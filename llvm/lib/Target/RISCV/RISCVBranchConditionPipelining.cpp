//===-- RISCVBranchConditionPipelining.cpp - Queue conditions ahead -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Software-pipelines the branch conditions of loops for the Non-Spec
// decoupled branches.
//
// A PB branch takes its condition from a FIFO in its branch register: each
// bmovc_* appends a bit and each PB consumes one. When the condition is
// computed just before the branch, typically from a load, the front end
// reaches the PB before the bmovc_* has produced its bit, and has to wait or
// guess. This pass computes such a condition D iterations before the branch
// consumes it, so that its bit is normally queued by the time the PB is
// fetched:
//
//   preheader:                            preheader:
//     br header                             if (tc <= D) goto drain.ph
//   header:                    =>         swp.ph:
//     c = cond(i)                           init(q)
//     br c, then, latch                     push(q, cond(0)) ... push(q, cond(D-1))
//   ...                                   kernel (tc - D iterations):
//                                           push(q, cond(i + D))
//                                           br pop(q), then', latch'
//                                         drain.ph:
//                                           the original loop runs what remains
//
// cond(j) is a clone of the instructions the branch condition depends on,
// evaluated with every induction variable advanced to iteration j. The
// kernel stops D iterations early so that it never evaluates the condition
// of an iteration that does not exist, and the original loop finishes the
// last D iterations with its ordinary branches. It cannot instead consume
// the D conditions the kernel queued for it: a branch register is bound to
// the address of a single PB. Those conditions are discarded when init(q)
// resets the queue the next time the loop is entered.
//
// Conditions are evaluated early, so the transformation requires that:
//  - the branch executes exactly once per iteration, so that pushes and pops
//    pair up;
//  - the condition depends only on loop-invariant values, affine induction
//    variables and side-effect-free instructions, which dominate the branch
//    and so already execute in every iteration;
//  - no store in the loop can write what a load in the condition reads in
//    the iterations the load now moves across;
//  - the loop has a computable trip count and no exit but its latch,
//    other than into undefined behavior.
//
// With -riscv-ns-swp-speculative, a loop without a computable trip count is
// also pipelined, if each instruction of its conditions is safe to execute
// on any operands (no loads, no division that could trap or be undefined).
// Its conditions are then also computed for iterations that never run, so
// there is no kernel copy, guard or drain loop: the loop is rewritten in
// place, its latch's condition can be pipelined too, and the conditions
// left queued when it exits are discarded by init the next time it is
// entered. The clones drop flags such as nsw, which an iteration that never
// runs could violate.
//
// The queue operations are the llvm.riscv.ns.cond.* intrinsics. Instruction
// selection turns them into PseudoNSQ* instructions, which RISCVExpandPseudo
// assigns a branch register.
//
// The pass runs before HardwareLoops, which then turns the kernel and the
// remainder loop into hardware loops.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVSubtarget.h"
#include "RISCVTargetMachine.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopIterator.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicsRISCV.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Verifier.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <optional>

using namespace llvm;
using namespace llvm::PatternMatch;

#define DEBUG_TYPE "riscv-branch-condition-pipelining"
#define PASS_NAME "RISC-V branch condition pipelining"

static cl::opt<bool> DisableSoftwarePipelining(
    "riscv-ns-disable-software-pipelining", cl::Hidden, cl::init(false),
    cl::desc("Disable computing loop branch conditions iterations ahead"));

static cl::opt<unsigned> PipelineDistance(
    "riscv-ns-swp-distance", cl::Hidden, cl::init(16),
    cl::desc("Number of iterations ahead of its branch that a pipelined "
             "loop computes a condition"));

static cl::opt<unsigned> MaxSliceSize(
    "riscv-ns-swp-max-slice-size", cl::Hidden, cl::init(16),
    cl::desc("Maximum number of instructions computing a pipelined branch "
             "condition"));

static cl::opt<unsigned>
    MaxLoopSize("riscv-ns-swp-max-loop-size", cl::Hidden, cl::init(256),
                cl::desc("Maximum number of instructions in a loop the "
                         "pipeliner duplicates"));

static cl::opt<unsigned> MaxPrologueSize(
    "riscv-ns-swp-max-prologue-size", cl::Hidden, cl::init(512),
    cl::desc("Maximum number of instructions in a pipelined loop's "
             "prologue; a branch's distance is reduced to fit"));

static cl::opt<unsigned> MaxFunctionBranches(
    "riscv-ns-swp-max-function-branches", cl::Hidden, cl::init(32),
    cl::desc("Do not pipeline a loop if the function could then need more "
             "branch registers than this"));

static cl::opt<bool> EnableShiftRecurrences(
    "riscv-ns-swp-shift-recurrences", cl::Hidden, cl::init(false),
    cl::desc("Also compute ahead conditions on values shifted by a constant "
             "every iteration, such as x >>= 1"));

static cl::opt<bool> EnableSpeculativePipelining(
    "riscv-ns-swp-speculative", cl::Hidden, cl::init(false),
    cl::desc("Also pipeline loops without a computable trip count when the "
             "conditions can be computed for iterations that never run"));

STATISTIC(NumLoopsPipelined, "Number of loops whose conditions were pipelined");
STATISTIC(NumBranchesPipelined, "Number of branches given a condition queue");
STATISTIC(NumLoopsPipelinedSpeculatively,
          "Number of loops without a trip count whose conditions were "
          "pipelined");

/// The FIFO of a branch register holds 63 conditions, and a pipelined
/// branch's holds its distance plus the one pushed before it is popped.
static constexpr unsigned MaxDistance = 62;

namespace {

/// A header phi whose value in iteration I is Start + I * Step.
struct InductionVar {
  PHINode *Phi;
  Value *Start;
  /// Null for a shift recurrence.
  const SCEV *Step;
  /// Step, materialized in the preheader.
  Value *StepV = nullptr;
  /// For a shift recurrence, whose value in iteration I is Start shifted by
  /// I * ShiftAmount: the shift (LShr, AShr or Shl) and its amount.
  Instruction::BinaryOps ShiftOp = Instruction::BinaryOpsEnd;
  unsigned ShiftAmount = 0;

  bool isShift() const { return ShiftOp != Instruction::BinaryOpsEnd; }
};

/// A branch whose condition the pipelined loop computes ahead of it.
struct PipelinedBranch {
  BranchInst *Br;
  /// The instructions of the loop the condition is computed by, operands
  /// before users. The condition is last.
  SmallVector<Instruction *, 8> Slice;
  /// Indices into LoopPlan::IVs of the induction variables the slice reads.
  SmallVector<unsigned, 2> IVs;
  /// How many iterations ahead of the branch the condition is computed.
  unsigned Distance = 0;
  unsigned Queue = 0;
};

struct LoopPlan {
  Loop *L;
  BasicBlock *Preheader;
  BasicBlock *Header;
  BasicBlock *Latch;
  /// The loop's blocks in reverse post-order.
  SmallVector<BasicBlock *, 8> Blocks;
  /// The trip count, as an XLen integer, materialized in the preheader.
  /// Null when Speculative.
  const SCEV *TripCount = nullptr;
  Value *TripCountV = nullptr;
  /// Whether the loop has no computable trip count, so its conditions are
  /// computed for iterations that may never run. The loop is then pipelined
  /// in place, without a kernel copy, a guard or a drain loop, and every
  /// condition must be safe to compute for any iteration.
  bool Speculative = false;
  SmallVector<InductionVar, 4> IVs;
  SmallVector<PipelinedBranch, 2> Branches;

  unsigned maxDistance() const {
    unsigned D = 0;
    for (const PipelinedBranch &PB : Branches)
      D = std::max(D, PB.Distance);
    return D;
  }
};

/// The integers from Lo to Hi.
struct Interval {
  int64_t Lo;
  int64_t Hi;
};

/// How the address of a memory access moves through a loop: in iteration t
/// it is Start plus between MinStep * t and MaxStep * t bytes.
struct AddressEvolution {
  const SCEV *Start;
  int64_t MinStep;
  int64_t MaxStep;
};

class RISCVBranchConditionPipelining : public FunctionPass {
public:
  static char ID;

  RISCVBranchConditionPipelining() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetPassConfig>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<ScalarEvolutionWrapperPass>();
    AU.addRequired<AAResultsWrapperPass>();
    AU.addRequired<OptimizationRemarkEmitterWrapperPass>();
  }

  StringRef getPassName() const override { return PASS_NAME; }

private:
  const DataLayout *DL = nullptr;
  DominatorTree *DT = nullptr;
  LoopInfo *LI = nullptr;
  ScalarEvolution *SE = nullptr;
  AAResults *AA = nullptr;
  OptimizationRemarkEmitter *ORE = nullptr;
  IntegerType *XLenTy = nullptr;

  std::optional<LoopPlan> analyzeLoop(Loop *L);
  bool canPipelineLoopBody(const LoopPlan &P);
  bool buildSlice(LoopPlan &P, PipelinedBranch &PB);
  std::optional<unsigned> findInductionVar(LoopPlan &P, PHINode *Phi);
  unsigned findSafeDistance(const LoopPlan &P, const PipelinedBranch &PB,
                            unsigned Distance);
  std::optional<AddressEvolution> getAddressEvolution(const LoopPlan &P,
                                                      Value *Ptr);
  std::optional<AddressEvolution> getTermEvolution(const LoopPlan &P,
                                                   const SCEV *Term);
  bool isLoadMovableAcross(const LoopPlan &P, const PipelinedBranch &PB,
                           LoadInst *Load, StoreInst *Store,
                           unsigned Distance);
  void materializeValues(LoopPlan &P);
  void transformLoop(LoopPlan &P);
  Value *cloneCondition(IRBuilder<> &B, const PipelinedBranch &PB,
                        ValueToValueMapTy &Map, const Twine &Suffix,
                        bool Speculative);
  void transformLoopInPlace(LoopPlan &P);
  void emitPush(IRBuilder<> &B, unsigned Queue, Value *Cond);
};

} // end anonymous namespace

char RISCVBranchConditionPipelining::ID = 0;

/// Whether a block of \p L can reach \p To without going around the loop:
/// whether, in one iteration, something in \p From can run before \p To.
static bool reachesInIteration(const Loop &L, const BasicBlock *From,
                               const BasicBlock *To) {
  SmallPtrSet<const BasicBlock *, 16> Visited;
  SmallVector<const BasicBlock *, 16> WorkList(succ_begin(From),
                                               succ_end(From));
  while (!WorkList.empty()) {
    const BasicBlock *BB = WorkList.pop_back_val();
    if (BB == L.getHeader() || !L.contains(BB) || !Visited.insert(BB).second)
      continue;
    if (BB == To)
      return true;
    llvm::append_range(WorkList, successors(BB));
  }
  return false;
}

/// Whether \p I may be evaluated for a later iteration: it computes its
/// result from its operands alone (or, for a load, from memory the caller
/// checks), so it can run again elsewhere without changing the program.
static bool isClonableIntoCondition(const Instruction &I) {
  if (auto *Load = dyn_cast<LoadInst>(&I))
    return Load->isSimple();
  if (auto *II = dyn_cast<IntrinsicInst>(&I))
    return II->doesNotAccessMemory() && II->willReturn() &&
           !II->mayHaveSideEffects() && !II->isConvergent() &&
           !II->getType()->isTokenTy();
  // freeze is left out: a second freeze of poison need not agree with the
  // first, while the branch and the rest of the iteration must agree.
  return isa<BinaryOperator>(I) || isa<UnaryOperator>(I) || isa<CastInst>(I) ||
         isa<CmpInst>(I) || isa<SelectInst>(I) || isa<GetElementPtrInst>(I) ||
         isa<ExtractElementInst>(I) || isa<InsertElementInst>(I) ||
         isa<ShuffleVectorInst>(I) || isa<ExtractValueInst>(I) ||
         isa<InsertValueInst>(I);
}

bool RISCVBranchConditionPipelining::canPipelineLoopBody(const LoopPlan &P) {
  Loop *L = P.L;
  unsigned Size = 0;
  for (BasicBlock *BB : P.Blocks) {
    if (BB->isEHPad() || BB->hasAddressTaken())
      return false;
    for (Instruction &I : *BB) {
      if (I.isDebugOrPseudoInst())
        continue;
      if (++Size > MaxLoopSize) {
        LLVM_DEBUG(dbgs() << "  loop is too large to duplicate\n");
        return false;
      }
      // The kernel is a copy of the loop.
      if (I.getType()->isTokenTy() || isa<AllocaInst>(I))
        return false;
      if (auto *CB = dyn_cast<CallBase>(&I)) {
        // A call could write anything a condition reads, and a call to a
        // pure intrinsic is the only kind certain to stay a non-call.
        auto *II = dyn_cast<IntrinsicInst>(CB);
        if (!II || II->isConvergent() || II->cannotDuplicate())
          return false;
        // A lifetime marker could let a stack slot be reused within an
        // iteration, under a load that now reads it earlier.
        if (II->isLifetimeStartOrEnd())
          return false;
        if (isa<DbgInfoIntrinsic>(II) || II->isAssumeLikeIntrinsic())
          continue;
        if (!II->doesNotAccessMemory() || II->mayHaveSideEffects())
          return false;
        continue;
      }
      if (auto *Store = dyn_cast<StoreInst>(&I)) {
        if (!Store->isSimple())
          return false;
        continue;
      }
      if (auto *Load = dyn_cast<LoadInst>(&I)) {
        if (!Load->isSimple())
          return false;
        continue;
      }
      // Fences, atomics, va_arg...
      if (I.mayReadOrWriteMemory() || I.mayHaveSideEffects())
        return false;
      // The loop has no exits but its latch, so a terminator can only be a
      // branch or switch within it.
      if (I.isTerminator() && !isa<BranchInst>(I) && !isa<SwitchInst>(I))
        return false;
    }
  }

  // Each block must run at most once per iteration, or a branch could
  // consume more conditions than the iteration queued. The loop is
  // innermost, so a cycle avoiding the header would be irreducible: look for
  // an edge that goes backwards in reverse post-order.
  DenseMap<BasicBlock *, unsigned> Order;
  for (auto [Idx, BB] : enumerate(P.Blocks))
    Order[BB] = Idx;
  for (BasicBlock *BB : P.Blocks)
    for (BasicBlock *Succ : successors(BB))
      if (L->contains(Succ) && Succ != P.Header && Order[Succ] <= Order[BB])
        return false;
  return true;
}

std::optional<unsigned>
RISCVBranchConditionPipelining::findInductionVar(LoopPlan &P, PHINode *Phi) {
  for (auto [Idx, IV] : enumerate(P.IVs))
    if (IV.Phi == Phi)
      return Idx;

  if (Phi->getParent() != P.Header || !SE->isSCEVable(Phi->getType()))
    return std::nullopt;
  auto *AR = dyn_cast<SCEVAddRecExpr>(SE->getSCEV(Phi));
  if (!AR || AR->getLoop() != P.L || !AR->isAffine()) {
    // x = x >> c (or x << c), the same constant every iteration: in iteration
    // I it is x shifted by I * c, so it can be advanced like an affine one.
    using namespace PatternMatch;
    Value *Next = Phi->getNumIncomingValues() == 2
                      ? Phi->getIncomingValueForBlock(P.Latch)
                      : nullptr;
    auto *Shift = dyn_cast_or_null<BinaryOperator>(Next);
    const APInt *C;
    unsigned Width = Phi->getType()->getScalarSizeInBits();
    if (!EnableShiftRecurrences || !Phi->getType()->isIntegerTy() || !Shift ||
        !P.L->contains(Shift) ||
        (Shift->getOpcode() != Instruction::LShr &&
         Shift->getOpcode() != Instruction::AShr &&
         Shift->getOpcode() != Instruction::Shl) ||
        Shift->getOperand(0) != Phi ||
        !match(Shift->getOperand(1), m_APInt(C)) || C->isZero() ||
        C->uge(Width))
      return std::nullopt;
    InductionVar IV = {Phi, Phi->getIncomingValueForBlock(P.Preheader),
                       nullptr, nullptr};
    IV.ShiftOp = Shift->getOpcode();
    IV.ShiftAmount = C->getZExtValue();
    P.IVs.push_back(IV);
    return P.IVs.size() - 1;
  }
  const SCEV *Step = AR->getStepRecurrence(*SE);
  SCEVExpander Expander(*SE, *DL, "swp");
  if (!isa<SCEVConstant>(Step) &&
      !Expander.isSafeToExpandAt(Step, P.Preheader->getTerminator()))
    return std::nullopt;

  P.IVs.push_back(
      {Phi, Phi->getIncomingValueForBlock(P.Preheader), Step, nullptr});
  return P.IVs.size() - 1;
}

bool RISCVBranchConditionPipelining::buildSlice(LoopPlan &P,
                                                PipelinedBranch &PB) {
  Loop *L = P.L;
  SmallPtrSet<Instruction *, 16> InSlice;
  SmallVector<Instruction *, 16> WorkList;
  SmallSetVector<unsigned, 2> IVs;

  auto Visit = [&](Value *V) {
    auto *I = dyn_cast<Instruction>(V);
    if (!I || !L->contains(I))
      return true;
    if (auto *Phi = dyn_cast<PHINode>(I)) {
      // Any other phi carries a value that the slice cannot recompute.
      std::optional<unsigned> IV = findInductionVar(P, Phi);
      if (!IV)
        return false;
      IVs.insert(*IV);
      return true;
    }
    if (!isClonableIntoCondition(*I))
      return false;
    // Without a trip count, conditions are also computed for iterations that
    // never run, so each instruction must be safe to execute on any operands.
    // No load is: its address may not be valid in such an iteration.
    if (P.Speculative &&
        (isa<LoadInst>(I) || !isSafeToSpeculativelyExecute(I)))
      return false;
    if (InSlice.insert(I).second)
      WorkList.push_back(I);
    return true;
  };

  if (!Visit(PB.Br->getCondition()))
    return false;
  while (!WorkList.empty()) {
    Instruction *I = WorkList.pop_back_val();
    for (Value *Op : I->operands())
      if (!Visit(Op))
        return false;
  }

  // An invariant condition gains nothing from being queued early, and a phi
  // condition is rare enough not to handle.
  if (InSlice.empty() || InSlice.size() > MaxSliceSize)
    return false;

  // Everything in the slice dominates the branch, so program order puts
  // every operand before its users.
  for (BasicBlock *BB : P.Blocks)
    for (Instruction &I : *BB)
      if (InSlice.contains(&I))
        PB.Slice.push_back(&I);
  assert(PB.Slice.back() == PB.Br->getCondition());
  PB.IVs.assign(IVs.begin(), IVs.end());
  return true;
}

/// Accesses at most this far apart, or with a step this large, are not
/// worth the trouble of checking exactly.
static constexpr int64_t MaxCheckedOffset = int64_t(1) << 40;

static bool isTooFar(std::optional<int64_t> V) {
  return !V || *V > MaxCheckedOffset || *V < -MaxCheckedOffset;
}

/// How many definitions offsetFromPhi follows back from a value.
static constexpr unsigned MaxOffsetDepth = 8;

/// Returns bounds on how much \p V exceeds the header phi \p Phi of \p L in
/// the iteration that computes V, if every path through the iteration
/// computes V by adding constants to Phi. With \p NoSignedWrap, each addition
/// must be nsw, so that the bounds also hold between the sign extensions of V
/// and Phi.
static std::optional<Interval> offsetFromPhi(const Loop &L, const PHINode *Phi,
                                             const Value *V, bool NoSignedWrap,
                                             const DataLayout &DL,
                                             unsigned Depth = 0) {
  if (V == Phi)
    return Interval{0, 0};
  auto *I = dyn_cast<Instruction>(V);
  if (!I || !L.contains(I) || Depth == MaxOffsetDepth)
    return std::nullopt;

  // A phi or select picks one of several values computed earlier in the
  // iteration. A phi in the header would pick one from the previous
  // iteration.
  SmallVector<const Value *, 4> Choices;
  if (auto *PN = dyn_cast<PHINode>(I)) {
    if (PN->getParent() == L.getHeader())
      return std::nullopt;
    append_range(Choices, PN->incoming_values());
  } else if (auto *Sel = dyn_cast<SelectInst>(I)) {
    Choices = {Sel->getTrueValue(), Sel->getFalseValue()};
  }
  if (!Choices.empty()) {
    std::optional<Interval> R;
    for (const Value *Choice : Choices) {
      std::optional<Interval> CR =
          offsetFromPhi(L, Phi, Choice, NoSignedWrap, DL, Depth + 1);
      if (!CR)
        return std::nullopt;
      if (R)
        R = Interval{std::min(R->Lo, CR->Lo), std::max(R->Hi, CR->Hi)};
      else
        R = CR;
    }
    return R;
  }

  const Value *Base = nullptr;
  std::optional<int64_t> Addend;
  const APInt *C;
  if (auto *GEP = dyn_cast<GetElementPtrInst>(I)) {
    APInt Offset(DL.getIndexTypeSizeInBits(GEP->getType()), 0);
    if (!GEP->accumulateConstantOffset(DL, Offset))
      return std::nullopt;
    Base = GEP->getPointerOperand();
    Addend = Offset.trySExtValue();
  } else if (match(I, m_c_Add(m_Value(Base), m_APInt(C))) &&
             (!NoSignedWrap || I->hasNoSignedWrap())) {
    Addend = C->trySExtValue();
  }
  if (isTooFar(Addend))
    return std::nullopt;
  std::optional<Interval> R =
      offsetFromPhi(L, Phi, Base, NoSignedWrap, DL, Depth + 1);
  if (!R)
    return std::nullopt;
  return Interval{R->Lo + *Addend, R->Hi + *Addend};
}

std::optional<AddressEvolution>
RISCVBranchConditionPipelining::getTermEvolution(const LoopPlan &P,
                                                 const SCEV *Term) {
  if (SE->isLoopInvariant(Term, P.L))
    return AddressEvolution{Term, 0, 0};

  if (auto *AR = dyn_cast<SCEVAddRecExpr>(Term)) {
    auto *Step = AR->getLoop() == P.L && AR->isAffine()
                     ? dyn_cast<SCEVConstant>(AR->getStepRecurrence(*SE))
                     : nullptr;
    std::optional<int64_t> V =
        Step ? Step->getAPInt().trySExtValue() : std::nullopt;
    if (isTooFar(V))
      return std::nullopt;
    return AddressEvolution{AR->getStart(), *V, *V};
  }

  // Otherwise the term must be Scale * K or Scale * sext(K), where K is a
  // header phi that each iteration changes by an amount within known bounds,
  // such as a counter that only some iterations increment.
  int64_t Scale = 1;
  const SCEV *X = Term;
  if (auto *Mul = dyn_cast<SCEVMulExpr>(X); Mul && Mul->getNumOperands() == 2) {
    auto *C = dyn_cast<SCEVConstant>(Mul->getOperand(0));
    std::optional<int64_t> V = C ? C->getAPInt().trySExtValue() : std::nullopt;
    if (isTooFar(V))
      return std::nullopt;
    Scale = *V;
    X = Mul->getOperand(1);
  }
  // The bounds survive sign extension only if K never wraps. Unextended, K
  // is as wide as the address, which wraps along with it.
  auto *Ext = dyn_cast<SCEVSignExtendExpr>(X);
  if (Ext)
    X = Ext->getOperand();
  auto *U = dyn_cast<SCEVUnknown>(X);
  auto *Phi = U ? dyn_cast<PHINode>(U->getValue()) : nullptr;
  if (!Phi || Phi->getParent() != P.Header)
    return std::nullopt;
  std::optional<Interval> Change =
      offsetFromPhi(*P.L, Phi, Phi->getIncomingValueForBlock(P.Latch),
                    /*NoSignedWrap=*/Ext != nullptr, *DL);
  if (!Change)
    return std::nullopt;
  std::optional<int64_t> MinStep = checkedMul(Scale, Change->Lo);
  std::optional<int64_t> MaxStep = checkedMul(Scale, Change->Hi);
  if (isTooFar(MinStep) || isTooFar(MaxStep))
    return std::nullopt;
  if (Scale < 0)
    std::swap(MinStep, MaxStep);
  // In iteration 0, K holds its value from the preheader.
  ValueToSCEVMapTy Initial;
  Initial[Phi] = SE->getSCEV(Phi->getIncomingValueForBlock(P.Preheader));
  return AddressEvolution{SCEVParameterRewriter::rewrite(Term, *SE, Initial),
                          *MinStep, *MaxStep};
}

std::optional<AddressEvolution>
RISCVBranchConditionPipelining::getAddressEvolution(const LoopPlan &P,
                                                    Value *Ptr) {
  // Each term of a sum moves on its own.
  const SCEV *S = SE->getSCEV(Ptr);
  ArrayRef<const SCEV *> Terms = S;
  if (auto *Add = dyn_cast<SCEVAddExpr>(S))
    Terms = Add->operands();
  SmallVector<const SCEV *, 4> Starts;
  int64_t MinStep = 0, MaxStep = 0;
  for (const SCEV *Term : Terms) {
    std::optional<AddressEvolution> TermEv = getTermEvolution(P, Term);
    if (!TermEv)
      return std::nullopt;
    Starts.push_back(TermEv->Start);
    MinStep += TermEv->MinStep;
    MaxStep += TermEv->MaxStep;
  }
  if (isTooFar(MinStep) || isTooFar(MaxStep))
    return std::nullopt;
  return AddressEvolution{SE->getAddExpr(Starts), MinStep, MaxStep};
}

bool RISCVBranchConditionPipelining::isLoadMovableAcross(
    const LoopPlan &P, const PipelinedBranch &PB, LoadInst *Load,
    StoreInst *Store, unsigned Distance) {
  // Scoped noalias metadata only speaks about one iteration, while TBAA
  // holds across all of them.
  auto TBAAOnly = [](const Instruction *I) {
    AAMDNodes Tags = I->getAAMetadata();
    Tags.Scope = nullptr;
    Tags.NoAlias = nullptr;
    return Tags;
  };
  if (AA->isNoAlias(
          MemoryLocation::getBeforeOrAfter(Load->getPointerOperand(),
                                           TBAAOnly(Load)),
          MemoryLocation::getBeforeOrAfter(Store->getPointerOperand(),
                                           TBAAOnly(Store))))
    return true;

  // Otherwise both addresses must move predictably from a known distance
  // apart. The load's advances by the same amount every iteration, while the
  // store's may advance by another amount, or by one that varies within
  // bounds.
  std::optional<AddressEvolution> LoadEv =
      getAddressEvolution(P, Load->getPointerOperand());
  std::optional<AddressEvolution> StoreEv =
      getAddressEvolution(P, Store->getPointerOperand());
  if (!LoadEv || !StoreEv || LoadEv->MinStep != LoadEv->MaxStep)
    return false;
  auto *Start =
      dyn_cast<SCEVConstant>(SE->getMinusSCEV(LoadEv->Start, StoreEv->Start));
  if (!Start)
    return false;
  TypeSize LoadSize = DL->getTypeStoreSize(Load->getType());
  TypeSize StoreSize = DL->getTypeStoreSize(Store->getValueOperand()->getType());
  if (LoadSize.isScalable() || StoreSize.isScalable())
    return false;
  int64_t Step = LoadEv->MinStep;
  std::optional<int64_t> Offset = Start->getAPInt().trySExtValue();
  if (isTooFar(Offset))
    return false;

  // From iteration 0 to the store's iteration t, the load's address gains
  // between (Step - MaxStep) * t and (Step - MinStep) * t bytes on the
  // store's. Unless the two always advance alike, that needs a bound on t.
  int64_t MinGain = 0, MaxGain = 0;
  if (StoreEv->MinStep != Step || StoreEv->MaxStep != Step) {
    auto *MaxBTC =
        dyn_cast<SCEVConstant>(SE->getConstantMaxBackedgeTakenCount(P.L));
    if (!MaxBTC || MaxBTC->getAPInt().getActiveBits() >= 64)
      return false;
    int64_t MaxT = MaxBTC->getAPInt().getZExtValue();
    std::optional<int64_t> Lo = checkedMul(Step - StoreEv->MaxStep, MaxT);
    std::optional<int64_t> Hi = checkedMul(Step - StoreEv->MinStep, MaxT);
    if (!Lo || !Hi)
      return false;
    MinGain = std::min<int64_t>(*Lo, 0);
    MaxGain = std::max<int64_t>(*Hi, 0);
  }

  // The load of iteration J now runs at the end of the branch's block in
  // iteration J - Distance, so it moves across the store of iteration
  // J - T for each T in [0, Distance]: always if 0 < T < Distance; for
  // T = 0 if the store can run before the load; and for T = Distance if the
  // store can run after the branch's block.
  BasicBlock *BrBB = PB.Br->getParent();
  BasicBlock *LoadBB = Load->getParent();
  BasicBlock *StoreBB = Store->getParent();
  bool StorePrecedesLoad = StoreBB == LoadBB
                               ? Store->comesBefore(Load)
                               : reachesInIteration(*P.L, StoreBB, LoadBB);
  bool StoreFollowsBranch =
      StoreBB != BrBB && reachesInIteration(*P.L, BrBB, StoreBB);
  for (unsigned T = 0; T <= Distance; ++T) {
    if ((T == 0 && !StorePrecedesLoad) ||
        (T == Distance && !StoreFollowsBranch))
      continue;
    // The load's address less the store's, computed without wrapping, lies
    // in [Lo, Hi]. As long as that range fits in 64 bits, the accesses can
    // only overlap if the difference is in (-LoadSize, StoreSize).
    int64_t Delta = *Offset + Step * int64_t(T);
    std::optional<int64_t> Lo = checkedAdd(Delta, MinGain);
    std::optional<int64_t> Hi = checkedAdd(Delta, MaxGain);
    if (!Lo || !Hi ||
        (*Lo < int64_t(StoreSize.getFixedValue()) &&
         *Hi > -int64_t(LoadSize.getFixedValue())))
      return false;
  }
  return true;
}

/// Returns the largest distance up to \p Distance at which every load of the
/// condition can move ahead of the stores of the iterations it skips, or 0.
unsigned RISCVBranchConditionPipelining::findSafeDistance(
    const LoopPlan &P, const PipelinedBranch &PB, unsigned Distance) {
  SmallVector<LoadInst *, 4> Loads;
  for (Instruction *I : PB.Slice)
    if (auto *Load = dyn_cast<LoadInst>(I))
      Loads.push_back(Load);
  if (Loads.empty())
    return Distance;

  SmallVector<StoreInst *, 8> Stores;
  for (BasicBlock *BB : P.Blocks)
    for (Instruction &I : *BB)
      if (auto *Store = dyn_cast<StoreInst>(&I))
        Stores.push_back(Store);

  for (; Distance > 0; --Distance) {
    bool Safe = all_of(Loads, [&](LoadInst *Load) {
      return all_of(Stores, [&](StoreInst *Store) {
        return isLoadMovableAcross(P, PB, Load, Store, Distance);
      });
    });
    if (Safe)
      break;
  }
  return Distance;
}

/// Whether every way out of \p L but its latch is undefined behavior, such as
/// the default of a switch that covers every value: if taken, it would leave
/// conditions evaluated for iterations that never run. The kernel shares
/// those exits, so they must also have no phis to update.
static bool exitsOnlyAtLatch(const Loop &L) {
  SmallVector<BasicBlock *, 4> Exiting;
  L.getExitingBlocks(Exiting);
  for (BasicBlock *BB : Exiting) {
    if (BB == L.getLoopLatch())
      continue;
    for (BasicBlock *Succ : successors(BB))
      if (!L.contains(Succ) &&
          (!Succ->phis().empty() ||
           !isa<UnreachableInst>(Succ->getFirstNonPHIOrDbgOrLifetime())))
        return false;
  }
  return true;
}

std::optional<LoopPlan> RISCVBranchConditionPipelining::analyzeLoop(Loop *L) {
  LLVM_DEBUG(dbgs() << "Loop " << L->getHeader()->getName() << ":\n");

  LoopPlan P;
  P.L = L;
  P.Preheader = L->getLoopPreheader();
  P.Header = L->getHeader();
  P.Latch = L->getLoopLatch();
  if (!L->isInnermost() || !L->isLoopSimplifyForm() || !P.Latch ||
      !L->isLoopExiting(P.Latch) || !exitsOnlyAtLatch(*L)) {
    LLVM_DEBUG(dbgs() << "  not an innermost loop exiting at its latch\n");
    return std::nullopt;
  }
  auto *LatchBr = dyn_cast<BranchInst>(P.Latch->getTerminator());
  if (!LatchBr || !LatchBr->isConditional() ||
      !isa<BranchInst>(P.Preheader->getTerminator()))
    return std::nullopt;

  LoopBlocksRPO RPOT(L);
  RPOT.perform(LI);
  P.Blocks.assign(RPOT.begin(), RPOT.end());
  if (!canPipelineLoopBody(P)) {
    LLVM_DEBUG(dbgs() << "  loop body cannot be pipelined\n");
    return std::nullopt;
  }

  const SCEV *BTC = SE->getExitCount(L, P.Latch);
  if (isa<SCEVCouldNotCompute>(BTC) || !BTC->getType()->isIntegerTy() ||
      BTC->getType()->getIntegerBitWidth() > XLenTy->getBitWidth()) {
    if (!EnableSpeculativePipelining) {
      LLVM_DEBUG(dbgs() << "  trip count is not computable\n");
      return std::nullopt;
    }
    // Without a trip count, the loop is pipelined in place: its conditions
    // are computed until it exits, also for iterations that never run.
    LLVM_DEBUG(dbgs() << "  trip count is not computable, pipelining "
                         "speculatively\n");
    P.Speculative = true;
  } else {
    // tc wraps to 0 for a loop of 2^XLen iterations, which only makes the
    // guard send it to the original loop.
    P.TripCount = SE->getAddExpr(SE->getNoopOrZeroExtend(BTC, XLenTy),
                                 SE->getOne(XLenTy));
    SCEVExpander Expander(*SE, *DL, "swp");
    if (!Expander.isSafeToExpandAt(P.TripCount, P.Preheader->getTerminator()))
      return std::nullopt;
  }

  unsigned Distance = std::min<unsigned>(PipelineDistance, MaxDistance);
  unsigned MaxTripCount = SE->getSmallConstantMaxTripCount(L);
  if (MaxTripCount && MaxTripCount <= Distance) {
    LLVM_DEBUG(dbgs() << "  never runs more than " << MaxTripCount
                      << " iterations\n");
    return std::nullopt;
  }

  unsigned PrologueSize = 0;
  for (BasicBlock *BB : P.Blocks) {
    // With a trip count, the kernel counts its iterations in place of the
    // latch's condition. Without one, the latch's condition decides when the
    // loop exits, and it can be pipelined like any other.
    if (BB == P.Latch && !P.Speculative)
      continue;
    auto *Br = dyn_cast<BranchInst>(BB->getTerminator());
    // The branch must run exactly once per iteration, to pop the condition
    // its iteration pushed. Blocks run at most once per iteration, and one
    // that dominates the latch runs at least once.
    if (!Br || !Br->isConditional() || !DT->dominates(BB, P.Latch) ||
        Br->getSuccessor(0) == Br->getSuccessor(1))
      continue;

    PipelinedBranch PB;
    PB.Br = Br;
    if (!buildSlice(P, PB)) {
      LLVM_DEBUG(dbgs() << "  cannot compute ahead: " << *Br << "\n");
      continue;
    }
    // Each condition the prologue queues costs a copy of the slice and a
    // push.
    unsigned PushSize = PB.Slice.size() + 1;
    unsigned Fit = std::min(Distance, (MaxPrologueSize - std::min<unsigned>(
                                           PrologueSize, MaxPrologueSize)) /
                                          PushSize);
    PB.Distance = findSafeDistance(P, PB, Fit);
    if (!PB.Distance) {
      LLVM_DEBUG(dbgs() << "  a store could clobber what the condition of "
                        << *Br << " reads\n");
      continue;
    }
    PrologueSize += PB.Distance * PushSize;
    LLVM_DEBUG(dbgs() << "  pipelining " << *Br << " " << PB.Distance
                      << " iterations ahead\n");
    P.Branches.push_back(std::move(PB));
  }

  if (P.Branches.empty())
    return std::nullopt;
  return P;
}

/// An upper bound on the branch registers \p F needs, if every control
/// transfer were a PB with a register of its own.
static unsigned estimateBranchRegisters(const Function &F) {
  unsigned N = 0;
  for (const BasicBlock &BB : F) {
    const Instruction *Term = BB.getTerminator();
    if (auto *SI = dyn_cast<SwitchInst>(Term))
      N += SI->getNumCases() + 1;
    else if (Term)
      N += Term->getNumSuccessors() > 0 || isa<ReturnInst>(Term);
    for (const Instruction &I : BB)
      if (auto *CB = dyn_cast<CallBase>(&I))
        if (!isa<IntrinsicInst>(CB) && !CB->isTerminator())
          ++N;
  }
  return N;
}

void RISCVBranchConditionPipelining::materializeValues(LoopPlan &P) {
  SCEVExpander Expander(*SE, *DL, "swp");
  Instruction *InsertPt = P.Preheader->getTerminator();
  if (!P.Speculative)
    P.TripCountV = Expander.expandCodeFor(P.TripCount, XLenTy, InsertPt);
  // Rejected branches may have found induction variables nothing uses.
  SmallSet<unsigned, 4> Used;
  for (const PipelinedBranch &PB : P.Branches)
    Used.insert(PB.IVs.begin(), PB.IVs.end());
  for (auto [Idx, IV] : enumerate(P.IVs)) {
    // A shift recurrence has no step to materialize.
    if (!Used.contains(Idx) || IV.isShift())
      continue;
    if (auto *C = dyn_cast<SCEVConstant>(IV.Step))
      IV.StepV = C->getValue();
    else
      IV.StepV = Expander.expandCodeFor(IV.Step, IV.Step->getType(), InsertPt);
  }
}

/// Returns the value \p IV has \p Iterations iterations after it has
/// \p Base.
static Value *advanceInductionVar(IRBuilder<> &B, const InductionVar &IV,
                                  Value *Base, unsigned Iterations,
                                  const Twine &Name) {
  if (Iterations == 0)
    return Base;
  if (IV.isShift()) {
    // Shifting by a then by b is shifting by a + b, and a shift by the width
    // or more leaves nothing of a logical shift and only the sign of an
    // arithmetic one. The shift emitted is therefore always less than the
    // width, which keeps it safe to compute for any iteration.
    Type *Ty = Base->getType();
    unsigned Width = Ty->getScalarSizeInBits();
    uint64_t Total = uint64_t(IV.ShiftAmount) * Iterations;
    if (Total >= Width) {
      if (IV.ShiftOp != Instruction::AShr)
        return Constant::getNullValue(Ty);
      Total = Width - 1;
    }
    assert(Total > 0 && Total < Width && "shift out of range");
    return B.CreateBinOp(IV.ShiftOp, Base, ConstantInt::get(Ty, Total), Name);
  }
  Value *Offset = B.CreateMul(
      IV.StepV, ConstantInt::get(IV.StepV->getType(), Iterations));
  if (Base->getType()->isPointerTy())
    return B.CreatePtrAdd(Base, Offset, Name);
  return B.CreateAdd(Base, Offset, Name);
}

Value *RISCVBranchConditionPipelining::cloneCondition(
    IRBuilder<> &B, const PipelinedBranch &PB, ValueToValueMapTy &Map,
    const Twine &Suffix, bool Speculative) {
  for (Instruction *I : PB.Slice) {
    Instruction *C = I->clone();
    // These describe the accesses of the iteration I belongs to, and the
    // clone belongs to a different one.
    C->setMetadata(LLVMContext::MD_alias_scope, nullptr);
    C->setMetadata(LLVMContext::MD_noalias, nullptr);
    C->setMetadata(LLVMContext::MD_access_group, nullptr);
    // In an iteration that never runs, an operation may overflow where the
    // original, in an iteration that runs, cannot. Without these flags the
    // clone computes the same value in every iteration that runs.
    if (Speculative)
      C->dropPoisonGeneratingAnnotations();
    RemapInstruction(C, Map,
                     RF_NoModuleLevelChanges | RF_IgnoreMissingLocals);
    B.Insert(C, I->hasName() ? I->getName() + Suffix : "");
    // The clone, with its operands and flags as emitted, may run for an
    // iteration that never does, so it must be safe to execute anywhere.
    assert((!Speculative ||
            (!isa<LoadInst>(C) && isSafeToSpeculativelyExecute(C))) &&
           "speculative condition clone is not safe to execute anywhere");
    Map[I] = C;
  }
  return Map[PB.Slice.back()];
}

void RISCVBranchConditionPipelining::emitPush(IRBuilder<> &B, unsigned Queue,
                                              Value *Cond) {
  // Fold an integer comparison into the push, so that it becomes one
  // BMOVC_*. Sign extension keeps both the signed and the unsigned order of
  // narrower integers.
  auto ToXLen = [&](Value *V) -> Value * {
    Type *Ty = V->getType();
    if (Ty->isPointerTy())
      return DL->getPointerSizeInBits(Ty->getPointerAddressSpace()) ==
                     XLenTy->getBitWidth()
                 ? B.CreatePtrToInt(V, XLenTy)
                 : nullptr;
    if (!Ty->isIntegerTy() || Ty->getIntegerBitWidth() > XLenTy->getBitWidth())
      return nullptr;
    return B.CreateSExtOrBitCast(V, XLenTy);
  };

  CmpInst::Predicate Pred = CmpInst::ICMP_NE;
  Value *LHS = nullptr, *RHS = nullptr;
  auto *Cmp = dyn_cast<ICmpInst>(Cond);
  if (Cmp && Cmp->getOperand(0)->getType()->isIntOrPtrTy()) {
    LHS = ToXLen(Cmp->getOperand(0));
    RHS = LHS ? ToXLen(Cmp->getOperand(1)) : nullptr;
    Pred = Cmp->getPredicate();
  }
  if (!LHS || !RHS) {
    Pred = CmpInst::ICMP_NE;
    LHS = B.CreateZExt(Cond, XLenTy);
    RHS = ConstantInt::get(XLenTy, 0);
    Cmp = nullptr;
  }
  B.CreateIntrinsic(Intrinsic::riscv_ns_cond_push, {XLenTy},
                    {B.getInt32(Queue), B.getInt32(Pred), LHS, RHS});
  if (Cmp)
    RecursivelyDeleteTriviallyDeadInstructions(Cmp);
}

/// Pipelines a loop without a trip count where it is: the preheader queues
/// each branch's conditions for its first iterations, and each branch
/// consumes the condition queued Distance iterations ago and queues the one
/// for Distance iterations ahead. The loop runs until it exits as before, so
/// up to Distance conditions per branch are left queued for iterations that
/// never run; init empties the queue the next time the loop is entered. No
/// block is added or removed.
void RISCVBranchConditionPipelining::transformLoopInPlace(LoopPlan &P) {
  assert(P.Speculative && !P.TripCountV && "loop has a trip count");
  unsigned Distance = P.maxDistance();

  // The preheader runs once each time the loop is entered.
  IRBuilder<> B(P.Preheader->getTerminator());
  for (const PipelinedBranch &PB : P.Branches)
    B.CreateIntrinsic(Intrinsic::riscv_ns_cond_init, {},
                      {B.getInt32(PB.Queue)});
  for (unsigned K = 0; K < Distance; ++K) {
    for (const PipelinedBranch &PB : P.Branches) {
      if (K >= PB.Distance)
        continue;
      ValueToValueMapTy Map;
      for (unsigned Idx : PB.IVs) {
        const InductionVar &IV = P.IVs[Idx];
        Map[IV.Phi] = advanceInductionVar(B, IV, IV.Start, K,
                                          IV.Phi->getName() + ".swp" + Twine(K));
      }
      emitPush(B, PB.Queue,
               cloneCondition(B, PB, Map, ".swp" + Twine(K),
                              /*Speculative=*/true));
    }
  }

  // The slices still compute the other branches' conditions, so the old
  // conditions are only deleted once every branch is rewritten. Two branches
  // may share one, which the handles track once it is deleted.
  SmallVector<WeakTrackingVH, 2> OldConds;
  for (const PipelinedBranch &PB : P.Branches) {
    B.SetInsertPoint(PB.Br);
    B.SetCurrentDebugLocation(PB.Br->getDebugLoc());
    ValueToValueMapTy Map;
    for (unsigned Idx : PB.IVs) {
      const InductionVar &IV = P.IVs[Idx];
      Map[IV.Phi] = advanceInductionVar(B, IV, IV.Phi, PB.Distance,
                                        IV.Phi->getName() + ".ahead");
    }
    emitPush(B, PB.Queue,
             cloneCondition(B, PB, Map, ".ahead", /*Speculative=*/true));
    OldConds.push_back(PB.Br->getCondition());
    PB.Br->setCondition(B.CreateIntrinsic(Intrinsic::riscv_ns_cond_pop, {},
                                          {B.getInt32(PB.Queue)}));
  }
  // A condition the loop still uses elsewhere is left in place.
  RecursivelyDeleteTriviallyDeadInstructionsPermissive(OldConds);
}

void RISCVBranchConditionPipelining::transformLoop(LoopPlan &P) {
  Function *F = P.Header->getParent();
  LLVMContext &Ctx = F->getContext();
  unsigned Distance = P.maxDistance();

  // The kernel: a copy of the loop.
  ValueToValueMapTy VMap;
  SmallVector<BasicBlock *, 8> Kernel;
  for (BasicBlock *BB : P.Blocks) {
    BasicBlock *NewBB = CloneBasicBlock(BB, VMap, ".swp", F);
    VMap[BB] = NewBB;
    Kernel.push_back(NewBB);
  }
  remapInstructionsInBlocks(Kernel, VMap);

  BasicBlock *PipePH = BasicBlock::Create(Ctx, "swp.ph", F, P.Header);
  for (BasicBlock *BB : Kernel)
    BB->moveBefore(P.Header);
  BasicBlock *KernelExit =
      BasicBlock::Create(Ctx, "swp.kernel.exit", F, P.Header);
  BasicBlock *DrainPH = BasicBlock::Create(Ctx, "swp.drain.ph", F, P.Header);
  auto *KHeader = cast<BasicBlock>(VMap[P.Header]);
  auto *KLatch = cast<BasicBlock>(VMap[P.Latch]);

  // The prologue queues each branch's conditions for its first iterations.
  IRBuilder<> B(PipePH);
  B.SetCurrentDebugLocation(P.Preheader->getTerminator()->getDebugLoc());
  for (const PipelinedBranch &PB : P.Branches)
    B.CreateIntrinsic(Intrinsic::riscv_ns_cond_init, {},
                      {B.getInt32(PB.Queue)});
  for (unsigned K = 0; K < Distance; ++K) {
    for (const PipelinedBranch &PB : P.Branches) {
      if (K >= PB.Distance)
        continue;
      ValueToValueMapTy Map;
      for (unsigned Idx : PB.IVs) {
        const InductionVar &IV = P.IVs[Idx];
        Map[IV.Phi] = advanceInductionVar(B, IV, IV.Start, K,
                                          IV.Phi->getName() + ".swp" + Twine(K));
      }
      emitPush(B, PB.Queue,
               cloneCondition(B, PB, Map, ".swp" + Twine(K),
                              /*Speculative=*/false));
    }
  }
  // The kernel stops Distance iterations early: its last iteration queues
  // the condition of the original loop's last.
  Value *KernelCount = B.CreateSub(P.TripCountV,
                                   ConstantInt::get(XLenTy, Distance),
                                   "swp.kernel.count", /*HasNUW=*/true);
  B.CreateBr(KHeader);

  for (PHINode &Phi : KHeader->phis())
    Phi.setIncomingBlock(Phi.getBasicBlockIndex(P.Preheader), PipePH);

  // Count the kernel's iterations in place of the original exit condition.
  PHINode *Counter = PHINode::Create(XLenTy, 2, "swp.kernel.iv");
  Counter->insertBefore(KHeader->begin());
  auto *OldLatchBr = cast<BranchInst>(KLatch->getTerminator());
  B.SetInsertPoint(OldLatchBr);
  Value *CounterNext =
      B.CreateAdd(Counter, ConstantInt::get(XLenTy, 1), "swp.kernel.iv.next",
                  /*HasNUW=*/true);
  Value *Done = B.CreateICmpEQ(CounterNext, KernelCount, "swp.kernel.done");
  B.CreateCondBr(Done, KernelExit, KHeader);
  Counter->addIncoming(ConstantInt::get(XLenTy, 0), PipePH);
  Counter->addIncoming(CounterNext, KLatch);
  Value *OldExitCond = OldLatchBr->getCondition();
  OldLatchBr->eraseFromParent();
  RecursivelyDeleteTriviallyDeadInstructions(OldExitCond);

  // Each kernel branch consumes the condition queued Distance iterations
  // ago and queues the one for Distance iterations ahead.
  for (const PipelinedBranch &PB : P.Branches) {
    auto *KBr = cast<BranchInst>(VMap[PB.Br]);
    B.SetInsertPoint(KBr);
    B.SetCurrentDebugLocation(KBr->getDebugLoc());
    ValueToValueMapTy Map;
    for (unsigned Idx : PB.IVs) {
      const InductionVar &IV = P.IVs[Idx];
      Map[IV.Phi] = advanceInductionVar(B, IV, VMap[IV.Phi], PB.Distance,
                                        IV.Phi->getName() + ".ahead");
    }
    emitPush(B, PB.Queue,
             cloneCondition(B, PB, Map, ".ahead", /*Speculative=*/false));
    Value *OldCond = KBr->getCondition();
    KBr->setCondition(B.CreateIntrinsic(Intrinsic::riscv_ns_cond_pop, {},
                                        {B.getInt32(PB.Queue)}));
    RecursivelyDeleteTriviallyDeadInstructions(OldCond);
  }

  // The original loop picks up where the kernel left off, or runs every
  // iteration if the loop is too short to pipeline.
  BranchInst::Create(DrainPH, KernelExit);
  B.SetInsertPoint(DrainPH);
  B.SetCurrentDebugLocation(P.Preheader->getTerminator()->getDebugLoc());
  for (PHINode &Phi : P.Header->phis()) {
    int PHIdx = Phi.getBasicBlockIndex(P.Preheader);
    Value *Next = Phi.getIncomingValueForBlock(P.Latch);
    Value *KernelNext = VMap.lookup(Next);
    PHINode *Resume =
        B.CreatePHI(Phi.getType(), 2, Phi.getName() + ".swp.resume");
    Resume->addIncoming(Phi.getIncomingValue(PHIdx), P.Preheader);
    Resume->addIncoming(KernelNext ? KernelNext : Next, KernelExit);
    Phi.setIncomingBlock(PHIdx, DrainPH);
    Phi.setIncomingValue(PHIdx, Resume);
  }
  B.CreateBr(P.Header);

  auto *PreheaderBr = cast<BranchInst>(P.Preheader->getTerminator());
  B.SetInsertPoint(PreheaderBr);
  Value *LongEnough = B.CreateICmpUGT(
      P.TripCountV, ConstantInt::get(XLenTy, Distance), "swp.long.enough");
  B.CreateCondBr(LongEnough, PipePH, DrainPH);
  PreheaderBr->eraseFromParent();
}

bool RISCVBranchConditionPipelining::runOnFunction(Function &F) {
  if (skipFunction(F) || DisableSoftwarePipelining || PipelineDistance == 0 ||
      F.hasOptSize())
    return false;

  // Only SelectionDAG knows how to select the queue intrinsics.
  auto &TM = getAnalysis<TargetPassConfig>().getTM<RISCVTargetMachine>();
  if (TM.Options.EnableGlobalISel)
    return false;
  const RISCVSubtarget &ST = TM.getSubtarget<RISCVSubtarget>(F);
  DL = &F.getDataLayout();
  XLenTy = IntegerType::get(F.getContext(), ST.getXLen());
  DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  SE = &getAnalysis<ScalarEvolutionWrapperPass>().getSE();
  AA = &getAnalysis<AAResultsWrapperPass>().getAAResults();
  ORE = &getAnalysis<OptimizationRemarkEmitterWrapperPass>().getORE();

  // Plan every loop before changing any: the analyses describe the function
  // as it was.
  unsigned Branches = estimateBranchRegisters(F);
  unsigned NextQueue = 0;
  for (Instruction &I : instructions(F))
    if (auto *II = dyn_cast<IntrinsicInst>(&I))
      if (II->getIntrinsicID() == Intrinsic::riscv_ns_cond_init ||
          II->getIntrinsicID() == Intrinsic::riscv_ns_cond_push) {
        auto *Queue = dyn_cast<ConstantInt>(II->getArgOperand(0));
        if (Queue)
          NextQueue = std::max(
              NextQueue, static_cast<unsigned>(Queue->getZExtValue() + 1));
      }
  SmallVector<LoopPlan, 4> Plans;
  for (Loop *L : LI->getLoopsInPreorder()) {
    std::optional<LoopPlan> P = analyzeLoop(L);
    if (!P)
      continue;

    // The kernel duplicates the loop's branches, and the guard and the
    // kernel exit add some.
    unsigned Added = 3;
    for (BasicBlock *BB : P->Blocks)
      if (BB->getTerminator()->getNumSuccessors() > 0)
        ++Added;
    if (Branches + Added > MaxFunctionBranches) {
      ORE->emit([&]() {
        return OptimizationRemarkMissed(DEBUG_TYPE, "TooManyBranches",
                                        L->getStartLoc(), L->getHeader())
               << "not pipelining branch conditions: the function could run "
                  "out of branch registers";
      });
      continue;
    }
    Branches += Added;

    for (PipelinedBranch &PB : P->Branches)
      PB.Queue = NextQueue++;
    ORE->emit([&]() {
      OptimizationRemark R(DEBUG_TYPE, "Pipelined", L->getStartLoc(),
                           L->getHeader());
      R << "computing " << ore::NV("Branches", unsigned(P->Branches.size()))
        << " branch condition(s) up to "
        << ore::NV("Distance", P->maxDistance()) << " iterations ahead";
      return R;
    });
    Plans.push_back(std::move(*P));
  }
  if (Plans.empty())
    return false;

  for (LoopPlan &P : Plans)
    materializeValues(P);
  for (LoopPlan &P : Plans) {
    if (P.Speculative) {
#ifndef NDEBUG
      size_t NumBlocks = F.size();
#endif
      transformLoopInPlace(P);
      // Pipelining in place adds and removes no block, so the dominator
      // tree and loop info the other plans rely on stay valid.
      assert(F.size() == NumBlocks &&
             "speculative pipelining changed the control flow");
      ++NumLoopsPipelinedSpeculatively;
    } else {
      transformLoop(P);
    }
    ++NumLoopsPipelined;
    NumBranchesPipelined += P.Branches.size();
  }
#ifndef NDEBUG
  if (verifyFunction(F, &errs()))
    report_fatal_error("RISC-V branch condition pipelining produced invalid IR");
#endif
  return true;
}

INITIALIZE_PASS_BEGIN(RISCVBranchConditionPipelining, DEBUG_TYPE, PASS_NAME,
                      false, false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(OptimizationRemarkEmitterWrapperPass)
INITIALIZE_PASS_END(RISCVBranchConditionPipelining, DEBUG_TYPE, PASS_NAME,
                    false, false)

FunctionPass *llvm::createRISCVBranchConditionPipeliningPass() {
  return new RISCVBranchConditionPipelining();
}
