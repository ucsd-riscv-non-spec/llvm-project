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
// With -riscv-ns-swp-toggles, a value flipped every iteration (x = !x, or
// x ^= m for an invariant m) can be computed ahead like an induction
// variable, and a branch may test it directly. Such a value repeats every two
// iterations whether or not those iterations run, so a loop whose pipelined
// conditions read nothing else is also pipelined in place, even if it has a
// trip count. Its latch stays a counted exit.
//
// The queue operations are the llvm.riscv.ns.cond.* intrinsics. Instruction
// selection turns them into PseudoNSQ* instructions, which RISCVExpandPseudo
// assigns a branch register.
//
// The pass runs before HardwareLoops, which then turns the kernel and the
// remainder loop into hardware loops.
//
// Early-exit chain check thinning is also an opt-in IR transform. Its
// contract mode trusts intact metadata only; callers must provide an image
// buffer of at least x_size*y_size bytes and a LUT covering the full uchar
// difference range. The pass does not prove those allocation extents.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVNSLoopOptions.h"
#include "RISCVSubtarget.h"
#include "RISCVTargetMachine.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopIterator.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsRISCV.h"
#include "llvm/IR/Module.h"
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

static cl::opt<bool> EnableToggles(
    "riscv-ns-swp-toggles", cl::Hidden, cl::init(false),
    cl::desc("Also compute ahead conditions on values flipped every "
             "iteration, such as x = !x, in place when a loop's conditions "
             "read nothing else"));

static cl::opt<bool> EnableSpeculativePipelining(
    "riscv-ns-swp-speculative", cl::Hidden, cl::init(false),
    cl::desc("Also pipeline loops without a computable trip count when the "
             "conditions can be computed for iterations that never run"));

static cl::opt<bool> EnableSusanCornersInterleave2(
    "riscv-ns-susan-corners-interleave2", cl::Hidden, cl::init(false),
    cl::desc("Interleave the first branch conditions of two SUSAN corner "
             "pixels"));

static cl::opt<unsigned> SusanCornersInterleaveFactor(
    "riscv-ns-susan-corners-interleave-factor", cl::Hidden, cl::init(0),
    cl::desc("Interleave SUSAN corner pixels by this factor (3-8); factor 2 "
             "uses the separate interleave2 option"));

static cl::opt<unsigned> SusanCornersLoadHoistDepth(
    "riscv-ns-susan-corners-load-hoist-depth", cl::Hidden, cl::init(0),
    cl::desc("Hoist SUSAN corner neighbor load chains across this many "
             "threshold checks (1-3)"));

static cl::opt<unsigned> ChainLoadHoistDistance(
    "riscv-ns-hoist-chain-loads", cl::Hidden, cl::init(0),
    cl::desc("Hoist safe load slices across this many early-exit checks "
             "(0 disables; maximum 8)"));

// Contract mode trusts only metadata attached by a transform that recognized
// the corresponding bounded access. It assumes the image holds at least
// x_size*y_size bytes and that the LUT covers the full uchar difference
// range. This pass does not prove either allocation extent.
static cl::opt<bool> AssumeChainLoadInBounds(
    "riscv-ns-hoist-assume-chain-inbounds", cl::Hidden, cl::init(false),
    cl::desc("Trust intact chain-inbounds metadata when hoisting or thinning "
             "loads (assumes the image buffer holds at least x_size*y_size "
             "bytes and the LUT covers the full uchar difference range; "
             "this pass does not prove either extent)"));

static cl::opt<unsigned> ThinChainChecks(
    "riscv-ns-thin-checks", cl::Hidden, cl::init(0),
    cl::desc("Check early-exit load chains every G contributions "
             "(0 disables; requires safe speculative loads)"));

static cl::opt<std::string> ThinChainChecksSchedule(
    "riscv-ns-thin-checks-schedule", cl::Hidden, cl::init("uniform"),
    cl::desc("Threshold-check schedule: uniform or histogram"));

// Response-map compaction relies on the interleave transform's chain metadata
// and on SUSAN's image allocation contract: at least x_size*y_size bytes.
// The pass does not prove that extent.
static cl::opt<bool> SusanCornersResponseCompact(
    "riscv-ns-susan-response-compact", cl::Hidden, cl::init(false),
    cl::desc("Compact SUSAN corner survivors before local-maximum checks "
             "(requires interleave chain metadata; assumes the image buffer "
             "holds at least x_size*y_size bytes, which this pass does not "
             "prove)"));

// Each entry of a loop without a trip count computes this many conditions
// for iterations that may never run, so it is kept shorter than the distance
// of counted loops.
static cl::opt<unsigned> SpeculativePipelineDistance(
    "riscv-ns-swp-speculative-distance", cl::Hidden, cl::init(4),
    cl::desc("Number of iterations ahead of its branch that a loop pipelined "
             "without a trip count computes a condition"));

STATISTIC(NumLoopsPipelined, "Number of loops whose conditions were pipelined");
STATISTIC(NumBranchesPipelined, "Number of branches given a condition queue");
STATISTIC(NumChainLoadHoists, "Number of early-exit load slices hoisted");
STATISTIC(NumChainLoadHoistRejectDereferenceability,
          "Load slices rejected because speculative dereferenceability was "
          "not proven");
STATISTIC(NumChainLoadHoistRejectMemory,
          "Load slices rejected because of an intervening memory effect");
STATISTIC(NumChainLoadHoistRejectControl,
          "Load slices rejected because of control-flow or loop guards");
STATISTIC(NumChainLoadHoistRejectPressure,
          "Load slices rejected by the live-value pressure limit");
STATISTIC(NumChainLoadHoistRejectDependence,
          "Load slices rejected because their address depends on the "
          "accumulator");
STATISTIC(NumChainChecksRemoved,
          "Intermediate early-exit threshold checks removed");
STATISTIC(NumChainCheckThinRejectContribution,
          "Threshold checks kept because contributions are not nonnegative");
STATISTIC(NumChainCheckThinRejectOverflow,
          "Threshold checks kept because accumulator overflow is possible");
STATISTIC(NumChainCheckThinRejectMemory,
          "Threshold checks kept because of intervening side effects");
STATISTIC(NumChainCheckThinRejectInBounds,
          "Threshold checks kept because speculative loads are not safe");
STATISTIC(NumChainCheckThinRejectLiveOut,
          "Threshold checks kept because intermediate values are live on exit");
STATISTIC(NumChainCheckThinRejectPressure,
          "Threshold checks kept because of live-value pressure");
STATISTIC(NumChainCheckThinRejectMonotonic,
          "Threshold checks kept because exit conditions are not monotone");
STATISTIC(NumChainCheckThinRejectControl,
          "Threshold checks kept because the chain or loop control is unsafe");
STATISTIC(NumChainCheckThinRejectShortChain,
          "Chains too short to thin at the requested spacing");
STATISTIC(NumSusanResponseCompactions,
          "Number of SUSAN response-map scans compacted");
STATISTIC(NumSusanResponseCompactRejectContract,
          "Response-map compactions rejected because chain metadata is "
          "missing");
STATISTIC(NumSusanResponseCompactRejectPattern,
          "Response-map compactions rejected because the IR shape is "
          "unsupported");
STATISTIC(NumLoopsPipelinedSpeculatively,
          "Number of loops without a trip count whose conditions were "
          "pipelined");
STATISTIC(NumToggleLoopsPipelined,
          "Number of loops with a trip count whose toggle conditions were "
          "pipelined in place");

/// The FIFO of a branch register holds 63 conditions, and a pipelined
/// branch's holds its distance plus the one pushed before it is popped.
static constexpr unsigned MaxDistance = 62;

namespace {

static MDNode *makeChainInBoundsContract(
    LLVMContext &Ctx, StringRef Kind, unsigned BaseArg, unsigned ImageArg,
    unsigned XSizeArg, unsigned YSizeArg, Loop *InnerLoop, Loop *OuterLoop,
    int64_t RowStart, int64_t RowEndTrim, int64_t ColumnStart,
    int64_t ColumnEndTrim, ArrayRef<int64_t> Offsets);

/// A header phi whose value in iteration I is Start + I * Step.
struct InductionVar {
  PHINode *Phi;
  Value *Start;
  /// Null for a shift recurrence or a toggle.
  const SCEV *Step;
  /// Step, materialized in the preheader.
  Value *StepV = nullptr;
  /// For a shift recurrence, whose value in iteration I is Start shifted by
  /// I * ShiftAmount: the shift (LShr, AShr or Shl) and its amount.
  Instruction::BinaryOps ShiftOp = Instruction::BinaryOpsEnd;
  unsigned ShiftAmount = 0;

  /// For a toggle, whose value in iteration I is Start for even I and
  /// Start ^ ToggleMask for odd I: the loop-invariant mask.
  Value *ToggleMask = nullptr;

  bool isShift() const { return ShiftOp != Instruction::BinaryOpsEnd; }
  bool isToggle() const { return ToggleMask != nullptr; }
};

/// A branch whose condition the pipelined loop computes ahead of it.
struct PipelinedBranch {
  BranchInst *Br;
  /// The instructions of the loop the condition is computed by, operands
  /// before users. The condition is last.
  SmallVector<Instruction *, 8> Slice;
  /// Indices into LoopPlan::IVs of the induction variables the slice reads.
  SmallVector<unsigned, 2> IVs;
  /// Affine IV advances for this branch's distance, materialized once in the
  /// preheader instead of multiplying the step on every kernel iteration.
  SmallVector<Value *, 2> StepOffsets;
  /// How many iterations ahead of the branch the condition is computed.
  unsigned Distance = 0;
  unsigned Queue = 0;
};

struct LoopPlan {
  RISCVNS::SWPSettings Settings;
  bool Annotated = false;
  Loop *L;
  BasicBlock *Preheader;
  BasicBlock *Header;
  BasicBlock *Latch;
  /// The loop's blocks in reverse post-order.
  SmallVector<BasicBlock *, 8> Blocks;
  /// The trip count, as an XLen integer. Null if it is not computable.
  const SCEV *TripCount = nullptr;
  /// TripCount, materialized in the preheader. Null when Speculative.
  Value *TripCountV = nullptr;
  /// Whether the loop's conditions are computed for iterations that may
  /// never run: it has no computable trip count, or its conditions read only
  /// toggles. The loop is then pipelined in place, without a kernel copy, a
  /// guard or a drain loop, and every condition must be safe to compute for
  /// any iteration.
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

  // Why canPipelineLoopBody or buildSlice said no, for the missed remarks
  std::string Why;
  bool reject(const Twine &W) {
    Why = W.str();
    return false;
  }
  void missedLoop(Loop *L, StringRef Name, const Twine &Msg) {
    ORE->emit([&]() {
      return OptimizationRemarkMissed(DEBUG_TYPE, Name, L->getStartLoc(),
                                      L->getHeader())
             << ("not pipelining branch conditions: " + Msg).str();
    });
  }
  void missedBranch(BranchInst *Br, StringRef Name, const Twine &Msg) {
    ORE->emit([&]() {
      return OptimizationRemarkMissed(DEBUG_TYPE, Name, Br->getDebugLoc(),
                                      Br->getParent())
             << ("not pipelining this branch's condition: " + Msg).str();
    });
  }
  IntegerType *XLenTy = nullptr;
  StringRef FailureReason;

  std::optional<LoopPlan> analyzeLoop(Loop *L, RISCVNS::SWPSettings Settings,
                                      bool Annotated);
  void missed(Loop *L, StringRef Name, StringRef Reason);
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
  bool interleaveSusanCorners(Loop *L, unsigned Queue);
  bool interleaveSusanCornersFactor(Loop *L, unsigned Queue, unsigned Factor);
  bool hoistSusanCornerLoads(Loop *L, unsigned Depth);
  bool annotateSusanChainInBounds(Loop *L, std::string &FailureReason);
  Value *cloneCondition(IRBuilder<> &B, const PipelinedBranch &PB,
                        ValueToValueMapTy &Map, const Twine &Suffix,
                        bool Speculative, bool Force);
  void transformLoopInPlace(LoopPlan &P);
  void emitPush(IRBuilder<> &B, unsigned Queue, Value *Cond);
};

} // end anonymous namespace

char RISCVBranchConditionPipelining::ID = 0;

static std::string loopName(const Loop &L) {
  if (L.getHeader()->hasName())
    return L.getHeader()->getName().str();
  std::string Name;
  raw_string_ostream OS(Name);
  L.getHeader()->printAsOperand(OS, false);
  return Name;
}

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

/// "a call to f at line 12", "'load' at line 12": what an instruction is, for
/// remarks.
static std::string describe(const Instruction &I) {
  std::string S;
  raw_string_ostream OS(S);
  if (auto *CB = dyn_cast<CallBase>(&I)) {
    if (const Function *F = CB->getCalledFunction())
      OS << "a call to " << F->getName();
    else
      OS << "an indirect call";
  } else {
    OS << "'" << I.getOpcodeName() << "'";
  }
  if (const DebugLoc &DL = I.getDebugLoc())
    OS << " at line " << DL.getLine();
  return S;
}

/// Whether \p I may be evaluated for a later iteration: it computes its
/// result from its operands alone (or, for a load, from memory the caller
/// checks), so it can run again elsewhere without changing the program.
static bool isClonableIntoCondition(const Instruction &I, bool Force) {
  // PHIs require a future recurrence (handled separately), and terminators,
  // EH pads and token results cannot be emitted as a straight-line condition.
  if (Force)
    return !I.isTerminator() && !I.isEHPad() && !I.getType()->isTokenTy();
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
  Why = "unknown";
  Loop *L = P.L;
  unsigned Size = 0;
  for (BasicBlock *BB : P.Blocks) {
    // The cloner does not remap exceptional entries or block addresses.
    if (BB->isEHPad() || BB->hasAddressTaken())
      return reject("a block is an exception handler or has its address taken");
    for (Instruction &I : *BB) {
      if (I.isDebugOrPseudoInst())
        continue;
      if (++Size > P.Settings.MaxLoopSize && !P.Settings.Force) {
        FailureReason = "loop size limit";
        LLVM_DEBUG(dbgs() << "  loop is too large to duplicate\n");
        return reject("the loop has more than " + Twine(P.Settings.MaxLoopSize) +
                      " instructions (-riscv-ns-swp-max-loop-size)");
      }
      // The kernel is a copy of the loop.
      if (I.getType()->isTokenTy())
        return reject("the loop contains " + describe(I) +
                      ", which cannot be duplicated");
      // The emitter only clones branch/switch CFGs. This is a construction
      // requirement even when force permits duplicated or reordered effects.
      if (I.isTerminator() && !isa<BranchInst>(I) && !isa<SwitchInst>(I))
        return reject("the loop contains " + describe(I));
      if (auto *II = dyn_cast<IntrinsicInst>(&I))
        if (II->getIntrinsicID() == Intrinsic::riscv_ns_cond_init ||
            II->getIntrinsicID() == Intrinsic::riscv_ns_cond_push ||
            II->getIntrinsicID() == Intrinsic::riscv_ns_cond_pop)
          return reject("the loop already uses a condition queue");
      if (P.Settings.Force)
        continue;
      if (isa<AllocaInst>(I))
        return reject("the loop contains " + describe(I) +
                      ", which cannot be duplicated");
      if (auto *CB = dyn_cast<CallBase>(&I)) {
        // A call could write anything a condition reads, and a call to a
        // pure intrinsic is the only kind certain to stay a non-call.
        auto *II = dyn_cast<IntrinsicInst>(CB);
        if (!II || II->isConvergent() || II->cannotDuplicate())
          return reject("the loop contains " + describe(I) +
                        " (a call could write what a condition reads)");
        // A lifetime marker could let a stack slot be reused within an
        // iteration, under a load that now reads it earlier.
        if (II->isLifetimeStartOrEnd())
          return reject("the loop contains a lifetime marker (a local "
                        "variable's scope ends inside the loop)");
        if (isa<DbgInfoIntrinsic>(II) || II->isAssumeLikeIntrinsic())
          continue;
        if (!II->doesNotAccessMemory() || II->mayHaveSideEffects())
          return reject("the loop contains " + describe(I) +
                        ", which accesses memory");
        continue;
      }
      if (auto *Store = dyn_cast<StoreInst>(&I)) {
        if (!Store->isSimple())
          return reject("the loop contains a volatile or atomic store (" +
                        describe(I) + ")");
        continue;
      }
      if (auto *Load = dyn_cast<LoadInst>(&I)) {
        if (!Load->isSimple())
          return reject("the loop contains a volatile or atomic load (" +
                        describe(I) + ")");
        continue;
      }
      // Fences, atomics, va_arg...
      if (I.mayReadOrWriteMemory() || I.mayHaveSideEffects())
        return reject("the loop contains " + describe(I) +
                      ", which accesses memory or has side effects");
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
        return reject("a block can run more than once per iteration "
                      "(irreducible control flow)");
  return true;
}

std::optional<unsigned>
RISCVBranchConditionPipelining::findInductionVar(LoopPlan &P, PHINode *Phi) {
  for (auto [Idx, IV] : enumerate(P.IVs))
    if (IV.Phi == Phi)
      return Idx;

  if (Phi->getParent() != P.Header)
    return std::nullopt;

  // x = x ^ m, the same m every iteration: two iterations on, x is what it
  // is now. An i1 toggle is also an affine recurrence of step 1, so this
  // comes first.
  Value *Mask;
  if (Phi->getType()->isIntegerTy() &&
      Phi->getNumIncomingValues() == 2 &&
      match(Phi->getIncomingValueForBlock(P.Latch),
            m_c_Xor(m_Specific(Phi), m_Value(Mask))) &&
      P.L->isLoopInvariant(Mask)) {
    // In particular, do not let a forced i1 toggle take the affine fallback:
    // force cannot turn on the toggle feature when it was explicitly disabled.
    if (!P.Settings.Toggles && P.Settings.Force)
      return std::nullopt;
    if (P.Settings.Toggles) {
      InductionVar IV = {Phi, Phi->getIncomingValueForBlock(P.Preheader), nullptr,
                         nullptr};
      IV.ToggleMask = Mask;
      P.IVs.push_back(IV);
      return P.IVs.size() - 1;
    }
  }

  if (!SE->isSCEVable(Phi->getType()))
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
    if (!P.Settings.ShiftRecurrences || !Phi->getType()->isIntegerTy() ||
        !Shift || !P.L->contains(Shift) ||
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
      !(P.Settings.Force
            ? SE->dominates(Step, P.Preheader)
            : Expander.isSafeToExpandAt(Step, P.Preheader->getTerminator())))
    return std::nullopt;

  P.IVs.push_back(
      {Phi, Phi->getIncomingValueForBlock(P.Preheader), Step, nullptr});
  return P.IVs.size() - 1;
}

bool RISCVBranchConditionPipelining::buildSlice(LoopPlan &P,
                                                PipelinedBranch &PB) {
  Why = "unknown";
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
      if (!IV) {
        FailureReason = "cannot construct future recurrence values with the "
                        "enabled features";
        return reject("it depends on a value carried from the previous "
                      "iteration (" +
                      Phi->getName() + ") that is not an induction variable");
      }
      IVs.insert(*IV);
      return true;
    }
    if (!isClonableIntoCondition(*I, P.Settings.Force)) {
      FailureReason = "condition contains an operation that cannot be cloned";
      return reject("it depends on " + describe(*I) +
                    ", which cannot be computed again for a later iteration");
    }
    // Without a trip count, conditions are also computed for iterations that
    // never run, so each instruction must be safe to execute on any operands.
    // No load is: its address may not be valid in such an iteration.
    if (P.Speculative && !P.Settings.Force &&
        (isa<LoadInst>(I) || !isSafeToSpeculativelyExecute(I))) {
      FailureReason = "condition is not safe to speculate";
      return reject("it needs " + describe(*I) +
                    ", and without a trip count conditions are also computed "
                    "for iterations that never run");
    }
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

  if (InSlice.size() > P.Settings.MaxSliceSize && !P.Settings.Force) {
    FailureReason = "condition slice size limit";
    return reject("it needs " + Twine(InSlice.size()) +
                  " instructions, more than " + Twine(P.Settings.MaxSliceSize) +
                  " (-riscv-ns-swp-max-slice-size)");
  }
  // An invariant condition gains nothing from being queued early, and a
  // condition that is itself a header phi is only handled if it is a toggle.
  if (!P.Settings.Force && InSlice.empty() &&
      (IVs.empty() || !P.IVs[IVs[0]].isToggle()))
    return reject("the condition is loop-invariant or itself a value carried "
                  "between iterations");

  // Everything in the slice dominates the branch, so program order puts
  // every operand before its users.
  for (BasicBlock *BB : P.Blocks)
    for (Instruction &I : *BB)
      if (InSlice.contains(&I))
        PB.Slice.push_back(&I);
  assert(PB.Slice.empty() || PB.Slice.back() == PB.Br->getCondition());
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
  if (P.Settings.Force)
    return Distance;
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
static bool exitsOnlyAtLatch(const Loop &L, bool Force) {
  SmallVector<BasicBlock *, 4> Exiting;
  L.getExitingBlocks(Exiting);
  for (BasicBlock *BB : Exiting) {
    if (BB == L.getLoopLatch())
      continue;
    for (BasicBlock *Succ : successors(BB))
      if (!L.contains(Succ) &&
          (!Succ->phis().empty() ||
           (!Force &&
            !isa<UnreachableInst>(Succ->getFirstNonPHIOrDbgOrLifetime()))))
        return false;
  }
  return true;
}

std::optional<LoopPlan> RISCVBranchConditionPipelining::analyzeLoop(
    Loop *L, RISCVNS::SWPSettings Settings, bool Annotated) {
  LLVM_DEBUG(dbgs() << "Loop " << L->getHeader()->getName() << ":\n");

  LoopPlan P;
  P.Settings = Settings;
  P.Annotated = Annotated;
  FailureReason = "unsupported loop shape: requires an innermost simplified "
                  "loop with a latch exit and mappable exit values";
  P.L = L;
  P.Preheader = L->getLoopPreheader();
  P.Header = L->getHeader();
  P.Latch = L->getLoopLatch();
  if (!L->isInnermost() || !L->isLoopSimplifyForm() || !P.Latch ||
      !L->isLoopExiting(P.Latch) || !exitsOnlyAtLatch(*L, Settings.Force) ||
      (Settings.Force && !exitsOnlyAtLatch(*L, false) &&
       !L->isLCSSAForm(*DT))) {
    LLVM_DEBUG(dbgs() << "  not an innermost loop exiting at its latch\n");
    if (!L->isInnermost())
      missedLoop(L, "NotInnermost", "the loop contains another loop");
    else if (!L->isLoopSimplifyForm() || !P.Latch)
      missedLoop(L, "NotSimplified",
                 "the loop has no preheader or no single latch");
    else
      missedLoop(L, "EarlyExit",
                 "the loop can exit other than at its latch (a break, a "
                 "return, or a call that does not return)");
    return std::nullopt;
  }
  auto *LatchBr = dyn_cast<BranchInst>(P.Latch->getTerminator());
  if (!LatchBr || !LatchBr->isConditional() ||
      !isa<BranchInst>(P.Preheader->getTerminator())) {
    missedLoop(L, "NotPlainBranches",
               "the latch or the preheader does not end in a plain branch");
    return std::nullopt;
  }

  LoopBlocksRPO RPOT(L);
  RPOT.perform(LI);
  P.Blocks.assign(RPOT.begin(), RPOT.end());
  FailureReason = "loop body cannot be cloned with valid control flow and "
                  "queue push/pop structure";
  if (!canPipelineLoopBody(P)) {
    LLVM_DEBUG(dbgs() << "  loop body cannot be pipelined\n");
    missedLoop(L, "BodyNotPipelinable", Why);
    return std::nullopt;
  }

  const SCEV *BTC = SE->getExitCount(L, P.Latch);
  if (isa<SCEVCouldNotCompute>(BTC) || !BTC->getType()->isIntegerTy() ||
      BTC->getType()->getIntegerBitWidth() > XLenTy->getBitWidth()) {
    if (!P.Settings.Speculative) {
      FailureReason = "trip count is not computable and speculative pipelining "
                      "is disabled";
      LLVM_DEBUG(dbgs() << "  trip count is not computable\n");
      missedLoop(L, "NoTripCount",
                 "the trip count cannot be computed (-riscv-ns-swp-speculative "
                 "pipelines such loops in place)");
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
    if (!(P.Settings.Force ? SE->dominates(P.TripCount, P.Preheader)
                           : Expander.isSafeToExpandAt(
                                 P.TripCount, P.Preheader->getTerminator()))) {
      FailureReason = "cannot materialize the trip count in the preheader";
      missedLoop(L, "TripCountNotExpandable",
                 "the trip count cannot be computed before the loop");
      return std::nullopt;
    }
  }

  unsigned RequestedDistance =
      P.Speculative ? P.Settings.SpeculativeDistance : P.Settings.Distance;
  unsigned Distance = std::min(RequestedDistance, MaxDistance);
  if (P.Annotated && Distance < RequestedDistance)
    ORE->emit([&]() {
      return OptimizationRemarkAnalysis(DEBUG_TYPE, "QueueBound",
                                        L->getStartLoc(), P.Header)
             << "loop " << ore::NV("Loop", loopName(*P.L))
             << ": requested distance "
             << ore::NV("Requested", RequestedDistance)
             << ", effective distance capped at "
             << ore::NV("Effective", Distance) << " by the hardware queue bound";
    });
  unsigned MaxTripCount = SE->getSmallConstantMaxTripCount(L);
  if (MaxTripCount && MaxTripCount <= Distance && !P.Settings.Force) {
    FailureReason = "short trip count profitability filter";
    LLVM_DEBUG(dbgs() << "  never runs more than " << MaxTripCount
                      << " iterations\n");
    missedLoop(L, "TooFewIterations",
               "the loop never runs more than " + Twine(MaxTripCount) +
                   " iterations, no more than the distance (" +
                   Twine(Distance) + ")");
    return std::nullopt;
  }

  uint64_t PrologueSize = 0;
  FailureReason = "no branch with the execution structure needed to pair "
                  "queue pushes and pops";
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
    if (!Br || !Br->isConditional() ||
        Br->getSuccessor(0) == Br->getSuccessor(1))
      continue;
    if (!DT->dominates(BB, P.Latch)) {
      missedBranch(Br, "NotEveryIteration",
                   "the branch does not run in every iteration (it is inside "
                   "an if); only branches that do are pipelined");
      continue;
    }

    PipelinedBranch PB;
    PB.Br = Br;
    if (!buildSlice(P, PB)) {
      LLVM_DEBUG(dbgs() << "  cannot compute ahead: " << *Br << "\n");
      missedBranch(Br, "ConditionNotComputable",
                   "its condition cannot be computed ahead: " + Why);
      continue;
    }
    // Each condition the prologue queues costs a copy of the slice and a
    // push.
    unsigned PushSize = PB.Slice.size() + 1;
    unsigned Fit =
        P.Settings.Force
            ? Distance
            : std::min<uint64_t>(
                  Distance, (P.Settings.MaxPrologueSize -
                             std::min<uint64_t>(PrologueSize,
                                                P.Settings.MaxPrologueSize)) /
                                PushSize);
    PB.Distance = findSafeDistance(P, PB, Fit);
    if (P.Annotated && PB.Distance < Distance) {
      StringRef Reason =
          Fit < Distance ? "prologue size limit" : "memory dependence";
      ORE->emit([&]() {
        return OptimizationRemarkAnalysis(DEBUG_TYPE, "DistanceReduced",
                                          Br->getDebugLoc(), BB)
               << "loop " << ore::NV("Loop", loopName(*P.L))
               << ": requested distance " << ore::NV("Requested", Distance)
               << ", effective distance " << ore::NV("Effective", PB.Distance)
               << " (" << ore::NV("Reason", Reason) << ")";
      });
    }
    if (!PB.Distance) {
      FailureReason = Fit == 0 ? "prologue size limit or zero in-place distance"
                               : "memory dependence";
      LLVM_DEBUG(dbgs() << "  a store could clobber what the condition of "
                        << *Br << " reads\n");
      missedBranch(Br, "StoreMayClobber",
                   "a store in the loop may change what its condition reads "
                   "in a later iteration (no distance is safe)");
      continue;
    }
    PrologueSize += PB.Distance * PushSize;
    LLVM_DEBUG(dbgs() << "  pipelining " << *Br << " " << PB.Distance
                      << " iterations ahead\n");
    P.Branches.push_back(std::move(PB));
  }

  if (P.Branches.empty()) {
    missedLoop(L, "NoBranchPipelinable",
               "no branch condition in the loop can be pipelined (see the "
               "remarks on its branches)");
    return std::nullopt;
  }

  // A toggle repeats every two iterations whether or not they run, so a
  // counted loop whose conditions read nothing else needs no kernel that
  // stops early: it is pipelined in place, like a loop without a trip count
  // and at the same distance. Its latch was skipped above and stays a
  // counted exit.
  unsigned InPlaceDistance =
      std::min<unsigned>(P.Settings.SpeculativeDistance, MaxDistance);
  if (!P.Speculative && InPlaceDistance &&
      all_of(P.Branches, [&](const PipelinedBranch &PB) {
        return !PB.IVs.empty() && all_of(PB.IVs, [&](unsigned Idx) {
          return P.IVs[Idx].isToggle();
        }) && all_of(PB.Slice, [&](const Instruction *I) {
          return P.Settings.Force ||
                 (!isa<LoadInst>(I) && isSafeToSpeculativelyExecute(I));
        });
      })) {
    P.Speculative = true;
    for (PipelinedBranch &PB : P.Branches) {
      PB.Distance = std::min(PB.Distance, InPlaceDistance);
      if (P.Annotated && PB.Distance < P.Settings.SpeculativeDistance)
        ORE->emit([&]() {
          return OptimizationRemarkAnalysis(DEBUG_TYPE, "ToggleDistance",
                                            PB.Br->getDebugLoc(),
                                            PB.Br->getParent())
                 << "loop " << ore::NV("Loop", loopName(*P.L))
                 << ": requested in-place distance "
                 << ore::NV("Requested", P.Settings.SpeculativeDistance)
                 << ", effective distance " << ore::NV("Effective", PB.Distance)
                 << "; counted toggle lookahead also respects the counted "
                    "distance and earlier distance bounds";
        });
    }
    LLVM_DEBUG(dbgs() << "  every condition reads only toggles, pipelining "
                         "in place "
                      << P.maxDistance() << " iterations ahead\n");
  }
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
    // A shift recurrence or a toggle has no step to materialize.
    if (!Used.contains(Idx) || IV.isShift() || IV.isToggle())
      continue;
    if (auto *C = dyn_cast<SCEVConstant>(IV.Step))
      IV.StepV = C->getValue();
    else
      IV.StepV = Expander.expandCodeFor(IV.Step, IV.Step->getType(), InsertPt);
  }

  IRBuilder<> B(InsertPt);
  for (PipelinedBranch &PB : P.Branches) {
    PB.StepOffsets.resize(PB.IVs.size());
    for (auto [I, Idx] : enumerate(PB.IVs)) {
      const InductionVar &IV = P.IVs[Idx];
      if (IV.isShift() || IV.isToggle())
        continue;
      PB.StepOffsets[I] = B.CreateMul(
          IV.StepV,
          ConstantInt::get(IV.StepV->getType(), PB.Distance),
          "swp.step.offset");
    }
  }
}

#ifndef NDEBUG
static APInt applyShift(Instruction::BinaryOps Op, const APInt &X,
                        uint64_t Amount) {
  switch (Op) {
  case Instruction::LShr:
    return X.lshr(Amount);
  case Instruction::AShr:
    return X.ashr(Amount);
  case Instruction::Shl:
    return X.shl(Amount);
  default:
    llvm_unreachable("not a shift recurrence");
  }
}

/// Checks that \p Result, emitted for a shift recurrence \p Iterations
/// iterations after \p Base, equals shifting \p Base by the recurrence's
/// amount \p Iterations times, one shift at a time. It reads back what was
/// emitted, a constant or one shift of \p Base by a constant, and compares on
/// sample values, or on \p Base itself when it is a constant.
static void checkShiftAdvance(const InductionVar &IV, Value *Base,
                              unsigned Iterations, Value *Result) {
  unsigned Width = Base->getType()->getScalarSizeInBits();
  SmallVector<APInt, 8> Samples;
  if (auto *C = dyn_cast<ConstantInt>(Base))
    Samples.push_back(C->getValue());
  else if (isa<Constant>(Base))
    return;
  else
    Samples = {APInt::getZero(Width),
               APInt::getOneBitSet(Width, 0),
               APInt::getAllOnes(Width),
               APInt::getSignedMinValue(Width),
               APInt::getSignedMaxValue(Width),
               APInt(64, 0x5a5a5a5a5a5a5a5aULL).zextOrTrunc(Width),
               APInt(64, 0x8badf00ddeadbeefULL).zextOrTrunc(Width)};
  auto *ResultC = dyn_cast<ConstantInt>(Result);
  auto *ResultShift = dyn_cast<BinaryOperator>(Result);
  assert((ResultC || (ResultShift && ResultShift->getOperand(0) == Base &&
                      isa<ConstantInt>(ResultShift->getOperand(1)))) &&
         "advanced shift recurrence is neither a constant nor one shift");
  for (const APInt &X : Samples) {
    APInt Expected = X;
    for (unsigned K = 0; K < Iterations; ++K)
      Expected = applyShift(IV.ShiftOp, Expected, IV.ShiftAmount);
    APInt Got =
        ResultC ? ResultC->getValue()
                : applyShift(
                      ResultShift->getOpcode(), X,
                      cast<ConstantInt>(ResultShift->getOperand(1))
                          ->getZExtValue());
    assert(Got == Expected &&
           "advanced shift recurrence differs from shifting once per "
           "iteration");
    (void)Got;
  }
}
#endif

/// Returns the value \p IV has \p Iterations iterations after it has
/// \p Base.
static Value *advanceInductionVar(IRBuilder<> &B, const InductionVar &IV,
                                  Value *Base, unsigned Iterations,
                                  const Twine &Name,
                                  Value *StepOffset = nullptr) {
  if (Iterations == 0)
    return Base;
  // Flipping the same bits twice restores them.
  if (IV.isToggle())
    return Iterations % 2 ? B.CreateXor(Base, IV.ToggleMask, Name) : Base;
  if (IV.isShift()) {
    // Shifting by a then by b is shifting by a + b, and a shift by the width
    // or more leaves nothing of a logical shift and only the sign of an
    // arithmetic one. The shift emitted is therefore always less than the
    // width, which keeps it safe to compute for any iteration.
    Type *Ty = Base->getType();
    unsigned Width = Ty->getScalarSizeInBits();
    uint64_t Total = uint64_t(IV.ShiftAmount) * Iterations;
    Value *Result;
    if (Total >= Width && IV.ShiftOp != Instruction::AShr) {
      Result = Constant::getNullValue(Ty);
    } else {
      Total = std::min<uint64_t>(Total, Width - 1);
      assert(Total > 0 && Total < Width && "shift out of range");
      Result =
          B.CreateBinOp(IV.ShiftOp, Base, ConstantInt::get(Ty, Total), Name);
    }
#ifndef NDEBUG
    checkShiftAdvance(IV, Base, Iterations, Result);
#endif
    return Result;
  }
  assert(IV.StepV && "affine induction variable without a materialized step");
  Value *Offset = StepOffset;
  if (!Offset)
    Offset = B.CreateMul(
        IV.StepV, ConstantInt::get(IV.StepV->getType(), Iterations));
  if (Base->getType()->isPointerTy())
    return B.CreatePtrAdd(Base, Offset, Name);
  return B.CreateAdd(Base, Offset, Name);
}

Value *RISCVBranchConditionPipelining::cloneCondition(
    IRBuilder<> &B, const PipelinedBranch &PB, ValueToValueMapTy &Map,
    const Twine &Suffix, bool Speculative, bool Force) {
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
    assert((Force || !Speculative ||
            (!isa<LoadInst>(C) && isSafeToSpeculativelyExecute(C))) &&
           "speculative condition clone is not safe to execute anywhere");
    Map[I] = C;
  }
  // A condition that is a toggle itself has no slice: it is the toggle's
  // value in the iteration Map describes.
  Value *Condition = Map.lookup(PB.Br->getCondition());
  return Condition ? Condition : PB.Br->getCondition();
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

bool RISCVBranchConditionPipelining::hoistSusanCornerLoads(Loop *L,
                                                           unsigned Depth) {
  auto Reject = [&](StringRef Reason) {
    FailureReason = Reason;
    return false;
  };
  if (!L || Depth < 1 || Depth > 3 || !L->isInnermost() ||
      L->getHeader()->getName() != "for.body14")
    return Reject("the requested loop or depth did not match");

  Function *Fn = L->getHeader()->getParent();
  BasicBlock *Latch = L->getLoopLatch();
  if (!Latch || Fn->arg_size() <= 3)
    return Reject("the corner scan loop has no recognized latch or threshold");

  struct ThresholdCheck {
    BranchInst *Branch;
    ICmpInst *Compare;
    BasicBlock *Continue;
  };
  SmallVector<ThresholdCheck, 20> Checks;
  DenseMap<BasicBlock *, unsigned> BlockOrder;
  unsigned NextBlockOrder = 0;
  for (BasicBlock &BB : *Fn) {
    BlockOrder[&BB] = NextBlockOrder++;
    if (!L->contains(&BB))
      continue;

    auto *Br = dyn_cast<BranchInst>(BB.getTerminator());
    auto *Cmp = Br && Br->isConditional()
                    ? dyn_cast<ICmpInst>(Br->getCondition())
                    : nullptr;
    if (!Cmp || Br->getSuccessor(1) != Latch ||
        (Cmp->getOperand(0) != Fn->getArg(3) &&
         Cmp->getOperand(1) != Fn->getArg(3)))
      continue;
    BasicBlock *Continue = Br->getSuccessor(0);
    if (Continue == Latch || !L->contains(Continue))
      return false;
    Checks.push_back({Br, Cmp, Continue});
  }
  if (Checks.empty() || Checks.size() > 19)
    return Reject("the number of threshold checks was not recognized");

  for (unsigned I = 1; I < Checks.size(); ++I)
    if (!DT->dominates(Checks[I - 1].Continue,
                       Checks[I].Branch->getParent()))
      return Reject("the threshold checks do not form one continuation chain");

  auto IsInputLoad = [&](LoadInst *Load) {
    if (!Load->isSimple())
      return false;
    Value *Object = getUnderlyingObject(Load->getPointerOperand());
    while (auto *GEP = dyn_cast<GEPOperator>(Object))
      Object = getUnderlyingObject(GEP->getPointerOperand());
    auto *Arg = dyn_cast<Argument>(Object);
    if (!Arg || (Arg->getArgNo() != 0 && Arg->getArgNo() != 2))
      return false;
    unsigned ArgNo = Arg->getArgNo();
    return Fn->getAttributes().hasParamAttr(ArgNo, Attribute::NoAlias) &&
           Fn->getAttributes().hasParamAttr(ArgNo, Attribute::ReadOnly);
  };

  DenseMap<Instruction *, unsigned> InstructionOrder;
  unsigned NextInstructionOrder = 0;
  for (Instruction &I : instructions(*Fn))
    InstructionOrder[&I] = NextInstructionOrder++;

  DenseMap<Instruction *, BasicBlock *> MoveTo;
  for (unsigned Source = 0; Source < Checks.size(); ++Source) {
    if (Source + 1 < Depth)
      continue;
    unsigned Destination = Source + 1 - Depth;
    ICmpInst *Anchor = Checks[Destination].Compare;

    // Every block on this true-edge chain must be free of writes and calls.
    // The hoisted image/LUT reads can then execute on an earlier threshold's
    // exit path without changing observable state.
    SmallPtrSet<BasicBlock *, 8> VisitedBlocks;
    for (unsigned I = Destination; I <= Source; ++I) {
      for (BasicBlock *BB :
           {Checks[I].Branch->getParent(), Checks[I].Continue}) {
        if (!VisitedBlocks.insert(BB).second)
          continue;
        for (Instruction &I : *BB)
          if (I.mayWriteToMemory() || isa<CallBase>(I))
                return Reject("a store or call lies on the crossed path");
      }
    }

    SmallPtrSet<Instruction *, 32> Slice;
    auto AddSlice = [&](auto &&Self, Value *V) -> bool {
      auto *I = dyn_cast<Instruction>(V);
      if (!I || DT->dominates(I, Anchor))
        return true;
      if (!L->contains(I->getParent()) || isa<PHINode>(I) ||
          !isClonableIntoCondition(*I, /*Force=*/false))
        return Reject("a load dependency is not movable within the scan loop");
      if (auto *Load = dyn_cast<LoadInst>(I)) {
        if (!IsInputLoad(Load))
          return Reject("a load is not from the readonly image or LUT");
      } else if (!isSafeToSpeculativelyExecute(I)) {
        return Reject("a load-address instruction is not safe to speculate");
      }
      if (!Slice.insert(I).second)
        return true;
      for (Value *Operand : I->operands())
        if (!Self(Self, Operand))
          return false;
      return true;
    };

    bool HasLoad = false;
    for (Instruction &I : *Checks[Source].Continue) {
      auto *Load = dyn_cast<LoadInst>(&I);
      if (!Load)
        continue;
      HasLoad = true;
      if (!AddSlice(AddSlice, Load))
        return false;
      for (User *U : Load->users())
        if (auto *ZExt = dyn_cast<ZExtInst>(U))
          if (!AddSlice(AddSlice, ZExt))
            return false;
    }
    if (!HasLoad)
      continue;

    for (Instruction *I : Slice) {
      auto It = MoveTo.find(I);
      if (It == MoveTo.end() ||
          BlockOrder[Checks[Destination].Compare->getParent()] <
              BlockOrder[It->second])
        MoveTo[I] = Checks[Destination].Compare->getParent();
    }
  }
  if (MoveTo.empty())
    return Reject("no independent image or LUT loads were found to hoist");

  DenseMap<BasicBlock *, SmallVector<Instruction *, 16>> ByDestination;
  for (auto &Entry : MoveTo)
    ByDestination[Entry.second].push_back(Entry.first);
  for (auto &Entry : ByDestination) {
    llvm::sort(Entry.second, [&](Instruction *A, Instruction *B) {
      return InstructionOrder[A] < InstructionOrder[B];
    });
    auto *Anchor = cast<ICmpInst>(Checks.front().Compare);
    for (const ThresholdCheck &Check : Checks)
      if (Check.Compare->getParent() == Entry.first) {
        Anchor = Check.Compare;
        break;
      }
    for (Instruction *I : Entry.second)
      I->moveBefore(Anchor->getIterator());
  }

  ORE->emit([&]() {
    return OptimizationRemark(DEBUG_TYPE, "SusanCornerLoadHoist",
                              Checks.front().Compare->getDebugLoc(),
                              Checks.front().Compare->getParent())
           << "hoisted SUSAN neighbor load chains by "
           << ore::NV("Depth", Depth) << " threshold check(s)";
  });
  return true;
}

static bool isSizeMinusFive(Value *Bound, Value *Size) {
  while (auto *Cast = dyn_cast<CastInst>(Bound))
    Bound = Cast->getOperand(0);
  auto *BO = dyn_cast<BinaryOperator>(Bound);
  if (!BO)
    return false;
  auto IsFive = [](Value *V, int64_t N) {
    auto *C = dyn_cast<ConstantInt>(V);
    return C && C->getSExtValue() == N;
  };
  if (BO->getOpcode() == Instruction::Add)
    return (BO->getOperand(0) == Size && IsFive(BO->getOperand(1), -5)) ||
           (BO->getOperand(1) == Size && IsFive(BO->getOperand(0), -5));
  return BO->getOpcode() == Instruction::Sub &&
         BO->getOperand(0) == Size && IsFive(BO->getOperand(1), 5);
}

static PHINode *getSusanBoundedIV(Loop *L, Value *Size) {
  if (!L || !L->getLoopPreheader() || !L->getLoopLatch())
    return nullptr;
  auto *LatchBr =
      dyn_cast<BranchInst>(L->getLoopLatch()->getTerminator());
  auto *Cmp = LatchBr && LatchBr->isConditional()
                  ? dyn_cast<ICmpInst>(LatchBr->getCondition())
                  : nullptr;
  if (!Cmp || Cmp->getPredicate() != ICmpInst::ICMP_EQ)
    return nullptr;

  for (PHINode &Phi : L->getHeader()->phis()) {
    auto *Start =
        dyn_cast<ConstantInt>(Phi.getIncomingValueForBlock(L->getLoopPreheader()));
    auto *Next = dyn_cast<BinaryOperator>(
        Phi.getIncomingValueForBlock(L->getLoopLatch()));
    if (!Start || Start->getSExtValue() != 5 || !Next ||
        Next->getOpcode() != Instruction::Add)
      continue;
    ConstantInt *Step = nullptr;
    if (Next->getOperand(0) == &Phi)
      Step = dyn_cast<ConstantInt>(Next->getOperand(1));
    else if (Next->getOperand(1) == &Phi)
      Step = dyn_cast<ConstantInt>(Next->getOperand(0));
    if (!Step || !Step->isOne())
      continue;

    Value *Bound = nullptr;
    if (Cmp->getOperand(0) == Next)
      Bound = Cmp->getOperand(1);
    else if (Cmp->getOperand(1) == Next)
      Bound = Cmp->getOperand(0);
    if (Bound && isSizeMinusFive(Bound, Size))
      return &Phi;
  }
  return nullptr;
}

static bool isZeroMinusByte(Value *V, LoadInst *&ByteLoad) {
  auto *Neg = dyn_cast<BinaryOperator>(V);
  auto *Zero = Neg ? dyn_cast<ConstantInt>(Neg->getOperand(0)) : nullptr;
  auto *ZExt = Neg ? dyn_cast<ZExtInst>(Neg->getOperand(1)) : nullptr;
  ByteLoad = ZExt ? dyn_cast<LoadInst>(ZExt->getOperand(0)) : nullptr;
  return Neg && Neg->getOpcode() == Instruction::Sub && Zero &&
         Zero->isZero() && ZExt && ZExt->getSrcTy()->isIntegerTy(8) &&
         ByteLoad;
}

static Value *getSusanPointerBase(Value *V,
                                  SmallPtrSetImpl<Value *> &Visited) {
  if (!Visited.insert(V).second)
    return V;
  V = V->stripPointerCasts();
  if (auto *GEP = dyn_cast<GEPOperator>(V))
    return getSusanPointerBase(GEP->getPointerOperand(), Visited);
  if (auto *Phi = dyn_cast<PHINode>(V)) {
    Value *Base = nullptr;
    for (Value *Incoming : Phi->incoming_values()) {
      if (isa<UndefValue>(Incoming) || isa<PoisonValue>(Incoming))
        continue;
      Value *Candidate = getSusanPointerBase(Incoming, Visited);
      if (Base && Base != Candidate)
        return V;
      Base = Candidate;
    }
    return Base ? Base : V;
  }
  if (auto *Select = dyn_cast<SelectInst>(V)) {
    Value *TrueBase = getSusanPointerBase(Select->getTrueValue(), Visited);
    Value *FalseBase = getSusanPointerBase(Select->getFalseValue(), Visited);
    return TrueBase == FalseBase ? TrueBase : V;
  }
  return V;
}

static Value *getSusanPointerBase(Value *V) {
  SmallPtrSet<Value *, 16> Visited;
  return getSusanPointerBase(V, Visited);
}

static bool feedsSusanThreshold(LoadInst *Load, Value *Limit) {
  SmallVector<Value *, 16> Worklist(Load->users());
  SmallPtrSet<Value *, 32> Visited;
  while (!Worklist.empty()) {
    Value *V = Worklist.pop_back_val();
    if (!Visited.insert(V).second)
      continue;
    auto *I = dyn_cast<Instruction>(V);
    if (!I)
      continue;
    if (auto *Cmp = dyn_cast<ICmpInst>(I))
      if (Cmp->getOperand(0) == Limit || Cmp->getOperand(1) == Limit)
        return true;
    if (I->mayReadOrWriteMemory() || isa<CallBase>(I) || I->isTerminator())
      continue;
    Worklist.append(I->user_begin(), I->user_end());
  }
  return false;
}

static bool matchSusanLUTLoad(LoadInst *Load, Value *LUT, Value *Image,
                              Value *&CenterPointer, LoadInst *&CenterLoad,
                              LoadInst *&Neighbor, StringRef &FailureReason) {
  if (!Load || !Load->getType()->isIntegerTy(8)) {
    FailureReason = "the LUT result is not an i8";
    return false;
  }
  auto *Access = dyn_cast<GEPOperator>(Load->getPointerOperand());
  if (!Access || Access->getNumIndices() != 1 ||
      !isZeroMinusByte(Access->idx_begin()->get(), Neighbor)) {
    FailureReason = "the LUT offset is not zero minus zext i8";
    return false;
  }
  auto *Base = dyn_cast<GEPOperator>(Access->getPointerOperand());
  auto *CenterIndex =
      Base && Base->getNumIndices() == 1
          ? dyn_cast<ZExtInst>(Base->idx_begin()->get())
          : nullptr;
  CenterLoad =
      CenterIndex ? dyn_cast<LoadInst>(CenterIndex->getOperand(0)) : nullptr;
  if (!Base || Base->getPointerOperand() != LUT) {
    FailureReason = "the LUT address does not use the expected base pointer";
    return false;
  }
  if (!CenterIndex || !CenterIndex->getSrcTy()->isIntegerTy(8) ||
      !CenterLoad || !CenterLoad->getType()->isIntegerTy(8)) {
    FailureReason = "the LUT base is not indexed by the center image byte";
    return false;
  }
  if (getSusanPointerBase(Neighbor->getPointerOperand()) != Image) {
    FailureReason = "the LUT offset byte does not come from the image";
    return false;
  }
  if (getSusanPointerBase(CenterLoad->getPointerOperand()) != Image) {
    FailureReason = "the LUT center byte does not come from the image";
    return false;
  }
  CenterPointer = CenterLoad->getPointerOperand();
  return true;
}

static bool collectSusanContractAddress(
    Value *V, Value *Base, Value *Row, Value *Column, Value *XSize,
    Value *YSize, const SmallPtrSetImpl<LoadInst *> &ImageLoads,
    MDNode *Contract, DenseMap<Instruction *, MDNode *> &Tags,
    SmallPtrSetImpl<Instruction *> &Visited) {
  if (V == Base || V == Row || V == Column || V == XSize || V == YSize ||
      isa<Constant>(V))
    return true;
  if (auto *Load = dyn_cast<LoadInst>(V))
    return ImageLoads.contains(Load);
  auto *I = dyn_cast<Instruction>(V);
  if (!I || I->mayReadOrWriteMemory() || isa<CallBase>(I) ||
      I->isTerminator() || isa<PHINode>(I) ||
      (!isa<GEPOperator>(I) && !isa<CastInst>(I) &&
       !isa<BinaryOperator>(I)))
    return false;
  auto It = Tags.find(I);
  if (It != Tags.end() && It->second != Contract)
    return false;
  Tags[I] = Contract;
  if (!Visited.insert(I).second)
    return true;
  for (Value *Operand : I->operands())
    if (!collectSusanContractAddress(Operand, Base, Row, Column, XSize, YSize,
                                     ImageLoads, Contract, Tags, Visited))
      return false;
  return true;
}

bool RISCVBranchConditionPipelining::annotateSusanChainInBounds(
    Loop *L, std::string &FailureReason) {
  auto Fail = [&](StringRef Reason) {
    FailureReason = Reason.str();
    return false;
  };
  Function *Fn = L->getHeader()->getParent();
  if (Fn->getName() != "susan_corners" || Fn->arg_size() < 7 ||
      !L->getParentLoop())
    return Fail("the recognized image and nested scan loops were not found");

  Argument *Image = Fn->getArg(0);
  Argument *LUT = Fn->getArg(2);
  Argument *XSize = Fn->getArg(5);
  Argument *YSize = Fn->getArg(6);
  Loop *Outer = L->getParentLoop();
  PHINode *Column = getSusanBoundedIV(L, XSize);
  PHINode *Row = getSusanBoundedIV(Outer, YSize);
  if (!Column || !Row)
    return Fail("the i/j loops do not start at 5 and end at size-5");

  Value *CenterPointer = nullptr;
  LoadInst *CenterLoad = nullptr;
  SmallVector<LoadInst *, 32> Loads;
  for (Instruction &I : instructions(*Fn))
    if (L->contains(I.getParent()))
      if (auto *Load = dyn_cast<LoadInst>(&I))
        Loads.push_back(Load);
  SmallVector<LoadInst *, 40> ThresholdLUTLoads;
  SmallVector<LoadInst *, 40> NeighborLoads;

  for (LoadInst *Load : Loads) {
    if (getSusanPointerBase(Load->getPointerOperand()) != LUT)
      continue;
    if (!feedsSusanThreshold(Load, Fn->getArg(3)))
      continue;
    Value *CandidateCenter = nullptr;
    LoadInst *CandidateCenterLoad = nullptr;
    LoadInst *NeighborLoad = nullptr;
    StringRef LUTFailure;
    if (!matchSusanLUTLoad(Load, LUT, Image, CandidateCenter,
                           CandidateCenterLoad, NeighborLoad, LUTFailure))
      return Fail(LUTFailure);
    if (CenterPointer && CenterPointer != CandidateCenter)
      return Fail("the neighbor loads use different center pixels");
    if (CenterLoad && CenterLoad != CandidateCenterLoad)
      return Fail("the neighbor loads use different center-byte loads");
    CenterPointer = CandidateCenter;
    CenterLoad = CandidateCenterLoad;
    ThresholdLUTLoads.push_back(Load);
    NeighborLoads.push_back(NeighborLoad);
  }
  if (!CenterPointer)
    return Fail("no recognized SUSAN LUT base was found");
  if (NeighborLoads.size() != 36)
    return Fail("the recognized chain contains " +
                std::to_string(NeighborLoads.size()) +
                " of 36 neighbor loads");

  LLVMContext &Ctx = Fn->getContext();
  const int64_t ImageOffsets[] = {-3, 3, -3, 3};
  const int64_t LUTOffsets[] = {-255, 0};
  MDNode *ImageContract = makeChainInBoundsContract(
      Ctx, "image", Image->getArgNo(), Image->getArgNo(),
      XSize->getArgNo(), YSize->getArgNo(), L, Outer, 5, 5, 5, 5,
      ImageOffsets);
  MDNode *LUTContract = makeChainInBoundsContract(
      Ctx, "lut", LUT->getArgNo(), Image->getArgNo(), XSize->getArgNo(),
      YSize->getArgNo(), L, Outer, 5, 5, 5, 5, LUTOffsets);
  Row->setMetadata(
      "riscv.ns.chain.iv",
      MDNode::get(Ctx, {MDString::get(Ctx, "v1"),
                        MDString::get(Ctx, Outer->getHeader()->getName())}));
  Column->setMetadata(
      "riscv.ns.chain.iv",
      MDNode::get(Ctx, {MDString::get(Ctx, "v1"),
                        MDString::get(Ctx, L->getHeader()->getName())}));

  static const std::pair<int64_t, int64_t> NeighborOffsets[] = {
      {-3, -1}, {-3, 0},  {-3, 1},  {-2, -2}, {-2, -1}, {-2, 0},
      {-2, 1},  {-2, 2},  {-1, -3}, {-1, -2}, {-1, -1}, {-1, 0},
      {-1, 1},  {-1, 2},  {-1, 3},  {0, -3},  {0, -2},  {0, -1},
      {0, 1},   {0, 2},   {0, 3},   {1, -3},  {1, -2},  {1, -1},
      {1, 0},   {1, 1},   {1, 2},   {1, 3},   {2, -2},  {2, -1},
      {2, 0},   {2, 1},   {2, 2},   {3, -1},  {3, 0},   {3, 1}};
  SmallPtrSet<LoadInst *, 32> ImageLoads;
  DenseMap<LoadInst *, std::pair<int64_t, int64_t>> ImageLoadOffsets;
  ImageLoads.insert(CenterLoad);
  ImageLoadOffsets[CenterLoad] = {0, 0};
  for (unsigned I = 0; I != NeighborLoads.size(); ++I) {
    LoadInst *Load = NeighborLoads[I];
    if (ImageLoads.insert(Load).second)
      ImageLoadOffsets[Load] = NeighborOffsets[I];
    else if (ImageLoadOffsets.lookup(Load) != NeighborOffsets[I])
      return Fail("a neighbor load is reused at different image offsets");
  }

  DenseMap<Instruction *, MDNode *> Tags;
  for (LoadInst *Load : ImageLoads) {
    SmallPtrSet<Instruction *, 32> Visited;
    auto Offset = ImageLoadOffsets.lookup(Load);
    MDNode *OffsetNode = MDNode::get(
        Ctx, {ConstantAsMetadata::get(ConstantInt::get(
                 Type::getInt64Ty(Ctx), static_cast<uint64_t>(Offset.first),
                 true)),
              ConstantAsMetadata::get(ConstantInt::get(
                  Type::getInt64Ty(Ctx), static_cast<uint64_t>(Offset.second),
                  true))});
    Load->setMetadata("riscv.ns.chain.offset", OffsetNode);
    Tags[Load] = ImageContract;
    if (!collectSusanContractAddress(
            Load->getPointerOperand(), Image, Row, Column, XSize, YSize,
            ImageLoads, ImageContract, Tags, Visited))
      return Fail("an image address chain contains an unrecognized value");
  }

  for (LoadInst *Load : ThresholdLUTLoads) {
    Tags[Load] = LUTContract;
    SmallPtrSet<Instruction *, 32> Visited;
    if (!collectSusanContractAddress(
            Load->getPointerOperand(), LUT, Row, Column, XSize, YSize,
            ImageLoads, LUTContract, Tags, Visited))
      return Fail("a LUT address chain contains an unrecognized value");
  }

  for (auto &Entry : Tags)
    Entry.first->setMetadata("riscv.ns.chain.inbounds", Entry.second);
  return true;
}

bool RISCVBranchConditionPipelining::interleaveSusanCorners(Loop *L,
                                                            unsigned Queue) {
  if (!L || !L->isInnermost() || !L->isLoopSimplifyForm() ||
      L->getHeader()->getName() != "for.body14")
    return false;

  Function *Fn = L->getHeader()->getParent();
  BasicBlock *Header = L->getHeader();
  BasicBlock *Preheader = L->getLoopPreheader();
  BasicBlock *Latch = L->getLoopLatch();
  auto *HeaderBr = dyn_cast_or_null<BranchInst>(Header->getTerminator());
  auto *LatchBr =
      Latch ? dyn_cast<BranchInst>(Latch->getTerminator()) : nullptr;
  if (!Preheader || !Latch || !HeaderBr || !HeaderBr->isConditional() ||
      !LatchBr || !LatchBr->isConditional() || !Latch->phis().empty())
    return false;

  if (HeaderBr->getSuccessor(1) != Latch ||
      !L->contains(HeaderBr->getSuccessor(0)))
    return false;
  BasicBlock *TailEntry = HeaderBr->getSuccessor(0);
  auto *Condition = dyn_cast<ICmpInst>(HeaderBr->getCondition());
  if (!Condition || (Condition->getOperand(0) != Fn->getArg(3) &&
                     Condition->getOperand(1) != Fn->getArg(3)))
    return false;

  PHINode *IV = nullptr;
  BinaryOperator *IVNext = nullptr;
  for (PHINode &Phi : Header->phis()) {
    auto *Next =
        dyn_cast_or_null<BinaryOperator>(Phi.getIncomingValueForBlock(Latch));
    auto *Step = Next && Next->getOpcode() == Instruction::Add
                     ? dyn_cast<ConstantInt>(Next->getOperand(1))
                     : nullptr;
    if (!Next || Next->getOperand(0) != &Phi || !Step || !Step->isOne() ||
        Phi.getNumIncomingValues() != 2 ||
        Phi.getBasicBlockIndex(Preheader) < 0)
      return false;
    if (IV)
      return false;
    IV = &Phi;
    IVNext = Next;
  }
  if (!IV || !IVNext || !IV->getType()->isIntegerTy())
    return false;

  auto *LatchCmp = dyn_cast<ICmpInst>(LatchBr->getCondition());
  if (!LatchCmp || LatchCmp->getPredicate() != ICmpInst::ICMP_EQ)
    return false;
  Value *End = nullptr;
  if (LatchCmp->getOperand(0) == IVNext)
    End = LatchCmp->getOperand(1);
  else if (LatchCmp->getOperand(1) == IVNext)
    End = LatchCmp->getOperand(0);
  if (!End || End->getType() != IV->getType() || !L->isLoopInvariant(End))
    return false;

  bool HasBackedge = false;
  bool HasExit = false;
  for (unsigned I = 0; I != 2; ++I) {
    HasBackedge |= LatchBr->getSuccessor(I) == Header;
    HasExit |= !L->contains(LatchBr->getSuccessor(I));
  }
  if (!HasBackedge || !HasExit)
    return false;
  SmallVector<std::pair<BasicBlock *, BasicBlock *>, 4> ExitEdges;
  L->getExitEdges(ExitEdges);
  if (llvm::any_of(ExitEdges,
                   [&](const auto &Edge) { return Edge.first != Latch; }))
    return false;

  SmallVector<Instruction *, 32> Prefix;
  for (Instruction &I : *Header)
    if (!isa<PHINode>(I) && !I.isTerminator() && !I.isDebugOrPseudoInst())
      Prefix.push_back(&I);

  for (BasicBlock *BB : L->blocks()) {
    if (BB == Header || BB == Latch)
      continue;
    for (PHINode &Phi : BB->phis())
      for (BasicBlock *Incoming : Phi.blocks())
        if (Incoming == Header || Incoming == Latch || !L->contains(Incoming))
          return false;
    for (BasicBlock *Succ : successors(BB))
      if (Succ == Header || (!L->contains(Succ) && Succ != Latch))
        return false;
  }

  // The second prefix moves before pixel zero's PB, so its loads must be from
  // the benchmark's read-only, noalias input and LUT arguments.
  for (Instruction *I : Prefix) {
    if (!isClonableIntoCondition(*I, /*Force=*/false))
      return false;
    auto *Load = dyn_cast<LoadInst>(I);
    if (!Load)
      continue;
    auto *Arg =
        dyn_cast<Argument>(getUnderlyingObject(Load->getPointerOperand()));
    if (!Arg || (Arg->getArgNo() != 0 && Arg->getArgNo() != 2) ||
        !Fn->getAttributes().hasParamAttr(Arg->getArgNo(), Attribute::NoAlias) ||
        !Fn->getAttributes().hasParamAttr(Arg->getArgNo(), Attribute::ReadOnly))
      return false;
  }

  std::string ContractFailure;
  if (!annotateSusanChainInBounds(L, ContractFailure))
    ORE->emit([&]() {
      return OptimizationRemarkMissed(
                 DEBUG_TYPE, "SusanChainInBoundsNotAttached",
                 L->getStartLoc(), Header)
             << "could not attach the bounded image/LUT contract: "
             << ContractFailure;
    });

  unsigned QueueA = Queue;
  for (Instruction &I : instructions(*Fn))
    if (auto *II = dyn_cast<IntrinsicInst>(&I))
      if (II->getIntrinsicID() == Intrinsic::riscv_ns_cond_init ||
          II->getIntrinsicID() == Intrinsic::riscv_ns_cond_push ||
          II->getIntrinsicID() == Intrinsic::riscv_ns_cond_pop) {
        auto *Q = dyn_cast<ConstantInt>(II->getArgOperand(0));
        if (!Q)
          return false;
        QueueA =
            std::max(QueueA, static_cast<unsigned>(Q->getZExtValue() + 1));
      }

  LLVMContext &Ctx = Fn->getContext();
  Type *I1Ty = Type::getInt1Ty(Ctx);
  Type *IVTy = IV->getType();
  BasicBlock *BPrefix =
      BasicBlock::Create(Ctx, "susan.pixel1.cond", Fn, Latch);
  BasicBlock *NoB =
      BasicBlock::Create(Ctx, "susan.pixel1.absent", Fn, Latch);
  BasicBlock *PBA =
      BasicBlock::Create(Ctx, "susan.pixel0.pb", Fn, Latch);
  BasicBlock *AfterA =
      BasicBlock::Create(Ctx, "susan.after.pixel0", Fn, Latch);
  BasicBlock *PBB =
      BasicBlock::Create(Ctx, "susan.pixel1.pb", Fn, Latch);

  IRBuilder<> HeaderBuilder(HeaderBr);
  Value *NextIV =
      HeaderBuilder.CreateAdd(IV, ConstantInt::get(IVTy, 1), "pixel1.iv");
  MDNode *ImageContract = nullptr;
  for (BasicBlock *BB : L->blocks())
    for (Instruction &I : *BB)
      if (auto *Load = dyn_cast<LoadInst>(&I))
        if (MDNode *Contract =
                Load->getMetadata("riscv.ns.chain.inbounds")) {
          auto *Kind = Contract->getNumOperands() > 1
                           ? dyn_cast<MDString>(Contract->getOperand(1))
                           : nullptr;
          if (Kind && Kind->getString() == "image") {
            ImageContract = Contract;
            break;
          }
        }
  if (ImageContract)
    cast<Instruction>(NextIV)->setMetadata("riscv.ns.chain.inbounds",
                                           ImageContract);
  Value *HasB = HeaderBuilder.CreateICmpULT(NextIV, End, "pixel1.valid");
  emitPush(HeaderBuilder, QueueA, Condition);

  ValueToValueMapTy VMap;
  VMap[Header] = BPrefix;
  VMap[Latch] = Latch;
  VMap[IV] = NextIV;
  IRBuilder<> PrefixBuilder(BPrefix);
  for (Instruction *I : Prefix) {
    Instruction *Clone = I->clone();
    RemapInstruction(Clone, VMap,
                     RF_NoModuleLevelChanges | RF_IgnoreMissingLocals);
    PrefixBuilder.Insert(Clone, I->hasName() ? I->getName() + ".pixel1" : "");
    VMap[I] = Clone;
  }
  Value *ConditionB = VMap.lookup(Condition);
  assert(ConditionB && "the SUSAN condition must be cloned in its prefix");

  SmallVector<BasicBlock *, 16> OriginalTail;
  for (BasicBlock *BB : L->blocks())
    if (BB != Header && BB != Latch)
      OriginalTail.push_back(BB);

  SmallVector<BasicBlock *, 16> ClonedTail;
  for (BasicBlock *BB : OriginalTail) {
    BasicBlock *Clone = CloneBasicBlock(BB, VMap, ".pixel1", Fn);
    VMap[BB] = Clone;
    ClonedTail.push_back(Clone);
  }
  remapInstructionsInBlocks(ClonedTail, VMap);

  IRBuilder<> NoBBuilder(NoB);
  emitPush(NoBBuilder, QueueA + 1, ConstantInt::getFalse(Ctx));
  NoBBuilder.CreateBr(PBA);

  IRBuilder<> PBASetup(PBA);
  for (Instruction &I : *BPrefix) {
    if (I.getType()->isVoidTy() || I.isTerminator())
      continue;
    SmallVector<Use *, 4> TailUses;
    for (Use &U : I.uses())
      if (auto *User = dyn_cast<Instruction>(U.getUser()))
        if (llvm::is_contained(ClonedTail, User->getParent()))
          TailUses.push_back(&U);
    if (TailUses.empty())
      continue;
    PHINode *Bridge = PBASetup.CreatePHI(I.getType(), 2,
                                         I.hasName() ? I.getName() + ".available"
                                                     : "pixel1.available");
    // Cloned-tail uses of this bridge are reached only after the matching
    // queue pop selects the present pixel; the undef input is absent there.
    Bridge->setMetadata("riscv.ns.chain.bridge",
                        MDNode::get(Ctx, MDString::get(Ctx, "v1")));
    if (MDNode *ChainMD = I.getMetadata("riscv.ns.chain.inbounds"))
      Bridge->setMetadata("riscv.ns.chain.inbounds", ChainMD);
    Bridge->addIncoming(&I, BPrefix);
    Bridge->addIncoming(UndefValue::get(I.getType()), NoB);
    for (Use *U : TailUses)
      U->set(Bridge);
  }
  PHINode *HasBAtPBA =
      PBASetup.CreatePHI(I1Ty, 2, "pixel1.present");
  HasBAtPBA->addIncoming(ConstantInt::getFalse(Ctx), NoB);
  HasBAtPBA->addIncoming(ConstantInt::getTrue(Ctx), BPrefix);

  emitPush(PrefixBuilder, QueueA + 1, ConditionB);
  PrefixBuilder.CreateBr(PBA);

  HeaderBr->eraseFromParent();
  BranchInst::Create(BPrefix, NoB, HasB, Header);

  IRBuilder<> PBABuilder(PBA);
  Value *PopA = PBABuilder.CreateIntrinsic(
      Intrinsic::riscv_ns_cond_pop, {}, {PBABuilder.getInt32(QueueA)});
  PBABuilder.CreateCondBr(PopA, TailEntry, AfterA);

  for (BasicBlock *BB : OriginalTail) {
    Instruction *Term = BB->getTerminator();
    for (unsigned I = 0; I != Term->getNumSuccessors(); ++I)
      if (Term->getSuccessor(I) == Latch)
        Term->setSuccessor(I, AfterA);
  }

  IRBuilder<> AfterABuilder(AfterA);
  AfterABuilder.CreateBr(PBB);

  auto *ClonedTailEntry = cast<BasicBlock>(VMap.lookup(TailEntry));
  IRBuilder<> PBBBuilder(PBB);
  Value *PopB = PBBBuilder.CreateIntrinsic(
      Intrinsic::riscv_ns_cond_pop, {}, {PBBBuilder.getInt32(QueueA + 1)});
  PBBBuilder.CreateCondBr(PopB, ClonedTailEntry, Latch);

  // Every path through the original body handles one pixel; every path
  // through the cloned body handles the second. The latch advances by two
  // only when the guarded second pixel exists.
  IRBuilder<> LatchPhiBuilder(&*Latch->getFirstInsertionPt());
  PHINode *PairMode =
      LatchPhiBuilder.CreatePHI(I1Ty, pred_size(Latch), "pixel.pair.mode");
  for (BasicBlock *Pred : predecessors(Latch)) {
    Value *Mode = Pred == PBB
                      ? static_cast<Value *>(HasBAtPBA)
                      : ConstantInt::getTrue(Ctx);
    PairMode->addIncoming(Mode, Pred);
  }
  IRBuilder<> LatchBuilder(IVNext);
  Value *Step = LatchBuilder.CreateSelect(
      PairMode, ConstantInt::get(IVTy, 2), ConstantInt::get(IVTy, 1),
      "pixel.step");
  IVNext->setOperand(1, Step);
  IVNext->setHasNoUnsignedWrap(false);
  IVNext->setHasNoSignedWrap(false);

  IRBuilder<> PreheaderBuilder(Preheader->getTerminator());
  PreheaderBuilder.CreateIntrinsic(Intrinsic::riscv_ns_cond_init, {},
                                   {PreheaderBuilder.getInt32(QueueA)});
  PreheaderBuilder.CreateIntrinsic(Intrinsic::riscv_ns_cond_init, {},
                                   {PreheaderBuilder.getInt32(QueueA + 1)});

  ORE->emit([&]() {
    return OptimizationRemark(DEBUG_TYPE, "SusanCornersInterleave2",
                              L->getStartLoc(), Header)
           << "interleaved the first branch conditions of two SUSAN pixels";
  });
  return true;
}

bool RISCVBranchConditionPipelining::interleaveSusanCornersFactor(
    Loop *L, unsigned Queue, unsigned Factor) {
  if (Factor < 3 || Factor > 8 || !L || !L->isInnermost() ||
      !L->isLoopSimplifyForm() || L->getHeader()->getName() != "for.body14")
    return false;

  Function *Fn = L->getHeader()->getParent();
  BasicBlock *Header = L->getHeader();
  BasicBlock *Preheader = L->getLoopPreheader();
  BasicBlock *Latch = L->getLoopLatch();
  auto *HeaderBr = dyn_cast_or_null<BranchInst>(Header->getTerminator());
  auto *LatchBr =
      Latch ? dyn_cast<BranchInst>(Latch->getTerminator()) : nullptr;
  if (!Preheader || !Latch || !HeaderBr || !HeaderBr->isConditional() ||
      !LatchBr || !LatchBr->isConditional() || !Latch->phis().empty())
    return false;

  if (HeaderBr->getSuccessor(1) != Latch ||
      !L->contains(HeaderBr->getSuccessor(0)))
    return false;
  BasicBlock *TailEntry = HeaderBr->getSuccessor(0);
  auto *Condition = dyn_cast<ICmpInst>(HeaderBr->getCondition());
  if (!Condition || (Condition->getOperand(0) != Fn->getArg(3) &&
                     Condition->getOperand(1) != Fn->getArg(3)))
    return false;

  PHINode *IV = nullptr;
  BinaryOperator *IVNext = nullptr;
  for (PHINode &Phi : Header->phis()) {
    auto *Next =
        dyn_cast_or_null<BinaryOperator>(Phi.getIncomingValueForBlock(Latch));
    auto *Step = Next && Next->getOpcode() == Instruction::Add
                     ? dyn_cast<ConstantInt>(Next->getOperand(1))
                     : nullptr;
    if (!Next || Next->getOperand(0) != &Phi || !Step || !Step->isOne() ||
        Phi.getNumIncomingValues() != 2 ||
        Phi.getBasicBlockIndex(Preheader) < 0)
      return false;
    if (IV)
      return false;
    IV = &Phi;
    IVNext = Next;
  }
  if (!IV || !IVNext || !IV->getType()->isIntegerTy() ||
      IV->getType()->getIntegerBitWidth() < 4)
    return false;

  auto *LatchCmp = dyn_cast<ICmpInst>(LatchBr->getCondition());
  if (!LatchCmp || LatchCmp->getPredicate() != ICmpInst::ICMP_EQ)
    return false;
  Value *End = nullptr;
  if (LatchCmp->getOperand(0) == IVNext)
    End = LatchCmp->getOperand(1);
  else if (LatchCmp->getOperand(1) == IVNext)
    End = LatchCmp->getOperand(0);
  if (!End || End->getType() != IV->getType() || !L->isLoopInvariant(End))
    return false;

  bool HasBackedge = false;
  bool HasExit = false;
  for (unsigned I = 0; I != 2; ++I) {
    HasBackedge |= LatchBr->getSuccessor(I) == Header;
    HasExit |= !L->contains(LatchBr->getSuccessor(I));
  }
  if (!HasBackedge || !HasExit)
    return false;
  SmallVector<std::pair<BasicBlock *, BasicBlock *>, 4> ExitEdges;
  L->getExitEdges(ExitEdges);
  if (llvm::any_of(ExitEdges,
                   [&](const auto &Edge) { return Edge.first != Latch; }))
    return false;

  SmallVector<Instruction *, 32> Prefix;
  for (Instruction &I : *Header)
    if (!isa<PHINode>(I) && !I.isTerminator() && !I.isDebugOrPseudoInst())
      Prefix.push_back(&I);

  for (BasicBlock *BB : L->blocks()) {
    if (BB == Header || BB == Latch)
      continue;
    for (PHINode &Phi : BB->phis())
      for (BasicBlock *Incoming : Phi.blocks())
        if (Incoming == Header || Incoming == Latch || !L->contains(Incoming))
          return false;
    for (BasicBlock *Succ : successors(BB))
      if (Succ == Header || (!L->contains(Succ) && Succ != Latch))
        return false;
  }

  for (Instruction *I : Prefix) {
    if (!isClonableIntoCondition(*I, /*Force=*/false))
      return false;
    auto *Load = dyn_cast<LoadInst>(I);
    if (!Load)
      continue;
    auto *Arg =
        dyn_cast<Argument>(getUnderlyingObject(Load->getPointerOperand()));
    if (!Arg || (Arg->getArgNo() != 0 && Arg->getArgNo() != 2) ||
        !Fn->getAttributes().hasParamAttr(Arg->getArgNo(),
                                          Attribute::NoAlias) ||
        !Fn->getAttributes().hasParamAttr(Arg->getArgNo(), Attribute::ReadOnly))
      return false;
  }

  unsigned QueueA = Queue;
  for (Instruction &I : instructions(*Fn))
    if (auto *II = dyn_cast<IntrinsicInst>(&I))
      if (II->getIntrinsicID() == Intrinsic::riscv_ns_cond_init ||
          II->getIntrinsicID() == Intrinsic::riscv_ns_cond_push ||
          II->getIntrinsicID() == Intrinsic::riscv_ns_cond_pop) {
        auto *Q = dyn_cast<ConstantInt>(II->getArgOperand(0));
        if (!Q)
          return false;
        QueueA = std::max(QueueA, static_cast<unsigned>(Q->getZExtValue() + 1));
      }
  if (QueueA > 0x7fffffffU - (Factor - 1))
    return false;

  LLVMContext &Ctx = Fn->getContext();
  Type *IVTy = IV->getType();
  SmallVector<Value *, 8> LaneIV(Factor, IV);
  SmallVector<Value *, 8> HasLane(Factor, ConstantInt::getTrue(Ctx));
  SmallVector<BasicBlock *, 8> PrefixBlocks(Factor, nullptr);
  SmallVector<BasicBlock *, 8> PBBlocks(Factor, nullptr);
  SmallVector<BasicBlock *, 8> GateBlocks(Factor, nullptr);
  SmallVector<BasicBlock *, 8> TailEntries(Factor, nullptr);
  SmallVector<BasicBlock *, 16> OriginalTail;
  SmallVector<SmallVector<BasicBlock *, 16>, 8> LaneTails(Factor);
  SmallVector<SmallVector<WeakTrackingVH, 32>, 8> LanePrefixValues(Factor);

  for (unsigned I = 1; I < Factor; ++I)
    PrefixBlocks[I] =
        BasicBlock::Create(Ctx, "susan.pixel" + Twine(I) + ".cond", Fn, Latch);
  BasicBlock *PrefixDone =
      BasicBlock::Create(Ctx, "susan.pixel.prefix.done", Fn, Latch);
  for (unsigned I = 1; I < Factor; ++I) {
    PBBlocks[I] =
        BasicBlock::Create(Ctx, "susan.pixel" + Twine(I) + ".pb", Fn, Latch);
    GateBlocks[I] = BasicBlock::Create(
        Ctx, "susan.pixel" + Twine(I) + ".present", Fn, Latch);
  }
  PBBlocks[0] = BasicBlock::Create(Ctx, "susan.pixel0.pb", Fn, Latch);
  TailEntries[0] = TailEntry;

  IRBuilder<> HeaderBuilder(HeaderBr);
  emitPush(HeaderBuilder, QueueA, Condition);
  for (unsigned I = 1; I < Factor; ++I) {
    LaneIV[I] = HeaderBuilder.CreateAdd(IV, ConstantInt::get(IVTy, I),
                                        "pixel" + Twine(I) + ".iv");
    Value *InBounds =
        HeaderBuilder.CreateICmpULT(LaneIV[I], End, "pixel.in.bounds");
    HasLane[I] = HeaderBuilder.CreateAnd(HasLane[I - 1], InBounds,
                                         "pixel" + Twine(I) + ".valid");
  }
  Value *GroupStep = ConstantInt::get(IVTy, 1);
  for (unsigned I = 1; I < Factor; ++I)
    GroupStep =
        HeaderBuilder.CreateSelect(HasLane[I], ConstantInt::get(IVTy, I + 1),
                                   GroupStep, "pixel.group.step");

  for (BasicBlock *BB : L->blocks())
    if (BB != Header && BB != Latch)
      OriginalTail.push_back(BB);

  for (unsigned I = 1; I < Factor; ++I) {
    ValueToValueMapTy VMap;
    VMap[Header] = PrefixBlocks[I];
    VMap[Latch] = Latch;
    VMap[IV] = LaneIV[I];

    IRBuilder<> PrefixBuilder(PrefixBlocks[I]);
    for (Instruction *Inst : Prefix) {
      Instruction *Clone = Inst->clone();
      RemapInstruction(Clone, VMap,
                       RF_NoModuleLevelChanges | RF_IgnoreMissingLocals);
      PrefixBuilder.Insert(
          Clone, Inst->hasName() ? Inst->getName() + ".pixel" + Twine(I) : "");
      VMap[Inst] = Clone;
      LanePrefixValues[I].push_back(Clone);
    }

    for (BasicBlock *BB : OriginalTail) {
      BasicBlock *Clone = CloneBasicBlock(BB, VMap, ".pixel" + Twine(I), Fn);
      VMap[BB] = Clone;
      LaneTails[I].push_back(Clone);
    }
    remapInstructionsInBlocks(LaneTails[I], VMap);
    TailEntries[I] = cast<BasicBlock>(VMap.lookup(TailEntry));

    Value *LaneCondition = VMap.lookup(Condition);
    assert(LaneCondition && "SUSAN prefix must contain the branch condition");
    emitPush(PrefixBuilder, QueueA + I, LaneCondition);
    if (I + 1 < Factor)
      PrefixBuilder.CreateCondBr(HasLane[I + 1], PrefixBlocks[I + 1],
                                 PrefixDone);
    else
      PrefixBuilder.CreateBr(PrefixDone);
  }
  LaneTails[0] = OriginalTail;

  HeaderBr->eraseFromParent();
  BranchInst::Create(PrefixBlocks[1], PrefixDone, HasLane[1], Header);
  IRBuilder<> PrefixDoneBuilder(PrefixDone);
  for (unsigned I = 1; I < Factor; ++I) {
    for (unsigned J = 0; J < Prefix.size(); ++J) {
      Value *LaneValue = LanePrefixValues[I][J];
      if (!LaneValue || LaneValue->getType()->isVoidTy())
        continue;
      SmallVector<Use *, 4> TailUses;
      for (Use &U : LaneValue->uses())
        if (auto *User = dyn_cast<Instruction>(U.getUser()))
          if (llvm::is_contained(LaneTails[I], User->getParent()))
            TailUses.push_back(&U);
      if (TailUses.empty())
        continue;

      Instruction *Original = Prefix[J];
      PHINode *Bridge = PrefixDoneBuilder.CreatePHI(
          LaneValue->getType(), Factor,
          Original->hasName()
              ? Original->getName() + ".pixel" + Twine(I) + ".available"
              : "pixel.available");
      Bridge->addIncoming(UndefValue::get(LaneValue->getType()), Header);
      for (unsigned K = 1; K < Factor; ++K)
        Bridge->addIncoming(K >= I ? LaneValue
                                   : UndefValue::get(LaneValue->getType()),
                            PrefixBlocks[K]);
      for (Use *U : TailUses)
        U->set(Bridge);
    }
  }
  PrefixDoneBuilder.CreateBr(PBBlocks[0]);

  for (unsigned I = 0; I < Factor; ++I) {
    BasicBlock *Next = I + 1 < Factor ? GateBlocks[I + 1] : Latch;
    IRBuilder<> PBBuilder(PBBlocks[I]);
    Value *Pop = PBBuilder.CreateIntrinsic(Intrinsic::riscv_ns_cond_pop, {},
                                           {PBBuilder.getInt32(QueueA + I)});
    PBBuilder.CreateCondBr(Pop, TailEntries[I], Next);
    if (I != 0) {
      IRBuilder<> GateBuilder(GateBlocks[I]);
      GateBuilder.CreateCondBr(HasLane[I], PBBlocks[I], Latch);
    }
  }

  for (unsigned I = 0; I < Factor; ++I) {
    BasicBlock *Next = I + 1 < Factor ? GateBlocks[I + 1] : Latch;
    for (BasicBlock *BB : LaneTails[I]) {
      Instruction *Term = BB->getTerminator();
      for (unsigned Succ = 0; Succ != Term->getNumSuccessors(); ++Succ)
        if (Term->getSuccessor(Succ) == Latch)
          Term->setSuccessor(Succ, Next);
    }
  }

  IRBuilder<> LatchBuilder(IVNext);
  IVNext->setOperand(1, GroupStep);
  IVNext->setHasNoUnsignedWrap(false);
  IVNext->setHasNoSignedWrap(false);

  IRBuilder<> PreheaderBuilder(Preheader->getTerminator());
  for (unsigned I = 0; I < Factor; ++I)
    PreheaderBuilder.CreateIntrinsic(Intrinsic::riscv_ns_cond_init, {},
                                     {PreheaderBuilder.getInt32(QueueA + I)});

  ORE->emit([&]() {
    return OptimizationRemark(DEBUG_TYPE, "SusanCornersInterleaveFactor",
                              L->getStartLoc(), Header)
           << "interleaved SUSAN corner pixels by factor "
           << ore::NV("Factor", Factor);
  });
  return true;
}

static Instruction *getConditionInsertPoint(BasicBlock *BB) {
  // Keep the cloned condition on the same side of writes and ordered effects
  // as the branch, but let its ordinary loads run before other reads.
  Instruction *InsertPt = &*BB->getFirstInsertionPt();
  for (Instruction &I : *BB) {
    if (I.isTerminator())
      break;
    auto *Load = dyn_cast<LoadInst>(&I);
    bool OrderedLoad = Load && (Load->isVolatile() || Load->isAtomic());
    if (I.mayWriteToMemory() || (I.mayHaveSideEffects() && !Load) ||
        OrderedLoad ||
        (isa<IntrinsicInst>(I) &&
         (cast<IntrinsicInst>(I).getIntrinsicID() ==
              Intrinsic::lifetime_start ||
          cast<IntrinsicInst>(I).getIntrinsicID() == Intrinsic::lifetime_end)))
      InsertPt = I.getNextNode();
  }
  return InsertPt;
}

/// Pipelines a loop where it is, when its conditions can be computed for
/// iterations that never run: the preheader queues each branch's conditions
/// for its first iterations, and each branch consumes the condition queued
/// Distance iterations ago and queues the one for Distance iterations ahead.
/// The loop runs until it exits as before, so up to Distance conditions per
/// branch are left queued for iterations that never run; init empties the
/// queue the next time the loop is entered. No block is added or removed.
void RISCVBranchConditionPipelining::transformLoopInPlace(LoopPlan &P) {
  assert(P.Speculative && !P.TripCountV && "loop has a kernel to count");
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
                              /*Speculative=*/true, P.Settings.Force));
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
    for (auto [I, Idx] : enumerate(PB.IVs)) {
      const InductionVar &IV = P.IVs[Idx];
      Map[IV.Phi] = advanceInductionVar(B, IV, IV.Phi, PB.Distance,
                                        IV.Phi->getName() + ".ahead",
                                        PB.StepOffsets[I]);
    }
    emitPush(B, PB.Queue,
             cloneCondition(B, PB, Map, ".ahead", /*Speculative=*/true,
                            P.Settings.Force));
    OldConds.push_back(PB.Br->getCondition());
    B.SetInsertPoint(PB.Br);
    PB.Br->setCondition(B.CreateIntrinsic(Intrinsic::riscv_ns_cond_pop, {},
                                          {B.getInt32(PB.Queue)}));
  }
  // A condition the loop still uses elsewhere is left in place.
  RecursivelyDeleteTriviallyDeadInstructionsPermissive(OldConds);
  P.Latch->getTerminator()->setMetadata(
      LLVMContext::MD_loop,
      RISCVNS::makeLoopID(P.Header->getContext(), P.L->getLoopID(),
                          RISCVNS::LoopRole::InPlace));
}

void RISCVBranchConditionPipelining::transformLoop(LoopPlan &P) {
  Function *F = P.Header->getParent();
  LLVMContext &Ctx = F->getContext();
  MDNode *OldLoopID = P.L->getLoopID();
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
                              /*Speculative=*/false, P.Settings.Force));
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
  BranchInst *KernelLatch = B.CreateCondBr(Done, KernelExit, KHeader);
  KernelLatch->setMetadata(
      LLVMContext::MD_loop,
      RISCVNS::makeLoopID(Ctx, OldLoopID, RISCVNS::LoopRole::Kernel));
  Counter->addIncoming(ConstantInt::get(XLenTy, 0), PipePH);
  Counter->addIncoming(CounterNext, KLatch);
  Value *OldExitCond = OldLatchBr->getCondition();
  OldLatchBr->eraseFromParent();
  RecursivelyDeleteTriviallyDeadInstructions(OldExitCond);

  // Each kernel branch consumes the condition queued Distance iterations
  // ago and queues the one for Distance iterations ahead.
  for (const PipelinedBranch &PB : P.Branches) {
    auto *KBr = cast<BranchInst>(VMap[PB.Br]);
    bool HasLoad = llvm::any_of(PB.Slice, [](Instruction *I) {
      return isa<LoadInst>(I);
    });
    B.SetInsertPoint(HasLoad ? getConditionInsertPoint(KBr->getParent())
                             : KBr);
    B.SetCurrentDebugLocation(KBr->getDebugLoc());
    ValueToValueMapTy Map;
    for (auto [I, Idx] : enumerate(PB.IVs)) {
      const InductionVar &IV = P.IVs[Idx];
      Map[IV.Phi] = advanceInductionVar(B, IV, VMap[IV.Phi], PB.Distance,
                                        IV.Phi->getName() + ".ahead",
                                        PB.StepOffsets[I]);
    }
    emitPush(B, PB.Queue,
             cloneCondition(B, PB, Map, ".ahead", /*Speculative=*/false,
                            P.Settings.Force));
    Value *OldCond = KBr->getCondition();
    B.SetInsertPoint(KBr);
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
  P.Latch->getTerminator()->setMetadata(
      LLVMContext::MD_loop,
      RISCVNS::makeLoopID(Ctx, OldLoopID, RISCVNS::LoopRole::Drain));
}

void RISCVBranchConditionPipelining::missed(Loop *L, StringRef Name,
                                            StringRef Reason) {
  ORE->emit([&]() {
    return OptimizationRemarkMissed(DEBUG_TYPE, Name, L->getStartLoc(),
                                    L->getHeader())
           << "not pipelining branch conditions in function "
           << ore::NV("Function", L->getHeader()->getParent()->getName())
           << ", loop " << ore::NV("Loop", loopName(*L)) << ": "
           << ore::NV("Reason", Reason);
  });
}

bool RISCVBranchConditionPipelining::runOnFunction(Function &F) {
  if (!RISCVNS::validateLoopOptions(F))
    return false;

  if (SusanCornersLoadHoistDepth > 3)
    report_fatal_error("SUSAN corner load-hoist depth must be 1-3");
  if (EnableSusanCornersInterleave2 && SusanCornersInterleaveFactor)
    report_fatal_error("select either SUSAN interleave2 or an interleave "
                       "factor, not both");
  if (SusanCornersInterleaveFactor &&
      (SusanCornersInterleaveFactor < 3 || SusanCornersInterleaveFactor > 8))
    report_fatal_error("SUSAN interleave factor must be 3-8; use "
                       "-riscv-ns-susan-corners-interleave2 for factor 2");

  // Only SelectionDAG knows how to select the queue intrinsics.
  auto &TM = getAnalysis<TargetPassConfig>().getTM<RISCVTargetMachine>();
  const RISCVSubtarget &ST = TM.getSubtarget<RISCVSubtarget>(F);
  DL = &F.getDataLayout();
  XLenTy = IntegerType::get(F.getContext(), ST.getXLen());
  DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  SE = &getAnalysis<ScalarEvolutionWrapperPass>().getSE();
  AA = &getAnalysis<AAResultsWrapperPass>().getAAResults();
  ORE = &getAnalysis<OptimizationRemarkEmitterWrapperPass>().getORE();

  bool Changed = false;
  if (SusanCornersLoadHoistDepth && F.getName() == "susan_corners") {
    bool Applied = false;
    for (Loop *L : LI->getLoopsInPreorder())
      if (L->getHeader()->getName() == "for.body14") {
        Applied = hoistSusanCornerLoads(L, SusanCornersLoadHoistDepth);
        break;
      }
    if (!Applied)
      report_fatal_error(Twine("could not safely apply the requested SUSAN "
                               "corner load-hoist depth: ") +
                         FailureReason);
    Changed = true;
  }

  unsigned SusanFactor =
      EnableSusanCornersInterleave2 ? 2 : SusanCornersInterleaveFactor;
  if (SusanFactor && F.getName() == "susan_corners")
    for (Loop *L : LI->getLoopsInPreorder())
      if (L->getHeader()->getName() == "for.body14") {
        bool Applied =
            SusanFactor == 2
                ? interleaveSusanCorners(L, /*Queue=*/0)
                : interleaveSusanCornersFactor(L, /*Queue=*/0, SusanFactor);
        if (Applied) {
          assert(!verifyFunction(F, &errs()) &&
                 "invalid SUSAN interleave transformation");
          return true;
        }
        missed(L,
               SusanFactor == 2 ? "SusanCornersInterleave2NotApplied"
                                : "SusanCornersInterleaveFactorNotApplied",
               "the scan loop did not meet the conservative SUSAN "
               "interleaving requirements");
        break;
      }

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
    auto Options = RISCVNS::parseLoopOptions(L->getLoopID(),
                                             *L->getHeader()->getTerminator());
    if (!Options)
      continue;
    const RISCVNS::SWPSettings Settings = Options->resolveSWP(
        {true, false, PipelineDistance, SpeculativePipelineDistance,
         EnableSpeculativePipelining, EnableShiftRecurrences, EnableToggles,
         MaxSliceSize, MaxLoopSize, MaxPrologueSize},
        DisableSoftwarePipelining);
    bool Annotated = Options->hasRequests();
    if (Annotated) {
      ORE->emit([&]() {
        std::string Description;
        raw_string_ostream OS(Description);
        auto Add = [&](StringRef Name, unsigned Value, bool Local) {
          OS << ", " << Name << '=' << Value
             << (Local ? " (metadata)" : " (default)");
        };
        OS << "Non-Spec settings for function " << F.getName() << ", loop ";
        L->getHeader()->printAsOperand(OS, false);
        Add("swp", Settings.Enable, Options->SWPEnable.has_value());
        Add("distance", Settings.Distance, Options->Distance.has_value());
        Add("speculative_distance", Settings.SpeculativeDistance,
            Options->SpeculativeDistance.has_value());
        Add("speculative", Settings.Speculative,
            Options->Speculative.has_value());
        Add("shift_recurrences", Settings.ShiftRecurrences,
            Options->ShiftRecurrences.has_value());
        Add("toggles", Settings.Toggles, Options->Toggles.has_value());
        Add("max_slice_size", Settings.MaxSliceSize,
            Options->MaxSliceSize.has_value());
        Add("max_loop_size", Settings.MaxLoopSize,
            Options->MaxLoopSize.has_value());
        Add("max_prologue_size", Settings.MaxPrologueSize,
            Options->MaxPrologueSize.has_value());
        Add("force", Settings.Force, Options->Force.has_value());
        return OptimizationRemarkAnalysis(DEBUG_TYPE, "LoopSettings",
                                          L->getStartLoc(), L->getHeader())
               << Description;
      });
    }
    StringRef Blocked;
    if (skipFunction(F))
      Blocked = "function is unavailable for optimization (optnone)";
    else if (TM.Options.EnableGlobalISel)
      Blocked = "queue lowering is unavailable with GlobalISel";
    else if (DisableSoftwarePipelining)
      Blocked = "global software pipelining disable";
    else if (!Settings.Enable)
      Blocked = "loop software pipelining disable or inherited zero distance";
    else if (F.hasOptSize() && !Settings.Force)
      Blocked = "optsize exclusion";
    else if (RISCVNS::isIRFissioned(L->getLoopID()) ||
             !RISCVNS::getSWPApplied(L->getLoopID()).empty())
      Blocked = "loop already transformed";
    if (!Blocked.empty()) {
      if (Annotated)
        missed(L, "RequestBlocked", Blocked);
      continue;
    }
    if (Annotated && Settings.Force)
      ORE->emit([&]() {
        return OptimizationRemarkAnalysis(DEBUG_TYPE, "Force", L->getStartLoc(),
                                          L->getHeader())
               << "force active for loop "
               << ore::NV("Loop", loopName(*L))
               << ": bypassing size, profitability, optsize and function "
                  "branch "
                  "budgets, memory dependence and speculative safety proofs";
      });
    std::optional<LoopPlan> P = analyzeLoop(L, Settings, Annotated);
    if (!P) {
      if (Annotated)
        missed(L, "RequestBlocked", FailureReason);
      continue;
    }

    // The kernel duplicates the loop's branches, and the guard and the
    // kernel exit add some.
    unsigned Added = 3;
    for (BasicBlock *BB : P->Blocks)
      if (BB->getTerminator()->getNumSuccessors() > 0)
        ++Added;
    if (Branches + Added > MaxFunctionBranches && !Settings.Force) {
      missed(L, "TooManyBranches",
             "function branch budget: the function could run out of branch "
             "registers");
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
      if (Annotated)
        R << " in function " << ore::NV("Function", F.getName())
          << ", loop " << ore::NV("Loop", loopName(*L))
          << "; requested distance "
          << ore::NV("RequestedDistance", P->Speculative
                                              ? Settings.SpeculativeDistance
                                              : Settings.Distance)
          << ", effective distance "
          << ore::NV("EffectiveDistance", P->maxDistance())
          << ", force=" << ore::NV("Force", Settings.Force);
      return R;
    });
    Plans.push_back(std::move(*P));
  }
  if (Plans.empty())
    return Changed;

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
      if (P.TripCount)
        ++NumToggleLoopsPipelined;
      else
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

namespace {

struct ChainInBoundsContract {
  StringRef Kind;
  Value *Base;
  Value *ImageBase;
  Value *XSize;
  Value *YSize;
  StringRef InnerLoop;
  StringRef OuterLoop;
  int64_t RowStart;
  int64_t RowEndTrim;
  int64_t ColumnStart;
  int64_t ColumnEndTrim;
  SmallVector<int64_t, 4> Offsets;
};

static bool getContractInt(Metadata *MD, int64_t &Value) {
  auto *CAM = dyn_cast<ConstantAsMetadata>(MD);
  auto *CI = CAM ? dyn_cast<ConstantInt>(CAM->getValue()) : nullptr;
  if (!CI || CI->getBitWidth() > 64)
    return false;
  Value = CI->getSExtValue();
  return true;
}

static bool parseChainInBoundsContract(const Instruction &I,
                                       ChainInBoundsContract &Contract) {
  MDNode *Node = I.getMetadata("riscv.ns.chain.inbounds");
  if (!Node || Node->getNumOperands() != 13)
    return false;
  auto *Version = dyn_cast<MDString>(Node->getOperand(0));
  auto *Kind = dyn_cast<MDString>(Node->getOperand(1));
  auto *InnerLoop = dyn_cast<MDString>(Node->getOperand(6));
  auto *OuterLoop = dyn_cast<MDString>(Node->getOperand(7));
  auto *Offsets = dyn_cast<MDNode>(Node->getOperand(12));
  if (!Version || Version->getString() != "v1" || !Kind ||
      (Kind->getString() != "image" && Kind->getString() != "lut") ||
      !InnerLoop || !OuterLoop || !Offsets)
    return false;

  Contract.Kind = Kind->getString();
  int64_t BaseArg = -1, ImageArg = -1, XSizeArg = -1, YSizeArg = -1;
  if (!getContractInt(Node->getOperand(2), BaseArg) ||
      !getContractInt(Node->getOperand(3), ImageArg) ||
      !getContractInt(Node->getOperand(4), XSizeArg) ||
      !getContractInt(Node->getOperand(5), YSizeArg) || BaseArg < 0 ||
      ImageArg < 0 || XSizeArg < 0 || YSizeArg < 0 ||
      static_cast<uint64_t>(BaseArg) >= I.getFunction()->arg_size() ||
      static_cast<uint64_t>(ImageArg) >= I.getFunction()->arg_size() ||
      static_cast<uint64_t>(XSizeArg) >= I.getFunction()->arg_size() ||
      static_cast<uint64_t>(YSizeArg) >= I.getFunction()->arg_size())
    return false;
  Contract.Base = I.getFunction()->getArg(BaseArg);
  Contract.ImageBase = I.getFunction()->getArg(ImageArg);
  Contract.XSize = I.getFunction()->getArg(XSizeArg);
  Contract.YSize = I.getFunction()->getArg(YSizeArg);
  Contract.InnerLoop = InnerLoop->getString();
  Contract.OuterLoop = OuterLoop->getString();
  if (!getContractInt(Node->getOperand(8), Contract.RowStart) ||
      !getContractInt(Node->getOperand(9), Contract.RowEndTrim) ||
      !getContractInt(Node->getOperand(10), Contract.ColumnStart) ||
      !getContractInt(Node->getOperand(11), Contract.ColumnEndTrim))
    return false;

  Contract.Offsets.clear();
  for (const MDOperand &Offset : Offsets->operands()) {
    int64_t Value = 0;
    if (!getContractInt(Offset.get(), Value))
      return false;
    Contract.Offsets.push_back(Value);
  }
  return true;
}

static MDNode *makeChainInBoundsContract(
    LLVMContext &Ctx, StringRef Kind, unsigned BaseArg, unsigned ImageArg,
    unsigned XSizeArg, unsigned YSizeArg, Loop *InnerLoop, Loop *OuterLoop,
    int64_t RowStart, int64_t RowEndTrim, int64_t ColumnStart,
    int64_t ColumnEndTrim, ArrayRef<int64_t> Offsets) {
  SmallVector<Metadata *, 16> OffsetMD;
  for (int64_t Offset : Offsets)
    OffsetMD.push_back(ConstantAsMetadata::get(ConstantInt::get(
        Type::getInt64Ty(Ctx), static_cast<uint64_t>(Offset), true)));
  Metadata *Fields[] = {
      MDString::get(Ctx, "v1"),
      MDString::get(Ctx, Kind),
      ConstantAsMetadata::get(
          ConstantInt::get(Type::getInt64Ty(Ctx), BaseArg)),
      ConstantAsMetadata::get(
          ConstantInt::get(Type::getInt64Ty(Ctx), ImageArg)),
      ConstantAsMetadata::get(
          ConstantInt::get(Type::getInt64Ty(Ctx), XSizeArg)),
      ConstantAsMetadata::get(
          ConstantInt::get(Type::getInt64Ty(Ctx), YSizeArg)),
      MDString::get(Ctx, InnerLoop->getHeader()->getName()),
      MDString::get(Ctx, OuterLoop ? OuterLoop->getHeader()->getName()
                                   : StringRef()),
      ConstantAsMetadata::get(ConstantInt::get(
          Type::getInt64Ty(Ctx), static_cast<uint64_t>(RowStart), true)),
      ConstantAsMetadata::get(ConstantInt::get(
          Type::getInt64Ty(Ctx), static_cast<uint64_t>(RowEndTrim), true)),
      ConstantAsMetadata::get(ConstantInt::get(
          Type::getInt64Ty(Ctx), static_cast<uint64_t>(ColumnStart), true)),
      ConstantAsMetadata::get(ConstantInt::get(
          Type::getInt64Ty(Ctx), static_cast<uint64_t>(ColumnEndTrim), true)),
      MDNode::get(Ctx, OffsetMD)};
  return MDNode::get(Ctx, Fields);
}

class RISCVChainLoadHoisting : public FunctionPass {
  enum class RejectKind {
    Dereferenceability,
    Memory,
    Control,
    Pressure,
    Dependence
  };

  struct ThresholdCheck {
    BranchInst *Branch;
    BasicBlock *Exit;
    BasicBlock *Continue;
    SmallVector<BasicBlock *, 4> Segment;
  };

  static constexpr unsigned MaxDistance = 8;
  static constexpr unsigned MaxLiveLoadResults = 8;

  bool findSegment(BasicBlock *Start, BasicBlock *End, Loop *L,
                   SmallVectorImpl<BasicBlock *> &Segment) const {
    SmallPtrSet<BasicBlock *, 8> Seen;
    BasicBlock *BB = Start;
    while (BB != End) {
      if (!L->contains(BB) || !Seen.insert(BB).second)
        return false;
      Segment.push_back(BB);
      auto *Br = dyn_cast<BranchInst>(BB->getTerminator());
      if (!Br || !Br->isUnconditional() || !L->contains(Br->getSuccessor(0)))
        return false;
      BB = Br->getSuccessor(0);
    }
    Segment.push_back(End);
    return true;
  }

  void markAccumulator(Value *V, SmallPtrSetImpl<Value *> &Values,
                       SmallPtrSetImpl<Value *> &Visited) const {
    auto *I = dyn_cast<Instruction>(V);
    if (!I || isa<LoadInst>(I) || !Visited.insert(V).second)
      return;
    if (I->getType()->isIntegerTy() && !isa<PHINode>(I))
      Values.insert(I);
    for (Value *Operand : I->operands())
      markAccumulator(Operand, Values, Visited);
  }

  void collectConditionLoads(Value *V,
                             const SmallPtrSetImpl<BasicBlock *> &Segment,
                             SmallPtrSetImpl<Value *> &Visited,
                             SmallPtrSetImpl<LoadInst *> &Loads) const {
    auto *I = dyn_cast<Instruction>(V);
    if (!I || !Visited.insert(V).second)
      return;
    if (auto *Load = dyn_cast<LoadInst>(I)) {
      if (Segment.contains(Load->getParent()))
        Loads.insert(Load);
      return;
    }
    if (isa<PHINode>(I))
      return;
    for (Value *Operand : I->operands())
      collectConditionLoads(Operand, Segment, Visited, Loads);
  }

  bool isProvenDereferenceable(LoadInst *Load, Instruction *At,
                               DominatorTree &DT,
                               std::string &FailureReason) const {
    const DataLayout &DL = Load->getDataLayout();
    if (isDereferenceableAndAlignedPointer(
            Load->getPointerOperand(), Load->getType(), Load->getAlign(), DL,
            At, nullptr, &DT))
      return true;

    auto *Base = dyn_cast<Argument>(
        getUnderlyingObject(Load->getPointerOperand()));
    if (!Base) {
      FailureReason = "pointer base is not a function argument";
      return false;
    }
    uint64_t DereferenceableBytes =
        Load->getFunction()->getAttributes().getParamDereferenceableBytes(
            Base->getArgNo());
    TypeSize AccessSize = DL.getTypeStoreSize(Load->getType());
    if (!DereferenceableBytes || AccessSize.isScalable() ||
        AccessSize.getFixedValue() > DereferenceableBytes) {
      FailureReason = "base argument has no sufficient dereferenceable-bytes "
                      "attribute";
      return false;
    }

    int64_t MinOffset = 0;
    int64_t MaxOffset = 0;
    Value *Ptr = Load->getPointerOperand();
    while (Ptr != Base) {
      auto *GEP = dyn_cast<GEPOperator>(Ptr);
      if (!GEP) {
        FailureReason = "pointer expression is not a simple GEP";
        return false;
      }
      APInt ConstantOffset(64, 0);
      if (GEP->accumulateConstantOffset(DL, ConstantOffset)) {
        MinOffset += ConstantOffset.getSExtValue();
        MaxOffset += ConstantOffset.getSExtValue();
      } else {
        if (GEP->getNumIndices() != 1 ||
            !GEP->getSourceElementType()->isIntegerTy(8)) {
          FailureReason = "dynamic pointer offset is not a byte GEP";
          return false;
        }
        Value *Index = *GEP->idx_begin();
        ConstantRange Range =
            [&]() -> ConstantRange {
          if (auto *ZExt = dyn_cast<ZExtInst>(Index)) {
            Type *SrcTy = ZExt->getSrcTy();
            if (SrcTy->isIntegerTy() && SrcTy->getIntegerBitWidth() < 63)
              return ConstantRange(
                  APInt(64, 0),
                  APInt(64, uint64_t(1) << SrcTy->getIntegerBitWidth()));
          }
          return computeConstantRange(Index, /*ForSigned=*/true,
                                      /*UseInstrInfo=*/true, nullptr, At, &DT);
        }();
        if (Range.isFullSet() || Range.isEmptySet() ||
            Range.isSizeLargerThan(DereferenceableBytes)) {
          FailureReason = "dynamic byte offset has no bounded range";
          return false;
        }
        MinOffset += Range.getSignedMin().getSExtValue();
        MaxOffset += Range.getSignedMax().getSExtValue();
      }
      Ptr = GEP->getPointerOperand();
    }

    bool InBounds =
        MinOffset >= 0 && MaxOffset >= MinOffset &&
        static_cast<uint64_t>(MaxOffset) + AccessSize.getFixedValue() <=
            DereferenceableBytes;
    if (!InBounds)
      FailureReason = "offset range exceeds the dereferenceable byte extent";
    return InBounds;
  }

  bool matchesContractLoop(const ChainInBoundsContract &Contract, Loop *L,
                           Instruction *At, DominatorTree &DT) const {
    if (!L || !L->contains(At->getParent()) ||
        L->getHeader()->getName() != Contract.InnerLoop)
      return false;
    Loop *Outer = L->getParentLoop();
    if ((Outer ? Outer->getHeader()->getName() : StringRef()) !=
        Contract.OuterLoop ||
        Contract.RowStart != 5 || Contract.RowEndTrim != 5 ||
        Contract.ColumnStart != 5 || Contract.ColumnEndTrim != 5)
      return false;
    return true;
  }

  bool isChainContractSafeLoad(
      LoadInst *Load, Loop *L, Instruction *At, DominatorTree &DT,
      ScalarEvolution &SE,
      SmallPtrSetImpl<Instruction *> &ContractAddressInstructions,
      SmallPtrSetImpl<LoadInst *> &ContractLoads,
      SmallPtrSetImpl<LoadInst *> &VisitedLoads,
      SmallPtrSetImpl<Instruction *> &VisitedAddress) const {
    if (!VisitedLoads.insert(Load).second)
      return true;

    ChainInBoundsContract Contract;
    if (!parseChainInBoundsContract(*Load, Contract) ||
        !matchesContractLoop(Contract, L, At, DT))
      return false;
    MDNode *ContractNode = Load->getMetadata("riscv.ns.chain.inbounds");

    if (Contract.Kind == "image") {
      if (Contract.Offsets.size() != 4 ||
          Contract.Offsets[0] != -3 || Contract.Offsets[1] != 3 ||
          Contract.Offsets[2] != -3 || Contract.Offsets[3] != 3 ||
          Contract.Base != Contract.ImageBase)
        return false;
      if (getSusanPointerBase(Load->getPointerOperand()) != Contract.Base)
        return false;
      MDNode *OffsetNode = Load->getMetadata("riscv.ns.chain.offset");
      if (!OffsetNode || OffsetNode->getNumOperands() != 2)
        return false;
      int64_t Dy = 0, Dx = 0;
      if (!getContractInt(OffsetNode->getOperand(0), Dy) ||
          !getContractInt(OffsetNode->getOperand(1), Dx) ||
          Dy < Contract.Offsets[0] || Dy > Contract.Offsets[1] ||
          Dx < Contract.Offsets[2] || Dx > Contract.Offsets[3])
        return false;
    } else {
      if (Contract.Offsets.size() != 2 || Contract.Offsets[0] != -255 ||
          Contract.Offsets[1] != 0)
        return false;
      auto *Access = dyn_cast<GEPOperator>(Load->getPointerOperand());
      if (!Access || Access->getNumIndices() != 1)
        return false;
      auto *Neg = dyn_cast<BinaryOperator>(Access->idx_begin()->get());
      auto *Zero = Neg ? dyn_cast<ConstantInt>(Neg->getOperand(0)) : nullptr;
      auto *ZExt = Neg ? dyn_cast<ZExtInst>(Neg->getOperand(1)) : nullptr;
      auto *ByteLoad =
          ZExt ? dyn_cast<LoadInst>(ZExt->getOperand(0)) : nullptr;
      if (!Neg || Neg->getOpcode() != Instruction::Sub || !Zero ||
          !Zero->isZero() || !ZExt ||
          !ZExt->getSrcTy()->isIntegerTy(8) || !ByteLoad ||
          Access->getPointerOperand()->getType() != Contract.Base->getType())
        return false;
      if (!isSusanLUTBase(Access->getPointerOperand(), Contract, ContractNode,
                          L, At, DT, SE, ContractAddressInstructions,
                          ContractLoads, VisitedLoads, VisitedAddress))
        return false;
      if (!isChainContractSafeLoad(ByteLoad, L, At, DT, SE,
                                   ContractAddressInstructions, ContractLoads,
                                   VisitedLoads, VisitedAddress))
        return false;
    }

    if (!collectContractAddress(
            Load->getPointerOperand(), Contract.Base, Contract,
            Load->getMetadata("riscv.ns.chain.inbounds"), L, At, DT, SE,
            ContractAddressInstructions, ContractLoads, VisitedLoads,
            VisitedAddress))
      return false;
    ContractLoads.insert(Load);
    return true;
  }

  bool isSusanLUTBase(
      Value *V, const ChainInBoundsContract &Contract, MDNode *ContractNode,
      Loop *L, Instruction *At, DominatorTree &DT, ScalarEvolution &SE,
      SmallPtrSetImpl<Instruction *> &ContractAddressInstructions,
      SmallPtrSetImpl<LoadInst *> &ContractLoads,
      SmallPtrSetImpl<LoadInst *> &VisitedLoads,
      SmallPtrSetImpl<Instruction *> &VisitedAddress) const {
    if (auto *Phi = dyn_cast<PHINode>(V)) {
      if (!Phi->getType()->isPointerTy() ||
          Phi->getMetadata("riscv.ns.chain.inbounds") != ContractNode)
        return false;
      ContractAddressInstructions.insert(Phi);
      if (!VisitedAddress.insert(Phi).second)
        return true;
      for (Value *Incoming : Phi->incoming_values())
        if (!isa<UndefValue>(Incoming) && !isa<PoisonValue>(Incoming) &&
            !isSusanLUTBase(Incoming, Contract, ContractNode, L, At, DT, SE,
                            ContractAddressInstructions, ContractLoads,
                            VisitedLoads, VisitedAddress))
          return false;
      return true;
    }

    auto *GEP = dyn_cast<GEPOperator>(V);
    auto *GEPInst = dyn_cast<Instruction>(V);
    if (!GEP || !GEPInst || GEP->getNumIndices() != 1 ||
        GEPInst->getMetadata("riscv.ns.chain.inbounds") != ContractNode)
      return false;
    auto *ZExt = dyn_cast<ZExtInst>(GEP->idx_begin()->get());
    auto *CenterLoad =
        ZExt ? dyn_cast<LoadInst>(ZExt->getOperand(0)) : nullptr;
    if (!ZExt || !ZExt->getSrcTy()->isIntegerTy(8) || !CenterLoad ||
        GEP->getPointerOperand() != Contract.Base ||
        getSusanPointerBase(CenterLoad->getPointerOperand()) !=
            Contract.ImageBase)
      return false;
    ContractAddressInstructions.insert(GEPInst);
    if (!VisitedAddress.insert(GEPInst).second)
      return true;
    return isChainContractSafeLoad(CenterLoad, L, At, DT, SE,
                                   ContractAddressInstructions, ContractLoads,
                                   VisitedLoads, VisitedAddress) &&
           collectContractAddress(
               GEP->getPointerOperand(), Contract.Base, Contract,
               ContractNode, L, At, DT, SE, ContractAddressInstructions,
               ContractLoads, VisitedLoads, VisitedAddress) &&
           collectContractAddress(
               GEP->idx_begin()->get(), Contract.Base, Contract,
               ContractNode, L, At, DT, SE, ContractAddressInstructions,
               ContractLoads, VisitedLoads, VisitedAddress);
  }

  bool collectContractAddress(
      Value *V, Value *Base, const ChainInBoundsContract &Contract,
      MDNode *ContractNode, Loop *L, Instruction *At, DominatorTree &DT,
      ScalarEvolution &SE,
      SmallPtrSetImpl<Instruction *> &ContractAddressInstructions,
      SmallPtrSetImpl<LoadInst *> &ContractLoads,
      SmallPtrSetImpl<LoadInst *> &VisitedLoads,
      SmallPtrSetImpl<Instruction *> &VisitedAddress) const {
    if (V == Base || V == Contract.XSize || V == Contract.YSize ||
        isa<Constant>(V))
      return true;
    if (auto *Load = dyn_cast<LoadInst>(V))
      return isChainContractSafeLoad(
          Load, L, At, DT, SE, ContractAddressInstructions, ContractLoads,
          VisitedLoads, VisitedAddress);

    auto *I = dyn_cast<Instruction>(V);
    if (!I || isa<CallBase>(I) || I->mayReadOrWriteMemory() ||
        I->isTerminator())
      return false;
    if (!VisitedAddress.insert(I).second)
      return true;
    if (auto *Phi = dyn_cast<PHINode>(I)) {
      if (Phi->getMetadata("riscv.ns.chain.inbounds") == ContractNode) {
        if (!Phi->getType()->isPointerTy())
          return false;
      } else {
        MDNode *IV = Phi->getMetadata("riscv.ns.chain.iv");
        auto *Version =
            IV && IV->getNumOperands() == 2
                ? dyn_cast<MDString>(IV->getOperand(0))
                : nullptr;
        auto *Header =
            IV && IV->getNumOperands() == 2
                ? dyn_cast<MDString>(IV->getOperand(1))
                : nullptr;
        if (!Phi->getType()->isIntegerTy() || !Version ||
            Version->getString() != "v1" || !Header ||
            (Header->getString() != Contract.InnerLoop &&
             Header->getString() != Contract.OuterLoop) ||
            Phi->getParent()->getName() != Header->getString())
          return false;
        return true;
      }
      ContractAddressInstructions.insert(Phi);
      for (Value *Incoming : Phi->incoming_values())
        if (!isa<UndefValue>(Incoming) && !isa<PoisonValue>(Incoming) &&
            !collectContractAddress(
                Incoming, Base, Contract, ContractNode, L, At, DT, SE,
                ContractAddressInstructions, ContractLoads, VisitedLoads,
                VisitedAddress))
          return false;
      return true;
    }
    if (I->getMetadata("riscv.ns.chain.inbounds") != ContractNode)
      return false;
    ContractAddressInstructions.insert(I);
    for (Value *Operand : I->operands())
      if (!collectContractAddress(
              Operand, Base, Contract, ContractNode, L, At, DT, SE,
              ContractAddressInstructions, ContractLoads, VisitedLoads,
              VisitedAddress))
        return false;
    return true;
  }

  bool reject(RejectKind Kind, StringRef Reason, Function &F,
              BranchInst *At, OptimizationRemarkEmitter &ORE) const {
    switch (Kind) {
    case RejectKind::Dereferenceability:
      ++NumChainLoadHoistRejectDereferenceability;
      break;
    case RejectKind::Memory:
      ++NumChainLoadHoistRejectMemory;
      break;
    case RejectKind::Control:
      ++NumChainLoadHoistRejectControl;
      break;
    case RejectKind::Pressure:
      ++NumChainLoadHoistRejectPressure;
      break;
    case RejectKind::Dependence:
      ++NumChainLoadHoistRejectDependence;
      break;
    }
    DEBUG_WITH_TYPE("riscv-chain-load-hoisting",
                    dbgs() << "rejecting load slice in " << F.getName()
                           << " at " << At->getParent()->getName() << ": "
                           << Reason << '\n');
    ORE.emit([&]() {
      return OptimizationRemarkMissed(
                 "riscv-chain-load-hoisting", "Rejected", At->getDebugLoc(),
                 At->getParent())
             << "did not hoist a load slice in " << F.getName() << ": "
             << Reason;
    });
    return false;
  }

  bool processChain(ArrayRef<ThresholdCheck> Checks, Loop *L, Function &F,
                    DominatorTree &DT, ScalarEvolution &SE,
                    BranchProbabilityInfo &BPI,
                    AAResults &AA, OptimizationRemarkEmitter &ORE,
                    DenseMap<Instruction *, BranchInst *> &MoveDestination) {
    if (Checks.size() < 2)
      return false;

    SmallPtrSet<Value *, 32> Accumulators;
    SmallPtrSet<Value *, 32> VisitedAccumulators;
    for (const ThresholdCheck &Check : Checks)
      if (auto *Cmp = dyn_cast<ICmpInst>(Check.Branch->getCondition()))
        for (Value *Operand : Cmp->operands())
          markAccumulator(Operand, Accumulators, VisitedAccumulators);

    DenseMap<unsigned, SmallVector<LoadInst *, 8>> LiveLoadsAtDestination;
    unsigned HoistedSlices = 0;
    for (unsigned Source = 1; Source < Checks.size(); ++Source) {
      unsigned Destination =
          Source > ChainLoadHoistDistance ? Source - ChainLoadHoistDistance : 0;

      double ReachProbability = 1.0;
      for (unsigned I = Destination; I < Source; ++I) {
        BranchInst *Br = Checks[I].Branch;
        BranchProbability ContinueProbability =
            BPI.getEdgeProbability(Br->getParent(), Checks[I].Continue);
        ReachProbability *=
            static_cast<double>(ContinueProbability.getNumerator()) /
            BranchProbability::getDenominator();
      }
      if (1.0 - ReachProbability > 0.99) {
        ++NumChainLoadHoistRejectControl;
        DEBUG_WITH_TYPE("riscv-chain-load-hoisting",
                        dbgs() << "rejecting load slice in " << F.getName()
                               << " at "
                               << Checks[Source].Branch->getParent()->getName()
                               << ": estimated exit-path waste exceeds 99%\n");
        continue;
      }

      SmallPtrSet<BasicBlock *, 8> SourceSegment;
      for (BasicBlock *BB : Checks[Source].Segment)
        SourceSegment.insert(BB);
      SmallPtrSet<LoadInst *, 8> ConditionLoads;
      SmallPtrSet<Value *, 32> VisitedCondition;
      collectConditionLoads(Checks[Source].Branch->getCondition(),
                            SourceSegment, VisitedCondition, ConditionLoads);
      if (ConditionLoads.empty())
        continue;

      SmallPtrSet<Instruction *, 32> ContractAddressInstructions;
      SmallPtrSet<LoadInst *, 8> ContractLoads;
      if (AssumeChainLoadInBounds)
        for (LoadInst *Load : ConditionLoads) {
          SmallPtrSet<Instruction *, 32> CandidateAddressInstructions;
          SmallPtrSet<LoadInst *, 8> CandidateLoads;
          SmallPtrSet<LoadInst *, 8> VisitedLoads;
          SmallPtrSet<Instruction *, 32> VisitedAddress;
          if (isChainContractSafeLoad(
                  Load, L, Checks[Destination].Branch, DT, SE,
                  CandidateAddressInstructions, CandidateLoads, VisitedLoads,
                  VisitedAddress)) {
            ContractAddressInstructions.insert(
                CandidateAddressInstructions.begin(),
                CandidateAddressInstructions.end());
            ContractLoads.insert(CandidateLoads.begin(), CandidateLoads.end());
          }
        }

      SmallPtrSet<Instruction *, 32> Slice;
      SmallPtrSet<LoadInst *, 8> SliceLoads;
      SmallPtrSet<LoadInst *, 8> AddressLoads;
      unsigned Failure = 0;
      LoadInst *UnsafeLoad = nullptr;
      std::string DereferenceabilityFailure;
      auto AddSlice = [&](auto &&Self, Value *V, bool IsAddressValue) -> bool {
        auto *I = dyn_cast<Instruction>(V);
        if (!I || DT.dominates(I, Checks[Destination].Branch))
          return true;
        if (Accumulators.contains(I)) {
          Failure = 5;
          return false;
        }
        if (!L->contains(I->getParent()) || isa<PHINode>(I) ||
            I->isTerminator() || isa<CallBase>(I)) {
          Failure = 3;
          return false;
        }
        if (auto *Load = dyn_cast<LoadInst>(I)) {
          if (!Load->isSimple()) {
            Failure = 2;
            return false;
          }
          if (IsAddressValue)
            AddressLoads.insert(Load);
          std::string ProofFailure;
          if (!isSafeToSpeculativelyExecute(
                  Load, Checks[Destination].Branch, nullptr, &DT, nullptr) &&
              !isProvenDereferenceable(Load, Checks[Destination].Branch, DT,
                                       ProofFailure) &&
              !ContractLoads.contains(Load)) {
            Failure = 1;
            UnsafeLoad = Load;
            DereferenceabilityFailure =
                ProofFailure.empty()
                    ? "missing or invalid chain-inbounds metadata"
                    : ProofFailure;
            return false;
          }
          SliceLoads.insert(Load);
          if (!Slice.insert(I).second)
            return true;
          return Self(Self, Load->getPointerOperand(), true);
        }
        if (I->mayReadOrWriteMemory() ||
            (!isSafeToSpeculativelyExecute(
                 I, Checks[Destination].Branch, nullptr, &DT, nullptr, true,
                 false) &&
             !ContractAddressInstructions.contains(I))) {
          Failure = 3;
          return false;
        }
        if (!Slice.insert(I).second)
          return true;
        for (Value *Operand : I->operands())
          if (!Self(Self, Operand, IsAddressValue))
            return false;
        return true;
      };

      bool Safe = true;
      for (LoadInst *Load : ConditionLoads)
        if (!AddSlice(AddSlice, Load, false)) {
          Safe = false;
          break;
        }
      if (!Safe) {
        if (Failure == 1) {
          std::string Detail =
              (Twine("load '") + UnsafeLoad->getName() +
               "' through pointer '" +
               UnsafeLoad->getPointerOperand()->getName() + "' is not proven "
               "dereferenceable at the hoist point (" +
               DereferenceabilityFailure + ")")
                  .str();
          reject(RejectKind::Dereferenceability,
                 Detail, F, Checks[Source].Branch, ORE);
        } else if (Failure == 2)
          reject(RejectKind::Memory,
                 "volatile or atomic load in the slice", F,
                 Checks[Source].Branch, ORE);
        else if (Failure == 5)
          reject(RejectKind::Dependence,
                 "address depends on the threshold accumulator", F,
                 Checks[Source].Branch, ORE);
        else
          reject(RejectKind::Control,
                 "address slice contains a control-dependent or unsafe "
                 "instruction",
                 F, Checks[Source].Branch, ORE);
        continue;
      }

      SmallVector<BasicBlock *, 16> CrossedBlocks;
      for (unsigned I = Destination + 1; I <= Source; ++I)
        CrossedBlocks.append(Checks[I].Segment.begin(),
                             Checks[I].Segment.end());
      llvm::sort(CrossedBlocks);
      CrossedBlocks.erase(
          std::unique(CrossedBlocks.begin(), CrossedBlocks.end()),
          CrossedBlocks.end());

      bool HasMemoryConflict = false;
      for (LoadInst *Load : SliceLoads) {
        MemoryLocation LoadLoc = MemoryLocation::get(Load);
        for (BasicBlock *BB : CrossedBlocks) {
          for (Instruction &I : *BB) {
            if (&I == Load)
              continue;
            if (isa<CallBase>(I) || isa<FenceInst>(I)) {
              HasMemoryConflict = true;
              break;
            }
            if (auto *Store = dyn_cast<StoreInst>(&I)) {
              if (AA.alias(LoadLoc, MemoryLocation::get(Store)) !=
                  AliasResult::NoAlias) {
                HasMemoryConflict = true;
                break;
              }
            } else if (I.mayWriteToMemory()) {
              HasMemoryConflict = true;
              break;
            }
          }
          if (HasMemoryConflict)
            break;
        }
        if (HasMemoryConflict)
          break;
      }
      if (HasMemoryConflict) {
        reject(RejectKind::Memory,
               "a call, fence, or may-alias write lies on the crossed path",
               F, Checks[Source].Branch, ORE);
        continue;
      }

      SmallPtrSet<LoadInst *, 8> TerminalLoads;
      for (LoadInst *Load : SliceLoads)
        if (!AddressLoads.contains(Load))
          TerminalLoads.insert(Load);
      SmallVector<LoadInst *, 8> &LiveLoads =
          LiveLoadsAtDestination[Destination];
      unsigned AddedLiveLoads = 0;
      for (LoadInst *Load : TerminalLoads)
        AddedLiveLoads += !llvm::is_contained(LiveLoads, Load);
      if (LiveLoads.size() + AddedLiveLoads > MaxLiveLoadResults) {
        reject(RejectKind::Pressure,
               "the in-flight loaded values exceed the live-value cap", F,
               Checks[Source].Branch, ORE);
        continue;
      }

      for (LoadInst *Load : TerminalLoads)
        if (!llvm::is_contained(LiveLoads, Load))
          LiveLoads.push_back(Load);
      for (Instruction *I : Slice) {
        auto It = MoveDestination.find(I);
        BranchInst *Anchor = Checks[Destination].Branch;
        if (It == MoveDestination.end() ||
            DT.dominates(Anchor, It->second))
          MoveDestination[I] = Anchor;
      }
      unsigned Distance = Source - Destination;
      Checks[Source].Branch->setMetadata(
          "riscv.ns.chain-hoist",
          MDNode::get(F.getContext(),
                      MDString::get(F.getContext(),
                                    Twine(Distance).str())));
      ++HoistedSlices;
      ++NumChainLoadHoists;
    }

    if (!HoistedSlices)
      return false;
    ORE.emit([&]() {
      return OptimizationRemark("riscv-chain-load-hoisting", "Hoisted",
                                Checks.front().Branch->getDebugLoc(),
                                Checks.front().Branch->getParent())
             << "hoisted " << ore::NV("Slices", HoistedSlices)
             << " early-exit load slice(s) with distance "
             << ore::NV("Distance", ChainLoadHoistDistance);
    });
    return true;
  }

public:
  bool isLoadSafeForThinning(LoadInst *Load, Loop *L, Instruction *At,
                             DominatorTree &DT, ScalarEvolution &SE) const {
    if (!Load->isSimple())
      return false;
    if (AssumeChainLoadInBounds) {
      SmallPtrSet<Instruction *, 32> ContractAddressInstructions;
      SmallPtrSet<LoadInst *, 8> ContractLoads;
      SmallPtrSet<LoadInst *, 8> VisitedLoads;
      SmallPtrSet<Instruction *, 32> VisitedAddress;
      if (isChainContractSafeLoad(Load, L, At, DT, SE,
                                  ContractAddressInstructions, ContractLoads,
                                  VisitedLoads, VisitedAddress))
        return true;
    }

    std::string FailureReason;
    return isSafeToSpeculativelyExecute(Load, At, nullptr, &DT, nullptr) ||
           isProvenDereferenceable(Load, At, DT, FailureReason);
  }

public:
  static char ID;
  RISCVChainLoadHoisting() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override {
    if (!ChainLoadHoistDistance || skipFunction(F))
      return false;
    if (ChainLoadHoistDistance > MaxDistance)
      report_fatal_error("early-exit load-hoist distance must be in [0, 8]");

    auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    auto &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    auto &SE = getAnalysis<ScalarEvolutionWrapperPass>().getSE();
    auto &BPI = getAnalysis<BranchProbabilityInfoWrapperPass>().getBPI();
    auto &AA = getAnalysis<AAResultsWrapperPass>().getAAResults();
    auto &ORE = getAnalysis<OptimizationRemarkEmitterWrapperPass>().getORE();
    DenseMap<Instruction *, BranchInst *> MoveDestination;

    for (Loop *L : LI.getLoopsInPreorder()) {
      DenseMap<BasicBlock *, SmallVector<BranchInst *, 4>> ExitCandidates;
      for (BasicBlock *BB : L->blocks()) {
        if (LI.getLoopFor(BB) != L)
          continue;
        auto *Br = dyn_cast<BranchInst>(BB->getTerminator());
        if (!Br || !Br->isConditional())
          continue;
        if (L->contains(Br->getSuccessor(0)) !=
            L->contains(Br->getSuccessor(1))) {
          ++NumChainLoadHoistRejectControl;
          DEBUG_WITH_TYPE("riscv-chain-load-hoisting",
                          dbgs() << "not hoisting across loop exit/guard in "
                                 << F.getName() << " at " << BB->getName()
                                 << '\n');
          ORE.emit([&]() {
            return OptimizationRemarkMissed(
                       "riscv-chain-load-hoisting", "LoopExitOrGuard",
                       Br->getDebugLoc(), BB)
                   << "load hoisting does not cross a loop-exit or loop-guard "
                      "branch";
          });
          continue;
        }
        for (unsigned I = 0; I != 2; ++I)
          if (L->contains(Br->getSuccessor(I)))
            ExitCandidates[Br->getSuccessor(I)].push_back(Br);
      }
      DenseMap<BranchInst *, ThresholdCheck> Candidates;
      for (auto &Entry : ExitCandidates) {
        if (Entry.second.size() < 2)
          continue;
        BasicBlock *Exit = Entry.first;
        for (BranchInst *Br : Entry.second) {
          if (Br->getSuccessor(0) == Exit &&
              Br->getSuccessor(1) == Exit)
            continue;
          BasicBlock *Continue = Br->getSuccessor(0) == Exit
                                     ? Br->getSuccessor(1)
                                     : Br->getSuccessor(0);
          if (!L->contains(Continue) || Continue == Exit)
            continue;
          if (Candidates.count(Br)) {
            Candidates.erase(Br);
            ++NumChainLoadHoistRejectControl;
            continue;
          }
          Candidates[Br] = {Br, Exit, Continue, {}};
        }
      }
      DenseMap<BasicBlock *, BranchInst *> CandidateAt;
      for (auto &Entry : Candidates)
        CandidateAt[Entry.first->getParent()] = Entry.first;
      DEBUG_WITH_TYPE("riscv-chain-check-thinning",
                      dbgs() << "found " << Candidates.size()
                             << " threshold candidates in " << F.getName()
                             << " loop " << L->getHeader()->getName() << '\n');

      DenseMap<BranchInst *, BranchInst *> NextCheck;
      for (auto &Entry : Candidates) {
        BranchInst *Br = Entry.first;
        BasicBlock *BB = Entry.second.Continue;
        SmallPtrSet<BasicBlock *, 8> Seen;
        while (L->contains(BB) && Seen.insert(BB).second) {
          auto It = CandidateAt.find(BB);
          if (It != CandidateAt.end()) {
            if (Candidates[It->second].Exit == Entry.second.Exit)
              NextCheck[Br] = It->second;
            break;
          }
          auto *Step = dyn_cast<BranchInst>(BB->getTerminator());
          if (!Step || !Step->isUnconditional())
            break;
          BB = Step->getSuccessor(0);
        }
      }

      SmallPtrSet<BranchInst *, 16> HasPrevious;
      for (auto &Entry : NextCheck)
        HasPrevious.insert(Entry.second);
      SmallPtrSet<BranchInst *, 16> Visited;
      for (auto &Entry : Candidates) {
        BranchInst *Start = Entry.first;
        if (HasPrevious.contains(Start) || Visited.contains(Start))
          continue;

        SmallVector<ThresholdCheck, 20> Checks;
        BranchInst *Current = Start;
        while (Current && Visited.insert(Current).second) {
          Checks.push_back(Candidates[Current]);
          auto It = NextCheck.find(Current);
          Current = It == NextCheck.end() ? nullptr : It->second;
        }
        DEBUG_WITH_TYPE("riscv-chain-check-thinning",
                        dbgs() << "found threshold chain of " << Checks.size()
                               << " checks in " << F.getName() << '\n');
        if (Checks.size() < 2)
          continue;
        bool Valid = true;
        for (unsigned I = 1; I < Checks.size(); ++I)
          if (!findSegment(Checks[I - 1].Continue,
                           Checks[I].Branch->getParent(), L,
                           Checks[I].Segment)) {
            Valid = false;
            ++NumChainLoadHoistRejectControl;
            DEBUG_WITH_TYPE(
                "riscv-chain-load-hoisting",
                dbgs() << "rejecting chain in " << F.getName()
                       << ": a loop guard or non-linear branch lies between "
                          "threshold checks\n");
            break;
          }
        if (!Valid)
          continue;
        processChain(Checks, L, F, DT, SE, BPI, AA, ORE, MoveDestination);
      }
    }

    if (!MoveDestination.empty()) {
      DenseMap<BranchInst *, SmallVector<Instruction *, 16>> ByDestination;
      for (auto &Entry : MoveDestination)
        ByDestination[Entry.second].push_back(Entry.first);

      for (auto &Entry : ByDestination) {
        BranchInst *Anchor = Entry.first;
        SmallPtrSet<Instruction *, 32> Visited;
        SmallVector<Instruction *, 16> Ordered;
        auto Visit = [&](auto &&Self, Instruction *I) -> void {
          if (!Visited.insert(I).second)
            return;
          for (Value *Operand : I->operands())
            if (auto *Dep = dyn_cast<Instruction>(Operand))
              if (MoveDestination.lookup(Dep) == Anchor)
                Self(Self, Dep);
          Ordered.push_back(I);
        };
        for (Instruction *I : Entry.second)
          Visit(Visit, I);
        for (Instruction *I : Ordered)
          I->moveBefore(Anchor->getIterator());
      }
      return true;
    }
    return false;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<ScalarEvolutionWrapperPass>();
    AU.addRequired<BranchProbabilityInfoWrapperPass>();
    AU.addRequired<AAResultsWrapperPass>();
    AU.addRequired<OptimizationRemarkEmitterWrapperPass>();
    AU.setPreservesCFG();
  }
};

class RISCVSusanResponseMapCompaction : public FunctionPass {
  static BasicBlock *findBlock(Function &F, StringRef Name) {
    for (BasicBlock &BB : F)
      if (BB.getName() == Name)
        return &BB;
    return nullptr;
  }

public:
  static char ID;
  RISCVSusanResponseMapCompaction() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override {
    if (!SusanCornersResponseCompact || skipFunction(F) ||
        F.getName() != "susan_corners")
      return false;

    auto &ORE = getAnalysis<OptimizationRemarkEmitterWrapperPass>().getORE();
    auto Reject = [&](bool MissingContract, StringRef Reason) {
      if (MissingContract)
        ++NumSusanResponseCompactRejectContract;
      else
        ++NumSusanResponseCompactRejectPattern;
      DEBUG_WITH_TYPE("riscv-susan-response-compaction",
                      dbgs() << "rejecting SUSAN response-map compaction in "
                             << F.getName() << ": " << Reason << '\n');
      ORE.emit([&]() {
        return OptimizationRemarkMissed(
                   "riscv-susan-response-compaction", "NotCompacted",
                   DebugLoc(), &F.getEntryBlock())
               << "did not compact the SUSAN response-map scan: " << Reason;
      });
      return false;
    };

    bool HasChainContract = false;
    for (Instruction &I : instructions(F))
      if (auto *Load = dyn_cast<LoadInst>(&I))
        HasChainContract |=
            Load->getMetadata("riscv.ns.chain.inbounds") != nullptr;
    if (!HasChainContract)
      return Reject(true, "interleave chain-inbounds metadata is missing");

    if (F.arg_size() < 7 ||
        !F.getAttributes().hasParamAttr(1, Attribute::NoAlias))
      return Reject(false, "the SUSAN argument and aliasing contract changed");

    BasicBlock *Scan = findBlock(F, "for.body916");
    BasicBlock *ScanPreheader = findBlock(F, "for.body916.lr.ph");
    BasicBlock *ScanLatch = findBlock(F, "for.inc1378");
    BasicBlock *RowExit = findBlock(F, "for.inc1381.loopexit");
    BasicBlock *RowLatch = findBlock(F, "for.inc1381");
    if (!Scan || !ScanPreheader || !ScanLatch || !RowExit || !RowLatch ||
        ScanPreheader->getTerminator()->getNumSuccessors() != 1 ||
        ScanPreheader->getTerminator()->getSuccessor(0) != Scan)
      return Reject(false, "the response-map loop blocks are not recognized");
    BasicBlock *RowHeader = ScanPreheader->getSinglePredecessor();
    auto *RowBr = RowHeader
                      ? dyn_cast<BranchInst>(RowHeader->getTerminator())
                      : nullptr;
    auto *RowGuard =
        RowBr && RowBr->isConditional()
            ? dyn_cast<ICmpInst>(RowBr->getCondition())
            : nullptr;
    bool StartsAtInteriorRow = false;
    if (RowHeader)
      for (PHINode &Phi : RowHeader->phis())
        for (Value *Incoming : Phi.incoming_values())
          if (auto *Start = dyn_cast<ConstantInt>(Incoming))
            StartsAtInteriorRow |= Start->getSExtValue() == 5;
    auto *MinimumWidth = RowGuard
                             ? dyn_cast<ConstantInt>(RowGuard->getOperand(1))
                             : nullptr;
    if (!RowGuard || RowGuard->getPredicate() != ICmpInst::ICMP_SGT ||
        RowGuard->getOperand(0) != F.getArg(5) ||
        !MinimumWidth || MinimumWidth->getSExtValue() != 10 ||
        (RowBr->getSuccessor(0) != ScanPreheader &&
         RowBr->getSuccessor(1) != ScanPreheader) ||
        !StartsAtInteriorRow)
      return Reject(false, "the row guard and interior bounds are not recognized");

    PHINode *Pixel = nullptr;
    PHINode *CornerCount = nullptr;
    for (PHINode &Phi : Scan->phis()) {
      if (Phi.getType()->isIntegerTy(64)) {
        if (Pixel)
          return Reject(false, "the pixel induction variable is ambiguous");
        Pixel = &Phi;
      } else if (Phi.getType()->isIntegerTy(32)) {
        if (CornerCount)
          return Reject(false, "the corner-count recurrence is ambiguous");
        CornerCount = &Phi;
      } else {
        return Reject(false, "the response-map loop has an unexpected phi");
      }
    }
    if (!Pixel || !CornerCount ||
        Pixel->getBasicBlockIndex(ScanPreheader) < 0 ||
        Pixel->getBasicBlockIndex(ScanLatch) < 0 ||
        CornerCount->getBasicBlockIndex(ScanPreheader) < 0 ||
        CornerCount->getBasicBlockIndex(ScanLatch) < 0)
      return Reject(false, "the response-map induction variables changed");

    auto *PixelStart =
        dyn_cast<ConstantInt>(Pixel->getIncomingValueForBlock(ScanPreheader));
    if (!PixelStart || PixelStart->getSExtValue() != 5)
      return Reject(false, "the response-map scan no longer starts at column 5");

    auto *ScanBr = dyn_cast<BranchInst>(Scan->getTerminator());
    if (!ScanBr || !ScanBr->isConditional())
      return Reject(false, "the response-map scan has no conditional branch");
    BasicBlock *Body = nullptr;
    for (unsigned I = 0; I != 2; ++I)
      if (ScanBr->getSuccessor(I)->getName() == "if.then923")
        Body = ScanBr->getSuccessor(I);
    if (!Body ||
        !((ScanBr->getSuccessor(0) == Body &&
           ScanBr->getSuccessor(1) == ScanLatch) ||
          (ScanBr->getSuccessor(1) == Body &&
           ScanBr->getSuccessor(0) == ScanLatch)))
      return Reject(false, "the survivor test no longer guards the local-max body");

    LoadInst *ResponseLoad = nullptr;
    ICmpInst *SurvivorTest = nullptr;
    for (Instruction &I : *Scan) {
      auto *Load = dyn_cast<LoadInst>(&I);
      auto *GEP = Load ? dyn_cast<GetElementPtrInst>(Load->getPointerOperand())
                       : nullptr;
      if (Load && Load->getType()->isIntegerTy(32) && GEP &&
          getUnderlyingObject(GEP->getPointerOperand()) == F.getArg(1)) {
        ResponseLoad = Load;
        break;
      }
    }
    for (Instruction &I : *Scan)
      if (auto *Cmp = dyn_cast<ICmpInst>(&I))
        if (Cmp->getPredicate() == ICmpInst::ICMP_SGT &&
            ResponseLoad && Cmp->getOperand(0) == ResponseLoad &&
            match(Cmp->getOperand(1), m_Zero()))
          SurvivorTest = Cmp;
    if (!ResponseLoad || !SurvivorTest)
      return Reject(false, "the response-map load or positive-value test changed");

    auto *PixelGEP =
        cast<GetElementPtrInst>(ResponseLoad->getPointerOperand());
    if (!PixelGEP->getSourceElementType()->isIntegerTy(32))
      return Reject(false, "the response map is not an int array");
    Value *PixelOffset = PixelGEP->getOperand(1);
    auto *PixelOffsetAdd = dyn_cast<BinaryOperator>(PixelOffset);
    if (!PixelOffsetAdd || PixelOffsetAdd->getOpcode() != Instruction::Add)
      return Reject(false, "the response-map address is not row-base plus column");
    Value *RowBase = nullptr;
    if (PixelOffsetAdd->getOperand(0) == Pixel)
      RowBase = PixelOffsetAdd->getOperand(1);
    else if (PixelOffsetAdd->getOperand(1) == Pixel)
      RowBase = PixelOffsetAdd->getOperand(0);
    if (!RowBase || RowBase->getType() != Pixel->getType())
      return Reject(false, "the response-map row base is not recognized");

    PHINode *LatchCornerCount = nullptr;
    for (PHINode &Phi : ScanLatch->phis())
      if (Phi.getType() == CornerCount->getType()) {
        if (LatchCornerCount)
          return Reject(false, "the corner-count latch phi is ambiguous");
        LatchCornerCount = &Phi;
      }
    if (!LatchCornerCount ||
        LatchCornerCount->getBasicBlockIndex(Scan) < 0)
      return Reject(false, "the corner-count latch does not match the scan");

    auto *LatchBr = dyn_cast<BranchInst>(ScanLatch->getTerminator());
    if (!LatchBr || !LatchBr->isConditional())
      return Reject(false, "the response-map latch has no loop test");
    BasicBlock *LoopExit = nullptr;
    for (unsigned I = 0; I != 2; ++I)
      if (LatchBr->getSuccessor(I) == Scan)
        continue;
      else
        LoopExit = LatchBr->getSuccessor(I);
    if (LoopExit != RowExit)
      return Reject(false, "the response-map loop exit is not recognized");

    BinaryOperator *PixelNext = nullptr;
    ICmpInst *ScanDone = nullptr;
    for (Instruction &I : *ScanLatch)
      if (auto *Add = dyn_cast<BinaryOperator>(&I))
        if (Add->getOpcode() == Instruction::Add &&
            Add->getOperand(0) == Pixel)
          if (auto *One = dyn_cast<ConstantInt>(Add->getOperand(1)))
            if (One->isOne())
              PixelNext = Add;
    if (PixelNext)
      for (Instruction &I : *ScanLatch)
        if (auto *Cmp = dyn_cast<ICmpInst>(&I))
          if (Cmp->getPredicate() == ICmpInst::ICMP_EQ &&
              (Cmp->getOperand(0) == PixelNext ||
               Cmp->getOperand(1) == PixelNext) &&
              LatchBr->getCondition() == Cmp)
            ScanDone = Cmp;
    if (!PixelNext || !ScanDone)
      return Reject(false, "the response-map loop increment is not recognized");
    Value *TripCount = ScanDone->getOperand(
        ScanDone->getOperand(0) == PixelNext ? 1 : 0);

    CallBase *CGX = nullptr;
    for (Instruction &I : F.getEntryBlock())
      if (auto *CB = dyn_cast<CallBase>(&I))
        if (Function *Callee = CB->getCalledFunction())
          if (Callee->getName() == "malloc") {
            CGX = CB;
            break;
          }
    if (!CGX || !CGX->getType()->isPointerTy())
      return Reject(false, "the SUSAN candidate scratch allocation is missing");

    SmallPtrSet<BasicBlock *, 32> BodyBlocks;
    SmallVector<BasicBlock *, 32> WorkList{Body};
    while (!WorkList.empty()) {
      BasicBlock *BB = WorkList.pop_back_val();
      if (BB == ScanLatch || !BodyBlocks.insert(BB).second)
        continue;
      if (BB == Scan || BB == ScanPreheader || BB == RowExit)
        return Reject(false, "the local-max body loops back into the scan");
      Instruction *Term = BB->getTerminator();
      if (!Term)
        return Reject(false, "the local-max body has an unterminated block");
      for (BasicBlock *Succ : successors(BB))
        if (Succ != ScanLatch)
          WorkList.push_back(Succ);
    }
    if (!BodyBlocks.contains(Body))
      return Reject(false, "the local-max body is empty");
    for (PHINode &Phi : Body->phis())
      if (Phi.getBasicBlockIndex(Scan) >= 0)
        return Reject(false, "the local-max body has a scan-edge phi");

    auto UsesOnlyIn = [&](Value *V, const SmallPtrSetImpl<BasicBlock *> &Allowed,
                          Instruction *AlsoAllowed = nullptr) {
      for (Use &U : V->uses()) {
        auto *User = dyn_cast<Instruction>(U.getUser());
        if (!User || isa<DbgInfoIntrinsic>(User))
          continue;
        if (User == AlsoAllowed || Allowed.contains(User->getParent()))
          continue;
        return false;
      }
      return true;
    };
    if (!UsesOnlyIn(CornerCount, BodyBlocks, LatchCornerCount))
      return Reject(false, "the local-max body has unsupported live-ins");

    Value *OuterCornerCount =
        CornerCount->getIncomingValueForBlock(ScanPreheader);
    if (!OuterCornerCount)
      return Reject(false, "the row corner-count value is missing");

    PHINode *RowNextPhi = nullptr;
    PHINode *RowCornerCountPhi = nullptr;
    Value *RowNext = nullptr;
    for (PHINode &Phi : RowLatch->phis()) {
      int Index = Phi.getBasicBlockIndex(RowExit);
      if (Index < 0)
        continue;
      if (Phi.getType() == Pixel->getType()) {
        if (RowNextPhi)
          return Reject(false, "the row-exit induction phi is ambiguous");
        RowNextPhi = &Phi;
        RowNext = Phi.getIncomingValue(Index);
      } else if (Phi.getType() == CornerCount->getType()) {
        if (RowCornerCountPhi ||
            Phi.getIncomingValue(Index) != LatchCornerCount)
          return Reject(false, "the row-exit corner-count phi is ambiguous");
        RowCornerCountPhi = &Phi;
      }
    }
    if (!RowNextPhi || !RowCornerCountPhi || !RowNext)
      return Reject(false, "the row-exit phis are not recognized");

    LLVMContext &Ctx = F.getContext();
    Type *I32Ty = Type::getInt32Ty(Ctx);
    Type *IndexTy = Pixel->getType();
    MDNode *CompactMD = MDNode::get(
        Ctx, MDString::get(Ctx, "image-buffer-xysize-v1"));

    BasicBlock *ScanContinue =
        BasicBlock::Create(Ctx, "susan.response.scan.continue", &F, Body);
    BasicBlock *Candidates =
        BasicBlock::Create(Ctx, "susan.response.candidates", &F, Body);
    BasicBlock *CandidateBody =
        BasicBlock::Create(Ctx, "susan.response.candidate.body", &F, Body);

    PHINode *CandidateCount =
        PHINode::Create(IndexTy, 2, "response.candidate.count");
    CandidateCount->insertBefore(Scan->getFirstNonPHIIt());
    CandidateCount->addIncoming(ConstantInt::get(IndexTy, 0), ScanPreheader);

    IRBuilder<> ScanBuilder(Scan->getTerminator());
    ScanBuilder.SetCurrentDebugLocation(ResponseLoad->getDebugLoc());
    Value *ScratchSlot = ScanBuilder.CreateGEP(
        I32Ty, CGX, CandidateCount, "response.candidate.slot");
    Value *Pixel32 = ScanBuilder.CreateTrunc(Pixel, I32Ty, "response.candidate.x");
    auto *ScratchStore =
        ScanBuilder.CreateStore(Pixel32, ScratchSlot);
    ScratchStore->setAlignment(Align(4));
    ScratchStore->setMetadata("riscv.ns.susan-response-compact", CompactMD);
    Value *CountIncrement =
        ScanBuilder.CreateZExt(SurvivorTest, IndexTy, "response.candidate.valid");
    Value *CountNext = ScanBuilder.CreateAdd(
        CandidateCount, CountIncrement, "response.candidate.count.next");
    ScanBuilder.CreateBr(ScanContinue);
    ScanBr->eraseFromParent();
    CandidateCount->addIncoming(CountNext, ScanContinue);

    BranchInst *ContinuePlaceholder = BranchInst::Create(Scan, ScanContinue);
    IRBuilder<> ContinueBuilder(ContinuePlaceholder);
    ContinueBuilder.SetCurrentDebugLocation(ResponseLoad->getDebugLoc());
    Value *PixelNextNew =
        ContinueBuilder.CreateAdd(Pixel, ConstantInt::get(IndexTy, 1),
                                  PixelNext->getName());
    cast<BinaryOperator>(PixelNextNew)->setHasNoUnsignedWrap(
        PixelNext->hasNoUnsignedWrap());
    cast<BinaryOperator>(PixelNextNew)->setHasNoSignedWrap(
        PixelNext->hasNoSignedWrap());
    Value *ScanDoneNew =
        ContinueBuilder.CreateICmpEQ(PixelNextNew, TripCount, "response.scan.done");
    ContinueBuilder.CreateCondBr(ScanDoneNew, Candidates, Scan);
    ContinuePlaceholder->eraseFromParent();
    int PixelLatchIndex = Pixel->getBasicBlockIndex(ScanLatch);
    Pixel->setIncomingValue(PixelLatchIndex, PixelNextNew);
    Pixel->setIncomingBlock(PixelLatchIndex, ScanContinue);

    for (int I = LatchCornerCount->getNumIncomingValues() - 1; I >= 0; --I)
      if (LatchCornerCount->getIncomingBlock(I) == Scan)
        LatchCornerCount->removeIncomingValue(I, false);

    BranchInst *CandidatesPlaceholder = BranchInst::Create(RowExit, Candidates);
    PHINode *CandidateCountFinal =
        PHINode::Create(IndexTy, 1, "response.candidate.count.final");
    CandidateCountFinal->insertBefore(CandidatesPlaceholder->getIterator());
    CandidateCountFinal->addIncoming(CountNext, ScanContinue);
    IRBuilder<> CandidateBuilder(CandidatesPlaceholder);
    CandidateBuilder.SetCurrentDebugLocation(ResponseLoad->getDebugLoc());
    Value *NoCandidates = CandidateBuilder.CreateICmpEQ(
        CandidateCountFinal, ConstantInt::get(IndexTy, 0),
        "response.no.candidates");
    CandidateBuilder.CreateCondBr(NoCandidates, RowExit, CandidateBody);
    CandidatesPlaceholder->eraseFromParent();

    BranchInst *OldLatchBr = cast<BranchInst>(ScanLatch->getTerminator());
    OldLatchBr->eraseFromParent();
    SmallVector<DbgInfoIntrinsic *, 4> DebugUsers;
    for (User *U : PixelNext->users())
      if (auto *Dbg = dyn_cast<DbgInfoIntrinsic>(U))
        DebugUsers.push_back(Dbg);
    for (DbgInfoIntrinsic *Dbg : DebugUsers)
      Dbg->replaceUsesOfWith(PixelNext, UndefValue::get(PixelNext->getType()));
    DebugUsers.clear();
    for (User *U : ScanDone->users())
      if (auto *Dbg = dyn_cast<DbgInfoIntrinsic>(U))
        DebugUsers.push_back(Dbg);
    for (DbgInfoIntrinsic *Dbg : DebugUsers)
      Dbg->replaceUsesOfWith(ScanDone, UndefValue::get(ScanDone->getType()));
    ScanDone->eraseFromParent();
    PixelNext->eraseFromParent();

    PHINode *CandidateIndex =
        PHINode::Create(IndexTy, 2, "response.candidate.index");
    BranchInst *CandidateBodyPlaceholder = BranchInst::Create(Body, CandidateBody);
    CandidateIndex->insertBefore(CandidateBodyPlaceholder->getIterator());
    CandidateIndex->addIncoming(ConstantInt::get(IndexTy, 0), Candidates);
    PHINode *CandidateCornerCount =
        PHINode::Create(CornerCount->getType(), 2,
                        "response.candidate.corner.count");
    CandidateCornerCount->insertBefore(CandidateBodyPlaceholder->getIterator());
    CandidateCornerCount->addIncoming(OuterCornerCount, Candidates);

    IRBuilder<> CandidateBodyBuilder(CandidateBodyPlaceholder);
    CandidateBodyBuilder.SetCurrentDebugLocation(ResponseLoad->getDebugLoc());
    Value *CandidateSlot = CandidateBodyBuilder.CreateGEP(
        I32Ty, CGX, CandidateIndex, "response.candidate.load.slot");
    auto *CandidatePixel32 = CandidateBodyBuilder.CreateLoad(
        I32Ty, CandidateSlot, "response.candidate.column");
    CandidatePixel32->setAlignment(Align(4));
    CandidatePixel32->setMetadata("riscv.ns.susan-response-compact",
                                  CompactMD);
    Value *CandidatePixel =
        CandidateBodyBuilder.CreateZExt(CandidatePixel32, IndexTy,
                                        "response.candidate.pixel");
    Value *CandidateOffset = CandidateBodyBuilder.CreateAdd(
        RowBase, CandidatePixel, "response.candidate.offset");
    Value *CandidateResponsePtr = CandidateBodyBuilder.CreateGEP(
        I32Ty, F.getArg(1), CandidateOffset, "response.candidate.address");
    auto *CandidateResponse = CandidateBodyBuilder.CreateLoad(
        I32Ty, CandidateResponsePtr, "response.candidate.value");
    CandidateResponse->setAlignment(ResponseLoad->getAlign());
    CandidateResponse->copyMetadata(*ResponseLoad);
    CandidateResponse->setMetadata("riscv.ns.susan-response-compact",
                                   CompactMD);

    for (BasicBlock *BB : BodyBlocks)
      for (Instruction &I : *BB) {
        I.replaceUsesOfWith(Pixel, CandidatePixel);
        I.replaceUsesOfWith(ResponseLoad, CandidateResponse);
        I.replaceUsesOfWith(PixelOffset, CandidateOffset);
        I.replaceUsesOfWith(CornerCount, CandidateCornerCount);
      }
    DebugUsers.clear();
    for (User *U : CornerCount->users())
      if (auto *Dbg = dyn_cast<DbgInfoIntrinsic>(U))
        DebugUsers.push_back(Dbg);
    for (DbgInfoIntrinsic *Dbg : DebugUsers) {
      Value *Replacement = BodyBlocks.contains(Dbg->getParent())
                               ? static_cast<Value *>(CandidateCornerCount)
                               : Dbg->getParent() == ScanLatch
                                     ? static_cast<Value *>(LatchCornerCount)
                                     : OuterCornerCount;
      Dbg->replaceUsesOfWith(CornerCount, Replacement);
    }
    for (int I = LatchCornerCount->getNumIncomingValues() - 1; I >= 0; --I)
      if (LatchCornerCount->getIncomingValue(I) == CornerCount)
        LatchCornerCount->setIncomingValue(I, CandidateCornerCount);
    if (!CornerCount->use_empty())
      report_fatal_error("SUSAN response compaction left a stale corner-count use");
    CornerCount->eraseFromParent();

    auto *RowExitBr = dyn_cast<BranchInst>(RowExit->getTerminator());
    if (!RowExitBr || !RowExitBr->isUnconditional() ||
        RowExitBr->getSuccessor(0) != RowLatch)
      report_fatal_error("SUSAN response compaction lost the row-exit edge");
    PHINode *RowNextMerge =
        PHINode::Create(Pixel->getType(), 2, "response.row.next");
    RowNextMerge->insertBefore(RowExitBr->getIterator());
    RowNextMerge->addIncoming(RowNext, ScanLatch);
    RowNextMerge->addIncoming(RowNext, Candidates);
    PHINode *RowCornerCountMerge = PHINode::Create(
        LatchCornerCount->getType(), 2, "response.row.corner.count");
    RowCornerCountMerge->insertBefore(RowExitBr->getIterator());
    RowCornerCountMerge->addIncoming(LatchCornerCount, ScanLatch);
    RowCornerCountMerge->addIncoming(OuterCornerCount, Candidates);
    RowNextPhi->setIncomingValue(RowNextPhi->getBasicBlockIndex(RowExit),
                                 RowNextMerge);
    RowCornerCountPhi->setIncomingValue(
        RowCornerCountPhi->getBasicBlockIndex(RowExit), RowCornerCountMerge);

    BranchInst *LatchPlaceholder = BranchInst::Create(CandidateBody, ScanLatch);
    IRBuilder<> LatchBuilder(LatchPlaceholder);
    LatchBuilder.SetCurrentDebugLocation(ResponseLoad->getDebugLoc());
    Value *CandidateIndexNext = LatchBuilder.CreateAdd(
        CandidateIndex, ConstantInt::get(IndexTy, 1),
        "response.candidate.index.next");
    Value *CandidatesDone = LatchBuilder.CreateICmpEQ(
        CandidateIndexNext, CandidateCountFinal,
        "response.candidates.done");
    LatchBuilder.CreateCondBr(CandidatesDone, RowExit, CandidateBody);
    LatchPlaceholder->eraseFromParent();
    CandidateIndex->addIncoming(CandidateIndexNext, ScanLatch);
    CandidateCornerCount->addIncoming(LatchCornerCount, ScanLatch);

    NumSusanResponseCompactions++;
    ORE.emit([&]() {
      return OptimizationRemark("riscv-susan-response-compaction",
                                "Compacted", ResponseLoad->getDebugLoc(), Scan)
             << "compacted response-map survivors before local-maximum checks";
    });
    assert(!verifyFunction(F, &errs()) &&
           "invalid SUSAN response-map compaction transformation");
    return true;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<OptimizationRemarkEmitterWrapperPass>();
  }
};

char RISCVSusanResponseMapCompaction::ID = 0;
char RISCVChainLoadHoisting::ID = 0;

class RISCVChainCheckThinning : public FunctionPass {
  struct ThresholdCheck {
    BranchInst *Branch;
    BasicBlock *Exit;
    BasicBlock *Continue;
    SmallVector<BasicBlock *, 4> Segment;
  };

  struct BranchToRemove {
    BranchInst *Branch;
    BasicBlock *Exit;
    BasicBlock *Continue;
  };

  static constexpr unsigned MaxDistance = 18;
  static constexpr unsigned MaxLiveValues = 16;

  bool findSegment(BasicBlock *Start, BasicBlock *End, Loop *L,
                   SmallVectorImpl<BasicBlock *> &Segment) const {
    SmallPtrSet<BasicBlock *, 8> Seen;
    BasicBlock *BB = Start;
    while (BB != End) {
      if (!L->contains(BB) || !Seen.insert(BB).second)
        return false;
      Segment.push_back(BB);
      auto *Br = dyn_cast<BranchInst>(BB->getTerminator());
      if (!Br || !Br->isUnconditional() || !L->contains(Br->getSuccessor(0)))
        return false;
      BB = Br->getSuccessor(0);
    }
    Segment.push_back(End);
    return true;
  }

  bool containsChainLoad(Value *V, SmallPtrSetImpl<Value *> &Visited) const {
    auto *I = dyn_cast<Instruction>(V);
    if (!I || !Visited.insert(V).second)
      return false;
    if (auto *Load = dyn_cast<LoadInst>(I))
      return Load->getMetadata("riscv.ns.chain.inbounds") != nullptr;
    if (auto *Phi = dyn_cast<PHINode>(I)) {
      if (!Phi->getMetadata("riscv.ns.chain.bridge"))
        return false;
      for (Value *Incoming : Phi->incoming_values())
        if (!isa<UndefValue>(Incoming) && !isa<PoisonValue>(Incoming) &&
            containsChainLoad(Incoming, Visited))
          return true;
      return false;
    }
    for (Value *Operand : I->operands())
      if (containsChainLoad(Operand, Visited))
        return true;
    return false;
  }

  Value *stripChainBridgeAndCasts(Value *V) const {
    while (true) {
      if (auto *Cast = dyn_cast<CastInst>(V)) {
        if (isa<SExtInst>(Cast) && Cast->getType()->isIntegerTy() &&
            Cast->getOperand(0)->getType()->isIntegerTy()) {
          V = Cast->getOperand(0);
          continue;
        }
      }
      auto *Phi = dyn_cast<PHINode>(V);
      if (!Phi || !Phi->getMetadata("riscv.ns.chain.bridge"))
        return V;
      Value *IncomingValue = nullptr;
      for (Value *Incoming : Phi->incoming_values()) {
        if (isa<UndefValue>(Incoming) || isa<PoisonValue>(Incoming))
          continue;
        if (IncomingValue)
          return V;
        IncomingValue = Incoming;
      }
      if (!IncomingValue)
        return V;
      V = IncomingValue;
    }
  }

  bool getQueueCondition(BranchInst *Br, Value *&Accumulator,
                         Value *&Threshold,
                         CmpInst::Predicate &Predicate) const {
    auto *Pop = dyn_cast<IntrinsicInst>(Br->getCondition());
    auto *Queue = Pop && Pop->getIntrinsicID() == Intrinsic::riscv_ns_cond_pop
                      ? dyn_cast<ConstantInt>(Pop->getArgOperand(0))
                      : nullptr;
    if (!Queue)
      return false;
    for (Instruction &I : instructions(*Br->getFunction())) {
      auto *Push = dyn_cast<IntrinsicInst>(&I);
      if (!Push || Push->getIntrinsicID() != Intrinsic::riscv_ns_cond_push ||
          Push->arg_size() < 4)
        continue;
      auto *PushQueue = dyn_cast<ConstantInt>(Push->getArgOperand(0));
      auto *PushPredicate = dyn_cast<ConstantInt>(Push->getArgOperand(1));
      if (!PushQueue || PushQueue->getValue() != Queue->getValue() ||
          !PushPredicate ||
          PushPredicate->getZExtValue() >
              static_cast<unsigned>(CmpInst::LAST_ICMP_PREDICATE))
        continue;
      SmallPtrSet<Value *, 32> LeftVisited;
      SmallPtrSet<Value *, 32> RightVisited;
      bool LeftHasChain =
          containsChainLoad(Push->getArgOperand(2), LeftVisited);
      bool RightHasChain =
          containsChainLoad(Push->getArgOperand(3), RightVisited);
      if (LeftHasChain == RightHasChain)
        continue;
      if (Accumulator)
        return false;
      Accumulator =
          stripChainBridgeAndCasts(Push->getArgOperand(LeftHasChain ? 2 : 3));
      Threshold =
          stripChainBridgeAndCasts(Push->getArgOperand(LeftHasChain ? 3 : 2));
      Predicate = static_cast<CmpInst::Predicate>(
          PushPredicate->getZExtValue());
    }
    return Accumulator && Threshold;
  }

  Value *getAccumulatedValue(BranchInst *Br) const {
    auto *Cmp = dyn_cast<ICmpInst>(Br->getCondition());
    if (!Cmp) {
      Value *Accumulator = nullptr;
      Value *Threshold = nullptr;
      CmpInst::Predicate Predicate = CmpInst::BAD_ICMP_PREDICATE;
      if (getQueueCondition(Br, Accumulator, Threshold, Predicate))
        return Accumulator;
      return nullptr;
    }
    SmallPtrSet<Value *, 32> Visited0;
    SmallPtrSet<Value *, 32> Visited1;
    bool HasChainLoad0 =
        containsChainLoad(Cmp->getOperand(0), Visited0);
    bool HasChainLoad1 =
        containsChainLoad(Cmp->getOperand(1), Visited1);
    if (HasChainLoad0 == HasChainLoad1)
      return nullptr;
    return stripChainBridgeAndCasts(Cmp->getOperand(HasChainLoad0 ? 0 : 1));
  }

  bool getThresholdAndPredicate(BranchInst *Br, Value *Accumulator,
                                Value *&Threshold,
                                CmpInst::Predicate &Predicate) const {
    if (auto *Cmp = dyn_cast<ICmpInst>(Br->getCondition())) {
      Predicate = Cmp->getPredicate();
      if (Cmp->hasSameSign()) {
        switch (Predicate) {
        case CmpInst::ICMP_ULT:
          Predicate = CmpInst::ICMP_SLT;
          break;
        case CmpInst::ICMP_ULE:
          Predicate = CmpInst::ICMP_SLE;
          break;
        case CmpInst::ICMP_UGT:
          Predicate = CmpInst::ICMP_SGT;
          break;
        case CmpInst::ICMP_UGE:
          Predicate = CmpInst::ICMP_SGE;
          break;
        default:
          break;
        }
      }
      Value *Operand0 = stripChainBridgeAndCasts(Cmp->getOperand(0));
      Value *Operand1 = stripChainBridgeAndCasts(Cmp->getOperand(1));
      if (Operand0 == Accumulator)
        Threshold = Operand1;
      else if (Operand1 == Accumulator)
        Threshold = Operand0;
      else
        return false;
      return true;
    }
    Value *QueueAccumulator = nullptr;
    return getQueueCondition(Br, QueueAccumulator, Threshold, Predicate) &&
           QueueAccumulator == Accumulator;
  }

  bool hasSameThreshold(ArrayRef<ThresholdCheck> Checks, unsigned First,
                        unsigned Last) const {
    Value *FirstAccumulator = getAccumulatedValue(Checks[First].Branch);
    Value *Threshold = nullptr;
    CmpInst::Predicate Predicate = CmpInst::BAD_ICMP_PREDICATE;
    if (!FirstAccumulator ||
        !getThresholdAndPredicate(Checks[First].Branch, FirstAccumulator,
                                  Threshold, Predicate))
      return false;
    bool ExitOnTrue =
        Checks[First].Branch->getSuccessor(0) == Checks[First].Exit;
    for (unsigned I = First + 1; I <= Last; ++I) {
      Value *Accumulator = getAccumulatedValue(Checks[I].Branch);
      Value *CandidateThreshold = nullptr;
      CmpInst::Predicate CandidatePredicate = CmpInst::BAD_ICMP_PREDICATE;
      if (!Accumulator ||
          !getThresholdAndPredicate(Checks[I].Branch, Accumulator,
                                    CandidateThreshold, CandidatePredicate) ||
          CandidatePredicate != Predicate ||
          CandidateThreshold != Threshold ||
          (Checks[I].Branch->getSuccessor(0) == Checks[I].Exit) != ExitOnTrue)
        return false;
    }
    return true;
  }

  bool hasMonotonicExitCondition(ArrayRef<ThresholdCheck> Checks,
                                 unsigned Index) const {
    Value *Accumulator = getAccumulatedValue(Checks[Index].Branch);
    Value *Threshold = nullptr;
    CmpInst::Predicate Predicate = CmpInst::BAD_ICMP_PREDICATE;
    if (!Accumulator ||
        !getThresholdAndPredicate(Checks[Index].Branch, Accumulator,
                                 Threshold, Predicate))
      return false;

    bool ExitOnTrue =
        Checks[Index].Branch->getSuccessor(0) == Checks[Index].Exit;
    switch (Predicate) {
    case CmpInst::ICMP_UGT:
    case CmpInst::ICMP_UGE:
    case CmpInst::ICMP_SGT:
    case CmpInst::ICMP_SGE:
      return ExitOnTrue;
    case CmpInst::ICMP_ULT:
    case CmpInst::ICMP_ULE:
    case CmpInst::ICMP_SLT:
    case CmpInst::ICMP_SLE:
      return !ExitOnTrue;
    default:
      return false;
    }
  }

  bool dependsOn(Value *V, Value *Target,
                 SmallPtrSetImpl<Value *> &Visited) const {
    if (V == Target)
      return true;
    auto *I = dyn_cast<Instruction>(V);
    if (!I || isa<LoadInst>(I) || !Visited.insert(V).second)
      return false;
    if (auto *Phi = dyn_cast<PHINode>(I)) {
      if (!Phi->getMetadata("riscv.ns.chain.bridge"))
        return false;
      for (Value *Incoming : Phi->incoming_values())
        if (!isa<UndefValue>(Incoming) && !isa<PoisonValue>(Incoming) &&
            dependsOn(Incoming, Target, Visited))
          return true;
      return false;
    }
    for (Value *Operand : I->operands())
      if (dependsOn(Operand, Target, Visited))
        return true;
    return false;
  }

  bool getNonnegativeUpperBound(Value *V, const DataLayout &DL,
                                Instruction *At, DominatorTree &DT,
                                APInt &Upper,
                                SmallPtrSetImpl<Value *> &Visited) const {
    auto *Ty = dyn_cast<IntegerType>(V->getType());
    if (!Ty || !Visited.insert(V).second)
      return false;
    unsigned Width = Ty->getBitWidth();
    if (auto *CI = dyn_cast<ConstantInt>(V)) {
      if (CI->isNegative())
        return false;
      Upper = CI->getValue();
      return true;
    }
    if (auto *ZExt = dyn_cast<ZExtInst>(V)) {
      auto *SrcTy = dyn_cast<IntegerType>(ZExt->getSrcTy());
      if (!SrcTy || SrcTy->getBitWidth() >= Width)
        return false;
      Upper = APInt::getLowBitsSet(Width, SrcTy->getBitWidth());
      return true;
    }
    if (auto *Phi = dyn_cast<PHINode>(V)) {
      if (!Phi->getMetadata("riscv.ns.chain.bridge"))
        return false;
      bool HasIncoming = false;
      for (Value *Incoming : Phi->incoming_values()) {
        if (isa<UndefValue>(Incoming) || isa<PoisonValue>(Incoming))
          continue;
        SmallPtrSet<Value *, 32> IncomingVisited;
        for (Value *Seen : Visited)
          IncomingVisited.insert(Seen);
        APInt IncomingUpper(Width, 0);
        if (!getNonnegativeUpperBound(Incoming, DL, At, DT, IncomingUpper,
                                      IncomingVisited))
          return false;
        if (!HasIncoming || IncomingUpper.ugt(Upper))
          Upper = IncomingUpper;
        HasIncoming = true;
      }
      return HasIncoming;
    }
    if (auto *BO = dyn_cast<BinaryOperator>(V)) {
      if (BO->getOpcode() == Instruction::And) {
        auto *Mask = dyn_cast<ConstantInt>(BO->getOperand(1));
        if (!Mask)
          Mask = dyn_cast<ConstantInt>(BO->getOperand(0));
        if (Mask && !Mask->isNegative()) {
          Upper = Mask->getValue().zextOrTrunc(Width);
          return true;
        }
      }
      if (BO->getOpcode() == Instruction::Add) {
        SmallPtrSet<Value *, 32> LeftVisited;
        SmallPtrSet<Value *, 32> RightVisited;
        for (Value *Seen : Visited) {
          LeftVisited.insert(Seen);
          RightVisited.insert(Seen);
        }
        APInt Left(Width, 0), Right(Width, 0);
        if (!getNonnegativeUpperBound(BO->getOperand(0), DL, At, DT, Left,
                                      LeftVisited) ||
            !getNonnegativeUpperBound(BO->getOperand(1), DL, At, DT, Right,
                                      RightVisited))
          return false;
        APInt Sum = Left + Right;
        if (Sum.ult(Left) || Sum.ugt(APInt::getMaxValue(Width)))
          return false;
        Upper = Sum;
        return true;
      }
      if (BO->getOpcode() == Instruction::Sub)
        return false;
    }
    if (auto *SI = dyn_cast<SelectInst>(V)) {
      SmallPtrSet<Value *, 32> TrueVisited;
      SmallPtrSet<Value *, 32> FalseVisited;
      for (Value *Seen : Visited) {
        TrueVisited.insert(Seen);
        FalseVisited.insert(Seen);
      }
      APInt TrueUpper(Width, 0), FalseUpper(Width, 0);
      if (!getNonnegativeUpperBound(SI->getTrueValue(), DL, At, DT,
                                    TrueUpper, TrueVisited) ||
          !getNonnegativeUpperBound(SI->getFalseValue(), DL, At, DT,
                                    FalseUpper, FalseVisited))
        return false;
      Upper = TrueUpper.ugt(FalseUpper) ? TrueUpper : FalseUpper;
      return true;
    }

    ConstantRange Range =
        computeConstantRange(V, /*ForSigned=*/true, /*UseInstrInfo=*/true,
                            nullptr, At, &DT);
    if (Range.isFullSet() || Range.isEmptySet() || Range.isWrappedSet() ||
        Range.getSignedMin().isNegative())
      return false;
    Upper = Range.getSignedMax().zextOrTrunc(Width);
    return true;
  }

  bool isSafeAccumulation(Value *Previous, Value *Next,
                          Instruction *At, DominatorTree &DT,
                          StringRef &Failure) const {
    Previous = stripChainBridgeAndCasts(Previous);
    Next = stripChainBridgeAndCasts(Next);
    const DataLayout &DL = At->getModule()->getDataLayout();
    auto *Ty = dyn_cast<IntegerType>(Next->getType());
    if (!Ty || Previous->getType() != Next->getType()) {
      Failure = "accumulator type changed";
      return false;
    }
    SmallPtrSet<Value *, 32> BoundVisited;
    APInt AccumulatorUpper(Ty->getBitWidth(), 0);
    if (!getNonnegativeUpperBound(Previous, DL, At, DT, AccumulatorUpper,
                                  BoundVisited)) {
      Failure = "accumulator range is not proven nonnegative";
      return false;
    }

    SmallPtrSet<Value *, 32> Visited;
    Value *Current = Next;
    SmallVector<Value *, 16> Contributions;
    while (Current != Previous) {
      auto *Add = dyn_cast<BinaryOperator>(Current);
      if (!Add || Add->getOpcode() != Instruction::Add) {
        Failure = "intermediate value is not a plain accumulation";
        return false;
      }
      SmallPtrSet<Value *, 32> LeftVisited;
      SmallPtrSet<Value *, 32> RightVisited;
      bool LeftDepends =
          dependsOn(Add->getOperand(0), Previous, LeftVisited);
      bool RightDepends =
          dependsOn(Add->getOperand(1), Previous, RightVisited);
      if (LeftDepends == RightDepends) {
        Failure = "addition does not have one unique prior accumulator";
        return false;
      }
      Value *Prior =
          stripChainBridgeAndCasts(Add->getOperand(LeftDepends ? 0 : 1));
      Value *Contribution = Add->getOperand(LeftDepends ? 1 : 0);
      if (!Visited.insert(Add).second) {
        Failure = "accumulation contains a cycle";
        return false;
      }
      Contributions.push_back(Contribution);
      Current = Prior;
    }
    if (Contributions.empty()) {
      Failure = "no contributions were found between checks";
      return false;
    }
    std::reverse(Contributions.begin(), Contributions.end());
    for (Value *Contribution : Contributions) {
      SmallPtrSet<Value *, 32> ContributionVisited;
      APInt ContributionUpper(Ty->getBitWidth(), 0);
      if (!getNonnegativeUpperBound(Contribution, DL, At,
                                    DT,
                                    ContributionUpper, ContributionVisited)) {
        Failure = "a contribution is not proven nonnegative";
        return false;
      }
      APInt Sum = AccumulatorUpper + ContributionUpper;
      if (Sum.ult(AccumulatorUpper) ||
          Sum.ugt(APInt::getMaxValue(Ty->getBitWidth()))) {
        Failure = "accumulator may overflow";
        return false;
      }
      AccumulatorUpper = Sum;
    }
    return true;
  }

  bool hasUnsafeMemoryOperation(ArrayRef<ThresholdCheck> Checks,
                                unsigned First, unsigned Last) const {
    for (unsigned I = First; I < Last; ++I)
      for (BasicBlock *BB : Checks[I].Segment)
        for (Instruction &Inst : *BB) {
          if (isa<CallBase>(Inst) || isa<FenceInst>(Inst))
            return true;
          if (auto *Load = dyn_cast<LoadInst>(&Inst)) {
            if (!Load->isSimple())
              return true;
          } else if (auto *Store = dyn_cast<StoreInst>(&Inst)) {
            if (!Store->isSimple())
              return true;
            return true;
          } else if (Inst.mayWriteToMemory()) {
            return true;
          }
        }
    return false;
  }

  bool hasExitLiveOut(ArrayRef<ThresholdCheck> Checks, unsigned First,
                      unsigned Last, Loop *L) const {
    SmallPtrSet<BasicBlock *, 32> ExitPath;
    SmallPtrSet<BasicBlock *, 32> ChainBlocks;
    for (const ThresholdCheck &Check : Checks)
      ChainBlocks.insert(Check.Branch->getParent());
    SmallVector<BasicBlock *, 16> Worklist;
    Worklist.push_back(Checks[First].Exit);
    while (!Worklist.empty()) {
      BasicBlock *BB = Worklist.pop_back_val();
      if (!ExitPath.insert(BB).second || ChainBlocks.contains(BB) ||
          BB == L->getHeader())
        continue;
      if (!L->contains(BB))
        continue;
      for (BasicBlock *Succ : successors(BB))
        if (Succ != L->getHeader() && !ChainBlocks.contains(Succ))
          Worklist.push_back(Succ);
    }

    SmallPtrSet<Instruction *, 32> IntermediateDefs;
    for (unsigned I = First + 1; I < Last; ++I)
      for (BasicBlock *BB : Checks[I].Segment)
        for (Instruction &Inst : *BB)
          IntermediateDefs.insert(&Inst);
    for (Instruction *Def : IntermediateDefs)
      for (User *U : Def->users()) {
        auto *UseInst = dyn_cast<Instruction>(U);
        if (UseInst && ExitPath.contains(UseInst->getParent()))
          return true;
      }
    return false;
  }

  bool pressureIsAcceptable(ArrayRef<ThresholdCheck> Checks, unsigned First,
                            unsigned Last) const {
    SmallPtrSet<Instruction *, 32> Group;
    SmallVector<Instruction *, 64> Ordered;
    for (unsigned I = First + 1; I <= Last; ++I)
      for (BasicBlock *BB : Checks[I].Segment)
        for (Instruction &Inst : *BB) {
          if (Inst.isTerminator())
            continue;
          Group.insert(&Inst);
          Ordered.push_back(&Inst);
        }
    DenseMap<Value *, unsigned> Uses;
    for (Instruction *Inst : Ordered)
      for (Value *Operand : Inst->operands())
        if (Group.contains(dyn_cast<Instruction>(Operand)))
          ++Uses[Operand];
    for (unsigned I = First + 1; I <= Last; ++I) {
    Value *Condition = Checks[I].Branch->getCondition();
    if (Group.contains(dyn_cast<Instruction>(Condition)))
      ++Uses[Condition];
    }

    unsigned Live = 0;
    unsigned Peak = 0;
    for (Instruction *Inst : Ordered) {
      for (Value *Operand : Inst->operands()) {
        auto It = Uses.find(Operand);
        if (It != Uses.end() && It->second && --It->second == 0)
          --Live;
      }
      if (!Inst->getType()->isVoidTy() && Uses.lookup(Inst)) {
        ++Live;
        Peak = std::max(Peak, Live);
      }
    }
    return Peak <= MaxLiveValues;
  }

  bool isKeptCheck(unsigned Index, unsigned Count) const {
    if (Index == 0 || Index + 1 == Count)
      return true;
    if (ThinChainChecksSchedule == "histogram") {
      if (Index <= 12)
        return Index % 2 == 0;
      return (Index - 12) % 3 == 0;
    }
    return Index % ThinChainChecks == 0;
  }

  void reject(Function &F, BranchInst *Br, StringRef Reason,
              OptimizationRemarkEmitter &ORE, Statistic &Counter) const {
    ++Counter;
    DEBUG_WITH_TYPE("riscv-chain-check-thinning",
                    dbgs() << "keeping threshold check in " << F.getName()
                           << " at " << Br->getParent()->getName() << ": "
                           << Reason << '\n');
    ORE.emit([&]() {
      return OptimizationRemarkMissed(
                 "riscv-chain-check-thinning", "Rejected", Br->getDebugLoc(),
                 Br->getParent())
             << "kept a threshold check in " << F.getName() << ": " << Reason;
    });
  }

  bool thinChain(ArrayRef<ThresholdCheck> Checks, Loop *L, Function &F,
                 DominatorTree &DT, ScalarEvolution &SE,
                 OptimizationRemarkEmitter &ORE,
                 RISCVChainLoadHoisting &ContractValidator,
                 SmallVectorImpl<BranchToRemove> &BranchesToRemove) {
    if (Checks.size() < 3) {
      ++NumChainCheckThinRejectShortChain;
      ORE.emit([&]() {
        return OptimizationRemarkMissed(
                   "riscv-chain-check-thinning", "ShortChain",
                   Checks.front().Branch->getDebugLoc(),
                   Checks.front().Branch->getParent())
               << "kept threshold checks because the chain is too short to "
                  "thin";
      });
      return false;
    }
    if (ThinChainChecks > MaxDistance) {
      report_fatal_error("early-exit check spacing must be in [1, 18]");
    }
    if (ThinChainChecksSchedule != "uniform" &&
        ThinChainChecksSchedule != "histogram")
      report_fatal_error("early-exit check schedule must be uniform or "
                         "histogram");

    SmallVector<unsigned, 16> Keep;
    for (unsigned I = 0; I < Checks.size(); ++I)
      if (isKeptCheck(I, Checks.size()))
        Keep.push_back(I);
    if (Keep.size() == Checks.size())
      return false;

    SmallVector<BranchToRemove, 16> Remove;
    for (unsigned Group = 1; Group < Keep.size(); ++Group) {
      unsigned First = Keep[Group - 1];
      unsigned Last = Keep[Group];
      if (Last == First + 1)
        continue;

      bool Valid = true;
      StringRef Failure;
      Value *Previous = getAccumulatedValue(Checks[First].Branch);
      Value *Next = getAccumulatedValue(Checks[Last].Branch);
      if (!Previous || !Next ||
          !hasSameThreshold(Checks, First, Last)) {
        Valid = false;
        Failure = "threshold branch does not compare a unique chain accumulator";
        reject(F, Checks[First].Branch, Failure, ORE,
               NumChainCheckThinRejectControl);
      }
      if (Valid && !hasMonotonicExitCondition(Checks, First)) {
        Valid = false;
        Failure = "the exit condition is not monotone for a nonnegative "
                  "accumulator";
        reject(F, Checks[First].Branch, Failure, ORE,
               NumChainCheckThinRejectMonotonic);
      }
      if (Valid && !isSafeAccumulation(Previous, Next, Checks[Last].Branch, DT,
                                      Failure)) {
        Valid = false;
        reject(F, Checks[First].Branch, Failure, ORE,
               Failure == "accumulator may overflow"
                   ? NumChainCheckThinRejectOverflow
                   : NumChainCheckThinRejectContribution);
      }
      if (Valid && hasUnsafeMemoryOperation(Checks, First + 1, Last)) {
        Valid = false;
        reject(F, Checks[First].Branch,
               "a store, call, fence, volatile, or atomic access lies between "
               "checks",
               ORE, NumChainCheckThinRejectMemory);
      }
      if (Valid) {
        for (unsigned I = First + 1; I < Last && Valid; ++I)
          for (BasicBlock *BB : Checks[I].Segment)
            for (Instruction &Inst : *BB)
              if (auto *Load = dyn_cast<LoadInst>(&Inst))
                if (!ContractValidator.isLoadSafeForThinning(
                        Load, L, Checks[First].Branch, DT, SE)) {
                  Valid = false;
                  reject(F, Checks[First].Branch,
                         "a load is not safe on the newly reachable exit path",
                         ORE, NumChainCheckThinRejectInBounds);
                  break;
                }
      }
      if (Valid && hasExitLiveOut(Checks, First, Last, L)) {
        Valid = false;
        reject(F, Checks[First].Branch,
               "an intermediate value is live on the early-exit path", ORE,
               NumChainCheckThinRejectLiveOut);
      }
      if (Valid && !pressureIsAcceptable(Checks, First, Last)) {
        Valid = false;
        reject(F, Checks[First].Branch,
               "the in-flight values exceed the live-value pressure cap", ORE,
               NumChainCheckThinRejectPressure);
      }
      if (Valid)
        for (unsigned I = First + 1; I < Last; ++I)
          Remove.push_back({Checks[I].Branch, Checks[I].Exit,
                            Checks[I].Continue});
    }

    if (Remove.empty())
      return false;
    BranchesToRemove.append(Remove.begin(), Remove.end());

    ORE.emit([&]() {
      return OptimizationRemark("riscv-chain-check-thinning", "Thinned",
                                Checks.front().Branch->getDebugLoc(),
                                Checks.front().Branch->getParent())
             << "will remove " << ore::NV("Checks", Remove.size())
             << " early-exit threshold check(s) at spacing "
             << ore::NV("Spacing", ThinChainChecks);
    });
    return true;
  }

public:
  static char ID;
  RISCVChainCheckThinning() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override {
    if (!ThinChainChecks || skipFunction(F))
      return false;
    if (ThinChainChecks > MaxDistance)
      report_fatal_error("early-exit check spacing must be in [1, 18]");
    if (ThinChainChecksSchedule != "uniform" &&
        ThinChainChecksSchedule != "histogram")
      report_fatal_error("early-exit check schedule must be uniform or "
                         "histogram");

    auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    auto &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    auto &SE = getAnalysis<ScalarEvolutionWrapperPass>().getSE();
    auto &ORE = getAnalysis<OptimizationRemarkEmitterWrapperPass>().getORE();
    RISCVChainLoadHoisting ContractValidator;
    SmallVector<BranchToRemove, 32> BranchesToRemove;
    bool Changed = false;

    for (Loop *L : LI.getLoopsInPreorder()) {
      DenseMap<BasicBlock *, SmallVector<BranchInst *, 4>> ExitCandidates;
      for (BasicBlock *BB : L->blocks()) {
        if (LI.getLoopFor(BB) != L)
          continue;
        auto *Br = dyn_cast<BranchInst>(BB->getTerminator());
        if (!Br || !Br->isConditional())
          continue;
        if (L->contains(Br->getSuccessor(0)) !=
            L->contains(Br->getSuccessor(1))) {
          ++NumChainCheckThinRejectControl;
          DEBUG_WITH_TYPE("riscv-chain-check-thinning",
                          dbgs() << "rejecting loop-exit/guard branch in "
                                 << F.getName() << " at " << BB->getName()
                                 << '\n');
          ORE.emit([&]() {
            return OptimizationRemarkMissed(
                       "riscv-chain-check-thinning", "LoopExitOrGuard",
                       Br->getDebugLoc(), BB)
                   << "check thinning does not cross a loop-exit or "
                      "loop-guard branch";
          });
          continue;
        }
        for (unsigned I = 0; I != 2; ++I)
          if (L->contains(Br->getSuccessor(I)))
            ExitCandidates[Br->getSuccessor(I)].push_back(Br);
      }
      DenseMap<BranchInst *, ThresholdCheck> Candidates;
      for (auto &Entry : ExitCandidates) {
        if (Entry.second.size() < 2)
          continue;
        BasicBlock *Exit = Entry.first;
        for (BranchInst *Br : Entry.second) {
          if (Br->getSuccessor(0) == Exit &&
              Br->getSuccessor(1) == Exit)
            continue;
          if (!getAccumulatedValue(Br)) {
            DEBUG_WITH_TYPE("riscv-chain-check-thinning",
                            dbgs() << "skipping non-chain branch "
                                   << Br->getParent()->getName() << " in "
                                   << F.getName() << '\n');
            continue;
          }
          BasicBlock *Continue = Br->getSuccessor(0) == Exit
                                     ? Br->getSuccessor(1)
                                     : Br->getSuccessor(0);
          if (!L->contains(Continue) || Continue == Exit)
            continue;
          if (Candidates.count(Br)) {
            Candidates.erase(Br);
            ++NumChainCheckThinRejectControl;
            continue;
          }
          Candidates[Br] = {Br, Exit, Continue, {}};
        }
      }

      DenseMap<BasicBlock *, BranchInst *> CandidateAt;
      for (auto &Entry : Candidates)
        CandidateAt[Entry.first->getParent()] = Entry.first;
      DenseMap<BranchInst *, BranchInst *> NextCheck;
      for (auto &Entry : Candidates) {
        BranchInst *Br = Entry.first;
        BasicBlock *BB = Entry.second.Continue;
        SmallPtrSet<BasicBlock *, 8> Seen;
        while (L->contains(BB) && Seen.insert(BB).second) {
          auto It = CandidateAt.find(BB);
          if (It != CandidateAt.end()) {
            if (Candidates[It->second].Exit == Entry.second.Exit)
              NextCheck[Br] = It->second;
            break;
          }
          auto *Step = dyn_cast<BranchInst>(BB->getTerminator());
          if (!Step || !Step->isUnconditional())
            break;
          BB = Step->getSuccessor(0);
        }
      }

      SmallPtrSet<BranchInst *, 16> HasPrevious;
      for (auto &Entry : NextCheck)
        HasPrevious.insert(Entry.second);
      SmallPtrSet<BranchInst *, 16> Visited;
      for (auto &Entry : Candidates) {
        BranchInst *Start = Entry.first;
        if (HasPrevious.contains(Start) || Visited.contains(Start))
          continue;
        SmallVector<ThresholdCheck, 20> Checks;
        BranchInst *Current = Start;
        while (Current && Visited.insert(Current).second) {
          Checks.push_back(Candidates[Current]);
          auto It = NextCheck.find(Current);
          Current = It == NextCheck.end() ? nullptr : It->second;
        }
        if (Checks.size() < 2)
          continue;
        bool Valid = true;
        for (unsigned I = 1; I < Checks.size(); ++I)
          if (!findSegment(Checks[I - 1].Continue,
                           Checks[I].Branch->getParent(), L,
                           Checks[I].Segment)) {
            Valid = false;
            ++NumChainCheckThinRejectControl;
            DEBUG_WITH_TYPE(
                "riscv-chain-check-thinning",
                dbgs() << "rejecting chain in " << F.getName()
                       << ": a loop guard or non-linear branch lies between "
                          "threshold checks\n");
            break;
          }
        if (Valid)
          Changed |= thinChain(Checks, L, F, DT, SE, ORE,
                               ContractValidator, BranchesToRemove);
      }
    }
    SmallPtrSet<BranchInst *, 32> Removed;
    for (const BranchToRemove &Candidate : BranchesToRemove) {
      BranchInst *Br = Candidate.Branch;
      if (!Removed.insert(Br).second)
        continue;
      Candidate.Exit->removePredecessor(Br->getParent());
      BranchInst::Create(Candidate.Continue, Br->getIterator());
      Br->eraseFromParent();
      ++NumChainChecksRemoved;
    }
    return Changed;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<ScalarEvolutionWrapperPass>();
    AU.addRequired<OptimizationRemarkEmitterWrapperPass>();
  }
};

char RISCVChainCheckThinning::ID = 0;

} // namespace

INITIALIZE_PASS_BEGIN(RISCVChainLoadHoisting, "riscv-chain-load-hoisting",
                      "RISC-V early-exit chain load hoisting", false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(BranchProbabilityInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(OptimizationRemarkEmitterWrapperPass)
INITIALIZE_PASS_END(RISCVChainLoadHoisting, "riscv-chain-load-hoisting",
                    "RISC-V early-exit chain load hoisting", false, false)

FunctionPass *llvm::createRISCVChainLoadHoistingPass() {
  return new RISCVChainLoadHoisting();
}

INITIALIZE_PASS_BEGIN(RISCVChainCheckThinning, "riscv-chain-check-thinning",
                      "RISC-V early-exit chain check thinning", false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(OptimizationRemarkEmitterWrapperPass)
INITIALIZE_PASS_END(RISCVChainCheckThinning, "riscv-chain-check-thinning",
                    "RISC-V early-exit chain check thinning", false, false)

FunctionPass *llvm::createRISCVChainCheckThinningPass() {
  return new RISCVChainCheckThinning();
}

INITIALIZE_PASS_BEGIN(RISCVSusanResponseMapCompaction,
                      "riscv-susan-response-compaction",
                      "RISC-V SUSAN response-map compaction", false, false)
INITIALIZE_PASS_DEPENDENCY(OptimizationRemarkEmitterWrapperPass)
INITIALIZE_PASS_END(RISCVSusanResponseMapCompaction,
                    "riscv-susan-response-compaction",
                    "RISC-V SUSAN response-map compaction", false, false)

FunctionPass *llvm::createRISCVSusanResponseMapCompactionPass() {
  return new RISCVSusanResponseMapCompaction();
}
