//===-- RISCVLoopDecideAct.cpp - Split guarded loops: decide, then act ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Non-Spec: a pb waits at dispatch until its condition is computed, so a
// branch on a value loaded just before it stalls every iteration. Condition
// pipelining and loop fission compute such conditions early, but give up on a
// loop whose guarded body has a call or another loop in it. This pass splits
// such a loop
//
//   for (i...) { d = decide(i); if (d) act(i); }
//
// into a decide loop and an act loop:
//
//   n = 0;
//   for (i...) { list[n] = i; n += decide(i); }     // no branch but the latch
//   for (m = 0; m < n; ++m) { i = list[m]; act(i); } // the original loop
//
// The decide loop computes every condition without a branch (an if-converted
// copy of the decide part); the original loop, kept in place, becomes the act
// loop over the chosen iterations, with the branches that led only to the
// latch removed. It is legal when the decide part has no side effects and is
// safe to run for every iteration, and no act of an earlier iteration writes
// memory the decide part of a later one reads (writes on paths that end in a
// call that does not return do not count: the program ends there).
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVSubtarget.h"
#include "RISCVTargetMachine.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/Loads.h"
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
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Verifier.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-decide-act"

STATISTIC(NumDecideActLoops,
          "Number of loops split into a decide and an act loop");

static cl::opt<bool>
    DisableDecideAct("riscv-ns-disable-decide-act", cl::Hidden, cl::init(false),
                     cl::desc("Do not split guarded loops into a decide and an "
                              "act loop"));

static cl::opt<unsigned> MaxDecideActTrip(
    "riscv-ns-decide-act-max-trip", cl::Hidden, cl::init(1024),
    cl::desc("Largest bound on the trip count of a loop split into a decide "
             "and an act loop (the size of its list of chosen iterations)"));

namespace {

struct Candidate {
  Loop *L = nullptr;
  BasicBlock *Preheader = nullptr, *Header = nullptr, *Latch = nullptr;
  BasicBlock *Exit = nullptr;     // the latch's exit
  BasicBlock *ActEntry = nullptr; // the first block of the guarded part
  PHINode *IV = nullptr;
  Instruction *IVNext = nullptr;
  ICmpInst *ExitCmp = nullptr;
  BranchInst *LatchBr = nullptr;
  SmallVector<BasicBlock *, 8> Decide; // the decide part, in reverse post-order
  SmallVector<PHINode *, 4> Carried;   // header phis other than the IV
  SmallPtrSet<BasicBlock *, 4> NoReturnExits;
  unsigned MaxTrip = 0;
};

class RISCVLoopDecideAct : public FunctionPass {
public:
  static char ID;
  RISCVLoopDecideAct() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override;
  StringRef getPassName() const override {
    return "RISC-V decide/act loop splitting";
  }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetPassConfig>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<ScalarEvolutionWrapperPass>();
    AU.addRequired<AAResultsWrapperPass>();
    AU.addRequired<AssumptionCacheTracker>();
  }

private:
  bool analyze(Loop *L, Candidate &C);
  bool memoryIsIndependent(const Candidate &C);
  bool storeMissesLaterLoads(const Candidate &C, StoreInst *S, LoadInst *Ld);
  void transform(Candidate &C);

  std::string Why;
  bool reject(const Twine &W) {
    Why = W.str();
    return false;
  }

  DominatorTree *DT = nullptr;
  LoopInfo *LI = nullptr;
  ScalarEvolution *SE = nullptr;
  AAResults *AA = nullptr;
  AssumptionCache *AC = nullptr;
};

} // end anonymous namespace

char RISCVLoopDecideAct::ID = 0;

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

/// Whether \p I changes memory or has another effect the decide loop must not
/// run early. Debug and assume-like intrinsics do not.
static bool hasEffects(const Instruction &I) {
  if (auto *II = dyn_cast<IntrinsicInst>(&I))
    if (isa<DbgInfoIntrinsic>(II) || II->isAssumeLikeIntrinsic() ||
        II->isLifetimeStartOrEnd())
      return false;
  return I.mayWriteToMemory() || I.mayHaveSideEffects() ||
         (isa<CallBase>(I) && !isa<IntrinsicInst>(I));
}

/// Whether every path from \p BB ends in `unreachable` without leaving it: a
/// block that calls exit() or abort() and stops.
static bool isNoReturnExit(const BasicBlock *BB) {
  return isa<UnreachableInst>(BB->getTerminator());
}

bool RISCVLoopDecideAct::analyze(Loop *L, Candidate &C) {
  C.L = L;
  C.Preheader = L->getLoopPreheader();
  C.Header = L->getHeader();
  C.Latch = L->getLoopLatch();
  if (!C.Preheader || !C.Latch || C.Latch == C.Header)
    return reject("the loop has no preheader, no single latch, or is a "
                  "single block");
  auto *PreBr = dyn_cast<BranchInst>(C.Preheader->getTerminator());
  if (!PreBr || !PreBr->isUnconditional())
    return reject("the preheader does not end in a jump to the loop");
  C.LatchBr = dyn_cast<BranchInst>(C.Latch->getTerminator());
  if (!C.LatchBr || !C.LatchBr->isConditional())
    return reject("the latch does not end in a conditional branch");
  for (BasicBlock *Succ : successors(C.Latch))
    if (!L->contains(Succ))
      C.Exit = Succ;
  if (!C.Exit)
    return reject("the latch does not exit the loop");

  // Every other exit must stop the program (exit(), abort()): the act loop
  // still reaches it, and the decide loop's early loads cannot be observed.
  SmallVector<std::pair<BasicBlock *, BasicBlock *>, 4> ExitEdges;
  L->getExitEdges(ExitEdges);
  for (auto [From, To] : ExitEdges) {
    if (From == C.Latch && To == C.Exit)
      continue;
    if (!isNoReturnExit(To))
      return reject("the loop can exit other than at its latch, to a block "
                    "that does not end the program");
    C.NoReturnExits.insert(To);
  }

  // The induction variable: a header phi stepped by a constant in the latch,
  // whose next value the latch compares to leave the loop.
  auto *Cmp = dyn_cast<ICmpInst>(C.LatchBr->getCondition());
  if (!Cmp || !Cmp->hasOneUse() || Cmp->getParent() != C.Latch)
    return reject("the latch's exit condition is not a compare used only "
                  "there");
  for (PHINode &Phi : C.Header->phis()) {
    if (Phi.getNumIncomingValues() != 2)
      return reject("a header phi does not have exactly two incoming values");
    auto *Next =
        dyn_cast<BinaryOperator>(Phi.getIncomingValueForBlock(C.Latch));
    if (!C.IV && Next && Next->getOpcode() == Instruction::Add &&
        Next->getOperand(0) == &Phi && isa<ConstantInt>(Next->getOperand(1)) &&
        (Cmp->getOperand(0) == Next || Cmp->getOperand(0) == &Phi) &&
        L->isLoopInvariant(Cmp->getOperand(1))) {
      C.IV = &Phi;
      C.IVNext = Next;
      continue;
    }
    C.Carried.push_back(&Phi);
  }
  if (!C.IV || !C.IV->getType()->isIntegerTy())
    return reject("no integer induction variable compared in the latch");
  C.ExitCmp = Cmp;
  C.MaxTrip = SE->getSmallConstantMaxTripCount(L);
  if (!C.MaxTrip)
    return reject("the trip count has no known bound");
  if (C.MaxTrip > MaxDecideActTrip)
    return reject("the trip count can reach " + Twine(C.MaxTrip) +
                  ", more than -riscv-ns-decide-act-max-trip (" +
                  Twine(MaxDecideActTrip) + ")");

  // The latch only counts and branches; its phis merge the carried values.
  for (Instruction &I : *C.Latch) {
    if (isa<PHINode>(I) || &I == C.IVNext || &I == C.ExitCmp ||
        &I == C.LatchBr || isa<DbgInfoIntrinsic>(I))
      continue;
    if (I.mayReadOrWriteMemory() || hasEffects(I))
      return reject("the latch does more than count: " + describe(I));
  }

  // The decide part: the blocks from the header on that have no effects and
  // are in no inner loop. The first block that does, or is, starts the act.
  LoopBlocksRPO RPO(L);
  RPO.perform(LI);
  SmallPtrSet<BasicBlock *, 8> InDecide;
  SmallSetVector<BasicBlock *, 2> Frontier;
  for (BasicBlock *BB : RPO) {
    if (BB == C.Latch)
      continue;
    bool Reached = BB == C.Header ||
                   any_of(predecessors(BB),
                          [&](BasicBlock *P) { return InDecide.contains(P); });
    if (!Reached)
      continue;
    // A block that can leave the loop (to a call that never returns) belongs
    // to the guarded part, even without effects of its own.
    bool Pure =
        LI->getLoopFor(BB) == L &&
        none_of(*BB, [](Instruction &I) { return hasEffects(I); }) &&
        all_of(successors(BB), [&](BasicBlock *S) { return L->contains(S); });
    if (Pure && (BB == C.Header || all_of(predecessors(BB), [&](BasicBlock *P) {
                   return InDecide.contains(P);
                 }))) {
      InDecide.insert(BB);
      C.Decide.push_back(BB);
    } else {
      Frontier.insert(BB);
    }
  }
  if (Frontier.size() != 1)
    return reject("the conditions lead to " + Twine(Frontier.size()) +
                  " guarded blocks; one is handled");
  C.ActEntry = Frontier[0];
  if (!all_of(predecessors(C.ActEntry),
              [&](BasicBlock *P) { return InDecide.contains(P); }) ||
      !C.ActEntry->phis().empty())
    return reject("the guarded part is also entered from outside the "
                  "conditions, or starts with a phi");

  // Shape: decide blocks branch within the decide part, to the guarded part
  // or to the latch; the guarded part is everything else and is only entered
  // through its first block.
  for (BasicBlock *BB : C.Decide) {
    auto *Br = dyn_cast<BranchInst>(BB->getTerminator());
    if (!Br)
      return reject("a condition block ends in a switch or another "
                    "terminator");
    for (BasicBlock *S : successors(BB))
      if (!InDecide.contains(S) && S != C.ActEntry && S != C.Latch)
        return reject("a condition block branches outside the loop");
    if (BB != C.Header && !BB->phis().empty())
      return reject("a condition block merges values (a phi)");
  }
  bool ActHasCallOrLoop = false;
  for (BasicBlock *BB : L->blocks()) {
    if (InDecide.contains(BB) || BB == C.Latch)
      continue;
    if (!DT->dominates(C.ActEntry, BB))
      return reject("a block of the guarded part is not reached through its "
                    "first block");
    if (LI->getLoopFor(BB) != L)
      ActHasCallOrLoop = true;
    for (Instruction &I : *BB)
      if (isa<CallBase>(I) && !isa<IntrinsicInst>(I))
        ActHasCallOrLoop = true;
  }
  // Only worth it where pipelining and fission give up: the guarded part has
  // a call or a loop. And only if a condition waits on a load.
  if (!ActHasCallOrLoop)
    return reject("the guarded part has no call or inner loop; condition "
                  "pipelining and fission handle such loops");
  bool CondNeedsLoad = false;
  for (BasicBlock *BB : C.Decide)
    for (Instruction &I : *BB)
      if (isa<LoadInst>(I))
        CondNeedsLoad = true;
  if (!CondNeedsLoad)
    return reject("no condition reads memory: nothing waits on a load");

  // The decide loop runs the decide part for every iteration, first.
  for (BasicBlock *BB : C.Decide)
    for (Instruction &I : *BB) {
      if (isa<PHINode>(I) || I.isTerminator() || isa<DbgInfoIntrinsic>(I))
        continue;
      for (Value *Op : I.operands())
        if (auto *Phi = dyn_cast<PHINode>(Op))
          if (is_contained(C.Carried, Phi))
            return reject("a condition depends on a value carried from the "
                          "previous iteration (" +
                          Phi->getName() + ")");
      if (auto *Ld = dyn_cast<LoadInst>(&I)) {
        if (!Ld->isSimple())
          return reject("a condition reads memory with a volatile or atomic "
                        "load");
        if (BB != C.Header &&
            !isDereferenceableAndAlignedInLoop(Ld, L, *SE, *DT, AC))
          return reject("a condition needs " + describe(I) +
                        ", which is not provably safe to run in every "
                        "iteration");
        continue;
      }
      if (!isSafeToSpeculativelyExecute(&I))
        return reject("a condition needs " + describe(I) +
                      ", which is not safe to run in every iteration");
    }

  // A carried value must come back through a latch phi (or not change): a
  // value computed in the latch itself would be computed only for the chosen
  // iterations in the act loop.
  for (PHINode *H : C.Carried) {
    Value *Back = H->getIncomingValueForBlock(C.Latch);
    auto *BackPhi = dyn_cast<PHINode>(Back);
    if (Back != H && !(BackPhi && BackPhi->getParent() == C.Latch))
      return reject("a value carried to the next iteration is computed in the "
                    "latch (" +
                    H->getName() + ")");
  }

  // A latch phi stands for a carried value: on the paths that skip the
  // guarded part it must keep the value the iteration started with.
  for (PHINode &P : C.Latch->phis()) {
    if (&P == C.IV)
      continue;
    PHINode *Head = nullptr;
    for (PHINode *H : C.Carried)
      if (H->getIncomingValueForBlock(C.Latch) == &P)
        Head = H;
    for (unsigned I = 0; I < P.getNumIncomingValues(); ++I)
      if (InDecide.contains(P.getIncomingBlock(I)) &&
          P.getIncomingValue(I) != Head)
        return reject("a value merged in the latch changes on a path that "
                      "skips the guarded part");
  }

  // What is used after the loop must be one of: a value of the decide part or
  // the counter (the decide loop's last iteration has it), or a carried value
  // as the latch leaves it. Uses on the paths that end the program are left
  // alone: the act loop reaches them with the right values.
  for (BasicBlock *BB : L->blocks())
    for (Instruction &I : *BB)
      for (User *U : I.users()) {
        auto *UI = dyn_cast<Instruction>(U);
        if (!UI || L->contains(UI) || C.NoReturnExits.contains(UI->getParent()))
          continue;
        bool DecideValue = InDecide.contains(BB) && !isa<PHINode>(I);
        bool Counter = &I == C.IV || &I == C.IVNext;
        bool CarriedOut = isa<PHINode>(I) && BB == C.Latch;
        if (DecideValue && !DT->dominates(BB, C.Latch))
          return reject("a value used after the loop is not computed in "
                        "every iteration");
        if (!DecideValue && !Counter && !CarriedOut)
          return reject("a value used after the loop (" + describe(I) +
                        ") comes from the guarded part or the iteration's "
                        "start");
      }

  return memoryIsIndependent(C);
}

/// Whether \p S, run in an iteration j, can never write what \p Ld reads in a
/// later iteration i > j: both addresses step through the loop, a constant
/// distance apart.
bool RISCVLoopDecideAct::storeMissesLaterLoads(const Candidate &C, StoreInst *S,
                                               LoadInst *Ld) {
  const SCEV *SA = SE->getSCEV(S->getPointerOperand());
  const SCEV *LA = SE->getSCEV(Ld->getPointerOperand());
  auto split = [&](const SCEV *A, const SCEV *&Start, int64_t &Step) {
    if (auto *AR = dyn_cast<SCEVAddRecExpr>(A)) {
      auto *St = dyn_cast<SCEVConstant>(AR->getStepRecurrence(*SE));
      if (AR->getLoop() != C.L || !AR->isAffine() || !St)
        return false;
      Start = AR->getStart();
      Step = St->getAPInt().getSExtValue();
      return true;
    }
    if (!SE->isLoopInvariant(A, C.L))
      return false;
    Start = A;
    Step = 0;
    return true;
  };
  const SCEV *SS, *LS;
  int64_t SStep, LStep;
  if (!split(SA, SS, SStep) || !split(LA, LS, LStep))
    return false;
  auto *D = dyn_cast<SCEVConstant>(SE->getMinusSCEV(SS, LS));
  if (!D)
    return false;
  const DataLayout &DL = S->getDataLayout();
  int64_t SSize = DL.getTypeStoreSize(S->getValueOperand()->getType());
  int64_t LSize = DL.getTypeStoreSize(Ld->getType());
  int64_t Delta = D->getAPInt().getSExtValue();
  // Store in iteration j, load in iteration i, 0 <= j < i < MaxTrip: the
  // store's address less the load's is Delta + SStep * j - LStep * i. They
  // overlap if that lies in (-SSize, LSize).
  int64_t T = C.MaxTrip;
  auto overlaps = [&](int64_t Diff) { return Diff > -SSize && Diff < LSize; };
  if (SStep == LStep) {
    for (int64_t K = 1; K < T; ++K) // K = i - j
      if (overlaps(Delta - SStep * K))
        return false;
    return true;
  }
  if (SStep == 0) {
    for (int64_t I = 1; I < T; ++I)
      if (overlaps(Delta - LStep * I))
        return false;
    return true;
  }
  if (LStep == 0) {
    for (int64_t J = 0; J + 1 < T; ++J)
      if (overlaps(Delta + SStep * J))
        return false;
    return true;
  }
  return false;
}

bool RISCVLoopDecideAct::memoryIsIndependent(const Candidate &C) {
  SmallVector<LoadInst *, 8> Loads;
  for (BasicBlock *BB : C.Decide)
    for (Instruction &I : *BB)
      if (auto *Ld = dyn_cast<LoadInst>(&I))
        Loads.push_back(Ld);
  SmallPtrSet<BasicBlock *, 8> DecideSet(C.Decide.begin(), C.Decide.end());
  for (BasicBlock *BB : C.L->blocks()) {
    if (DecideSet.contains(BB) || BB == C.Latch)
      continue;
    for (Instruction &I : *BB) {
      if (!I.mayWriteToMemory())
        continue;
      for (LoadInst *Ld : Loads) {
        // Alias analysis answers for one iteration: the load and the write
        // seen with the same values. Ask about every access through each
        // pointer instead, keeping only TBAA, which holds across iterations
        // (scoped noalias metadata does not); the per-iteration distance is
        // checked below with SCEV.
        auto TBAAOnly = [](const Instruction *X) {
          AAMDNodes Tags = X->getAAMetadata();
          Tags.Scope = nullptr;
          Tags.NoAlias = nullptr;
          return Tags;
        };
        MemoryLocation Loc = MemoryLocation::getBeforeOrAfter(
            Ld->getPointerOperand(), TBAAOnly(Ld));
        ModRefInfo MR;
        if (auto *S = dyn_cast<StoreInst>(&I))
          MR = AA->isNoAlias(Loc, MemoryLocation::getBeforeOrAfter(
                                      S->getPointerOperand(), TBAAOnly(S)))
                   ? ModRefInfo::NoModRef
                   : ModRefInfo::Mod;
        else
          MR = AA->getModRefInfo(&I, Loc);
        if (!isModSet(MR))
          continue;
        auto *S = dyn_cast<StoreInst>(&I);
        if (S && S->isSimple() && storeMissesLaterLoads(C, S, Ld))
          continue;
        return reject(describe(I) +
                      " in the guarded part may change what a later "
                      "iteration's condition reads (" +
                      describe(*Ld) + ")");
      }
    }
  }
  return true;
}

void RISCVLoopDecideAct::transform(Candidate &C) {
  Function &F = *C.Header->getParent();
  LLVMContext &Ctx = F.getContext();
  Type *IVTy = C.IV->getType();
  DebugLoc DL = C.Header->getTerminator()->getDebugLoc();
  Value *Start = C.IV->getIncomingValueForBlock(C.Preheader);

  // The list of chosen iterations, on the stack.
  IRBuilder<> EntryB(&*F.getEntryBlock().getFirstInsertionPt());
  ArrayType *ListTy = ArrayType::get(IVTy, C.MaxTrip);
  AllocaInst *List = EntryB.CreateAlloca(ListTy, nullptr, "decide.list");

  BasicBlock *Dec = BasicBlock::Create(Ctx, "decide.loop", &F, C.Header);
  BasicBlock *ActPre = BasicBlock::Create(Ctx, "act.preheader", &F, C.Header);
  BasicBlock *ExitJoin = BasicBlock::Create(Ctx, "act.exit", &F, C.Exit);

  // --- The decide loop: the decide part if-converted, every iteration. ---
  IRBuilder<> B(Dec);
  B.SetCurrentDebugLocation(DL);
  PHINode *DIV = B.CreatePHI(IVTy, 2, "decide.iv");
  PHINode *DN = B.CreatePHI(IVTy, 2, "decide.n");
  ValueToValueMapTy VMap;
  VMap[C.IV] = DIV;
  DenseMap<BasicBlock *, Value *> Pred;
  DenseMap<std::pair<BasicBlock *, BasicBlock *>, Value *> EdgePred;
  Value *True = ConstantInt::getTrue(Ctx);
  for (BasicBlock *BB : C.Decide) {
    Value *P = True;
    if (BB != C.Header) {
      P = nullptr;
      for (BasicBlock *From : predecessors(BB)) {
        Value *E = EdgePred.lookup({From, BB});
        P = P ? B.CreateOr(P, E) : E;
      }
    }
    Pred[BB] = P;
    for (Instruction &I : *BB) {
      if (isa<PHINode>(I) || I.isTerminator() || isa<DbgInfoIntrinsic>(I))
        continue;
      Instruction *New = I.clone();
      New->setName(I.getName() + ".decide");
      // A copy runs whether or not its block would have: drop facts that only
      // held on the original path.
      New->dropUBImplyingAttrsAndMetadata();
      B.Insert(New);
      VMap[&I] = New;
      RemapInstruction(New, VMap,
                       RF_IgnoreMissingLocals | RF_NoModuleLevelChanges);
    }
    auto *Br = cast<BranchInst>(BB->getTerminator());
    if (Br->isUnconditional()) {
      EdgePred[{BB, Br->getSuccessor(0)}] = P;
      continue;
    }
    Value *Cond = VMap.lookup(Br->getCondition());
    if (!Cond)
      Cond = Br->getCondition(); // invariant
    Value *T = B.CreateAnd(P, Cond);
    Value *E = B.CreateAnd(P, B.CreateNot(Cond));
    auto add = [&](BasicBlock *S, Value *V) {
      Value *&Slot = EdgePred[{BB, S}];
      Slot = Slot ? B.CreateOr(Slot, V) : V;
    };
    add(Br->getSuccessor(0), T);
    add(Br->getSuccessor(1), E);
  }
  Value *Chosen = nullptr;
  for (BasicBlock *From : predecessors(C.ActEntry)) {
    Value *E = EdgePred.lookup({From, C.ActEntry});
    Chosen = Chosen ? B.CreateOr(Chosen, E) : E;
  }
  Value *Slot = B.CreateInBoundsGEP(
      ListTy, List, {ConstantInt::get(IVTy, 0), DN}, "decide.slot");
  B.CreateStore(DIV, Slot);
  Value *DNNext = B.CreateAdd(DN, B.CreateZExt(Chosen, IVTy), "decide.n.next");
  Instruction *DIVNext = C.IVNext->clone();
  DIVNext->setName("decide.iv.next");
  B.Insert(DIVNext);
  DIVNext->replaceUsesOfWith(C.IV, DIV);
  Instruction *DCmp = C.ExitCmp->clone();
  B.Insert(DCmp);
  DCmp->replaceUsesOfWith(C.IVNext, DIVNext);
  DCmp->replaceUsesOfWith(C.IV, DIV);
  bool ContinueOnTrue = C.LatchBr->getSuccessor(0) == C.Header;
  B.CreateCondBr(DCmp, ContinueOnTrue ? Dec : ActPre,
                 ContinueOnTrue ? ActPre : Dec);
  DIV->addIncoming(Start, C.Preheader);
  DIV->addIncoming(DIVNext, Dec);
  DN->addIncoming(ConstantInt::get(IVTy, 0), C.Preheader);
  DN->addIncoming(DNNext, Dec);
  VMap[C.IVNext] = DIVNext;

  // The preheader now enters the decide loop; the act loop runs if any
  // iteration was chosen.
  C.Preheader->getTerminator()->setSuccessor(0, Dec);
  IRBuilder<> PB(ActPre);
  PB.SetCurrentDebugLocation(DL);
  PB.CreateCondBr(PB.CreateICmpEQ(DNNext, ConstantInt::get(IVTy, 0)), ExitJoin,
                  C.Header);
  for (PHINode *H : C.Carried)
    H->setIncomingBlock(H->getBasicBlockIndex(C.Preheader), ActPre);

  // --- Uses after the loop. Rewritten before the act loop changes. ---
  IRBuilder<> XB(ExitJoin);
  XB.SetCurrentDebugLocation(DL);
  SmallVector<std::pair<Use *, Value *>, 8> Rewrites;
  for (BasicBlock *BB : C.L->blocks())
    for (Instruction &I : *BB) {
      Value *Out = nullptr;
      for (Use &U : I.uses()) {
        auto *UI = dyn_cast<Instruction>(U.getUser());
        if (!UI || C.L->contains(UI) ||
            C.NoReturnExits.contains(UI->getParent()))
          continue;
        if (!Out) {
          if (Value *V = VMap.lookup(&I)) {
            Out = V; // the decide loop's last iteration
          } else {
            // A carried value as the latch leaves it, or as it entered the
            // loop if no iteration was chosen.
            auto *P = cast<PHINode>(&I);
            Value *Init = nullptr;
            for (PHINode *H : C.Carried)
              if (H->getIncomingValueForBlock(C.Latch) == P)
                Init = H->getIncomingValueForBlock(ActPre);
            PHINode *J = XB.CreatePHI(P->getType(), 2, P->getName() + ".act");
            J->addIncoming(P, C.Latch);
            J->addIncoming(Init ? Init : PoisonValue::get(P->getType()),
                           ActPre);
            Out = J;
          }
        }
        Rewrites.push_back({&U, Out});
      }
    }
  for (auto [U, V] : Rewrites)
    U->set(V);
  XB.CreateBr(C.Exit);
  for (PHINode &P : C.Exit->phis())
    for (unsigned I = 0; I < P.getNumIncomingValues(); ++I)
      if (P.getIncomingBlock(I) == C.Latch)
        P.setIncomingBlock(I, ExitJoin);
  C.LatchBr->replaceSuccessorWith(C.Exit, ExitJoin);

  // --- The act loop: the original loop over the chosen iterations. ---
  IRBuilder<> HB(&*C.Header->getFirstInsertionPt());
  HB.SetCurrentDebugLocation(DL);
  PHINode *M = PHINode::Create(IVTy, 2, "act.m");
  M->insertBefore(C.Header->begin());
  Value *MSlot = HB.CreateInBoundsGEP(
      ListTy, List, {ConstantInt::get(IVTy, 0), M}, "act.slot");
  Value *ActI = HB.CreateLoad(IVTy, MSlot, "act.i");
  C.IV->replaceAllUsesWith(ActI);
  C.IV->eraseFromParent();
  IRBuilder<> LB(C.LatchBr);
  LB.SetCurrentDebugLocation(DL);
  Value *MNext = LB.CreateAdd(M, ConstantInt::get(IVTy, 1), "act.m.next");
  Value *More = LB.CreateICmpNE(MNext, DNNext, "act.more");
  C.LatchBr->setCondition(More);
  if (C.LatchBr->getSuccessor(0) != C.Header)
    C.LatchBr->swapSuccessors();
  M->addIncoming(ConstantInt::get(IVTy, 0), ActPre);
  M->addIncoming(MNext, C.Latch);
  // The old exit test, and the old increment if nothing else uses it
  RecursivelyDeleteTriviallyDeadInstructions(C.ExitCmp);

  // Every act iteration reaches the guarded part: a condition branch to the
  // latch is never taken there.
  for (BasicBlock *BB : C.Decide) {
    auto *Br = cast<BranchInst>(BB->getTerminator());
    if (!Br->isConditional())
      continue;
    int ToLatch = Br->getSuccessor(0) == C.Latch   ? 0
                  : Br->getSuccessor(1) == C.Latch ? 1
                                                   : -1;
    if (ToLatch < 0)
      continue;
    BasicBlock *Keep = Br->getSuccessor(1 - ToLatch);
    C.Latch->removePredecessor(BB);
    BranchInst *New = BranchInst::Create(Keep, Br->getIterator());
    New->setDebugLoc(Br->getDebugLoc());
    Value *Cond = Br->getCondition();
    Br->eraseFromParent();
    RecursivelyDeleteTriviallyDeadInstructions(Cond);
  }
  removeUnreachableBlocks(F);
}

bool RISCVLoopDecideAct::runOnFunction(Function &F) {
  if (skipFunction(F) || DisableDecideAct)
    return false;
  auto &TM = getAnalysis<TargetPassConfig>().getTM<RISCVTargetMachine>();
  if (TM.Options.EnableGlobalISel)
    return false;
  DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  SE = &getAnalysis<ScalarEvolutionWrapperPass>().getSE();
  AA = &getAnalysis<AAResultsWrapperPass>().getAAResults();
  AC = &getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);
  OptimizationRemarkEmitter ORE(&F);

  // Plan every loop first, while the analyses describe the function; then
  // split the outermost candidates (a split loop's inner loops are left
  // alone, as are loops around it).
  SmallVector<Candidate, 2> Plans;
  for (Loop *L : LI->getLoopsInPreorder()) {
    if (any_of(Plans, [&](const Candidate &C) {
          return C.L->contains(L) || L->contains(C.L);
        }))
      continue;
    Candidate C;
    if (!analyze(L, C)) {
      ORE.emit([&]() {
        return OptimizationRemarkMissed(DEBUG_TYPE, "NotSplit",
                                        L->getStartLoc(), L->getHeader())
               << "not splitting the loop into a decide and an act loop: "
               << Why;
      });
      continue;
    }
    ORE.emit([&]() {
      return OptimizationRemark(DEBUG_TYPE, "Split", L->getStartLoc(),
                                L->getHeader())
             << "split the loop: its conditions are computed for every "
                "iteration first, without a branch, then the guarded part "
                "runs for the chosen ones";
    });
    Plans.push_back(std::move(C));
  }
  for (Candidate &C : Plans) {
    transform(C);
    ++NumDecideActLoops;
  }
  assert((Plans.empty() || !verifyFunction(F, &dbgs())) &&
         "splitting a loop into decide and act loops left invalid IR");
  return !Plans.empty();
}

INITIALIZE_PASS_BEGIN(RISCVLoopDecideAct, DEBUG_TYPE,
                      "RISC-V decide/act loop splitting", false, false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_END(RISCVLoopDecideAct, DEBUG_TYPE,
                    "RISC-V decide/act loop splitting", false, false)

FunctionPass *llvm::createRISCVLoopDecideActPass() {
  return new RISCVLoopDecideAct();
}
