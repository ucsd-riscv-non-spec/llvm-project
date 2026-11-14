//===-- RISCVIRLoopFission.cpp - Fission conditional loops ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVNSLoopOptions.h"
#include "RISCVSubtarget.h"
#include "RISCVTargetMachine.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsRISCV.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-ir-loop-fission"

namespace {

static cl::opt<bool>
    DisableIRLoopFission("riscv-ns-disable-ir-loop-fission", cl::Hidden,
                         cl::init(false),
                         cl::desc("Disable RISC-V IR loop fission"));

struct LoopCandidate {
  RISCVNS::FissionSettings Settings;
  Loop *L;
  BasicBlock *Preheader;
  BasicBlock *Header;
  BasicBlock *Latch;
  BranchInst *DataBranch;
  BasicBlock *Body;
  BasicBlock *Merge;
  unsigned TripCount;
};

class RISCVIRLoopFission : public FunctionPass {
public:
  static char ID;

  RISCVIRLoopFission() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetPassConfig>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<ScalarEvolutionWrapperPass>();
    AU.addRequired<AAResultsWrapperPass>();
  }

private:
  bool findCandidate(Loop &L, LoopCandidate &C);
  bool fission(LoopCandidate &C, unsigned Queue);

  // Why findCandidate rejected the loop, for the missed remark
  std::string Why;
  bool reject(const Twine &W) {
    Why = W.str();
    return false;
  }

  DominatorTree *DT = nullptr;
  LoopInfo *LI = nullptr;
  ScalarEvolution *SE = nullptr;
  AAResults *AA = nullptr;
  unsigned XLen = 0;
};

char RISCVIRLoopFission::ID = 0;

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

static void markFissionedLoop(BranchInst &LatchBranch) {
  LatchBranch.setMetadata(
      LLVMContext::MD_loop,
      RISCVNS::makeLoopID(LatchBranch.getContext(),
                          LatchBranch.getMetadata(LLVMContext::MD_loop),
                          RISCVNS::LoopRole::Fission));
}

bool RISCVIRLoopFission::findCandidate(Loop &L, LoopCandidate &C) {
  C.L = &L;
  C.Preheader = L.getLoopPreheader();
  C.Header = L.getHeader();
  C.Latch = L.getLoopLatch();
  C.TripCount = SE->getSmallConstantTripCount(&L);

  if (!L.isInnermost())
    return reject("the loop contains another loop");
  if (!C.Preheader || !C.Latch)
    return reject("the loop has no preheader or no single latch");
  if (C.TripCount == 0)
    return reject("the trip count is not a compile-time constant");
  if (C.TripCount > C.Settings.QueueDepth)
    return reject("the trip count (" + Twine(C.TripCount) +
                  ") is more than the condition queue holds (" +
                  Twine(C.Settings.QueueDepth) + ")");
  if (L.getExitingBlock() != C.Latch || !L.getExitBlock())
    return reject("the loop can exit other than at its latch");

  auto *PreheaderBranch = dyn_cast<BranchInst>(C.Preheader->getTerminator());
  if (!PreheaderBranch || !PreheaderBranch->isUnconditional() ||
      PreheaderBranch->getSuccessor(0) != C.Header)
    return reject("the preheader does not end in a jump to the loop");

  auto *LatchBranch = dyn_cast<BranchInst>(C.Latch->getTerminator());
  if (!LatchBranch || !LatchBranch->isConditional())
    return reject("the latch does not end in a conditional branch");

  unsigned DataBranches = 0;
  for (BasicBlock *BB : L.blocks()) {
    if (BB == C.Latch)
      continue;
    auto *Branch = dyn_cast<BranchInst>(BB->getTerminator());
    if (!Branch || !Branch->isConditional())
      continue;
    ++DataBranches;
    C.DataBranch = Branch;
  }
  if (DataBranches != 1)
    return reject("the loop has " + Twine(DataBranches) +
                  " conditional branches besides the latch (one is handled)");
  if (C.DataBranch->getParent() != C.Header)
    return reject("the conditional branch is not in the loop header");

  auto *Condition = dyn_cast<ICmpInst>(C.DataBranch->getCondition());
  if (!Condition || !Condition->getOperand(0)->getType()->isIntegerTy() ||
      Condition->getOperand(0)->getType() != Condition->getOperand(1)->getType() ||
      Condition->getOperand(0)->getType()->getIntegerBitWidth() > XLen)
    return reject("the branch condition is not an integer compare");

  C.Body = C.DataBranch->getSuccessor(0);
  C.Merge = C.DataBranch->getSuccessor(1);
  if (C.Body == C.Merge || !L.contains(C.Body) || !L.contains(C.Merge) ||
      C.Body == C.Latch || C.Body == C.Header ||
      pred_size(C.Body) != 1 || *pred_begin(C.Body) != C.Header ||
      !C.Body->phis().empty())
    return reject("the branch does not guard a single block that "
                  "rejoins the loop (an if without else)");

  auto *BodyBranch = dyn_cast<BranchInst>(C.Body->getTerminator());
  if (!BodyBranch || !BodyBranch->isUnconditional() ||
      BodyBranch->getSuccessor(0) != C.Merge)
    return reject("the guarded block does not jump straight to the merge");
  for (PHINode &Phi : C.Merge->phis()) {
    int BodyIndex = Phi.getBasicBlockIndex(C.Body);
    if (BodyIndex < 0)
      continue;
    for (unsigned I = 0; I < Phi.getNumIncomingValues(); ++I)
      if (I != static_cast<unsigned>(BodyIndex) &&
          Phi.getIncomingValue(I) != Phi.getIncomingValue(BodyIndex))
        return reject("the guarded block changes a value used after it");
  }
  for (Instruction &I : *C.Body)
    if (!I.isTerminator())
      for (User *U : I.users())
        if (auto *Use = dyn_cast<Instruction>(U))
          if (Use->getParent() != C.Body)
            return reject("a value computed in the guarded block is used "
                          "outside it");

  SmallVector<Instruction *, 16> Worklist;
  SmallPtrSet<Instruction *, 32> Visited;
  auto AddConditionValue = [&](Value *V) {
    if (auto *I = dyn_cast<Instruction>(V))
      if (L.contains(I->getParent()) && Visited.insert(I).second)
        Worklist.push_back(I);
  };
  AddConditionValue(C.DataBranch->getCondition());
  while (!Worklist.empty()) {
    Instruction *I = Worklist.pop_back_val();
    if (!DT->dominates(I, C.DataBranch))
      return reject("the condition depends on " + describe(*I) +
                    ", which does not run before the branch");
    // The complete loop is cloned, so these are semantic proofs rather than
    // requirements for mapping the condition into the compute loop.
    if (C.Settings.Force) {
      if (isa<PHINode>(I))
        continue;
    } else if (auto *Load = dyn_cast<LoadInst>(I)) {
      if (!Load->isSimple())
        return reject("the condition depends on a volatile or atomic load");
    } else if (isa<PHINode>(I)) {
      continue;
    } else if (I->mayReadOrWriteMemory() || I->mayHaveSideEffects() ||
               isa<CallBase>(I)) {
      return reject("the condition depends on " + describe(*I));
    }
    for (Value *Operand : I->operands())
      AddConditionValue(Operand);
  }

  SmallVector<LoadInst *, 8> Loads;
  SmallVector<StoreInst *, 8> Stores;
  for (BasicBlock *BB : L.blocks()) {
    for (Instruction &I : *BB) {
      if (isa<DbgInfoIntrinsic>(I) || I.isTerminator() || isa<PHINode>(I))
        continue;
      // The compute loop drops the conditional body and clones the remaining
      // instructions. Force permits their reordered or duplicated effects.
      if (C.Settings.Force)
        continue;
      if (auto *Store = dyn_cast<StoreInst>(&I)) {
        if (!Store->isSimple())
          return reject("the loop has a volatile or atomic store");
        if (BB != C.Body)
          return reject("the loop stores outside the guarded block (" +
                        describe(I) + ")");
        Stores.push_back(Store);
        continue;
      }
      if (auto *Load = dyn_cast<LoadInst>(&I)) {
        if (!Load->isSimple())
          return reject("the loop has a volatile or atomic load");
        Loads.push_back(Load);
        continue;
      }
      if (I.mayReadOrWriteMemory() || I.mayHaveSideEffects() ||
          isa<CallBase>(I))
        return reject("the loop contains " + describe(I));
    }
  }

  if (!C.Settings.Force && Stores.empty())
    return reject("the guarded block stores nothing");
  for (LoadInst *Load : Loads)
    for (StoreInst *Store : Stores)
      if (AA->alias(MemoryLocation::get(Load), MemoryLocation::get(Store)) !=
          AliasResult::NoAlias)
        return reject("a load in the loop (" + describe(*Load) +
                      ") may read what the guarded store (" + describe(*Store) +
                      ") writes");

  for (PHINode &Phi : C.Header->phis())
    if (Phi.getBasicBlockIndex(C.Preheader) < 0)
      return reject("a header phi has no value from the preheader");

  return true;
}

bool RISCVIRLoopFission::fission(LoopCandidate &C, unsigned Queue) {
  Function *F = C.Header->getParent();
  LLVMContext &Context = F->getContext();
  IntegerType *XLenTy = IntegerType::get(Context, XLen);

  ValueToValueMapTy VMap;
  SmallVector<BasicBlock *, 8> ClonedBlocks;
  for (BasicBlock *BB : C.L->blocks()) {
    BasicBlock *Clone = CloneBasicBlock(BB, VMap, ".fission", F);
    F->splice(C.Header->getIterator(), F, Clone->getIterator());
    VMap[BB] = Clone;
    ClonedBlocks.push_back(Clone);
  }
  remapInstructionsInBlocks(ClonedBlocks, VMap);

  auto *ClonedHeader = cast<BasicBlock>(VMap[C.Header]);
  auto *ClonedLatch = cast<BasicBlock>(VMap[C.Latch]);
  auto *ClonedBranch = cast<BranchInst>(VMap[C.DataBranch]);
  auto *ClonedBody = cast<BasicBlock>(VMap[C.Body]);
  auto *ClonedMerge = cast<BasicBlock>(VMap[C.Merge]);
  auto *PreheaderBranch = cast<BranchInst>(C.Preheader->getTerminator());
  auto *LatchBranch = cast<BranchInst>(C.Latch->getTerminator());
  auto *ClonedLatchBranch = cast<BranchInst>(ClonedLatch->getTerminator());

  BasicBlock *FissionExit =
      BasicBlock::Create(Context, "fission.exit", F, C.Header);
  IRBuilder<> ExitBuilder(FissionExit);
  ExitBuilder.SetCurrentDebugLocation(LatchBranch->getDebugLoc());
  ExitBuilder.CreateBr(C.Header);

  for (unsigned I = 0; I < ClonedLatchBranch->getNumSuccessors(); ++I)
    if (!is_contained(ClonedBlocks,
                      ClonedLatchBranch->getSuccessor(I)))
      ClonedLatchBranch->setSuccessor(I, FissionExit);

  for (PHINode &Phi : C.Header->phis()) {
    int Index = Phi.getBasicBlockIndex(C.Preheader);
    Phi.setIncomingBlock(Index, FissionExit);
  }

  PreheaderBranch->setSuccessor(0, ClonedHeader);

  IRBuilder<> InitBuilder(PreheaderBranch);
  InitBuilder.SetCurrentDebugLocation(PreheaderBranch->getDebugLoc());
  InitBuilder.CreateIntrinsic(Intrinsic::riscv_ns_cond_init, {},
                              {InitBuilder.getInt32(Queue)});

  IRBuilder<> PushBuilder(ClonedBranch);
  PushBuilder.SetCurrentDebugLocation(C.DataBranch->getDebugLoc());
  Value *Condition = ClonedBranch->getCondition();
  auto *Compare = cast<ICmpInst>(Condition);
  Value *LHS = Compare->getOperand(0);
  Value *RHS = Compare->getOperand(1);
  LHS = PushBuilder.CreateSExtOrBitCast(LHS, XLenTy);
  RHS = PushBuilder.CreateSExtOrBitCast(RHS, XLenTy);
  PushBuilder.CreateIntrinsic(
      Intrinsic::riscv_ns_cond_push, {XLenTy},
      {PushBuilder.getInt32(Queue),
       PushBuilder.getInt32(static_cast<unsigned>(Compare->getPredicate())),
       LHS, RHS});

  DebugLoc BranchDL = ClonedBranch->getDebugLoc();
  BranchInst::Create(ClonedMerge, ClonedBranch->getIterator());
  ClonedBranch->eraseFromParent();
  for (Instruction &I : make_early_inc_range(*ClonedBody))
    if (isa<StoreInst>(I))
      I.eraseFromParent();
  for (Instruction &I : make_early_inc_range(*ClonedBody))
    if (!I.isTerminator())
      RecursivelyDeleteTriviallyDeadInstructions(&I);
  ClonedBody->getTerminator()->setDebugLoc(BranchDL);
  DeleteDeadBlock(ClonedBody);

  IRBuilder<> PopBuilder(C.DataBranch);
  PopBuilder.SetCurrentDebugLocation(C.DataBranch->getDebugLoc());
  Value *Pop = PopBuilder.CreateIntrinsic(Intrinsic::riscv_ns_cond_pop, {},
                                          {PopBuilder.getInt32(Queue)});
  Value *OldCondition = C.DataBranch->getCondition();
  C.DataBranch->setCondition(Pop);
  RecursivelyDeleteTriviallyDeadInstructions(
      cast<Instruction>(OldCondition));

  markFissionedLoop(*LatchBranch);
  markFissionedLoop(*ClonedLatchBranch);

  LLVM_DEBUG(dbgs() << "fission: transformed " << F->getName()
                    << " trip count " << C.TripCount << "\n");
  return true;
}

bool RISCVIRLoopFission::runOnFunction(Function &F) {
  if (!RISCVNS::validateLoopOptions(F) || skipFunction(F))
    return false;

  auto &TM = getAnalysis<TargetPassConfig>().getTM<RISCVTargetMachine>();
  if (TM.Options.EnableGlobalISel)
    return false;
  XLen = TM.getSubtarget<RISCVSubtarget>(F).getXLen();

  DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  SE = &getAnalysis<ScalarEvolutionWrapperPass>().getSE();
  AA = &getAnalysis<AAResultsWrapperPass>().getAAResults();

  SmallVector<BasicBlock *, 4> Headers;
  for (Loop *L : LI->getLoopsInPreorder())
    if (L->isInnermost())
      Headers.push_back(L->getHeader());

  unsigned Queue = 0;
  for (Instruction &I : instructions(F))
    if (auto *II = dyn_cast<IntrinsicInst>(&I))
      if (II->getIntrinsicID() == Intrinsic::riscv_ns_cond_init ||
          II->getIntrinsicID() == Intrinsic::riscv_ns_cond_push) {
        auto *ID = dyn_cast<ConstantInt>(II->getArgOperand(0));
        if (ID)
          Queue = std::max(Queue, static_cast<unsigned>(ID->getZExtValue() + 1));
      }

  OptimizationRemarkEmitter ORE(&F);
  for (BasicBlock *Header : Headers) {
    Loop *L = LI->getLoopFor(Header);
    if (!L || L->getHeader() != Header)
      continue;
    LoopCandidate Candidate;
    auto Options = RISCVNS::parseLoopOptions(L->getLoopID(),
                                             *L->getHeader()->getTerminator());
    if (!Options)
      continue;
    Candidate.Settings = Options->resolveIRFission(DisableIRLoopFission);
    if (!Candidate.Settings.Enable || RISCVNS::isIRFissioned(L->getLoopID()) ||
        !RISCVNS::getSWPApplied(L->getLoopID()).empty())
      continue;
    if (!findCandidate(*L, Candidate)) {
      ORE.emit([&]() {
        return OptimizationRemarkMissed(DEBUG_TYPE, "NotFissioned",
                                        L->getStartLoc(), L->getHeader())
               << "not splitting the loop (IR loop fission): " << Why;
      });
      continue;
    }
    // The transformation changes the loop: take what the remark needs first
    DebugLoc Loc = L->getStartLoc();
    if (fission(Candidate, Queue)) {
      ORE.emit([&]() {
        return OptimizationRemark(DEBUG_TYPE, "Fissioned", Loc, Header)
               << "split the loop: its conditions are computed first and "
                  "queued";
      });
      return true;
    }
  }
  return false;
}

} // namespace

INITIALIZE_PASS_BEGIN(RISCVIRLoopFission, DEBUG_TYPE,
                      "RISC-V IR loop fission", false, false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_END(RISCVIRLoopFission, DEBUG_TYPE,
                    "RISC-V IR loop fission", false, false)

FunctionPass *llvm::createRISCVIRLoopFissionPass() {
  return new RISCVIRLoopFission();
}
