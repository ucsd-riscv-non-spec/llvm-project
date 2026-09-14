//==- RISCVBranchSetupHoisting.cpp - Hoist BMOV insts. towards fn. entry -====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVBranchSetupAnalysis.h"
#include "RISCVInstrInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/Support/CommandLine.h"

#define DEBUG_TYPE "riscv-branch-setup-hoisting"
#define PASS_NAME "RISC-V branch setup hoisting pass"

using namespace llvm;

static cl::opt<bool> DisableBranchSetupHoisting("disable-branch-setup-hoisting",
                                                cl::Hidden,
                                                cl::desc("Disable " PASS_NAME),
                                                cl::init(false));

static cl::opt<bool>
    DisableCrossBBHoisting("disable-cross-bb-hoisting", cl::Hidden,
                           cl::desc("Disable Cross BB Hoisting"),
                           cl::init(false));

STATISTIC(NumBMOVHoisted, "Number of BMOV instructions able to be hoisted");

namespace {

class RISCVBranchSetupHoisting : public MachineFunctionPass {
public:
  static char ID;
  const RISCVBranchSetupInfo *BSI;
  const TargetRegisterInfo *TRI;
  MachineDominatorTree *MDT;
  MachineLoopInfo *MLI;

  RISCVBranchSetupHoisting() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<RISCVBranchSetupAnalysisWrapper>();
    AU.addRequired<MachineDominatorTreeWrapperPass>();
    AU.addRequired<MachineLoopInfoWrapperPass>();
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override { return PASS_NAME; }

  bool scheduleBranchSetup(MachineInstr *MI, MachineInstr *S, MachineInstr *T,
                           MachineInstr *C);
  /// Returns the block to hoist \p SetupMI into and the insertion point
  /// within it. The iterator may be that block's end(), so it must not be
  /// dereferenced to recover the block.
  std::pair<MachineBasicBlock *, MachineBasicBlock::iterator>
  findEarliestSafePoint(MachineInstr &SetupMI,
                        SmallSet<MachineInstr *, 4> Setup);
  bool isSafeToHoistTo(const MachineInstr &SetupMI, MachineBasicBlock &DestBB,
                       MachineBasicBlock::const_iterator InsertPt) const;
};

char RISCVBranchSetupHoisting::ID = 0;

static void printMachineCFG(const MachineFunction &MF) {
  errs() << "\nMachineFunction: " << MF.getName() << '\n';

  for (const MachineBasicBlock &MBB : MF) {
    errs() << "\nbb." << MBB.getNumber();

    if (MBB.hasName())
      errs() << " (" << MBB.getName() << ")";

    errs() << "\n  successors:";

    for (const MachineBasicBlock *Succ : MBB.successors())
      errs() << " bb." << Succ->getNumber();

    errs() << '\n';

    for (const MachineInstr &MI : MBB) {
      if (MI.isTerminator() || MI.isBMOV() || MI.isCall()) {
        errs() << "    ";
        MI.print(errs());
      }
    }
  }
}

bool RISCVBranchSetupHoisting::runOnMachineFunction(MachineFunction &MF) {
  if (DisableBranchSetupHoisting)
    return false;

  LLVM_DEBUG(printMachineCFG(MF));

  TRI = MF.getSubtarget().getRegisterInfo();
  BSI = &getAnalysis<RISCVBranchSetupAnalysisWrapper>().getInfo();
  MDT = &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  MLI = &getAnalysis<MachineLoopInfoWrapperPass>().getLI();

  constexpr unsigned MaxBranchSetups = 30;
  if (BSI->Branches.size() > MaxBranchSetups)
    return false;

  // Collect work in machine-function order. Iteration order of the analysis
  // map must not determine physical-register assignment.
  struct BranchWork {
    MachineInstr *MI;
    MachineInstr *S, *T, *C;
  };
  SmallVector<BranchWork, 16> WorkList;

  for (auto &MBB : MF) {
    for (auto &MI : MBB) {
      if (RISCVBranchSetup BS = BSI->Branches.lookup(&MI)) {
        WorkList.push_back({&MI, const_cast<MachineInstr *>(BS.S),
                            const_cast<MachineInstr *>(BS.T),
                            const_cast<MachineInstr *>(BS.C)});
      }
    }
  }

  // A shared setup instruction cannot be renamed independently for two branch
  // consumers. Refuse the transformation instead of silently miscompiling.
  SmallPtrSet<MachineInstr *, 32> ClaimedSetups;
  for (const BranchWork &W : WorkList) {
    if (!W.S || !W.T)
      return false;
    for (MachineInstr *SetupMI : {W.S, W.C, W.T})
      if (SetupMI && !ClaimedSetups.insert(SetupMI).second)
        return false;
  }

  bool Changed = false;
  for (auto [BranchIdx, W] : llvm::enumerate(WorkList)) {
    Register NewReg = Register(RISCV::B0 + static_cast<unsigned>(BranchIdx));
    Register OldReg = W.S->getOperand(0).getReg();

    auto Rewrite = [&](MachineInstr *SetupMI) {
      if (!SetupMI)
        return;
      MachineOperand &Reg = SetupMI->getOperand(0);
      if (Reg.getReg() == NewReg)
        return;
      Reg.setReg(NewReg);
      Changed = true;
    };

    Rewrite(W.S);
    Rewrite(W.C);
    Rewrite(W.T);
    Rewrite(W.MI);
  }

  for (BranchWork &W : llvm::reverse(WorkList)) {
    Changed |= scheduleBranchSetup(W.MI, W.S, W.T, W.C);
  }

  LLVM_DEBUG(printMachineCFG(MF));

  return Changed;
}

} // end of anonymous namespace

static bool isHoistBarrier(const MachineInstr &MI) {
  if (MI.isBMOV())
    return false;
  // isBarrier() means that control cannot fall through; it is not a generic
  // instruction-motion barrier. Calls, inline asm, and unmodelled effects are
  // the boundaries relevant to these branch-setup register writes.
  return MI.isCall() || MI.isIndirectBranch() || MI.isInlineAsm() ||
         MI.hasUnmodeledSideEffects();
}

bool RISCVBranchSetupHoisting::scheduleBranchSetup(MachineInstr *MI,
                                                   MachineInstr *S,
                                                   MachineInstr *T,
                                                   MachineInstr *C) {
  assert(S && "Branch must have BMOVS");
  assert(T && "Branch must have BMOVT");
  bool Changed = false;

  SmallSet<MachineInstr *, 4> Setups{{MI, S, T, C}};

  // Keep architectural setup order. The dependency checks also prevent a
  // later setup from being moved across an earlier setup of the same B-reg.
  for (MachineInstr *SetupMI : {S, C, T}) {
    if (!SetupMI)
      continue;

    LLVM_DEBUG(dbgs() << "Hoisting: " << *SetupMI);

    auto [DestBB, InsertPt] = findEarliestSafePoint(*SetupMI, Setups);
    if (InsertPt != SetupMI->getIterator()) {
      DestBB->splice(InsertPt, SetupMI->getParent(), SetupMI);
      ++NumBMOVHoisted;
      Changed = true;
    }
  }

  return Changed;
}

static bool hasRegisterDependency(const MachineInstr &MI,
                                  const MachineInstr &SetupMI,
                                  const TargetRegisterInfo &TRI) {
  for (const MachineOperand &MO : SetupMI.all_uses()) {
    if (MO.isReg() && MO.getReg() && MI.modifiesRegister(MO.getReg(), &TRI))
      return true;
  }

  for (const MachineOperand &MO : SetupMI.all_defs()) {
    if (!MO.isReg() || !MO.getReg())
      continue;
    if (MI.modifiesRegister(MO.getReg(), &TRI) ||
        MI.readsRegister(MO.getReg(), &TRI))
      return true;
  }

  return false;
}

static bool isHoistHazard(const MachineInstr &MI, const MachineInstr &SetupMI,
                          const TargetRegisterInfo &TRI) {
  return isHoistBarrier(MI) || hasRegisterDependency(MI, SetupMI, TRI);
}

/// Scan `BB` backwards from `From`, returning the earliest safe
/// insertion point. `HazardFound` is set if we stopped early.
/// Caller guarantees From != BB->rend().
static MachineBasicBlock::iterator
scanBlockBackward(MachineBasicBlock *BB,
                  MachineBasicBlock::reverse_iterator From,
                  const MachineInstr &SetupMI, const TargetRegisterInfo &TRI,
                  bool &HazardFound) {
  HazardFound = false;

  // Try to hoist to the top of the func.
  MachineBasicBlock::iterator SafePoint = BB->getFirstNonPHI();

  for (auto I = From, E = BB->rend(); I != E; ++I) {
    MachineInstr &CurrMI = *I;

    if (CurrMI.isPHI())
      break;

    if (isHoistHazard(CurrMI, SetupMI, TRI)) {
      LLVM_DEBUG(dbgs() << "  [Hazard] Cannot cross: " << CurrMI);
      LLVM_DEBUG(if (CurrMI.isCall()) dbgs() << "    - reason: Call");
      LLVM_DEBUG(if (CurrMI.isIndirectBranch()) dbgs()
                 << "    - reason: Indirect Branch");
      LLVM_DEBUG(if (CurrMI.isInlineAsm()) dbgs()
                 << "    - reason: Inline Assembly");
      LLVM_DEBUG(if (CurrMI.hasUnmodeledSideEffects()) dbgs()
                 << "    - reason: Has Side Effects");
      LLVM_DEBUG(dbgs() << " ; Cannot cross: " << CurrMI);
      HazardFound = true;
      return std::next(CurrMI.getIterator());
    }

    SafePoint = CurrMI.getIterator();
  }

  return SafePoint;
}

bool RISCVBranchSetupHoisting::isSafeToHoistTo(
    const MachineInstr &SetupMI, MachineBasicBlock &DestBB,
    MachineBasicBlock::const_iterator InsertPt) const {
  const MachineBasicBlock *SrcBB = SetupMI.getParent();
  if (SrcBB == &DestBB || !MDT->dominates(&DestBB, SrcBB))
    return false;

  auto rangeHasHazard = [&](MachineBasicBlock::const_iterator Begin,
                            MachineBasicBlock::const_iterator End) {
    return llvm::any_of(
        llvm::make_range(Begin, End), [&](const MachineInstr &MI) {
          return &MI != &SetupMI && isHoistHazard(MI, SetupMI, *TRI);
        });
  };

  // Walk the inverse CFG from the source to DestBB. Stopping at DestBB avoids
  // confusing a loop backedge with code crossed by the same dynamic setup.
  // Every predecessor retained here is dominated by DestBB, so it lies on a
  // realizable DestBB-to-SrcBB path rather than before the hoist point.
  SmallPtrSet<const MachineBasicBlock *, 32> Visited;
  SmallVector<const MachineBasicBlock *, 16> WorkList;
  WorkList.push_back(SrcBB);
  bool ReachedDest = false;

  while (!WorkList.empty()) {
    const MachineBasicBlock *BB = WorkList.pop_back_val();
    if (!Visited.insert(BB).second)
      continue;

    if (BB == &DestBB) {
      ReachedDest = true;
      if (rangeHasHazard(InsertPt, DestBB.end()))
        return false;
      continue;
    }

    MachineBasicBlock::const_iterator End = BB->end();
    if (BB == SrcBB)
      End = SetupMI.getIterator();
    if (rangeHasHazard(BB->begin(), End))
      return false;

    for (const MachineBasicBlock *Pred : BB->predecessors()) {
      if (Pred == &DestBB || MDT->dominates(&DestBB, Pred))
        WorkList.push_back(Pred);
    }
  }

  return ReachedDest;
}

static bool loopHasHazard(const MachineLoop &L, const MachineInstr &SetupMI,
                          const TargetRegisterInfo &TRI,
                          SmallSet<MachineInstr *, 4> Setup) {
  Register BR = SetupMI.getOperand(0).getReg();
  for (MachineBasicBlock *MBB : L.getBlocks()) {
    for (const MachineInstr &MI : *MBB) {
      if (MI.isBMOV()) {
        if (MI.getOperand(0).getReg() == BR)
          continue;
      }
      // if (Setup.contains(&MI)) {
      //   dbgs() << "  Skipping MI: " << MI;
      //   continue;
      // }
      bool IsHazard = isHoistHazard(MI, SetupMI, TRI);
      LLVM_DEBUG(dbgs() << "  Checking loop MI (isHazard=" << IsHazard << "): " << MI);
      if (IsHazard) {
        return true;
      }
    }
    LLVM_DEBUG(dbgs() << "  No loop hazard in BB#" << MBB->getNumber() << "\n");
  }

  return false;
}

std::pair<MachineBasicBlock *, MachineBasicBlock::iterator>
RISCVBranchSetupHoisting::findEarliestSafePoint(
    MachineInstr &SetupMI, SmallSet<MachineInstr *, 4> Setup) {

  MachineBasicBlock *CurBB = SetupMI.getParent();
  MachineBasicBlock::iterator SafePoint = SetupMI.getIterator();

  bool HazardFound = false;

  auto IntraStart = std::next(SetupMI.getReverseIterator());
  if (IntraStart != CurBB->rend())
    SafePoint =
        scanBlockBackward(CurBB, IntraStart, SetupMI, *TRI, HazardFound);
  else
    SafePoint = CurBB->getFirstNonPHI();

  if (HazardFound)
    return {CurBB, SafePoint};

  MachineLoop *SetupLoop = MLI->getLoopFor(CurBB);
  MachineBasicBlock *BB = CurBB;

  while (true) {
    MachineDomTreeNode *Node = MDT->getNode(BB);
    if (!Node)
      break;

    MachineDomTreeNode *IDomNode = Node->getIDom();
    if (!IDomNode)
      break;

    MachineBasicBlock *IDom = IDomNode->getBlock();
    if (!IDom || DisableCrossBBHoisting)
      break;

    // Don't hoist across loop boundaries.
#if 0
    MachineLoop *IDomLoop = MLI->getLoopFor(IDom);
    if (IDomLoop != SetupLoop) {
      LLVM_DEBUG(dbgs() << "  [Stop] Loop boundary at BB#" << IDom->getNumber()
                        << "\n");
      break;
    }
#endif

    MachineBasicBlock::reverse_iterator IDomScanStart = IDom->rbegin();
    while (IDomScanStart != IDom->rend() && IDomScanStart->isTerminator())
      ++IDomScanStart;

    MachineBasicBlock::iterator IDomSafePoint;
    if (IDomScanStart == IDom->rend()) {
      IDomSafePoint = IDom->getFirstTerminator();
      HazardFound = false;
    } else {
      IDomSafePoint =
          scanBlockBackward(IDom, IDomScanStart, SetupMI, *TRI, HazardFound);
    }

    // The dominator tree chooses candidates; the inverse-CFG walk proves that
    // every actual path from the candidate to SetupMI is free of hazards.
    if (!isSafeToHoistTo(SetupMI, *IDom, IDomSafePoint)) {
      LLVM_DEBUG(dbgs() << "  [Stop] Hazard on a CFG path from BB#"
                        << CurBB->getNumber() << " to BB#" << IDom->getNumber()
                        << "\n");
      break;
    }

    for (MachineLoop *L = MLI->getLoopFor(BB); L && !L->contains(IDom);
         L = L->getParentLoop()) {
      LLVM_DEBUG(dbgs() << "  Checking loop hazard...\n");
      if (loopHasHazard(*L, SetupMI, *TRI, Setup)) {
        goto EndSearch;
      }
    }

    LLVM_DEBUG(dbgs() << "  [Cross-BB] Hoisted into BB#" << IDom->getNumber()
                      << "\n");
    SafePoint = IDomSafePoint;
    BB = IDom;

    if (HazardFound)
      break;
  }
EndSearch:

  return {BB, SafePoint};
}

INITIALIZE_PASS(RISCVBranchSetupHoisting, DEBUG_TYPE, PASS_NAME,
                false, // is CFG only?
                false  // is analysis?
)

namespace llvm {

FunctionPass *createRISCVBranchSetupHoistingPass() {
  return new RISCVBranchSetupHoisting();
}

} // namespace llvm
