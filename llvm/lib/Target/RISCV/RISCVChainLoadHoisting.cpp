//===-- RISCVChainLoadHoisting.cpp - Hoist checked chain loads ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVInstrInfo.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "riscv-chain-load-hoisting"

using namespace llvm;

namespace {

class RISCVChainLoadHoistingMachine : public MachineFunctionPass {
public:
  static char ID;
  RISCVChainLoadHoistingMachine() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineDominatorTreeWrapperPass>();
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

char RISCVChainLoadHoistingMachine::ID = 0;

static bool isOnApprovedPath(MachineBasicBlock *BB, MachineBasicBlock *Dest,
                             MachineBasicBlock *Source,
                             MachineDominatorTree &MDT) {
  return BB != Dest && MDT.dominates(Dest, BB) && MDT.dominates(BB, Source);
}

static bool isSimpleLoad(const MachineInstr &MI) {
  if (!MI.mayLoad() || MI.mayStore() || MI.memoperands_empty())
    return false;
  for (const MachineMemOperand *MMO : MI.memoperands())
    if (MMO->isVolatile() || MMO->isAtomic())
      return false;
  return true;
}

static MachineInstr *findConditionalBranch(MachineBasicBlock &MBB) {
  for (MachineInstr &MI : MBB)
    if (MI.isConditionalBranch())
      return &MI;
  return nullptr;
}

bool RISCVChainLoadHoistingMachine::runOnMachineFunction(MachineFunction &MF) {
  MachineDominatorTree &MDT =
      getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  MachineRegisterInfo &MRI = MF.getRegInfo();

  DenseMap<MachineBasicBlock *, SmallPtrSet<MachineInstr *, 32>>
      MovesByDestination;
  for (MachineBasicBlock &Source : MF) {
    const BasicBlock *SourceIR = Source.getBasicBlock();
    if (!SourceIR || !SourceIR->getTerminator())
      continue;
    MDNode *Marker =
        SourceIR->getTerminator()->getMetadata("riscv.ns.chain-hoist");
    if (!Marker || Marker->getNumOperands() != 1)
      continue;
    auto *DestMetadata = dyn_cast<MDString>(Marker->getOperand(0));
    if (!DestMetadata)
      continue;

    unsigned Distance = 0;
    if (DestMetadata->getString().getAsInteger(10, Distance) || !Distance)
      continue;

    MachineBasicBlock *Dest = &Source;
    bool ValidPath = true;
    for (unsigned I = 0; I < Distance; ++I) {
      MachineBasicBlock *Predecessor = nullptr;
      for (MachineBasicBlock *Pred : Dest->predecessors()) {
        if (!findConditionalBranch(*Pred) ||
            !MDT.dominates(Pred, &Source))
          continue;
        if (Predecessor) {
          ValidPath = false;
          break;
        }
        Predecessor = Pred;
      }
      if (!ValidPath || !Predecessor) {
        ValidPath = false;
        break;
      }
      Dest = Predecessor;
    }
    if (!ValidPath)
      continue;
    if (!Dest || Dest == &Source ||
        !MDT.dominates(Dest, &Source))
      continue;

    MachineInstr *SourceBranch = findConditionalBranch(Source);
    if (!SourceBranch)
      continue;

    MachineInstr *DestBranch = findConditionalBranch(*Dest);
    if (!DestBranch)
      continue;

    SmallPtrSet<MachineBasicBlock *, 8> Segment;
    ValidPath = true;
    for (MachineBasicBlock &BB : MF) {
      if (&BB != &Source &&
          isOnApprovedPath(&BB, Dest, &Source, MDT))
        Segment.insert(&BB);
    }
    Segment.insert(&Source);

    SmallPtrSet<MachineInstr *, 32> Slice;
    SmallPtrSet<MachineInstr *, 32> VisitingValues;
    SmallPtrSet<MachineInstr *, 32> VisitingAddresses;

    auto CollectAddress = [&](auto &&Self, Register Reg) -> bool {
      if (!Reg.isVirtual())
        return true;
      MachineInstr *Def = MRI.getVRegDef(Reg);
      if (!Def || !Segment.contains(Def->getParent()))
        return true;
      if (!VisitingAddresses.insert(Def).second)
        return true;
      if (Def->mayLoad()) {
        if (!isSimpleLoad(*Def))
          return false;
        Slice.insert(Def);
      } else {
        if (Def->mayStore() || Def->isCall() || Def->isTerminator() ||
            Def->hasUnmodeledSideEffects() || Def->isPHI())
          return false;
        Slice.insert(Def);
      }
      for (const MachineOperand &MO : Def->operands())
        if (MO.isReg() && MO.isUse() && MO.getReg().isVirtual())
          if (!Self(Self, MO.getReg()))
            return false;
      return true;
    };

    auto CollectValue = [&](auto &&Self, Register Reg) -> bool {
      if (!Reg.isVirtual())
        return true;
      MachineInstr *Def = MRI.getVRegDef(Reg);
      if (!Def || !Segment.contains(Def->getParent()) ||
          !VisitingValues.insert(Def).second)
        return true;
      if (Def->isPHI())
        return true;
      if (Def->mayLoad()) {
        if (!isSimpleLoad(*Def))
          return false;
        Slice.insert(Def);
        for (const MachineOperand &MO : Def->operands())
          if (MO.isReg() && MO.isUse() && MO.getReg().isVirtual())
            if (!CollectAddress(CollectAddress, MO.getReg()))
              return false;
        return true;
      }
      if (Def->mayStore() || Def->isCall() || Def->isTerminator() ||
          Def->hasUnmodeledSideEffects())
        return false;
      for (const MachineOperand &MO : Def->operands())
        if (MO.isReg() && MO.isUse() && MO.getReg().isVirtual())
          if (!Self(Self, MO.getReg()))
            return false;
      return true;
    };

    for (const MachineOperand &MO : SourceBranch->operands())
      if (MO.isReg() && MO.isUse() && MO.getReg().isVirtual())
        if (!CollectValue(CollectValue, MO.getReg())) {
          ValidPath = false;
          break;
        }
    if (!ValidPath || Slice.empty())
      continue;

    SmallPtrSet<MachineInstr *, 32> &MoveSet = MovesByDestination[Dest];
    for (MachineInstr *MI : Slice)
      MoveSet.insert(MI);
    LLVM_DEBUG(dbgs() << "Recovered approved chain load slice from "
                      << Source.getName() << " for destination "
                      << Dest->getName() << ": " << Slice.size()
                      << " instructions\n");
  }

  bool Changed = false;
  for (auto &Entry : MovesByDestination) {
    MachineBasicBlock *Dest = Entry.first;
    MachineInstr *DestBranch = findConditionalBranch(*Dest);
    if (!DestBranch)
      continue;

    SmallPtrSet<MachineInstr *, 32> Visited;
    SmallVector<MachineInstr *, 32> Ordered;
    auto Visit = [&](auto &&Self, MachineInstr *MI) -> void {
      if (!Visited.insert(MI).second)
        return;
      for (const MachineOperand &MO : MI->operands()) {
        if (!MO.isReg() || !MO.isUse() || !MO.getReg().isVirtual())
          continue;
        MachineInstr *Def = MRI.getVRegDef(MO.getReg());
        if (Def && Entry.second.contains(Def))
          Self(Self, Def);
      }
      Ordered.push_back(MI);
    };
    for (MachineInstr *MI : Entry.second)
      Visit(Visit, MI);
    for (MachineInstr *MI : Ordered) {
      if (MI->getParent() == Dest)
        continue;
      Dest->splice(DestBranch->getIterator(), MI->getParent(),
                   MI->getIterator());
      Changed = true;
    }
  }
  return Changed;
}

} // namespace

INITIALIZE_PASS_BEGIN(RISCVChainLoadHoistingMachine,
                      "riscv-chain-load-hoisting-machine",
                      "RISC-V chain-load hoisting after machine sinking", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_END(RISCVChainLoadHoistingMachine,
                    "riscv-chain-load-hoisting-machine",
                    "RISC-V chain-load hoisting after machine sinking", false,
                    false)

FunctionPass *llvm::createRISCVChainLoadHoistingMachinePass() {
  return new RISCVChainLoadHoistingMachine();
}
