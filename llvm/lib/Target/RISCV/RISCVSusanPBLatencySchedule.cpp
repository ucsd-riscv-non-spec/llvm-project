//===-- RISCVSusanPBLatencySchedule.cpp - Schedule SUSAN PB latency
//--------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVInstrInfo.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Registry.h"

#define DEBUG_TYPE "riscv-susan-pb-latency-schedule"

using namespace llvm;

static cl::opt<bool> EnableSusanPBLatencySchedule(
    "riscv-ns-susan-pb-latency-schedule", cl::Hidden, cl::init(false),
    cl::desc("Move one independent instruction into a SUSAN BMOVC-to-PB gap"));

namespace {

class RISCVSusanPBLatencySchedule : public MachineFunctionPass {
public:
  static char ID;

  RISCVSusanPBLatencySchedule() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "RISC-V SUSAN PB latency scheduling";
  }
};

char RISCVSusanPBLatencySchedule::ID = 0;

static MachineInstr *previousNonDebug(MachineInstr &MI,
                                      MachineBasicBlock &MBB) {
  auto It = MI.getIterator();
  while (It != MBB.begin()) {
    --It;
    if (!It->isDebugInstr())
      return &*It;
  }
  return nullptr;
}

static MachineInstr *nextNonDebug(MachineInstr &MI, MachineBasicBlock &MBB) {
  auto It = std::next(MI.getIterator());
  for (auto End = MBB.end(); It != End; ++It)
    if (!It->isDebugInstr())
      return &*It;
  return nullptr;
}

static bool isOneInstructionLI(const MachineInstr &MI) {
  return MI.getOpcode() == RISCV::ADDI && MI.getNumExplicitOperands() == 3 &&
         MI.getOperand(0).isReg() && MI.getOperand(0).isDef() &&
         MI.getOperand(1).isReg() && MI.getOperand(1).getReg() == RISCV::X0 &&
         MI.getOperand(2).isImm() && MI.getOperand(2).getImm() == 1;
}

static bool isPBFor(const MachineInstr &MI, Register BReg) {
  if (MI.getOpcode() == RISCV::PseudoPBC)
    return MI.getOperand(0).isReg() && MI.getOperand(0).getReg() == BReg;
  if (MI.getOpcode() == RISCV::PBAL)
    return MI.getOperand(1).isReg() && MI.getOperand(1).getReg() == BReg;
  return false;
}

static bool isIndependentOfCondition(const MachineInstr &LI,
                                     const MachineInstr &BMOVC,
                                     const MachineInstr &PB,
                                     const TargetRegisterInfo &TRI) {
  if (LI.mayLoad() || LI.mayStore() || LI.isCall() || LI.isBranch() ||
      LI.isTerminator() || LI.hasUnmodeledSideEffects())
    return false;

  Register Def = LI.getOperand(0).getReg();
  for (const MachineOperand &MO : BMOVC.operands())
    if (MO.isReg() && TRI.regsOverlap(Def, MO.getReg()))
      return false;
  for (const MachineOperand &MO : PB.operands())
    if (MO.isReg() && TRI.regsOverlap(Def, MO.getReg()))
      return false;
  return true;
}

bool RISCVSusanPBLatencySchedule::runOnMachineFunction(MachineFunction &MF) {
  if (!EnableSusanPBLatencySchedule || MF.getName() != "susan_corners")
    return false;

  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  SmallVector<std::pair<MachineBasicBlock *, MachineInstr *>, 1> Candidates;

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &BMOVC : MBB) {
      if (BMOVC.getOpcode() != RISCV::BMOVC_BGEU ||
          BMOVC.getNumExplicitOperands() < 4 || !BMOVC.getOperand(0).isReg() ||
          !BMOVC.getOperand(2).isReg())
        continue;

      MachineInstr *PB = nextNonDebug(BMOVC, MBB);
      MachineInstr *LI = previousNonDebug(BMOVC, MBB);
      MachineInstr *Sum = LI ? previousNonDebug(*LI, MBB) : nullptr;
      bool MatchesPB = PB && isPBFor(*PB, BMOVC.getOperand(0).getReg());
      bool MatchesLI = LI && isOneInstructionLI(*LI);
      bool MatchesSum =
          Sum && Sum->getOpcode() == RISCV::ADDW &&
          Sum->getOperand(0).isReg() &&
          Sum->getOperand(0).getReg() == BMOVC.getOperand(2).getReg();
      bool Independent =
          LI && PB && isIndependentOfCondition(*LI, BMOVC, *PB, *TRI);
      if (!MatchesPB || !MatchesLI || !MatchesSum || !Independent)
        continue;

      Candidates.emplace_back(&MBB, LI);
    }
  }

  LLVM_DEBUG(dbgs() << "Found " << Candidates.size()
                    << " SUSAN PB scheduling candidates\n");
  if (Candidates.size() != 1)
    return false;

  MachineBasicBlock *MBB = Candidates.front().first;
  MachineInstr *LI = Candidates.front().second;
  MachineInstr *BMOVC = nextNonDebug(*LI, *MBB);
  assert(BMOVC && BMOVC->getOpcode() == RISCV::BMOVC_BGEU);
  MBB->splice(std::next(BMOVC->getIterator()), MBB, LI->getIterator());

  LLVM_DEBUG(dbgs() << "Moved independent SUSAN instruction after BMOVC:\n"
                    << *LI << "before\n"
                    << *BMOVC);
  return true;
}

} // namespace

INITIALIZE_PASS(RISCVSusanPBLatencySchedule, DEBUG_TYPE,
                "RISC-V SUSAN PB latency scheduling", false, false)

namespace llvm {
FunctionPass *createRISCVSusanPBLatencySchedulePass() {
  return new RISCVSusanPBLatencySchedule();
}
} // namespace llvm
