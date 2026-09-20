//===-- RISCVExpandPseudoInsts.cpp - Expand pseudo instructions -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a pass that expands pseudo instructions into target
// instructions. This pass should be run after register allocation but before
// the post-regalloc scheduling pass.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVInstrInfo.h"
#include "RISCVTargetMachine.h"

#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/MC/MCContext.h"

using namespace llvm;

#define RISCV_EXPAND_PSEUDO_NAME "RISC-V pseudo instruction expansion pass"
#define RISCV_PRERA_EXPAND_PSEUDO_NAME "RISC-V Pre-RA pseudo instruction expansion pass"

namespace {

class RISCVExpandPseudo : public MachineFunctionPass {
public:
  const RISCVSubtarget *STI;
  const RISCVInstrInfo *TII;
  const MachineLoopInfo *MLI;
  static char ID;

  RISCVExpandPseudo() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    MachineFunctionPass::getAnalysisUsage(AU);
    AU.addRequired<MachineLoopInfoWrapperPass>();
    AU.setPreservesAll();
  }

  StringRef getPassName() const override { return RISCV_EXPAND_PSEUDO_NAME; }

  MachineFunctionProperties getClearedProperties() const override {
    return MachineFunctionProperties().setNoVRegs();
  }

private:
  bool expandMBB(MachineBasicBlock &MBB);
  bool expandMI(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
                MachineBasicBlock::iterator &NextMBBI);
  bool expandCondBranch(MachineBasicBlock &MBB,
                        MachineBasicBlock::iterator MBBI);
  bool expandBranch(MachineBasicBlock &MBB,
                    MachineBasicBlock::iterator MBBI);
  bool expandCall(MachineBasicBlock &MBB,
                  MachineBasicBlock::iterator MBBI);
  bool expandReturn(MachineBasicBlock &MBB,
                    MachineBasicBlock::iterator MBBI);
  bool expandIndirect(MachineBasicBlock &MBB,
                      MachineBasicBlock::iterator MBBI, bool IsCall) const;
  bool expandLoopEnd(MachineBasicBlock &MBB,
                     MachineBasicBlock::iterator MBBI);
  bool expandCCOp(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
                  MachineBasicBlock::iterator &NextMBBI);
  bool expandVMSET_VMCLR(MachineBasicBlock &MBB,
                         MachineBasicBlock::iterator MBBI, unsigned Opcode);
  bool expandMV_FPR16INX(MachineBasicBlock &MBB,
                         MachineBasicBlock::iterator MBBI);
  bool expandMV_FPR32INX(MachineBasicBlock &MBB,
                         MachineBasicBlock::iterator MBBI);
  bool expandRV32ZdinxStore(MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator MBBI);
  bool expandRV32ZdinxLoad(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator MBBI);
  bool expandPseudoReadVLENBViaVSETVLIX0(MachineBasicBlock &MBB,
                                         MachineBasicBlock::iterator MBBI);
#ifndef NDEBUG
  unsigned getInstSizeInBytes(const MachineFunction &MF) const {
    unsigned Size = 0;
    for (auto &MBB : MF)
      for (auto &MI : MBB)
        Size += TII->getInstSizeInBytes(MI);
    return Size;
  }
#endif
};

char RISCVExpandPseudo::ID = 0;

bool RISCVExpandPseudo::runOnMachineFunction(MachineFunction &MF) {
  STI = &MF.getSubtarget<RISCVSubtarget>();
  TII = STI->getInstrInfo();
  MLI = &getAnalysis<MachineLoopInfoWrapperPass>().getLI();

#ifndef NDEBUG
  const unsigned OldSize = getInstSizeInBytes(MF);
#endif

  bool Modified = false;
  for (auto &MBB : MF)
    Modified |= expandMBB(MBB);

#ifndef NDEBUG
  const unsigned NewSize = getInstSizeInBytes(MF);
  assert(OldSize >= NewSize);
#endif
  return Modified;
}

bool RISCVExpandPseudo::expandMBB(MachineBasicBlock &MBB) {
  bool Modified = false;

  MachineBasicBlock::iterator MBBI = MBB.begin(), E = MBB.end();
  while (MBBI != E) {
    MachineBasicBlock::iterator NMBBI = std::next(MBBI);
    Modified |= expandMI(MBB, MBBI, NMBBI);
    MBBI = NMBBI;
  }

  return Modified;
}

bool RISCVExpandPseudo::expandMI(MachineBasicBlock &MBB,
                                 MachineBasicBlock::iterator MBBI,
                                 MachineBasicBlock::iterator &NextMBBI) {
  // RISCVInstrInfo::getInstSizeInBytes expects that the total size of the
  // expanded instructions for each pseudo is correct in the Size field of the
  // tablegen definition for the pseudo.
  switch (MBBI->getOpcode()) {
  case RISCV::PseudoBR:
    return expandBranch(MBB, MBBI);
  case RISCV::PseudoBREQ:
  case RISCV::PseudoBRNE:
  case RISCV::PseudoBRGE:
  case RISCV::PseudoBRGEU:
  case RISCV::PseudoBRLT:
  case RISCV::PseudoBRLTU:
    return expandCondBranch(MBB, MBBI);
  case RISCV::PseudoCALLReg:
  case RISCV::PseudoCALL:
  case RISCV::PseudoTAIL:
  case RISCV::PseudoJump:
    return expandCall(MBB, MBBI);
  case RISCV::PseudoRET:
    return expandReturn(MBB, MBBI);
  case RISCV::PseudoBRIND:
  case RISCV::PseudoBRINDX7:
  case RISCV::PseudoBRINDNonX7:
  case RISCV::PseudoTAILIndirect:
  case RISCV::PseudoTAILIndirectX7:
  case RISCV::PseudoTAILIndirectNonX7:
    return expandIndirect(MBB, MBBI, /*IsCall=*/false);
  case RISCV::PseudoCALLIndirect:
  case RISCV::PseudoCALLIndirectX7:
  case RISCV::PseudoCALLIndirectNonX7:
    return expandIndirect(MBB, MBBI, /*IsCall=*/true);
  case RISCV::PseudoLoopEnd:
    return expandLoopEnd(MBB, MBBI);
  case RISCV::PseudoMV_FPR16INX:
    return expandMV_FPR16INX(MBB, MBBI);
  case RISCV::PseudoMV_FPR32INX:
    return expandMV_FPR32INX(MBB, MBBI);
  case RISCV::PseudoRV32ZdinxSD:
    return expandRV32ZdinxStore(MBB, MBBI);
  case RISCV::PseudoRV32ZdinxLD:
    return expandRV32ZdinxLoad(MBB, MBBI);
  case RISCV::PseudoCCMOVGPRNoX0:
  case RISCV::PseudoCCMOVGPR:
  case RISCV::PseudoCCADD:
  case RISCV::PseudoCCSUB:
  case RISCV::PseudoCCAND:
  case RISCV::PseudoCCOR:
  case RISCV::PseudoCCXOR:
  case RISCV::PseudoCCADDW:
  case RISCV::PseudoCCSUBW:
  case RISCV::PseudoCCSLL:
  case RISCV::PseudoCCSRL:
  case RISCV::PseudoCCSRA:
  case RISCV::PseudoCCADDI:
  case RISCV::PseudoCCSLLI:
  case RISCV::PseudoCCSRLI:
  case RISCV::PseudoCCSRAI:
  case RISCV::PseudoCCANDI:
  case RISCV::PseudoCCORI:
  case RISCV::PseudoCCXORI:
  case RISCV::PseudoCCSLLW:
  case RISCV::PseudoCCSRLW:
  case RISCV::PseudoCCSRAW:
  case RISCV::PseudoCCADDIW:
  case RISCV::PseudoCCSLLIW:
  case RISCV::PseudoCCSRLIW:
  case RISCV::PseudoCCSRAIW:
  case RISCV::PseudoCCANDN:
  case RISCV::PseudoCCORN:
  case RISCV::PseudoCCXNOR:
  case RISCV::PseudoCCNDS_BFOS:
  case RISCV::PseudoCCNDS_BFOZ:
    return expandCCOp(MBB, MBBI, NextMBBI);
  case RISCV::PseudoVMCLR_M_B1:
  case RISCV::PseudoVMCLR_M_B2:
  case RISCV::PseudoVMCLR_M_B4:
  case RISCV::PseudoVMCLR_M_B8:
  case RISCV::PseudoVMCLR_M_B16:
  case RISCV::PseudoVMCLR_M_B32:
  case RISCV::PseudoVMCLR_M_B64:
    // vmclr.m vd => vmxor.mm vd, vd, vd
    return expandVMSET_VMCLR(MBB, MBBI, RISCV::VMXOR_MM);
  case RISCV::PseudoVMSET_M_B1:
  case RISCV::PseudoVMSET_M_B2:
  case RISCV::PseudoVMSET_M_B4:
  case RISCV::PseudoVMSET_M_B8:
  case RISCV::PseudoVMSET_M_B16:
  case RISCV::PseudoVMSET_M_B32:
  case RISCV::PseudoVMSET_M_B64:
    // vmset.m vd => vmxnor.mm vd, vd, vd
    return expandVMSET_VMCLR(MBB, MBBI, RISCV::VMXNOR_MM);
  case RISCV::PseudoReadVLENBViaVSETVLIX0:
    return expandPseudoReadVLENBViaVSETVLIX0(MBB, MBBI);
  }

  return false;
}

bool RISCVExpandPseudo::expandCCOp(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator MBBI,
                                   MachineBasicBlock::iterator &NextMBBI) {

  MachineFunction *MF = MBB.getParent();
  MachineInstr &MI = *MBBI;
  DebugLoc DL = MI.getDebugLoc();

  MachineBasicBlock *TrueBB = MF->CreateMachineBasicBlock(MBB.getBasicBlock());
  MachineBasicBlock *MergeBB = MF->CreateMachineBasicBlock(MBB.getBasicBlock());

  MF->insert(++MBB.getIterator(), TrueBB);
  MF->insert(++TrueBB->getIterator(), MergeBB);

  // We want to copy the "true" value when the condition is true which means
  // we need to invert the branch condition to jump over TrueBB when the
  // condition is false.
  auto CC = static_cast<RISCVCC::CondCode>(MI.getOperand(3).getImm());
  CC = RISCVCC::getOppositeBranchCondition(CC);

  // Insert branch instruction.
  BuildMI(MBB, MBBI, DL, TII->get(RISCVCC::getBrCond(CC)))
      .addReg(MI.getOperand(1).getReg())
      .addReg(MI.getOperand(2).getReg())
      .addMBB(MergeBB);

  Register DestReg = MI.getOperand(0).getReg();
  assert(MI.getOperand(4).getReg() == DestReg);

  if (MI.getOpcode() == RISCV::PseudoCCMOVGPR ||
      MI.getOpcode() == RISCV::PseudoCCMOVGPRNoX0) {
    // Add MV.
    BuildMI(TrueBB, DL, TII->get(RISCV::ADDI), DestReg)
        .add(MI.getOperand(5))
        .addImm(0);
  } else {
    unsigned NewOpc;
    switch (MI.getOpcode()) {
    default:
      llvm_unreachable("Unexpected opcode!");
    case RISCV::PseudoCCADD:   NewOpc = RISCV::ADD;   break;
    case RISCV::PseudoCCSUB:   NewOpc = RISCV::SUB;   break;
    case RISCV::PseudoCCSLL:   NewOpc = RISCV::SLL;   break;
    case RISCV::PseudoCCSRL:   NewOpc = RISCV::SRL;   break;
    case RISCV::PseudoCCSRA:   NewOpc = RISCV::SRA;   break;
    case RISCV::PseudoCCAND:   NewOpc = RISCV::AND;   break;
    case RISCV::PseudoCCOR:    NewOpc = RISCV::OR;    break;
    case RISCV::PseudoCCXOR:   NewOpc = RISCV::XOR;   break;
    case RISCV::PseudoCCADDI:  NewOpc = RISCV::ADDI;  break;
    case RISCV::PseudoCCSLLI:  NewOpc = RISCV::SLLI;  break;
    case RISCV::PseudoCCSRLI:  NewOpc = RISCV::SRLI;  break;
    case RISCV::PseudoCCSRAI:  NewOpc = RISCV::SRAI;  break;
    case RISCV::PseudoCCANDI:  NewOpc = RISCV::ANDI;  break;
    case RISCV::PseudoCCORI:   NewOpc = RISCV::ORI;   break;
    case RISCV::PseudoCCXORI:  NewOpc = RISCV::XORI;  break;
    case RISCV::PseudoCCADDW:  NewOpc = RISCV::ADDW;  break;
    case RISCV::PseudoCCSUBW:  NewOpc = RISCV::SUBW;  break;
    case RISCV::PseudoCCSLLW:  NewOpc = RISCV::SLLW;  break;
    case RISCV::PseudoCCSRLW:  NewOpc = RISCV::SRLW;  break;
    case RISCV::PseudoCCSRAW:  NewOpc = RISCV::SRAW;  break;
    case RISCV::PseudoCCADDIW: NewOpc = RISCV::ADDIW; break;
    case RISCV::PseudoCCSLLIW: NewOpc = RISCV::SLLIW; break;
    case RISCV::PseudoCCSRLIW: NewOpc = RISCV::SRLIW; break;
    case RISCV::PseudoCCSRAIW: NewOpc = RISCV::SRAIW; break;
    case RISCV::PseudoCCANDN:  NewOpc = RISCV::ANDN;  break;
    case RISCV::PseudoCCORN:   NewOpc = RISCV::ORN;   break;
    case RISCV::PseudoCCXNOR:  NewOpc = RISCV::XNOR;  break;
    case RISCV::PseudoCCNDS_BFOS: NewOpc = RISCV::NDS_BFOS; break;
    case RISCV::PseudoCCNDS_BFOZ: NewOpc = RISCV::NDS_BFOZ; break;
    }

    if (NewOpc == RISCV::NDS_BFOZ || NewOpc == RISCV::NDS_BFOS) {
      BuildMI(TrueBB, DL, TII->get(NewOpc), DestReg)
          .add(MI.getOperand(5))
          .add(MI.getOperand(6))
          .add(MI.getOperand(7));
    } else {
      BuildMI(TrueBB, DL, TII->get(NewOpc), DestReg)
          .add(MI.getOperand(5))
          .add(MI.getOperand(6));
    }
  }

  TrueBB->addSuccessor(MergeBB);

  MergeBB->splice(MergeBB->end(), &MBB, MI, MBB.end());
  MergeBB->transferSuccessors(&MBB);

  MBB.addSuccessor(TrueBB);
  MBB.addSuccessor(MergeBB);

  NextMBBI = MBB.end();
  MI.eraseFromParent();

  // Make sure live-ins are correctly attached to this new basic block.
  LivePhysRegs LiveRegs;
  computeAndAddLiveIns(LiveRegs, *TrueBB);
  computeAndAddLiveIns(LiveRegs, *MergeBB);

  return true;
}

bool RISCVExpandPseudo::expandVMSET_VMCLR(MachineBasicBlock &MBB,
                                          MachineBasicBlock::iterator MBBI,
                                          unsigned Opcode) {
  DebugLoc DL = MBBI->getDebugLoc();
  Register DstReg = MBBI->getOperand(0).getReg();
  const MCInstrDesc &Desc = TII->get(Opcode);
  BuildMI(MBB, MBBI, DL, Desc, DstReg)
      .addReg(DstReg, RegState::Undef)
      .addReg(DstReg, RegState::Undef);
  MBBI->eraseFromParent(); // The pseudo instruction is gone now.
  return true;
}

bool RISCVExpandPseudo::expandMV_FPR16INX(MachineBasicBlock &MBB,
                                          MachineBasicBlock::iterator MBBI) {
  DebugLoc DL = MBBI->getDebugLoc();
  const TargetRegisterInfo *TRI = STI->getRegisterInfo();
  Register DstReg = TRI->getMatchingSuperReg(
      MBBI->getOperand(0).getReg(), RISCV::sub_16, &RISCV::GPRRegClass);
  Register SrcReg = TRI->getMatchingSuperReg(
      MBBI->getOperand(1).getReg(), RISCV::sub_16, &RISCV::GPRRegClass);

  BuildMI(MBB, MBBI, DL, TII->get(RISCV::ADDI), DstReg)
      .addReg(SrcReg, getKillRegState(MBBI->getOperand(1).isKill()))
      .addImm(0);

  MBBI->eraseFromParent(); // The pseudo instruction is gone now.
  return true;
}

bool RISCVExpandPseudo::expandBranch(MachineBasicBlock &MBB,
  MachineBasicBlock::iterator MBBI) {
  MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();
  Register BReg = MRI.createVirtualRegister(&RISCV::PBRRegClass);
  MachineBasicBlock *TBB = MBBI->getOperand(0).getMBB();
  DebugLoc DL = MBBI->getDebugLoc();
  MCContext &Context = MBB.getParent()->getContext();
  MCSymbol *Sym = Context.createTempSymbol("ns_j_");

  // Emit BMOVS B0, ns_j_
  BuildMI(MBB, MBBI, DL, TII->get(RISCV::BMOVS_J))
      .addDef(BReg)
      .addSym(Sym);

  // Emit BMOVT B0, Target
  BuildMI(MBB, MBBI, DL, TII->get(RISCV::BMOVT_J))
      .addDef(BReg)
      .addReg(BReg)
      .addMBB(TBB);

  // Emit unconditional branch
  BuildMI(MBB, MBBI, DL, TII->get(RISCV::PseudoPBU))
      .addReg(BReg)
      .addMBB(TBB);

  MBBI->eraseFromParent();
  return true;
}

bool RISCVExpandPseudo::expandCondBranch(MachineBasicBlock &MBB,
                                         MachineBasicBlock::iterator MBBI) {
  MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();
  Register BReg = MRI.createVirtualRegister(&RISCV::PBRRegClass);
  Register Rs1 = MBBI->getOperand(0).getReg();
  Register Rs2 = MBBI->getOperand(1).getReg();
  MachineBasicBlock *TBB = MBBI->getOperand(2).getMBB();
  DebugLoc DL = MBBI->getDebugLoc();
  MCContext &Context = MBB.getParent()->getContext();

  MCSymbol* Sym;
  unsigned Opcode = 0;
  switch (MBBI->getOpcode()) {
    case RISCV::PseudoBREQ:
      Sym = Context.createTempSymbol("ns_beq_");
      Opcode = RISCV::BMOVC_BEQ;
    break;
    case RISCV::PseudoBRNE:
      Sym = Context.createTempSymbol("ns_bne_");
      Opcode = RISCV::BMOVC_BNE;
    break;
  case RISCV::PseudoBRGE:
    Sym = Context.createTempSymbol("ns_bge_");
    Opcode = RISCV::BMOVC_BGE;
    break;
  case RISCV::PseudoBRGEU:
    Sym = Context.createTempSymbol("ns_bgeu_");
    Opcode = RISCV::BMOVC_BGEU;
    break;
  case RISCV::PseudoBRLT:
    Sym = Context.createTempSymbol("ns_blt_");
    Opcode = RISCV::BMOVC_BLT;
    break;
  case RISCV::PseudoBRLTU:
    Sym = Context.createTempSymbol("ns_bltu_");
    Opcode = RISCV::BMOVC_BLTU;
    break;
    default:
      llvm_unreachable("invalid opcode");
  }

  // Emit BMOVS B0, ns_j_
  BuildMI(MBB, MBBI, DL, TII->get(RISCV::BMOVS_J))
      .addDef(BReg)
      .addSym(Sym);

  // Emit BMOVT B0, Target
  BuildMI(MBB, MBBI, DL, TII->get(RISCV::BMOVT_J))
      .addDef(BReg)
      .addReg(BReg)
      .addMBB(TBB);

  // Emit BMOVC_XX B0, Rs1, Rs2
  BuildMI(MBB, MBBI, DL, TII->get(Opcode))
      .addDef(BReg)
      .addReg(BReg)
      .addReg(Rs1)
      .addReg(Rs2);

  // Emit conditional branch
  BuildMI(MBB, MBBI, DL, TII->get(RISCV::PseudoPBC))
      .addReg(BReg)
      .addMBB(TBB);

  MBBI->eraseFromParent();
  return true;
}

/// Transfer what \p Old recorded implicitly -- the callee clobber mask, the
/// argument registers it reads, the values it defines -- onto \p MIB, the PB
/// instruction taking its place.
///
/// Dropping these leaves every later pass believing the call neither clobbers
/// nor defines anything, so a return value reads as undefined and caller-saved
/// registers read as preserved. Operands are tested directly rather than
/// sliced at getNumExplicitOperands(), which counts a trailing non-register
/// operand as explicit on a variadic instruction and would skip the mask.
static void transferImplicitOps(const MachineInstrBuilder &MIB,
                                const MachineInstr &Old) {
  for (const MachineOperand &MO : Old.operands()) {
    if (MO.isRegMask()) {
      MIB.add(MO);
      continue;
    }
    if (!MO.isReg() || !MO.isImplicit())
      continue;
    // PseudoPBCALL already carries Defs = [X1], so skip what the description
    // of the replacement instruction states for itself.
    if (MO.isDef() && llvm::is_contained(MIB->getDesc().implicit_defs(),
                                        MO.getReg().asMCReg()))
      continue;
    MIB.add(MO);
  }
}

bool RISCVExpandPseudo::expandCall(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator MBBI) {
  MCContext &Context = MBB.getParent()->getContext();
  MCInst TmpInst;
  const MachineOperand* Func = nullptr;
  MCRegister Ra;
  MCSymbol* Sym;
  switch (MBBI->getOpcode()) {
    case RISCV::PseudoTAIL:
      Func = &MBBI->getOperand(0);
      Ra = RISCVII::getTailExpandUseRegNo(STI->getFeatureBits());
      Sym = Context.createTempSymbol("ns_tail_");
    break;
    case RISCV::PseudoCALLReg:
      Func = &MBBI->getOperand(1);
      Ra = MBBI->getOperand(0).getReg();
      Sym = Context.createTempSymbol("ns_call_reg_");
    break;
    case RISCV::PseudoCALL:
      Func = &MBBI->getOperand(0);
      Ra = RISCV::X1;
      Sym = Context.createTempSymbol("ns_call_");
    break;
    case RISCV::PseudoJump:
      Func = &MBBI->getOperand(1);
      Ra = MBBI->getOperand(0).getReg();
      Sym = Context.createTempSymbol("ns_jump_");
    break;
  default:
    llvm_unreachable("invalid opcode");
  }

  DebugLoc DL = MBBI->getDebugLoc();
  MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();
  Register BReg = MRI.createVirtualRegister(&RISCV::PBRRegClass);

  // Emit BMOVS B0, Label
  BuildMI(MBB, MBBI, DL, TII->get(RISCV::BMOVS_J))
      .addDef(BReg)
      .addSym(Sym);

  // Emit AUIPC Ra, Func with R_RISCV_CALL relocation type.
  BuildMI(MBB, MBBI, DL, TII->get(RISCV::AUIPC), Ra)
      ->addOperand(*Func);

  // Emit BMOVT B0, Ra, 0
  BuildMI(MBB, MBBI, DL, TII->get(RISCV::BMOVT_I))
      .addDef(BReg)
      .addReg(BReg)
      .addReg(Ra)
      .addImm(0);

  MachineInstrBuilder PB;
  if (MBBI->getOpcode() == RISCV::PseudoTAIL ||
      MBBI->getOpcode() == RISCV::PseudoJump) {
    // Emit ~ JALR X0, Ra, 0. Neither a tail call nor a jump links; Ra is only
    // the scratch register holding the target, which the BMOVT above has
    // already consumed.
    PB = BuildMI(MBB, MBBI, DL, TII->get(RISCV::PseudoPBI)).addReg(BReg);
  }
  else {
    // Emit ~ JALR Ra, Ra, 0
    assert(Ra == RISCV::X1);
    PB = BuildMI(MBB, MBBI, DL, TII->get(RISCV::PseudoPBCALL)).addReg(BReg);
  }
  transferImplicitOps(PB, *MBBI);

  MBBI->eraseFromParent();
  return true;
}

bool RISCVExpandPseudo::expandReturn(MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator MBBI) {
  MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();
  Register BReg = MRI.createVirtualRegister(&RISCV::PBRRegClass);
  DebugLoc DL = MBBI->getDebugLoc();
  MCContext &Context = MBB.getParent()->getContext();
  MCSymbol* Sym = Context.createTempSymbol("ns_return_");

  // Emit BMOVS B0, ns_return_
  BuildMI(MBB, MBBI, DL, TII->get(RISCV::BMOVS_J))
      .addDef(BReg)
      .addSym(Sym);

  // Emit BMOVT B0, Ra, 0
  BuildMI(MBB, MBBI, DL, TII->get(RISCV::BMOVT_I))
      .addDef(BReg)
      .addReg(BReg)
      .addReg(RISCV::X1)
      .addImm(0);

  // Emit PBAL (JALR X0, Ra, 0)
  transferImplicitOps(
      BuildMI(MBB, MBBI, DL, TII->get(RISCV::PseudoPBI)).addReg(BReg), *MBBI);

  MBBI->eraseFromParent();
  return true;
}

bool RISCVExpandPseudo::expandIndirect(MachineBasicBlock &MBB,
                                       MachineBasicBlock::iterator MBBI,
                                       bool IsCall) const {
  MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();
  Register BReg = MRI.createVirtualRegister(&RISCV::PBRRegClass);
  DebugLoc DL = MBBI->getDebugLoc();
  MCRegister Rs1 = MBBI->getOperand(0).getReg();
  MCContext &Context = MBB.getParent()->getContext();
  MCSymbol* Sym = Context.createTempSymbol("ns_indirect_");

  // Emit BMOVS B0, 8
  BuildMI(MBB, MBBI, DL, TII->get(RISCV::BMOVS_J))
      .addDef(BReg)
      .addSym(Sym);

  // Emit BMOVT B0, Rs1, 0
  BuildMI(MBB, MBBI, DL, TII->get(RISCV::BMOVT_I))
      .addDef(BReg)
      .addReg(BReg)
      .addReg(Rs1)
      .addImm(0);

  // An indirect call has to link, so it needs PBCALL (PBAL X1, $rs1, X0).
  // PBI expands to PBAL X0, $rs1, X0, which is only right for an indirect
  // branch or an indirect tail call: neither writes a return address.
  transferImplicitOps(
      BuildMI(MBB, MBBI, DL,
              TII->get(IsCall ? RISCV::PseudoPBCALL : RISCV::PseudoPBI))
          .addReg(BReg),
      *MBBI);

  MBBI->eraseFromParent();
  return true;
}

bool RISCVExpandPseudo::expandLoopEnd(MachineBasicBlock &MBB,
                                        MachineBasicBlock::iterator MBBI) {
  // dbgs() << "expandLoopEnd: "; MBBI->dump();

  MachineInstr *Setup = nullptr;
  MachineLoop *ML = MLI->getLoopFor(&MBB);
  MachineBasicBlock *Preheader = ML->getLoopPreheader();
  for (MachineInstr &MI : *Preheader) {
    if (MI.getOpcode() == RISCV::PseudoLoopSetup) {
      Setup = &MI;
    }
  }

  assert(Setup && "Loop end must also contain a setup");
  // dbgs() << "Found: "; Setup->dump();

  MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();
  Register BReg = MRI.createVirtualRegister(&RISCV::PBRRegClass);
  DebugLoc DL = MBBI->getDebugLoc();
  MCContext &Context = MBB.getParent()->getContext();
  MCSymbol* Sym = Context.createTempSymbol("ns_loop_");

  MachineBasicBlock *TargetMBB = MBBI->getOperand(0).getMBB();
  Register Count = Setup->getOperand(0).getReg();

  // Emit BMOVS B0, ns_loop_
  BuildMI(*Setup->getParent(), Setup, Setup->getDebugLoc(),
    TII->get(RISCV::BMOVS_J))
      .addDef(BReg).addSym(Sym);

  // Emit BMOVT B0, Target
  BuildMI(*Setup->getParent(), Setup, Setup->getDebugLoc(), TII->get(RISCV::BMOVT_J))
      .addDef(BReg).addReg(BReg).addMBB(TargetMBB);

  // Emit BMOVC_LOOP B0, Rs1, 0
  BuildMI(*Setup->getParent(), Setup, Setup->getDebugLoc(), TII->get(RISCV::BMOVC_LOOP))
      .addDef(BReg).addReg(BReg).addReg(Count).addImm(0);

  // Emit conditional branch
  BuildMI(MBB, MBBI, DL, TII->get(RISCV::PseudoPBC))
      .addReg(BReg).addMBB(TargetMBB);

  Setup->eraseFromParent();
  MBBI->eraseFromParent();

  return true;
}

bool RISCVExpandPseudo::expandMV_FPR32INX(MachineBasicBlock &MBB,
                                          MachineBasicBlock::iterator MBBI) {
  DebugLoc DL = MBBI->getDebugLoc();
  const TargetRegisterInfo *TRI = STI->getRegisterInfo();
  Register DstReg = TRI->getMatchingSuperReg(
      MBBI->getOperand(0).getReg(), RISCV::sub_32, &RISCV::GPRRegClass);
  Register SrcReg = TRI->getMatchingSuperReg(
      MBBI->getOperand(1).getReg(), RISCV::sub_32, &RISCV::GPRRegClass);

  BuildMI(MBB, MBBI, DL, TII->get(RISCV::ADDI), DstReg)
      .addReg(SrcReg, getKillRegState(MBBI->getOperand(1).isKill()))
      .addImm(0);

  MBBI->eraseFromParent(); // The pseudo instruction is gone now.
  return true;
}

// This function expands the PseudoRV32ZdinxSD for storing a double-precision
// floating-point value into memory by generating an equivalent instruction
// sequence for RV32.
bool RISCVExpandPseudo::expandRV32ZdinxStore(MachineBasicBlock &MBB,
                                             MachineBasicBlock::iterator MBBI) {
  DebugLoc DL = MBBI->getDebugLoc();
  const TargetRegisterInfo *TRI = STI->getRegisterInfo();
  Register Lo =
      TRI->getSubReg(MBBI->getOperand(0).getReg(), RISCV::sub_gpr_even);
  Register Hi =
      TRI->getSubReg(MBBI->getOperand(0).getReg(), RISCV::sub_gpr_odd);
  if (Hi == RISCV::DUMMY_REG_PAIR_WITH_X0)
    Hi = RISCV::X0;

  auto MIBLo = BuildMI(MBB, MBBI, DL, TII->get(RISCV::SW))
                   .addReg(Lo, getKillRegState(MBBI->getOperand(0).isKill()))
                   .addReg(MBBI->getOperand(1).getReg())
                   .add(MBBI->getOperand(2));

  MachineInstrBuilder MIBHi;
  if (MBBI->getOperand(2).isGlobal() || MBBI->getOperand(2).isCPI()) {
    assert(MBBI->getOperand(2).getOffset() % 8 == 0);
    MBBI->getOperand(2).setOffset(MBBI->getOperand(2).getOffset() + 4);
    MIBHi = BuildMI(MBB, MBBI, DL, TII->get(RISCV::SW))
                .addReg(Hi, getKillRegState(MBBI->getOperand(0).isKill()))
                .add(MBBI->getOperand(1))
                .add(MBBI->getOperand(2));
  } else {
    assert(isInt<12>(MBBI->getOperand(2).getImm() + 4));
    MIBHi = BuildMI(MBB, MBBI, DL, TII->get(RISCV::SW))
                .addReg(Hi, getKillRegState(MBBI->getOperand(0).isKill()))
                .add(MBBI->getOperand(1))
                .addImm(MBBI->getOperand(2).getImm() + 4);
  }

  MachineFunction *MF = MBB.getParent();
  SmallVector<MachineMemOperand *> NewLoMMOs;
  SmallVector<MachineMemOperand *> NewHiMMOs;
  for (const MachineMemOperand *MMO : MBBI->memoperands()) {
    NewLoMMOs.push_back(MF->getMachineMemOperand(MMO, 0, 4));
    NewHiMMOs.push_back(MF->getMachineMemOperand(MMO, 4, 4));
  }
  MIBLo.setMemRefs(NewLoMMOs);
  MIBHi.setMemRefs(NewHiMMOs);

  MBBI->eraseFromParent();
  return true;
}

// This function expands PseudoRV32ZdinxLoad for loading a double-precision
// floating-point value from memory into an equivalent instruction sequence for
// RV32.
bool RISCVExpandPseudo::expandRV32ZdinxLoad(MachineBasicBlock &MBB,
                                            MachineBasicBlock::iterator MBBI) {
  DebugLoc DL = MBBI->getDebugLoc();
  const TargetRegisterInfo *TRI = STI->getRegisterInfo();
  Register Lo =
      TRI->getSubReg(MBBI->getOperand(0).getReg(), RISCV::sub_gpr_even);
  Register Hi =
      TRI->getSubReg(MBBI->getOperand(0).getReg(), RISCV::sub_gpr_odd);
  assert(Hi != RISCV::DUMMY_REG_PAIR_WITH_X0 && "Cannot write to X0_Pair");

  MachineInstrBuilder MIBLo, MIBHi;

  // If the register of operand 1 is equal to the Lo register, then swap the
  // order of loading the Lo and Hi statements.
  bool IsOp1EqualToLo = Lo == MBBI->getOperand(1).getReg();
  // Order: Lo, Hi
  if (!IsOp1EqualToLo) {
    MIBLo = BuildMI(MBB, MBBI, DL, TII->get(RISCV::LW), Lo)
                .addReg(MBBI->getOperand(1).getReg())
                .add(MBBI->getOperand(2));
  }

  if (MBBI->getOperand(2).isGlobal() || MBBI->getOperand(2).isCPI()) {
    auto Offset = MBBI->getOperand(2).getOffset();
    assert(Offset % 8 == 0);
    MBBI->getOperand(2).setOffset(Offset + 4);
    MIBHi = BuildMI(MBB, MBBI, DL, TII->get(RISCV::LW), Hi)
                .addReg(MBBI->getOperand(1).getReg())
                .add(MBBI->getOperand(2));
    MBBI->getOperand(2).setOffset(Offset);
  } else {
    assert(isInt<12>(MBBI->getOperand(2).getImm() + 4));
    MIBHi = BuildMI(MBB, MBBI, DL, TII->get(RISCV::LW), Hi)
                .addReg(MBBI->getOperand(1).getReg())
                .addImm(MBBI->getOperand(2).getImm() + 4);
  }

  // Order: Hi, Lo
  if (IsOp1EqualToLo) {
    MIBLo = BuildMI(MBB, MBBI, DL, TII->get(RISCV::LW), Lo)
                .addReg(MBBI->getOperand(1).getReg())
                .add(MBBI->getOperand(2));
  }

  MachineFunction *MF = MBB.getParent();
  SmallVector<MachineMemOperand *> NewLoMMOs;
  SmallVector<MachineMemOperand *> NewHiMMOs;
  for (const MachineMemOperand *MMO : MBBI->memoperands()) {
    NewLoMMOs.push_back(MF->getMachineMemOperand(MMO, 0, 4));
    NewHiMMOs.push_back(MF->getMachineMemOperand(MMO, 4, 4));
  }
  MIBLo.setMemRefs(NewLoMMOs);
  MIBHi.setMemRefs(NewHiMMOs);

  MBBI->eraseFromParent();
  return true;
}

bool RISCVExpandPseudo::expandPseudoReadVLENBViaVSETVLIX0(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI) {
  DebugLoc DL = MBBI->getDebugLoc();
  Register Dst = MBBI->getOperand(0).getReg();
  unsigned Mul = MBBI->getOperand(1).getImm();
  RISCVVType::VLMUL VLMUL = RISCVVType::encodeLMUL(Mul, /*Fractional=*/false);
  unsigned VTypeImm = RISCVVType::encodeVTYPE(
      VLMUL, /*SEW=*/8, /*TailAgnostic=*/true, /*MaskAgnostic=*/true);

  BuildMI(MBB, MBBI, DL, TII->get(RISCV::PseudoVSETVLIX0))
      .addReg(Dst, RegState::Define)
      .addReg(RISCV::X0, RegState::Kill)
      .addImm(VTypeImm);

  MBBI->eraseFromParent();
  return true;
}

class RISCVPreRAExpandPseudo : public MachineFunctionPass {
public:
  const RISCVSubtarget *STI;
  const RISCVInstrInfo *TII;
  static char ID;

  RISCVPreRAExpandPseudo() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
  StringRef getPassName() const override {
    return RISCV_PRERA_EXPAND_PSEUDO_NAME;
  }

private:
  bool expandMBB(MachineBasicBlock &MBB);
  bool expandMI(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
                MachineBasicBlock::iterator &NextMBBI);
  bool expandAuipcInstPair(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator MBBI,
                           MachineBasicBlock::iterator &NextMBBI,
                           unsigned FlagsHi, unsigned SecondOpcode);
  bool expandLoadLocalAddress(MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator MBBI,
                              MachineBasicBlock::iterator &NextMBBI);
  bool expandLoadGlobalAddress(MachineBasicBlock &MBB,
                               MachineBasicBlock::iterator MBBI,
                               MachineBasicBlock::iterator &NextMBBI);
  bool expandLoadTLSIEAddress(MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator MBBI,
                              MachineBasicBlock::iterator &NextMBBI);
  bool expandLoadTLSGDAddress(MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator MBBI,
                              MachineBasicBlock::iterator &NextMBBI);
  bool expandLoadTLSDescAddress(MachineBasicBlock &MBB,
                                MachineBasicBlock::iterator MBBI,
                                MachineBasicBlock::iterator &NextMBBI);

#ifndef NDEBUG
  unsigned getInstSizeInBytes(const MachineFunction &MF) const {
    unsigned Size = 0;
    for (auto &MBB : MF)
      for (auto &MI : MBB)
        Size += TII->getInstSizeInBytes(MI);
    return Size;
  }
#endif
};

char RISCVPreRAExpandPseudo::ID = 0;

bool RISCVPreRAExpandPseudo::runOnMachineFunction(MachineFunction &MF) {
  STI = &MF.getSubtarget<RISCVSubtarget>();
  TII = STI->getInstrInfo();

#ifndef NDEBUG
  const unsigned OldSize = getInstSizeInBytes(MF);
#endif

  bool Modified = false;
  for (auto &MBB : MF)
    Modified |= expandMBB(MBB);

#ifndef NDEBUG
  const unsigned NewSize = getInstSizeInBytes(MF);
  assert(OldSize >= NewSize);
#endif
  return Modified;
}

bool RISCVPreRAExpandPseudo::expandMBB(MachineBasicBlock &MBB) {
  bool Modified = false;

  MachineBasicBlock::iterator MBBI = MBB.begin(), E = MBB.end();
  while (MBBI != E) {
    MachineBasicBlock::iterator NMBBI = std::next(MBBI);
    Modified |= expandMI(MBB, MBBI, NMBBI);
    MBBI = NMBBI;
  }

  return Modified;
}

bool RISCVPreRAExpandPseudo::expandMI(MachineBasicBlock &MBB,
                                      MachineBasicBlock::iterator MBBI,
                                      MachineBasicBlock::iterator &NextMBBI) {

  switch (MBBI->getOpcode()) {
  case RISCV::PseudoLLA:
    return expandLoadLocalAddress(MBB, MBBI, NextMBBI);
  case RISCV::PseudoLGA:
    return expandLoadGlobalAddress(MBB, MBBI, NextMBBI);
  case RISCV::PseudoLA_TLS_IE:
    return expandLoadTLSIEAddress(MBB, MBBI, NextMBBI);
  case RISCV::PseudoLA_TLS_GD:
    return expandLoadTLSGDAddress(MBB, MBBI, NextMBBI);
  case RISCV::PseudoLA_TLSDESC:
    return expandLoadTLSDescAddress(MBB, MBBI, NextMBBI);
  }
  return false;
}

bool RISCVPreRAExpandPseudo::expandAuipcInstPair(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
    MachineBasicBlock::iterator &NextMBBI, unsigned FlagsHi,
    unsigned SecondOpcode) {
  MachineFunction *MF = MBB.getParent();
  MachineInstr &MI = *MBBI;
  DebugLoc DL = MI.getDebugLoc();

  Register DestReg = MI.getOperand(0).getReg();
  Register ScratchReg =
      MF->getRegInfo().createVirtualRegister(&RISCV::GPRRegClass);

  MachineOperand &Symbol = MI.getOperand(1);
  Symbol.setTargetFlags(FlagsHi);
  MCSymbol *AUIPCSymbol = MF->getContext().createNamedTempSymbol("pcrel_hi");

  MachineInstr *MIAUIPC =
      BuildMI(MBB, MBBI, DL, TII->get(RISCV::AUIPC), ScratchReg).add(Symbol);
  MIAUIPC->setPreInstrSymbol(*MF, AUIPCSymbol);

  MachineInstr *SecondMI =
      BuildMI(MBB, MBBI, DL, TII->get(SecondOpcode), DestReg)
          .addReg(ScratchReg)
          .addSym(AUIPCSymbol, RISCVII::MO_PCREL_LO);

  if (MI.hasOneMemOperand())
    SecondMI->addMemOperand(*MF, *MI.memoperands_begin());

  MI.eraseFromParent();
  return true;
}

bool RISCVPreRAExpandPseudo::expandLoadLocalAddress(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
    MachineBasicBlock::iterator &NextMBBI) {
  return expandAuipcInstPair(MBB, MBBI, NextMBBI, RISCVII::MO_PCREL_HI,
                             RISCV::ADDI);
}

bool RISCVPreRAExpandPseudo::expandLoadGlobalAddress(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
    MachineBasicBlock::iterator &NextMBBI) {
  unsigned SecondOpcode = STI->is64Bit() ? RISCV::LD : RISCV::LW;
  return expandAuipcInstPair(MBB, MBBI, NextMBBI, RISCVII::MO_GOT_HI,
                             SecondOpcode);
}

bool RISCVPreRAExpandPseudo::expandLoadTLSIEAddress(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
    MachineBasicBlock::iterator &NextMBBI) {
  unsigned SecondOpcode = STI->is64Bit() ? RISCV::LD : RISCV::LW;
  return expandAuipcInstPair(MBB, MBBI, NextMBBI, RISCVII::MO_TLS_GOT_HI,
                             SecondOpcode);
}

bool RISCVPreRAExpandPseudo::expandLoadTLSGDAddress(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
    MachineBasicBlock::iterator &NextMBBI) {
  return expandAuipcInstPair(MBB, MBBI, NextMBBI, RISCVII::MO_TLS_GD_HI,
                             RISCV::ADDI);
}

bool RISCVPreRAExpandPseudo::expandLoadTLSDescAddress(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
    MachineBasicBlock::iterator &NextMBBI) {
  MachineFunction *MF = MBB.getParent();
  MachineInstr &MI = *MBBI;
  DebugLoc DL = MI.getDebugLoc();

  const auto &STI = MF->getSubtarget<RISCVSubtarget>();
  unsigned SecondOpcode = STI.is64Bit() ? RISCV::LD : RISCV::LW;

  Register FinalReg = MI.getOperand(0).getReg();
  Register DestReg =
      MF->getRegInfo().createVirtualRegister(&RISCV::GPRRegClass);
  Register ScratchReg =
      MF->getRegInfo().createVirtualRegister(&RISCV::GPRRegClass);

  MachineOperand &Symbol = MI.getOperand(1);
  Symbol.setTargetFlags(RISCVII::MO_TLSDESC_HI);
  MCSymbol *AUIPCSymbol = MF->getContext().createNamedTempSymbol("tlsdesc_hi");

  MachineInstr *MIAUIPC =
      BuildMI(MBB, MBBI, DL, TII->get(RISCV::AUIPC), ScratchReg).add(Symbol);
  MIAUIPC->setPreInstrSymbol(*MF, AUIPCSymbol);

  BuildMI(MBB, MBBI, DL, TII->get(SecondOpcode), DestReg)
      .addReg(ScratchReg)
      .addSym(AUIPCSymbol, RISCVII::MO_TLSDESC_LOAD_LO);

  BuildMI(MBB, MBBI, DL, TII->get(RISCV::ADDI), RISCV::X10)
      .addReg(ScratchReg)
      .addSym(AUIPCSymbol, RISCVII::MO_TLSDESC_ADD_LO);

  BuildMI(MBB, MBBI, DL, TII->get(RISCV::PseudoTLSDESCCall), RISCV::X5)
      .addReg(DestReg)
      .addImm(0)
      .addSym(AUIPCSymbol, RISCVII::MO_TLSDESC_CALL);

  BuildMI(MBB, MBBI, DL, TII->get(RISCV::ADD), FinalReg)
      .addReg(RISCV::X10)
      .addReg(RISCV::X4);

  MI.eraseFromParent();
  return true;
}

} // end of anonymous namespace

INITIALIZE_PASS(RISCVExpandPseudo, "riscv-expand-pseudo",
                RISCV_EXPAND_PSEUDO_NAME, false, false)

INITIALIZE_PASS(RISCVPreRAExpandPseudo, "riscv-prera-expand-pseudo",
                RISCV_PRERA_EXPAND_PSEUDO_NAME, false, false)

namespace llvm {

FunctionPass *createRISCVExpandPseudoPass() { return new RISCVExpandPseudo(); }
FunctionPass *createRISCVPreRAExpandPseudoPass() { return new RISCVPreRAExpandPseudo(); }

} // end of namespace llvm
