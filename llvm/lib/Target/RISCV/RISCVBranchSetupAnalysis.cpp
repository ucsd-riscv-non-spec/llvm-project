//===-- RISCVBranchSetupAnalysis.cpp - Branch support analysis ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RISCVBranchSetupAnalysis.h"
#include "RISCVSubtarget.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-branch-support-analysis"

//STATISTIC(NumFunctionsWithResidualPseudos,
//          "Number of functions that still contained a pseudo instruction "
//          "after RISCVExpandPseudoInsts/RISCVExpandAtomicPseudoInsts");

/// When set, dump a one-line summary of RISCVBranchSetupInfo for every function,
/// e.g. via `llc -riscv-print-branch-support-analysis`.
static cl::opt<bool> PrintRISCVBranchSetupAnalysis(
    "riscv-print-branch-support-analysis", cl::Hidden,
    cl::desc("Print RISCVBranchSetupAnalysis results for every function"),
    cl::init(false));

/// When set, a residual (unexpanded) pseudo instruction found by this
/// analysis is reported as a hard, fatal error instead of a warning. Useful
/// to turn on in CI so an unexpanded pseudo fails the build loudly instead
/// of silently reaching the AsmPrinter.
//static cl::opt<bool> RISCVBranchSetupAnalysisStrict(
//    "riscv-late-mir-analysis-strict", cl::Hidden,
//    cl::desc("Treat residual pseudo instructions found by "
//             "RISCVBranchSetupAnalysis as a fatal zerror"),
//    cl::init(false));


void RISCVBranchSetup::dump() const {
  print(dbgs());
  dbgs() << '\n';
}
void RISCVBranchSetup::print(raw_ostream &OS) const {
  OS << "S = "; if (S) { OS << *S; } else { OS << "nullptr\n"; }
  OS << "T = "; if (T) { OS << *T; } else { OS << "nullptr\n"; }
  OS << "C = "; if (C) { OS << *C; } else { OS << "nullptr\n"; }
}

static RISCVBranchSetup FindBranchSetup(const MachineInstr &PB,
                                        const MachineDominatorTree &MDT) {
  RISCVBranchSetup Setup = {};
  const MachineBasicBlock *MBB = PB.getParent();
  Register BR = PB.getOperand(0).getReg();

  auto ScanBlock = [&](const MachineBasicBlock &Block,
                       MachineBasicBlock::const_iterator It) -> bool {
    while (It != Block.begin()) {
      --It;
      const MachineInstr &I = *It;
      if (!I.isBMOV())
        continue;

      switch (I.getOpcode()) {
      case RISCV::BMOVS_I:
      case RISCV::BMOVS_J:
        if (!Setup.S && I.getOperand(0).getReg() == BR) {
          Setup.S = &I;
        }
        break;
      case RISCV::BMOVT_I:
      case RISCV::BMOVT_J:
        if (!Setup.T && I.getOperand(0).getReg() == BR) {
          Setup.T = &I;
        }
        break;
      case RISCV::BMOVC_BEQ:
      case RISCV::BMOVC_BNE:
      case RISCV::BMOVC_BLT:
      case RISCV::BMOVC_BLTU:
      case RISCV::BMOVC_BGE:
      case RISCV::BMOVC_BGEU:
      case RISCV::BMOVC_BITS:
      case RISCV::BMOVC_LOOP:
        if (!Setup.C && I.getOperand(0).getReg() == BR) {
          Setup.C = &I;
        }
        break;
      default:
        llvm_unreachable("invalid BMOV instruction");
      }
      if (Setup.S && Setup.T)
        return true;
    }
    return false;
  };

  if (ScanBlock(*MBB, PB.getIterator()))
    return Setup;

  auto *DomNode = MDT.getNode(MBB);
  while (DomNode && DomNode->getIDom()) {
    DomNode = DomNode->getIDom();
    const MachineBasicBlock *DomMBB = DomNode->getBlock();
    if (ScanBlock(*DomMBB, DomMBB->end()))
      break;
  }

  return Setup;
}

/// Core, PM-agnostic traversal shared by the legacy wrapper pass and the
/// new-PM analysis below.
static RISCVBranchSetupInfo computeRISCVBranchSetupInfo(const MachineFunction &MF,
                                                        const MachineDominatorTree &MDT,
                                                        const ReachingDefAnalysis &RDA) {
  RISCVBranchSetupInfo Info;

  for (const MachineBasicBlock &MBB : MF) {
    for (const MachineInstr &MI : MBB) {
      if (MI.isDebugInstr() || MI.isMetaInstruction())
        continue;

      switch (MI.getOpcode()) {
      case RISCV::BMOVS_I:
      case RISCV::BMOVS_J:
        Info.NumBMOVS += 1;
        break;
      case RISCV::BMOVT_I:
      case RISCV::BMOVT_J:
        Info.NumBMOVT += 1;
        break;
      case RISCV::BMOVC_BEQ:
      case RISCV::BMOVC_BNE:
      case RISCV::BMOVC_BLT:
      case RISCV::BMOVC_BLTU:
      case RISCV::BMOVC_BGE:
      case RISCV::BMOVC_BGEU:
        Info.NumBMOVC += 1;
        break;
      case RISCV::PseudoPBCALL:
      case RISCV::PseudoPBU:
      case RISCV::PseudoPBC:
      case RISCV::PseudoPBI: {
        Info.NumPB += 1;
        // TODO(mitch): Look more into better ways of locating branch setup instructions
        // MachineInstr *Def = RDA.getUniqueReachingMIDef(const_cast<MachineInstr*>(&MI), MI.getOperand(0).getReg());
        // dbgs() << "ReachingDef: "; if (Def) Def->dump(); else dbgs() << "None\n";
        RISCVBranchSetup Setup = FindBranchSetup(MI, MDT);
        auto [_, Inserted] = Info.Branches.try_emplace(&MI, Setup);
        assert(Inserted);
        break;
      }
      default:
        break;
      }
    }
  }

  return Info;
}

void RISCVBranchSetupInfo::print(raw_ostream &OS, const MachineFunction &MF) const {
  OS << "RISCVBranchSetupAnalysis for function '" << MF.getName() << "':\n";
    //<< "  " << Branches.size() << " branches" << '\n'
  for (auto& B : Branches) {
    B.getFirst()->print(OS);
  }
  OS << "  BMOVS:  " << NumBMOVS << " instructions" << '\n'
    << "  BMOVT:  " << NumBMOVT << " instructions" << '\n'
    << "  BMOVC:  " << NumBMOVC << " instructions" << '\n'
    << "  PBAL:   " << NumPB << " instructions" << '\n';
}

void RISCVBranchSetupInfo::dump(const MachineFunction &MF) const {
  print(dbgs(), MF);
}

//===----------------------------------------------------------------------===//
// Legacy PassManager wrapper pass.
//===----------------------------------------------------------------------===//

char RISCVBranchSetupAnalysisWrapper::ID = 0;

INITIALIZE_PASS(RISCVBranchSetupAnalysisWrapper,
                "riscv-branch-support-analysis",
                "RISC-V Branch Setup Analysis", false, true)

RISCVBranchSetupAnalysisWrapper::RISCVBranchSetupAnalysisWrapper()
    : MachineFunctionPass(ID) {}

bool RISCVBranchSetupAnalysisWrapper::runOnMachineFunction(
    MachineFunction &MF) {
  const MachineDominatorTree &MDT = getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  ReachingDefAnalysis RDA;// = getAnalysis<ReachingDefAnalysis>();
  Info = computeRISCVBranchSetupInfo(MF, MDT, RDA);
  if (PrintRISCVBranchSetupAnalysis)
    Info.print(errs(), MF);
  // This is a pure analysis: it never changes the MachineFunction.
  return false;
}

void RISCVBranchSetupAnalysisWrapper::print(raw_ostream &OS,
                                            const Module *) const {
  // `print()` is invoked without a MachineFunction handle (see
  // MachineFunctionPass::print / the -p / MIR pass-printing machinery), so
  // we don't have a name to print here; dump the raw counters instead.
  OS << __func__ << " was called\n";
}

FunctionPass *llvm::createRISCVBranchSetupAnalysisPass() {
  return new RISCVBranchSetupAnalysisWrapper();
}

//===----------------------------------------------------------------------===//
// New PassManager analysis + printer.
//===----------------------------------------------------------------------===//

AnalysisKey RISCVBranchSetupAnalysis::Key;

RISCVBranchSetupAnalysis::Result
RISCVBranchSetupAnalysis::run(MachineFunction &MF,
                          MachineFunctionAnalysisManager &MFM) {
  const MachineDominatorTree &MDT = MFM.getResult<MachineDominatorTreeAnalysis>(MF);
  ReachingDefAnalysis RDA;// = MFM.getResult<ReachingDefAnalysis>(MF);
  RISCVBranchSetupInfo Info = computeRISCVBranchSetupInfo(MF, MDT, RDA);
  if (PrintRISCVBranchSetupAnalysis)
    Info.print(errs(), MF);
  return Info;
}

PreservedAnalyses
RISCVBranchSetupAnalysisPrinterPass::run(MachineFunction &MF,
                                     MachineFunctionAnalysisManager &MFAM) {
  MFAM.getResult<RISCVBranchSetupAnalysis>(MF).print(OS, MF);
  return PreservedAnalyses::all();
}