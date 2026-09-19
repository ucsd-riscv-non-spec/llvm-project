//===-- RISCVBranchSetupAnalysis.cpp - Branch setup analysis --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares RISCVBranchSetupAnalysis, a read-only MachineFunction
// analysis meant to run at the very end of the RISC-V backend pipeline,
// after both RISCVExpandPseudoInsts and RISCVExpandAtomicPseudoInsts have
// run. At that point every remaining MachineInstr should be a "real" (i.e.
// non-pseudo) target instruction -- exactly the instruction stream that is
// about to be handed to the AsmPrinter/MC layer.
//
// The analysis never mutates the MachineFunction (AU.setPreservesAll()), so
// it can be inserted anywhere late in the pipeline without perturbing
// codegen.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RISCV_RISCVBRANCHSETUPANALYSIS_H
#define LLVM_LIB_TARGET_RISCV_RISCVBRANCHSETUPANALYSIS_H

#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/ReachingDefAnalysis.h"

namespace llvm {

class MachineFunction;
class Module;
class raw_ostream;

struct RISCVBranchSetup {
  const MachineInstr *S; // Source
  const MachineInstr *T; // Target
  const MachineInstr *C; // Condition (optional)

  operator bool() const { return S || T || C; }
  void dump() const;
  void print(raw_ostream &OS) const;
};

struct RISCVBranchSetupInfo {
  unsigned NumBMOVS = 0;
  unsigned NumBMOVT = 0;
  unsigned NumBMOVC = 0;
  unsigned NumPB = 0;

  DenseMap<const MachineInstr*, RISCVBranchSetup> Branches;

  void print(raw_ostream &OS, const MachineFunction &MF) const;
  void dump(const MachineFunction &MF) const;
};

/// Legacy PassManager wrapper. Other legacy-PM passes that run after this
/// one in the pipeline can retrieve the cached result with:
///
///   void getAnalysisUsage(AnalysisUsage &AU) const override {
///     AU.addRequired<RISCVBranchSetupAnalysisWrapper>();
///     AU.setPreservesAll();
///   }
///   ...
///   const RISCVBranchSetupInfo &Info =
///       getAnalysis<RISCVBranchSetupAnalysisWrapper>().getInfo();
class RISCVBranchSetupAnalysisWrapper : public MachineFunctionPass {
  RISCVBranchSetupInfo Info;

public:
  static char ID;

  RISCVBranchSetupAnalysisWrapper();

  RISCVBranchSetupInfo &getInfo() { return Info; }
  const RISCVBranchSetupInfo &getInfo() const { return Info; }

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineDominatorTreeWrapperPass>();
    // AU.addRequired<ReachingDefAnalysis>();
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  void releaseMemory() override { Info = RISCVBranchSetupInfo(); }

  void print(raw_ostream &OS, const Module *M = nullptr) const override;

  StringRef getPassName() const override {
    return "RISC-V Branch Setup Analysis";
  }
};

/// New-PM analysis, exposed for tools/pipelines that build a
/// MachineFunctionAnalysisManager directly. NOTE: as of LLVM 21 the in-tree
/// `llc` RISC-V pipeline is still driven by the legacy TargetPassConfig
/// (RISCVPassConfig), so RISCVBranchSetupAnalysisWrapper above is what
/// actually runs for ordinary `llc`/`RISCVTargetMachine` codegen. This
/// class is provided so the same analysis logic is available to any
/// MIR new-PM based driver without duplicating the traversal code.
class RISCVBranchSetupAnalysis : public AnalysisInfoMixin<RISCVBranchSetupAnalysis> {
  friend AnalysisInfoMixin<RISCVBranchSetupAnalysis>;
  static AnalysisKey Key;

public:
  using Result = RISCVBranchSetupInfo;

  Result run(MachineFunction &MF, MachineFunctionAnalysisManager &MFM);
};

/// New-PM printer, usable as `-passes=print<riscv-branch-support>` by tools
/// that wire up the MIR new-PM pipeline.
class RISCVBranchSetupAnalysisPrinterPass
    : public PassInfoMixin<RISCVBranchSetupAnalysisPrinterPass> {
  raw_ostream &OS;

public:
  explicit RISCVBranchSetupAnalysisPrinterPass(raw_ostream &OS) : OS(OS) {}

  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);

  static bool isRequired() { return true; }
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_RISCV_RISCVBRANCHSUPPORTANALYSIS_H