//===- RISCVNSLoopOptions.h - Non-Spec loop options ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RISCV_RISCVNSLOOPOPTIONS_H
#define LLVM_LIB_TARGET_RISCV_RISCVNSLOOPOPTIONS_H

#include "llvm/ADT/StringRef.h"
#include <optional>

namespace llvm {
class Function;
class Instruction;
class LLVMContext;
class MDNode;

namespace RISCVNS {

struct SWPSettings {
  bool Enable = true;
  bool Force = false;
  unsigned Distance = 16;
  unsigned SpeculativeDistance = 4;
  bool Speculative = false;
  bool ShiftRecurrences = false;
  bool Toggles = false;
  unsigned MaxSliceSize = 16;
  unsigned MaxLoopSize = 256;
  unsigned MaxPrologueSize = 512;
};

struct FissionSettings {
  bool Enable = true;
  bool Force = false;
  unsigned QueueDepth = 64;
};

/// Only explicitly supplied properties have values. In particular, false is
/// different from absent. Resolution never writes to a command-line option.
struct LoopOptions {
  std::optional<bool> SWPEnable, IRFissionEnable, MachineFissionEnable, Force;
  std::optional<bool> Speculative, ShiftRecurrences, Toggles;
  std::optional<unsigned> Distance, SpeculativeDistance;
  std::optional<unsigned> MaxSliceSize, MaxLoopSize, MaxPrologueSize,
      QueueDepth;

  bool hasRequests() const;
  SWPSettings resolveSWP(SWPSettings Defaults, bool GloballyDisabled) const;
  FissionSettings resolveIRFission(bool GloballyDisabled) const;
  FissionSettings resolveMachineFission(unsigned DefaultDepth,
                                        bool GloballyDisabled) const;
};

/// Diagnose malformed recognized properties with an error, even on loops that
/// are disabled or cannot be transformed. Unknown properties are preserved.
std::optional<LoopOptions> parseLoopOptions(const MDNode *ID,
                                            const Instruction &Latch);
bool validateLoopOptions(const Function &F);

bool isIRFissioned(const MDNode *ID);
StringRef getSWPApplied(const MDNode *ID);
bool hasLoopControls(const MDNode *ID);

enum class LoopRole { Fission, Kernel, Drain, InPlace };

/// Create a fresh self-referencing ID, consume requests, preserve unrelated
/// properties and record the completed transformation. Only the drain retains
/// machine-fission controls, the common depth and force.
MDNode *makeLoopID(LLVMContext &Context, const MDNode *OldID, LoopRole Role);

} // namespace RISCVNS
} // namespace llvm

#endif
