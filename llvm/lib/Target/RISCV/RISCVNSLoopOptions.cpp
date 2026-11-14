//===-- RISCVNSLoopOptions.cpp - Non-Spec loop options
//----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RISCVNSLoopOptions.h"
#include "RISCV.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include <algorithm>
#include <cstdint>

using namespace llvm;
using namespace llvm::RISCVNS;

namespace {
struct Option {
  StringLiteral Name;
  std::optional<bool> LoopOptions::*Boolean = nullptr;
  std::optional<unsigned> LoopOptions::*Number = nullptr;
  unsigned Max = 0;
};

constexpr Option Options[] = {
    {"llvm.loop.riscv.ns.swp.enable", &LoopOptions::SWPEnable},
    {"llvm.loop.riscv.ns.fission.ir.enable", &LoopOptions::IRFissionEnable},
    {"llvm.loop.riscv.ns.fission.machine.enable",
     &LoopOptions::MachineFissionEnable},
    {"llvm.loop.riscv.ns.force", &LoopOptions::Force},
    {"llvm.loop.riscv.ns.swp.speculative", &LoopOptions::Speculative},
    {"llvm.loop.riscv.ns.swp.shift_recurrences",
     &LoopOptions::ShiftRecurrences},
    {"llvm.loop.riscv.ns.swp.toggles", &LoopOptions::Toggles},
    {"llvm.loop.riscv.ns.swp.distance", nullptr, &LoopOptions::Distance, 62},
    {"llvm.loop.riscv.ns.swp.speculative_distance", nullptr,
     &LoopOptions::SpeculativeDistance, 62},
    {"llvm.loop.riscv.ns.swp.max_slice_size", nullptr,
     &LoopOptions::MaxSliceSize, INT32_MAX},
    {"llvm.loop.riscv.ns.swp.max_loop_size", nullptr, &LoopOptions::MaxLoopSize,
     INT32_MAX},
    {"llvm.loop.riscv.ns.swp.max_prologue_size", nullptr,
     &LoopOptions::MaxPrologueSize, INT32_MAX},
    {"llvm.loop.riscv.ns.fission.queue_depth", nullptr,
     &LoopOptions::QueueDepth, 64}};

StringRef propertyName(const Metadata *MD) {
  if (auto *Name = dyn_cast_or_null<MDString>(MD))
    return Name->getString();
  auto *Node = dyn_cast_or_null<MDNode>(MD);
  if (!Node || Node->getNumOperands() == 0)
    return {};
  auto *Name = dyn_cast_or_null<MDString>(Node->getOperand(0));
  return Name ? Name->getString() : StringRef();
}

const Option *lookupOption(StringRef Name) {
  for (const Option &O : Options)
    if (O.Name == Name)
      return &O;
  return nullptr;
}

class RISCVNSLoopMetadata : public FunctionPass {
public:
  static char ID;
  RISCVNSLoopMetadata() : FunctionPass(ID) {}
  bool runOnFunction(Function &F) override {
    // Deliberately precedes skipFunction and all transformation gates.
    validateLoopOptions(F);
    return false;
  }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }
};
char RISCVNSLoopMetadata::ID = 0;
} // namespace

bool LoopOptions::hasRequests() const {
  for (const Option &O : Options)
    if (O.Boolean ? (this->*O.Boolean).has_value()
                  : (this->*O.Number).has_value())
      return true;
  return false;
}

SWPSettings LoopOptions::resolveSWP(SWPSettings D, bool Disabled) const {
  D.Enable = !Disabled && SWPEnable.value_or(D.Enable);
  D.Force = Force.value_or(false);
  D.Distance = Distance.value_or(D.Distance);
  D.SpeculativeDistance = SpeculativeDistance.value_or(D.SpeculativeDistance);
  D.Speculative = Speculative.value_or(D.Speculative);
  D.ShiftRecurrences = ShiftRecurrences.value_or(D.ShiftRecurrences);
  D.Toggles = Toggles.value_or(D.Toggles);
  D.MaxSliceSize = MaxSliceSize.value_or(D.MaxSliceSize);
  D.MaxLoopSize = MaxLoopSize.value_or(D.MaxLoopSize);
  D.MaxPrologueSize = MaxPrologueSize.value_or(D.MaxPrologueSize);
  // A legacy counted distance of zero disables all SWP paths for this loop.
  D.Enable &= D.Distance != 0;
  return D;
}

FissionSettings LoopOptions::resolveIRFission(bool Disabled) const {
  return {!Disabled && IRFissionEnable.value_or(true), Force.value_or(false),
          QueueDepth.value_or(64)};
}

FissionSettings LoopOptions::resolveMachineFission(unsigned DefaultDepth,
                                                   bool Disabled) const {
  unsigned Depth = QueueDepth.value_or(DefaultDepth);
  return {!Disabled && MachineFissionEnable.value_or(true) && Depth != 0,
          Force.value_or(false), std::min(Depth, 64u)};
}

std::optional<LoopOptions>
llvm::RISCVNS::parseLoopOptions(const MDNode *ID, const Instruction &Latch) {
  LoopOptions Result;
  if (!ID)
    return Result;
  StringSet<> Seen;
  bool Valid = true;
  auto Error = [&](StringRef Name, const Twine &Expected) {
    std::string Block;
    raw_string_ostream OS(Block);
    Latch.getParent()->printAsOperand(OS, false);
    DiagnosticLocation Location(Latch.getDebugLoc());
    if (!Location.isValid())
      for (unsigned I = 1; I < ID->getNumOperands(); ++I)
        if (auto *Loc = dyn_cast_or_null<DILocation>(ID->getOperand(I))) {
          Location = DiagnosticLocation(DebugLoc(Loc));
          break;
        }
    if (!Location.isValid())
      Location = DiagnosticLocation(Latch.getFunction()->getSubprogram());
    Latch.getContext().diagnose(DiagnosticInfoUnsupported(
        *Latch.getFunction(),
        "invalid Non-Spec loop metadata '" + Name + "' in function '" +
            Latch.getFunction()->getName() + "', loop latch " + Block + ": " +
            Expected,
        Location, DS_Error));
    Valid = false;
  };
  if (!ID->isDistinct() || ID->getNumOperands() == 0 ||
      ID->getOperand(0) != ID) {
    for (const MDOperand &MD : ID->operands())
      if (lookupOption(propertyName(MD)))
        Error(propertyName(MD),
              "expected a distinct, self-referencing !llvm.loop ID");
    if (!Valid)
      return std::nullopt;
  }
  for (unsigned I = 1; I < ID->getNumOperands(); ++I) {
    StringRef Name = propertyName(ID->getOperand(I));
    const Option *O = lookupOption(Name);
    if (!O)
      continue;
    std::string Form = O->Boolean
                           ? "expected !{!\"" + Name.str() + "\", i1 <boolean>}"
                           : "expected !{!\"" + Name.str() +
                                 "\", i32 <integer in [1, " +
                                 std::to_string(O->Max) + "]>}";
    if (!Seen.insert(Name).second) {
      Error(Name, "duplicate key; " + Form);
      continue;
    }
    auto *Node = dyn_cast_or_null<MDNode>(ID->getOperand(I));
    if (!Node || Node->getNumOperands() != 2) {
      Error(Name, Form);
      continue;
    }
    auto *Value =
        mdconst::dyn_extract_or_null<ConstantInt>(Node->getOperand(1));
    if (!Value || !Value->getType()->isIntegerTy(O->Boolean ? 1 : 32) ||
        (O->Number && (Value->isZero() || Value->getZExtValue() > O->Max))) {
      Error(Name, Form);
      continue;
    }
    if (O->Boolean)
      Result.*O->Boolean = !Value->isZero();
    else
      Result.*O->Number = Value->getZExtValue();
  }
  if (!Valid)
    return std::nullopt;
  return Result;
}

bool llvm::RISCVNS::validateLoopOptions(const Function &F) {
  SmallPtrSet<MDNode *, 8> Seen;
  bool Valid = true;
  for (const BasicBlock &BB : F)
    if (const Instruction *Term = BB.getTerminator())
      if (MDNode *ID = Term->getMetadata(LLVMContext::MD_loop))
        if (Seen.insert(ID).second)
          Valid &= parseLoopOptions(ID, *Term).has_value();
  return Valid;
}

bool llvm::RISCVNS::isIRFissioned(const MDNode *ID) {
  if (ID)
    for (unsigned I = 1; I < ID->getNumOperands(); ++I)
      if (propertyName(ID->getOperand(I)) == "riscv.loop-fissioned")
        return true;
  return false;
}

StringRef llvm::RISCVNS::getSWPApplied(const MDNode *ID) {
  if (ID)
    for (unsigned I = 1; I < ID->getNumOperands(); ++I)
      if (propertyName(ID->getOperand(I)) == "llvm.loop.riscv.ns.swp.applied") {
        auto *Node = dyn_cast_or_null<MDNode>(ID->getOperand(I));
        if (Node && Node->getNumOperands() == 2)
          if (auto *Role = dyn_cast_or_null<MDString>(Node->getOperand(1)))
            return Role->getString();
      }
  return {};
}

bool llvm::RISCVNS::hasLoopControls(const MDNode *ID) {
  if (ID)
    for (unsigned I = 1; I < ID->getNumOperands(); ++I)
      if (lookupOption(propertyName(ID->getOperand(I))))
        return true;
  return isIRFissioned(ID) || !getSWPApplied(ID).empty();
}

MDNode *llvm::RISCVNS::makeLoopID(LLVMContext &Context, const MDNode *OldID,
                                  LoopRole Role) {
  SmallVector<Metadata *, 8> MDs(1, nullptr);
  if (OldID)
    for (unsigned I = 1; I < OldID->getNumOperands(); ++I) {
      Metadata *MD = OldID->getOperand(I);
      StringRef Name = propertyName(MD);
      if (Name == "riscv.loop-fissioned" ||
          Name == "llvm.loop.riscv.ns.swp.applied")
        continue;
      if (lookupOption(Name) &&
          !(Role == LoopRole::Drain &&
            (Name == "llvm.loop.riscv.ns.fission.machine.enable" ||
             Name == "llvm.loop.riscv.ns.fission.queue_depth" ||
             Name == "llvm.loop.riscv.ns.force")))
        continue;
      MDs.push_back(MD);
    }
  if (Role == LoopRole::Fission) {
    MDs.push_back(
        MDNode::get(Context, {MDString::get(Context, "riscv.loop-fissioned")}));
  } else {
    StringRef RoleName = Role == LoopRole::Kernel  ? "kernel"
                         : Role == LoopRole::Drain ? "drain"
                                                   : "inplace";
    MDs.push_back(MDNode::get(
        Context, {MDString::get(Context, "llvm.loop.riscv.ns.swp.applied"),
                  MDString::get(Context, RoleName)}));
  }
  MDNode *ID = MDNode::getDistinct(Context, MDs);
  ID->replaceOperandWith(0, ID);
  return ID;
}

INITIALIZE_PASS(RISCVNSLoopMetadata, "riscv-ns-loop-metadata",
                "RISC-V Non-Spec loop metadata validation", false, true)

FunctionPass *llvm::createRISCVNSLoopMetadataPass() {
  return new RISCVNSLoopMetadata();
}
