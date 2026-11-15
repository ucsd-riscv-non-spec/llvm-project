//===- RISCVSelectionDAGTest.cpp - RISC-V DAG tests --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RISCVISelLowering.h"
#include "RISCVSubtarget.h"
#include "RISCVTargetMachine.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

class RISCVSelectionDAGTest : public testing::TestWithParam<const char *> {
protected:
  LLVMContext Context;
  std::unique_ptr<RISCVTargetMachine> TM;
  std::unique_ptr<Module> M;
  std::unique_ptr<MachineModuleInfo> MMI;
  std::unique_ptr<MachineFunction> MF;
  std::unique_ptr<OptimizationRemarkEmitter> ORE;
  std::unique_ptr<SelectionDAG> DAG;

  static void SetUpTestSuite() {
    LLVMInitializeRISCVTargetInfo();
    LLVMInitializeRISCVTarget();
    LLVMInitializeRISCVTargetMC();
  }

  void SetUp() override {
    Triple TT(GetParam());
    std::string Error;
    const Target *T = TargetRegistry::lookupTarget(TT, Error);
    ASSERT_NE(T, nullptr);
    TM.reset(static_cast<RISCVTargetMachine *>(T->createTargetMachine(
        TT, "generic", "", TargetOptions(), std::nullopt)));
    M = std::make_unique<Module>("test", Context);
    M->setDataLayout(TM->createDataLayout());
    auto *FT = FunctionType::get(Type::getVoidTy(Context), false);
    auto *F = Function::Create(FT, GlobalValue::ExternalLinkage, "test", *M);
    MMI = std::make_unique<MachineModuleInfo>(TM.get());
    MF = std::make_unique<MachineFunction>(*F, *TM, *TM->getSubtargetImpl(*F),
                                           MMI->getContext(), 0);
    ORE = std::make_unique<OptimizationRemarkEmitter>(F);
    DAG = std::make_unique<SelectionDAG>(*TM, CodeGenOptLevel::Default);
    DAG->init(*MF, *ORE, nullptr, nullptr, nullptr, nullptr, nullptr, *MMI,
              nullptr, false);
  }

  MVT getVT() const { return MF->getSubtarget<RISCVSubtarget>().getXLenVT(); }

  SDValue reg(unsigned Reg, EVT VT) {
    return DAG->getCopyFromReg(DAG->getEntryNode(), SDLoc(), Reg, VT);
  }

  SDValue lowerSelect(SDValue T, SDValue F) {
    MVT VT = getVT();
    SDValue C = DAG->getNode(ISD::AND, SDLoc(), VT, reg(RISCV::X10, VT),
                             DAG->getConstant(1, SDLoc(), VT));
    SDValue Select = DAG->getSelect(SDLoc(), VT, C, T, F);
    EXPECT_EQ(Select.getOpcode(), ISD::SELECT);
    return DAG->getTargetLoweringInfo().LowerOperation(Select, *DAG);
  }
};

TEST_P(RISCVSelectionDAGTest,
       BranchlessSelectDoesNotPropagateUnselectedPoison) {
  MVT VT = getVT();
  SDLoc DL;
  SDValue X = reg(RISCV::X11, VT);
  SDValue Y = reg(RISCV::X12, VT);
  SDNodeFlags Flags;
  Flags.setNoSignedWrap(true);
  SDValue MaybePoison = DAG->getNode(ISD::ADD, DL, VT, X, Y, Flags);
  ASSERT_FALSE(DAG->isGuaranteedNotToBeUndefOrPoison(MaybePoison));

  // Test either arm independently: poison in the unchosen arm must not leak
  // through the bitwise blend. Assembly tests cannot observe these freezes.
  for (bool PoisonOnTrue : {false, true}) {
    SDValue Result = PoisonOnTrue ? lowerSelect(MaybePoison, Y)
                                  : lowerSelect(Y, MaybePoison);
    EXPECT_TRUE(DAG->isGuaranteedNotToBeUndefOrPoison(Result));
  }
}

TEST_P(RISCVSelectionDAGTest, BranchlessSelectFreezesDuplicatedUndef) {
  MVT VT = getVT();
  SDLoc DL;
  SDValue T = reg(RISCV::X11, VT);
  SDValue F = DAG->getNode(ISD::ANY_EXTEND, DL, VT, reg(RISCV::X12, MVT::i8));
  ASSERT_FALSE(DAG->isGuaranteedNotToBeUndefOrPoison(F));
  SDValue Result = lowerSelect(T, F);
  ASSERT_EQ(Result.getOpcode(), ISD::XOR);

  // The *same* frozen value must cancel when the true arm is chosen; two
  // independent uses of ANY_EXTEND may choose different upper bits.
  SDValue FrozenF = DAG->getFreeze(F);
  EXPECT_EQ(Result.getOperand(0), FrozenF);
  ASSERT_EQ(Result.getOperand(1).getOpcode(), ISD::AND);
  SDValue Diff = Result.getOperand(1).getOperand(0);
  ASSERT_EQ(Diff.getOpcode(), ISD::XOR);
  EXPECT_EQ(Diff.getOperand(1), FrozenF);
}

INSTANTIATE_TEST_SUITE_P(RISCV, RISCVSelectionDAGTest,
                         testing::Values("riscv32", "riscv64"));

} // namespace
