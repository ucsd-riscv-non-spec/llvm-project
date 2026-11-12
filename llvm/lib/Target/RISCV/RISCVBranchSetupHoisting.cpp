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
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/Support/CommandLine.h"

#define DEBUG_TYPE "riscv-branch-setup-hoisting"
#define PASS_NAME "RISC-V branch setup hoisting pass"

using namespace llvm;

static cl::opt<bool> DisableBranchSetupHoisting("riscv-ns-disable-branch-setup-hoisting",
                                                cl::Hidden,
                                                cl::desc("Disable " PASS_NAME),
                                                cl::init(false));

static cl::opt<bool>
    DisableCrossBBHoisting("riscv-ns-disable-cross-bb-hoisting", cl::Hidden,
                           cl::desc("Disable Cross BB Hoisting"),
                           cl::init(false));

static cl::opt<unsigned> MaxHoistFreqRatio(
    "riscv-ns-hoist-max-freq-ratio", cl::Hidden,
    cl::desc("Do not hoist a branch setup into a block expected to run more "
             "than this many times as often as its own (0 for no limit)"),
    cl::init(2));

static cl::opt<bool> DisableHoistFreqFromBest(
    "riscv-ns-disable-hoist-freq-from-best", cl::Hidden, cl::init(false),
    cl::desc("Compare a hoist point's frequency with the setup's own block, "
             "instead of with the coolest point it was already hoisted to"));

static cl::opt<bool> DisableBranchRegHints(
    "riscv-ns-disable-branch-reg-hints", cl::Hidden,
    cl::desc("Leave the B-register of each branch to the register allocator, "
             "without giving it a preferred one"),
    cl::init(false));

STATISTIC(NumBMOVHoisted, "Number of BMOV instructions able to be hoisted");
STATISTIC(NumBMOVSLimited,
          "Number of BMOVS hoisted less far to keep enough B-registers free");
STATISTIC(NumBRegHints, "Number of branches given a preferred B-register");
STATISTIC(NumHoistPointsTooHotFromBest,
          "Number of hoist points rejected for running too often compared "
          "with a point the setup was already hoisted to");

namespace {

/// Where each B-register is live, recorded at the boundaries of the blocks.
///
/// Every B-register the register allocator could assign has an id: B<i> is i,
/// and a virtual one is NumPhysRegs plus its index. Reserved ones have none,
/// as no branch can be given one.
class BRegLiveness {
public:
  static constexpr unsigned NumPhysRegs = 32;

  BRegLiveness(const MachineFunction &MF, const TargetRegisterInfo &TRI);

  /// Returns the id of \p Reg, or -1 if it is not a B-register the register
  /// allocator could assign.
  int getId(Register Reg) const;
  static Register getReg(unsigned Id) {
    return Id < NumPhysRegs ? Register(RISCV::B0 + Id)
                            : Register::index2VirtReg(Id - NumPhysRegs);
  }
  static bool isPhysId(unsigned Id) { return Id < NumPhysRegs; }

  /// Appends the ids of the B-registers \p MI reads and those it writes, once
  /// each.
  void getOperands(const MachineInstr &MI, SmallVectorImpl<unsigned> &Uses,
                   SmallVectorImpl<unsigned> &Defs) const;

  /// Recomputes where \p Reg is live, after an instruction using it moved.
  void update(Register Reg) { compute(Reg, /*Clear=*/true); }

  const BitVector &getLiveIn(const MachineBasicBlock &MBB) const {
    return LiveIn[MBB.getNumber()];
  }
  const BitVector &getLiveOut(const MachineBasicBlock &MBB) const {
    return LiveOut[MBB.getNumber()];
  }

  /// Number of B-registers the register allocator may assign.
  unsigned getNumAssignable() const { return NumAssignable; }
  /// Number of those the function reads or writes, virtual or physical.
  unsigned getNumUsed() const { return NumUsed; }

  /// Returns the most B-registers live at once in the blocks where \p Reg is
  /// live.
  unsigned getMaxPressureAround(Register Reg) const;

private:
  void compute(Register Reg, bool Clear);
  unsigned getMaxPressure(const MachineBasicBlock &MBB) const;

  const MachineFunction &MF;
  const MachineRegisterInfo &MRI;
  BitVector Reserved;
  unsigned NumAssignable = 0;
  unsigned NumUsed = 0;
  /// By block number, the ids live on entry to the block and on exit.
  SmallVector<BitVector, 0> LiveIn, LiveOut;
};

/// A place to move a setup to: a block and the instruction to insert before,
/// which may be that block's end().
using HoistPoint = std::pair<MachineBasicBlock *, MachineBasicBlock::iterator>;

class RISCVBranchSetupHoisting : public MachineFunctionPass {
public:
  static char ID;
  const RISCVBranchSetupInfo *BSI;
  const TargetRegisterInfo *TRI;
  MachineDominatorTree *MDT;
  MachineLoopInfo *MLI;
  MachineBlockFrequencyInfo *MBFI;

  RISCVBranchSetupHoisting() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<RISCVBranchSetupAnalysisWrapper>();
    AU.addRequired<MachineDominatorTreeWrapperPass>();
    AU.addRequired<MachineLoopInfoWrapperPass>();
    AU.addRequired<MachineBlockFrequencyInfoWrapperPass>();
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override { return PASS_NAME; }

  bool hoistBranchSetups(MachineFunction &MF, BRegLiveness &Liveness);
  /// Gives the B-register of every branch a preferred physical register,
  /// picked so that the branches share as few as possible.
  bool hintBranchRegs(MachineFunction &MF, const BRegLiveness &Liveness) const;
  /// If \p LimitPressure is set, hoists \p S only as far as keeps the
  /// B-registers live at once within those the register allocator may assign.
  bool scheduleBranchSetup(MachineInstr *MI, MachineInstr *S, MachineInstr *T,
                           MachineInstr *C, BRegLiveness &Liveness,
                           bool LimitPressure);
  bool repairBranchTargets(const MachineFunction &MF) const;
  /// Returns the places \p SetupMI can be hoisted to, nearest first; each one
  /// dominates the one before it. The iterators may be their block's end(),
  /// so they must not be dereferenced to recover the block.
  /// \p OwnBranch, if set, is the branch reading \p SetupMI, which does not
  /// stop it from being hoisted.
  SmallVector<HoistPoint, 8> findHoistPoints(MachineInstr &SetupMI,
                                             const MachineInstr *OwnBranch);
  /// Returns how many of \p Points \p S can be hoisted through while the
  /// B-registers live at once still fit in those that may be assigned.
  unsigned countFittingPoints(MachineInstr &S, ArrayRef<HoistPoint> Points,
                              BRegLiveness &Liveness) const;
  bool isSafeToHoistTo(const MachineInstr &SetupMI, MachineBasicBlock &DestBB,
                       MachineBasicBlock::const_iterator InsertPt,
                       const MachineInstr *OwnBranch) const;
};

char RISCVBranchSetupHoisting::ID = 0;

bool RISCVBranchSetupHoisting::repairBranchTargets(
    const MachineFunction &MF) const {
  DenseMap<MachineInstr *, MachineBasicBlock *> Targets;
  SmallPtrSet<MachineInstr *, 16> ConflictingTargets;

  for (const auto &[Branch, Setup] : BSI->Branches) {
    if (!Setup.T || Setup.T->getOpcode() != RISCV::BMOVT_J ||
        Branch->getNumExplicitOperands() < 2 ||
        !Branch->getOperand(1).isMBB())
      continue;

    MachineInstr *TargetSetup = const_cast<MachineInstr *>(Setup.T);
    MachineBasicBlock *Target = Branch->getOperand(1).getMBB();
    auto [It, Inserted] = Targets.try_emplace(TargetSetup, Target);
    if (!Inserted && It->second != Target)
      ConflictingTargets.insert(TargetSetup);
  }

  bool Changed = false;
  for (const auto &[TargetSetup, Target] : Targets) {
    if (ConflictingTargets.contains(TargetSetup))
      report_fatal_error("RISC-V branch setup is shared by branches with "
                         "different targets");

    MachineOperand &TargetOperand = TargetSetup->getOperand(2);
    if (!TargetOperand.isMBB() || TargetOperand.getMBB() == Target)
      continue;

    assert(TargetSetup->getParent()->getParent() == &MF &&
           "branch setup must belong to the current machine function");
    TargetOperand.setMBB(Target);
    Changed = true;
  }
  return Changed;
}

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

/// Whether \p MI appends to its B-register's condition FIFO.
static bool pushesCondition(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  case RISCV::BMOVC_BEQ:
  case RISCV::BMOVC_BNE:
  case RISCV::BMOVC_BLT:
  case RISCV::BMOVC_BGE:
  case RISCV::BMOVC_BLTU:
  case RISCV::BMOVC_BGEU:
  case RISCV::BMOVC_BITS:
    return true;
  default:
    return false;
  }
}

BRegLiveness::BRegLiveness(const MachineFunction &MF,
                           const TargetRegisterInfo &TRI)
    : MF(MF), MRI(MF.getRegInfo()), Reserved(TRI.getReservedRegs(MF)) {
  assert(RISCV::PBRRegClass.getNumRegs() == NumPhysRegs &&
         "unexpected number of B-registers");
  unsigned NumIds = NumPhysRegs + MRI.getNumVirtRegs();
  LiveIn.assign(MF.getNumBlockIDs(), BitVector(NumIds));
  LiveOut.assign(MF.getNumBlockIDs(), BitVector(NumIds));

  for (unsigned Id = 0; Id != NumPhysRegs; ++Id) {
    assert(getReg(Id) == Register(RISCV::PBRRegClass.getRegister(Id)) &&
           "B-registers are not numbered in order");
    if (getId(getReg(Id)) < 0)
      continue;
    ++NumAssignable;
    if (!MRI.reg_nodbg_empty(getReg(Id))) {
      ++NumUsed;
      compute(getReg(Id), /*Clear=*/false);
    }
  }
  for (unsigned I = 0, E = MRI.getNumVirtRegs(); I != E; ++I) {
    Register Reg = Register::index2VirtReg(I);
    if (getId(Reg) >= 0 && !MRI.reg_nodbg_empty(Reg)) {
      ++NumUsed;
      compute(Reg, /*Clear=*/false);
    }
  }
}

int BRegLiveness::getId(Register Reg) const {
  if (Reg.isVirtual()) {
    const TargetRegisterClass *RC = MRI.getRegClassOrNull(Reg);
    if (!RC || !RISCV::PBRRegClass.hasSubClassEq(RC))
      return -1;
    return NumPhysRegs + Reg.virtRegIndex();
  }
  if (!Reg.isPhysical() || !RISCV::PBRRegClass.contains(Reg) ||
      Reserved.test(Reg))
    return -1;
  return Reg.id() - RISCV::B0;
}

void BRegLiveness::getOperands(const MachineInstr &MI,
                               SmallVectorImpl<unsigned> &Uses,
                               SmallVectorImpl<unsigned> &Defs) const {
  for (const MachineOperand &MO : MI.operands()) {
    if (!MO.isReg())
      continue;
    int Id = getId(MO.getReg());
    if (Id < 0)
      continue;
    SmallVectorImpl<unsigned> &Ids = MO.isDef() ? Defs : Uses;
    if ((MO.isDef() || MO.readsReg()) && !llvm::is_contained(Ids, Id))
      Ids.push_back(Id);
  }
}

void BRegLiveness::compute(Register Reg, bool Clear) {
  unsigned Id = getId(Reg);
  if (Clear) {
    for (BitVector &Live : LiveIn)
      Live.reset(Id);
    for (BitVector &Live : LiveOut)
      Live.reset(Id);
  }

  // For each block using Reg, whether it reads the value Reg enters with, and
  // whether it writes Reg without reading it, which ends that value.
  SmallPtrSet<const MachineBasicBlock *, 8> Seen, Overwrites;
  SmallVector<const MachineBasicBlock *, 16> WorkList;
  for (const MachineInstr &RefMI : MRI.reg_nodbg_instructions(Reg)) {
    const MachineBasicBlock *MBB = RefMI.getParent();
    if (!Seen.insert(MBB).second)
      continue;
    bool Overwritten = false, ReadOnEntry = false;
    for (const MachineInstr &MI : *MBB) {
      if (MI.isDebugInstr())
        continue;
      bool Reads = false, Writes = false;
      for (const MachineOperand &MO : MI.operands()) {
        if (!MO.isReg() || MO.getReg() != Reg)
          continue;
        if (MO.isDef())
          Writes = true;
        else if (MO.readsReg())
          Reads = true;
      }
      ReadOnEntry |= Reads && !Overwritten;
      Overwritten |= Writes && !Reads;
    }
    if (Overwritten)
      Overwrites.insert(MBB);
    if (ReadOnEntry) {
      LiveIn[MBB->getNumber()].set(Id);
      WorkList.push_back(MBB);
    }
  }

  while (!WorkList.empty()) {
    const MachineBasicBlock *MBB = WorkList.pop_back_val();
    for (const MachineBasicBlock *Pred : MBB->predecessors()) {
      LiveOut[Pred->getNumber()].set(Id);
      if (!Overwrites.contains(Pred) && !LiveIn[Pred->getNumber()].test(Id)) {
        LiveIn[Pred->getNumber()].set(Id);
        WorkList.push_back(Pred);
      }
    }
  }
}

unsigned BRegLiveness::getMaxPressure(const MachineBasicBlock &MBB) const {
  BitVector Live = getLiveOut(MBB);
  unsigned NumLive = Live.count();
  unsigned MaxLive = NumLive;
  SmallVector<unsigned, 4> Uses, Defs;
  for (const MachineInstr &MI : llvm::reverse(MBB)) {
    if (MI.isDebugInstr())
      continue;
    Uses.clear();
    Defs.clear();
    getOperands(MI, Uses, Defs);
    // A register written here needs to be free here, even if nothing reads
    // it afterwards.
    unsigned NumWritten =
        llvm::count_if(Defs, [&](unsigned Id) { return !Live.test(Id); });
    MaxLive = std::max(MaxLive, NumLive + NumWritten);
    for (unsigned Id : Defs)
      if (Live.test(Id)) {
        Live.reset(Id);
        --NumLive;
      }
    for (unsigned Id : Uses)
      if (!Live.test(Id)) {
        Live.set(Id);
        ++NumLive;
      }
    MaxLive = std::max(MaxLive, NumLive);
  }
  return MaxLive;
}

unsigned BRegLiveness::getMaxPressureAround(Register Reg) const {
  unsigned Id = getId(Reg);
  SmallPtrSet<const MachineBasicBlock *, 16> Blocks;
  for (const MachineInstr &MI : MRI.reg_nodbg_instructions(Reg))
    Blocks.insert(MI.getParent());
  for (const MachineBasicBlock &MBB : MF)
    if (getLiveIn(MBB).test(Id) || getLiveOut(MBB).test(Id))
      Blocks.insert(&MBB);

  unsigned MaxLive = 0;
  for (const MachineBasicBlock *MBB : Blocks)
    MaxLive = std::max(MaxLive, getMaxPressure(*MBB));
  return MaxLive;
}

bool RISCVBranchSetupHoisting::runOnMachineFunction(MachineFunction &MF) {
  if (DisableBranchSetupHoisting && DisableBranchRegHints)
    return false;

  LLVM_DEBUG(printMachineCFG(MF));

  TRI = MF.getSubtarget().getRegisterInfo();
  BSI = &getAnalysis<RISCVBranchSetupAnalysisWrapper>().getInfo();
  MDT = &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  MLI = &getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  MBFI = &getAnalysis<MachineBlockFrequencyInfoWrapperPass>().getMBFI();

  BRegLiveness Liveness(MF, *TRI);
  bool Changed = false;
  if (!DisableBranchSetupHoisting)
    Changed |= hoistBranchSetups(MF, Liveness);
  if (!DisableBranchRegHints)
    Changed |= hintBranchRegs(MF, Liveness);

  LLVM_DEBUG(printMachineCFG(MF));

  return Changed;
}

bool RISCVBranchSetupHoisting::hoistBranchSetups(MachineFunction &MF,
                                                 BRegLiveness &Liveness) {
  bool Changed = repairBranchTargets(MF);

  // Collect work in machine-function order, so the order the setups are
  // moved in does not depend on the iteration order of the analysis map.
  struct BranchWork {
    MachineInstr *MI;
    MachineInstr *S, *T, *C;
    /// Whether several BMOVCs fill the B-register's condition FIFO, as for a
    /// queue from RISCVBranchConditionPipelining. C is then only the BMOVC
    /// nearest the branch.
    bool IsQueue = false;
  };
  SmallVector<BranchWork, 16> WorkList;

  const MachineRegisterInfo &MRI = MF.getRegInfo();
  for (auto &MBB : MF) {
    for (auto &MI : MBB) {
      RISCVBranchSetup BS = BSI->Branches.lookup(&MI);
      if (!BS)
        continue;
      BranchWork W = {&MI, const_cast<MachineInstr *>(BS.S),
                      const_cast<MachineInstr *>(BS.T),
                      const_cast<MachineInstr *>(BS.C)};
      if (BS.S) {
        // RISCVExpandPseudo gives each branch a fresh virtual B-register, so
        // every instruction using it belongs to this branch. The main
        // register allocator assigns those of RISCVLoopFission's branches,
        // which are physical by now, and may give one to several branches.
        // Their setups cannot be moved for one branch alone, so a physical
        // register is only hoisted when no other branch uses it.
        Register Reg = BS.S->getOperand(0).getReg();
        if (Liveness.getId(Reg) < 0)
          continue;
        if (Reg.isPhysical() &&
            !llvm::all_of(MRI.reg_nodbg_instructions(Reg),
                          [&](const MachineInstr &RefMI) {
                            return llvm::is_contained({W.S, W.C, W.T, W.MI},
                                                      &RefMI);
                          }))
          continue;
        W.IsQueue =
            llvm::count_if(MRI.def_instructions(Reg), pushesCondition) > 1;
      }
      WorkList.push_back(W);
    }
  }

  // A setup shared by two branches cannot be scheduled for each of them
  // independently. Refuse the transformation instead of silently
  // miscompiling.
  SmallPtrSet<MachineInstr *, 32> ClaimedSetups;
  for (const BranchWork &W : WorkList) {
    if (!W.S || !W.T)
      return Changed;
    for (MachineInstr *SetupMI : {W.S, W.C, W.T})
      if (SetupMI && !ClaimedSetups.insert(SetupMI).second)
        return Changed;
  }

  // Hoisting lengthens the live ranges of the B-registers, and the register
  // allocator that assigns them runs after frame finalization, so it cannot
  // spill. If the function uses more B-registers than may be assigned, each
  // BMOVS is hoisted only as far as keeps the number live at once within
  // that.
  bool LimitPressure = Liveness.getNumUsed() > Liveness.getNumAssignable();

  SmallVector<BranchWork *, 16> Order;
  for (BranchWork &W : llvm::reverse(WorkList))
    Order.push_back(&W);
  // The setups hoisted first take the registers that are left, so start with
  // the branches that run most often.
  if (LimitPressure)
    llvm::stable_sort(Order, [&](const BranchWork *A, const BranchWork *B) {
      return MBFI->getBlockFreq(A->MI->getParent()) >
             MBFI->getBlockFreq(B->MI->getParent());
    });

  for (BranchWork *W : Order) {
    // A queue's BMOVCs run iterations ahead of its branch, and its BMOVS and
    // BMOVT empty the FIFO, so they must stay where the pipeliner put them:
    // moved above another entry to the loop, they would leave the previous
    // run's surplus conditions queued.
    if (W->IsQueue)
      continue;
    Changed |= scheduleBranchSetup(W->MI, W->S, W->T, W->C, Liveness,
                                   LimitPressure);
  }

  return Changed;
}

/// Whether \p MI is a branch reading the B-register in its first operand.
static bool isPB(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  case RISCV::PseudoPBC:
  case RISCV::PseudoPBU:
  case RISCV::PseudoPBI:
  case RISCV::PseudoPBCALL:
    return true;
  default:
    return false;
  }
}

bool RISCVBranchSetupHoisting::hintBranchRegs(
    MachineFunction &MF, const BRegLiveness &Liveness) const {
  // At fetch, a PB is predicted from the B-register whose source address is
  // the PB's own. Each BMOVS of a register replaces that address, so a branch
  // that shares its register is not predicted once another branch has set
  // it up, as happens every iteration when the two run in turn in a loop.
  // Left alone, the register allocator takes the lowest free register, which
  // gives most branches the same few, so each branch is given its own as a
  // hint. The hints are colored to be a valid assignment, which the
  // allocator then follows. A branch left without one, as when every
  // register it could take is in use, gets whichever is free.
  //
  // Colors are the indices of B0-B31.
  constexpr unsigned NumColors = BRegLiveness::NumPhysRegs;
  MachineRegisterInfo &MRI = MF.getRegInfo();

  BitVector Assignable(NumColors), CalleeSaved(NumColors);
  for (unsigned C = 0; C != NumColors; ++C)
    if (Liveness.getId(BRegLiveness::getReg(C)) >= 0)
      Assignable.set(C);
  for (const MCPhysReg *R = MRI.getCalleeSavedRegs(); *R; ++R)
    if (RISCV::PBRRegClass.contains(*R))
      CalleeSaved.set(*R - RISCV::B0);

  // The i-th branch in layout order prefers the i-th caller-saved register,
  // as this pass once renamed them, so that no two branches share one until
  // there are more branches than registers. The callee-saved ones are left
  // to the B-registers live across calls, which can only use those.
  SmallVector<unsigned, NumColors> PreferredOrder;
  for (unsigned C : Assignable.set_bits())
    if (!CalleeSaved.test(C))
      PreferredOrder.push_back(C);
  DenseMap<Register, unsigned> Preferred;
  if (!PreferredOrder.empty())
    for (const MachineBasicBlock &MBB : MF)
      for (const MachineInstr &MI : MBB) {
        if (!isPB(MI))
          continue;
        Register Reg = MI.getOperand(0).getReg();
        if (Reg.isVirtual() && !Preferred.contains(Reg)) {
          unsigned C = PreferredOrder[Preferred.size() % PreferredOrder.size()];
          Preferred[Reg] = C;
        }
      }

  // The colors each virtual B-register cannot take: those of the physical
  // ones live anywhere it is, and for one live across a call, those the call
  // clobbers.
  DenseMap<unsigned, BitVector> Forbidden;
  DenseSet<unsigned> AcrossCall;
  auto forbid = [&](unsigned Id, const BitVector &Colors) {
    Forbidden.try_emplace(Id, NumColors).first->second |= Colors;
  };
  // Notes the physical B-registers among \p Regs, all live at one point, as
  // forbidden to the virtual ones among them.
  auto noteOverlap = [&](const BitVector &Regs) {
    int First = Regs.find_first();
    if (First < 0 || !BRegLiveness::isPhysId(First))
      return;
    BitVector Phys(NumColors);
    for (unsigned Id : Regs.set_bits()) {
      if (!BRegLiveness::isPhysId(Id))
        break;
      Phys.set(Id);
    }
    for (int Id = Regs.find_next(NumColors - 1); Id >= 0;
         Id = Regs.find_next(Id))
      forbid(Id, Phys);
  };

  SmallVector<unsigned, 4> Uses, Defs;
  for (const MachineBasicBlock &MBB : MF) {
    BitVector Live = Liveness.getLiveOut(MBB);
    noteOverlap(Live);
    for (const MachineInstr &MI : llvm::reverse(MBB)) {
      if (MI.isDebugInstr())
        continue;
      Uses.clear();
      Defs.clear();
      Liveness.getOperands(MI, Uses, Defs);

      BitVector LiveOrWritten = Live;
      for (unsigned Id : Defs)
        LiveOrWritten.set(Id);
      noteOverlap(LiveOrWritten);

      for (const MachineOperand &MO : MI.operands()) {
        if (!MO.isRegMask())
          continue;
        BitVector Clobbered(NumColors);
        for (unsigned C = 0; C != NumColors; ++C)
          if (MO.clobbersPhysReg(BRegLiveness::getReg(C)))
            Clobbered.set(C);
        for (unsigned Id : Live.set_bits()) {
          if (BRegLiveness::isPhysId(Id) || llvm::is_contained(Defs, Id))
            continue;
          AcrossCall.insert(Id);
          forbid(Id, Clobbered);
        }
      }

      for (unsigned Id : Defs)
        Live.reset(Id);
      for (unsigned Id : Uses)
        Live.set(Id);
      noteOverlap(Live);
    }
  }

  // Color the virtual B-registers at their definitions, walking the
  // dominator tree in preorder. Each one's BMOVS dominates where it is live,
  // so every register live there is colored before it, and as hoisting keeps
  // no more of them live at once than there are registers to assign, a free
  // color is always left.
  DenseMap<Register, unsigned> Color;
  SmallVector<unsigned, NumColors> LastPicked(NumColors, 0);
  unsigned Clock = 0;
  auto getColor = [&](unsigned Id) -> int {
    if (BRegLiveness::isPhysId(Id))
      return Id;
    auto It = Color.find(BRegLiveness::getReg(Id));
    return It == Color.end() ? -1 : int(It->second);
  };
  auto pickColor = [&](Register Reg, unsigned Id,
                       ArrayRef<unsigned> NumHolding) -> int {
    BitVector Free = Assignable;
    for (unsigned C = 0; C != NumColors; ++C)
      if (NumHolding[C])
        Free.reset(C);
    if (auto It = Forbidden.find(Id); It != Forbidden.end())
      Free.reset(It->second);
    if (Free.none())
      return -1;
    bool LiveAcrossCall = AcrossCall.contains(Id);

    auto PreferredIt = Preferred.find(Reg);
    if (PreferredIt != Preferred.end() && Free.test(PreferredIt->second))
      return PreferredIt->second;

    // Otherwise take the free one picked least recently, which the fewest
    // branches nearby share.
    int Best = -1;
    for (unsigned C : Free.set_bits()) {
      auto Key = [&](unsigned C) {
        return std::make_pair(!LiveAcrossCall && CalleeSaved.test(C),
                              LastPicked[C]);
      };
      if (Best < 0 || Key(C) < Key(Best))
        Best = C;
    }
    return Best;
  };

  bool Changed = false;
  DenseMap<const MachineInstr *, SmallVector<unsigned, 2>> Kills, DeadDefs;
  for (const MachineDomTreeNode *Node : depth_first(MDT->getRootNode())) {
    const MachineBasicBlock &MBB = *Node->getBlock();

    // What each instruction reads for the last time, and what it writes
    // that nothing reads.
    Kills.clear();
    DeadDefs.clear();
    BitVector Live = Liveness.getLiveOut(MBB);
    for (const MachineInstr &MI : llvm::reverse(MBB)) {
      if (MI.isDebugInstr())
        continue;
      Uses.clear();
      Defs.clear();
      Liveness.getOperands(MI, Uses, Defs);
      for (unsigned Id : Uses)
        if (!Live.test(Id))
          Kills[&MI].push_back(Id);
      for (unsigned Id : Defs)
        if (!Live.test(Id))
          DeadDefs[&MI].push_back(Id);
      for (unsigned Id : Defs)
        Live.reset(Id);
      for (unsigned Id : Uses)
        Live.set(Id);
    }

    // How many of the live B-registers hold each color.
    SmallVector<unsigned, NumColors> NumHolding(NumColors, 0);
    Live = Liveness.getLiveIn(MBB);
    for (unsigned Id : Live.set_bits())
      if (int C = getColor(Id); C >= 0)
        ++NumHolding[C];
    auto release = [&](unsigned Id) {
      if (!Live.test(Id))
        return;
      Live.reset(Id);
      if (int C = getColor(Id); C >= 0)
        --NumHolding[C];
    };

    for (const MachineInstr &MI : MBB) {
      if (MI.isDebugInstr())
        continue;
      Uses.clear();
      Defs.clear();
      Liveness.getOperands(MI, Uses, Defs);
      for (unsigned Id : Kills.lookup(&MI))
        release(Id);
      for (unsigned Id : Defs) {
        if (Live.test(Id))
          continue;
        Live.set(Id);
        Register Reg = BRegLiveness::getReg(Id);
        if (!BRegLiveness::isPhysId(Id) && getColor(Id) < 0) {
          int C = pickColor(Reg, Id, NumHolding);
          if (C >= 0) {
            Color[Reg] = C;
            LastPicked[C] = ++Clock;
            MRI.setSimpleHint(Reg, BRegLiveness::getReg(C));
            ++NumBRegHints;
            Changed = true;
          }
        }
        if (int C = getColor(Id); C >= 0)
          ++NumHolding[C];
      }
      for (unsigned Id : DeadDefs.lookup(&MI))
        release(Id);
    }
  }
  return Changed;
}

} // end of anonymous namespace

static bool isHoistBarrier(const MachineInstr &MI) {
  if (MI.isBMOV())
    return false;
  // A callee-saved B-register holds the caller's value until the prologue
  // saves it, so a setup above the save could not be given that register,
  // and a B-register live across a call has to get one.
  if (MI.getOpcode() == RISCV::BPUSH && MI.getFlag(MachineInstr::FrameSetup))
    return true;
  // isBarrier() means that control cannot fall through; it is not a generic
  // instruction-motion barrier. Calls, inline asm, and unmodelled effects are
  // the boundaries relevant to these branch-setup register writes.
  return MI.isCall() || MI.isIndirectBranch() || MI.isInlineAsm() ||
         MI.hasUnmodeledSideEffects();
}

/// Whether each condition pushed for Branch is consumed by that branch,
/// leaving an empty FIFO without repeating its BMOVS/BMOVT resets.
static bool hasPairedCondition(const MachineInstr &Branch,
                               const MachineInstr &S, const MachineInstr &T,
                               const MachineInstr *C) {
  if (Branch.getOpcode() != RISCV::PseudoPBC || !C || !pushesCondition(*C) ||
      C->getParent() != Branch.getParent())
    return false;

  // In particular, exclude other producers, consumers, or resets of the
  // register. Counting static BMOVCs alone does not prove the FIFO is balanced.
  const MachineRegisterInfo &MRI = Branch.getMF()->getRegInfo();
  Register Reg = S.getOperand(0).getReg();
  if (!llvm::all_of(MRI.reg_nodbg_instructions(Reg),
                    [&](const MachineInstr &MI) {
                      return llvm::is_contained({&S, &T, C, &Branch}, &MI);
                    }))
    return false;

  // A preceding conditional terminator could bypass this PB after the push.
  for (MachineBasicBlock::const_iterator I = std::next(C->getIterator()),
                                         E = Branch.getParent()->end();
       I != E; ++I) {
    if (&*I == &Branch)
      return true;
    if (I->isTerminator())
      break;
  }
  return false;
}

bool RISCVBranchSetupHoisting::scheduleBranchSetup(
    MachineInstr *MI, MachineInstr *S, MachineInstr *T, MachineInstr *C,
    BRegLiveness &Liveness, bool LimitPressure) {
  assert(S && "Branch must have BMOVS");
  assert(T && "Branch must have BMOVT");
  bool Changed = false;

  // PB preserves the source and target. An unconditional PB consumes nothing;
  // a conditional PB with a paired push leaves its FIFO empty, so neither
  // needs its source/target setup repeated on each iteration.
  bool PairedCondition =
      MLI->getLoopFor(MI->getParent()) && hasPairedCondition(*MI, *S, *T, C);
  const MachineInstr *OwnBranch =
      ((!C && MI->getOpcode() == RISCV::PseudoPBU) || PairedCondition)
          ? MI
          : nullptr;

  // Keep architectural setup order. The dependency checks also prevent a
  // later setup from being moved across an earlier setup of the same B-reg.
  for (MachineInstr *SetupMI : {S, C, T}) {
    if (!SetupMI)
      continue;

    // Once the per-iteration resets are removed, speculating even an invariant
    // condition push onto a path that bypasses PB could leave surplus bits.
    // Keep the paired push in place, and never exempt PB when moving BMOVC.
    if (SetupMI == C && PairedCondition)
      continue;

    LLVM_DEBUG(dbgs() << "Hoisting: " << *SetupMI);

    SmallVector<HoistPoint, 8> Points = findHoistPoints(*SetupMI, OwnBranch);
    unsigned NumPoints = Points.size();
    // The BMOVS starts the live range of the B-register, and the setups
    // after it cannot move above it, so only the BMOVS can lengthen it.
    if (SetupMI == S && LimitPressure && NumPoints) {
      NumPoints = countFittingPoints(*S, Points, Liveness);
      if (NumPoints < Points.size()) {
        ++NumBMOVSLimited;
        LLVM_DEBUG(dbgs() << "  [Limit] Hoisted " << NumPoints << " of "
                          << Points.size()
                          << " steps, to keep enough B-registers free\n");
      }
    }
    if (!NumPoints)
      continue;

    auto [DestBB, InsertPt] = Points[NumPoints - 1];
    DestBB->splice(InsertPt, SetupMI->getParent(), SetupMI);
    ++NumBMOVHoisted;
    Changed = true;
  }

  if (!Changed)
    return false;

  // The branches hoisted after this one, and the choice of the registers,
  // depend on where its B-register is now live.
  Register Reg = S->getOperand(0).getReg();
  Liveness.update(Reg);

  // The register allocator that follows fills in the live-in lists for the
  // registers it assigns, but a physical one now enters the blocks between
  // its new setup and its old one, which have to list it.
  MachineFunction &MF = *S->getMF();
  if (Reg.isPhysical() && MF.getRegInfo().tracksLiveness())
    for (MachineBasicBlock &MBB : MF)
      if (Liveness.getLiveIn(MBB).test(Liveness.getId(Reg)) &&
          !MBB.isLiveIn(Reg))
        MBB.addLiveIn(Reg);
  return true;
}

unsigned
RISCVBranchSetupHoisting::countFittingPoints(MachineInstr &S,
                                             ArrayRef<HoistPoint> Points,
                                             BRegLiveness &Liveness) const {
  Register Reg = S.getOperand(0).getReg();
  MachineBasicBlock *HomeBB = S.getParent();
  MachineBasicBlock::iterator HomeNext = std::next(S.getIterator());

  auto Fits = [&](unsigned NumPoints) {
    auto [DestBB, InsertPt] = Points[NumPoints - 1];
    DestBB->splice(InsertPt, S.getParent(), &S);
    Liveness.update(Reg);
    bool Result =
        Liveness.getMaxPressureAround(Reg) <= Liveness.getNumAssignable();
    HomeBB->splice(HomeNext, S.getParent(), &S);
    Liveness.update(Reg);
    return Result;
  };

  // Each point dominates the one before it, and every path from it to the
  // branch passes through that one, so hoisting further only adds to where
  // the register is live. Whether it fits can then be searched for.
  if (Fits(Points.size()))
    return Points.size();
  unsigned Lo = 0, Hi = Points.size(); // Lo fits and Hi does not.
  while (Hi - Lo > 1) {
    unsigned Mid = Lo + (Hi - Lo) / 2;
    (Fits(Mid) ? Lo : Hi) = Mid;
  }
  return Lo;
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
                          const TargetRegisterInfo &TRI,
                          const MachineInstr *OwnBranch) {
  if (&MI == OwnBranch)
    return false;
  return isHoistBarrier(MI) || hasRegisterDependency(MI, SetupMI, TRI);
}

/// Scan `BB` backwards from `From`, returning the earliest safe
/// insertion point. `HazardFound` is set if we stopped early.
/// Caller guarantees From != BB->rend().
static MachineBasicBlock::iterator
scanBlockBackward(MachineBasicBlock *BB,
                  MachineBasicBlock::reverse_iterator From,
                  const MachineInstr &SetupMI, const TargetRegisterInfo &TRI,
                  const MachineInstr *OwnBranch, bool &HazardFound) {
  HazardFound = false;

  // Try to hoist to the top of the func.
  MachineBasicBlock::iterator SafePoint = BB->getFirstNonPHI();

  for (auto I = From, E = BB->rend(); I != E; ++I) {
    MachineInstr &CurrMI = *I;

    if (CurrMI.isPHI())
      break;

    if (isHoistHazard(CurrMI, SetupMI, TRI, OwnBranch)) {
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

/// Whether \p MI, in a cycle that \p SetupMI would be hoisted out of, keeps
/// the setup in that cycle. The other setups of the same B-register do not:
/// they are hoisted with it, or checked on their own.
static bool isCycleHazard(const MachineInstr &MI, const MachineInstr &SetupMI,
                          const TargetRegisterInfo &TRI,
                          const MachineInstr *OwnBranch) {
  if (MI.isBMOV() &&
      MI.getOperand(0).getReg() == SetupMI.getOperand(0).getReg())
    return false;
  return isHoistHazard(MI, SetupMI, TRI, OwnBranch);
}

bool RISCVBranchSetupHoisting::isSafeToHoistTo(
    const MachineInstr &SetupMI, MachineBasicBlock &DestBB,
    MachineBasicBlock::const_iterator InsertPt,
    const MachineInstr *OwnBranch) const {
  const MachineBasicBlock *SrcBB = SetupMI.getParent();
  if (SrcBB == &DestBB || !MDT->dominates(&DestBB, SrcBB))
    return false;
  bool SrcInCycle = false;

  auto rangeHasHazard = [&](MachineBasicBlock::const_iterator Begin,
                            MachineBasicBlock::const_iterator End) {
    return llvm::any_of(
        llvm::make_range(Begin, End), [&](const MachineInstr &MI) {
          return &MI != &SetupMI &&
                 isHoistHazard(MI, SetupMI, *TRI, OwnBranch);
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
      // SrcBB reaches itself again without passing DestBB.
      if (Pred == SrcBB)
        SrcInCycle = true;
      if (Pred == &DestBB || MDT->dominates(&DestBB, Pred))
        WorkList.push_back(Pred);
    }
  }

  // The walk covers every block of a cycle around SrcBB except the part of
  // SrcBB after the setup, which the hoisted B-register would now stay live
  // through on every trip round. MachineLoopInfo reports only natural loops,
  // so loopHasHazard() does not see an irreducible cycle, such as one left by
  // tail duplication giving a loop a second entry.
  MachineBasicBlock::const_iterator AfterSetup = SetupMI.getIterator();
  ++AfterSetup;
  if (ReachedDest && SrcInCycle &&
      llvm::any_of(llvm::make_range(AfterSetup, SrcBB->end()),
                   [&](const MachineInstr &MI) {
                     return isCycleHazard(MI, SetupMI, *TRI, OwnBranch);
                   })) {
    LLVM_DEBUG(dbgs() << "  [Stop] Hazard after the setup in BB#"
                      << SrcBB->getNumber() << ", in a cycle not through BB#"
                      << DestBB.getNumber() << "\n");
    return false;
  }

  return ReachedDest;
}

static bool loopHasHazard(const MachineLoop &L, const MachineInstr &SetupMI,
                          const TargetRegisterInfo &TRI,
                          const MachineInstr *OwnBranch) {
  for (MachineBasicBlock *MBB : L.getBlocks()) {
    for (const MachineInstr &MI : *MBB) {
      bool IsHazard = isCycleHazard(MI, SetupMI, TRI, OwnBranch);
      LLVM_DEBUG(dbgs() << "  Checking loop MI (isHazard=" << IsHazard << "): " << MI);
      if (IsHazard) {
        return true;
      }
    }
    LLVM_DEBUG(dbgs() << "  No loop hazard in BB#" << MBB->getNumber() << "\n");
  }

  return false;
}

SmallVector<HoistPoint, 8>
RISCVBranchSetupHoisting::findHoistPoints(MachineInstr &SetupMI,
                                          const MachineInstr *OwnBranch) {
  SmallVector<HoistPoint, 8> Points;
  MachineBasicBlock *CurBB = SetupMI.getParent();
  MachineBasicBlock::iterator SafePoint;
  // How often the setup runs where it is, so far: its own block, then the
  // coolest point it has been hoisted to. Out of a loop body, that is the
  // preheader, not the body, whose frequency includes the trip count.
  BlockFrequency Ref = MBFI->getBlockFreq(CurBB);

  bool HazardFound = false;

  auto IntraStart = std::next(SetupMI.getReverseIterator());
  if (IntraStart != CurBB->rend())
    SafePoint =
        scanBlockBackward(CurBB, IntraStart, SetupMI, *TRI, OwnBranch,
                          HazardFound);
  else
    SafePoint = CurBB->getFirstNonPHI();

  if (SafePoint != SetupMI.getIterator())
    Points.push_back({CurBB, SafePoint});
  if (HazardFound)
    return Points;

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

    // The latch of a loop that ends before the setup can dominate it, but a
    // setup left in that loop would run on every iteration instead of once,
    // and would overwrite its B-register while the loop's own branches run.
    // It can still be hoisted through the loop to a block before it.
    MachineLoop *IDomLoop = MLI->getLoopFor(IDom);
    bool InOtherLoop = IDomLoop && !IDomLoop->contains(CurBB);
    // Nor is a setup left in a block that runs much more often than its own,
    // such as the test before a rarely taken path, as the setups of that path
    // would then run whether it is taken or not.
    bool TooHot = false;
    if (MaxHoistFreqRatio) {
      BlockFrequency Base =
          DisableHoistFreqFromBest ? MBFI->getBlockFreq(CurBB) : Ref;
      if (std::optional<BlockFrequency> Limit = Base.mul(MaxHoistFreqRatio))
        TooHot = MBFI->getBlockFreq(IDom) > *Limit;
      if (TooHot && !DisableHoistFreqFromBest)
        if (std::optional<BlockFrequency> Own =
                MBFI->getBlockFreq(CurBB).mul(MaxHoistFreqRatio))
          if (MBFI->getBlockFreq(IDom) <= *Own)
            ++NumHoistPointsTooHotFromBest;
    }

    MachineBasicBlock::reverse_iterator IDomScanStart = IDom->rbegin();
    while (IDomScanStart != IDom->rend() && IDomScanStart->isTerminator())
      ++IDomScanStart;

    MachineBasicBlock::iterator IDomSafePoint;
    if (IDomScanStart == IDom->rend()) {
      IDomSafePoint = IDom->getFirstTerminator();
      HazardFound = false;
    } else {
      IDomSafePoint =
          scanBlockBackward(IDom, IDomScanStart, SetupMI, *TRI, OwnBranch,
                            HazardFound);
    }

    // The dominator tree chooses candidates; the inverse-CFG walk proves that
    // every actual path from the candidate to SetupMI is free of hazards.
    if (!isSafeToHoistTo(SetupMI, *IDom, IDomSafePoint, OwnBranch)) {
      LLVM_DEBUG(dbgs() << "  [Stop] Hazard on a CFG path from BB#"
                        << CurBB->getNumber() << " to BB#" << IDom->getNumber()
                        << "\n");
      break;
    }

    for (MachineLoop *L = MLI->getLoopFor(BB); L && !L->contains(IDom);
         L = L->getParentLoop()) {
      LLVM_DEBUG(dbgs() << "  Checking loop hazard...\n");
      if (loopHasHazard(*L, SetupMI, *TRI, OwnBranch))
        return Points;
    }

    if (InOtherLoop || TooHot) {
      LLVM_DEBUG(dbgs() << "  [Cross-BB] Passing BB#" << IDom->getNumber()
                        << (InOtherLoop ? ", in a loop the setup is not in"
                                        : ", which runs too often")
                        << "\n");
    } else {
      LLVM_DEBUG(dbgs() << "  [Cross-BB] Can hoist into BB#"
                        << IDom->getNumber() << "\n");
#ifndef NDEBUG
      // Recomputed from the points themselves, not from Ref: a point accepted
      // runs at most MaxHoistFreqRatio times as often as the setup's own
      // block and as every point accepted before it.
      if (MaxHoistFreqRatio && !DisableHoistFreqFromBest) {
        BlockFrequency Coolest = MBFI->getBlockFreq(CurBB);
        for (const HoistPoint &P : Points)
          Coolest = std::min(Coolest, MBFI->getBlockFreq(P.first));
        if (std::optional<BlockFrequency> Limit =
                Coolest.mul(MaxHoistFreqRatio))
          assert(MBFI->getBlockFreq(IDom) <= *Limit &&
                 "Hoist point runs too often for an earlier point");
      }
#endif
      Points.push_back({IDom, IDomSafePoint});
      if (MBFI->getBlockFreq(IDom) < Ref)
        Ref = MBFI->getBlockFreq(IDom);
    }
    BB = IDom;

    if (HazardFound)
      break;
  }

  return Points;
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
