//==- RISCVLoopFissionDecomposedBranch.cpp - Fission loops for BMOV slack -==//
//

/*
// test_fission.c
#include "test.h"

#define N 200

void test_fission(int *restrict a, int *restrict out, int t) {
    for (int i = 0; i < N; i++)
        if (a[i] < t)
            out[i] = a[i] * 3;
}

void _start(void) {
    static int data[N];
    static int out[N];

    for (int i = 0; i < N; i++)
        data[i] = i % 37;

    test_fission(data, out, 20);

    EMIT64(out[0]);   // emit a computed scalar value

    _exit(0);
}
*/

#include "RISCV.h"
#include "RISCVBranchSetupAnalysis.h"
#include "RISCVInstrInfo.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineOptimizationRemarkEmitter.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "riscv-loop-fission-decomposed-branch"
#define PASS_NAME "RISC-V loop fission for decomposed branches pass"

using namespace llvm;

// Follows the same convention as RISCVBranchSetupHoisting: a Hidden,
// disable-style flag, defaulted to enabled (false = not disabled).
static cl::opt<bool> DisableLoopFissionDecomposedBranch(
    "disable-loop-fission-decomposed-branch", cl::Hidden,
    cl::desc("Disable " PASS_NAME), cl::init(false));

// Optional queue-depth override; the paper assumes a 64-entry branch
// outcome queue, so tiling must not exceed this bound.
static cl::opt<unsigned>
    LoopFissionQueueDepth("loop-fission-queue-depth", cl::Hidden,
                          cl::desc("Branch outcome queue depth used to "
                                   "bound fission tile size"),
                          cl::init(64));

STATISTIC(NumLoopsFissioned,
          "Number of loops split into compute/consume pairs");

namespace {

class RISCVLoopFissionDecomposedBranch : public MachineFunctionPass {
public:
  static char ID;
  const RISCVBranchSetupInfo *BSI;
  const TargetRegisterInfo *TRI;
  MachineDominatorTree *MDT;
  MachineLoopInfo *MLI;
  AAResults *AA = nullptr;
  // Why buildSlice said no, for the missed remark
  std::string Why;

  RISCVLoopFissionDecomposedBranch() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<RISCVBranchSetupAnalysisWrapper>();
    AU.addRequired<MachineDominatorTreeWrapperPass>();
    AU.addRequired<MachineLoopInfoWrapperPass>();
    AU.addRequired<AAResultsWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override { return PASS_NAME; }

private:
  /// Returns true if \p L is a countable loop containing exactly one
  /// internal conditional branch whose condition does not depend on
  /// values produced by a prior iteration's body.
  bool isFissionCandidate(MachineLoop &L) const;

  /// Returns true if the branch condition computed in \p L has any
  /// register dependency reaching back through the loop body from a
  /// prior iteration (i.e. control-dependent control flow per the
  /// paper's Section 8.4 -- unsafe to fission).
  bool hasCrossIterationDependence(MachineLoop &L,
                                   MachineInstr &CondBranch) const;

  bool buildSlice(MachineLoop &L, MachineInstr &Br,
                  SmallVectorImpl<MachineInstr *> &Slice);

  /// Splits \p L into a compute loop (condition only, BMOVC enqueue)
  /// and a consume loop (original body, PB dequeue), tiled to
  /// LoopFissionQueueDepth if the trip count exceeds it.
  bool fissionLoop(MachineLoop &L);
};

char RISCVLoopFissionDecomposedBranch::ID = 0;

/// "a call to f at line 12", "LW at line 12": what an instruction is, for
/// remarks.
static std::string describe(const MachineInstr &MI) {
  std::string S;
  raw_string_ostream OS(S);
  const MachineOperand *Callee = nullptr;
  if (MI.isCall())
    for (const MachineOperand &MO : MI.operands())
      if (MO.isGlobal() || MO.isSymbol()) {
        Callee = &MO;
        break;
      }
  if (Callee && Callee->isGlobal())
    OS << "a call to " << Callee->getGlobal()->getName();
  else if (Callee)
    OS << "a call to " << Callee->getSymbolName();
  else if (MI.isCall())
    OS << "an indirect call";
  else
    OS << MI.getMF()->getSubtarget().getInstrInfo()->getName(MI.getOpcode());
  if (const DebugLoc &DL = MI.getDebugLoc())
    OS << " at line " << DL.getLine();
  return S;
}

static bool isIRFissioned(const MachineLoop &L) {
  const MachineBasicBlock *Latch = L.getLoopLatch();
  const BasicBlock *IRLatch = Latch ? Latch->getBasicBlock() : nullptr;
  const Instruction *Term = IRLatch ? IRLatch->getTerminator() : nullptr;
  const MDNode *LoopID = Term ? Term->getMetadata(LLVMContext::MD_loop)
                              : nullptr;
  if (!LoopID)
    return false;
  for (unsigned I = 1; I < LoopID->getNumOperands(); ++I) {
    auto *Option = dyn_cast<MDNode>(LoopID->getOperand(I));
    if (!Option || Option->getNumOperands() != 1)
      continue;
    auto *Name = dyn_cast<MDString>(Option->getOperand(0));
    if (Name && Name->getString() == "riscv.loop-fissioned")
      return true;
  }
  return false;
}

bool RISCVLoopFissionDecomposedBranch::isFissionCandidate(
    MachineLoop &L) const {

  if (!L.getLoopLatch() || !L.getLoopPreheader()) {
    llvm::dbgs() << "  rejected: no latch/preheader\n";
    return false;
  }

  unsigned bc = 0;
  MachineInstr *CondBranch = nullptr;
  for (MachineBasicBlock *MBB : L.getBlocks())
    for (MachineInstr &MI : *MBB) {
      if (MI.getOpcode() != RISCV::PseudoPBC)
        continue;
      RISCVBranchSetup BS = BSI->Branches.lookup(&MI);
      if (!BS.S || !BS.T || !BS.C)
        continue;
      if (BS.C->getOpcode() == RISCV::BMOVC_LOOP)
        continue;
      CondBranch = &MI;
      ++bc;
    }

  llvm::dbgs() << "  branch count (excluding loop-counter)=" << bc << "\n";

  if (bc == 0) {
    llvm::dbgs() << "  rejected: no candidate branch\n";
    return false;
  }
  if (bc > 1) {
    llvm::dbgs() << "  rejected: multiple candidate branches\n";
    return false;
  }

  bool Cross = hasCrossIterationDependence(L, *CondBranch);
  llvm::dbgs() << "  crossIterationDependence=" << Cross << "\n";
  return !Cross;
}

bool RISCVLoopFissionDecomposedBranch::hasCrossIterationDependence(
    MachineLoop &L, MachineInstr &CondBranch) const {

  MachineBasicBlock *Header = L.getHeader();
  MachineBasicBlock *Latch = L.getLoopLatch();
  MachineBasicBlock *Preheader = L.getLoopPreheader();

  if (!Header || !Latch || !Preheader)
    return true;

  const MachineRegisterInfo &MRI = CondBranch.getMF()->getRegInfo();
  SmallVector<Register, 16> WorkList;
  SmallPtrSet<const MachineInstr *, 32> Visited;

  for (const MachineOperand &MO : CondBranch.all_uses()) {
    if (MO.isReg() && MO.getReg().isVirtual())
      WorkList.push_back(MO.getReg());
  }

  while (!WorkList.empty()) {
    Register Reg = WorkList.pop_back_val();

    for (MachineInstr &Def : MRI.def_instructions(Reg)) {
      if (!Visited.insert(&Def).second)
        continue;

      MachineBasicBlock *DefMBB = Def.getParent();
      if (!L.contains(DefMBB))
        continue;

      if (Def.isPHI() && DefMBB == Header) {
        for (unsigned i = 1; i < Def.getNumOperands(); i += 2) {
          if (!Def.getOperand(i + 1).isMBB())
            continue;

          Register IncomingReg = Def.getOperand(i).getReg();
          MachineBasicBlock *IncomingMBB = Def.getOperand(i + 1).getMBB();

          if (IncomingMBB == Preheader) {
            continue;
          }

          if (IncomingReg.isVirtual()) {
            MachineInstr *IncomingDef = MRI.getVRegDef(IncomingReg);
            if (IncomingDef && L.contains(IncomingDef->getParent()))
              return true;
          }

          continue;
        }
      }

      for (const MachineOperand &MO : Def.all_uses()) {
        if (MO.isReg() && MO.getReg().isVirtual())
          WorkList.push_back(MO.getReg());
      }
    }
  }
  return false;
}

bool RISCVLoopFissionDecomposedBranch::buildSlice(
    MachineLoop &L, MachineInstr &Br, SmallVectorImpl<MachineInstr *> &Slice) {
  MachineRegisterInfo &MRI = Br.getMF()->getRegInfo();
  MachineBasicBlock *Header = L.getHeader(), *Latch = L.getLoopLatch();
  SmallPtrSet<MachineInstr *, 16> Seen;
  SmallVector<Register, 8> Work;
  BatchAAResults BAA(*AA);
  auto enqueue = [&](Register Reg) {
    if (Reg.isPhysical())
      return Reg == RISCV::X0;
    if (Reg.isVirtual())
      Work.push_back(Reg);
    return true;
  };
  auto push = [&](MachineInstr &MI) {
    for (const MachineOperand &MO : MI.uses())
      if (MO.isReg() && !enqueue(MO.getReg()))
        return false;
    return true;
  };
  // The compute loop cannot reuse a physical value produced during the
  // consume loop (for example, a call result still in a0).
  Why = "unknown";
  auto fail = [&](const Twine &W) {
    Why = W.str();
    return false;
  };
  if (!push(Br))
    return fail("the branch reads a physical register other than x0");
  while (!Work.empty()) {
    MachineInstr *Def = MRI.getVRegDef(Work.pop_back_val());

    if (!Def)
      continue; // live-in
    if (Def->getNumDefs() != 1)
      return fail("the condition depends on " + describe(*Def) +
                  ", which defines more than one value");

    if (!L.contains(Def->getParent()))
      continue; // loop invariant
    if (!Seen.insert(Def).second)
      continue;
    // Must run every iteration.
    if (!MDT->dominates(Def->getParent(), Latch))
      return fail("the condition depends on " + describe(*Def) +
                  ", which does not run in every iteration");
    if (Def->isPHI()) {
      if (Def->getParent() != Header)
        return fail("the condition depends on a value merged inside the "
                    "loop body");
      // follow only the latch operand
      for (unsigned i = 1; i < Def->getNumOperands(); i += 2)
        if (Def->getOperand(i + 1).getMBB() == Latch &&
            !enqueue(Def->getOperand(i).getReg()))
          return fail("the condition depends on a physical register "
                      "carried around the loop");
    } else {
      if (Def->mayStore() || Def->isCall() || Def->hasUnmodeledSideEffects())
        return fail("the condition depends on " + describe(*Def) +
                    ", which stores, calls or has side effects");
      if (Def->mayLoad())
        for (MachineBasicBlock *B : L.blocks())
          for (MachineInstr &S : *B)
            if (S.mayStore() && S.mayAlias(AA, *Def, /*UseTBAA=*/true))
              return fail("the condition loads (" + describe(*Def) +
                          ") from memory that a store in the loop (" +
                          describe(S) + ") may write");
      if (!push(*Def))
        return fail("the condition depends on " + describe(*Def) +
                    ", which reads a physical register other than x0");
    }
    Slice.push_back(Def);
  }
  if (!MDT->dominates(Br.getParent(), Latch))
    return fail("the branch does not run in every iteration");
  return true;
}
static bool isDataBranch(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  case RISCV::PseudoBREQ:
  case RISCV::PseudoBRNE:
  case RISCV::PseudoBRLT:
  case RISCV::PseudoBRGE:
  case RISCV::PseudoBRLTU:
  case RISCV::PseudoBRGEU:
    return true;
  }
  return false;
}

static std::pair<unsigned, const char *> bmovcFor(unsigned Opc) {
  switch (Opc) {
  case RISCV::PseudoBREQ:
    return {RISCV::BMOVC_BEQ, "ns_beq_"};
  case RISCV::PseudoBRNE:
    return {RISCV::BMOVC_BNE, "ns_bne_"};
  case RISCV::PseudoBRLT:
    return {RISCV::BMOVC_BLT, "ns_blt_"};
  case RISCV::PseudoBRGE:
    return {RISCV::BMOVC_BGE, "ns_bge_"};
  case RISCV::PseudoBRLTU:
    return {RISCV::BMOVC_BLTU, "ns_bltu_"};
  case RISCV::PseudoBRGEU:
    return {RISCV::BMOVC_BGEU, "ns_bgeu_"};
  }
  llvm_unreachable("not a data branch");
}

bool RISCVLoopFissionDecomposedBranch::fissionLoop(MachineLoop &L) {
  MachineBasicBlock *Pre = L.getLoopPreheader(), *Header = L.getHeader(),
                    *Latch = L.getLoopLatch();
  MachineFunction &MF = *Header->getParent();
  MachineOptimizationRemarkEmitter ORE(MF, nullptr);
  DebugLoc LoopLoc = L.getStartLoc();
  auto Reject = [&](const Twine &Why) {
    LLVM_DEBUG(dbgs() << "fission: reject " << MF.getName() << ": " << Why
                      << "\n");
    ORE.emit([&]() {
      return MachineOptimizationRemarkMissed(DEBUG_TYPE, "NotFissioned",
                                             LoopLoc, Header)
             << ("not splitting the loop (decomposed-branch fission): " + Why)
                    .str();
    });
    return false;
  };
  if (!Pre || !Latch)
    return Reject("no preheader or latch");

  MachineBasicBlock *Exit = L.getExitBlock();
  if (!L.getExitingBlock() || !Exit)
    return Reject("loop does not have a single exit");
  if (L.getExitingBlock() != Latch)
    return Reject("loop exit is not taken from the latch");

  MachineRegisterInfo &MRI = MF.getRegInfo();
  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();

  // Exactly one data branch in the loop.
  MachineInstr *Br = nullptr;
  unsigned N = 0;
  for (MachineBasicBlock *B : L.blocks())
    for (MachineInstr &MI : B->terminators())
      if (isDataBranch(MI)) {
        Br = &MI;
        ++N;
      }
  if (N != 1)
    return Reject("the loop has " + Twine(N) +
                  " conditional branches besides the latch (one is handled)");

  // Constant trip count from PseudoLoopSetup %x, where %x = ADDI $x0, imm.
  MachineInstr *Setup = nullptr;
  int64_t Trip = -1;
  for (MachineInstr &MI : *Pre)
    if (MI.getOpcode() == RISCV::PseudoLoopSetup) {
      Setup = &MI;
      MachineInstr *D = MRI.getVRegDef(MI.getOperand(0).getReg());
      if (D && D->getOpcode() == RISCV::ADDI &&
          D->getOperand(1).getReg() == RISCV::X0 && D->getOperand(2).isImm())
        Trip = D->getOperand(2).getImm();
    }
  if (!Setup)
    return Reject("the loop is not a counted hardware loop (no loop setup in "
                  "its preheader)");
  if (Trip < 1)
    return Reject("the trip count is not a compile-time constant");

  // The hardware queue holds 64 outcomes. A tile must not exceed it.
  int64_t Depth = std::min<unsigned>(LoopFissionQueueDepth, 64);
  if (Depth < 1)
    return Reject("queue depth is below 1");
  const bool Tiled = Trip > Depth;

  // The latch operands that point at the exit block. Needed when tiling.
  SmallVector<MachineOperand *, 2> ExitOps;
  for (MachineInstr &T : Latch->terminators())
    for (MachineOperand &MO : T.operands())
      if (MO.isMBB() && MO.getMBB() == Exit)
        ExitOps.push_back(&MO);
  if (Tiled && ExitOps.empty())
    return Reject("latch has no explicit branch to the exit block");

  SmallVector<MachineInstr *, 16> Slice;
  if (!buildSlice(L, *Br, Slice))
    return Reject("its condition cannot be computed in a loop of its own: " +
                  Why);
  SmallPtrSet<MachineInstr *, 16> InSlice(Slice.begin(), Slice.end());

  DebugLoc DL = Br->getDebugLoc();
  auto [BmovcOpc, Prefix] = bmovcFor(Br->getOpcode());
  MachineBasicBlock *TBB = Br->getOperand(2).getMBB();

  // ---- Blocks. ----
  // untiled: Pre -> Compute -> CLatch -> Mid -> Header ... Latch -> Exit
  // tiled:   Pre -> TileHeader -> Compute -> CLatch -> Mid -> Header ... Latch
  //                  ^                                                   |
  //                  +------------------- TileLatch <--------------------+
  MachineBasicBlock *TileHeader = nullptr, *TileLatch = nullptr;
  if (Tiled) {
    TileHeader = MF.CreateMachineBasicBlock();
    TileLatch = MF.CreateMachineBasicBlock();
    MF.insert(Header->getIterator(), TileHeader);
  }
  MachineBasicBlock *Compute = MF.CreateMachineBasicBlock();
  MachineBasicBlock *CLatch = MF.CreateMachineBasicBlock();
  MachineBasicBlock *Mid = MF.CreateMachineBasicBlock();
  MF.insert(Header->getIterator(), Compute);
  MF.insert(Header->getIterator(), CLatch);
  MF.insert(Header->getIterator(), Mid);
  if (Tiled)
    MF.insert(std::next(Latch->getIterator()), TileLatch);

  MachineBasicBlock *Entry = Tiled ? TileHeader : Pre;

  // ---- PBR values: one vreg per value, so the function stays in SSA. ----
  Register P0 = MRI.createVirtualRegister(&RISCV::PBRRegClass);   // BMOVS
  Register P1 = MRI.createVirtualRegister(&RISCV::PBRRegClass);   // BMOVT
  Register PPhi = MRI.createVirtualRegister(&RISCV::PBRRegClass); // compute PHI
  Register P2 = MRI.createVirtualRegister(&RISCV::PBRRegClass);   // after BMOVC

  // S and T once, in the preheader.
  MCSymbol *Sym = MF.getContext().createTempSymbol(Prefix);
  BuildMI(*Pre, Setup->getIterator(), DL, TII->get(RISCV::BMOVS_J), P0)
      .addSym(Sym);
  BuildMI(*Pre, Setup->getIterator(), DL, TII->get(RISCV::BMOVT_J), P1)
      .addReg(P0)
      .addMBB(TBB);
  Setup->getOperand(0).setIsKill(false);
  Register Cnt = Setup->getOperand(0).getReg();

  // ---- Header PHIs: entry value from Pre, back value from Latch. ----
  SmallVector<MachineInstr *, 4> HPhis;
  for (MachineInstr &PHI : Header->phis())
    HPhis.push_back(&PHI);
  auto incoming = [](MachineInstr &PHI, MachineBasicBlock *BB) -> Register {
    for (unsigned i = 1; i + 1 < PHI.getNumOperands(); i += 2)
      if (PHI.getOperand(i + 1).getMBB() == BB)
        return PHI.getOperand(i).getReg();
    return Register();
  };
  DenseMap<MachineInstr *, Register> EntryVal; // value each loop starts with
  for (MachineInstr *PHI : HPhis)
    EntryVal[PHI] = incoming(*PHI, Pre);

  // ---- Tile header: tile size, remaining count, PBR and value PHIs. ----
  Register CountV = Cnt;    // iterations of the inner loops
  Register EntryPBR = P1;   // PBR value the compute loop starts from
  Register RemV, Rem2;
  if (Tiled) {
    const TargetRegisterClass *GPRRC = &RISCV::GPRRegClass;
    int64_t FirstN = Trip % Depth;  // first tile takes the remainder
    if (FirstN == 0)
      FirstN = Depth;

    Register FirstV = MRI.createVirtualRegister(GPRRC);
    Register TileV = MRI.createVirtualRegister(GPRRC);
    BuildMI(*Pre, Setup->getIterator(), DL, TII->get(RISCV::ADDI), FirstV)
        .addReg(RISCV::X0)
        .addImm(FirstN);
    BuildMI(*Pre, Setup->getIterator(), DL, TII->get(RISCV::ADDI), TileV)
        .addReg(RISCV::X0)
        .addImm(Depth);
    // The tiles replace the original setup. Cnt keeps its ADDI def.
    Setup->eraseFromParent();

    CountV = MRI.createVirtualRegister(GPRRC);
    RemV = MRI.createVirtualRegister(GPRRC);
    Rem2 = MRI.createVirtualRegister(GPRRC);
    EntryPBR = MRI.createVirtualRegister(&RISCV::PBRRegClass);

    BuildMI(*TileHeader, TileHeader->end(), DL, TII->get(TargetOpcode::PHI),
            CountV)
        .addReg(FirstV)
        .addMBB(Pre)
        .addReg(TileV)
        .addMBB(TileLatch);
    BuildMI(*TileHeader, TileHeader->end(), DL, TII->get(TargetOpcode::PHI),
            RemV)
        .addReg(Cnt)
        .addMBB(Pre)
        .addReg(Rem2)
        .addMBB(TileLatch);
    BuildMI(*TileHeader, TileHeader->end(), DL, TII->get(TargetOpcode::PHI),
            EntryPBR)
        .addReg(P1)
        .addMBB(Pre)
        .addReg(P2)
        .addMBB(TileLatch);

    // One PHI per loop-carried value. Each tile starts where the last ended.
    for (MachineInstr *PHI : HPhis) {
      Register TPhi =
          MRI.createVirtualRegister(MRI.getRegClass(PHI->getOperand(0).getReg()));
      BuildMI(*TileHeader, TileHeader->end(), DL, TII->get(TargetOpcode::PHI),
              TPhi)
          .addReg(EntryVal[PHI])
          .addMBB(Pre)
          .addReg(incoming(*PHI, Latch))
          .addMBB(TileLatch);
      EntryVal[PHI] = TPhi;
    }

    BuildMI(*TileHeader, TileHeader->end(), DL,
            TII->get(RISCV::PseudoLoopSetup))
        .addReg(CountV);
    BuildMI(*TileHeader, TileHeader->end(), DL, TII->get(RISCV::PseudoBR))
        .addMBB(Compute);
  }

  // ---- CFG edges. ----
  MachineBasicBlock *AfterPre = Tiled ? TileHeader : Compute;
  for (MachineInstr &T : Pre->terminators())
    if (T.getOpcode() == RISCV::PseudoBR && T.getOperand(0).getMBB() == Header)
      T.getOperand(0).setMBB(AfterPre);
  Pre->replaceSuccessor(Header, AfterPre);
  if (Tiled)
    TileHeader->addSuccessor(Compute);
  Compute->addSuccessor(CLatch);
  CLatch->addSuccessor(Compute);
  CLatch->addSuccessor(Mid);
  Mid->addSuccessor(Header);

  if (Tiled) {
    Latch->replaceSuccessor(Exit, TileLatch);
    for (MachineOperand *MO : ExitOps)
      MO->setMBB(TileLatch);
    TileLatch->addSuccessor(TileHeader);
    TileLatch->addSuccessor(Exit);
    for (MachineInstr &PHI : Exit->phis())
      for (unsigned i = 1; i + 1 < PHI.getNumOperands(); i += 2)
        if (PHI.getOperand(i + 1).getMBB() == Latch)
          PHI.getOperand(i + 1).setMBB(TileLatch);
  }

  // Mid: second loop setup for the consume loop.
  BuildMI(*Mid, Mid->end(), DL, TII->get(RISCV::PseudoLoopSetup))
      .addReg(CountV);
  BuildMI(*Mid, Mid->end(), DL, TII->get(RISCV::PseudoBR)).addMBB(Header);

  // ---- Compute block. ----
  DenseMap<Register, Register> VMap;
  for (MachineInstr *MI : Slice) {
    Register R = MI->getOperand(0).getReg();
    VMap[R] = MRI.cloneVirtualRegister(R);
  }
  auto Remap = [&](Register R) {
    auto It = VMap.find(R);
    return It == VMap.end() ? R : It->second;
  };

  // PHIs first: the PBR PHI, then the cloned header PHIs.
  BuildMI(*Compute, Compute->end(), DL, TII->get(TargetOpcode::PHI), PPhi)
      .addReg(EntryPBR)
      .addMBB(Entry)
      .addReg(P2)
      .addMBB(CLatch);
  for (MachineInstr *PHI : HPhis) {
    if (!InSlice.count(PHI))
      continue;
    BuildMI(*Compute, Compute->end(), PHI->getDebugLoc(),
            TII->get(TargetOpcode::PHI), VMap[PHI->getOperand(0).getReg()])
        .addReg(EntryVal[PHI])
        .addMBB(Entry)
        .addReg(Remap(incoming(*PHI, Latch)))
        .addMBB(CLatch);
  }

  // Header PHIs now start from Mid. Done after the compute PHIs read Pre.
  for (MachineInstr *PHI : HPhis)
    for (unsigned i = 1; i + 1 < PHI->getNumOperands(); i += 2)
      if (PHI->getOperand(i + 1).getMBB() == Pre) {
        PHI->getOperand(i).setReg(EntryVal[PHI]);
        PHI->getOperand(i + 1).setMBB(Mid);
      }

  // Non-PHI slice instructions, in loop block order.
  for (MachineBasicBlock *B : L.blocks())
    for (MachineInstr &MI : *B) {
      if (MI.isPHI() || !InSlice.count(&MI))
        continue;
      MachineInstr *C = MF.CloneMachineInstr(&MI);
      for (MachineOperand &MO : C->operands())
        if (MO.isReg() && MO.getReg().isVirtual()) {
          MO.setReg(Remap(MO.getReg()));
          if (MO.isUse())
            MO.setIsKill(false);
        }
      Compute->push_back(C);
    }

  // BMOVC enqueues one outcome per iteration.
  BuildMI(*Compute, Compute->end(), DL, TII->get(BmovcOpc), P2)
      .addReg(PPhi)
      .addReg(Remap(Br->getOperand(0).getReg()))
      .addReg(Remap(Br->getOperand(1).getReg()));
  BuildMI(*Compute, Compute->end(), DL, TII->get(RISCV::PseudoBR))
      .addMBB(CLatch);

  // CLatch holds the loop end, so back-edge copies land after the BMOVC.
  BuildMI(*CLatch, CLatch->end(), DL, TII->get(RISCV::PseudoLoopEnd))
      .addMBB(Compute);
  BuildMI(*CLatch, CLatch->end(), DL, TII->get(RISCV::PseudoBR)).addMBB(Mid);

  // ---- Tile latch: rem2 = rem - n ; loop while rem2 != 0. ----
  if (Tiled) {
    BuildMI(*TileLatch, TileLatch->end(), DL, TII->get(RISCV::SUB), Rem2)
        .addReg(RemV)
        .addReg(CountV);
    BuildMI(*TileLatch, TileLatch->end(), DL, TII->get(RISCV::PseudoBRNE))
        .addReg(Rem2)
        .addReg(RISCV::X0)
        .addMBB(TileHeader);
    BuildMI(*TileLatch, TileLatch->end(), DL, TII->get(RISCV::PseudoBR))
        .addMBB(Exit);
  }

  // ---- Consume loop: the branch becomes a PB that reads the outcome. ----
  BuildMI(*Br->getParent(), Br, DL, TII->get(RISCV::PseudoPBC))
      .addReg(P2)
      .addMBB(TBB);
  Br->eraseFromParent();

  MDT->recalculate(MF);
  MLI->releaseMemory();
  MLI->analyze(*MDT);
  ++NumLoopsFissioned;
  ORE.emit([&]() {
    return MachineOptimizationRemark(DEBUG_TYPE, "Fissioned", LoopLoc, Header)
           << "split the loop: its conditions are computed first and queued"
           << (Tiled ? " (in tiles)" : "");
  });
  return true;
}

bool RISCVLoopFissionDecomposedBranch::runOnMachineFunction(
    MachineFunction &MF) {
  if (DisableLoopFissionDecomposedBranch)
    return false;
  if (MF.getName() == "_start" || MF.getName() == "_exit")
    return false;

  MDT = &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  MLI = &getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  AA = &getAnalysis<AAResultsWrapperPass>().getAAResults();

  // Snapshot headers first. fissionLoop rebuilds MLI, which frees MachineLoops.
  SmallVector<MachineBasicBlock *, 4> Headers;
  for (MachineLoop *L : MLI->getLoopsInPreorder())
    if (L->isInnermost() && !isIRFissioned(*L))
      Headers.push_back(L->getHeader());

  bool Changed = false;
  for (MachineBasicBlock *H : Headers) {
    MachineLoop *L = MLI->getLoopFor(H);
    if (!L || L->getHeader() != H)
      continue;
    Changed |= fissionLoop(*L);
    LLVM_DEBUG(dbgs() << "fission: reject " << MF.getName()
                      << ": trip count\n");
  }
  return Changed;
}

} // end of anonymous namespace

INITIALIZE_PASS(RISCVLoopFissionDecomposedBranch, DEBUG_TYPE, PASS_NAME,
                false, // is CFG only?
                false  // is analysis?
)

namespace llvm {

FunctionPass *createRISCVLoopFissionDecomposedBranchPass() {
  return new RISCVLoopFissionDecomposedBranch();
}

} // namespace llvm