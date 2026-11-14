//===-- RISCVNSVersionPromote.cpp - Version loops to keep memory in registers -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Non-Spec: a pb waits at dispatch until its condition is computed. When the
// condition of a branch in a loop reads a loop-invariant memory location that
// LICM cannot keep in a register, every iteration reloads it, and the pb waits
// on that load. LICM gives up when
//  - a store in the loop may write the location (alias analysis cannot rule
//    it out: the address of a store depends on loaded data, or the pointers
//    come from places it cannot see), or
//  - the loop stores to the location only conditionally, and LICM cannot prove
//    that a store after the loop is allowed (the object is not known to be
//    writable and private to this thread).
//
// This pass versions such a loop on a runtime check that the location does not
// overlap any memory the loop accesses through other pointers, and in the
// checked copy keeps the location in a register:
//
//   if (no overlap) {                        // once, before the loop
//     v = *p;
//     for (...) { ... v ... v = x; ... }     // *p read and written as v
//     if (v != *p) *p = v;                   // write back only if changed
//   } else {
//     for (...) { ... *p ... *p = x; ... }   // the original loop
//   }
//
// The ranges the check compares come from scalar evolution: a loop-invariant
// address covers its own bytes, an affine address {S,+,step} covers its
// iterations up to the trip count, and any other address based on a
// loop-invariant pointer covers that pointer plus the range of its offset (for
// an index computed from loaded bytes, such as histogram[r][g][b] with
// r, g, b < 32, the offset's value range bounds it). Accesses alias analysis
// proves disjoint from the location need no check.
//
// It is correct for every program:
//  - The loop has no calls, no atomic or volatile accesses, no fences and no
//    instruction that may throw: it exits only through its exit block.
//  - Every promoted location is accessed in the loop by an access that runs
//    whenever the loop is entered, so loading it before the loop is safe.
//  - If no other access in the loop touches the location (the check), the loop
//    reads and writes it only through the register, so the copy computes the
//    same values (by induction over its iterations, the check covering every
//    address the copy's iterations use, and so the original's).
//  - The write back stores only if the value changed, which happens only if
//    the loop stored to the location: no store is added on a path that had
//    none (so read-only memory and other threads that only read it are safe).
//    Leaving out a store of the value already there is allowed: with no
//    synchronization in the loop, another thread cannot tell without a race.
//  - Delaying the stores to the end of the loop is allowed for the same
//    reason. Only integer locations are promoted, so "changed" is exact.
//  - The ranges are computed in the checked copy, where the hoisted loads are
//    invariant (its trip count may depend on them); the induction above makes
//    that valid. Address arithmetic that could wrap fails the check.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/LoopAccessAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/Analysis/MustExecute.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/LoopSimplify.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/LoopVersioning.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"
#include <optional>

using namespace llvm;

#define DEBUG_TYPE "riscv-ns-version-promote"

STATISTIC(NumVersioned, "Number of loops versioned on a memory overlap check");
STATISTIC(NumInPlace, "Number of loops promoted without a runtime check");
STATISTIC(NumPromoted, "Number of memory locations kept in registers");
STATISTIC(NumHoisted, "Number of loop-invariant loads hoisted");
STATISTIC(NumAbandoned, "Number of versioned loops whose check could not be "
                        "built (the original loop always runs)");

static cl::opt<bool> DisableVersionPromote(
    "riscv-ns-disable-version-promote", cl::Hidden, cl::init(false),
    cl::desc("Do not version loops on a runtime overlap check to keep the "
             "memory their branch conditions read in registers"));

static cl::opt<unsigned> MaxLoopSize(
    "riscv-ns-version-promote-max-loop-size", cl::Hidden, cl::init(256),
    cl::desc("Largest loop (in instructions) the versioning promotion copies"));

static cl::opt<uint64_t> MaxRange(
    "riscv-ns-version-promote-max-range", cl::Hidden, cl::init(16 << 20),
    cl::desc("Largest range (in bytes) a data-indexed access may be bounded "
             "to; a wider one would make the overlap check fail anyway"));

static cl::opt<unsigned> MaxChecks(
    "riscv-ns-version-promote-max-checks", cl::Hidden, cl::init(8),
    cl::desc("Most range-overlap tests in the runtime check of a versioned "
             "loop"));

namespace {

// A loop-invariant memory location the loop reads (and maybe writes).
struct Candidate {
  const SCEV *Addr = nullptr;
  Value *Ptr = nullptr; // a pointer operand with that address
  Type *Ty = nullptr;
  Align Alignment;
  SmallVector<Instruction *, 4> Accesses;
  bool Stored = false;
  bool Alive = true;
  bool FeedsBranch = false;
  // Other accesses of the loop the runtime check has to separate from it.
  SmallVector<Instruction *, 8> Checks;
  // Filled in by the transformation.
  Value *PtrInCheck = nullptr;
  LoadInst *Init = nullptr;
};

// Bytes [Lo, Hi) the accesses of a group may touch during the loop.
struct Range {
  const SCEV *Lo, *Hi;
  // Affine ranges are computed from the trip count: Step * BTC must not wrap.
  SmallVector<std::pair<const SCEV *, uint64_t>, 2> Guards; // (BTC, |Step|)
  // Affine, but the trip count is not known yet (only in the analysis, where
  // it may still depend on a load the checked copy hoists).
  bool PendingTrip = false;
};

class VersionPromote {
  Function &F;
  LoopInfo &LI;
  DominatorTree &DT;
  ScalarEvolution &SE;
  AAResults &AA;
  AssumptionCache &AC;
  LoopAccessInfoManager &LAIs;
  OptimizationRemarkEmitter &ORE;
  const DataLayout &DL;

public:
  VersionPromote(Function &F, LoopInfo &LI, DominatorTree &DT,
                 ScalarEvolution &SE, AAResults &AA, AssumptionCache &AC,
                 LoopAccessInfoManager &LAIs, OptimizationRemarkEmitter &ORE)
      : F(F), LI(LI), DT(DT), SE(SE), AA(AA), AC(AC), LAIs(LAIs), ORE(ORE),
        DL(F.getDataLayout()) {}

  bool run();

private:
  bool processLoop(Loop &L);
  std::optional<Range> rangeOf(Instruction *I, Loop &L, bool Final);
  bool feedsBranch(Loop &L, const DenseMap<Instruction *, unsigned> &Owner,
                   SmallVectorImpl<Candidate> &Cands);
  void missed(Loop &L, StringRef Tag, const Twine &Msg);
};

// Scoped noalias metadata speaks about one iteration only; TBAA holds across
// all of them.
static AAMDNodes tbaaOnly(const Instruction *I) {
  AAMDNodes Tags = I->getAAMetadata();
  Tags.Scope = nullptr;
  Tags.NoAlias = nullptr;
  return Tags;
}

void VersionPromote::missed(Loop &L, StringRef Tag, const Twine &Msg) {
  LLVM_DEBUG(dbgs() << "  not versioned: " << Msg << "\n");
  ORE.emit([&]() {
    return OptimizationRemarkMissed(DEBUG_TYPE, Tag, L.getStartLoc(),
                                    L.getHeader())
           << "not keeping loop memory in registers: " << Msg.str();
  });
}

std::optional<Range> VersionPromote::rangeOf(Instruction *I, Loop &L,
                                             bool Final) {
  Value *Ptr = getLoadStorePointerOperand(I);
  const SCEV *S = SE.getSCEV(Ptr);
  Type *IntPtrTy = DL.getIntPtrType(Ptr->getType());
  uint64_t Size = DL.getTypeStoreSize(getLoadStoreType(I)).getFixedValue();
  const SCEV *SizeS = SE.getConstant(IntPtrTy, Size);

  if (SE.isLoopInvariant(S, &L))
    return Range{S, SE.getAddExpr(S, SizeS), {}};

  if (auto *AR = dyn_cast<SCEVAddRecExpr>(S);
      AR && AR->getLoop() == &L && AR->isAffine()) {
    auto *StepC = dyn_cast<SCEVConstant>(AR->getStepRecurrence(SE));
    const SCEV *BTC = SE.getBackedgeTakenCount(&L);
    if (StepC && StepC->getAPInt().getSignificantBits() <= 32) {
      if (isa<SCEVCouldNotCompute>(BTC) ||
          BTC->getType()->getScalarSizeInBits() > 64) {
        // Before the transformation the trip count may still depend on a load
        // the checked copy hoists: the caller allows for that.
        if (Final)
          return std::nullopt;
        Range R{AR->getStart(), AR->getStart(), {}};
        R.PendingTrip = true;
        return R;
      }
      int64_t Step = StepC->getAPInt().getSExtValue();
      const SCEV *N = SE.getTruncateOrZeroExtend(BTC, IntPtrTy);
      const SCEV *Start = AR->getStart();
      const SCEV *End =
          SE.getAddExpr(Start, SE.getMulExpr(SE.getConstant(IntPtrTy, Step), N));
      Range R{Step >= 0 ? Start : End,
              SE.getAddExpr(Step >= 0 ? End : Start, SizeS),
              {}};
      R.Guards.push_back({N, (uint64_t)(Step < 0 ? -Step : Step)});
      return R;
    }
  }

  // An address based on a loop-invariant pointer, at an offset whose value
  // range scalar evolution can bound.
  const SCEV *Base = SE.getPointerBase(S);
  if (!isa<SCEVCouldNotCompute>(Base) && SE.isLoopInvariant(Base, &L)) {
    const SCEV *Off = SE.getMinusSCEV(S, Base);
    if (!isa<SCEVCouldNotCompute>(Off) && Off->getType() == IntPtrTy) {
      ConstantRange CR = SE.getSignedRange(Off);
      if (!CR.isFullSet() && !CR.isSignWrappedSet() && !CR.isEmptySet()) {
        APInt Min = CR.getSignedMin(), Max = CR.getSignedMax();
        APInt Span = Max - Min;
        if (Span.ult(APInt(Span.getBitWidth(), MaxRange)))
          return Range{SE.getAddExpr(Base, SE.getConstant(Min)),
                       SE.getAddExpr(Base, SE.getAddExpr(SE.getConstant(Max),
                                                         SizeS)),
                       {}};
      }
    }
  }
  return std::nullopt;
}

// Marks the candidates whose loads a branch condition (or a select's, or a
// min/max's, which the backend also lowers to a branch) of the loop reads.
bool VersionPromote::feedsBranch(Loop &L,
                                 const DenseMap<Instruction *, unsigned> &Owner,
                                 SmallVectorImpl<Candidate> &Cands) {
  SmallVector<Value *, 16> Roots;
  for (BasicBlock *BB : L.blocks())
    for (Instruction &I : *BB) {
      if (auto *Br = dyn_cast<BranchInst>(&I); Br && Br->isConditional())
        Roots.push_back(Br->getCondition());
      else if (auto *Sel = dyn_cast<SelectInst>(&I))
        Roots.push_back(Sel->getCondition());
      else if (auto *MM = dyn_cast<MinMaxIntrinsic>(&I)) {
        Roots.push_back(MM->getLHS());
        Roots.push_back(MM->getRHS());
      }
    }
  bool Any = false;
  for (Value *Root : Roots) {
    SmallVector<Value *, 16> Work{Root};
    SmallPtrSet<Value *, 16> Seen;
    while (!Work.empty() && Seen.size() < 32) {
      auto *I = dyn_cast<Instruction>(Work.pop_back_val());
      if (!I || !L.contains(I) || !Seen.insert(I).second)
        continue;
      if (isa<LoadInst>(I)) {
        auto It = Owner.find(I);
        if (It != Owner.end() && Cands[It->second].Alive) {
          Cands[It->second].FeedsBranch = true;
          Any = true;
        }
        continue;
      }
      if (isa<PHINode>(I))
        continue;
      for (Value *Op : I->operands())
        Work.push_back(Op);
    }
  }
  return Any;
}

bool VersionPromote::processLoop(Loop &L) {
  LLVM_DEBUG(dbgs() << "riscv-ns-version-promote: " << L);
  if (!L.isInnermost() || !L.getLoopPreheader() || !L.getExitingBlock() ||
      !L.getExitBlock() || !L.isLoopSimplifyForm())
    return false;

  SmallVector<Instruction *, 32> MemInsts;
  unsigned Size = 0;
  for (BasicBlock *BB : L.blocks())
    for (Instruction &I : *BB) {
      if (++Size > MaxLoopSize)
        return false;
      if (auto *Ld = dyn_cast<LoadInst>(&I)) {
        if (!Ld->isSimple())
          return false;
        MemInsts.push_back(Ld);
      } else if (auto *St = dyn_cast<StoreInst>(&I)) {
        if (!St->isSimple())
          return false;
        MemInsts.push_back(St);
      } else if (I.mayReadOrWriteMemory() || I.mayThrow()) {
        return false; // calls, atomics, fences, ...
      }
    }

  // The candidates: loop-invariant addresses of integer loads and stores.
  SmallVector<Candidate, 8> Cands;
  DenseMap<const SCEV *, unsigned> ByAddr;
  DenseMap<Instruction *, unsigned> Owner;
  for (Instruction *I : MemInsts) {
    Value *Ptr = getLoadStorePointerOperand(I);
    if (Ptr->getType()->getPointerAddressSpace() != 0)
      continue;
    const SCEV *S = SE.getSCEV(Ptr);
    if (!SE.isLoopInvariant(S, &L))
      continue;
    auto [It, New] = ByAddr.try_emplace(S, Cands.size());
    if (New) {
      Cands.emplace_back();
      Cands.back().Addr = S;
      Cands.back().Ptr = Ptr;
      Cands.back().Ty = getLoadStoreType(I);
      Cands.back().Alignment = getLoadStoreAlignment(I);
    }
    Candidate &C = Cands[It->second];
    C.Accesses.push_back(I);
    Owner[I] = It->second;
    Type *Ty = getLoadStoreType(I);
    if (Ty != C.Ty || !Ty->isIntegerTy() || !DL.typeSizeEqualsStoreSize(Ty))
      C.Alive = false;
    C.Alignment = std::min(C.Alignment, getLoadStoreAlignment(I));
    C.Stored |= isa<StoreInst>(I);
  }
  if (Cands.empty())
    return false;

  // A load before the loop is safe only if the loop surely accesses the
  // location once entered; the address must be computable there.
  SimpleLoopSafetyInfo Safety;
  Safety.computeLoopSafetyInfo(&L);
  SCEVExpander Probe(SE, DL, "nsvp");
  Instruction *PHTerm = L.getLoopPreheader()->getTerminator();
  for (Candidate &C : Cands) {
    if (!C.Alive)
      continue;
    bool Guaranteed = any_of(C.Accesses, [&](Instruction *I) {
      return Safety.isGuaranteedToExecute(*I, &DT, &L);
    });
    if (!Guaranteed || !Probe.isSafeToExpandAt(C.Addr, PHTerm))
      C.Alive = false;
  }

  // Worth it only if a branch condition reads one of them.
  if (!feedsBranch(L, Owner, Cands))
    return false;

  // Whether the trip count is known now, or the exit condition reads a
  // location the copy will hoist (then it is known in the copy).
  bool TripKnown = !isa<SCEVCouldNotCompute>(SE.getBackedgeTakenCount(&L));
  auto ExitReadsHoisted = [&]() {
    auto *Br = dyn_cast<BranchInst>(L.getExitingBlock()->getTerminator());
    if (!Br || !Br->isConditional())
      return false;
    SmallVector<Value *, 8> Work{Br->getCondition()};
    SmallPtrSet<Value *, 8> Seen;
    while (!Work.empty() && Seen.size() < 16) {
      auto *I = dyn_cast<Instruction>(Work.pop_back_val());
      if (!I || !L.contains(I) || !Seen.insert(I).second)
        continue;
      if (isa<LoadInst>(I)) {
        auto It = Owner.find(I);
        if (It != Owner.end() && Cands[It->second].Alive &&
            !Cands[It->second].Stored)
          return true;
        continue;
      }
      if (isa<PHINode>(I))
        continue;
      for (Value *Op : I->operands())
        Work.push_back(Op);
    }
    return false;
  };

  // Separate each candidate from the other accesses of the loop: proven by
  // alias analysis, or left to the runtime check. Repeat until no candidate
  // drops out (the accesses of a dropped one are then other accesses too).
  bool Dropped = true;
  while (Dropped) {
    Dropped = false;
    for (Candidate &C : Cands) {
      if (!C.Alive)
        continue;
      C.Checks.clear();
      uint64_t CSize = DL.getTypeStoreSize(C.Ty).getFixedValue();
      MemoryLocation CLoc(C.Ptr, LocationSize::precise(CSize),
                          tbaaOnly(C.Accesses.front()));
      for (Instruction *I : MemInsts) {
        auto It = Owner.find(I);
        if (It != Owner.end() && &Cands[It->second] == &C)
          continue;
        bool IsStore = isa<StoreInst>(I);
        if (It != Owner.end() && Cands[It->second].Alive) {
          // Another candidate: both are kept in registers, so they must be
          // disjoint for sure.
          Candidate &D = Cands[It->second];
          if (!C.Stored && !D.Stored)
            continue;
          MemoryLocation DLoc(
              D.Ptr,
              LocationSize::precise(DL.getTypeStoreSize(D.Ty).getFixedValue()),
              tbaaOnly(I));
          if (AA.alias(CLoc, DLoc) != AliasResult::NoAlias) {
            C.Alive = false;
            break;
          }
          continue;
        }
        if (!C.Stored && !IsStore)
          continue; // two loads never conflict
        MemoryLocation ILoc = MemoryLocation::getBeforeOrAfter(
            getLoadStorePointerOperand(I), tbaaOnly(I));
        AliasResult R = AA.alias(CLoc, ILoc);
        if (R == AliasResult::NoAlias)
          continue;
        if (R != AliasResult::MayAlias) {
          C.Alive = false; // overlaps for sure: the check would always fail
          break;
        }
        std::optional<Range> RI = rangeOf(I, L, /*Final=*/false);
        if (!RI || (RI->PendingTrip && !TripKnown && !ExitReadsHoisted())) {
          C.Alive = false;
          break;
        }
        C.Checks.push_back(I);
      }
      if (!C.Alive)
        Dropped = true;
    }
  }

  SmallVector<Candidate *, 8> Live;
  bool Profitable = false, NeedCheck = false;
  for (Candidate &C : Cands)
    if (C.Alive) {
      Live.push_back(&C);
      Profitable |= C.FeedsBranch;
      NeedCheck |= !C.Checks.empty();
    }
  if (!Profitable) {
    if (any_of(Cands, [](const Candidate &C) { return C.FeedsBranch; }))
      missed(L, "NoCandidate",
             "a branch condition reads memory the loop may also write "
             "through a pointer whose range cannot be bounded");
    return false;
  }
  SmallPtrSet<Instruction *, 16> Checked;
  for (Candidate *C : Live)
    Checked.insert(C->Checks.begin(), C->Checks.end());
  if (Checked.size() > 4 * MaxChecks) {
    missed(L, "TooManyChecks", "the runtime check would be too large");
    return false;
  }
  // Without a runtime check, a location that is only read would already have
  // been hoisted by LICM: what is left to do is the stores.
  if (!NeedCheck && none_of(Live, [](Candidate *C) { return C->Stored; }))
    return false;

  // --- Transform -------------------------------------------------------------
  LLVM_DEBUG(dbgs() << "  versioning: " << Live.size() << " locations, "
                    << (NeedCheck ? "with" : "without") << " a check\n");
  const LoopAccessInfo *LAI = NeedCheck ? &LAIs.getInfo(L) : nullptr;

  BasicBlock *Preheader = L.getLoopPreheader();
  SCEVExpander Exp(SE, DL, "nsvp");
  Exp.disableCanonicalMode();
  // The initial values, loaded where the check is (the copy needs them, and
  // the check may need them for the copy's trip count).
  for (Candidate *C : Live) {
    C->PtrInCheck =
        Exp.expandCodeFor(C->Addr, C->Ptr->getType(), Preheader->getTerminator());
    C->Init = new LoadInst(C->Ty, C->PtrInCheck, "nsvp.init", false,
                           C->Alignment, Preheader->getTerminator()->getIterator());
  }

  BasicBlock *CheckBB = nullptr;
  if (NeedCheck) {
    LoopVersioning LVer(*LAI, {}, &L, &LI, &DT, &SE);
    LVer.versionLoop();
    CheckBB = Preheader; // the check block: it now ends in the version branch
  }
  BasicBlock *FastPH = L.getLoopPreheader();
  BasicBlock *FastExit = L.getExitBlock();
  assert(FastPH && FastExit && "versioned loop lost its preheader or exit");

  // In the copy: the hoisted loads use the initial value, the promoted
  // locations a slot that mem2reg turns into SSA values.
  DomTreeUpdater DTU(&DT, DomTreeUpdater::UpdateStrategy::Eager);
  SmallVector<AllocaInst *, 8> Slots;
  BasicBlock &Entry = F.getEntryBlock();
  Instruction *WBPoint = &*FastExit->getFirstInsertionPt();
  for (Candidate *C : Live) {
    if (!C->Stored) {
      for (Instruction *I : C->Accesses) {
        I->replaceAllUsesWith(C->Init);
        I->eraseFromParent();
      }
      ++NumHoisted;
      continue;
    }
    auto *Slot = new AllocaInst(C->Ty, DL.getAllocaAddrSpace(), nullptr,
                                C->Alignment, "nsvp.slot",
                                Entry.getFirstNonPHIOrDbgOrAlloca());
    new StoreInst(C->Init, Slot, FastPH->getTerminator()->getIterator());
    for (Instruction *I : C->Accesses) {
      if (auto *Ld = dyn_cast<LoadInst>(I)) {
        auto *New = new LoadInst(C->Ty, Slot, Ld->getName(), I->getIterator());
        Ld->replaceAllUsesWith(New);
      } else {
        new StoreInst(cast<StoreInst>(I)->getValueOperand(), Slot,
                      I->getIterator());
      }
      I->eraseFromParent();
    }
    // Write back only if the loop changed it.
    auto *Final = new LoadInst(C->Ty, Slot, "nsvp.final", WBPoint->getIterator());
    auto *Changed = new ICmpInst(WBPoint->getIterator(), ICmpInst::ICMP_NE,
                                 Final, C->Init, "nsvp.changed");
    Instruction *Then =
        SplitBlockAndInsertIfThen(Changed, WBPoint->getIterator(), false,
                                  nullptr, &DTU, &LI);
    new StoreInst(Final, C->PtrInCheck, false, C->Alignment,
                  Then->getIterator());
    Slots.push_back(Slot);
    ++NumPromoted;
  }
  if (!Slots.empty())
    PromoteMemToReg(Slots, DT, &AC);

  if (!NeedCheck) {
    ++NumInPlace;
    ORE.emit([&]() {
      return OptimizationRemark(DEBUG_TYPE, "Promoted", L.getStartLoc(),
                                L.getHeader())
             << "kept " << ore::NV("Locations", (unsigned)Live.size())
             << " memory locations in registers (no overlap check needed)";
    });
    return true;
  }

  // --- The check, from the copy's ranges ------------------------------------
  SE.forgetLoop(&L);
  auto *BI = cast<BranchInst>(CheckBB->getTerminator());
  auto Abandon = [&](const Twine &Why) {
    BI->setCondition(ConstantInt::getTrue(F.getContext()));
    ++NumAbandoned;
    missed(L, "CheckFailed", Why);
    return true;
  };

  // Group ranges whose bounds differ by constants into one: the promoted
  // locations among themselves, the other accesses among themselves.
  SmallVector<Range, 8> Groups;
  SmallVector<bool, 8> IsCandGroup;
  auto AddToGroup = [&](Range R, bool Cand) -> unsigned {
    for (unsigned G = 0; G < Groups.size(); ++G) {
      if (IsCandGroup[G] != Cand)
        continue;
      auto *DLo = dyn_cast<SCEVConstant>(SE.getMinusSCEV(R.Lo, Groups[G].Lo));
      auto *DHi = dyn_cast<SCEVConstant>(SE.getMinusSCEV(R.Hi, Groups[G].Hi));
      if (!DLo || !DHi)
        continue;
      if (DLo->getAPInt().isNegative())
        Groups[G].Lo = R.Lo;
      if (DHi->getAPInt().isStrictlyPositive())
        Groups[G].Hi = R.Hi;
      Groups[G].Guards.append(R.Guards.begin(), R.Guards.end());
      return G;
    }
    Groups.push_back(R);
    IsCandGroup.push_back(Cand);
    return Groups.size() - 1;
  };
  SmallVector<std::pair<unsigned, unsigned>, 8> Pairs;
  for (Candidate *C : Live) {
    if (C->Checks.empty())
      continue;
    Type *IntPtrTy = DL.getIntPtrType(C->Ptr->getType());
    unsigned CG = AddToGroup(
        Range{C->Addr,
              SE.getAddExpr(C->Addr,
                            SE.getConstant(IntPtrTy, DL.getTypeStoreSize(C->Ty)
                                                         .getFixedValue())),
              {}},
        /*Cand=*/true);
    for (Instruction *I : C->Checks) {
      std::optional<Range> R = rangeOf(I, L, /*Final=*/true);
      if (!R)
        return Abandon("the range of an access could not be computed");
      unsigned OG = AddToGroup(*R, /*Cand=*/false);
      auto P = std::make_pair(CG, OG);
      if (!is_contained(Pairs, P))
        Pairs.push_back(P);
    }
  }
  if (Pairs.size() > MaxChecks)
    return Abandon("the check would need too many range tests");
  for (Range &G : Groups)
    if (!Exp.isSafeToExpandAt(G.Lo, BI) || !Exp.isSafeToExpandAt(G.Hi, BI) ||
        any_of(G.Guards, [&](auto &Gd) {
          return !Exp.isSafeToExpandAt(Gd.first, BI);
        }))
      return Abandon("a range bound is not available before the loop");

  IRBuilder<> B(BI);
  Type *PtrTy = PointerType::get(F.getContext(), 0);
  SmallVector<std::pair<Value *, Value *>, 8> Bounds;
  Value *Bad = B.getFalse();
  for (Range &G : Groups) {
    Value *Lo = Exp.expandCodeFor(G.Lo, PtrTy, BI);
    Value *Hi = Exp.expandCodeFor(G.Hi, PtrTy, BI);
    B.SetInsertPoint(BI);
    // A range that wraps around the address space fails the check.
    Bad = B.CreateOr(Bad, B.CreateICmpULT(Hi, Lo, "nsvp.wrap"));
    for (auto &[N, Step] : G.Guards) {
      Value *NV = Exp.expandCodeFor(N, N->getType(), BI);
      B.SetInsertPoint(BI);
      uint64_t Limit = (uint64_t)INT64_MAX / Step;
      Bad = B.CreateOr(Bad, B.CreateICmpUGT(
                                NV, ConstantInt::get(N->getType(), Limit),
                                "nsvp.long"));
    }
    Bounds.push_back({Lo, Hi});
  }
  for (auto [A, C] : Pairs) {
    Value *Overlap = B.CreateAnd(B.CreateICmpULT(Bounds[A].first, Bounds[C].second),
                                 B.CreateICmpULT(Bounds[C].first, Bounds[A].second),
                                 "nsvp.overlap");
    Bad = B.CreateOr(Bad, Overlap, "nsvp.conflict");
  }
  BI->setCondition(Bad);

  ++NumVersioned;
  ORE.emit([&]() {
    return OptimizationRemark(DEBUG_TYPE, "Versioned", L.getStartLoc(),
                              L.getHeader())
           << "kept " << ore::NV("Locations", (unsigned)Live.size())
           << " memory locations in registers, versioned on "
           << ore::NV("Tests", (unsigned)Pairs.size())
           << " range-overlap tests";
  });
  return true;
}

bool VersionPromote::run() {
  SmallVector<Loop *, 8> Loops;
  for (Loop *L : LI.getLoopsInPreorder())
    if (L->isInnermost())
      Loops.push_back(L);
  bool Changed = false;
  for (Loop *L : Loops) {
    if (!L->isLoopSimplifyForm())
      Changed |= simplifyLoop(L, &DT, &LI, &SE, &AC, nullptr, false);
    if (!L->isRecursivelyLCSSAForm(DT, LI))
      Changed |= formLCSSARecursively(*L, DT, &LI, &SE);
    Changed |= processLoop(*L);
  }
  return Changed;
}

} // namespace

PreservedAnalyses RISCVNSVersionPromotePass::run(Function &F,
                                                 FunctionAnalysisManager &AM) {
  if (DisableVersionPromote || F.hasOptNone())
    return PreservedAnalyses::all();
  auto &LI = AM.getResult<LoopAnalysis>(F);
  if (LI.empty())
    return PreservedAnalyses::all();
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
  auto &SE = AM.getResult<ScalarEvolutionAnalysis>(F);
  auto &AA = AM.getResult<AAManager>(F);
  auto &AC = AM.getResult<AssumptionAnalysis>(F);
  auto &LAIs = AM.getResult<LoopAccessAnalysis>(F);
  auto &ORE = AM.getResult<OptimizationRemarkEmitterAnalysis>(F);
  if (!VersionPromote(F, LI, DT, SE, AA, AC, LAIs, ORE).run())
    return PreservedAnalyses::all();
  return PreservedAnalyses::none();
}
