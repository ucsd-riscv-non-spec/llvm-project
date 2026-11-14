# Decide/act loop splitting (`-riscv-ns-disable-decide-act`)

Record of a new IR pass, `riscv-decide-act` (`RISCVLoopDecideAct.cpp`), and of how it was verified.
On by default, with `-riscv-ns-disable-decide-act` to turn it off, like the other Non-Spec options.
Started 2026-11-13 on `non-spec` at `90a583dfb8c6`.

## 1. Why

A pb waits at dispatch until its condition is computed. In a loop like dijkstra's relax loop

```c
for (i = 0; i < NUM_NODES; i++)
  if ((iCost = AdjMatrix[iNode][i]) != NONE)              // load, then branch
    if (NONE == rgnNodes[i].iDist || rgnNodes[i].iDist > iCost + iDist) {   // load, then branch
      rgnNodes[i].iDist = iDist + iCost; rgnNodes[i].iPrev = iNode;
      enqueue(i, iDist + iCost, iNode);                   // malloc, a list walk
    }
```

both conditions wait on a load every iteration: 56% of dijkstra's pb stall cycles (the pb stall
profile, gem5 `--pb_profile`). Condition pipelining gives up because the loop contains another loop
(`enqueue`'s list walk, inlined) and a call (`malloc`); IR loop fission because the trip count (100)
exceeds the condition queue (64), and for the call and stores (the passes' remarks say so).

Splitting it by hand into a loop that only decides and one that acts took pb from 58.52M to 32.90M
cycles and vanilla by only -1.4%: a pb-specific gain worth a pass.

## 2. What it does

```
for (i...) { if (decide(i)) act(i); }
  =>
n = 0;
for (i...) { list[n] = i; n += decide(i); }         // decide loop: no branch but its latch
for (m = 0; m < n; ++m) { i = list[m]; act(i); }    // act loop: the original loop, in place
```

- **The decide part** is the blocks from the header on that have no effects, are in no inner loop and do
  not leave the loop; the first block that does starts the guarded (act) part, which must be the only
  way into the rest of the loop.
- **The decide loop** is one new block: the decide part if-converted (each block's predicate, each
  branch an `and`/`or` of 0/1 values), storing every index into a stack list and counting the chosen
  ones. Its copies drop UB-implying attributes and metadata, as they now run on every path.
- **The act loop** is the original loop, kept in place: its induction variable becomes `list[m]`, its
  exit `m + 1 != n`, and every condition branch that led only to the latch becomes unconditional
  (every act iteration reaches the guarded part). Its inner loops, calls and exits are untouched.
- **After the loop:** values of the decide part (and the counter) are taken from the decide loop's last
  iteration; carried values (dijkstra's globals promoted to registers) from the act loop, or as they
  entered if no iteration was chosen.

**Legal when:**
- the decide part has no side effects and every instruction in it is safe to run in every iteration
  (a load not in the header must be dereferenceable over the loop, `isDereferenceableAndAlignedInLoop`);
- no write in the guarded part of an iteration j can change what the decide part of a later iteration
  i > j reads. Alias analysis is asked about every access through each pointer
  (`MemoryLocation::getBeforeOrAfter`, TBAA kept, scoped noalias dropped, as condition pipelining does),
  since a per-iteration query says nothing across iterations; what it cannot rule out is checked with
  SCEV: both addresses affine in the loop with the same constant step (or one invariant), a constant
  apart, and no overlap for any j < i below the trip-count bound. Calls must not modify the condition's
  memory at all (e.g. `malloc`: `memory(inaccessiblemem: readwrite)`).
- writes on paths that end in a call that never returns (`exit()`) do not count: the program ends there,
  and the decide loop only loaded early;
- the decide part does not use a value carried between iterations; a carried value comes back through
  a latch phi that keeps it unchanged on the paths that skip the guarded part.

**Only where it pays:** the guarded part has a call or an inner loop (otherwise condition pipelining
and fission handle the loop), a condition reads memory, and the trip count has a bound of at most
`-riscv-ns-decide-act-max-trip` (1024; the list's size). Every rejection is a missed remark
(`-Rpass-missed=riscv-decide-act`).

It runs after IR loop fission and before condition pipelining.

## 3. Layer 1: nothing changes with the flag given

- `hoistfix/buildall.sh` (musl `96ae942`, compiler-rt, the 8 MiBench programs): with
  `-riscv-ns-disable-decide-act`, 1,544 of 1,544 files are byte-identical to `90a583dfb8c6`.
- RISC-V lit: the same 1,684 failing tests by name; the non-spec tests pass, with the new one.
- **With the pass on, only `apps/dijkstra` changes**: no loop in musl, compiler-rt or the other 7
  programs is split.

## 4. Layer 2: tests

`ns-decide-act.ll` (IR from C, `llc -O2 -stop-after=riscv-decide-act`; on, off, remarks):

| Function | Expected |
|---|---|
| `split` | split: two loads and `&&` become one 0/1 value, the list is stored, the act loop runs over it; the guarded store to `D[i]` only meets the load of `D[i]` in the same iteration |
| `live_out` | split; the returned value is the decide loop's last load |
| `noreturn` | split, although the guarded part has an error path that calls `report()` and `die()` |
| `clobber` | not split: the guarded store to `D[i + 1]` is what the next iteration reads |
| `unknown_call` | not split: `opaque()` may write the condition's memory |
| `no_call` | not split: for pipelining and fission |
| `carried` | not split: the condition reads a value carried between iterations |

Mutation testing (each caught by the test): skipping the memory check; per-iteration alias queries
(the first version's bug, below); not rewriting the uses after the loop; letting a block that leaves
the loop into the decide part; skipping the carried-value check.

**What did not go as planned:**
- The first version asked alias analysis about the load and the store as they are within one iteration.
  For `D[i + 1] = ...` against a load of `D[i]` that says "no alias", and `clobber` was split, which is
  wrong. Fixed as above; the test case is that one.
- A block without effects that branches out of the loop (the error check before `die()`) was taken into
  the decide part, so `noreturn` was rejected. Such a block now starts the guarded part.
- `isDereferenceableAndAlignedInLoop` dereferences its assumption cache even though it defaults to null:
  the first build crashed on dijkstra. The pass now requires `AssumptionCacheTracker`.

## 5. Layer 3: interplay

- The non-spec tests (pipelining with queues, speculative and shift; fission; setup hoisting; exit-jump
  split) pass unchanged.
- `-verify-machineinstrs` on dijkstra: no errors. musl is byte-identical (nothing split), so its
  verifier results are those of `setup-placement.md`.

## 6. Layer 4: gem5

dijkstra is the only program that changes. Built with the from-scratch runner (`fromscratch-optimized`,
musl and compiler-rt from source with the same compiler, `-falign-loops=32`, nonspec inline threshold
500) and run with `gem5.opt.pbprofile`, `configs/pbtb` defaults:

| | before (`90a583dfb8c6`) | with the pass | |
|---|---|---|---|
| cycles | 58,516,768 | **32,737,201** | -44.1% |
| committed instructions | 54,711,534 | 50,740,979 | -7.3% |
| pb stall cycles | 43,475,156 | 19,112,634 | -56.0% |
| mispredicts | 117,638 | 90,565 | -23.0% |
| pb / vanilla (vanilla 18,602,960) | 3.146 | **1.760** | |

- Output identical to vanilla's. No ordinary branch in the binary; all of the runner's checks pass.
- On the unchanged source it does slightly better than the hand split (32,904,179 cycles).
- The remaining stalls are `enqueue`'s list walk (64%): pointer chasing that no pass can compute early.
- The other 7 programs are byte-identical, so unchanged.

## 7. Layer 5: checks inside the pass

In assertion builds the pass runs the IR verifier on each function it changed. The rebuild above (all
of musl, compiler-rt, the 8 programs) is with it active; it never fires.

## 8. Files

- `llvm/lib/Target/RISCV/RISCVLoopDecideAct.cpp` (new), `RISCV.h`, `RISCVTargetMachine.cpp`, `CMakeLists.txt`
- `llvm/test/CodeGen/RISCV/ns-decide-act.ll`
- `ns-docs/decide-act.md`
