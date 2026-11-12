# Branch setup placement (`-riscv-ns-disable-split-exit-jumps`, `-riscv-ns-disable-hoist-freq-from-best`)

Record of changes to where branch setups end up, and of how each was verified. Branch `setup-placement`, on top of `non-spec` at `23c5fa3fc5cb`. Started 2026-11-11.

Unlike `-riscv-ns-swp-speculative` and `-riscv-ns-swp-shift-recurrences`, these changes are **on by default**, with a `-riscv-ns-disable-...` flag to turn each off, like the other Non-Spec options.

Verification is done in layers, as in `swp-speculative.md`.

---

## 1. Why

Setups (`bmovs`/`bmovt`) are a large share of what pb code executes. Counted from committed-instruction traces in gem5:

| | Setups among committed instructions | Setups whose branch is not reached afterwards |
|---|---|---|
| `qsort` small | 28% | 7.4% |
| `patricia` small | 18% | 4.7% |
| `stringsearch` small | 33% | 11.8% |

Two placement problems cause much of this:

1. **The exit jump of a loop runs its setup on every iteration** (this change, change 3). When a loop's branch does not lead to the block placed after the loop, the loop leaves through a jump. Its setup ends up in the loop's last block, which runs every iteration, while the jump runs once.
   - `bitcount`'s `main`: 1.05M of its 2.10M extra instructions.
   - `dijkstra`'s relax loop: about 3M.
2. **A rarely entered loop's setups are hoisted to the function entry** (change 4). The hoisting pass's frequency guard compares against the frequency of the setup's own block, which inside a loop includes the trip count.
   - musl's `memcpy` sets up the branches of its unaligned-copy loops on every call: about 18 wasted setups per call, 3.6M instructions in `qsort`.
   - Most of those turned out to be exit-jump setups, which change 3 already keeps out of the entry, so change 4 has much less left to do than this estimate (section 9).

## 2. Change 3: the exit jump and its setup get a block of their own

### 2.1 What happened before

`RISCVExpandPseudo` expands a block ending in a conditional branch and a jump into:

```
PseudoPBC %a, %bb.taken
%b = BMOVS_J ; %b = BMOVT_J %b, %bb.other     <- after a terminator
PseudoPBU %b, %bb.other
```

The jump's setup is after the `PseudoPBC`, which is not valid MIR. This was noted in `swp-speculative.md` (5.2), where the machine verifier failed on it. `RISCVBranchSetupHoisting` then always moves the setup above the `PseudoPBC`, which makes the MIR valid, but the setup now runs whenever the block runs.

- In a loop's last block, that is every iteration.
- If the loop has a call, the call keeps the setup there.
- If not, the setup is hoisted out of the loop, and then possibly further (change 4).

### 2.2 What was changed

In `RISCVExpandPseudoInsts.cpp`, about 90 lines, a new step `splitJumpsAfterCondBranches` runs after the expansion loop. It is on unless `-riscv-ns-disable-split-exit-jumps` is given, with the statistic `NumExitJumpsSplit`.

- **What it matches:** a block ending in `PseudoPBC`, then `BMOVS_J`, `BMOVT_J`, `PseudoPBU`, all on the jump's register, with the `BMOVT_J` target equal to the jump's.
  - This covers ordinary compares, hardware-loop ends and queue branches, which all expand to `PseudoPBC`.
  - It walks back from the end of the block. `getFirstTerminator()` cannot find the `PseudoPBC`: it only sees the terminators at the very end, and the setup is in between. The first version used it and matched nothing.
- **What it does:** moves the last three instructions into a new block inserted right after, which the conditional falls through to.
  - `replaceSuccessor` gives the new block the fall-through edge with its probability; the new block's only successor is the jump's destination.
  - Its live-ins are the destination's, since it uses and defines no physical register.
  - The emitted instructions are in the same order as before hoisting.
- **Skipped:**
  - when the conditional's target and the jump's are the same block, because then both edges share one successor entry;
  - when the destination is an EH pad.
- The hoisting pass is unchanged. It now sees the setup in a block that runs as often as the jump, so its call barrier and frequency guard keep the setup on the exit path in a loop with a call, and let it leave a loop without one.
- **Checks in assertion builds:**
  - the conditional's block has exactly two successors, the target and the new block, and is no longer a successor of the destination;
  - the new block follows it in layout;
  - the new block has one predecessor and one successor and holds exactly the setup and the jump.

## 3. Change 3, layer 1: nothing changes with the flag given

**Setup:**
- The `llvm-ns-latest` build was reconfigured with `LLVM_INCLUDE_TESTS=ON` (the compiler itself unchanged), and `llc` and the test tools were built.
- Saved in `claude-scratch/hoistfix/`: the original pass files (`orig/`), `llc.orig`, `clang.orig`.
- `buildall.sh <tag> [flags]` builds, with this tree's clang at `-O2 -mllvm -riscv-ns-swp-speculative -mllvm -riscv-ns-swp-shift-recurrences`:
  - musl `96ae942`;
  - compiler-rt builtins and crt from this tree;
  - the 8 MiBench programs (basicmath, bitcount, qsort, susan, dijkstra, patricia, stringsearch, sha), linked against those two.
- `cmpall.sh A B` compares every musl object and archive, compiler-rt object and program, with `.comment` removed.

**Results:**
- **Unmodified compiler against the new one with `-riscv-ns-disable-split-exit-jumps`:** 1,544 of 1,544 files identical.
- **The new compiler with the change on:**
  - the 13 non-spec tests pass;
  - the RISC-V suite fails the same 1,684 of 2,223 tests, by name, as the unmodified compiler (the fork's branches break upstream tests; see `swp-speculative.md` 4.3);
  - 224 files change, 197 of them musl objects.

## 4. Change 3, layer 2: tests

Two new files.

**`ns-split-exit-jumps.mir`** runs `riscv-expand-pseudo` alone, with the change on and off:

| Function | Checks |
|---|---|
| `split` | The new block follows the conditional's block, which keeps the fall-through probability (`0x04000000`) toward it; the new block's successor is the destination at 100%, its live-ins are the destination's (`$x12`), and it holds `BMOVS_J`, `BMOVT_J`, `PseudoPBU` only. Off: all four stay in one block. |
| `same_target` | Both edges to one block: not split. |
| `queue` | A queue branch (`PseudoNSQBR`) followed by a jump: split. |
| `fallthrough_only` | No jump: untouched. |

**`ns-split-exit-jumps-asm.ll`** runs full `llc -O2 -mattr=+m`:

| Function | Checks |
|---|---|
| `call_loop` (a counted loop calling a function, in a counted outer loop) | Each loop's exit-jump setup comes after the loop's `pb`, in its own block, and no `.Lns_j_` setup is in either loop body. Off: both setups are in the loop bodies, before the loop's `pb`. |
| `nocall_loop` (the same without a call) | The exit jumps' blocks hold only the `pb`, and no setup is left in either loop: they are hoisted out, as before. |

**Mutation testing.** The pass was broken four ways, `llc` rebuilt each time, and lit's exit code used (`lit -q` prints no `FAIL:` lines):

| Mistake | `.mir` test | asm test |
|---|---|---|
| Never split | caught | caught |
| Old edge kept and a new one added | caught | caught |
| Live-ins not copied | caught | caught |
| Same-target guard removed | caught | not caught (no such case in it) |

The pass was restored byte-identical from `l2/pass.layer2.cpp`; 15 of 15 non-spec tests pass.

**What did not go as planned:**
- The MIR printer writes two spaces on empty lines, so `CHECK-EMPTY` does not match them, and `{{^ *$}}` matches the end of the previous line. The checks use a `CHECK-NOT` for instructions followed by a `CHECK` instead.
- `nocall_loop`'s first draft took the first `.Lns_j_` setup in the entry block for an exit jump's; it is the jump into the outer loop. It was rewritten to check the exit jumps' blocks and loop bodies directly.

## 5. Change 3, layer 3: interplay

- **Queues:** the setups of `-riscv-ns-swp-speculative` queues stay in their preheader. `ns-branch-condition-pipelining-speculative-asm.ll` and `-shift-asm.ll` still pass.
- **Machine verifier**, on all of musl with `-mllvm -verify-machineinstrs` (`hoistfix/l3/verify.sh`):

  | | Without the change | With it |
  |---|---|---|
  | "Non-terminator instruction after the first terminator" | 842+ | **0** |
  | "Expected implicit register after groups" | present | present |

  The second error is a pre-existing problem with inline-asm operands, the same both ways. No new kind of error appears. (The two builds ran at once, so some messages in the logs are interleaved.)

  So this change also fixes the verifier failure noted in `swp-speculative.md` 5.2.

## 6. Change 3, layer 4: gem5

The baseline (`bse`) and change-3 (`on3`) builds of musl, compiler-rt and the 8 programs (`hoistfix/buildall.sh`) were copied to directories of equal name length, since paths go on the simulated stack (`hoistfix/g4/bin/`). They were run with the final core config and every PBTB option at its default. Instruction counts per PC come from `Exec` traces streamed through awk (`g4/out-prof/`), classified with `pbprof/setups/classify.py`.

**Outputs:** identical to vanilla in all 8 programs (sha differs from vanilla as it always has, by the same amount in both builds). The leak (committed ordinary control minus pbs) is 0, or 2 in bitcount, in both.

**Committed instructions, `on3` against `bse`:** dijkstra -5.18%, stringsearch -3.32%, qsort -2.92%, bitcount -2.08%, basicmath -1.45%, patricia -1.43%, sha -0.65%, susan -0.05%.

**Where they go.** Setups whose branch is not reached afterwards:

| Program | `bse` | `on3` |
|---|---|---|
| bitcount | 1,057,846 | 5,866 |
| dijkstra | 5,383,770 | 2,373,922 |
| qsort | 5,101,645 | 3,087,748 |

- bitcount's is `main`'s exit jump (`b12`); dijkstra's is the relax loop's exit (`b17`), as predicted in section 1.
- Setups run again on every iteration are unchanged (bitcount 1.05M, dijkstra 9.38M, qsort 2.26M): those are the setups a call keeps in their loop, which this change does not touch.

**Cycles** were measured on two gem5 versions:

| Program | gem5 `16a4f2f808` | gem5 `a4485a886e` (adds dispatch resolution and the return address stack) |
|---|---|---|
| basicmath | -0.35% | -1.00% |
| patricia | -1.45% | -0.96% |
| qsort | -0.72% | -0.33% |
| bitcount | +0.24% | -0.27% |
| dijkstra | +0.09% | +0.04% |
| susan | +0.10% | +0.10% |
| stringsearch | -0.67% | +1.19% |
| sha | +0.25% | -5.51% (unreliable) |

- sha's memory-dependence prediction varies with timing, so its cycles are not used.
- These programs are mostly limited by pb dispatch stalls, not by instruction count, so a few percent fewer instructions move cycles by about 1% at most.
- These builds are not loop-aligned (`-falign-loops`): moving code by a few bytes can change cycles by more than these differences.

**What did not go as planned: mispredicts rose in stringsearch (2,923 -> 3,737) and qsort on the newer gem5.** The stringsearch increase was traced per PC (`hoistfix/mp/`):
- Most of it is `printf_core`'s `pb b5` (192 -> 716), the compare of the format character with `%`.
- Its code is identical in both builds. It only sits 8 bytes earlier, because one exit-jump setup before it (`b10`'s) moved to the exit block.
- The rest is also data-dependent branches in memchr and memcpy, not the moved exit jumps.
- So the change shifted where branches sit, and with it which predictor entries they share. The change itself does not mispredict.

## 7. Change 3, layer 5: checks inside the pass

- With the asserts active, musl, compiler-rt and the 8 programs are rebuilt: 1,544 of 1,544 files are identical to the build without them, so none fires.
- **Do they fire:**

  | Mistake | Result |
  |---|---|
  | Old edge kept and a new one added | The successor assert fires |
  | New block placed at the end of the function | The layout assert fires |

- The pass was restored byte-identical from `l5/pass.layer5.cpp`; 15 of 15 non-spec tests pass.

**Correction (found during change 4).** `cmpall.sh` selected the programs with `-path '*/apps/*'`, which never matches the relative paths it lists, so the 8 programs were left out of every comparison above, and the totals said 1,536. With the pattern fixed to `apps/*`, the comparisons of sections 3 and 7 were redone: 1,544 of 1,544 identical in both (the numbers above are the corrected ones).

## 8. Change 4: compare a hoist point with the coolest point reached

### 8.1 What happened before

`RISCVBranchSetupHoisting::findHoistPoints` walks up the dominator tree from the setup and accepts each dominator as a hoist point, unless it runs more than `-riscv-ns-hoist-max-freq-ratio` (2) times as often as **the setup's own block**. The setup goes to the furthest point accepted.

For a setup in a loop body, that block's frequency includes the trip count. A loop entered 1 time in 16 with 32 iterations has a body frequency of 2 per call, against the entry's 1, so the entry is accepted, and the setup runs on every call, whether or not the loop is entered. The preheader in between runs 1/16 per call.

### 8.2 What was changed

In `RISCVBranchSetupHoisting.cpp`, about 20 lines. On unless `-riscv-ns-disable-hoist-freq-from-best` is given, with the statistic `NumHoistPointsTooHotFromBest` (points this change rejects that the old rule would have accepted).

- `findHoistPoints` keeps `Ref`, the lowest frequency among the setup's own block and every point accepted so far, and compares each candidate with `Ref` instead of with the setup's block.
- In the example, the preheader (1/16) is accepted and becomes `Ref`. The entry (1) is more than twice that, so it is passed, and the setup stays in the preheader.
- A point that is passed does not end the walk, as before. A cooler block further up can still be accepted.
- `Ref` only goes down, so a point accepted later is never more than twice as hot as any earlier one.
- **Check in assertion builds:** when a point is accepted, the lowest frequency is recomputed from the setup's block and the points already in the list, independently of `Ref`, and the new point must be within the ratio of it.

## 9. Change 4, layer 1: nothing changes with the flag given

All with the final compiler (assertion build):

| Comparison | Result |
|---|---|
| Change 3 (`on3`) against change 4 with `-riscv-ns-disable-hoist-freq-from-best` | 1,544 of 1,544 files identical |
| RISC-V lit suite | the same 1,684 of 2,227 failing tests, by name, as the unmodified compiler; the two new tests pass |
| Non-spec tests (`ns-*`, `branch-setup-*`) | 17 of 17 pass |
| Change 3 against change 4 on | 86 files differ (73 musl objects, and all 8 programs) |

Over musl, the change rejects 1,956 hoist points that the old rule accepted.

What the 86 files show: only setups moving between neighbouring blocks. In `memcpy`, which motivated the change, it does nothing: change 3 had already moved the exit-jump setups that reached its entry.

## 10. Change 4, layer 2: tests

Two new files.

**`ns-hoist-freq-from-best.mir`** runs `riscv-branch-setup-hoisting` alone, with the change on and off:

| Function | Checks |
|---|---|
| `rare_loop` (loop entered 1 time in 16, 32 iterations) | The loop's `BMOVS`/`BMOVT` go to the preheader, not the entry, and none are left in the body. Off: they go to the entry. |
| `common_loop` (the same loop, entered 3 times in 4) | The setups still go to the entry: the entry is within twice the preheader. |
| `coolest_not_last` (a chain: entry 1, `bb.1` 0.25, `bb.2` 0.15, preheader 0.09) | The setups end in `bb.2`. `bb.1` is within twice `bb.2`, the last point accepted, but not twice the preheader, the coolest. Off: the entry. |

**`ns-hoist-freq-from-best-asm.ll`** runs full `llc -O2 -mattr=+m` on the same two loops, written in IR with branch weights and with a data-dependent branch in the body. The loop itself is a hardware loop, whose setups the hardware-loop pass already puts in the preheader, so the test checks the inner branch's setup (`.Lns_bge_`):
- `rare_loop`: in the preheader, not the entry. Off: in the entry.
- `common_loop`: in the entry, both ways.

**Mutation testing.** The pass was broken three ways, `llc` rebuilt each time, and lit's exit code used:

| Mistake | `.mir` test | asm test |
|---|---|---|
| Compare with the setup's own block (the old rule) | caught | caught |
| `Ref` is the last point accepted, not the coolest | caught | not caught (only one step in it) |
| `Ref` never updated | caught | caught |

The pass was restored byte-identical from `l2c4/pass.layer2.cpp`.

**What did not go as planned:**
- The mutation script printed `$?` after a `$(basename ...)` in the same `echo`, which reset it, so the first run reported every mutation as passing. Checked by hand, fixed to save the exit code first, and rerun; the table is from the rerun.
- The first asm test used a loop without an inner branch; its setups start in the preheader (hardware loop), so the change had nothing to move.

## 11. Change 4, layer 3: interplay

- **Queues:** `ns-branch-condition-pipelining-speculative-asm.ll` and `-shift-asm.ll` still pass (among the 17 above).
- **Machine verifier**, on all of musl with `-mllvm -verify-machineinstrs` (`hoistfix/l3/verify.sh`, now with `make -O` so the logs of parallel compiles are not interleaved):

  | | Change 4 off | Change 4 on |
  |---|---|---|
  | "Expected implicit register after groups" (the inline-asm problem of section 5) | 602, in 329 files | 602, in the same functions |
  | Any other error | 0 | 0 |

## 12. Change 4, layer 4: gem5

Built and run as in section 6, change 4 (`on4`) against change 3 (`on3`), on gem5 `a4485a886e` with that commit's core config (before the L1-I prefetcher and the 32-entry fetch queue of `5a358108ea`).

**Outputs:** identical to vanilla in all 8 programs (sha as always). The leak is unchanged (0, or 2 in bitcount).

| Program | Committed instructions | Cycles |
|---|---|---|
| stringsearch | -1.43% | -2.11% |
| basicmath | -0.41% | -0.36% |
| patricia | -0.33% | +0.59% |
| qsort | -0.09% | -0.53% |
| dijkstra | -0.06% | +0.34% |
| bitcount | -0.00% | -0.00% |
| susan | -0.01% | +0.02% |
| sha | -0.00% | -0.00% (unreliable) |

- Committed instructions go down or stay flat in every program; cycle changes under about 1.5% are within the code-placement noise described in section 6.
- The final compiler's build (`on6`, with the assertion) is byte-identical to `on4`, so these numbers hold for it.

## 13. Change 4, layer 5: checks inside the pass

- With the assertion active, musl, compiler-rt and the 8 programs are rebuilt (`on6`): 1,544 of 1,544 files are identical to the build without it (`on4`), and it does not fire.
- **Does it fire:**

  | Mistake | Result |
  |---|---|
  | `Ref` is the last point accepted, not the coolest | The assertion fires (on `coolest_not_last`) |
  | `Ref` never updated | The assertion fires (on `rare_loop`) |

- The pass was restored byte-identical from `l2c4/l5/pass.layer5.cpp`; 17 of 17 non-spec tests pass.

## 14. Files

- `llvm/lib/Target/RISCV/RISCVExpandPseudoInsts.cpp` (change 3)
- `llvm/lib/Target/RISCV/RISCVBranchSetupHoisting.cpp` (change 4)
- `llvm/test/CodeGen/RISCV/ns-split-exit-jumps.mir`
- `llvm/test/CodeGen/RISCV/ns-split-exit-jumps-asm.ll`
- `llvm/test/CodeGen/RISCV/ns-hoist-freq-from-best.mir`
- `llvm/test/CodeGen/RISCV/ns-hoist-freq-from-best-asm.ll`
- `ns-docs/setup-placement.md`
