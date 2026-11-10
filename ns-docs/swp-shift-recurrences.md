# Shift recurrences in branch condition pipelining (`-riscv-ns-swp-shift-recurrences`)

Record of the second change to `RISCVBranchConditionPipelining`: conditions on a value shifted by the same constant every iteration (`x >>= 1`) can be computed ahead, alongside affine induction variables. Branch `swp-speculative`, on top of change 1 (`dbcdc6c8096e`, `-riscv-ns-swp-speculative`, recorded in `swp-speculative.md`). Started 2026-11-10.

The change is verified in the same layers as change 1; all five are done (layer 5 in section 8). Section 7 records a distance sweep and a per-call mispredict that first looked like a compiler problem with leftover queue bits and turned out to be a gem5 bug. Section 9 records a follow-up: loops without a trip count now default to distance 4 (`-riscv-ns-swp-speculative-distance`).

---

## 1. Why

After `acb2007` and the paired-condition patch, MiBench `bitcount` spends about 12.4M of 23.7M cycles with dispatch blocked behind a `pb` (v3 in `bitcount-v3-gap-analysis`). Three while-loops cause 88% of it. Change 1 removed their trip-count rejection. Their conditions still could not be computed ahead, because the pass only advanced affine induction variables (`Start + I * Step`).

The IR just before the pass:

| Loop | Share of stall cycles | Recurrence | Closed form |
|---|---:|---|---|
| `bit_shifter` | 55.7% | `x = ashr x, 1`, plus affine `i` | `ashr x0, j` |
| `ntbl_bitcnt` | 12.1% | `x = ashr x, 4` | `ashr x0, 4j` |
| `bit_count` | 20.1% | `x = and (x - 1), x` | none |

### Exit merging, the question left open by the gap analysis

`bit_shifter`'s source condition is `x && i < 64`. The binary had two `bmovc`/`pb` pairs per iteration, while vanilla had one branch, so exit merging looked missing.

**It is not missing at the IR level.** Before this pass, the two tests are already one branch:

```llvm
%tobool = icmp ugt i64 %x.addr.08, 1
%cmp    = icmp samesign ult i32 %i.09, 63
%1      = select i1 %tobool, i1 %cmp, i1 false
br i1 %1, label %for.body, label %for.end.loopexit
```

So the split into two branches happens later, in code generation. LLVM turns a logical-and branch condition into two branches when it considers branches cheap; which step does it here was not confirmed. A pipelined loop branches on a `pop` instead, so there is nothing left to split, and `bit_shifter` needs no separate exit merging once its condition can be computed ahead.

## 2. What was changed

All in `RISCVBranchConditionPipelining.cpp`, about 56 lines.

- **New flag `-riscv-ns-swp-shift-recurrences`,** default off, separate from change 1's flag. bitcount's loops need both, as they also have no trip count.
- **`InductionVar`:** gains `ShiftOp` (`LShr`, `AShr` or `Shl`) and `ShiftAmount`, with `isShift()`. `Step` is null for a shift recurrence.
- **`findInductionVar`:** if the phi is not an affine recurrence and the flag is on, it is accepted when it is an integer header phi with two incoming values, the start from the preheader and `next` from the latch, where `next` is `lshr`/`ashr`/`shl` of the phi itself by a constant `c` with `0 < c < width`.
- **`advanceInductionVar`:** for K iterations ahead, it emits one shift by `c * K`.
  - Shifts compose by adding amounts, so this is exact.
  - At `c * K >= width`: `lshr` and `shl` give the constant 0, and `ashr` gives `ashr x, width - 1`, the sign.
  - The emitted amount is therefore always below the width. A larger amount would be poison in IR.
  - An assertion checks the range.
- **`materializeValues`:** skips expanding a step for shift recurrences.
- **Both modes:** this works with or without a trip count. The kernel, the prologue and the in-place transform all advance induction variables through `advanceInductionVar`.
- **Unchanged:** clones still drop flags (`exact`, `nuw`, `samesign`) through `cloneCondition` in speculative mode.

**Smoke test on bitcount**, both flags:

- `bit_shifter` and both of `bitcnt_4`'s loops (`ntbl_bitcnt`, `btbl_bitcnt`) are pipelined; `bit_count` is rejected ("cannot compute ahead").
- `bit_shifter`'s loop now has one `pb` per iteration on the merged condition. The kernel pushes `(x >> 16) > 1 && i + 16 < 63`; the prologue pushes `(x >> k) > 1` for k = 0 to 15, with `i < 63` folded away as always true there.
- It runs 11 instructions per iteration, against 9 before and 8 in vanilla, in exchange for a `pb` that should no longer wait.

---

## 3. Layer 1: nothing changes when the flag is off

Change 1's compiler was saved as the reference (`swpspec/c2/pass.change1.cpp`, `llc.change1`).

| Check | Result |
|---|---|
| 10 non-spec tests | all pass |
| Full RISC-V suite | the same 1,684 tests fail as before any change |
| 16 MiBench programs, both new flags off, against change 1 flag off | 112 ELF files, all identical |
| speculative only, against change 1 flag on | 112 ELF files, all identical: change 1's behavior is untouched |
| both flags on, against change 1 flag on | exactly one file differs: bitcount's `bench` |

Only bitcount has a shift recurrence in a loop whose condition can be computed ahead. fft's `NumberOfBitsNeeded` does not shift its loop variable.

**Two harness problems, fixed before the numbers above:**

1. **A flaky jpeg build.** The rebuild script copied sources with `cp -r`, which does not keep timestamps. When `jconfig.doc` landed in a later second than `jconfig.h`, `make` ran jpeg's "prepare `jconfig.h`" rule, which always fails. It hit one variant only, by timing. The script now uses `cp -rp`.
2. **A comparison that could pass falsely.** For a missing file, the comparison silently reused the previous one. It now counts a missing or unreadable file as a difference.

   Change 1's comparisons were not redone with the fixed check. All of their builds succeeded, so no file should have been missing there.

---

## 4. Layer 2: IR tests

New file `llvm/test/CodeGen/RISCV/ns-branch-condition-pipelining-shift.ll`. RUN lines:

- distance 4, both flags (`CHECK`);
- distance 16, both flags (`D16`), for totals past the width;
- distance 4, speculative only (`OFF`): none of the shift cases may be transformed.

Each function's output was read and checked by hand before its checks were written.

| Function | Checked |
|---|---|
| `ashr_by_4` (`ntbl_bitcnt`'s shape) | prologue shifts by 0, 4, 8, 12, then the update's `>> 4`; kernel `ashr x, 16` then `>> 4`; the pop drives the latch |
| `merged_exits` (`bit_shifter`'s shape) | the `select` of both tests is computed ahead whole; `i` advances as affine; the clone drops `samesign` |
| `lshr_by_1` | kernel `lshr x, 4`; the original keeps `exact`, the clone does not |
| `shl_by_3` | totals `3K` (`shl x0, 9` for K = 3, `shl x, 12` in the kernel); the original keeps `nuw` |
| `lshr_past_width` (`i8`, distance 16) | iterations below 8 shift by K; from 8 on, and in the kernel (shift by 16), `x` is the constant 0 |
| `ashr_past_width` (`i8`, distance 16) | `ashr x0, 6`, then `ashr x0, 7` for K = 7 and every K after; kernel `ashr x, 7` |
| `counted_with_shift` | with a trip count, the guard, kernel counter and drain path; the body branch on `x & 1` advances the shift in the prologue and kernel; the drain resumes from the kernel's `x` |
| `reject_variable_shift` | shift by a variable: no queue |
| `reject_not_own_shift` | `lshr (x + 1), 1`: no queue |
| `reject_double_shift` | two shifts in a row: no queue |
| `reject_clear_lowest` (`bit_count`'s shape) | `x & (x - 1)`: no queue |

(Shift amounts above are as emitted by this pass. Code generation later merges `(x >> a) >> b` into one shift.)

### What did not go as planned

- **`lshr_past_width`, first version** (`while (x != 0) x >>= 1` on an `i8`) was not transformed at distance 16. The debug output said "never runs more than 9 iterations": the compiler knows an `i8` shifted right by 1 reaches 0 within 9 iterations, and the pass skips loops that never run more than D iterations. That is correct; the test was badly chosen. It was rewritten with condition `zext(x) + k*k < n`, so the loop can run longer.
- **The last function's `CHECK-NOT`** ran to the end of the output and matched the intrinsic declarations. It now ends at the function's `ret`.

### Does the test catch mistakes

Four bugs introduced on purpose, `llc` rebuilt each time:

| Mistake | Result |
|---|---|
| `ashr` past the width gives 0 instead of the sign | caught (line 181, `D16`) |
| The total shift ignores the iteration count | caught (line 28) |
| The "shift of the phi itself" check is removed | caught (line 264, `reject_not_own_shift`) |
| The feature never activates | caught (line 23) |

The pass was restored from `swpspec/c2/pass.c2.l2.cpp`, byte-identical, and rebuilt. All 11 non-spec tests pass.

---

## 5. Layer 3: the hoisting pass keeps the queue's setup in place

New file `llvm/test/CodeGen/RISCV/ns-branch-condition-pipelining-shift-asm.ll`: full `llc` to assembly (`-mattr=+m`, both flags, distance 4). The function `reentry_shift` has an `ntbl_bitcnt`-shaped inner loop (`do { c += x & 15; x >>= 4; } while (x >= 16)`) re-entered by an outer loop, with a new `x` loaded from an array each time.

**The property:** the queue's `bmovs`/`bmovt` are in `%outer` (the inner loop's preheader), after the load of the new `x`, while the other setups (the outer hardware loop, the exit jump, the return) are hoisted to the entry.

**The test checks:**

- no queue setup before `%outer`;
- the setup and the prologue in `%outer`: pushes on `x >> 4` and on `x >> 16`, to the captured register;
- no `bmovs` of that register in `%inner`;
- the kernel's push on `x >> 20`, which is `(x >> 16) >> 4`, the next value in iteration i + 4, merged into one shift by code generation;
- the `pb`.

**First version too strict:** I copied change 1's check that the setup is the first instruction after the label. Here the load of the new `x` comes first. What matters is that the setup is in the preheader, so the check was changed. I had also written the kernel shift as 16 instead of 20.

**Catches the regression:** the same mutation as change 1's layer 3 (the queue rule off, and the queue's own pushes and `pb` treated as never conflicting). The setup moves to the entry, and the test fails at "no queue setup before `%outer`" (line 14). The hoisting pass was restored from `swpspec/l3/hoisting.orig.cpp` (it had not changed since change 1). All 12 non-spec tests pass.

---

## 6. Layer 4: running the programs in gem5

Setup as in change 1's layer 4: `gem5.opt.fixC`, all PBTB flags, model 2 and tage. Each run must give vanilla's output with `earlyDispatchOutcomeChanged` = 0. All runs went 32 at a time.

### 6.1 Generated shift-recurrence loops

- **Generator:** `swpspec/c2/l4/gen2.py`, seed 20261111. 32 programs, 6 `noinline` kernels each, 192 kernels in total.
- **Families:**

  | Family | Loop | Count |
  |---|---|---:|
  | `lshr_count` | `while (x) { …; x >>= c; }`, unsigned | 29 |
  | `ashr_table` | `ntbl_bitcnt`'s shape: `do { … x >>= c; } while (x >= T)`, signed | 26 |
  | `ashr_tozero` | `while (x != 0 && x != -1) x >>= c;` | 30 |
  | `merged` | `bit_shifter`'s shape: `x && i < LIM`, with `LIM` in {8, 16, 17, 31, 64} | 24 |
  | `shl_count` | `while (x) x <<= c;` | 26 |
  | `mixed` | `(x & 0xff) + i*i < LIM`, with `x >>= c`, so the loop outlives `x` | 31 |
  | `nested` | an inner `ashr` loop re-entered by an outer loop with a `volatile` count | 26 |

- **Shift amounts** are 1, 2, 3, 4, 5 and 7.
- **Inputs** (`volatile`): 0, 1, 2, 3, 15, 16, 17, 255, 256, 65535, 2^31, 2^62, -1, -2, -16, -17, `LONG_MIN` and `LONG_MAX`. So trip counts run from 0 to 64, and arithmetic shifts reach -1.
- **Re-entry:** each kernel is also called through all inputs again, twice.
- **Variants:** `van` (host clang), `off` (both flags off), and `on16` and `on4` (both flags, distances 16 and 4).

**Coverage:**

- At distance 4: all 192 loops pipelined.
- At distance 16: 181 pipelined. The other 11 the compiler proves never run more than 16 iterations (for example an unsigned 64-bit `x >>= 4`), which the pass skips at that distance, as designed.
- No condition was rejected.

**Results:**

- **Correctness:** all 128 runs give vanilla's output, with `earlyDispatchOutcomeChanged` = 0.
- **Performance is not meaningful here.** These programs are dominated by `printf`: about 250 lines each, through the pass-built musl, which has its own `pb` stalls (6.7M stall cycles even in the vanilla builds). Totals over the 32 programs:

  | Variant | Cycles | Instructions |
  |---|---:|---:|
  | off | 16.77M | 11.94M |
  | on16 | 16.25M | 12.48M |
  | on4 | 16.19M | 12.26M |
  | van | 15.92M | |

### 6.2 bitcount small (`75000`)

| Build | Cycles | x vanilla | Instructions | Mispredicts | pb stall cycles |
|---|---:|---:|---:|---:|---:|
| vanilla | 11,060,583 | 1.00 | 37.45M | 13,050 | 29,124 |
| v3 (change 1's compiler, default flags) | 23,726,885 | 2.15 | 43.66M | 12,241 | 12,402,406 |
| change 2, both flags, distance 16 | 21,297,655 | 1.93 | 55.96M | 31,774 | **4,377,606** |

- **Correctness:** all three print the same bit counts.
- **Stall cycles fall by 65%** (12.40M to 4.38M), as hoped: `bit_shifter` and `ntbl_bitcnt` no longer wait.
- **Cycles only fall by 10%,** because instructions grow by 28% (+12.3M). At distance 16, each call pays a 16-push prologue: in `bit_shifter`, about 4 instructions per push, so 64 per call, 75,000 times per function. Each kernel iteration also computes the condition 16 ahead. These loops run only about 30 iterations per call, so the prologue is a large share.
- **Mispredicts** rise from 12k to 32k. Traced later: mostly a gem5 bug, not the compiler (section 7.2).
- The distance was not tuned. A shorter distance should cut the prologue cost. That is a follow-up, not part of this layer.

---

## 7. Distance sweep, and a per-call mispredict that was a simulator bug

### 7.1 Distance sweep on bitcount

bitcount small, rebuilt with both flags at distances 1 to 16, run in parallel in gem5 (`swpspec/c2/dist/`). All outputs match vanilla, with `earlyDispatchOutcomeChanged` = 0. These are the first runs, on the gem5 build of the time (`fixC`); section 7.3 has the numbers after the gem5 fix.

| Distance | Cycles | x vanilla | Instructions | Mispredicts | pb stall cycles |
|---|---:|---:|---:|---:|---:|
| v3 (reference) | 23.73M | 2.15 | 43.66M | 12,241 | 12.40M |
| 1 | 19.87M | 1.80 | 47.04M | 87,848 | 6.44M |
| 2 | 20.51M | 1.85 | 49.66M | 87,761 | 4.23M |
| 3 | 20.10M | 1.82 | 50.11M | 50,230 | 4.42M |
| 4 | 20.73M | 1.87 | 50.57M | 87,706 | 4.53M |
| 6 | 21.04M | 1.90 | 51.47M | 87,988 | 4.53M |
| 8 | 21.19M | 1.92 | 52.37M | 88,049 | 4.53M |
| 12 | 21.11M | 1.91 | 54.16M | 50,474 | 4.42M |
| 16 | 21.30M | 1.93 | 55.96M | 31,728 | 4.38M |

- **Stall cycles flatten from distance 2 on.** The remainder is presumably `bit_count`, which has no closed form, and `main`'s call.
- **Instructions grow by about 450k per step,** from the longer prologues.
- **Mispredicts did not follow the distance**, so they were traced.

### 7.2 The cause: gem5 applied a decode-updated `bmovs` twice

**Correction.** An earlier version of this section blamed the bits a previous call leaves queued: fetch would predict the loop's first `pb` from them before the queue's `bmovs` reached decode. That was inferred from the mispredict pattern and a fetch-to-decode-delay experiment, and it was **wrong**. A trace with debug prints on that one `pb` showed a different cause, in the simulator, not the compiler.

- **Where:** almost all extra mispredicts are one branch, `ntbl_bitcnt`'s pipelined loop `pb`, about once per call (74,999 of 75,000 calls at distance 4).
- **The simulator option involved:** with `pbtbDecodeUpdate`, a PC-relative `bmovs` or `bmovt` is applied to fetch's view of the branch registers at decode. Every `pb` fetched afterwards starts from the reset queue, as it should.
- **The bug:** gem5 applied the same `bmovs` again when it executed, to the younger instructions' snapshots and fetch's state. Applying the source again is harmless. Emptying the queue again is not: it also erases what `pb`s fetched in between recorded. A `pb` that consumes from an empty queue records a debt, so that the next push pays it off instead of being queued. With the debt erased, every later bit reaches its iteration one too late, and the loop's last iteration mispredicts.
- **When it shows:** only when `pb`s are fetched between the `bmovs`'s decode and its execute, while the queue is still empty: the first iterations of each entry, before the prologue's pushes have executed. That explains the once-per-call pattern, why the gap between the setup and the loop seemed to matter, and why a fetch-to-decode delay of 1 hid most of it.
- **Correctness:** unaffected. Every `pb` is checked against the real register when it dispatches, so every output was right and `earlyDispatchOutcomeChanged` stayed 0. Leftover queued bits remain harmless for correctness, as change 1 says: `init` empties the queue on the next entry.
- **The fix,** in gem5 (`8bc685160b` on `jv-pbtb`): at execute, skip a `bmovs` or `bmovt` that decode update already applied with the same address. A companion fix (`ac63d421ee`) restores the branch registers' fetch state on a squash from decode. A decode-time redirect of `pb`s predicted from state a later decode update changes (`8786891c80`, `pbtbDecodeRedirect`, off by default) turned out neutral (within 0.02%).

### 7.3 After the gem5 fix

Same binaries, gem5 `8786891c80` (fetch-to-decode delay 3, as before), all flags. All outputs match vanilla, with `earlyDispatchOutcomeChanged` = 0.

| Distance | Before the fix | x vanilla | Mispredicts | After the fix | x vanilla | Mispredicts |
|---|---:|---:|---:|---:|---:|---:|
| v3 | 23.73M | 2.15 | 12,241 | 23.73M | 2.15 | 12,241 |
| 1 | 19.87M | 1.80 | 87,848 | 19.87M | 1.80 | 87,848 |
| 2 | 20.51M | 1.85 | 87,761 | 20.51M | 1.85 | 87,761 |
| 3 | 20.10M | 1.82 | 50,230 | **19.38M** | **1.75** | 12,745 |
| 4 | 20.73M | 1.87 | 87,706 | **19.53M** | **1.77** | 12,713 |
| 8 | 21.19M | 1.92 | 88,049 | 20.46M | 1.85 | 38,043 |
| 16 | 21.30M | 1.93 | 31,728 | 21.23M | 1.92 | 27,970 |

- **At distances 3 and 4 the extra mispredicts are gone**: about 12,700, vanilla's level. Distance 4 goes from 1.87x to 1.77x vanilla.
- **At distances 1 and 2 nothing changed, and 8 and 16 improved only partly.** About 75,000 extra mispredicts remain there, again once per call, so a second mechanism with the same symptom is at work. It is not traced yet, and it is not known whether it is in gem5 or the compiler.
- **Distance 4 is now the default** for loops without a trip count (section 9).

---

## 8. Layer 5: checks inside the pass

Layers 1 to 4 test examples. This layer adds checks that run every time the pass advances a shift recurrence, in builds with assertions, such as ours. They are compiled out of release builds (`NDEBUG`).

### 8.1 The checks

All in `advanceInductionVar` and a new helper, `checkShiftAdvance`.

1. **The value emitted equals shifting once per iteration.** After emitting the advanced value, the pass reads back what it emitted and compares it with applying the recurrence's own shift `K` times, one at a time, using `APInt`.
   - The emitted value must be a constant or one shift of the base by a constant; anything else fails.
   - When the base is a non-constant value, the comparison runs on 7 sample values: 0, 1, all ones, the signed minimum and maximum, and two mixed patterns. When the base is a constant, which happens in a prologue that starts from a constant, it runs on that constant.
   - Because it reads back the emitted IR rather than recomputing the formula, it also catches a wrong opcode or amount in what is emitted, not only in the arithmetic.
2. **The emitted shift amount is below the width** (`shift out of range`). This existed in change 2; it is now mutation-tested.
3. **An affine induction variable has a materialized step.** A shift recurrence has no step, so it must never take the affine path.

To read back what was emitted, the shift path now builds its result in one place and returns it after the check. The values emitted are unchanged (8.3).

### 8.2 Does each check fire

Each mutation was applied to a copy of the pass, `llc` rebuilt, and `ns-branch-condition-pipelining-shift.ll` compiled at distance 4 and 16. The pass was restored from `swpspec/c2/l5/pass.l5.cpp` after each one (byte-identical, then rebuilt).

| Mutation | Distance 4 | Distance 16 | Check that fires |
|---|---|---|---|
| `ashr` past the width gives 0 instead of the sign | no error | fires | 1 |
| The total amount is `c + K` instead of `c * K` | fires | fires | 1 |
| Always emit `lshr`, whatever the recurrence | fires | fires | 1 |
| No clamp at `width - 1` for `ashr` | no error | fires | 2 |
| Shift recurrences take the affine path | fires | fires | 3 |

The two "no error" cases are expected: those mutations only matter once the total amount reaches the width, which this test first does at distance 16 (`ashr_by_4`, 4 x 16 = 64).

**A slip in the process, not in the pass:** the restore used `cp -p`, which kept the copy's older timestamp, so `ninja` did not rebuild and two tests failed on the last mutant's `llc`. Touching the file and rebuilding fixed it; all tests below are on the correct build.

### 8.3 Results with the checks active

| Run | Result |
|---|---|
| The 538 RISC-V tests that passed before (536 from layer 1, plus this change's 2) | all pass |
| 32 generated programs, rebuilt `off`, `on16`, `on4` | 96 builds, no check fires; byte-identical to the layer 4 binaries |
| 16 MiBench programs, rebuilt flags off, speculative only and both flags (speculative distance 16) | 48 builds, no check fires; all 345 ELF files byte-identical to layer 1's builds |

Byte identity, ignoring `.comment` and debug info, also shows the restructured shift path and the checks do not change what the compiler emits. The full RISC-V suite's failing tests were not rerun, by agreement: they fail on the unmodified fork.

### 8.4 What these checks do not prove

That the recurrence the pass recognized is the one the loop computes. That rests on the pattern match in `findInductionVar` (a header phi whose latch value is a constant shift of the phi itself) and on layers 2 to 4.

---

## 9. Follow-up: distance 4 by default for loops without a trip count

### 9.1 The change

A new option, `-riscv-ns-swp-speculative-distance`, default 4, sets the distance for loops pipelined without a trip count (`-riscv-ns-swp-speculative`). Counted loops keep `-riscv-ns-swp-distance`, default 16. The two are independent: each applies only to its own kind of loop.

Why 4: each entry of a loop without a trip count computes up to D conditions for iterations that may never run, and these loops often run only a few dozen iterations per entry (section 6.2). The sweep in 7.3 is best at 3 and 4, and computing more than a few iterations ahead buys nothing once the condition is ready in time.

### 9.2 Tests

- Every existing RUN line that turns on `-riscv-ns-swp-speculative` now also passes `-riscv-ns-swp-speculative-distance` with the same value as its `-riscv-ns-swp-distance`, so every existing check keeps its meaning.
- **New `DEF` RUN lines** check the defaults with no distance given:
  - `ns-branch-condition-pipelining-speculative.ll`: `while_square` queues 4 conditions and computes 4 ahead.
  - `ns-branch-condition-pipelining-shift.ll`: in one compile, the speculative loop `ashr_by_4` is 4 ahead, and the counted loop `counted_with_shift` is 16 ahead. So neither option leaks into the other.

### 9.3 Effect on code

Rebuilt with the default (no distance options), with layer 5's checks active, compared with the distance 16 builds of 8.3:

- **Generated programs:** all 32 are byte-identical to `on4`. They contain only loops without a trip count.
- **MiBench, speculative only:** only fft changes (`fftmisc.o` and `fft`).
- **MiBench, both flags:** only bitcount and fft change.
- **bitcount at the default is not byte-identical to the distance-4 build of 7.1,** as expected: that build passed `-riscv-ns-swp-distance=4`, which also moved its one pipelined counted loop (`bitstring`'s `while (--biz >= 0)` in `bitstrng.c`) to 4. Now that loop stays at 16 and the three loops without a trip count are 4 ahead (from the pass's debug output, each source file compiled on its own).

### 9.4 gem5

gem5 `8786891c80`, all flags, as in 7.3.

| Program | Build | Cycles | x vanilla | Instructions | Mispredicts | pb stall cycles |
|---|---|---:|---:|---:|---:|---:|
| bitcount small | vanilla | 11,060,583 | 1.00 | 37.45M | 13,050 | 29,124 |
| | both flags, default | 19,532,473 | **1.77** | 50.57M | 12,631 | 4.38M |
| fft small (`4 4096`) | vanilla | 52,240,157 | 1.00 | 51.06M | 2,245,071 | |
| | speculative only, default | 52,187,038 | 1.00 | 51.30M | 2,278,426 | |
| | both flags, default | 52,186,759 | 1.00 | 51.30M | 2,278,416 | |

- **Every output matches vanilla,** with `earlyDispatchOutcomeChanged` = 0. bitcount's counts are compared with the timings stripped; fft's full `RealOut`/`ImagOut` output is compared.
- **bitcount** is where distance 4 was in 7.3 (19.53M, 1.77x), with counted loops kept at 16.
- **fft** matches vanilla's cycles. fft's distance 16 build was only measured on the older gem5 (change 1's layer 4), so no same-simulator comparison with 16 is given here.

---

## 10. Not done

- **The remaining per-call mispredicts** at distances 1 and 2 (and partly 8 and 16), section 7.3. Not traced; not known whether gem5 or the compiler is responsible.
- **Which code-generation step splits `bit_shifter`'s merged condition** back into two branches (section 1).
- **`bit_count`** (`x &= x - 1`) has no closed form and still stalls.
