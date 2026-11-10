# Shift recurrences in branch condition pipelining (`-riscv-ns-swp-shift-recurrences`)

Record of the second change to `RISCVBranchConditionPipelining`: conditions on a value shifted by the same constant every iteration (`x >>= 1`) can be computed ahead, alongside affine induction variables. Branch `swp-speculative`, on top of change 1 (`dbcdc6c8096e`, `-riscv-ns-swp-speculative`, recorded in `swp-speculative.md`). Started 2026-11-10.

The change is verified in the same layers as change 1. Layers 1 to 4 are done, layer 5 is not (section 8). Section 7 records a distance sweep and a prediction problem with leftover queue bits that affects change 1 too.

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
- **Mispredicts** rise from 12k to 32k. Not investigated.
- The distance was not tuned. A shorter distance should cut the prologue cost. That is a follow-up, not part of this layer.

---

## 7. Distance sweep, and leftover bits used for prediction

### 7.1 Distance sweep on bitcount

bitcount small, rebuilt with both flags at distances 1 to 16, run in parallel in gem5 (`swpspec/c2/dist/`). All outputs match vanilla, with `earlyDispatchOutcomeChanged` = 0.

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

### 7.2 The cause: fetch predicts from the previous entry's leftover bits

Traced with the instrumented gem5 (`gem5.opt.pbcause`):

- **Where:** almost all extra mispredicts are one branch, `ntbl_bitcnt`'s pipelined loop `pb`. It mispredicts on 74,999 of 75,000 calls at distance 4, and on 37,500 at distance 3.
- **What fetch saw:** each time, the register held this `pb`'s own source with the queue type. So fetch predicted from bits that looked valid.

**Mechanism (inferred from the numbers, then confirmed by the delay experiment below):**

- The core fetches 6 instructions per cycle, with a fetch-to-decode delay of 3. Decode update applies the queue's `bmovs` (which empties the queue) to fetch's state only when it reaches decode.
- If the loop's first `pb` is fetched before that, fetch still sees the previous call's register for this function. That means the same source, the same type, and the bits queued for iterations past the previous loop's exit, which say "exit".
- The first iteration of the new call usually continues, so it mispredicts.

**The setup-to-`pb` gap decides it:**

| Function | Gap | Effect |
|---|---:|---|
| `ntbl_bitcnt`, distance 4 | 18 instructions | every call |
| `ntbl_bitcnt`, distance 3 | 16 instructions | half the calls |
| `bit_shifter` | 28 instructions | not affected |

**This corrects change 1's claim** that leftover bits are harmless. They are harmless for correctness: `init` empties the queue before the next entry's `pb` executes, and every output is right. They are not harmless for prediction.

### 7.3 Confirmation: fetch-to-decode delay 1

`fetchToDecodeDelay` set to 1 in a scratch copy of the config (`swpspec/c2/f2d1/sim`; only that line differs; the baseline config is unchanged). All outputs match vanilla, with `earlyDispatchOutcomeChanged` = 0. Vanilla moves from 11.06M to 11.03M cycles.

| Build | Cycles | x vanilla (same delay) | Mispredicts | pb stall cycles |
|---|---:|---:|---:|---:|
| v3 | 23.31M | 2.11 | 12,942 | 12.11M |
| distance 1 | **18.39M** | **1.67** | 13,027 | 6.71M |
| distance 2 | 19.70M | 1.79 | **37,925** | 4.38M |
| distance 3 | 19.30M | 1.75 | 12,755 | 4.00M |
| distance 4 | 19.45M | 1.76 | 13,026 | 4.08M |
| distance 8 | 19.90M | 1.80 | 12,928 | 4.00M |
| distance 16 | 20.80M | 1.89 | 12,975 | 4.00M |

- **The extra mispredicts disappear** at distances 1, 3, 4, 8 and 16; what remains is `bit_count`'s ~9k real ones.
- **Shorter distance is then simply better.** Distance 1 is fastest, 21% below v3. Distance 4 is within 6% of it and safer for slower conditions. 4 remains a reasonable default.
- **Distance 2 still mispredicts on a third of the calls** (`ntbl_bitcnt`, setup 12 instructions before the loop, more than one fetch cycle). The likely reason: fetch can run up to `fetchQueueSize` (70) instructions ahead of decode while decode is blocked, which happens constantly here because of dispatch stalls. So no fixed gap is safe, and the delay is not a fix.

### 7.4 Possible fixes, not built

1. **gem5:** when a queue's `bmovs` reaches decode, re-check the younger `pb`s already predicted from that register, and redirect fetch if they used stale state. This covers any gap or fetch-queue depth. It is close to the "early redirect" item in gem5 docs 06/07.
2. **Compiler:** also empty the queue on loop exit (`bmovs`/`bmovt` in the exit block, 2 instructions per exit). The next entry's first `pb` then finds an empty queue and falls back to the predictor, which is allowed for a known source.
3. **Compiler:** move the queue's setup earlier, staying within the preheader's loop. This helps only partly, given the distance-2 result.

---

## 8. Not done

- **Layer 5** (assertions for change 2) is not done. The range assertion in `advanceInductionVar` exists but was not mutation-tested, and the full rebuild with checks active was not rerun.
- **The fixes in 7.4.**
- **Choosing a default distance.** It is still 16; the data suggests 4.
- **Which code-generation step splits `bit_shifter`'s merged condition** back into two branches (section 1).
