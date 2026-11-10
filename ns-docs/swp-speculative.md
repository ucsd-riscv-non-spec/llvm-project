# Speculative branch condition pipelining (`-riscv-ns-swp-speculative`)

Record of the change to `RISCVBranchConditionPipelining` that lets it pipeline loops without a computable trip count, and of how it was verified. Branch `swp-speculative`, on top of `archit-debug` (`b553d41cb062`: `non-spec` `5dbf685` + Mitchell's `acb2007` + his uncommitted paired-condition patch). Written 2026-11-10.

Verification is done in layers. This document covers layers 1 to 5.

---

## 1. Why

On MiBench `bitcount`, built with `acb2007` and the paired-condition patch, dispatch is blocked behind a `pb` for about 12.4M of 23.7M cycles. 88% of those stalls are in three while-loops (`bit_shifter`, `bit_count`, `ntbl_bitcnt`). The pass rejected all of them with "trip count is not computable".

The pass needs a trip count only to stop computing conditions for iterations that do not exist. Its header comment says the kernel runs `tc - D` iterations and the original loop finishes the last `D` "so that it never evaluates the condition of an iteration that does not exist". The danger is the condition's inputs, mainly loads past the end of an array. Bits queued for iterations that never run are already harmless for correctness: `init` empties the queue the next time the loop is entered.

**A note on prediction** (`swp-shift-recurrences.md`, section 7): a per-call mispredict found on bitcount was at first blamed on these leftover bits, with fetch predicting the next entry's first `pb` from them. That was wrong; the cause was a gem5 bug (a decode-updated `bmovs` applied a second time at execute, erasing the record of `pb`s fetched in between). Leftover bits are harmless for correctness. Some per-call mispredicts remain at distances 1 and 2 and are not traced yet, so whether leftover bits play any part in prediction there is still open.

So when every instruction of a condition is safe to compute for any iteration, the trip count is not needed. This is the change described here.

It is the first of two separate changes:

1. **This one:** drop the trip-count requirement for conditions that are safe to compute speculatively.
2. **Not done:** accept recurrences with an exact formula, such as `x >>= 1` (`x0 >> j`), alongside affine induction variables.

Change 1 alone fixes none of bitcount's three loops, because their recurrences are not affine. It fixes loops such as fft's `NumberOfBitsNeeded` (section 4.4).

## 2. What the pass did before

- `analyzeLoop` rejected any loop where `SE->getExitCount` could not compute a trip count.
- `transformLoop` used the trip count three times:
  1. **Guard:** `tc > D`, otherwise skip to the original loop.
  2. **Kernel stop:** the kernel's own exit condition is replaced by a counter that stops after `tc - D` iterations.
  3. **Drain:** the original loop runs the last `D` iterations, with resume phis.
- Because the kernel exit is a counter, the latch branch was never pipelined (`if (BB == P.Latch) continue;`). In a while-loop, the latch branch is the loop condition.
- The slice may contain loads, any binary operator including division, and instructions with `nsw`/`nuw`/`exact`/`inbounds`. That is safe only because each clone is evaluated for an iteration that really runs.

## 3. What was changed

All in `llvm/lib/Target/RISCV/RISCVBranchConditionPipelining.cpp`, about 130 lines. Nothing changes unless the new flag is given.

- **New flag `-riscv-ns-swp-speculative`,** default off. New statistic `NumLoopsPipelinedSpeculatively`.
- **`LoopPlan::Speculative`.** `TripCount` is null in this mode.
- **`analyzeLoop`:** if there is no trip count and the flag is on, the loop is marked speculative instead of rejected. In that mode the latch branch is a candidate too.
- **`buildSlice`:** in speculative mode, every slice instruction must also pass `isSafeToSpeculativelyExecute` and must not be a load.
  - This rejects division by anything not known to be non-zero, `sdiv` that could be `INT_MIN / -1`, calls and memory accesses.
  - Loads are rejected outright for now. They could be allowed later when every advanced address is known to be valid.
- **`cloneCondition`:** in speculative mode, each clone drops `nsw`/`nuw`/`exact`/`disjoint`/`inbounds` and similar flags (`dropPoisonGeneratingAnnotations`).
  - An iteration that never runs may overflow where the original cannot.
  - For iterations that do run, the clone computes the same value.
- **`materializeValues`:** no trip count is expanded in speculative mode.
- **New `transformLoopInPlace`** for speculative loops:
  - The preheader gets `init` and the pushes for iterations `0..D-1`.
  - Each pipelined branch gets a push for `i + D` and a pop as its condition.
  - Old conditions are deleted only after every branch is rewritten, through tracking handles, because two branches may share one condition.
  - No block is added or removed: no kernel copy, guard, counter or drain.
- **Header comment** updated to describe the mode.

### Design change from the first plan

The first plan copied the loop as a kernel that exits directly to the original exit and deleted the original loop. That needs every use of the loop's values after the loop to be redirected, which needs LCSSA form, not guaranteed at this point in the codegen pipeline. Rewriting the loop in place avoids it, changes no control flow, and leaves the dominator tree and loop info valid. This was agreed before implementing.

### Unchanged on purpose

- Single exit at the latch (`exitsOnlyAtLatch`).
- Each pipelined branch must run exactly once per iteration.
- No calls in the loop.
- Affine induction variables only.
- The register-estimate gate (`MaxFunctionBranches`). It still counts the kernel copy, which a speculative loop does not have, so it is conservative here.

### Queue safety argument

Up to `D` bits per queue are left after the loop exits. They are never read:

- The only `pb` bound to the queue is in the loop.
- `init`, the queue's `bmovs`/`bmovt`, is in the preheader, which runs every time the loop is entered.

A speculative queue has `D + 1` pushes, so the hoisting pass treats it as a queue (`IsQueue`) and does not move its setup. Layer 3 tests this.

---

## 4. Layer 1: nothing changes when the flag is off

### 4.1 Setup

- Our Release+assertions build in `/home/archit/vault/llvm-non-spec/build`, reconfigured with `LLVM_INCLUDE_TESTS=ON`, with `llvm-lit`, `FileCheck` and the other test tools built. That did not change the compiler itself.
- Before any change, the following were saved in `/home/archit/vault/claude-scratch/swpspec/`:
  - the original pass and test files, in `orig/`;
  - the unmodified `llc`, as `llc.orig`.

### 4.2 The non-spec tests: passed

The 8 tests for the fork's own passes all passed before and after the change:

- `ns-branch-condition-pipelining.ll`
- `ns-cond-queue.ll`
- `ns-hwloop-call.ll`
- `branch-setup-hoisting-bregs.mir`
- `branch-setup-hoisting-conditional.mir`
- `branch-setup-stale-target.mir`
- `ns-cond-queue-branch-folding.mir`
- `ns-cond-queue-expand.mir`

### 4.3 The full RISC-V test suite: same failures before and after

**Unexpected:** 1,684 of the 2,218 tests in `llvm/test/CodeGen/RISCV` fail on the fork **before any change**.

- The fork turns every ordinary branch into `bmovs`/`bmovt`/`bmovc`/`pb`, and those tests expect upstream RISC-V branches. One was checked by hand (`zext-with-load-is-free.ll`): a real output mismatch, not a missing tool.
- So "the suite passes" means nothing here. We agreed on a weaker check plus a stronger one.
- **Weaker check:** the set of failing tests must be identical before and after. **Result: identical, 1,684 and 1,684, same names.**
- **Stronger check:** byte-identical output, below.

### 4.4 Byte-identical output

**(a) Every `.ll` test input through the old and new `llc`: not completed, skipped.**

- 1,801 files for `riscv64` and `riscv32`, with the flag off and on, comparing output, errors and exit codes.
- The first run reported 1,813 differences with the flag off, far too many to be real.
  - Many inputs make `llc` print an error, and the error starts with the program name, `llc.orig` against `llc`.
  - Many crash with a stack dump containing addresses and the binary's path.
- A rerun with both binaries named `llc` and stack dumps removed was started, and you stopped it ("skip the ll checks, move on"). So this check was not completed.
- **Two things are not confirmed:**
  - that program names and stack dumps explain all of those differences;
  - that no real difference was mixed in.

**(b) MiBench rebuilt three ways: done.**

- **Variants:** the old compiler (`llvm-bins/acb2007-uncommitted`, the same source as before the change), the new compiler with the flag off, and the new one with it on.
- **Commands:** each program was rebuilt from its recorded build command in `~/vault/mibench/current/*/bench.cmd`, with Robert's paths replaced by ours.
- **Plain-clang programs:** only the 4 automotive ones (basicmath, bitcount, qsort, susan) have sources in our copy.
- **Make-based programs:** these carry their own sources; 14 are distinct. They were copied to scratch with our own build wrapper. Fixes needed along the way:
  - replay the recorded `make` command, since targets differ (`crc`, `sha`, `djpeg`);
  - keep subfolders (`jpeg-6a`);
  - put the wrapper on `PATH`, since some Makefiles call plain `gcc`.
- **Not rebuilt:**
  - `tiff2bw`: `make clean` removes parts the recorded command expects to exist.
  - `sphinx`: its configured Makefiles hard-code paths into Robert's folder.

  Both are limits of the rebuild script, not compiler problems.
- **Result:** 16 programs, 112 ELF files (objects and executables), compared with `.comment` removed, since it holds the compiler's version string.
  - **Flag off: identical**, after also removing debug info from ispell and fft. Their Makefiles use `-g`, and debug info records the build folder (`old/` against `off/`). All loaded sections (`.text`, `.rodata`, `.data` and the rest) were identical before that.
  - **Flag on: exactly one object file changes,** fft's `fftmisc.o`. Its `NumberOfBitsNeeded` has `for (i = 0; ; i++) if (PowerOfTwo & (1 << i)) return i;`. That loop has no trip count, its condition is an induction variable with a shift and an AND, and it is now pipelined 16 iterations ahead (4 since the speculative distance got its own default; `swp-shift-recurrences.md`, section 9). That is the kind of loop this mode is for.
  - It also exercises the poison rules: for iterations that never run, `1 << i` can go past 31, so the pushed value is meaningless, but that bit is never popped.
- **Not yet checked:** that fft's output is still correct. That is layer 4.

---

## 5. Layer 2: IR tests

### 5.1 The test

New file `llvm/test/CodeGen/RISCV/ns-branch-condition-pipelining-speculative.ll`, in the style of `ns-branch-condition-pipelining.ll`: `llc -stop-after=riscv-branch-condition-pipelining` with FileCheck. RUN lines:

- distance 4, flag on: the main checks;
- distance 1 and distance 62, the most a queue holds;
- distance 4, flag off.

Each function's output was read and checked against the design by hand before its checks were written.

| Function | Expected and checked |
|---|---|
| `while_square` (`while (i*i < n)`) | Exact output: `init` and 4 prologue pushes in the preheader; in the latch, push for `i + 4` and pop. No guard, kernel, counter or drain. Distance 1: 1 push, looking 1 ahead. Distance 62: 62 pushes. Flag off: untouched. |
| `flags_dropped` | Clones of `mul nsw`, `shl nuw nsw`, `or disjoint`, `lshr exact`, `add nsw` have no flags; the originals keep all of them. |
| `udiv_const` | Division by 7 is accepted. |
| `shared_condition` | A body branch and the latch share one condition: two queues, and the old condition is deleted once. |
| `reentry` | Inner while-loop in an outer loop: `init` is in the inner loop's preheader, so it runs on every entry. The outer loop is not pipelined. |
| `partial_branch` | A branch that does not run every iteration keeps its own condition; the header branch and the latch are pipelined. |
| `counted_loop_unchanged` | A loop with a trip count still takes the guard, kernel and drain path, with the flag on or off. |
| `reject_load` | No queue: loads are rejected. |
| `reject_udiv_var` | No queue: divisor may be 0. |
| `reject_sdiv_minus_one` | No queue: `INT_MIN / -1`. |
| `reject_call` | No queue: calls are rejected. |
| `reject_shift_recurrence` (`x >>= 1`) | No queue: not affine. That is change 2. |
| `reject_two_exits` | No queue: the loop does not exit only at its latch. |

The rejection reasons were also confirmed in the `-debug-only` output. The test itself checks only the IR, so it does not need a build with assertions.

### 5.2 What did not go as planned

- **`flags_dropped`, first version:** its condition `(i << 2) | 1 < n` had a computable trip count, so it took the normal path and tested nothing. It was rewritten on `i * i`.
- **Machine verifier, a pre-existing fork problem:** a planned RUN line with `-verify-machineinstrs` fails, with the unmodified `llc` too.
  - The error is "non-terminator instruction after the first terminator" in a function with a hardware loop (`flags_dropped`, first version).
  - The fork's expansion places the `bmovs`/`bmovt` of a fall-through jump after the loop's `pb`, which ends the block. The hoisting pass later moves them, so the final assembly is correct.
  - The functions this mode transforms passed the verifier.
  - The existing non-spec tests do not use the verifier either.
  - We decided to drop the RUN line. The fork's verifier failure is not investigated.
- **My own false alarm:** while checking that, a copy of the old `llc` had not been made, because the command was interrupted, so the "old compiler" run did not execute. Rerun with `llc.orig`: the same failure.

### 5.3 Does the test catch mistakes

I introduced three bugs on purpose, rebuilt `llc` each time, and ran the test:

| Mistake | Result |
|---|---|
| Clones keep their flags | Fails, line 29 (`%sq.swp0` became `mul nsw`) |
| Loads allowed in the slice | Fails, `reject_load` |
| Mode never taken | Fails, line 28 |

The first attempt reported all three as not caught. That was my detection: `lit -q` does not print `FAIL:` lines. The second attempt used lit's exit code. The pass was restored from a saved copy (byte-identical), rebuilt, and the test passes again.

**Final state:** the 8 non-spec tests and the new one, 9 of 9, pass.

---

## 5b. Layer 3: the hoisting pass keeps the queue's setup in place

### 5b.1 The property

A speculative loop leaves up to `D` conditions in its queue when it exits. Only the queue's setup, its `bmovs`/`bmovt` (`init`), discards them before the next entry.

`RISCVBranchSetupHoisting` runs after `RISCVExpandPseudo` has turned the queue pseudos into `bmovs`/`bmovt`/`bmovc`/`pb`. If it moved that setup out of an enclosing loop, it would run once, and the second entry would consume the first entry's leftover conditions: wrong branches.

### 5b.2 The test

New file `llvm/test/CodeGen/RISCV/ns-branch-condition-pipelining-speculative-asm.ll`, full `llc` to assembly with `-mattr=+m`. Two functions with a speculative inner loop inside an outer loop:

- `reentry`: limit `n + j`, which changes per outer iteration;
- `reentry_invariant`: limit `n`, invariant.

It checks:

- no queue setup (`.Lns_q_`) appears before the `%outer` label;
- the queue's `bmovs` is the first instruction after that label, followed by its `bmovt`;
- 4 prologue pushes go to the same register, captured by FileCheck;
- the inner loop has no `bmovs` of that register, and the `pb` uses it.

The other setups in these functions (the outer hardware loop, the exit jump, the return) are hoisted to the entry, so the test shows the pass is active and only the queue's setup stays.

**Why `-mattr=+m`:** without the M extension, the `mul` in the condition becomes a call to `__muldi3` inside the loop. That call is created after the pipelining pass, which only saw a `mul`. The result is still correct: the queue register lives across the call, so it gets callee-saved `b25` with `bpush`/`bpop`. Real builds have M. The existing non-speculative pass has the same interaction.

### 5b.3 What protects the setup, found from the debug output

Disabling what I assumed were the protections did not make the test fail, so I followed `-debug-only=riscv-branch-setup-hoisting` instead of guessing further. My first three explanations were wrong.

| I disabled | Hoisted? | What the debug output showed |
|---|---|---|
| `if (W->IsQueue) continue;` | No | The setup reached the scheduler, then: `[Stop] Hazard on a CFG path from BB#1 to BB#0` |
| plus the paired-condition rule that nothing else uses the register | No | the same stop |
| plus the check that skips physical registers with other users | No | Not applicable: the queue register is still virtual (`%0:pbr`) at this point |

**Two protections exist:**

1. **The explicit queue rule** in `hoistBranchSetups`: a register with more than one push keeps its setup where the pipeliner put it.
2. **The general conflict walk** in `isSafeToHoistTo`. Moving the `bmovs` from `%outer` to the entry crosses, through the outer loop's back edge, the inner loop with the queue's own pushes and `pb`. `hasRegisterDependency` counts them as conflicts.

   Note: `loopHasHazard` skips other `bmov`s on the same register, but `isSafeToHoistTo` does not, and the `pb` is a conflict in both.

### 5b.4 Showing the test catches the regression

Mutation: queue rule off, and `isHoistHazard` treats the pushes and `pb` of the setup's own register as never conflicting. This is the kind of change someone might make to hoist more. `bmovs`/`bmovt` on the same register still conflict, so they keep their order.

- **Result:** the queue's `bmovs`/`bmovt` move to the entry block, and the test fails at the "no queue setup before `%outer`" check in both functions (lines 15 and 57).
- **A broader first attempt crashed:** treating every same-register instruction as non-conflicting also let `bmovt` pass its own `bmovs`, and `llc` failed in live interval analysis ("Use not jointly dominated by defs"). That counted as caught, but by a crash, so it was narrowed.
- **Restored:** the hoisting pass was restored from `swpspec/l3/hoisting.orig.cpp`, with no diff against the branch. All 10 non-spec tests pass.

### 5b.5 Part B, not done

`hasPairedCondition` accepts `BMOVC_BITS` without checking its bit count (see the review of the uncommitted patch). Fixing it with its own test was planned for this layer. It was deprioritized because no pass emits `BMOVC_BITS` for the next few weeks.

---

## 5c. Layer 4: running the programs in gem5

The earlier layers check what the compiler emits. This one checks that the programs behave the same: every program is built vanilla and with the pass, run on the PBTB core, and its output compared against vanilla's.

### 5c.1 Setup

- **Simulator:** `gem5.opt.fixC` (the `updateCheckpoints` fix), with all three PBTB flags (early dispatch, keep branch type, decode update), base predictor model 2 and tage. That is the same setup as the bitcount analysis.
- **Variants:**

  | Variant | Compiler |
  |---|---|
  | `van` | host `/usr/bin/clang` (Ubuntu clang 14) with the vanilla MiBench flags, no `-DPBTB`. The reference. |
  | `off` | new compiler, flag off |
  | `on16` | flag on, distance 16 (the default at the time) |
  | `on4` | flag on, distance 4 |

- **Passing criteria for every run:** output identical to `van`, and `earlyDispatchOutcomeChanged` = 0, meaning early dispatch never concluded a wrong outcome.
- **Parallelism:** all 131 runs, 32 at a time.

### 5c.2 Generated while-loops

- **Generator:** `swpspec/l4/gen.py`, seed 20261110. 32 programs, 6 kernels each, 192 kernels in total.
- **Kernels:** each is a `noinline` while-loop whose condition is arithmetic on an affine counter, so it has no trip count. Families:

  | Family | Condition | Count |
  |---|---|---:|
  | square | `i*i + a*i < n` | 34 |
  | probe | `((i*m + s) & 2047) != t`, `m` odd; runs exactly `k` iterations | 30 |
  | xor | `(i ^ c) != t` | 30 |
  | udiv | `i/3 + i*i < n` | 28 |
  | bodybranch | a body branch on `(i*i + c*i) % 7 == r`, run every iteration, with a store in one arm | 32 |
  | nested | the while-loop re-entered by an outer loop in the same function, with a `volatile` count | 38 |

- **Trip counts:** each kernel runs at 0, 1, 2, 3, 4, 5, 15, 16, 17, 31, 32, 33, 100 and 1000, which covers `0`, `1`, `D-1`, `D` and `D+1` for both `D = 4` and `D = 16`.
- **Re-entry:** each kernel is also called 3 times in a row through all trip counts, so every entry starts with the previous entry's leftover bits.
- **Inputs** come from `volatile` globals, so nothing is folded at compile time. Each program prints one line per run, 93 lines in all.

**Getting the generator right took three tries**, all caught from the pass's debug output before running anything:

1. **`nested` was unrolled.** The outer loop's constant count let the compiler unroll it into separate while-loops, so there was no in-function re-entry. Fixed with a `volatile` count.
2. **`bodybranch` lost its branch.** The `if` was turned into a select. Fixed with a store in one arm.
3. **The `volatile` store blocked the pass.** It is not a "simple" store, so the pass rejected those loops (160 of 192 pipelined). Fixed with an ordinary global, printed at the end.

**After the fixes:** in both `on16` and `on4`, all 192 loops are pipelined speculatively, plus 29 of the 32 body branches (221 branches). The other 3 body branches were not examined; they were probably turned into selects.

**Results:**

- **Correctness:** all 128 runs (32 programs × 4 variants) give output identical to vanilla, with `earlyDispatchOutcomeChanged` = 0.
- **Performance**, a side result and not the purpose of this layer. Totals over the 32 programs:

| Variant | Cycles | Instructions | Mispredicts |
|---|---:|---:|---:|
| van | 17,342,925 | 20,646,548 | 555,574 |
| off | 29,291,121 | 23,980,962 | 561,415 |
| on16 | **18,725,407** | 24,367,654 | 539,431 |
| on4 | 18,829,363 | 23,933,058 | 563,267 |

On these loops, the speculative mode takes 36% fewer cycles than the flag-off build, within 8% of vanilla, for 1.6% more instructions. That is the effect the change was made for.

### 5c.3 fft small

- **Binaries:** vanilla `~/vault/mibench/vanilla/telecomm-fft-small-forward/bench`, and the layer 1 rebuilds with the flag off and on. The flag-on build differs only in `NumberOfBitsNeeded`.
- **Run:** `fft 4 4096`, forward, the small input.
- **Two problems in my run script before it worked:**
  1. The arguments were passed as one string (`--args "$*"`), so fft printed its usage message and exited after about 4,600 instructions. Fixed by passing separate words; the config takes `--args` as a list.
  2. The comparison flagged the config's banner line `Workload: <path>` as a difference. That line is now excluded.
- **Results:** all three outputs are identical (the full real and imaginary arrays), with `earlyDispatchOutcomeChanged` = 0.

| Variant | Cycles | Instructions | Mispredicts |
|---|---:|---:|---:|
| van | 52,240,157 | 51,063,051 | 2,245,071 |
| off | 52,112,358 | 51,296,967 | 2,256,985 |
| on | 52,682,563 | 51,297,006 | 2,306,558 |

The flag-on build takes 1.1% more cycles than flag off, with 39 more instructions. `NumberOfBitsNeeded` is called rarely and exits within a few iterations, so this run does not show a speedup. The difference is more likely code layout or noise than the loop itself; that is not investigated.

### 5c.4 The rest of MiBench

Of the 16 MiBench programs rebuilt in layer 1, only fft's code changes with the flag on. The other 15 are identical to the flag-off build apart from `.comment` and debug info, so running them would only repeat the flag-off results.

---

## 5d. Layer 5: checks inside the pass

Layers 1 to 4 test examples. This layer adds checks that run on every loop the pass transforms, in builds with assertions, such as ours.

### 5d.1 The checks

1. **In `cloneCondition`, speculative mode:** every clone, after its flags are dropped and its operands remapped, must not be a load and must pass `isSafeToSpeculativelyExecute`.
   - This checks the instruction as emitted, not only the original the slice rule looked at.
   - A future change that loosened the slice rule, or cloned something the rule never saw, would trip it.
2. **In `runOnFunction`:** after `transformLoopInPlace`, the function has the same number of blocks as before. Pipelining in place must not change control flow, because the other loops' plans rely on the dominator tree and loop info.
3. **In `runOnFunction`:** after the pass changes a function, `verifyFunction` must succeed, otherwise `report_fatal_error`. This applies to both modes.

All three are compiled out of release builds (`NDEBUG`).

### 5d.2 Results with the checks active

| Run | Result |
|---|---|
| 10 non-spec tests | all pass |
| Full RISC-V suite | the same 1,684 tests fail as before any change |
| 32 generated programs, rebuilt `on16` and `on4` | 64 builds, no check fires; byte-identical to the layer 4 binaries |
| 16 MiBench programs, rebuilt flag on | 16 builds, no check fires; all 112 ELF files byte-identical to layer 1's flag-on builds |

Byte identity, ignoring `.comment` and debug info, also shows the checks do not change what the compiler emits.

### 5d.3 Does the clone check fire

Mutation: the slice rule lets `udiv` through even when it is not safe to execute anywhere. Compiling the layer 2 test then stops at the new assertion in `cloneCondition`, on `@reject_udiv_var`, the function made to catch exactly that. The pass was restored from `swpspec/pass.layer5.cpp`, byte-identical, and both new tests pass again.

The other two checks were not mutation-tested. The block-count check would need a deliberately broken transform. `verifyFunction` is LLVM's own verifier.

### 5d.4 Not done

- **Alive2**, to check that dropping the flags refines the original, was optional. It is not installed here and cannot model the queue intrinsics, so it was not used.
- **What these checks do not prove:** that each queue has one `pb` and `init` on every entry, for every possible CFG. That rests on the existing structural rules (single exit at the latch, each branch run once per iteration, `init` in the dedicated preheader) and on layers 2 to 4.

---

## 6. Not done, and why

| Item | Why not |
|---|---|
| `.ll` corpus byte-identity (4.4a) | Stopped by you; differences from error text and crash dumps were not separated from real ones. |
| `tiff2bw` and `sphinx` rebuilds | Limits of the rebuild script. |
| Machine verifier RUN line | Fails on the unmodified fork; dropped by agreement. |
| The 3 body branches not pipelined in the generated programs | Not examined; probably turned into selects. |
| Why fft's flag-on build is 1.1% slower | Not investigated; likely layout or noise. |
| Closed-form recurrences (`x >>= 1`) | A separate change (change 2). |
| Exit merging for `bit_shifter` | Not yet triaged why it is missing. |
| Loads in speculative slices | Rejected for now. Would need proof that every advanced address is valid. |
| `BMOVC_BITS` bit-count check in `hasPairedCondition` (layer 3 part B) | Deprioritized; `BMOVC_BITS` is not planned for a few weeks. |
| Fork's machine-verifier failure and the `__muldi3` call created inside pipelined loops without M | Pre-existing, not investigated. |

## 7. Files

- Change: `llvm/lib/Target/RISCV/RISCVBranchConditionPipelining.cpp`; new tests `llvm/test/CodeGen/RISCV/ns-branch-condition-pipelining-speculative.ll` (layer 2) and `ns-branch-condition-pipelining-speculative-asm.ll` (layer 3).
- Layer 3 scratch: `swpspec/l3/`: original hoisting pass, debug logs, mutation logs.
- Scratch in `/home/archit/vault/claude-scratch/swpspec/`:
  - `orig/`: original pass and tests;
  - `llc.orig`: unmodified `llc`;
  - `pass.layer1.cpp`: the pass at the end of layer 1;
  - `lit-*`: suite results and failing-test lists;
  - `l1mib/`: MiBench rebuilds;
  - `l2/`: layer 2 outputs, including the mutation logs.
