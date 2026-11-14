# Non-Spec optimization guide

What each Non-Spec compiler flag does, when to change it, and what to watch for. It covers every `riscv-ns-*` option in `llvm/lib/Target/RISCV`, plus the two loop fission options that do not have that prefix. Written against `302081b0c246` (branch `call-target-hoisting`), 2026-11-12.

The other files in `ns-docs/` hold the design and measurements behind the newer options. This guide links to them and does not repeat them. Every example output below came from the `llc-dev` build at that commit (section 1.4).

---

## 1. Before you start

### 1.1 Quick reference

| Flag | Default | Section |
|---|---|---|
| `-riscv-ns-disable-software-pipelining` | pipelining on | 2.1 |
| `-riscv-ns-swp-distance=N` | 16 (max 62) | 2.2 |
| `-riscv-ns-swp-speculative` | off | 2.3 |
| `-riscv-ns-swp-speculative-distance=N` | 4 (max 62) | 2.4 |
| `-riscv-ns-swp-shift-recurrences` | off | 2.5 |
| `-riscv-ns-swp-toggles` | off | 2.6 |
| `-riscv-ns-swp-max-slice-size=N` | 16 | 2.7 |
| `-riscv-ns-swp-max-loop-size=N` | 256 | 2.7 |
| `-riscv-ns-swp-max-prologue-size=N` | 512 | 2.7 |
| `-riscv-ns-swp-max-function-branches=N` | 32 | 2.7 |
| `-riscv-ns-disable-ir-loop-fission` | IR fission on | 3.1 |
| `-disable-loop-fission-decomposed-branch` | machine fission on | 3.2 |
| `-loop-fission-queue-depth=N` | 64 (max 64) | 3.3 |
| `-riscv-ns-disable-hardware-loops` | hardware loops on | 4 |
| `-riscv-ns-disable-split-exit-jumps` | splitting on | 5.1 |
| `-riscv-ns-disable-branch-setup-hoisting` | hoisting on | 5.2 |
| `-riscv-ns-disable-cross-bb-hoisting` | cross-block hoisting on | 5.3 |
| `-riscv-ns-hoist-max-freq-ratio=N` | 2 (0 = no limit) | 5.4 |
| `-riscv-ns-disable-hoist-freq-from-best` | new rule on | 5.5 |
| `-riscv-ns-disable-call-target-hoisting` | call-target hoisting on | 5.6 |
| `-riscv-ns-hoist-call-target-max-freq-ratio=N` | 1 (0 = no limit) | 5.7 |
| `-riscv-ns-disable-branch-reg-hints` | hints on | 5.8 |

Everything is on by default except the three `-riscv-ns-swp-*` extensions (`speculative`, `shift-recurrences`, `toggles`), which are opt-in.

### 1.2 Passing the flags

- All of them are hidden `cl::opt`s, so `llc --help-hidden` lists them.
- With `llc`, pass them directly. With `clang`, pass each one through `-mllvm`:
  `clang -O2 -mllvm -riscv-ns-swp-speculative -mllvm -riscv-ns-swp-speculative-distance=3 ...`
- A flag supplies defaults for the whole compile. Clang also accepts
  [`#pragma clang loop riscv_ns_*` hints](loop-annotations.md), lowered to
  per-loop IR metadata, for independent pass controls,
  distances, feature gates, size limits, fission depth and force. Explicit
  pass-disable flags remain compile-wide kill switches.
- **At `-O0`,** IR loop fission and branch condition pipelining are not added to the pipeline. Hardware loops, machine fission, exit-jump splitting and hoisting still run.
- **With GlobalISel** (`-global-isel`), IR loop fission and branch condition pipelining do nothing. Only SelectionDAG selects the queue intrinsics.

### 1.3 The passes, in pipeline order

| | Pass (`-debug-only` name) | File | Its flags |
|---|---|---|---|
| IR | Non-Spec metadata validation (`riscv-ns-loop-metadata`) | `RISCVNSLoopOptions.cpp` | [Per-loop metadata](loop-annotations.md) |
| IR | IR loop fission (`riscv-ir-loop-fission`) | `RISCVIRLoopFission.cpp` | 3.1 |
| IR | Branch condition pipelining (`riscv-branch-condition-pipelining`) | `RISCVBranchConditionPipelining.cpp` | 2.x |
| IR | Generic `HardwareLoops`, which asks `RISCVTTIImpl::isHardwareLoopProfitable` | `RISCVTargetTransformInfo.cpp` | 4 |
| Before RA | Machine loop fission (`riscv-loop-fission-decomposed-branch`) | `RISCVLoopFission.cpp` | 3.2, 3.3 |
| After RA | Pseudo expansion (`riscv-expand-pseudo`) | `RISCVExpandPseudoInsts.cpp` | 5.1 |
| After RA | Branch setup hoisting and B-register hints (`riscv-branch-setup-hoisting`), then the B-register allocator | `RISCVBranchSetupHoisting.cpp` | 5.2 to 5.8 |

The order matters in two places:

- **An earlier pass takes loops away from a later one.** Pipelining skips a loop that IR fission split, because the split loop contains queue intrinsics. Machine fission skips loops that IR fission marked (`riscv.loop-fissioned`) and rejects a pipelined kernel. It can still split the drain loop that pipelining leaves behind (3.2).
- **Machine fission only works on hardware loops,** so it does nothing with `-riscv-ns-disable-hardware-loops`.

### 1.4 Seeing what a flag did

| Tool | What it shows |
|---|---|
| `-pass-remarks=riscv-branch-condition-pipelining` | Each pipelined loop and its distance. Without `-g`, the location prints as `<unknown>:0:0`. |
| `-pass-remarks-missed=riscv-branch-condition-pipelining` | Branch-budget rejections and reasons annotated requests were blocked. |
| `-pass-remarks-analysis=riscv-branch-condition-pipelining` | Annotated settings, their origin, force and distance reductions. |
| `-debug-only=<name>` | Why each loop or setup was taken or skipped. The names are in 1.3. Needs an assertion build such as `llc-dev`. |
| `-stats` | The counters below. Also needs an assertion build. |
| `-stop-after=<name>` | The IR or MIR right after one pass. |

| Pass | What `-stats` prints |
|---|---|
| pipelining | `Number of loops whose conditions were pipelined`, `Number of branches given a condition queue`, `Number of loops without a trip count whose conditions were pipelined`, `Number of loops with a trip count whose toggle conditions were pipelined in place` |
| machine fission | `Number of loops split into compute/consume pairs` |
| pseudo expansion | `Number of jumps after a conditional PB moved to their own block` |
| hoisting | `Number of BMOV instructions able to be hoisted`, `Number of BMOVS hoisted less far to keep enough B-registers free`, `Number of branches given a preferred B-register`, `Number of hoist points rejected for running too often compared with a point the setup was already hoisted to` |

IR fission has no counter. With `-debug-only`, it prints `fission: transformed <function> trip count <n>`.

**Static instruction counts are misleading here.** Most of these flags move instructions rather than add or remove them. For example, with `-riscv-ns-disable-branch-setup-hoisting`, `clamp_below` (below) still has 28 instructions, but two setups go back into the loop body. Compare committed instructions and cycles in gem5. Before you credit a flag with a change in cycles, rule out two things (`call-target-hoisting.md` 7.6 to 7.8, `setup-placement.md` 6):

- gem5's fetch buffer is 16 bytes. Moving a hot loop by a few bytes changed cycles by up to 27% with the same committed instructions. Compare `simInsts` and `fetch.cacheLines` first.
- `sha`'s cycles vary with memory-dependence prediction, so they can't be used.

### 1.5 The examples

Each kernel was compiled with `clang --target=riscv64-unknown-linux-gnu -O2 -fno-unroll-loops -S -emit-llvm`, then with `llc -mtriple=riscv64 -mattr=+m -O2` plus the flag being discussed. The listings below are trimmed to the relevant lines. The kernel used most often:

```c
// A counted loop whose branch reads a load.
void clamp_below(const int *restrict a, int *restrict out, long n, int t) {
  for (long i = 0; i < n; i++)
    if (a[i] < t)
      out[i] = t;
}
```

---

## 2. Branch condition pipelining

A `pb` takes its condition from a FIFO in its B-register. Each `bmovc_*` appends a bit, and each `pb` consumes one. When the condition is computed just before the branch, the front end reaches the `pb` before the bit exists. This pass computes the condition D iterations ahead, so the bit is normally already queued.

- **A loop with a trip count** gets a guard (`tc <= D` runs the original loop only), then a prologue that queues the first D conditions, then a kernel that runs `tc - D` iterations and pushes the condition of iteration `i + D` while it pops iteration `i`. The original loop then runs the last D iterations ("drain"). Both the kernel and the drain become hardware loops.
- **A loop without a trip count** (2.3), **or one whose conditions read only toggles** (2.6), is rewritten in place. There is no guard, kernel copy or drain. Conditions left in the queue when the loop exits are discarded by the reset the next time the loop is entered.

The header comment of `RISCVBranchConditionPipelining.cpp` has the full design. A loop qualifies only if all of these hold:

- It is innermost, in loop-simplify form, and exits only at its latch.
- Each pipelined branch runs exactly once per iteration, so its block dominates the latch.
- The condition is computed from loop invariants, induction variables, simple loads and side-effect-free instructions.
- The loop contains no call except to pure intrinsics, no `alloca`, no EH pad, no address-taken block and no lifetime marker.
- No store in the loop can write what the condition loads during the D iterations it is computed ahead. Otherwise that branch's distance is shortened, or the branch is dropped.
- It has a computable trip count, unless 2.3 or 2.6 applies.

The queue's setups (`bmovs`/`bmovt`) stay where the pipeliner puts them: the hoisting pass never moves a queue's setups.

`clamp_below` at the defaults:

```
Loop for.body:
  pipelining   br i1 %cmp1, label %if.then, label %for.inc 16 iterations ahead
```

```asm
	li	a4, 16
	bmovc_bgeu	b1, a4, a2        # guard: tc <= 16 goes to the original loop
	pb	b1
.LBB0_2:                          # %swp.ph
	bmovs	b5, .Lns_q_0              # the queue, reset on every entry
	bmovt	b5, .LBB0_3
	lw	a4, 0(a0)                     # prologue: the conditions of iterations 0 to 15
	bmovc_bge	b5, a4, a3
	...                               # 14 more load/push pairs
	lw	a4, 60(a0)
	bmovc_bge	b5, a4, a3
	...
.LBB0_4:                          # %for.body.swp, the kernel
	lw	a4, 0(a5)                     # a5 = &a[i + 16]
	bmovc_bge	b5, a4, a3            # pushes iteration i + 16
	pb	b5                            # pops iteration i
```

- In this kernel, each iteration has the same number of instructions as the original loop: the push takes the place of the `bmovc` that came before the `pb`.
- The cost is the guard, the 32-instruction prologue on every entry, and the drain. The function grows from 28 to 94 instructions.

### 2.1 `-riscv-ns-disable-software-pipelining`

- **Default:** off, so pipelining runs (above `-O0`).
- **With the flag,** the pass does nothing. Every loop branch computes its condition right before its `pb`, and none of the other `-riscv-ns-swp-*` flags has any effect.
- **Use it** as the A/B baseline, to rule the pass out when chasing a crash or miscompile in a pipelined loop, or for code size (`clamp_below`: 94 down to 28 instructions).
- **Side effect:** loops that pipelining would have rewritten become available to machine fission (3.2).

### 2.2 `-riscv-ns-swp-distance=N`

- **Default:** 16. Values above 62 are cut to 62, because the FIFO holds 63 conditions, the distance plus the one pushed just before the pop.
- **Counted loops use this distance.** Loops pipelined in place use 2.4;
  counted toggle analysis can also cap that distance by the counted distance.
  A value of zero disables every SWP path on loops that inherit it. A positive
  local `llvm.loop.riscv.ns.swp.distance` can override that zero.
- **A larger N** gives the condition more time to arrive, but:
  - the prologue computes N conditions on every entry to the loop (32 instructions for `clamp_below` at 16);
  - an entry with `tc <= N` runs only the original loop and pays for the guard;
  - a loop whose maximum trip count is known to be at most N is not pipelined (`never runs more than N iterations`).
- **A smaller N** suits short loops that are entered often. More of them qualify, and each entry costs less.
- **The distance can end up shorter than N:** `-riscv-ns-swp-max-prologue-size` (2.7) and the store check above can each shorten it per branch.
- **Measured:** no sweep of this option for counted loops is recorded in `ns-docs`. 16 has not been tuned. The only sweep (2.4) was for loops without a trip count.
- **Example:** `-riscv-ns-swp-distance=4` on `clamp_below` prints `4 iterations ahead`. `=100` prints `62 iterations ahead`.

### 2.3 `-riscv-ns-swp-speculative`

- **Default:** off.
- **With the flag,** a loop without a computable trip count (typically a `while`) is also pipelined, provided every instruction of its conditions is safe to run for an iteration that never happens. That rules out loads and any division that could trap. The cloned instructions drop flags such as `nsw`.
- The loop is rewritten in place (section 2). Its latch condition, which in a `while` is the loop's own test, can be pipelined too.
- Its distance comes from 2.4.
- **Example:**

  ```c
  long first_square(long a, long n) {
    long i = 0;
    while (i * i + a * i < n)
      i++;
    return i;
  }
  ```

  ```
  default:                 trip count is not computable
  -riscv-ns-swp-speculative:
                           trip count is not computable, pipelining speculatively
                           pipelining   br i1 %cmp, ... 4 iterations ahead
  ```

- **Measured** (`swp-speculative.md` 5c):
  - On 192 generated while-loops, 36% fewer cycles than with the flag off and within 8% of vanilla, for 1.6% more instructions. All outputs matched vanilla.
  - In MiBench, only fft's `NumberOfBitsNeeded` changes on its own. It showed no speedup, because it runs rarely and only for a few iterations.
- **It does not help** a loop whose condition loads, such as `while (*p)`: loads are always rejected.
- For loops that shift a value each iteration, combine it with 2.5.

### 2.4 `-riscv-ns-swp-speculative-distance=N`

- **Default:** 4. Values above 62 are cut to 62.
- **This is the distance for every loop pipelined in place:** loops without a trip count (2.3) and toggle loops (2.6). It also sets the toggle loops' distance when `-riscv-ns-swp-speculative` is not given.
- **Why 4:** every entry computes up to N conditions for iterations that may never run, and these loops often run only a few dozen iterations.
- **Measured** (`swp-shift-recurrences.md` 7.3, 9). On bitcount small, with 2.3 and 2.5 on and the fixed gem5:

  | N | 1 | 2 | 3 | 4 | 8 | 16 |
  |---|---|---|---|---|---|---|
  | Cycles, × vanilla | 1.80 | 1.85 | **1.75** | 1.77 | 1.85 | 1.92 |
  | Mispredicts | 87,848 | 87,761 | 12,745 | 12,713 | 38,043 | 27,970 |

- **Avoid 1 and 2:** about 75,000 extra mispredicts, one per call, remain at those distances, and their cause has not been traced.

### 2.5 `-riscv-ns-swp-shift-recurrences`

- **Default:** off.
- **With the flag,** `x = x >> c` or `x = x << c` (`lshr`, `ashr` or `shl` by a constant) counts as an induction variable: in iteration j, x is `x0` shifted by `j × c`.
- **Most of these loops have no trip count** (`while (x) { ...; x >>= 1; }`), so for them the flag needs 2.3 as well. In a counted loop, a shift uses 2.2's distance.
- **Example,** bitcount's `bit_shifter`:

  ```c
  for (i = n = 0; x && (i < (sizeof(long) * 8)); ++i, x >>= 1)
    n += (int)(x & 1L);
  ```

  | Flags | Debug output |
  |---|---|
  | none, or only shift | `trip count is not computable` |
  | only speculative | `cannot compute ahead: br i1 %1, ...` |
  | both | `pipelining br i1 %1, ... 4 iterations ahead` |

- **Measured** on bitcount small, with both flags at default distances (`swp-shift-recurrences.md` 6.2, 9.4): cycles fall from 2.15× to 1.77× vanilla, and stall cycles behind a `pb` from 12.40M to 4.38M. Instructions rise from 43.66M to 50.57M.
- **Not handled:** `x &= x - 1` (bitcount's `bit_count`) has no closed form and still stalls.

### 2.6 `-riscv-ns-swp-toggles`

- **Default:** off.
- **With the flag,** a value flipped every iteration (`x = !x`, or `x ^= m` with `m` invariant) counts as an induction variable.
- **How the loop is rewritten:**
  - If every pipelined condition of the loop reads only toggles, with no loads, the loop is pipelined in place even when it has a trip count, at 2.4's distance. Its latch stays an ordinary counted exit.
  - Otherwise, the toggle is just another induction variable in a counted loop, at 2.2's distance.
- **Example,** ADPCM's `bufferstep` pattern:

  ```c
  for (int i = 0; i < n; i++) {
    ...
    if (step) { ... } else { ... }
    step = !step;
  }
  ```

  ```
  default:  cannot compute ahead:   br i1 %step.09, label %if.else, label %if.then
  toggles:  pipelining   br i1 %step.09, ... 16 iterations ahead
            every condition reads only toggles, pipelining in place 4 iterations ahead
  ```

- **Evidence** (`swp-toggles.md` 6 to 9):
  - Only ADPCM's code changes among the 27 MiBench workloads that build.
  - Its loops match a hand-edited version that the gap analysis measured at about 11% (encode) and 17% (decode) fewer cycles.
  - **This flag has not been run in gem5.** Treat it as unverified.

### 2.7 Size limits

| Flag | Default | What it limits | When it is hit | Debug output |
|---|---|---|---|---|
| `-riscv-ns-swp-max-slice-size` | 16 | IR instructions in the loop that compute one branch's condition | That branch is not pipelined | `cannot compute ahead: <br>` |
| `-riscv-ns-swp-max-loop-size` | 256 | IR instructions in the loop, for every loop, including those pipelined in place | The loop is not pipelined | `loop is too large to duplicate` |
| `-riscv-ns-swp-max-prologue-size` | 512 | The loop's prologue, in IR instructions. Each branch costs `distance × (slice + 1)`, and branches are served in block order | Later branches get a shorter distance | `pipelining ... K iterations ahead`, with K below the distance |
| `-riscv-ns-swp-max-function-branches` | 32 | An estimate of the B-registers the function needs: one per branch, return and call, plus the branches each pipelined loop adds | The loop is not pipelined | missed remark `not pipelining branch conditions: the function could run out of branch registers` |

Examples on `clamp_below`: `max-slice-size=1` gives `cannot compute ahead`, `max-loop-size=4` gives `loop is too large to duplicate`, `max-prologue-size=10` gives `2 iterations ahead`, and `max-function-branches=8` gives the missed remark.

**Be careful raising `-riscv-ns-swp-max-function-branches`.** There are 32 B-registers. The B-register allocator runs after frame finalization and cannot spill. When a function uses more B-registers than can be assigned, the hoisting pass places the most frequently run branches first and moves the later branches' `bmovs` less far. Watch `Number of BMOVS hoisted less far to keep enough B-registers free` in `-stats`.

---

## 3. Loop fission

Two passes split a loop with a data-dependent branch into two loops:

- a **compute loop** that only pushes each iteration's condition into a queue;
- a **consume loop** that runs the body and pops one condition per iteration.

Unlike pipelining, all the conditions are computed before the body runs, so the queue has to hold the whole trip, or a tile of it.

### 3.1 `-riscv-ns-disable-ir-loop-fission`

- **Default:** off, so IR fission runs (above `-O0`).
- **It takes a loop when all of these hold:**
  - It is innermost, its preheader ends in an unconditional branch to the header, and it has a single exit, from the latch.
  - Its trip count is a constant from 1 to 64. The 64 is fixed in the code, not a flag.
  - Apart from the latch, it has exactly one conditional branch, in the header, testing an integer `icmp` no wider than XLEN.
  - The branch's true side is a single block that only jumps to the join and whose values are not used outside it.
  - The condition depends only on simple loads and side-effect-free instructions.
  - Every store in the loop is in that block, there is at least one, and alias analysis proves that no load in the loop aliases any store.
- **Only the first such loop** in each function is split.
- **Example:**

  ```c
  #define N 32
  void fission32(int *restrict a, int *restrict out, int t) {
    for (int i = 0; i < N; i++)
      if (a[i] < t)
        out[i] = a[i] * 3;
  }
  ```

  ```asm
  .LBB0_1:                    # %for.body.fission, the compute loop
  	lw	a4, 0(a3)
  	bmovc_bge	b5, a4, a2      # push
  	addi	a3, a3, 4
  	pb	b0                      # hardware loop, 32 iterations
  ...
  .LBB0_4:                    # %for.body, the consume loop
  	pb	b5                      # pop
  ```

- **With the flag,** the same loop is pipelined 16 ahead instead.
- **Where the two differ:** pipelining rejects a loop whose maximum trip count is at most its distance (2.2), so a constant loop of 16 iterations or fewer is left to fission.
- **Measured:** nothing recorded in `ns-docs`. Compare builds with and without the flag.

### 3.2 `-disable-loop-fission-decomposed-branch`

Note there is no `riscv-ns-` prefix.

- **Default:** off, so machine fission runs, at every optimization level.
- **It takes a loop when all of these hold:**
  - It is an innermost hardware loop, and its trip count is a constant loaded with a single `li` (`ADDI x0, imm`), so from 1 to 2047. A larger constant needs a `lui`, and the loop is rejected with `trip count is not a positive constant`.
  - It has exactly one data branch and a single exit, from the latch.
  - The branch's condition has no dependence that blocks the split.
  - It is not a loop IR fission marked, and the function is not `_start` or `_exit`.
- **Tiling:** a trip count above the queue depth (3.3) is split into tiles. The first tile takes `trip % depth` iterations, or the full depth if that is 0, and every later tile takes `depth`.
- **At the defaults, it mostly sees what pipelining leaves behind.** With 200 iterations of the 3.1 loop:
  - at the defaults, pipelining takes the loop. Machine fission skips the marked
    kernel, but can split the drain loop, whose trip count is the constant 16;
  - with `-riscv-ns-disable-software-pipelining`, it splits the whole loop into a tile of 8 and three of 64.
- **It does nothing with `-riscv-ns-disable-hardware-loops`.**

### 3.3 `-loop-fission-queue-depth=N`

- **Default:** 64. Values above 64 are cut to 64, and 0 disables machine fission
  on loops that inherit it. A positive local depth overrides that zero.
- **This flag is the tile size of 3.2 only.** Without metadata, IR fission's
  limit is 64. Per-loop `llvm.loop.riscv.ns.fission.queue_depth` overrides the
  whole-loop cutoff in IR fission and the maximum tile size in machine fission.
- **Lower it** to model a smaller queue in hardware. Each extra tile reruns the tile's setup. For example, `=16` turns the 200-iteration loop into one tile of 8 and twelve of 16.

---

## 4. `-riscv-ns-disable-hardware-loops`

- **Default:** off, so hardware loops are on.
- **What the default does:** `isHardwareLoopProfitable` returns true for every loop the generic `HardwareLoops` pass can convert. Profitability is not checked yet; see the `TODO` there. A converted loop gets `bmovc_loop bN, count-1` in its preheader and a single `pb bN` at its latch, and no compare.
- **With the flag,** loops keep their compare. On `clamp_below` with pipelining off:

  | | Preheader | Latch, every iteration |
  |---|---|---|
  | hardware loop | `bmovc_loop b2, -1(a2)` | `addi`, `addi`, `pb b2` |
  | with the flag | `slli`, `add` (the end pointer) | `addi`, `addi`, `bmovc_beq b2, a0, a2`, `pb b2` |

- **Cost in a loop with a call:** the loop's B-register has to survive the call, so it is a callee-saved one (B24 to B31), which the prologue saves with `bpush` and the epilogue restores with `bpop`. In `relax` (5.1), that is `bpush b24, 0(sp)` with a 48-byte slot.
- **Interplay:** the kernel and drain loops from pipelining become hardware loops, and machine fission (3.2) needs hardware loops.

---

## 5. Branch setup placement

Every `pb` needs a `bmovs` (the source: the `pb`'s own address) and a `bmovt` (the target), and a conditional one also needs a `bmovc`. `RISCVExpandPseudo` puts these setups right before their `pb`. `RISCVBranchSetupHoisting` then moves them up the dominator tree, so the target is known early and loop-invariant setups leave their loops. After that, it gives each branch a preferred B-register, and the B-register allocator runs.

These rules always apply and no flag changes them:

- A setup stops at a call, inline asm, an indirect branch, or anything it depends on.
- A setup is never left in a loop it was not in, but it can pass through one to a block before it.
- A queue's setups from pipelining never move.
- When a function uses more B-registers than can be assigned, each `bmovs` moves only as far as keeps the number live at once within that. The most frequently run branches are placed first.

The examples in this section use `-riscv-ns-disable-software-pipelining` to keep them short. `relax` and `straight` contain calls, so pipelining would leave them alone anyway.

### 5.1 `-riscv-ns-disable-split-exit-jumps`

- **Default:** off, so splitting is on. This one is in `RISCVExpandPseudo`.
- **What the default does:** when a block ends in a conditional `pb` followed by a jump, the jump and its setup move into a new block after the conditional, so the setup runs only when the jump does.
- **With the flag,** the setup stays in the conditional's block, and the hoisting pass moves it above the conditional `pb`. It then runs every time that block runs. In a loop's last block, if the loop has a call that keeps the setup in the loop, that means every iteration.
- **Example:**

  ```c
  void enqueue(int);
  void relax(int *dist, const int *w, int n, int d) {
    for (int i = 0; i < n; i++) {
      int nd = d + w[i];
      if (nd < dist[i]) { dist[i] = nd; enqueue(i); }
    }
  }
  ```

  ```asm
  .LBB0_2:                    # %if.end, the latch
  	addiw	s1, s1, 1
  	addi	s0, s0, 4
  	addi	s3, s3, 4
  	pb	b24                     # hardware loop
  # %bb.6:                    # the exit jump, in its own block
  	bmovs	b3, .Lns_j_1            # with the flag, these two move up into
  	bmovt	b3, .LBB0_5             # the latch and run every iteration
  	pb	b3
  ```

- **Measured** (`setup-placement.md` 6): committed instructions fell in dijkstra (-5.18%), stringsearch (-3.32%), qsort (-2.92%) and bitcount (-2.08%). Cycles moved by about 1% or less.
- **Don't use it with `-verify-machineinstrs`.** The verifier fails with `Non-terminator instruction after the first terminator`, because between pseudo expansion and hoisting the setup sits after the conditional `pb`. That was also the behavior before splitting was added.

### 5.2 `-riscv-ns-disable-branch-setup-hoisting`

- **Default:** off, so hoisting runs.
- **With the flag,** no setup moves. Each one runs right before its `pb`, every time the `pb` runs, so the setups of a loop's branches run on every iteration. The register hints (5.8) still apply unless you turn them off too.
- **Example,** `clamp_below`: by default, the `bmovs`/`bmovt` of all seven branches are at the function entry. With the flag, each stays by its `pb`, so the setups of the loop body's branch run on every iteration.
- **Use it** as an A/B baseline, or to rule hoisting out when chasing a bug.

### 5.3 `-riscv-ns-disable-cross-bb-hoisting`

- **Default:** off.
- **With the flag,** setups move only within their own block, up to the first hazard. In `clamp_below`, the body branch's `bmovs` moves above the load but stays in the loop.
- **Use it** to measure how much of hoisting's effect comes from leaving the block.

### 5.4 `-riscv-ns-hoist-max-freq-ratio=N`

- **Default:** 2. 0 means no limit.
- **The rule:** as a setup walks up the dominator tree, any dominator that runs more than N times as often as the coolest point already reached is passed over. The walk does not stop there, so a cooler block further up can still take the setup.
- **Why:** a setup placed above a rarely taken test runs whether or not the rare path is taken.
- **Example:**

  ```c
  void work(long);
  long rare(long *a, long n, int mode) {
    if (__builtin_expect_with_probability(mode == 3, 1, 0.0625))
      for (long i = 0; i < n; i++)
        if (a[i] > 0)
          a[i] = 0;
    work(0);
    return n;
  }
  ```

  At the default, the loop's 12 setups stay in the block before the loop, which runs on 1 call in 16. With `=0`, all 12 move to the function entry and run on every call.

- **Lower it** (1) and setups never move into a hotter block. **Raise it** (or use 0) and setups arrive earlier, but more of them run for branches that are never reached.
- Call targets have their own limit (5.7).
- **Measured:** no sweep is recorded.

### 5.5 `-riscv-ns-disable-hoist-freq-from-best`

- **Default:** off, so the new rule is on.
- **The default rule** compares each candidate with the coolest point reached so far. **With the flag,** it uses the old rule and compares with the setup's own block. Inside a loop body, that block's frequency includes the trip count, so the old rule lets a rarely entered loop's setups climb to the function entry.
- **Example,** `rare` above: with the flag, the setups of the loop body's branch (`b7`, `b8`) move from the block before the loop to the function entry.
- **Measured** (`setup-placement.md` 12): committed instructions fell or stayed flat in every program, for example stringsearch -1.43%, basicmath -0.41% and patricia -0.33%. Cycles stayed within noise.
- **Use it** for A/B only. Its counter is `Number of hoist points rejected for running too often ...`.

### 5.6 `-riscv-ns-disable-call-target-hoisting`

- **Default:** off, so call-target hoisting is on.
- **What the default does:** for a direct call, tail call or long jump, the target is `auipc rX, f` followed by `bmovt b, rX`, and the two move as one. They have to, because the `auipc`'s `R_RISCV_CALL_PLT` relocation also patches the instruction after it.
- **With the flag,** the pair stays right before the call.
- **Where a pair stops:**
  - at an instruction that reads or writes `rX`, which keeps it below `sd ra` and below an earlier call;
  - at a terminator, if `rX` is live into a successor;
  - at a `frame-setup` instruction;
  - at a call, as for every setup.
- **Example:** in `int straight(int x, int y) { ...; return g(c); }`, the pair moves to the top of the function. In `relax`, it moves above the store and the argument move in its own block. It then stops below the call's own `bmovs`, which a hazard keeps in that block (`[Stop] Hazard on a CFG path` in the debug output). Raising the limit in 5.7 changes nothing there.
- **Measured** at the default limit, over 37 MiBench workloads (`call-target-hoisting.md` 7.4, 7.11): committed instructions unchanged and cycles -0.04% in total. Predictions made with no target known fell by 85%, but that did not turn into fewer cycles.

### 5.7 `-riscv-ns-hoist-call-target-max-freq-ratio=N`

- **Default:** 1. 0 means no limit.
- **This is 5.4's limit for call-target pairs.** At 1, a pair never moves into a block that runs more often than the coolest point it has reached.
- **2 was the first version, and it was a regression overall:** +1.46% geomean cycles, not counting sha (`call-target-hoisting.md` 7.4 to 7.6, 7.10).
  - dijkstra ran 5.4% to 5.9% more instructions and 3.4% more cycles, because `enqueue`'s pair was hoisted above the test that guards the call.
  - susan smoothing took 27% more cycles, because an 8-byte code shift pushed a hot loop across a fetch-buffer boundary.
  - qsort large ran 2.38% faster.
- **Per program:** `-mllvm -riscv-ns-hoist-call-target-max-freq-ratio=2` gets qsort large's 2.38% back. Using it everywhere brings the regressions back too.

### 5.8 `-riscv-ns-disable-branch-reg-hints`

- **Default:** off, so hints are on.
- **What the default does:**
  - Each branch's B-register gets a preferred physical register. The i-th branch in layout order prefers the i-th caller-saved B-register.
  - The hints are colored so that together they form a valid assignment, and the allocator follows them.
  - The callee-saved B-registers are left for values that live across calls.
- **Why:** fetch predicts a `pb` from the B-register whose source address matches it. If two branches share a register, each `bmovs` of one wipes the other's prediction, which happens on every iteration when both run in a loop.
- **With the flag,** the allocator takes the lowest free register.
  - With hoisting on, setups live long, so branches rarely share anyway.
  - With hoisting off, `clamp_below`'s seven branches get `b0 b1 b0 b0 b1 b1 b0` without hints, and `b0` to `b6` with them.
- Its counter is `Number of branches given a preferred B-register`.

---

## 6. Recipes

### 6.1 A loop still waits on its `pb`

Run with `-pass-remarks=riscv-branch-condition-pipelining -debug-only=riscv-branch-condition-pipelining` and look up the message:

| Message | What to try |
|---|---|
| `trip count is not computable` | `-riscv-ns-swp-speculative`. This only works if the condition does not load. |
| `cannot compute ahead`, with a constant shift of the tested value | `-riscv-ns-swp-shift-recurrences`. For a `while` loop, add `-riscv-ns-swp-speculative`. |
| `cannot compute ahead`, with a value flipped every iteration | `-riscv-ns-swp-toggles` |
| `cannot compute ahead`, otherwise | The condition uses something that cannot be cloned, or it is longer than `-riscv-ns-swp-max-slice-size`. |
| `a store could clobber what the condition of ... reads` | Prove the pointers don't alias (`restrict`), or accept that the loop writes what it reads. |
| `loop is too large to duplicate` | `-riscv-ns-swp-max-loop-size` |
| `never runs more than N iterations` | A smaller `-riscv-ns-swp-distance`. A constant trip count of 64 or fewer may already be handled by IR fission. |
| `loop body cannot be pipelined` | A call, `alloca`, EH pad, or similar, and no flag helps. Or IR fission already split the loop: check `-debug-only=riscv-ir-loop-fission`. |
| `not an innermost loop exiting at its latch` | No flag helps. |
| missed remark about branch registers | `-riscv-ns-swp-max-function-branches` (see the warning in 2.7) |

### 6.2 Trying the opt-in extensions

```
-mllvm -riscv-ns-swp-speculative -mllvm -riscv-ns-swp-shift-recurrences -mllvm -riscv-ns-swp-toggles
```

So far, the first two have been measured together on bitcount (1.77× vanilla, from 2.15×), the first alone on fft and on generated loops, and the third only compiled.

### 6.3 A/B testing a flag in gem5

- In the verification repo (`~/drive1/verif`), add a config that passes the `-mllvm` flag, rather than keeping a second compiler build.
- After rebuilding `clang-dev`, rebuild MiBench with `--force`. Its make-based workloads are not stamped with the compiler, so they otherwise report that they are up to date.
- Compare committed instructions first, then cycles (1.4).

### 6.4 Narrowing down a miscompile

Turn the passes off one at a time, roughly from the most to the least intrusive:

1. `-riscv-ns-disable-software-pipelining`
2. `-riscv-ns-disable-ir-loop-fission`
3. `-disable-loop-fission-decomposed-branch`
4. `-riscv-ns-disable-branch-setup-hoisting`
5. `-riscv-ns-disable-hardware-loops`

Add `-verify-machineinstrs`, except with `-riscv-ns-disable-split-exit-jumps` (5.1). The verifier can also fail for reasons unrelated to these passes (`swp-speculative.md` 6).
