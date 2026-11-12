# Toggles in branch condition pipelining (`-riscv-ns-swp-toggles`)

Record of the third change to `RISCVBranchConditionPipelining`: a condition on a value flipped every iteration (`x = !x`) can be computed ahead, and a counted loop whose conditions read nothing else is pipelined in place. Branch `swp-toggles`, on top of `non-spec` at `ded096df5f78` (changes 1 and 2, recorded in `swp-speculative.md` and `swp-shift-recurrences.md`). Written 2026-11-11.

The change is verified through layer 3 of the earlier changes' five layers, plus an inspection of the MiBench ADPCM binaries. **Nothing was run in gem5**, so the gain in section 1 is the estimate from the gap analysis, not a result of this change. Section 9 lists what is left.

---

## 1. Why

`adpcm-gap.md` (in the verification repo) traces ADPCM's slowdown to five conditional branches in the per-sample loop. Four are made by the backend from `select`/`min`/`max`. The fifth is the `bufferstep` toggle, taken on every second sample:

```c
if (bufferstep) { ... } else { ... }
bufferstep = !bufferstep;
```

Its `bmovc` sits directly in front of its `pb`, so the `pb` holds dispatch for 2.6 (encode) and 3.9 (decode) cycles per sample, and the 2-bit fallback predictor is wrong on half of all samples.

The gap analysis measured a hand-edited copy of the compiler's assembly in which the toggle's condition is queued ahead (`tq`). Small inputs, 2-bit fallback, million cycles:

| Starting point | Encode | Decode |
|---|---|---|
| the compiler's code | 39.03 to 34.86 (1.86x to 1.66x vanilla) | 33.55 to 27.70 (2.94x to 2.43x) |
| the four selects branch-free | 22.79 to 22.78 (no gain) | 15.97 to 9.48 (1.40x to 0.83x) |

These are what-ifs. They are the reason for the change and the target its output is compared with in section 6.

### The IR before the pass

Both loops have a trip count (`len`) and one conditional branch besides the latch. The encoder's (`adpcm_coder`):

```llvm
for.body:
  %bufferstep.0128 = phi i1 [ %7, %if.end62 ], [ false, %for.body.preheader ]
  ...
  br i1 %bufferstep.0128, label %if.else57, label %if.then56
if.end62:
  %7 = xor i1 %bufferstep.0128, true
```

The decoder's is the same with the start `true`.

### Why the pass rejected it: a correction to the gap analysis

The gap analysis says the toggle is "neither an affine nor a shift recurrence", so the branch is rejected. For ADPCM that is not the reason.

- **`bufferstep` is an `i1`, and SCEV already sees it as affine**: `{false,+,true}<%for.body>` (`opt -passes='print<scalar-evolution>'`). Adding 1 to one bit flips it. `findInductionVar` accepted it.
- **The branch tests the phi itself**, so the slice of instructions computing the condition is empty, and `buildSlice` returned false for an empty slice ("a phi condition is rare enough not to handle"). That is what printed `cannot compute ahead`.
- A wider toggle (`i32 x ^= 1`) is not affine, so the pass could not have advanced that one. The new recurrence is needed for those.

---

## 2. What was changed

All in `RISCVBranchConditionPipelining.cpp`, 99 lines added and 24 removed.

- **New flag `-riscv-ns-swp-toggles`,** default off, independent of the other two flags.
- **`InductionVar`:** gains `ToggleMask` and `isToggle()`. `Step` is null for a toggle.
- **`findInductionVar`:** with the flag on, an integer header phi whose latch value is `xor` of the phi itself with a loop-invariant value `m` is a toggle. This is checked before the affine test, so that an `i1` toggle is recorded as a toggle and not as an affine recurrence of step 1.
- **`advanceInductionVar`:** K iterations ahead, a toggle is `x ^ m` for odd K and `x` itself for even K. Flipping the same bits twice restores them.
- **`buildSlice`:** an empty slice is accepted when the condition is a toggle's phi. Any other empty slice is rejected as before.
- **`cloneCondition`:** returns the mapped value of the branch's condition, where it returned the mapped last instruction of the slice. The two are the same value when the slice is not empty.
- **`materializeValues`:** skips expanding a step for toggles.
- **`analyzeLoop`:** in place for toggle-only loops (section 3).
- **A new statistic,** "loops with a trip count whose toggle conditions were pipelined in place". Such loops are not counted as loops without a trip count.

**Not recognized:** `x = c - x`, which also has period 2, and a mask that changes between iterations (`t ^= i`).

---

## 3. In place, although the loop has a trip count

A counted loop is normally split into a kernel, which stops D iterations early, and the original loop, which drains the last D. The kernel must stop early because it may not compute a condition for an iteration that does not exist: the condition may load.

A toggle has no such limit. Its value two iterations on is its value now, whether or not those iterations run, and computing it reads no memory. So when the flag is on and **every** pipelined branch of a counted loop

- reads at least one induction variable, all of them toggles, and
- has a slice with no load in which every instruction is safe to execute on any operands,

the loop is pipelined in place by `transformLoopInPlace`, the transform change 1 added for loops without a trip count.

- **No kernel copy, guard or drain loop.** Code grows by the prologue only, and loops shorter than the distance are pipelined too.
- **The latch is untouched.** It was already skipped while the plan was built, since the loop has a trip count. It still counts the loop and becomes a hardware loop.
- **Distance:** `-riscv-ns-swp-speculative-distance` (default 4), not `-riscv-ns-swp-distance`. The reasoning is change 2's section 9: each entry queues D conditions for iterations that may never run. The gap analysis found `tq` to take the same cycles at 2, 4, 8 and 16 ahead.
- **A loop that also has a condition on a load** keeps the kernel and the drain loop for all its branches, and the toggle is advanced in the prologue and the kernel like any other induction variable.

This matches `tq`, which also leaves the loop where it is. A kernel copy of the ADPCM loops would instead duplicate both loop bodies, including the four branches the backend makes later in each, each with a branch register of its own.

### Queue safety

As in change 1: an in-place loop leaves up to D conditions queued when it exits, for iterations that never ran. `init` empties the queue the next time the loop is entered, which holds only if the queue's `bmovs`/`bmovt` stay in the loop's preheader. Section 5 tests that.

### Known wart

The pass still skips a loop whose constant maximum trip count is at most `-riscv-ns-swp-distance` (16), before it knows the loop will be pipelined in place at distance 4. A toggle loop that provably runs 5 to 16 iterations is therefore not pipelined.

---

## 4. Layer 1 and 2: flag off, and IR tests

### Flag off

| Check | Result |
|---|---|
| `adpcm.c`, MiBench `current` flags, assembly before the change against after | byte-identical |
| `adpcm.o` from the rebuilt compiler against the verification repo's existing `current` build (compiler `ded096df5f78`) | byte-identical |
| The 13 non-spec tests that existed before | all pass |

Only ADPCM was compared with the flag off. The full RISC-V suite and the flag-off byte comparison of all of MiBench, which changes 1 and 2 did, were not run.

### The IR test

New file `llvm/test/CodeGen/RISCV/ns-branch-condition-pipelining-toggle.ll`. RUN lines:

- distance 4 for both kinds of loop, flag on (`CHECK`);
- in-place distance 3 (`D3`), for the odd case;
- no distance given (`DEF`): the default in-place distance is 4;
- with `-riscv-ns-swp-speculative` as well (`SPEC`);
- flag off (`OFF`).

| Function | Checked |
|---|---|
| `toggle_i1` (`bufferstep`'s shape) | no guard, kernel or drain; the preheader pushes the constants 0, 1, 0, 1; the loop pushes `zext(t)` and branches on the pop; the latch still exits on `i < n`. At distance 3: three pushes and `t.ahead = xor t, true` |
| `toggle_i32` | `t ^= 1` on an `i32`, tested as `t != 0`: the prologue pushes `t0`, `t0 ^ 1`, `t0`, `t0 ^ 1`, and the comparison is folded into the push |
| `toggle_mask` | `t ^= m` for an argument `m`, tested as `t & 4`: `xor 0, %m` for odd iterations; at distance 3 the loop computes `xor t, m` |
| `toggle_and_load` | with a second branch on a load: guard, kernel and drain; both branches get a queue; the drain loop resumes the toggle from the kernel and branches on it as before. Flag off: only the load's branch is pipelined |
| `varying_mask` | `t ^= i`: no queue |
| `no_trip_count` | a `do`/`while` on a loaded value: untouched without `-riscv-ns-swp-speculative`, pipelined in place on the toggle with it |

Flag off, `toggle_i1`, `toggle_i32` and `toggle_mask` get no queue.

### What did not go as planned

- **Two `-NOT` checks ran past their function**, one into the intrinsic declarations (the same slip as change 2's section 4) and one into the next function's guard. Both are now bounded.

### Not done

Mutation testing of the test, as in the earlier changes' "does the test catch mistakes".

---

## 5. Layer 3: the queue's setup stays in the preheader, and the loop stays a hardware loop

New file `llvm/test/CodeGen/RISCV/ns-branch-condition-pipelining-toggle-asm.ll`: full `llc` to assembly (`-mattr=+m`, flag on, default distance). The function `reentry_toggle` has a counted inner loop with a toggle, re-entered by an outer loop:

```c
for (j = 0; j < m; j++) { t = 0; for (i = 0; i < n; i++) { if (t) s += a[i]; t = !t; } }
```

**The test checks:**

- no queue setup before `%outer`, the inner loop's preheader;
- in `%outer`: the toggle's `li`, the queue's `bmovs`/`bmovt`, four `bmovc_beq` on `zero`, 1, `zero`, 1, then the inner loop's `bmovc_loop`;
- in `%latch`: the `xori` of the toggle and the hardware loop's `pb`;
- in `%inner`: no `bmovs` of the queue's register, then `andi`, `bmovc_beq` and the queue's `pb`.

This is the first combination of an in-place loop with a hardware-loop latch: loops without a trip count never had one.

**Not done:** the mutation of the hoisting pass that changes 1 and 2 used to show their tests catch a hoisted setup.

All 15 non-spec tests pass.

---

## 6. The ADPCM binaries

Built in the verification repo as config `swp-toggle`: `current` (`-O2 -DPBTB`) plus `-mllvm -riscv-ns-swp-toggles`. The config is added in memory by `experiments/swp-toggle/run.py`; the live config files are only read. The binaries were disassembled and compared with the existing `current` ones. They were not run.

The pass's debug output for `adpcm.c`:

```
Loop for.body:
  pipelining   br i1 %bufferstep.0128, label %if.else57, label %if.then56 16 iterations ahead
  every condition reads only toggles, pipelining in place 4 iterations ahead
```

and the same for the decoder's loop.

| | `current` | `swp-toggle` |
|---|---:|---:|
| `adpcm_coder`, instructions | 144 | 149 |
| `adpcm_decoder`, instructions | 124 | 128 |
| `.text`, bytes | 46,200 | 46,234 |

**The encoder's preheader and toggle**, from `telecomm-adpcm-encode-small`:

```
   12cc8:  li          t3, 0x0                # bufferstep = 0
   ...
   12cd4:  li          t0, 0x1
   ...
   12cea:  bmovs       b8, 0x12df8            # the toggle's pb
   12cee:  bmovt       b8, 0x12d14            # if.else57
   12cf2:  bmovc_bne   b8, zero, zero         # samples 0 to 3: fall through, take, fall through, take
   12cf6:  bmovc_bne   b8, t0, zero
   12cfa:  bmovc_bne   b8, zero, zero
   12cfe:  bmovc_bne   b8, t0, zero
   12d02:  addi        a5, a5, 0x1
   12d04:  bmovc_loop  b2, -0x1(a5)
   ...
   12d24:  xori        t3, t3, 0x1            # bufferstep = !bufferstep
   12d28:  pb          b2                     # the hardware loop
   ...
   12df0:  andi        a4, t3, 0x1
   12df4:  bmovc_bne   b8, a4, zero           # appends the bit for 4 samples later
   12df8:  pb          b8                     # consumes the bit queued 4 samples ago
```

- **The loop bodies are the ones `current` has.** Function by function, the sequence of instructions (operands ignored) differs only in the preheaders. Register names differ in places.
- **What the preheaders gain:** the queue's `bmovs`/`bmovt`, moved there from the function entry, and four pushes. The encoder also gains the `li t0, 1` the pushes use. The decoder's toggle starts at 1, so its pushes use the toggle's own register, and its queue starts with a fall-through into the block that reads a new byte.
- **The directions are right.** Encoder: `bufferstep` starts 0, the branch is taken to `if.else57` when it is not 0, and the queue starts not taken, taken. Decoder: starts 1, taken to `if.end` when it is 0, queue starts not taken, taken.
- **`t0` is not clobbered.** In the encoder it also holds `outputbuffer` inside the loop, whose value on entry is `undef` in the IR.

### Against the hand-written `tq`

Compared with `build/adpcm-gap/trees/tq/ns.s` in the verification repo, the loops have the same instruction sequence. Outside the loops there are two differences:

| | `tq` | this change |
|---|---|---|
| Prologue | `li a5, 43690` (`0xaaaa`) and one `bmovc_bits`, 16 conditions | four `bmovc_bne` (`bmovc_beq` in the decoder), plus `li t0, 1` in the encoder |
| Distance | 16 | 4 |
| Queue's `bmovs`/`bmovt` | at the function entry | in the preheader |

Each runs once per call (once per 1,000 samples). The gap analysis found the distance not to matter.

---

## 7. The rest of MiBench, compiled only

The 32 small workloads were built with config `swp-toggle` and compared with the existing `current` binaries.

- **27 build.** The other five (`lame`, `mad`, `typeset`, `ghostscript`, `jpeg` decode) have no `current` binary either.
  - `jpeg` decode stops at `Bad machine code: Using an undefined physical register` in `read_color_map`. Compiling `rdcolmap.c` by hand fails the same way with the flag off.
  - `lame`, `typeset` and `ghostscript` also stop on compiler errors (`Unhandled encodeInstruction length`, the same bad machine code in `TrieRead`, `Unknown conditional branch`), and `mad` on an autotools error. These were not rerun with the flag off.
- **No assertion of this pass fired** in any of the builds.
- **Only ADPCM's code changes.** In the 25 other binaries, every function has the same instruction sequence as in `current`, operands ignored. The one exception is not a code change: in `tiffmedian`'s `TIFFUnlinkDirectory`, an `addi a1, a1, 0x8` that forms a data address becomes `addi a1, a1, 0`, which the disassembler prints as `mv`. The binaries are not byte-identical, because the build path, which is three characters longer, is in each of them, and data addresses move with it.

So among the workloads that build, the flag only does something for ADPCM.

---

## 8. Expected effect, not measured

The loops this change emits are `tq`'s, so the cycle counts in section 1 are the expectation: about 11% fewer cycles on encode and 17% on decode from today's code, and 1.40x to 0.83x vanilla on decode once the four selects are branch-free. The decode gain depends on the PBTB early-dispatch fix (`all` is 12.09M without it and 9.48M with it).

---

## 9. Not done

- **gem5.** No run of any kind: neither output against vanilla with `earlyDispatchOutcomeChanged` = 0, nor cycles. This is the main gap.
- **Generated programs** with toggle loops (layer 4 of the earlier changes).
- **Checks inside the pass** (layer 5). Change 2 reads back each advanced shift and compares it with shifting once per iteration. There is no such check for a toggle.
- **Mutation testing** of the two new tests.
- **The full RISC-V suite,** and the flag-off byte comparison for MiBench programs other than ADPCM.
- **`x = c - x`** and the small-trip-count wart of section 3.
- **One `bmovc_bits`** in place of the prologue's constant pushes.

---

## 10. Files and how to reproduce

| | |
|---|---|
| Pass | `llvm/lib/Target/RISCV/RISCVBranchConditionPipelining.cpp` |
| IR test | `llvm/test/CodeGen/RISCV/ns-branch-condition-pipelining-toggle.ll` |
| Assembly test | `llvm/test/CodeGen/RISCV/ns-branch-condition-pipelining-toggle-asm.ll` |
| Verification config | `experiments/swp-toggle/run.py` in the verification repo |
| Scratch (IR dumps, disassembly, `fdiff.py`, `sweep.log`) | `scratch/toggle/` |

```sh
# Tests, from llvm/cmake-build-debug
./bin/llvm-lit ../test/CodeGen/RISCV/ns-*.ll

# What the pass decides for a file
clang-dev ... -mllvm -riscv-ns-swp-toggles -mllvm -debug-only=riscv-branch-condition-pipelining -c adpcm.c

# The ADPCM binaries, from the verification repo; `run` in place of `build` simulates them
python3 experiments/swp-toggle/run.py build --build-timeout 180 -c swp-toggle \
    -t adpcm-encode-small -t adpcm-decode-small

# Functions whose instructions differ between two binaries
python3 scratch/toggle/fdiff.py build/mibench/current/<test>/bench build/mibench/swp-toggle/<test>/bench
```
