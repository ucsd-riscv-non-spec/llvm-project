# Non-Spec backend loop metadata

The backend accepts the IR metadata from [the loop annotations proposal](loop-annotations-proposal.md).
Clang's proposed `riscv_ns_*` pragma syntax is not implemented by this patch.
IR producers can attach the properties directly to distinct, self-referencing
`!llvm.loop` IDs on latch branches. Properties apply only to that loop.

For example, request SWP at distance 8 while disabling both fission passes:

```llvm
!0 = distinct !{!0, !1, !2, !3}
!1 = !{!"llvm.loop.riscv.ns.swp.distance", i32 8}
!2 = !{!"llvm.loop.riscv.ns.fission.ir.enable", i1 false}
!3 = !{!"llvm.loop.riscv.ns.fission.machine.enable", i1 false}
```

Every key below has the prefix `llvm.loop.riscv.ns.`. Emit only explicitly
supplied options; an absent boolean inherits its default and differs from false.

| Suffix | Encoding | Default |
|---|---|---|
| `swp.enable` | `i1` | enabled |
| `fission.ir.enable` | `i1` | enabled |
| `fission.machine.enable` | `i1` | enabled |
| `force` | `i1` | false |
| `swp.distance` | `i32`, 1–62 | `-riscv-ns-swp-distance` (16) |
| `swp.speculative_distance` | `i32`, 1–62 | `-riscv-ns-swp-speculative-distance` (4) |
| `swp.speculative` | `i1` | `-riscv-ns-swp-speculative` (false) |
| `swp.shift_recurrences` | `i1` | `-riscv-ns-swp-shift-recurrences` (false) |
| `swp.toggles` | `i1` | `-riscv-ns-swp-toggles` (false) |
| `swp.max_slice_size` | `i32`, 1–2147483647 | `-riscv-ns-swp-max-slice-size` (16) |
| `swp.max_loop_size` | `i32`, 1–2147483647 | `-riscv-ns-swp-max-loop-size` (256) |
| `swp.max_prologue_size` | `i32`, 1–2147483647 | `-riscv-ns-swp-max-prologue-size` (512) |
| `fission.queue_depth` | `i32`, 1–64 | IR: 64; machine: `-loop-fission-queue-depth` (64) |

Metadata overrides matching command-line tuning values, including explicitly
supplied values. The three explicit disable flags remain compile-wide kill
switches: `-riscv-ns-disable-software-pipelining`,
`-riscv-ns-disable-ir-loop-fission`, and
`-disable-loop-fission-decomposed-branch`. A numeric option or force does not
turn on a disabled pass or an opt-in feature.

The legacy counted SWP distance of zero disables every SWP path for loops that
inherit it. A positive local counted distance overrides that zero. Likewise,
a positive local fission depth overrides a machine default of zero. Metadata
zero is an error. IR fission uses depth as a whole-loop cutoff and does not tile;
machine fission uses it as the maximum tile size, including partial tiles.

## Force and retained requirements

`!{!"llvm.loop.riscv.ns.force", i1 true}` bypasses size and profitability limits,
SWP's `optsize` exclusion and shared function branch budget, memory dependence
proofs, speculative safety checks, and semantic instruction exclusions. This
includes explicitly supplied local soft limits. Force may change program
behavior or introduce traps when those assumptions do not hold.

Force retains the machinery needed to emit valid code: preheaders and latches,
representable trip counts or future recurrences, valid value mapping, supported
control flow and queue push/pop pairing. For example, SWP cannot reconstruct
an arbitrary non-affine recurrence, IR fission cannot delete a conditional body
whose values escape it, and machine fission cannot move a slice needing an
unavailable physical register value into its compute loop. Hardware bounds,
queue lowering availability, `optnone`, explicit disables and opt-in feature
gates still apply. Force does not change register allocation.

Counted toggle-only loops can use in-place SWP without enabling speculative
SWP. Loops without a computable trip count require the speculative feature,
including under force. Distances remain upper bounds; ordinary prologue and
memory checks can shorten them. Forced plans bypass those reductions but still
respect the queue bound. Forced SWP plans contribute to the branch budget used
for subsequent ordinary plans. IR fission still performs at most one successful
transformation per function per invocation to keep its analyses valid.

## Validation, remarks and metadata lifetime

At optimization levels above `-O0`, the independent
`riscv-ns-loop-metadata` pass validates every recognized property on IR
terminators immediately before IR fission. Errors identify the key, expected
encoding, function and latch, with a source location when available. Wrong
arity, types, ranges and duplicate keys (including identical duplicates) are
compilation errors even with all passes disabled or on ineligible loops.
Unknown properties are accepted for forward compatibility. Each consumer also
validates the metadata it encounters.

Earlier optimizations may remove loops and their metadata; validation does not
run ahead of those optimizations to preserve diagnostics for removed metadata.
Backend support and validation at `-O0` are not guaranteed. Metadata does not
schedule missing passes or add queue lowering to GlobalISel.

SWP reports selected settings and their metadata/default origin with
`-pass-remarks-analysis=riscv-branch-condition-pipelining`, including force and
distance reductions. Use `-pass-remarks` with the same pass name for effective
transformation distances and `-pass-remarks-missed` for blocked requests.
These diagnostics work without assertions or debug information.

IR fission gives compute and consume loops fresh IDs, consumes requests and
retains `riscv.loop-fissioned`. Counted SWP gives kernel and drain fresh IDs with
`llvm.loop.riscv.ns.swp.applied` string roles `kernel` and `drain`. Only the drain
retains machine-fission enablement, depth and force, so it remains available to
machine fission. In-place SWP consumes requests and adds role `inplace`.
Unrelated properties are preserved. All passes skip completed queue consumers;
machine fission tracks generated loops without mutating their shared IR IDs.

Machine fission first uses `MachineLoop::getLoopID()`. Target-owned copies of
typed metadata properties and source block provenance preserve controls when
lowering splits a latch. An unassociated or ambiguous machine loop in an
annotated function produces an error instead of silently restoring defaults.
Known unannotated loops retain legacy settings. Runtime unrolling now preserves
the loop ID when replacing a remainder latch; preservation tests also cover
rotation, peeling, vectorization, distribution, ordinary optimization and LTO pipelines.

## Verification

Build an assertion-enabled compiler and run the regression tests:

```sh
ninja -C llvm/cmake-build-debug llc opt FileCheck
llvm/cmake-build-debug/bin/llvm-lit -v llvm/test/CodeGen/RISCV/ns-*.ll
llvm/cmake-build-debug/bin/llvm-lit -v llvm/test/Transforms/LoopUnroll/
```

The optional simulator check uses the existing verification repository's
`sim/se.py` and its Non-Spec gem5 build. It compiles the regression kernels with
annotations and with all three transformations disabled, then compares 84 output
hashes. It covers both fission passes, partial tiles, ordinary and forced SWP,
speculative loads with valid future addresses, shifts, toggles, short trips,
re-entry and nested loops. All output and build artifacts go to `--output`.

```sh
python3 ns-docs/verify-loop-metadata.py \
  --build-dir llvm/cmake-build-debug \
  --gem5 ~/gem5/pbtb/build/ALL/gem5.opt \
  --sim-script ~/drive1/verif/sim/se.py \
  --output /tmp/ns-loop-metadata
```
