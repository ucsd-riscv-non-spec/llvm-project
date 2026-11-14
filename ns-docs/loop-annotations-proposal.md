# Proposal: per-loop controls for Non-Spec branch transformations

**Status: backend metadata and Clang source hints implemented.**
See [the source and backend interfaces and verification instructions](loop-annotations.md).
Originally written
against `50c7eab35402`, 2026-11-13. Existing behavior and flags are described in
[the optimization guide](optimization-guide.md).

Add target-specific Clang loop hints that lower to `!llvm.loop` metadata. The
RISC-V backend reads that metadata to enable or disable branch condition
pipelining and either loop fission pass, and to override their tuning parameters
for one loop. Add a per-loop `force` flag to bypass soft restrictions and semantic
correctness checks. Keep the existing command-line options as defaults and
explicit disable flags as compile-wide kill switches.

## 1. Motivation and current constraints

A compile-wide distance or feature switch cannot express "pipeline this loop
eight iterations ahead, pipeline that while-loop three ahead, and fission the
third loop in batches of sixteen." It also makes experiments change unrelated
loops in the same translation unit.

There are three consumers, in this order:

| Pass | Current controls and constraints relevant to this proposal |
|---|---|
| [`RISCVIRLoopFission`](../llvm/lib/Target/RISCV/RISCVIRLoopFission.cpp) | Enabled by default; constant trip count of 1–64; transforms at most one loop per function per invocation. Its queue limit is fixed in the code. |
| [`RISCVBranchConditionPipelining`](../llvm/lib/Target/RISCV/RISCVBranchConditionPipelining.cpp) | Enabled by default; counted distance 16; in-place distance 4; speculative conditions, shift recurrences and toggles are opt-in. It has slice, loop, prologue and function branch-count limits. |
| [`RISCVLoopFissionDecomposedBranch`](../llvm/lib/Target/RISCV/RISCVLoopFission.cpp) | Enabled by default; requires a hardware loop and a positive constant trip count recognized from machine instructions; tiles to `-loop-fission-queue-depth`, default 64. |

IR fission runs before pipelining and hardware-loop formation. Machine fission
runs before register allocation. An earlier transformation can therefore take
away the loop a later pass would have used. Enabling one pass does not select it
exclusively; the interface needs independent enable/disable controls.

Ordinary annotations request an optimization under the backend's existing
checks. A forced transformation bypasses profitability limits and correctness
proofs, including dependence analysis and safe-speculation checks. The user
takes responsibility for the resulting program behavior. Structural requirements
needed to construct the transformation, hardware bounds and target/pipeline
availability still apply; section 4 defines that boundary.

## 2. Source interface

Extend `#pragma clang loop` with a `riscv_ns_` family. It applies to the immediately
following `for`, range-for, `while` or `do` loop, using Clang's existing loop-hint
attachment rules. It does not apply recursively to nested loops or to other
loops in the function. Numeric arguments must be integer constant expressions.

For example, select pipelining for one counted loop:

```c
#pragma clang loop riscv_ns_swp(enable) riscv_ns_swp_distance(8)
#pragma clang loop riscv_ns_ir_fission(disable) riscv_ns_machine_fission(disable)
for (long i = 0; i < n; ++i)
  if (a[i] < threshold)
    out[i] = threshold;
```

Select machine fission, including tiling, without IR fission or pipelining
changing the loop first:

```c
#pragma clang loop riscv_ns_swp(disable) riscv_ns_ir_fission(disable)
#pragma clang loop riscv_ns_machine_fission(enable) riscv_ns_fission_queue_depth(16)
for (int i = 0; i < 200; ++i)
  if (a[i] < threshold)
    out[i] = a[i] * 3;
```

Enable speculative pipelining and shift recurrences for just one while-loop:

```c
#pragma clang loop riscv_ns_swp(enable) riscv_ns_swp_speculative(enable)
#pragma clang loop riscv_ns_swp_shift_recurrences(enable) riscv_ns_swp_speculative_distance(3)
#pragma clang loop riscv_ns_ir_fission(disable) riscv_ns_machine_fission(disable)
while (x) {
  count += x & 1;
  x >>= 1;
}
```

Force pipelining when the compiler cannot prove that condition loads are
independent of stores, and bypass its size and profitability limits:

```c
#pragma clang loop riscv_ns_swp(enable) riscv_ns_swp_distance(8) riscv_ns_force(enable)
#pragma clang loop riscv_ns_ir_fission(disable) riscv_ns_machine_fission(disable)
for (long i = 0; i < n; ++i)
  if (a[i] < threshold)
    out[i] = threshold;
```

Here `force` allows the requested load motion without proving it safe. If an
alias or another dependence makes that motion change program behavior, the
generated code can produce different results.

For a baseline without either optimization, disable all three passes on the chosen
loop. Disabling SWP alone still permits fission.

### Options

All names in the first column have the source prefix `riscv_ns_`. All metadata
suffixes in the second column have the prefix `llvm.loop.riscv.ns.`.

| Source hint | Metadata suffix | Accepted argument | Default when absent |
|---|---|---|---|
| `swp` | `swp.enable` | `enable` / `disable` | Current pass enablement |
| `ir_fission` | `fission.ir.enable` | `enable` / `disable` | Current pass enablement |
| `machine_fission` | `fission.machine.enable` | `enable` / `disable` | Current pass enablement |
| `force` | `force` | `enable` / `disable` | Off |
| `swp_distance` | `swp.distance` | 1–62 | `-riscv-ns-swp-distance` (16) |
| `swp_speculative_distance` | `swp.speculative_distance` | 1–62 | `-riscv-ns-swp-speculative-distance` (4) |
| `swp_speculative` | `swp.speculative` | `enable` / `disable` | `-riscv-ns-swp-speculative` (off) |
| `swp_shift_recurrences` | `swp.shift_recurrences` | `enable` / `disable` | `-riscv-ns-swp-shift-recurrences` (off) |
| `swp_toggles` | `swp.toggles` | `enable` / `disable` | `-riscv-ns-swp-toggles` (off) |
| `swp_max_slice_size` | `swp.max_slice_size` | 1–2³¹−1 | `-riscv-ns-swp-max-slice-size` (16) |
| `swp_max_loop_size` | `swp.max_loop_size` | 1–2³¹−1 | `-riscv-ns-swp-max-loop-size` (256) |
| `swp_max_prologue_size` | `swp.max_prologue_size` | 1–2³¹−1 | `-riscv-ns-swp-max-prologue-size` (512) |
| `fission_queue_depth` | `fission.queue_depth` | 1–64 | IR: fixed 64; machine: `-loop-fission-queue-depth` (64) |

An omitted hint inherits the compile's setting. In particular, an omitted
boolean differs from an explicit `disable`. A numeric hint alone does not enable
a disabled transformation or an opt-in feature; `swp_speculative_distance(3)`
does not imply `swp_speculative(enable)`.

`force` is a loop-local flag shared by the three passes. It affects whichever
transformations are enabled for that loop; it does not enable a disabled pass or
an opt-in feature. To force just one transformation, use the independent pass
controls shown above. `force(disable)` retains the ordinary checks.

`fission_queue_depth` supplies a common per-loop limit to both fission passes.
For IR fission, it is the maximum whole-loop trip count: that pass still does
not tile. For machine fission, it is the maximum tile size. Thus a depth of 16
makes IR fission skip a 32-iteration loop; machine fission can later tile it if
pipelining has not taken it. Without metadata, the existing command-line depth
continues to affect machine fission only.

Keep `-riscv-ns-swp-max-function-branches` compile-wide. It controls an estimate
for the whole function, including transformations already planned for sibling
loops; it is not a per-loop budget. A forced SWP plan bypasses this estimate but
still contributes to accounting for later, unforced plans. Queue identities,
B-register assignment and hardware-loop formation remain backend decisions.

Do not reuse `pipeline`, `pipeline_initiation_interval` or `distribute`. Those
existing hints describe other transformations; condition lookahead distance is
not a machine scheduling initiation interval.

## 3. LLVM IR representation

Use ordinary, distinct, self-referencing loop IDs on latch branches. Emit only
explicitly supplied options. Encode booleans as `i1` and numeric parameters as
`i32`. This also lets other frontends and IR producers use the interface without
Clang changes.

```llvm
define void @annotated_loop(i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %next, %loop ]
  %next = add i64 %i, 1
  %again = icmp ult i64 %next, %n
  br i1 %again, label %loop, label %exit, !llvm.loop !0

exit:
  ret void
}

!0 = distinct !{!0, !1, !2, !3, !4, !5}
!1 = !{!"llvm.loop.riscv.ns.swp.enable", i1 true}
!2 = !{!"llvm.loop.riscv.ns.swp.distance", i32 8}
!3 = !{!"llvm.loop.riscv.ns.fission.ir.enable", i1 false}
!4 = !{!"llvm.loop.riscv.ns.fission.machine.enable", i1 false}
!5 = !{!"llvm.loop.riscv.ns.force", i1 true}
```

This example illustrates encoding only; it has no internal data branch for SWP
to pipeline. Keep the existing `riscv.loop-fissioned` marker internal. It records
a completed transformation and is not a user request.

The parser must check node arity, operand types, ranges and duplicate keys before
reading values. Clang diagnoses invalid arguments and duplicate hints as errors.
For hand-written IR and other IR producers, any malformed recognized option
present at the validation stage described below is a compilation error.
This applies to all recognized `llvm.loop.riscv.ns.*`
options, including shared fission depth and `force`. Duplicate recognized keys
are errors even when their values agree. Ignore unknown keys for forward
compatibility. Diagnose the offending key and expected form, with function and
loop identification and a source location when available. Do not assert,
silently substitute a default, or merely skip a transformation on invalid input.

In supported optimized RISC-V compilations, validate all recognized options
still present in the IR immediately before the Non-Spec transformations,
starting with IR fission, and before their enablement, eligibility or force gates.
Disabling a transformation does not hide malformed metadata. Validation must
not depend on a loop becoming a transformation candidate. Metadata removed by
earlier optimizations, including those in LTO pipelines, is outside this
validation guarantee; validation does not run before those optimizations merely
to diagnose metadata that they might remove. Annotation support
and backend metadata validation at `-O0` are outside the first-version guarantee;
Clang still diagnoses invalid source pragmas during semantic analysis.

## 4. Resolution and override rules

Resolve each loop's configuration independently, before analyzing candidates:

1. Respect pass availability and structural prerequisites. These hints
   do not insert the IR passes at `-O0`, add queue lowering to GlobalISel, override
   `optnone`, or remove machine fission's hardware-loop requirement. SWP's
   `optsize` exclusion is a soft restriction that `force` can bypass.
2. An explicit true disable flag wins over metadata for its pass:
   `-riscv-ns-disable-software-pipelining`,
   `-riscv-ns-disable-ir-loop-fission`, or
   `-disable-loop-fission-decomposed-branch`.
3. Otherwise, explicit loop metadata wins over the matching command-line
   tuning value, even when that command-line value was explicitly supplied.
4. Missing metadata uses the existing command-line value and built-in default.
5. Resolve `force` for this loop. When it is off, apply the existing correctness
   checks and budgets. When it is on, bypass the soft restrictions and semantic
   checks listed below, while retaining construction prerequisites and hardware
   bounds.

For example, `swp_distance(8)` wins over `-riscv-ns-swp-distance=16`;
`swp_toggles(disable)` wins over `-riscv-ns-swp-toggles`; and `swp(enable)` cannot
defeat `-riscv-ns-disable-software-pipelining`. There is no last-option-wins rule
between pragmas and backend flags.

Preserve the behavior of legacy numeric zero settings on unannotated loops.
In particular, `-riscv-ns-swp-distance=0` currently disables the entire SWP pass.
Move that gate to per-loop resolution: a loop with a positive local
`swp_distance` can override it, while loops inheriting zero still skip SWP,
including its in-place paths. Similarly, a positive local fission depth can
override a machine depth default of zero. Use the explicit disable flags when
the intent is to suppress a pass throughout the compile. Source-level numeric
zero is rejected; use the corresponding `disable` hint.

### Forced transformations

`enable` requests an attempt under the resolved settings and ordinary checks.
Raising a local size limit overrides that particular heuristic.
`riscv_ns_force(enable)` additionally bypasses both soft restrictions and
semantic correctness checks for enabled transformations:

| Kind of check | Behavior under `force` |
|---|---|
| Profitability and size heuristics | Bypass slice, loop and prologue size limits, the SWP function branch-count estimate, `optsize` exclusion, and short-trip profitability filters. This includes locally specified soft limits. |
| Memory and cross-iteration dependence proofs | Bypass alias/dependence rejection in SWP and both fission passes. SWP does not shorten the distance because a store might affect a condition load. |
| Speculative execution safety | Bypass checks that reject computing a condition for an iteration that may never execute, including loads or potentially trapping operations. SWP without a computable trip count still requires `swp_speculative(enable)` or its command-line equivalent. |
| Other semantic checks | Bypass checks whose sole purpose is to establish that moving or duplicating operations preserves program behavior, including checks expressed as exclusions of particular instruction or loop forms. Separate semantic checks from construction requirements rather than treating a whole candidate-analysis failure as success. |
| Construction prerequisites | Retain requirements the implementation needs to construct structurally valid IR/MIR: required preheaders/latches, computable future values, usable trip-count or recurrence representations, valid SSA/value mapping, and branch execution structure required to pair queue pushes and pops. Force does not supply missing transformation machinery. |
| Hardware and lowering bounds | Retain queue capacities, representable operands, actual register-allocation requirements, available lowering, and metadata validation. |

The force flag deliberately permits transformations without proving semantic
correctness. Violated assumptions can produce wrong results or traps. It does
not suppress IR/MIR verification or allow the compiler to continue with an
incomplete transformation plan. Explicit pass-disable flags and per-loop
`disable` hints still win over `force`.

A current rejection does not by itself establish a construction prerequisite.
Classify each check by why it is needed, not by whether the pass currently
accepts that instruction or loop form. If the implementation can construct and
lower the transformation within hardware bounds but cannot prove it preserves
program behavior, force bypasses that proof requirement. A retained construction
check must identify what the implementation cannot build without it. Explicit
pass and feature gates, metadata validation, and the analysis-maintenance
restriction described in section 6 still apply.

Separate construction checks, heuristic checks and semantic proof checks in the
implementation. Candidate analysis, transformation planning and transformation
helpers must all use the resolved per-loop force flag. Accepting a forced
candidate during analysis is insufficient if a later helper still rejects it
or asserts the semantic property that force deliberately bypasses. For example,
a recurrence for which SWP cannot construct the future value still prevents
pipelining, but failure to prove a load independent of stores
does not reject a forced plan. Bypassing the branch-count estimate must not
disable or modify the branch-register allocator; register exhaustion can still
prevent compilation.

Distances retain the current meaning: they are requested upper bounds, not
guarantees. Counted loops use `swp.distance`; loops pipelined in place use
`swp.speculative_distance`, including counted toggle-only loops. The current
toggle analysis can also cap that distance by the counted distance. Without
force, prologue budget and memory dependences may reduce a branch's distance
further or exclude the branch; force bypasses those two reductions. Report the
effective result in either mode. Force does not guarantee that the requested
distance or transformation is constructible. An exact-distance requirement would
need a separate future policy.

The `swp_speculative` switch gates loops without a computable trip count, not
every use of the in-place transformation. Counted toggle-only loops can still
be pipelined in place with `swp_toggles` enabled even when `swp_speculative` is
disabled, preserving current behavior. Force does not change either feature gate.

Keep the existing limits of 62 for SWP and 64 for fission. They are the bounds
currently used by the respective implementations; this proposal does not
reconcile their differing queue-capacity assumptions or change the architecture.

## 5. Pass ordering and metadata lifetime

Keep the existing pass order. Independent controls give these recipes:

| Desired attempt | SWP | IR fission | Machine fission |
|---|---|---|---|
| Existing automatic behavior | absent | absent | absent |
| SWP only | enable | disable | disable |
| IR fission only | disable | enable | disable |
| Machine fission only | disable | disable | enable |
| Neither transformation | disable | disable | disable |

If several passes remain enabled, retain today's opportunities: IR fission gets
first refusal, SWP sees what remains, and machine fission may process a SWP drain
loop. Failure of a requested pass does not imply permission to run a pass
explicitly disabled by metadata.

Loop IDs must survive hardware-loop formation and instruction selection. Use
`Loop::getLoopID()` in IR and `MachineLoop::getLoopID()` in machine fission.
The latter already reads `!llvm.loop` through the IR blocks associated with
machine latches, requiring consistent IDs across all latches. Extend the current
IR-fission marker lookup to use this common route. Do not use source locations,
block names, or raw metadata pointers as persistent loop identities.

Generated loops need explicit metadata handling:

| Transformation | Required handling |
|---|---|
| IR fission | Give compute and consume loops fresh IDs; preserve applicable unrelated properties; consume user tuning requests and retain `riscv.loop-fissioned` on both. Both remain ineligible for another fission or SWP attempt. |
| Counted SWP | Give kernel and drain distinct IDs. Add an internal `llvm.loop.riscv.ns.swp.applied` marker with a string role (`kernel` or `drain`) to each, and consume SWP tuning requests. SWP and IR fission skip both roles; machine fission skips the kernel. Carry the original machine-fission enable/disable state, fission depth and force flag to the drain, which can still be fissioned. |
| In-place SWP | Replace the loop ID with a fresh ID carrying the applied marker with role `inplace` and applicable unrelated properties; consume tuning requests. All three passes skip this queue consumer. |
| Machine fission | Consume the request for this invocation and keep generated compute, consume and tile loops from being candidates again. Newly created machine blocks need not have corresponding IR blocks; do not mutate a shared IR loop ID to mark them. |

In particular, SWP currently creates a new kernel latch branch. The implementation
must attach the new loop ID there explicitly; cloning the original blocks is
insufficient. Update the IR-fission helper that currently copies all old loop
operands so it consumes requests instead of forwarding them indiscriminately.

Preserve current code generation for unannotated input. Adding internal markers
must not remove today's machine-fission opportunity on an unannotated SWP drain.
If a supported lowering path loses an annotated loop's controls, fix propagation
or add a target-owned machine-loop record before supporting that path. Falling
back to defaults is appropriate for loops without annotations, but must not
silently undo an explicit disable.
An absent recoverable loop ID alone is not evidence that a loop was unannotated.
Preserve enough provenance through lowering, in metadata or a target-owned
record, to distinguish unannotated loops from loops whose controls were lost.

### Earlier generic loop transformations

Clang emits these controls as persistent loop properties, including in the
follow-up metadata it builds for requested unrolling or vectorization. Audit
rotation, partial/runtime unrolling, vectorization, peeling and distribution so
surviving descendant loops retain the controls where applicable. Full unrolling
can remove the loop completely; no backend transformation is then promised.
Annotations do not implicitly disable these generic optimizations.

Distance and size units describe the loop as seen by the backend. An unrolled
iteration can represent several source iterations. Users who need a direct
source-iteration comparison should also disable unrolling and vectorization on
that loop. Test metadata preservation with both ordinary optimization and LTO;
preserving unknown metadata in one pass does not establish an end-to-end path.

## 6. Implementation plan

1. **Backend metadata support.** Add a small shared RISC-V loop-options helper
   that parses an `MDNode` into typed optional fields, validates them, and resolves
   immutable per-loop settings, including force, against existing flags. Share
   it across the three passes. Never assign local overrides into process-global
   `cl::opt` variables.
   Schedule an independent validation step immediately before IR fission in supported
   optimized RISC-V compilations. Check loop metadata on IR terminators, not
   just eligible transformation candidates, and emit compilation errors for
   malformed recognized options even when all three transformations are disabled.
   This guarantee covers metadata still present at that stage, not metadata
   removed by earlier optimizations.
   Consumers must also diagnose malformed recognized options they encounter.
2. **SWP integration.** Store the resolved settings in `LoopPlan` and pass them
   through loop-size checks, slice construction, recurrence recognition,
   speculative eligibility and distance/prologue calculations. Move the numeric
   zero and `optsize` early returns into resolution. Separate construction
   requirements from soft limits and semantic proofs so force bypasses only the
   latter two categories.
   Carry force through candidate analysis, plan construction and transformation
   helpers. In particular, make `cloneCondition()`'s assertion that speculative
   clones contain no loads and are safe to speculate conditional on force;
   retain structural assertions and IR/MIR verification. Keep function branch
   accounting shared across plans;
   forced plans contribute to it but bypass the estimate's rejection threshold.
3. **Fission integration and propagation.** Resolve controls before candidate
   selection, use the local depth in both stages, and implement the lifetime
   rules above. Make dependence and other semantic proof checks conditional on
   force while preserving the prerequisites needed to build each loop pair.
   Audit candidate analysis, transformation planning and emission helpers for
   checks or assertions that must distinguish forced from ordinary plans.
   Keep IR fission's one-success-per-function restriction initially.
   Force does not bypass this analysis-maintenance restriction. Supporting every
   eligible loop requires rebuilding or updating its analyses after each change
   and should be a separate change.
4. **Clang support.** Extend loop-hint parsing, AST representation, semantic
   validation and compatibility checks; add the corresponding optional fields
   to `CGLoopInfo`. Ensure a loop with only Non-Spec hints gets a loop ID even
   without debug information, and clear the fields after each loop. Preserve
   them through Clang's follow-up metadata construction and serialization.
5. **Documentation.** Document the source syntax, metadata contract and force
   semantics. Update the optimization guide when implementation lands. An IR-only
   first patch makes backend behavior reviewable before the frontend syntax.

## 7. Diagnostics and acceptance checks

Extend SWP's existing optimization remarks using its existing pass name.
Neither fission pass currently emits optimization remarks; adding them is
optional and is not required for the first version. The reporting requirements
below apply to SWP only. When SWP runs, report annotated loops' selected settings,
whether values came from metadata or defaults, requested and effective
distance, and a reason when a request is blocked or reduced. Useful reasons
include a global disable,
an unavailable lowering, unsupported shape, dependence, prologue limit, function
branch budget. Use source locations when
available and function/loop identification otherwise. Do not require an
assertion build to understand whether a request took effect.

For forced transformations, report that force is active and which restrictions
or proof requirements were bypassed. If a forced request is still rejected,
identify the construction prerequisite, hardware bound or explicit disable
responsible.

Malformed recognized metadata present at the validation stage produces
compilation errors independently of
optimization remarks and their enablement. A valid request rejected because of
an unsupported loop shape or another transformation prerequisite is still a
missed optimization, not a compilation error.

An entirely unscheduled pass, such as IR SWP at `-O0`, cannot emit a pass remark.
Document that annotation support and backend metadata validation at `-O0` are
not guaranteed in the first version.

The implementation should pass these checks before the syntax is considered
supported:

- **Frontend:** source-to-IR mapping for every hint; false versus absent;
  constant expressions and templates; duplicate/out-of-range/nonconstant
  diagnostics; hints on non-loops; no leakage between sibling or nested loops;
  emission without `-g`.
- **Resolution:** two loops with different settings in one function; local
  metadata versus explicitly supplied CLI values; all three global disable
  flags; numeric zero compatibility; feature disable overriding a global
  enable; a distance hint that does not enable speculative pipelining; force
  versus pass disables and opt-in feature switches; counted toggle-only
  in-place SWP with speculative pipelining disabled.
- **Metadata errors:** wrong arity, operand types, out-of-range values and
  duplicate recognized keys present at the validation stage fail compilation,
  including equal-value duplicates
  and malformed shared `force` or fission depth. Check errors with all three
  transformations disabled, on ineligible loops and with remarks disabled;
  unknown keys remain accepted for forward compatibility. Cover loops removed
  by earlier optimizations without requiring a backend metadata diagnostic.
- **Transforms:** the selection recipes above; IR fission's local depth cutoff;
  machine tiling with a remainder; counted, speculative, shift and toggle SWP;
  ordinary distance reduction and rejection of unsafe loads and dependences;
  the shared function branch budget across several annotated loops.
- **Force:** unknown-alias and speculative-safety cases rejected normally but
  transformed with force; bypass of all soft limits, including explicit local
  limits and `optsize`; unchanged rejection of unsupported shapes, invalid
  metadata and hardware-bound violations; valid IR/MIR after every forced
  transformation; accounting for forced and unforced sibling loops. Exercise
  forced speculative loads in assertion builds.
- **Lifetime:** inspect IR after fission, SWP and hardware-loop formation;
  inspect MIR before machine fission; verify both SWP kernel and drain carry
  the intended policies, including force on the drain; run transforms again to
  check idempotence; cover generic optimization and LTO paths and machine loops
  without recoverable IDs.
- **Compatibility:** unannotated regression inputs retain their generated
  assembly; existing SWP tests still pass; unavailable passes retain their
  existing behavior. A machine loop with no recoverable ID uses legacy defaults
  only when known to be unannotated; lost controls must not silently restore
  defaults.
- **Execution:** compare annotated and baseline kernels in the existing Non-Spec
  simulator verification flow, including short trips, re-entry, nested loops
  and partial final tiles. Include forced cases whose semantic assumptions hold
  and verify output equality before measuring cycles. For deliberately violated
  assumptions, test that the requested checks were bypassed and the output is
  structurally valid; do not require semantic equivalence.

## 8. Scope of the first version

Implement the metadata contract, independent pass controls, listed local
parameters, force support, diagnostics and Clang spelling. Special handling for
non-RISC-V targets and a feature-test macro are outside the first version.
Leave per-branch selection, exact distances, profile-driven automatic tuning,
per-function budgets, generic
hardware-loop controls and a combined "prefer this transform" policy for later.
Those features need additional semantics; none is necessary to tune and compare
individual loops with the existing transformations.
