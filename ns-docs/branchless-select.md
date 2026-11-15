# Branchless integer selects (`-riscv-ns-branchless-select`)

Scalar integer selects that reach the Non-Spec SelectionDAG fallback are
lowered to a bitwise mask blend instead of a branch over a move. This also
covers min/max and saturating arithmetic when their expansion produces such
selects.

Added in `685426a4c0b3` as an opt-in; `48bfcf153beb` made the fallback poison-safe
and **enabled it by default**. Documented against `48bfcf153beb`, 2026-11-14.
Use `-riscv-ns-branchless-select=false` for the disabled baseline.

## 1. Why

Without a native conditional instruction or an earlier simplifying combine,
a select previously became a conditional branch around a move. Its `pb` waits
at dispatch for the condition, even though both values are already available
at this lowering step. The mask blend removes that selection branch and its
B-register setup, replacing the control dependency with integer data
dependencies.

This complements [version promotion](version-promote.md): promotion removes
repeated memory accesses, later cleanup can turn conditional register updates
into selects, and this lowering can remove their selection branches.

## 2. What it emits

For `select c, t, f`, the fallback builds these DAG operations:

```text
T = freeze(t)
F = freeze(f)
C = zero_extend_or_truncate(c, result_width)
C = (C != 0) if known-bits analysis cannot prove its upper bits are zero
M = 0 - C                  // modular arithmetic, no no-wrap flags
R = F ^ ((T ^ F) & M)
```

A defined scalar SelectionDAG condition is 0 or 1. Its negation is therefore
zero or all ones: the result is `F` when false and `T` when true. Type
legalization handles narrow and wide IR integers before this identity is used
on each legal scalar result.

For the test's `select_i64` on RV64, with an already normalized condition in
`a0`, true value in `a1` and false value in `a2`, the selection itself is:

```asm
neg a0, a0
xor a1, a1, a2
and a0, a1, a0
xor a0, a2, a0
```

With the option disabled, this example uses `bmovc_bne`, a `pb` and a
conditional move path, plus the branch's `bmovs`/`bmovt` setup. The function's
return still uses a `pb` in either case. Constants, comparisons and later
combines can change the exact arithmetic sequence.

## 3. Poison, undefined values and scope

LLVM `select` ignores poison in its unchosen arm; XOR and AND propagate it.
Freezing both arms before blending preserves the select's semantics. The
same frozen false value is reused in both XORs, so undefined bits cannot take
different values and invalidate cancellation. Freeze preserves a defined
chosen value and may choose a concrete value for poison or undefined input.
It is a DAG semantic operation, not a runtime check or memory barrier.

Condition normalization relies on RISC-V's `ZeroOrOneBooleanContent` contract.
For example, truncating a wider integer to `i1` selects using its low bit;
this change does not reinterpret that source integer as a general nonzero test.
The [detailed refinement proof](../llvm/docs/RISCV/NonSpecBranchlessSelect.md)
explains these assumptions and the SMT checks.

The scope is deliberately local:

- The implementation is in `RISCVTargetLowering::lowerSELECT`, not a separate
  IR pass. There is no `opt -passes=riscv-ns-branchless-select` pipeline entry.
- It applies to scalar integer results on RV32 and RV64, including integer
  results selected by a floating-point comparison. Floating-point results and
  vectors have their own lowering paths.
- Native conditional lowering for Zicond/XVentanaCondOps and earlier select
  combines keep their priority. The flag controls only selects that reach this
  fallback, so disabling it does not force all selects to become branches.
- It consumes existing DAG values and does not move loads, stores, calls or
  other side effects onto a previously unexecuted path.
- Earlier passes may form branches instead of selects. This is not a promise
  that every source ternary, min/max or saturation becomes branchless.

## 4. Controls and tradeoffs

| Option | Default | Effect |
|---|---|---|
| `-riscv-ns-branchless-select` or `=true` | true | Enable the scalar integer mask fallback |
| `-riscv-ns-branchless-select=false` | | Restore the previous fallback for A/B testing |

Pass it directly to `llc` or through Clang's `-mllvm`:

```sh
llc -mtriple=riscv64 -mattr=+m input.ll -o enabled.s
llc -mtriple=riscv64 -mattr=+m -riscv-ns-branchless-select=false \
  input.ll -o disabled.s
clang --target=riscv64 -O2 -S -mllvm -riscv-ns-branchless-select=false \
  input.c -o disabled.s
```

This is a SelectionDAG option; it does not add a GlobalISel lowering. It has
no optimization-level gate of its own and no per-loop pragma or metadata
control. There are no dedicated pass remarks or statistics: compare generated
assembly with the explicit opt-out.

The expected benefit is avoiding selection-related `pb` stalls and B-register
use. The cost is arithmetic instructions and a dependency chain that execute
for either condition value. Measure cycles and committed instructions on the
target; removing a branch does not by itself establish a speedup.

## 5. Verification

The [code-generation test](../llvm/test/CodeGen/RISCV/ns-branchless-select.ll)
covers default and explicit enablement, opt-out, Zicond priority, sign-extended
and truncated conditions, narrow/wide integers, integer and floating-point
comparisons, poison-producing arms, min/max and saturation. Its RV32/RV64 runs
enable the machine verifier.

[SelectionDAG unit tests](../llvm/unittests/Target/RISCV/RISCVSelectionDAGTest.cpp)
check unselected-poison protection and reuse of the frozen false arm on both
targets. These properties are not observable from assembly alone.
The [SMT file](../llvm/test/CodeGen/RISCV/Inputs/ns-branchless-select.smt2)
checks the local 32-bit and 64-bit refinement; both queries should return
`unsat`.

From the repository root, with a configured build and Z3 installed:

```sh
ninja -C build llc FileCheck RISCVTests
build/bin/llvm-lit -v llvm/test/CodeGen/RISCV/ns-branchless-select.ll
build/unittests/Target/RISCV/RISCVTests --gtest_filter='*RISCVSelectionDAGTest*'
z3 llvm/test/CodeGen/RISCV/Inputs/ns-branchless-select.smt2
```

Commit `48bfcf153beb` records both SMT queries as unsatisfiable, all 26 RISC-V
unit tests passing, 168 compilation/machine-verifier checks passing, and no
new failures in the complete RISC-V suite (1,684 pre-existing failures).
It also records fresh GSM encode-small and decode-small builds, enabled and
disabled, producing byte-identical output to the recorded vanilla/Non-Spec
runs and MiBench's expected files in gem5. These are results recorded for that
commit, not a claim of whole-pipeline formal verification or coverage of every
benchmark input.

## 6. Recorded measurements

Commit `685426a4c0b3` records MiBench `telecomm-gsm-decode-small` on gem5's PBTB
core, compiled with `-O2 -riscv-ns-disable-hardware-loops`:

| Configuration | Cycles |
|---|---:|
| Non-Spec, branchless select off | 10,403,417 |
| Non-Spec, branchless select on | 4,259,427 |
| Vanilla | 2,652,088 |

The commit reports identical output and a disabled binary byte-identical to
the compiler before the change. The [version promotion measurement](version-promote.md#6-recorded-measurement)
also records a benefit from combining the two optimizations on tiffmedian.
These timings predate the freeze/default-on update; the later GSM validation
above establishes output agreement, not a fresh timing comparison here.
For current measurements, explicitly use `=false` for the disabled case.

Implementation: [RISCVISelLowering.cpp](../llvm/lib/Target/RISCV/RISCVISelLowering.cpp),
`NSBranchlessSelect` and `RISCVTargetLowering::lowerSELECT`.
