# Keeping loop memory in registers (`riscv-ns-version-promote`)

The `riscv-ns-version-promote` IR pass keeps loop-invariant integer memory
locations in registers when ordinary loop-invariant code motion (LICM) cannot.
It uses a runtime overlap check when alias analysis cannot prove that other
loop accesses are disjoint, retaining the original loop as the fallback.

Added in `cacfb06bd366`; documented against `48bfcf153beb`, 2026-11-14.
Enabled by default in optimized RISC-V IR pipelines, with
`-riscv-ns-disable-version-promote` to turn it off.

## 1. Why

A Non-Spec `pb` waits at dispatch for its condition. If that condition reads a
fixed memory location on every iteration, it repeatedly waits for a load even
though the location's address never changes. LICM can leave those loads in place
for two reasons:

- A store through another pointer may alias the location. This includes stores
  indexed by loaded data, such as `histogram[r][g][b]`.
- The loop conditionally stores to the location, and LICM cannot prove that an
  unconditional store after the loop is allowed. The object may be read-only on
  the paths that never update it.

Version promotion handles both cases. Conditional min/max updates to structure
fields and a loop bound repeatedly loaded from a global are the motivating
examples in the regression test.

## 2. What it changes

For a location `p` that the loop reads and may update, the transformation has
this shape (pseudocode):

```text
if the loop is entered:
    initial = *p
    if the checked ranges are disjoint and do not wrap:
        value = initial
        run the loop, replacing its accesses to *p with value
        if value != initial:
            *p = value
    else:
        run the original loop with its original loads and stores
```

The initial load is in the loop preheader, before the overlap check. It is only
introduced when an access to the location is guaranteed once the loop is
entered, and its address can be computed there safely. Existing guards that
skip the loop still skip this load.

A read-only location needs no writeback: its loads are replaced by the initial
value. A location that is written becomes SSA values and PHIs, with a store at
the exit only when its final value differs from its saved initial value. This
avoids adding a store on a path that never wrote the object. Storing the same
value or changing it and changing it back needs no final store.

If alias analysis proves all relevant accesses disjoint, the pass promotes
conditional updates **in place**, without copying the loop or adding an overlap
check. It leaves a loop alone when only read-only candidates remain and no
check is needed; ordinary LICM already handles that case.

The pass itself preserves the conditional control flow inside the promoted
loop. Subsequent InstCombine and SimplifyCFG can turn register updates into
selects or min/max operations. [Branchless select lowering](branchless-select.md)
can then remove the corresponding selection branches. Hoisting a loop-bound
load can also make the fast loop's trip count computable for later passes.

## 3. Eligibility and overlap checks

- The loop must be innermost, in loop-simplify form, with a preheader, one
  exiting block and one exit block. The pass first attempts loop simplification
  and LCSSA formation.
- Loads and stores must be non-atomic and non-volatile. Other instructions that
  may read or write memory, or may throw, reject the loop; this rules out ordinary
  calls, fences and atomic operations.
- A candidate has an invariant address in address space zero. Its accesses must
  agree on an integer type whose bit size equals its storage size. Floating-point,
  vector and mixed-type locations are not promoted.
- At least one surviving candidate must feed a conditional branch, a select's
  condition, or a min/max operand. This search is bounded and stops at PHIs and
  loads; it does not trace arbitrary recurrences. Other eligible locations can
  be promoted along with that candidate.
- A candidate must be separable from all conflicting accesses. Read/read pairs
  need no check. Candidates promoted together must be statically disjoint when
  either is written; this pass does not version possible aliases between those
  candidates. Other accesses proven disjoint also need no check.

For remaining may-alias accesses, scalar evolution supplies half-open byte
ranges `[lo, hi)`:

| Address shape | Range used by the check |
|---|---|
| Loop-invariant address | The accessed object's bytes |
| Affine address with a constant step | First through last access, using the backedge count and access size; positive and negative steps are supported |
| Other address based on an invariant pointer | Base plus the bounded signed range of its offset, including the final access's size |

An affine trip count may become available only after promotion, for example
when the exit condition reads a global that the checked loop hoists. The pass
recomputes the ranges in that loop. Unbounded data-dependent indices, bounds
that cannot be safely computed before the loop, and accesses known to overlap
prevent the corresponding promotion.

Ranges whose bounds differ by constants are grouped to reduce overlap tests.
The runtime guard sends overlapping or wrapping ranges to the original loop;
affine ranges also have a trip-count/step overflow guard. If constructing the
final check fails after cloning, the version branch is set to always run the
original loop, and a missed remark records the reason.

## 4. Pipeline and controls

The pass runs at the new pass manager's **vectorizer-start extension point**,
after inlining and before the vectorizer pipeline's cleanup. It is registered
for optimization levels other than `-O0`, including size optimization, and
skips `optnone` functions. It runs before the later code-generation IR passes
for Non-Spec fission, decide/act splitting, pipelining and hardware loops.

`riscv-ns-version-promote` is the name for `opt -passes=...`, not an enable flag.
A normal `llc` invocation does not run this optimization pipeline. To compare
the pass on and off, set the option when Clang or `opt` optimizes the IR;
disabling it only when lowering already-optimized IR cannot undo promotion.
There is no pass-specific loop pragma or metadata control.

| Option | Default | Meaning |
|---|---|---|
| `-riscv-ns-disable-version-promote` | false | Disable the pass, including in-place promotion |
| `-riscv-ns-version-promote-max-loop-size=N` | 256 | Maximum IR instructions in an eligible loop; also applies to in-place promotion |
| `-riscv-ns-version-promote-max-range=N` | 16777216 (16 MiB) | The offset span for a data-indexed access must be strictly less than N bytes; access size is added to the upper bound separately. This does not cap affine trip ranges. |
| `-riscv-ns-version-promote-max-checks=N` | 8 | Maximum grouped range-overlap tests; a preliminary filter also rejects more than `4 * N` distinct accesses needing checks. Wrap/overflow guards are additional. |

Clang accepts these hidden options through `-mllvm`. For example:

```sh
clang --target=riscv64 -O2 -S -emit-llvm input.c -o promoted.ll
clang --target=riscv64 -O2 -S -emit-llvm \
  -mllvm -riscv-ns-disable-version-promote input.c -o baseline.ll
```

Versioning trades extra entry checks, a second loop body and conditional
writeback branches for fewer loads and stores inside the fast loop. Short
loops or frequent overlap can pay the check without recovering its cost.
Raising the limits admits more code and checks; it does not guarantee a speedup.

## 5. Inspecting and testing it

From the repository root, using a configured build with current `opt` and
`FileCheck` binaries:

```sh
build/bin/opt -mtriple=riscv64 -passes=riscv-ns-version-promote \
  -verify-each -S llvm/test/CodeGen/RISCV/ns-version-promote.ll \
  -o /tmp/ns-version-promote.ll \
  -pass-remarks=riscv-ns-version-promote \
  -pass-remarks-missed=riscv-ns-version-promote
build/bin/llvm-lit -v llvm/test/CodeGen/RISCV/ns-version-promote.ll
```

In Clang, use `-Rpass=riscv-ns-version-promote` and
`-Rpass-missed=riscv-ns-version-promote`. An assertion build also supports
`-debug-only=riscv-ns-version-promote` and `-stats`. Successful remarks report
the location count and overlap-test count, or say that no check was needed.
Missed remarks cover unbounded conflicting accesses, check size and final
check construction; early eligibility failures do not all emit a remark.

The counters distinguish loops versioned, loops promoted in place, writable
locations promoted, read-only loads hoisted and abandoned checks. In IR, look
for `nsvp.init`, `nsvp.slot` PHIs, `nsvp.changed`, range checks and the
`lver.orig` fallback.

The [regression test](../llvm/test/CodeGen/RISCV/ns-version-promote.ll) covers:

| Function | Expected behavior |
|---|---|
| `minmax` | Six structure fields promoted, with two overlap tests against affine input and a bounded data-indexed histogram |
| `counted` | Global loop bound hoisted, with one overlap test against output stores |
| `with_call` | No promotion across an ordinary call |
| `unbounded` | No promotion with an unbounded loaded index |
| `inplace` | Conditional update promoted without a check; write back only if changed |

The disable run checks that neither promotion nor versioning appears.

## 6. Recorded measurement

Commit `cacfb06bd366` records MiBench `consumer-tiffmedian-small` on the gem5
PBTB core, with `get_histogram`'s min/max loop and `quant`'s loop versioned:

| Configuration | Cycles |
|---|---:|
| Non-Spec baseline | 160.83M |
| Version promotion | 104.34M |
| Version promotion and branchless selects | 67.29M |
| Vanilla | 46.23M |

The commit reports unchanged output. These are historical measurements from
before the poison-safe branchless select update in `48bfcf153beb`, not a fresh
benchmark of the current defaults. For a current A/B run, explicitly disable
each optimization in the baseline; omitting the options now enables both.

Implementation: [RISCVNSVersionPromote.cpp](../llvm/lib/Target/RISCV/RISCVNSVersionPromote.cpp).
Registration: [RISCVTargetMachine.cpp](../llvm/lib/Target/RISCV/RISCVTargetMachine.cpp).
