# Non-Spec branchless integer selects

`-riscv-ns-branchless-select` defaults to true. Pass
`-riscv-ns-branchless-select=false` to `llc` (or through Clang's `-mllvm`)
to disable it. It changes the scalar integer fallback in SelectionDAG's
`RISCVTargetLowering::lowerSELECT`; native conditional instructions and earlier
select combines retain their existing priority. It does not guarantee that
every source-level select is branchless: other passes can form branches, and
floating-point/vector results have their own lowering paths.

## Correctness of the fallback

The transformation is a refinement of `select c, t, f`:

```
T = freeze(t)
F = freeze(f)
C = zero_extend_or_truncate(c, result_width)
C = (C != 0) if its upper bits are not known to be zero
M = 0 - C                   // modular subtraction, no no-wrap flags
R = F ^ ((T ^ F) & M)
```

The proof relies on the SelectionDAG contract, not on arbitrary integer
conditions being valid inputs. `ISD::SELECT` requires boolean contents as
specified by `TargetLowering::getBooleanContents`. RISC-V uses
`ZeroOrOneBooleanContent` for scalar integer and floating-point comparisons.
Thus a defined condition is exactly 0 or 1 before changing its width. Type
legalization normalizes promoted i1 conditions, including sign-extended ABI
arguments and truncations of wider integers. Truncating a general integer to
i1 tests its low bit, not whether the general integer is nonzero.

For any positive result width, zero extension/truncation preserves 0 and 1,
and the optional nonzero comparison preserves those values. In modular
arithmetic, `-0 = 0` and `-1` has every bit set. Therefore:

* If `c = 0`, `R = F ^ 0 = F`.
* If `c = 1`, `R = F ^ (T ^ F) = T`.

This identity holds for every bit pattern, including signed minima/maxima,
without overflow assumptions. Narrow/wide IR integers are promoted/split by
type legalization; the identity applies to each legal scalar result.

Both freezes are necessary for the semantic argument. LLVM select does not
propagate poison from its unchosen arm, while integer XOR and AND do. Freezing
each arm prevents such propagation. Reusing the same frozen `F` also makes
the cancellation valid if `f` contains undefined bits: two unfrozen uses may
choose different values. If the chosen arm is defined, freeze preserves it;
if it is poison or undefined, picking one allowed concrete value is a valid
refinement. An undefined condition may choose either arm; a poison condition
places no additional requirement on the result. See the LLVM language
reference for [select](https://llvm.org/docs/LangRef.html#select-instruction)
and [freeze](https://llvm.org/docs/LangRef.html#freeze-instruction).

The rewrite consumes already-present DAG values. It does not move a load,
store, call, or other side effect onto a previously unexecuted control-flow
path. Its new operations are integer arithmetic and freezes, which introduce
no traps or memory effects.

## Verification and limits

`llvm/test/CodeGen/RISCV/Inputs/ns-branchless-select.smt2` checks the 32-bit
and 64-bit identity and refinement in the presence of unselected poison and
undefined bits. Run `z3` on that file; both queries must report `unsat` (no
counterexample). The symbolic frozen values are constrained to preserve
defined input bits and are otherwise arbitrary.

`RISCVSelectionDAGTest` checks the actual lowering's poison protection and
reuse of the frozen false arm on RV32 and RV64. `ns-branchless-select.ll`
checks default/explicit enablement, opt-out, Zicond priority, condition
normalization, legalized widths, comparisons, poison-producing expressions,
min/max, and saturating arithmetic with the machine verifier enabled.

With a configured build directory named `build`, run:

```
ninja -C build llc FileCheck RISCVTests
build/unittests/Target/RISCV/RISCVTests
build/bin/llvm-lit -v llvm/test/CodeGen/RISCV/ns-branchless-select.ll
z3 llvm/test/CodeGen/RISCV/Inputs/ns-branchless-select.smt2
```

This is a proof of the local rewrite under LLVM's documented semantics,
supported by implementation tests. It is not a formal verification of the
entire LLVM pipeline, earlier target combines, instruction selection, or the
Non-Spec hardware implementation.
