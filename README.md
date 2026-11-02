# The LLVM Compiler Infrastructure

Non-speculative RISCV modification to the [LLVM Project](https://github.com/llvm/llvm-project).

## Overview of changes
- Added new BMOV/PBAL instructions to RISCV ISA
- Removed tablegen instruction selection for Pseudo branch instructions
- Added Pseudo branch instruction expansion after optimizations
	- See RISCVExpandPseudoInsts.cpp
- Added Branch Support Analysis pass (used to emit PBAL labels)
	- See RISCVBranchSupportAnalysis.{h,cpp} and RISCVAsmPrinter.cpp
- Added Branch Condition Pipelining pass (software pipelining of loop branch conditions)
	- Pushes each in-loop branch's condition into its branch register's FIFO
	  N iterations before the PB consumes it
	- See RISCVBranchConditionPipelining.cpp and the llvm.riscv.ns.cond.* intrinsics
	
## New Compiler Flags
| Flag                                      | Description                                              |
|-------------------------------------------|----------------------------------------------------------|
| `-riscv-ns-disable-hardware-loops`        | Disables BMOVC_LOOP transformation/emission              |
| `-riscv-ns-disable-branch-setup-hoisting` | Disables BMOV hoisting optimization pass                 |
| `-riscv-ns-disable-cross-bb-hoisting`     | Disables cross basic block hoisting of BMOV instructions |
| `-riscv-ns-disable-software-pipelining`   | Disables branch condition pipelining                     |
| `-riscv-ns-swp-distance=N`                | Iterations ahead to compute conditions (default 16, max 62) |

Branch condition pipelining also has tuning limits: `-riscv-ns-swp-max-slice-size`,
`-riscv-ns-swp-max-loop-size`, `-riscv-ns-swp-max-prologue-size` and
`-riscv-ns-swp-max-function-branches`. To see which loops it pipelines, use
`-pass-remarks=riscv-branch-condition-pipelining`, or
`-debug-only=riscv-branch-condition-pipelining` for why it skips the others.
From clang, pass these flags through `-mllvm`.

## Known Issues
- `RISCVExpandPseudo` claims to preserve every analysis (`AU.setPreservesAll()`), but
  `expandCCOp` splits blocks when it expands the `PseudoCC*` instructions (selected with
  `short-forward-branch-opt`), leaving its `MachineDominatorTree` and `MachineLoopInfo` stale
	- Latent for now: `RISCVExpandAtomicPseudo` runs next and preserves neither, so the
	  pass manager frees them and `RISCVBranchSetupHoisting` gets freshly computed ones
	- Becomes a real bug if a pass that needs them is scheduled right after
	  `RISCVExpandPseudo`, or if `RISCVExpandAtomicPseudo` starts preserving them
	- Fix by dropping `setPreservesAll()`, or by updating both analyses in `expandCCOp`
	- See RISCVExpandPseudoInsts.cpp


## My CMake Setup
```bash
-G Ninja
-DCMAKE_C_COMPILER=clang
-DCMAKE_CXX_COMPILER=clang++
-DCMAKE_BUILD_TYPE=Debug
"-DLLVM_ENABLE_PROJECTS=clang;lld"
-DLLVM_TARGETS_TO_BUILD=RISCV
-DLLVM_DEFAULT_TARGET_TRIPLE=riscv64-unknown-linux-gnu
-DCLANG_DEFAULT_RTLIB=compiler-rt
-DCLANG_DEFAULT_LINKER=lld
-DCLANG_DEFAULT_CXX_STDLIB=libc++
-DCLANG_DEFAULT_UNWINDLIB=libunwind
-DLLVM_RUNTIME_TARGETS=riscv64-unknown-linux-gnu
"-DLLVM_ENABLE_RUNTIMES=compiler-rt;libunwind;libc;libcxx;libcxxabi"
-DRUNTIMES_riscv64-unknown-linux-gnu_LIBUNWIND_USE_COMPILER_RT=ON
-DRUNTIMES_riscv64-unknown-linux-gnu_LIBUNWIND_ENABLE_SHARED=OFF
-DRUNTIMES_riscv64-unknown-linux-gnu_LIBCXXABI_ENABLE_SHARED=OFF
-DRUNTIMES_riscv64-unknown-linux-gnu_LIBCXX_ENABLE_SHARED=OFF
-DRUNTIMES_riscv64-unknown-linux-gnu_LIBC_KERNEL_HEADERS=/home/mitchell/riscv-kernel-headers/include
"-DRUNTIMES_riscv64-unknown-linux-gnu_LLVM_ENABLE_RUNTIMES=compiler-rt;libunwind;libc;libcxx;libcxxabi"
-DRUNTIMES_riscv64-unknown-linux-gnu_LLVM_INCLUDE_TESTS=OFF
-DRUNTIMES_riscv64-unknown-linux-gnu_COMPILER_RT_BUILD_SANITIZERS=OFF
-DRUNTIMES_riscv64-unknown-linux-gnu_COMPILER_RT_BUILD_XRAY=OFF
-DRUNTIMES_riscv64-unknown-linux-gnu_COMPILER_RT_BUILD_LIBFUZZER=OFF
-DRUNTIMES_riscv64-unknown-linux-gnu_COMPILER_RT_BUILD_MEMPROF=OFF
```

## How to Build
```bash
ninja -C build llc llvm-mc llvm-objdump
```

List of Implicit LLVM Assumptions that are not documented anywhere:
- Branch instructions will always hold their branch target
- All terminators of a basic block will be at the end of a basic block
- All Phi instructions must be at the beginning of a basic block
- LLVM will just copy MachineInstr's as it pleases without letting you know

# NOTES TO SELF
- Try reverting back to old MachineBasicBlock.cpp, I don't think my changes matter anymore..

