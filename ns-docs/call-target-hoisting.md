# Hoisting a call's target setup (`-riscv-ns-disable-call-target-hoisting`, `-riscv-ns-hoist-call-target-max-freq-ratio`)

Record of a change to `RISCVBranchSetupHoisting` that lets the `bmovt` of a direct call move, and of how it was verified. Branch `call-target-hoisting`, on top of `non-spec` at `a9eceb797b5e`. 2026-11-12.

Like the changes in `setup-placement.md`, this one is **on by default**, with a `-riscv-ns-disable-...` flag to turn it off.

Verification is done in layers, as in `swp-speculative.md`. The gem5 layer here uses the `~/drive1/verif` bench, not the `hoistfix` scripts of `setup-placement.md`, so its numbers are not directly comparable with that file's (section 7.1).

---

## 1. Why

A direct call, tail call or long jump expands to:

```
%b = BMOVS_J <.Lns_call_>
$x1 = AUIPC target-flags(riscv-call) @g
%b = BMOVT_I %b, $x1, 0
PseudoPBCALL %b
```

The hoisting pass moves the `BMOVS_J` as far up as it can, but the `BMOVT_I` never moved:

- It reads the register the `AUIPC` writes, so the `AUIPC` right above it is a register dependency it cannot cross.
- The two cannot be separated either. The `AUIPC` carries one `R_RISCV_CALL_PLT` relocation, which patches the instruction after it as well.

So every direct call had its target set up immediately before its `pb`, whatever room there was above it. This change moves the two instructions as one.

## 2. What was changed

In `RISCVBranchSetupHoisting.cpp`, about 145 lines added. A machine bundle was the alternative; it was not used because the bundle would also have to pass through the late B-register allocator, `VirtRegRewriter` and the asm printer, with an unpack step after them.

### 2.1 The pair is moved as one

- New `SetupGroup`: a setup and, optionally, its `Lead`. For a `BMOVT_I` whose previous instruction (debug instructions skipped) is an `AUIPC` writing the register the `BMOVT_I` reads, `Lead` is that `AUIPC`.
- `findHoistPoints`, `isSafeToHoistTo`, `scanBlockBackward`, `isHoistHazard`, `isCycleHazard` and `loopHasHazard` take a `SetupGroup` instead of a `MachineInstr`. The scan starts above the group's first instruction, and `scheduleBranchSetup` splices the whole range.
- A group without a `Lead` behaves exactly as the single setup did before.

### 2.2 Where a pair stops

| Stops at | Why |
|---|---|
| An instruction that reads or writes the `AUIPC`'s register | The register dependency check now covers both instructions of the group. This keeps the pair below `sd ra` and below an earlier call. |
| A terminator, if the register is live into any successor of its block | Above the terminator the `AUIPC` also runs on the paths that skip the call, which may still read the value it overwrites. Taken from the successors' live-in lists; if the function does not track liveness, any successor counts as reading it. |
| A `frame-setup` instruction | The unwind information says a callee-saved register still holds the caller's value until the CFI directives after the prologue's saves. |
| A call, inline asm, indirect branch | As for every setup. A pair therefore never leaves a loop that contains its own call. |

Nothing is inserted between an `AUIPC` and its `BMOVT_I` by another setup's hoist: a second pair on the same register stops below the first pair's `BMOVT_I`, which reads that register.

### 2.3 A pair is not left in a block hotter than its call

- The first version used the limit every setup has, `-riscv-ns-hoist-max-freq-ratio` (2).
- MiBench showed that costs instructions (section 7.5), so pairs now have their own limit, `-riscv-ns-hoist-call-target-max-freq-ratio`, **default 1**: a pair is hoisted only into blocks that run no more often than the coolest point it has reached.
- Within its own block a pair still moves freely.

### 2.4 Options

| Option | Default | Effect |
|---|---|---|
| `-riscv-ns-disable-call-target-hoisting` | off | No `Lead` is ever found, so the `BMOVT_I` stays next to its call as before. |
| `-riscv-ns-hoist-call-target-max-freq-ratio=N` | 1 | The frequency limit for pairs. 2 is the first version; 0 is no limit. |

## 3. Layer 1: nothing changes with the switch off

`scratch/auipc-bmovt/eq.sh` compiles the 479 `llvm/test/CodeGen/RISCV/*.ll` files that contain a call or a branch, with `-mtriple=riscv64 -mattr=+m,+a,+f,+d,+c`, three ways: an unmodified `llc` built from the same tree with the change stashed, the new `llc` with `-riscv-ns-disable-call-target-hoisting`, and the new `llc` as it is.

| | `-O2` | `-O0` |
|---|---|---|
| Files that compile (same set, same exit statuses, in all three) | 352 | 365 |
| Switch off, byte-identical to the unmodified compiler | 352 of 352 | 365 of 365 |
| Changed with the default | 134 | 160 |

The files that do not compile fail in the unmodified compiler too (the fork does not support what they use).

## 4. Layer 2: tests

New file `llvm/test/CodeGen/RISCV/branch-setup-hoisting-call-target.mir`, running `riscv-branch-setup-hoisting` alone. The functions that test the safety rules run with the limit at 2, so that the frequency rule does not hide them; one function checks the default limit, and one the switch.

| Function | Checks |
|---|---|
| `call_in_block` | The `AUIPC` and `BMOVT_I` move together to just below the `BMOVS_J`, above two unrelated instructions. With the switch: they stay directly above the `PseudoPBCALL`. |
| `scratch_in_use` | A tail call on `$x6`: the pair stops below the last instruction that reads or writes `$x6`. |
| `across_blocks` | With the limit at 2, the pair goes to the block that branches around the call. By default it stays at the top of the call's block. |
| `scratch_live_on_other_path` | The path that skips the call reads `$x1`, so the pair does not move above the branch to it, even with the limit at 2. |
| `below_prologue` | The pair lands after `frame-setup CFI_INSTRUCTION offset $x1, -8`, not between the `sd ra` and it. |

As in the other hoisting tests, it also runs the pass twice (same result) and runs `greedy,virtregrewriter` after it.

**Result:** the 20 existing `branch-setup-*`, `ns-*` and `non-spec-*` tests, the new test, and clang's `non-spec-hwloop-call-spill.c` pass: 22 of 22.

**Not done:** mutation testing of the new test, which the earlier changes had.

## 5. Layer 3: compiled output

On the files of section 3, compiled with the default:

| | `-O2` | `-O0` |
|---|---|---|
| Call `auipc`s | 1,201 | 1,573 |
| Not immediately followed by their `bmovt` | 0 | 0 |
| Hoisted away from their `pb` | 779 | 1,056 |
| Hoisted away with the limit at 2 (first version) | 918 | 1,262 |

- **Only order changes.** With the first version, every `.s` file was compared with the unmodified compiler's as a multiset of lines: all 350 that then compiled at `-O2`, and all at `-O0`, hold the same instructions with the same registers. The default hoists a subset of what that version hoisted.
- **Machine verifier.** With `-verify-machineinstrs`, the new compiler and the unmodified one fail on the same files (`lpad`, `miss-sp-restore-eh` at `-O2`; those and `compress-opt-branch`, `dwarf-eh`, `frameaddr-returnaddr` at `-O0`), with errors that predate this change. No file fails only with the change.
- **Relocations.** In an object file, each hoisted `auipc` still carries `R_RISCV_CALL_PLT` and is directly followed by its `bmovt`.

## 6. Layer 4: gem5, the gold-verified suite

`python3 -m bench run` in `~/drive1/verif`, which builds each workload with `-DVERIFY` and compares what it prints with the vanilla build's.

| Compiler | Result |
|---|---|
| Limit 2 (first version) and switch off, verify and perf | 144 runs: 72 pass, 72 ran |
| Final default, verify | 36 of 36 pass |

## 7. Layer 4: gem5, MiBench

### 7.1 Setup

- `~/drive1/verif` at `682db68`, `mibench/run.py`, gem5 `~/gem5/pbtb` at `5a35810`, the default machine config (TAGE, every PBTB fix on).
- Four builds, results under the label `auipc-bmovt`:

  | Name here | Config | Compiler |
  |---|---|---|
  | vanilla | `vanilla` | `clang-21 -O2` |
  | off | `no-call-hoist` | `clang-dev -O2 -DPBTB -mllvm -riscv-ns-disable-call-target-hoisting` |
  | limit 2 | `current`, as it was when run | `clang-dev -O2 -DPBTB`, when the limit's default was still 2 |
  | limit 1 | `call-hoist-ratio1` | `clang-dev -O2 -DPBTB -mllvm -riscv-ns-hoist-call-target-max-freq-ratio=1` |

- "Off" is the same compiler binary as the other two, so the comparison isolates this change.
- **The final default was not simulated again.** After the default became 1, `current` was rebuilt and compared with the limit-1 binaries: 35 of 49 are byte-identical. The other 14 (the tiff programs, ispell, fft) have the same instructions and differ in data addresses only, because the config name is part of build paths that end up in strings.
- **musl was not rebuilt.** The programs link the prebuilt `~/bin/musl-dev`, so only the benchmarks' own code has hoisted pairs. `setup-placement.md` rebuilt musl with each compiler; this did not.
- Each run was stopped after one hour of wall time by a watchdog (`scratch/auipc-bmovt/mibench-run.sh`), since the runner has no time limit.

**Pairs in the 49 linked binaries** (36,108 call pairs each way, musl's included):

| | Hoisted away from their `pb` |
|---|---|
| off | 49 |
| limit 1 | 12,085 |
| limit 2 | 12,697 |

Limit 1 keeps 95% of what limit 2 hoists. Instruction counts per binary are the same in all three.

### 7.2 What ran

Of the 66 workloads, 56 were attempted and 37 exit with status 0 under all three non-spec builds. The rest:

| Workloads | What happens | Same with the switch off? |
|---|---|---|
| `lame`, `ghostscript`, `pgp` | The non-spec compiler crashes, in pseudo expansion, the asm printer and PHI elimination | yes |
| `mad` | Does not build, vanilla included | yes |
| `sphinx` | Left out: over an hour in earlier runs | - |
| `basicmath`, `jpeg` encode | gem5 aborts (`stalls[tid].iew` assertion; "pb left dispatch early but its outcome changed") | yes |
| `typeset` | Completes under no build, vanilla included (section 7.9) | no |
| `patricia` large | Stopped at one hour | yes |
| `tiff2bw`, `tiffdither`, `tiffmedian`, `ispell`, `patricia` small, `blowfish` | Exit with a nonzero status under every build, vanilla included | yes |
| `fft` large inverse | gem5's skid-buffer assertion with the switch off and with limit 1; completed with limit 2 | - |

### 7.3 Outputs

- **Against off:** for all 37 workloads, stdout and every file the program wrote are byte-identical with limit 2 and with limit 1.
- **Against vanilla:** identical in 30. `bitcount` differs only in the times it prints. `sha` and `fft` differ, as they do with the switch off (section 9).

### 7.4 Cycles and committed instructions, against off

| Workload | Cycles, off | Cycles, limit 2 | Cycles, limit 1 | Instructions, limit 2 | Instructions, limit 1 |
|---|---|---|---|---|---|
| `automotive-bitcount-large` | 260,997,757 | -0.00% | -0.01% | 0.00% | 0.00% |
| `automotive-bitcount-small` | 16,956,010 | -0.01% | -0.01% | 0.00% | 0.00% |
| `automotive-qsort-large` | 437,337,812 | -2.38% | -0.00% | 0.00% | 0.00% |
| `automotive-qsort-small` | 50,874,676 | +0.42% | +0.00% | 0.00% | 0.00% |
| `automotive-susan-large-corners` | 16,226,182 | -0.11% | -0.06% | 0.00% | 0.00% |
| `automotive-susan-large-edges` | 34,676,857 | +0.99% | +0.00% | 0.00% | 0.00% |
| `automotive-susan-large-smoothing` | 82,567,410 | **+27.73%** | -0.00% | 0.00% | 0.00% |
| `automotive-susan-small-corners` | 624,028 | -1.50% | -0.00% | 0.00% | 0.00% |
| `automotive-susan-small-edges` | 1,028,474 | +1.36% | +0.02% | 0.00% | 0.00% |
| `automotive-susan-small-smoothing` | 5,457,531 | **+27.38%** | -0.02% | 0.00% | 0.00% |
| `consumer-jpeg-large-decode` | 8,320,161 | +0.75% | +0.01% | +0.01% | 0.00% |
| `consumer-jpeg-small-decode` | 2,456,228 | +0.54% | +0.38% | +0.02% | 0.00% |
| `consumer-tiff2rgba-large` | 26,346 | -0.79% | -0.91% | -3.45% | -0.61% |
| `consumer-tiff2rgba-small` | 26,389 | -2.03% | -0.80% | -4.07% | +0.01% |
| `network-dijkstra-large` | 276,876,200 | **+3.46%** | +0.03% | **+5.90%** | 0.00% |
| `network-dijkstra-small` | 59,540,359 | **+3.43%** | -0.03% | **+5.37%** | 0.00% |
| `office-rsynth-large` | 425,053,006 | -0.05% | -0.04% | +0.02% | 0.00% |
| `office-rsynth-small` | 29,859,612 | -0.05% | -0.04% | +0.02% | 0.00% |
| `office-stringsearch-large` | 7,484,243 | +4.54% | +4.54% | 0.00% | 0.00% |
| `office-stringsearch-small` | 330,150 | -4.51% | -4.51% | 0.00% | 0.00% |
| `security-rijndael-large` | 129,077,811 | -0.16% | -0.16% | 0.00% | 0.00% |
| `security-rijndael-small` | 12,413,358 | -0.16% | -0.16% | 0.00% | 0.00% |
| `security-sha-large` | 36,990,488 | -2.60% | -2.60% | 0.00% | 0.00% |
| `security-sha-small` | 5,418,001 | -17.78% | -17.78% | 0.00% | 0.00% |
| `telecomm-adpcm-decode-large` | 465,427,997 | -0.00% | +0.00% | 0.00% | 0.00% |
| `telecomm-adpcm-decode-small` | 24,902,759 | -0.01% | +0.00% | 0.00% | 0.00% |
| `telecomm-adpcm-encode-large` | 653,485,602 | -0.00% | +0.00% | 0.00% | 0.00% |
| `telecomm-adpcm-encode-small` | 33,818,749 | -0.01% | -0.00% | 0.00% | 0.00% |
| `telecomm-crc32-large` | 828,863,004 | -0.00% | -0.00% | 0.00% | 0.00% |
| `telecomm-crc32-small` | 828,863,004 | -0.00% | -0.00% | 0.00% | 0.00% |
| `telecomm-fft-large-forward` | 400,570,012 | -0.24% | -0.02% | +0.09% | 0.00% |
| `telecomm-fft-small-forward` | 38,037,952 | -0.17% | -0.00% | +0.06% | 0.00% |
| `telecomm-fft-small-inverse` | 83,423,956 | -0.15% | +0.02% | +0.06% | 0.00% |
| `telecomm-gsm-decode-large` | 31,751 | +0.00% | +0.24% | +0.03% | 0.00% |
| `telecomm-gsm-decode-small` | 31,751 | +0.00% | +0.24% | +0.03% | 0.00% |
| `telecomm-gsm-encode-large` | 708,159,363 | -0.27% | -0.02% | +0.03% | 0.00% |
| `telecomm-gsm-encode-small` | 13,120,562 | -0.29% | +0.01% | +0.02% | 0.00% |

`tiff2rgba` and `gsm` decode run about 30,000 cycles and say nothing. `crc32` small and large are the same program on the same input.

**Summary:**

| | Limit 2 | Limit 1 (default) |
|---|---|---|
| Cycles, geomean, 37 workloads | +0.78% | -0.64% |
| Cycles, geomean, without `sha` (35) | +1.46% | -0.04% |
| Cycles, total | +0.35% | -0.04% |
| Committed instructions, total | +0.22% | 0.00% |
| Workloads slower by more than 1% | 6 | 1 |
| Workloads faster by more than 1% | 6 | 3 |

- `sha`'s cycles are not reliable (section 7.8), so the rows without it are the ones to read.
- **Limit 2 is a regression:** it executes more instructions and is slower overall.
- **Limit 1 is neutral on these programs:** no extra instructions, and cycles within noise of off apart from `stringsearch`, whose two inputs move by the same amount in opposite directions (section 7.7).

### 7.5 `dijkstra`: the pair runs where the call does not

- With limit 2, `dijkstra` commits 5.4% (small) and 5.9% (large) more instructions and takes 3.4% more cycles; mispredicts in the small run go from 112,873 to 126,878.
- The relax loop calls `enqueue` only when a distance improves. The pair for that call was hoisted above the branches that guard it, so its two instructions ran on every iteration that skips the call.
- This is the frequency limit of 2 doing what it allows: a block that runs up to twice as often as the call is accepted. For a one-instruction setup that was judged acceptable; a pair is two instructions, one of which is only there to feed the other.
- With limit 1: instructions identical to off, cycles +0.03% and -0.03%.

### 7.6 `susan` smoothing: an 8-byte shift

- +27% cycles with limit 2, with committed instructions and mispredicts unchanged.
- `fetch.cacheLines` rises by 1,508,225 in the small run, and cycles by 1,494,259.
- The hot loop of `susan_smoothing` is 38 bytes. A pair hoisted earlier in the function moved everything after it by 8 bytes: the loop went from `0x16104..0x1612a` to `0x1610c..0x16132`. gem5's fetch buffer is 16 bytes (`fetchBufferSize=16`), so the loop went from three buffers to four.
- So this is code placement, the noise `setup-placement.md` (section 6) warns about, at its worst. With limit 1 that pair does not move, and the run is back to -0.02%. Another program could land the other way.

### 7.7 `stringsearch`

- +4.54% (large) and -4.51% (small), the same with both limits, with identical instruction counts.
- `fetch.cacheLines` moves with it: +12.2% in the large run, -11.6% in the small one. Placement again.

### 7.8 `sha`

- -17.8% (small) and -2.6% (large): the same 963,000 cycles in both, with identical instruction counts, fetch buffers and mispredicts.
- `MemDepUnit__0.conflictingLoads` goes from 1,185,968 to 640,330 in the small run. This is the memory-dependence predictor varying with timing, for which `setup-placement.md` already sets `sha`'s cycles aside.

### 7.9 `typeset`

It completes under no build, so it is not in the 37, and what it shows is not attributable:

| Build | small | large |
|---|---|---|
| vanilla | gem5 `stalls[tid].iew` assertion | gem5 skid-buffer assertion |
| off | page fault at address `0x3` | reaches `renameat2`, which gem5 does not implement |
| limit 2 | reaches `renameat2` | reaches `renameat2` |

Stdout up to the point each stops is identical between off and limit 2.

### 7.10 `qsort` large: what limit 1 gives up

`qsort` large was 2.38% faster with limit 2 and is unchanged with limit 1.

- Only three pairs in `qsort_large.c` differ between the two: one for `fopen`, and those of the second and third `fscanf` in the read loop's condition. With limit 2 they sit above the branch that leaves the loop when the previous `fscanf` fails.
- Instruction counts are identical in all three builds:

  | | off | limit 1 | limit 2 |
  |---|---|---|---|
  | Cycles | 437,337,812 | 437,317,173 | 426,934,245 |
  | Mispredicts | 5,213,365 | 5,209,408 | 4,925,036 |
  | of which from an exhausted B-register | 565,458 | 565,257 | 244,455 |
  | Cycles stalled in dispatch on a dependence | 214,924,572 | 214,925,686 | 205,314,216 |

- Why two call setups save ten million cycles was not traced.
- To get it back: `-mllvm -riscv-ns-hoist-call-target-max-freq-ratio=2` for this program. Used everywhere, it brings back sections 7.5 and 7.6.
- **Untested idea:** the limit is an integer, so 1 rejects a block that runs even slightly more often than the call. `qsort`'s branch is a loop exit and `dijkstra`'s is an ordinary `if`, so a fractional limit between 1 and 2 might accept the first and reject the second. The block frequencies the pass sees in the two programs were not looked at.

### 7.11 Does the target arrive earlier?

Summed over the 37 workloads:

| | off | limit 1 | limit 2 |
|---|---|---|---|
| `pbtb.predictPbTargetNotFound` | 63,571,030 | 9,574,679 (-85%) | 8,383,729 (-87%) |
| `pbtb.resolvePbTgtNotFoundAtPred` | 7,873,913 | 7,117,672 (-9.6%) | 5,852,322 (-25.7%) |
| ...of which mispredicted | 1,008,680 | 1,006,261 (-0.2%) | 968,625 (-4.0%) |
| `commit.branchMispredicts` | 16,474,586 | 16,527,486 (+0.3%) | 16,222,835 (-1.5%) |
| `pbtb.NumCyclesStalledInDispatchDueToDependence` | 3,423,506,324 | +0.05% | +0.07% |
| Cycles | 5,979,355,551 | -0.04% | +0.35% |

- The change does what it is for: predictions made with no target known fall by 85%.
- That does not turn into cycles on these programs. Mispredicts and dispatch stalls barely move, which suggests a call's target was seldom what its `pb` waited for. This was not investigated further.

## 8. What did not go as planned

- **The first MiBench run used stale binaries.** `mibench/run.py` stamps a make-based workload with its command line and sources, not with the compiler, so after `clang-dev` was rebuilt, the `current` binaries that already existed were reported up to date (39 of the 198 in that build, vanilla's included). It showed as `current` binaries with no hoisted pairs for most small inputs, while the large inputs of the same programs, never built before, had hundreds. The run was stopped, both non-spec configs were rebuilt with `--force` (112 of 112 compiled), every `current` binary was checked for hoisted pairs, and everything was run again. Results taken earlier, after a compiler rebuild without `--force`, may have the same problem.
- **The first version was a regression** (sections 7.4 and 7.5). The frequency limit of section 2.3 came out of the MiBench run, not out of the design.
- **The first reading of `sha` was wrong.** Its 18% was first counted as a gain of the change; it is the predictor effect of section 7.8.
- **`across_blocks` in the new test** first checked the instruction after the pair, which differs when the pass runs twice (an unrelated `BMOVC` moves on the second run). The line was dropped.

## 9. Found along the way, not caused by this change

- **Non-spec `sha` and `fft` print different output from vanilla**: a different digest, and truncated FFT output. The output is byte-identical with the switch off and to the non-spec output in the `new-gem5` and `vanilla-musl` results of 2026-11-11. `setup-placement.md` notes the `sha` difference.
- **The non-spec compiler crashes** on `lame` (`fft.c`, asm printer), `ghostscript` (`gscie.c`, PHI elimination) and `pgp` (`fileio.c`, pseudo expansion).
- **gem5 aborts** on non-spec `basicmath` and `jpeg` encode, and hits its own assertions on some long runs, vanilla's `typeset` among them.
- **The MiBench runner does not see a new compiler** (section 8).

## 10. Not covered

- musl built with the change (section 7.1). Most calls a program executes are in libc.
- The final default simulated as `current`: its numbers are the limit-1 run's, on the strength of the binary comparison in section 7.1.
- Mutation testing of the new test.
- The workloads of section 7.2 that do not build, do not finish, or abort.
- Why hoisted targets do not shorten dispatch stalls (section 7.11), and the fractional limit (section 7.10).

## 11. Files

- `llvm/lib/Target/RISCV/RISCVBranchSetupHoisting.cpp`
- `llvm/test/CodeGen/RISCV/branch-setup-hoisting-call-target.mir`
- `ns-docs/call-target-hoisting.md`
- Outside the tree:
  - `~/.config/bench/configs.toml` gained `[configs.no-call-hoist]`.
  - `~/drive1/verif/results/auipc-bmovt/` holds the MiBench runs; `current` there is limit 2, and `call-hoist-ratio1` no longer has a config.
  - `~/drive1/verif/results/auipc-bmovt-suite/` holds the gold-verified suite.
- Scripts and logs in `scratch/auipc-bmovt/`: `eq.sh` (section 3), `mibench-run.sh` and `mibench-run2.sh` (the runs), `compare.py` (sections 7.3 and 7.4), `pairs.py` (pair counts).
