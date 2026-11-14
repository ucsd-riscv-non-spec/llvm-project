// RUN: %clang_cc1 -triple riscv64 -target-feature +m -O2 -S -mllvm -riscv-ns-swp-distance=0 -mllvm -stop-after=riscv-branch-condition-pipelining -o - %s | FileCheck %s --check-prefix=IR
// RUN: %clang_cc1 -triple riscv64 -target-feature +m -O2 -S -mllvm -riscv-ns-swp-distance=0 -Rpass-analysis=riscv-branch-condition-pipelining -Rpass=riscv-branch-condition-pipelining -Rpass-missed=riscv-branch-condition-pipelining -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=REMARK
// RUN: %clang_cc1 -triple riscv64 -target-feature +m -O2 -S -mllvm -riscv-ns-disable-software-pipelining -mllvm -stop-after=riscv-branch-condition-pipelining -o - %s | FileCheck %s --check-prefix=OFF

// A local positive distance overrides the legacy global zero distance.
void pipelined(const int *restrict a, int *restrict out, long n) {
#pragma clang loop vectorize(disable) unroll(disable)
#pragma clang loop riscv_ns_swp(enable) riscv_ns_swp_distance(3)
#pragma clang loop riscv_ns_ir_fission(disable) riscv_ns_machine_fission(disable)
  for (long i = 0; i < n; ++i)
    if (a[i] > 0)
      out[i] = a[i];
}

// Numeric values and force do not turn a disabled pass back on.
void disabled(const int *restrict a, int *restrict out, long n) {
#pragma clang loop vectorize(disable) unroll(disable)
#pragma clang loop riscv_ns_swp(disable) riscv_ns_swp_distance(8) riscv_ns_force(enable)
#pragma clang loop riscv_ns_ir_fission(disable) riscv_ns_machine_fission(disable)
  for (long i = 0; i < n; ++i)
    if (a[i] > 0)
      out[i] = a[i];
}

// Force bypasses even explicitly supplied local size limits.
void forced(const int *a, int *out, long n) {
#pragma clang loop vectorize(disable) unroll(disable)
#pragma clang loop riscv_ns_swp(enable) riscv_ns_swp_distance(5) riscv_ns_force(enable)
#pragma clang loop riscv_ns_swp_max_slice_size(1) riscv_ns_swp_max_loop_size(1)
#pragma clang loop riscv_ns_swp_max_prologue_size(1)
#pragma clang loop riscv_ns_ir_fission(disable) riscv_ns_machine_fission(disable)
  for (long i = 0; i < n; ++i)
    if (a[i] > 0)
      out[i] = a[i];
}

// IR-LABEL: define {{.*}} @pipelined(
// IR: icmp ugt i64 {{.*}}, 3
// IR: call void @llvm.riscv.ns.cond.init
// IR: call void @llvm.riscv.ns.cond.push
// IR: call i1 @llvm.riscv.ns.cond.pop
// IR-LABEL: define {{.*}} @disabled(
// IR-NOT: call {{.*}}@llvm.riscv.ns.cond
// IR: ret void
// IR-LABEL: define {{.*}} @forced(
// IR: icmp ugt i64 {{.*}}, 5
// IR: call void @llvm.riscv.ns.cond.init
// IR: call i1 @llvm.riscv.ns.cond.pop

// REMARK: Non-Spec settings for function pipelined, loop {{.*}}distance=3 (metadata)
// REMARK: requested distance 3, effective distance 3, force=false
// REMARK: Non-Spec settings for function disabled, loop {{.*}}swp=0 (metadata){{.*}}force=1 (metadata)
// REMARK: loop software pipelining disable or inherited zero distance
// REMARK: Non-Spec settings for function forced, loop {{.*}}distance=5 (metadata){{.*}}max_slice_size=1 (metadata){{.*}}max_loop_size=1 (metadata){{.*}}max_prologue_size=1 (metadata){{.*}}force=1 (metadata)
// REMARK: force active for loop
// REMARK: requested distance 5, effective distance 5, force=true

// OFF-NOT: call {{.*}}@llvm.riscv.ns.cond
