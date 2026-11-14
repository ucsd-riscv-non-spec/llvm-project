// RUN: %clang_cc1 -triple riscv64 -std=c99 -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64 -std=c99 -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

// Target properties are emitted without debug information or generic hints.
void all_options(int *a, int n) {
// CHECK-LABEL: define {{.*}} @all_options(
#pragma clang loop riscv_ns_swp(enable) riscv_ns_ir_fission(enable)
#pragma clang loop riscv_ns_machine_fission(enable) riscv_ns_force(enable)
#pragma clang loop riscv_ns_swp_speculative(enable)
#pragma clang loop riscv_ns_swp_shift_recurrences(enable) riscv_ns_swp_toggles(enable)
#pragma clang loop riscv_ns_swp_distance(8) riscv_ns_swp_speculative_distance(3)
#pragma clang loop riscv_ns_swp_max_slice_size(17) riscv_ns_swp_max_loop_size(257)
#pragma clang loop riscv_ns_swp_max_prologue_size(513) riscv_ns_fission_queue_depth(16)
  for (int i = 0; i < n; ++i)
    a[i]++;
// CHECK: br label {{.*}}, !llvm.loop ![[ALL:[0-9]+]]
}

void explicit_disable(int *a, int n) {
// CHECK-LABEL: define {{.*}} @explicit_disable(
#pragma clang loop riscv_ns_swp(disable) riscv_ns_ir_fission(disable)
#pragma clang loop riscv_ns_machine_fission(disable) riscv_ns_force(disable)
#pragma clang loop riscv_ns_swp_speculative(disable)
#pragma clang loop riscv_ns_swp_shift_recurrences(disable) riscv_ns_swp_toggles(disable)
  while (n > 0)
    a[--n]++;
// CHECK: br label {{.*}}, !llvm.loop ![[DISABLED:[0-9]+]]
}

// Numeric controls do not implicitly enable any pass, feature, or force.
void numeric_only(int *a, int n) {
// CHECK-LABEL: define {{.*}} @numeric_only(
#pragma clang loop riscv_ns_swp_distance(62) riscv_ns_swp_speculative_distance(1)
#pragma clang loop riscv_ns_swp_max_slice_size(2147483647)
#pragma clang loop riscv_ns_swp_max_loop_size(1) riscv_ns_swp_max_prologue_size(1)
#pragma clang loop riscv_ns_fission_queue_depth(64)
  do {
    a[n]++;
  } while (--n > 0);
// CHECK: br i1 {{.*}}, !llvm.loop ![[NUMERIC:[0-9]+]]
}

void nested(int *a, int n) {
// CHECK-LABEL: define {{.*}} @nested(
#pragma clang loop riscv_ns_swp_distance(9)
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j)
      a[i * n + j]++;
// CHECK: br label %for.cond{{[0-9.]*}}{{$}}
  }
// CHECK: br label {{.*}}, !llvm.loop ![[OUTER:[0-9]+]]
  for (int i = 0; i < n; ++i)
    a[i]++;
// CHECK: br label %for.cond{{[0-9.]*}}{{$}}
}

void annotated_inner(int *a, int n) {
// CHECK-LABEL: define {{.*}} @annotated_inner(
  for (int i = 0; i < n; ++i) {
#pragma clang loop riscv_ns_swp_toggles(disable)
    for (int j = 0; j < n; ++j)
      a[i * n + j]++;
// CHECK: br label {{.*}}, !llvm.loop ![[INNER:[0-9]+]]
  }
// CHECK: br label %for.cond{{[0-9.]*}}{{$}}
}

void followups(int *a, int n) {
// CHECK-LABEL: define {{.*}} @followups(
#pragma clang loop distribute(enable) vectorize(enable) unroll_count(4)
#pragma clang loop riscv_ns_force(disable) riscv_ns_swp_distance(5)
  for (int i = 0; i < n; ++i)
    a[i]++;
// CHECK: br label {{.*}}, !llvm.loop ![[FOLLOWUPS:[0-9]+]]
}

void jam_inner(int *a, int n) {
// CHECK-LABEL: define {{.*}} @jam_inner(
#pragma unroll_and_jam(2)
  for (int i = 0; i < n; ++i) {
#pragma clang loop riscv_ns_fission_queue_depth(9)
    for (int j = 0; j < n; ++j)
      a[i * n + j]++;
// CHECK: br label {{.*}}, !llvm.loop ![[JAM_INNER:[0-9]+]]
  }
// CHECK: br label {{.*}}, !llvm.loop ![[JAM_OUTER:[0-9]+]]
}

// CHECK-DAG: ![[ALL]] = distinct !{![[ALL]], ![[SWP:[0-9]+]], ![[IR:[0-9]+]], ![[MACHINE:[0-9]+]], ![[FORCE:[0-9]+]], ![[SPEC:[0-9]+]], ![[SHIFT:[0-9]+]], ![[TOGGLE:[0-9]+]], ![[DIST:[0-9]+]], ![[SPECDIST:[0-9]+]], ![[SLICE:[0-9]+]], ![[SIZE:[0-9]+]], ![[PROLOGUE:[0-9]+]], ![[DEPTH:[0-9]+]]}
// CHECK-DAG: ![[SWP]] = !{!"llvm.loop.riscv.ns.swp.enable", i1 true}
// CHECK-DAG: ![[IR]] = !{!"llvm.loop.riscv.ns.fission.ir.enable", i1 true}
// CHECK-DAG: ![[MACHINE]] = !{!"llvm.loop.riscv.ns.fission.machine.enable", i1 true}
// CHECK-DAG: ![[FORCE]] = !{!"llvm.loop.riscv.ns.force", i1 true}
// CHECK-DAG: ![[SPEC]] = !{!"llvm.loop.riscv.ns.swp.speculative", i1 true}
// CHECK-DAG: ![[SHIFT]] = !{!"llvm.loop.riscv.ns.swp.shift_recurrences", i1 true}
// CHECK-DAG: ![[TOGGLE]] = !{!"llvm.loop.riscv.ns.swp.toggles", i1 true}
// CHECK-DAG: ![[DIST]] = !{!"llvm.loop.riscv.ns.swp.distance", i32 8}
// CHECK-DAG: ![[SPECDIST]] = !{!"llvm.loop.riscv.ns.swp.speculative_distance", i32 3}
// CHECK-DAG: ![[SLICE]] = !{!"llvm.loop.riscv.ns.swp.max_slice_size", i32 17}
// CHECK-DAG: ![[SIZE]] = !{!"llvm.loop.riscv.ns.swp.max_loop_size", i32 257}
// CHECK-DAG: ![[PROLOGUE]] = !{!"llvm.loop.riscv.ns.swp.max_prologue_size", i32 513}
// CHECK-DAG: ![[DEPTH]] = !{!"llvm.loop.riscv.ns.fission.queue_depth", i32 16}

// CHECK-DAG: ![[DISABLED]] = distinct !{![[DISABLED]], ![[SWP_OFF:[0-9]+]], ![[IR_OFF:[0-9]+]], ![[MACHINE_OFF:[0-9]+]], ![[FORCE_OFF:[0-9]+]], ![[SPEC_OFF:[0-9]+]], ![[SHIFT_OFF:[0-9]+]], ![[TOGGLE_OFF:[0-9]+]]}
// CHECK-DAG: ![[SWP_OFF]] = !{!"llvm.loop.riscv.ns.swp.enable", i1 false}
// CHECK-DAG: ![[IR_OFF]] = !{!"llvm.loop.riscv.ns.fission.ir.enable", i1 false}
// CHECK-DAG: ![[MACHINE_OFF]] = !{!"llvm.loop.riscv.ns.fission.machine.enable", i1 false}
// CHECK-DAG: ![[FORCE_OFF]] = !{!"llvm.loop.riscv.ns.force", i1 false}
// CHECK-DAG: ![[SPEC_OFF]] = !{!"llvm.loop.riscv.ns.swp.speculative", i1 false}
// CHECK-DAG: ![[SHIFT_OFF]] = !{!"llvm.loop.riscv.ns.swp.shift_recurrences", i1 false}
// CHECK-DAG: ![[TOGGLE_OFF]] = !{!"llvm.loop.riscv.ns.swp.toggles", i1 false}

// CHECK-DAG: ![[NUMERIC]] = distinct !{![[NUMERIC]], ![[MAXDIST:[0-9]+]], ![[MINSPEC:[0-9]+]], ![[MAXSLICE:[0-9]+]], ![[MINSIZE:[0-9]+]], ![[MINPROLOGUE:[0-9]+]], ![[MAXDEPTH:[0-9]+]]}
// CHECK-DAG: ![[MAXDIST]] = !{!"llvm.loop.riscv.ns.swp.distance", i32 62}
// CHECK-DAG: ![[MINSPEC]] = !{!"llvm.loop.riscv.ns.swp.speculative_distance", i32 1}
// CHECK-DAG: ![[MAXSLICE]] = !{!"llvm.loop.riscv.ns.swp.max_slice_size", i32 2147483647}
// CHECK-DAG: ![[MINSIZE]] = !{!"llvm.loop.riscv.ns.swp.max_loop_size", i32 1}
// CHECK-DAG: ![[MINPROLOGUE]] = !{!"llvm.loop.riscv.ns.swp.max_prologue_size", i32 1}
// CHECK-DAG: ![[MAXDEPTH]] = !{!"llvm.loop.riscv.ns.fission.queue_depth", i32 64}
// CHECK-DAG: ![[OUTER]] = distinct !{![[OUTER]], ![[OUTERDIST:[0-9]+]]}
// CHECK-DAG: ![[OUTERDIST]] = !{!"llvm.loop.riscv.ns.swp.distance", i32 9}
// CHECK-DAG: ![[INNER]] = distinct !{![[INNER]], ![[TOGGLE_OFF]]}

// CHECK-DAG: ![[FOLLOWUPS]] = distinct !{![[FOLLOWUPS]], ![[FORCE_OFF]], ![[FIVEDIST:[0-9]+]], ![[DISTRIBUTE:[0-9]+]], ![[DFOLLOW:[0-9]+]]}
// CHECK-DAG: ![[FIVEDIST]] = !{!"llvm.loop.riscv.ns.swp.distance", i32 5}
// CHECK-DAG: ![[DISTRIBUTE]] = !{!"llvm.loop.distribute.enable", i1 true}
// CHECK-DAG: ![[DFOLLOW]] = !{!"llvm.loop.distribute.followup_all", ![[FORCE_OFF]], ![[FIVEDIST]], ![[VECTORIZE:[0-9]+]], ![[VFOLLOW:[0-9]+]]}
// CHECK-DAG: ![[VECTORIZE]] = !{!"llvm.loop.vectorize.enable", i1 true}
// CHECK-DAG: ![[VFOLLOW]] = !{!"llvm.loop.vectorize.followup_all", ![[FORCE_OFF]], ![[FIVEDIST]], ![[VECTORIZED:[0-9]+]], ![[UNROLL:[0-9]+]], ![[UFOLLOW:[0-9]+]]}
// CHECK-DAG: ![[VECTORIZED]] = !{!"llvm.loop.isvectorized"}
// CHECK-DAG: ![[UNROLL]] = !{!"llvm.loop.unroll.count", i32 4}
// CHECK-DAG: ![[UFOLLOW]] = !{!"llvm.loop.unroll.followup_all", ![[FORCE_OFF]], ![[FIVEDIST]], ![[VECTORIZED]], ![[UNROLLED:[0-9]+]]}
// CHECK-DAG: ![[UNROLLED]] = !{!"llvm.loop.unroll.disable"}

// CHECK-DAG: ![[JAM_INNER]] = distinct !{![[JAM_INNER]], ![[JAM_DEPTH:[0-9]+]]}
// CHECK-DAG: ![[JAM_DEPTH]] = !{!"llvm.loop.riscv.ns.fission.queue_depth", i32 9}
// CHECK-DAG: ![[JAM_OUTER]] = distinct !{![[JAM_OUTER]], ![[JAM_COUNT:[0-9]+]], ![[JAM_FOLLOW:[0-9]+]]}
// CHECK-DAG: ![[JAM_COUNT]] = !{!"llvm.loop.unroll_and_jam.count", i32 2}
// CHECK-DAG: ![[JAM_FOLLOW]] = !{!"llvm.loop.unroll_and_jam.followup_inner", ![[JAM_DEPTH]]}
