// RUN: %clang_cc1 -triple riscv64 -std=c++14 -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple riscv64 -std=c++14 -emit-pch -o %t %s
// RUN: %clang_cc1 -triple riscv64 -std=c++14 -include-pch %t -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple riscv64 -std=c++14 -include-pch %t -ast-print -o - %s | FileCheck %s --check-prefix=PRINT

#ifndef PRAGMA_LOOP_RISCV_NS_H
#define PRAGMA_LOOP_RISCV_NS_H

void range_loop(int (&a)[16]) {
// CHECK-LABEL: define {{.*}} @_Z10range_loop
#pragma clang loop riscv_ns_swp(enable) riscv_ns_ir_fission(disable)
#pragma clang loop riscv_ns_machine_fission(disable) riscv_ns_force(disable)
#pragma clang loop riscv_ns_swp_speculative(enable)
#pragma clang loop riscv_ns_swp_shift_recurrences(enable) riscv_ns_swp_toggles(disable)
#pragma clang loop riscv_ns_swp_distance(4) riscv_ns_swp_speculative_distance(2)
#pragma clang loop riscv_ns_swp_max_slice_size(8) riscv_ns_swp_max_loop_size(32)
#pragma clang loop riscv_ns_swp_max_prologue_size(64) riscv_ns_fission_queue_depth(16)
  for (int &value : a)
    value++;
// CHECK: br label {{.*}}, !llvm.loop ![[RANGE:[0-9]+]]
}

template <int Distance, int SpeculativeDistance, int Size, int Depth>
void loop_template(int *a, int n) {
#pragma clang loop riscv_ns_swp_distance(Distance) riscv_ns_swp_speculative_distance(SpeculativeDistance)
#pragma clang loop riscv_ns_swp_max_slice_size(Size) riscv_ns_swp_max_loop_size(Size)
#pragma clang loop riscv_ns_swp_max_prologue_size(Size) riscv_ns_fission_queue_depth(Depth)
  for (int i = 0; i < n; ++i)
    a[i]++;
}

template <int N> struct DistanceTag { enum { value = N }; };

template <int Base> void lambda_template(int *a, int n) {
  // Substituting Base leaves a dependent expression until the generic lambda
  // is instantiated. Do not evaluate the hint prematurely.
  auto loop = [=](auto tag) {
#pragma clang loop riscv_ns_swp_distance(Base + decltype(tag)::value)
    for (int i = 0; i < n; ++i)
      a[i]++;
  };
  loop(DistanceTag<1>{});
}

void instantiate(int *a, int n) {
  loop_template<1, 1, 1, 1>(a, n);
  loop_template<62, 62, 2147483647, 64>(a, n);
  lambda_template<2>(a, n);
}

// A value of one remains a numeric Non-Spec request after instantiation.
// CHECK-LABEL: define linkonce_odr {{.*}} @_Z13loop_templateILi1ELi1ELi1ELi1EEvPii(
// CHECK: br label {{.*}}, !llvm.loop ![[MIN:[0-9]+]]
// CHECK-LABEL: define linkonce_odr {{.*}} @_Z13loop_templateILi62ELi62ELi2147483647ELi64EEvPii(
// CHECK: br label {{.*}}, !llvm.loop ![[MAX:[0-9]+]]
// CHECK-LABEL: define linkonce_odr {{.*}} @_ZZ15lambda_template
// CHECK: br label {{.*}}, !llvm.loop ![[LAMBDA:[0-9]+]]

// CHECK-DAG: ![[RANGE]] = distinct !{![[RANGE]], {{.*}}![[SWP:[0-9]+]], ![[IR:[0-9]+]], ![[MACHINE:[0-9]+]], ![[FORCE:[0-9]+]], ![[SPEC:[0-9]+]], ![[SHIFT:[0-9]+]], ![[TOGGLE:[0-9]+]], ![[DIST:[0-9]+]], ![[SDIST:[0-9]+]], ![[SLICE:[0-9]+]], ![[SIZE:[0-9]+]], ![[PROLOGUE:[0-9]+]], ![[DEPTH:[0-9]+]]}
// CHECK-DAG: ![[SWP]] = !{!"llvm.loop.riscv.ns.swp.enable", i1 true}
// CHECK-DAG: ![[IR]] = !{!"llvm.loop.riscv.ns.fission.ir.enable", i1 false}
// CHECK-DAG: ![[MACHINE]] = !{!"llvm.loop.riscv.ns.fission.machine.enable", i1 false}
// CHECK-DAG: ![[FORCE]] = !{!"llvm.loop.riscv.ns.force", i1 false}
// CHECK-DAG: ![[SPEC]] = !{!"llvm.loop.riscv.ns.swp.speculative", i1 true}
// CHECK-DAG: ![[SHIFT]] = !{!"llvm.loop.riscv.ns.swp.shift_recurrences", i1 true}
// CHECK-DAG: ![[TOGGLE]] = !{!"llvm.loop.riscv.ns.swp.toggles", i1 false}
// CHECK-DAG: ![[DIST]] = !{!"llvm.loop.riscv.ns.swp.distance", i32 4}
// CHECK-DAG: ![[SDIST]] = !{!"llvm.loop.riscv.ns.swp.speculative_distance", i32 2}
// CHECK-DAG: ![[SLICE]] = !{!"llvm.loop.riscv.ns.swp.max_slice_size", i32 8}
// CHECK-DAG: ![[SIZE]] = !{!"llvm.loop.riscv.ns.swp.max_loop_size", i32 32}
// CHECK-DAG: ![[PROLOGUE]] = !{!"llvm.loop.riscv.ns.swp.max_prologue_size", i32 64}
// CHECK-DAG: ![[DEPTH]] = !{!"llvm.loop.riscv.ns.fission.queue_depth", i32 16}

// CHECK-DAG: ![[MIN]] = distinct !{![[MIN]], {{.*}}![[MIN0:[0-9]+]], ![[MIN1:[0-9]+]], ![[MIN2:[0-9]+]], ![[MIN3:[0-9]+]], ![[MIN4:[0-9]+]], ![[MIN5:[0-9]+]]}
// CHECK-DAG: ![[MIN0]] = !{!"llvm.loop.riscv.ns.swp.distance", i32 1}
// CHECK-DAG: ![[MIN1]] = !{!"llvm.loop.riscv.ns.swp.speculative_distance", i32 1}
// CHECK-DAG: ![[MIN2]] = !{!"llvm.loop.riscv.ns.swp.max_slice_size", i32 1}
// CHECK-DAG: ![[MIN3]] = !{!"llvm.loop.riscv.ns.swp.max_loop_size", i32 1}
// CHECK-DAG: ![[MIN4]] = !{!"llvm.loop.riscv.ns.swp.max_prologue_size", i32 1}
// CHECK-DAG: ![[MIN5]] = !{!"llvm.loop.riscv.ns.fission.queue_depth", i32 1}

// CHECK-DAG: ![[MAX]] = distinct !{![[MAX]], {{.*}}![[MAX0:[0-9]+]], ![[MAX1:[0-9]+]], ![[MAX2:[0-9]+]], ![[MAX3:[0-9]+]], ![[MAX4:[0-9]+]], ![[MAX5:[0-9]+]]}
// CHECK-DAG: ![[MAX0]] = !{!"llvm.loop.riscv.ns.swp.distance", i32 62}
// CHECK-DAG: ![[MAX1]] = !{!"llvm.loop.riscv.ns.swp.speculative_distance", i32 62}
// CHECK-DAG: ![[MAX2]] = !{!"llvm.loop.riscv.ns.swp.max_slice_size", i32 2147483647}
// CHECK-DAG: ![[MAX3]] = !{!"llvm.loop.riscv.ns.swp.max_loop_size", i32 2147483647}
// CHECK-DAG: ![[MAX4]] = !{!"llvm.loop.riscv.ns.swp.max_prologue_size", i32 2147483647}
// CHECK-DAG: ![[MAX5]] = !{!"llvm.loop.riscv.ns.fission.queue_depth", i32 64}

// CHECK-DAG: ![[LAMBDA]] = distinct !{![[LAMBDA]], {{.*}}![[LAMBDADIST:[0-9]+]]}
// CHECK-DAG: ![[LAMBDADIST]] = !{!"llvm.loop.riscv.ns.swp.distance", i32 3}

// PRINT: #pragma clang loop riscv_ns_swp(enable)
// PRINT: #pragma clang loop riscv_ns_ir_fission(disable)
// PRINT: #pragma clang loop riscv_ns_machine_fission(disable)
// PRINT: #pragma clang loop riscv_ns_force(disable)
// PRINT: #pragma clang loop riscv_ns_swp_speculative(enable)
// PRINT: #pragma clang loop riscv_ns_swp_shift_recurrences(enable)
// PRINT: #pragma clang loop riscv_ns_swp_toggles(disable)
// PRINT: #pragma clang loop riscv_ns_swp_distance(4)
// PRINT: #pragma clang loop riscv_ns_swp_speculative_distance(2)
// PRINT: #pragma clang loop riscv_ns_swp_max_slice_size(8)
// PRINT: #pragma clang loop riscv_ns_swp_max_loop_size(32)
// PRINT: #pragma clang loop riscv_ns_swp_max_prologue_size(64)
// PRINT: #pragma clang loop riscv_ns_fission_queue_depth(16)
// PRINT: #pragma clang loop riscv_ns_swp_distance(Distance)
// PRINT: #pragma clang loop riscv_ns_swp_speculative_distance(SpeculativeDistance)
// PRINT: #pragma clang loop riscv_ns_swp_max_slice_size(Size)
// PRINT: #pragma clang loop riscv_ns_swp_max_loop_size(Size)
// PRINT: #pragma clang loop riscv_ns_swp_max_prologue_size(Size)
// PRINT: #pragma clang loop riscv_ns_fission_queue_depth(Depth)

#endif
