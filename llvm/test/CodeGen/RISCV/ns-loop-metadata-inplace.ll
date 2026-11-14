; RUN: llc -mtriple=riscv64 -O2 -riscv-ns-swp-speculative-distance=9 -riscv-ns-swp-toggles -riscv-ns-swp-shift-recurrences -stop-after=riscv-branch-condition-pipelining < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -O2 -riscv-ns-swp-speculative -stop-after=riscv-branch-condition-pipelining < %s | FileCheck %s --check-prefix=LOCAL-OFF
; RUN: llc -mtriple=riscv64 -O2 -riscv-ns-swp-distance=0 -stop-after=riscv-branch-condition-pipelining < %s | FileCheck %s --check-prefix=OFF
; RUN: llc -mtriple=riscv64 -mattr=+m -O2 -stop-after=riscv-loop-fission-decomposed-branch < %s -o /dev/null

; OFF-NOT: call {{.*}}@llvm.riscv.ns.cond

; A counted toggle-only loop needs no speculative opt-in. Local toggles enable
; and distance override the CLI; a local speculative disable remains harmless.
define void @toggle(ptr %out, i64 %n) {
; CHECK-LABEL: define {{.*}} @toggle(
; CHECK-COUNT-3: call void @llvm.riscv.ns.cond.push
; CHECK: br label %loop
; CHECK-NOT: swp.kernel
; CHECK: %t.ahead = xor i1 %t, true
; CHECK: !llvm.loop ![[TOGGLE:[0-9]+]]
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %next, %latch ]
  %t = phi i1 [ false, %entry ], [ %tnext, %latch ]
  br i1 %t, label %body, label %latch
body:
  store i64 %i, ptr %out
  br label %latch
latch:
  %tnext = xor i1 %t, true
  %next = add nuw i64 %i, 1
  %again = icmp ult i64 %next, %n
  br i1 %again, label %loop, label %exit, !llvm.loop !0
exit:
  ret void
}

; A force flag and a speculative distance do not enable the speculative path.
define i64 @feature_gate(i64 %x) {
; CHECK-LABEL: define {{.*}} @feature_gate(
; CHECK-NOT: call {{.*}}@llvm.riscv.ns.cond
; CHECK: ret i64
; LOCAL-OFF-LABEL: define {{.*}} @feature_gate(
; LOCAL-OFF-NOT: call {{.*}}@llvm.riscv.ns.cond
; LOCAL-OFF: ret i64
entry:
  br label %loop
loop:
  %v = phi i64 [ %x, %entry ], [ %half, %loop ]
  %half = lshr i64 %v, 1
  %again = icmp ne i64 %half, 0
  br i1 %again, label %loop, label %exit, !llvm.loop !10
exit:
  ret i64 %half
}

; A local false feature overrides an explicit global true, including force.
define void @toggle_disabled(ptr %out, i64 %n) {
; CHECK-LABEL: define {{.*}} @toggle_disabled(
; CHECK-NOT: call {{.*}}@llvm.riscv.ns.cond
; CHECK: ret void
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %next, %latch ]
  %t = phi i1 [ false, %entry ], [ %tnext, %latch ]
  br i1 %t, label %body, label %latch
body:
  store i64 %i, ptr %out
  br label %latch
latch:
  %tnext = xor i1 %t, true
  %next = add nuw i64 %i, 1
  %again = icmp ult i64 %next, %n
  br i1 %again, label %loop, label %exit, !llvm.loop !20
exit:
  ret void
}

; A speculative load is normally rejected even when speculative is enabled.
define void @unsafe_load(ptr %a, ptr %out) {
; CHECK-LABEL: define {{.*}} @unsafe_load(
; CHECK-NOT: call {{.*}}@llvm.riscv.ns.cond
; CHECK: ret void
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %next, %loop ]
  %p = getelementptr i32, ptr %a, i64 %i
  %v = load i32, ptr %p
  store i32 %v, ptr %out
  %next = add i64 %i, 1
  %again = icmp ne i32 %v, 0
  br i1 %again, label %loop, label %exit, !llvm.loop !30
exit:
  ret void
}

; Force reaches both slice analysis and cloneCondition's assertion. The user
; promises valid future addresses here; alias and speculation are not proved.
define void @forced_load(ptr %a, ptr %out) {
; CHECK-LABEL: define {{.*}} @forced_load(
; CHECK: %p.swp0 = getelementptr i32, ptr %a, i64 0
; CHECK: load i32, ptr %p.swp0
; CHECK: %p.swp1 = getelementptr i32, ptr %a, i64 1
; CHECK: load i32, ptr %p.swp1
; CHECK: br label %loop
; CHECK: %i.ahead = add i64 %i, 2
; CHECK: !llvm.loop ![[LOAD:[0-9]+]]
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %next, %loop ]
  %p = getelementptr i32, ptr %a, i64 %i
  %v = load i32, ptr %p
  store i32 %v, ptr %out
  %next = add i64 %i, 1
  %again = icmp ne i32 %v, 0
  br i1 %again, label %loop, label %exit, !llvm.loop !40
exit:
  ret void
}

; Local shift-recognition enable works without the global feature flag.
define i64 @shift(i64 %x) {
; CHECK-LABEL: define {{.*}} @shift(
; CHECK: call void @llvm.riscv.ns.cond.init
; CHECK: %v.ahead = lshr i64 %v, 2
; CHECK: ret i64
entry:
  br label %loop
loop:
  %v = phi i64 [ %x, %entry ], [ %half, %loop ]
  %half = lshr i64 %v, 1
  %again = icmp ne i64 %half, 0
  br i1 %again, label %loop, label %exit, !llvm.loop !50
exit:
  ret i64 %half
}

; Distance and force do not implicitly enable speculative SWP.
define i64 @feature_gate_absent(i64 %x) {
; CHECK-LABEL: define {{.*}} @feature_gate_absent(
; CHECK-NOT: call {{.*}}@llvm.riscv.ns.cond
; CHECK: ret i64
entry:
  br label %loop
loop:
  %v = phi i64 [ %x, %entry ], [ %half, %loop ]
  %half = lshr i64 %v, 1
  %again = icmp ne i64 %half, 0
  br i1 %again, label %loop, label %exit, !llvm.loop !60
exit:
  ret i64 %half
}

; Explicit shift disable beats the command-line feature enable and force.
define i64 @shift_disabled(i64 %x) {
; CHECK-LABEL: define {{.*}} @shift_disabled(
; CHECK-NOT: call {{.*}}@llvm.riscv.ns.cond
; CHECK: ret i64
entry:
  br label %loop
loop:
  %v = phi i64 [ %x, %entry ], [ %half, %loop ]
  %half = lshr i64 %v, 1
  %again = icmp ne i64 %half, 0
  br i1 %again, label %loop, label %exit, !llvm.loop !70
exit:
  ret i64 %half
}

; CHECK-DAG: ![[TOGGLE]] = distinct !{![[TOGGLE]], ![[APPLIED:[0-9]+]]}
; CHECK-DAG: ![[LOAD]] = distinct !{![[LOAD]], ![[APPLIED]]}
; CHECK-DAG: ![[APPLIED]] = !{!"llvm.loop.riscv.ns.swp.applied", !"inplace"}
!0 = distinct !{!0, !1, !2, !3, !4, !5}
!1 = !{!"llvm.loop.riscv.ns.swp.toggles", i1 true}
!2 = !{!"llvm.loop.riscv.ns.swp.speculative", i1 false}
!3 = !{!"llvm.loop.riscv.ns.swp.speculative_distance", i32 3}
!4 = !{!"llvm.loop.riscv.ns.fission.ir.enable", i1 false}
!5 = !{!"llvm.loop.riscv.ns.fission.machine.enable", i1 false}
!10 = distinct !{!10, !2, !11, !12, !4, !5}
!11 = !{!"llvm.loop.riscv.ns.force", i1 true}
!12 = !{!"llvm.loop.riscv.ns.swp.speculative_distance", i32 2}
!20 = distinct !{!20, !21, !11, !4, !5}
!21 = !{!"llvm.loop.riscv.ns.swp.toggles", i1 false}
!30 = distinct !{!30, !31, !12, !4, !5}
!31 = !{!"llvm.loop.riscv.ns.swp.speculative", i1 true}
!40 = distinct !{!40, !31, !12, !11, !4, !5}
!50 = distinct !{!50, !31, !12, !51, !4, !5}
!51 = !{!"llvm.loop.riscv.ns.swp.shift_recurrences", i1 true}

!60 = distinct !{!60, !11, !12, !51, !4, !5}
!70 = distinct !{!70, !31, !12, !71, !11, !4, !5}
!71 = !{!"llvm.loop.riscv.ns.swp.shift_recurrences", i1 false}
