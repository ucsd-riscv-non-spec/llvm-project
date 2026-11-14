; RUN: llc -mtriple=riscv64 -O2 -riscv-ns-swp-distance=9 -stop-after=riscv-branch-condition-pipelining < %s | FileCheck %s --check-prefixes=CHECK,DEFAULT
; RUN: llc -mtriple=riscv64 -O2 -riscv-ns-swp-distance=0 -stop-after=riscv-branch-condition-pipelining < %s | FileCheck %s --check-prefixes=CHECK,ZERO
; RUN: llc -mtriple=riscv64 -O2 -riscv-ns-disable-software-pipelining -stop-after=riscv-branch-condition-pipelining < %s | FileCheck %s --check-prefix=OFF
; RUN: llc -mtriple=riscv64 -O2 -riscv-ns-swp-max-slice-size=1 -riscv-ns-swp-max-loop-size=1 -riscv-ns-swp-max-prologue-size=1 -riscv-ns-swp-max-function-branches=1 -stop-after=riscv-branch-condition-pipelining < %s | FileCheck %s --check-prefix=FORCE
; RUN: llc -mtriple=riscv64 -O2 -riscv-ns-swp-max-slice-size=1 -riscv-ns-swp-max-loop-size=1 -riscv-ns-swp-max-prologue-size=1 -stop-after=riscv-branch-condition-pipelining < %s | FileCheck %s --check-prefix=LIMITS
; RUN: llc -mtriple=riscv64 -O2 -pass-remarks-analysis=riscv-branch-condition-pipelining -pass-remarks-missed=riscv-branch-condition-pipelining -pass-remarks=riscv-branch-condition-pipelining -stop-after=riscv-branch-condition-pipelining < %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=REMARK

; RUN: sed 's/!0 = distinct !{!0,/!0 = distinct !{!0, !22,/' %s > %t.ll
; RUN: llc -mtriple=riscv64 -O2 -riscv-ns-swp-max-function-branches=1 -stop-after=riscv-branch-condition-pipelining < %t.ll | FileCheck %s --check-prefix=BUDGET

; OFF-NOT: call {{.*}}@llvm.riscv.ns.cond
; REMARK: Non-Spec settings for function siblings, loop %first{{.*}}distance=2 (metadata){{.*}}speculative=0 (default)
; REMARK: requested distance 2, effective distance 2, force=false
; REMARK: Non-Spec settings for function siblings, loop %second{{.*}}distance=5 (metadata)
; REMARK: force active for loop loop: bypassing size, profitability, optsize and function branch budgets, memory dependence and speculative safety proofs
; REMARK: not pipelining branch conditions in function disabled, loop loop: loop software pipelining disable
; REMARK: not pipelining branch conditions in function unsupported_recurrence{{.*}}cannot construct future recurrence values

; Sibling loops resolve independently, and explicit metadata overrides even an
; explicitly supplied CLI distance (including legacy zero). No settings leak.
define void @siblings(ptr %a, i64 %n) {
; BUDGET-LABEL: define {{.*}} @siblings(
; BUDGET: %swp.long.enough = icmp ugt i64 {{%.*}}, 2
; BUDGET-NOT: icmp ugt i64 {{%.*}}, 5
; BUDGET: second:
; BUDGET-NOT: call {{.*}}@llvm.riscv.ns.cond
; BUDGET: ret void
; CHECK-LABEL: define {{.*}} @siblings(
; CHECK: %swp.long.enough = icmp ugt i64 {{%.*}}, 2
; CHECK: %swp.long.enough{{[0-9]*}} = icmp ugt i64 {{%.*}}, 5
entry:
  br label %first
first:
  %i = phi i64 [ 0, %entry ], [ %inext, %first.latch ]
  %p = getelementptr i32, ptr %a, i64 %i
  %v = load i32, ptr %p
  %c = icmp sgt i32 %v, 0
  br i1 %c, label %first.body, label %first.latch
first.body:
  store i32 0, ptr %p
  br label %first.latch
first.latch:
  %inext = add nuw i64 %i, 1
  %again = icmp ult i64 %inext, %n
  br i1 %again, label %first, label %second.ph, !llvm.loop !0
second.ph:
  br label %second
second:
  %j = phi i64 [ 0, %second.ph ], [ %jnext, %second.latch ]
  %q = getelementptr i32, ptr %a, i64 %j
  %w = load i32, ptr %q
  %d = icmp slt i32 %w, 0
  br i1 %d, label %second.body, label %second.latch
second.body:
  store i32 0, ptr %q
  br label %second.latch
second.latch:
  %jnext = add nuw i64 %j, 1
  %more = icmp ult i64 %jnext, %n
  br i1 %more, label %second, label %exit, !llvm.loop !10
exit:
  ret void
}

; A loop without local metadata continues to inherit the CLI value.
define void @unannotated(ptr %a, i64 %n) {
; DEFAULT-LABEL: define {{.*}} @unannotated(
; DEFAULT: icmp ugt i64 {{%.*}}, 9
; ZERO-LABEL: define {{.*}} @unannotated(
; ZERO-NOT: call {{.*}}@llvm.riscv.ns.cond
; ZERO: ret void
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %next, %latch ]
  %p = getelementptr i32, ptr %a, i64 %i
  %v = load i32, ptr %p
  %c = icmp sgt i32 %v, 0
  br i1 %c, label %body, label %latch
body:
  store i32 0, ptr %p
  br label %latch
latch:
  %next = add nuw i64 %i, 1
  %again = icmp ult i64 %next, %n
  br i1 %again, label %loop, label %exit
exit:
  ret void
}

; Force bypasses alias proofs, optsize and all local and global soft budgets.
; It still constructs guarded, valid IR when a trip is shorter than distance.
define void @forced(ptr %a, ptr %out) optsize {
; CHECK-LABEL: define {{.*}} @forced(
; CHECK: swp.ph:
; CHECK: %i.ahead = add i64 %i.swp, 8
; CHECK: ret void
; FORCE-LABEL: define {{.*}} @forced(
; FORCE: %i.ahead = add i64 %i.swp, 8
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %next, %latch ]
  %p = getelementptr i32, ptr %a, i64 %i
  %v = load i32, ptr %p
  %c = icmp sgt i32 %v, 0
  br i1 %c, label %body, label %latch
body:
  %q = getelementptr i32, ptr %out, i64 %i
  store i32 0, ptr %q
  br label %latch
latch:
  %next = add nuw i64 %i, 1
  %again = icmp ult i64 %next, 4
  br i1 %again, label %loop, label %exit, !llvm.loop !20
exit:
  ret void
}

; A per-loop disable takes precedence over force.
define void @disabled(ptr %a, i64 %n) {
; CHECK-LABEL: define {{.*}} @disabled(
; CHECK-NOT: call {{.*}}@llvm.riscv.ns.cond
; CHECK: ret void
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %next, %latch ]
  %p = getelementptr i32, ptr %a, i64 %i
  %v = load i32, ptr %p
  %c = icmp sgt i32 %v, 0
  br i1 %c, label %body, label %latch
body:
  store i32 0, ptr %p
  br label %latch
latch:
  %next = add nuw i64 %i, 1
  %again = icmp ult i64 %next, %n
  br i1 %again, label %loop, label %exit, !llvm.loop !30
exit:
  ret void
}

; Local size limits can also raise the corresponding CLI limits without force.
define void @raised_limits(ptr %a, i64 %n) {
; LIMITS-LABEL: define {{.*}} @raised_limits(
; LIMITS: %swp.long.enough = icmp ugt i64 {{%.*}}, 3
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %next, %latch ]
  %p = getelementptr i32, ptr %a, i64 %i
  %v = load i32, ptr %p
  %c = icmp sgt i32 %v, 0
  br i1 %c, label %body, label %latch
body:
  store i32 0, ptr %p
  br label %latch
latch:
  %next = add nuw i64 %i, 1
  %again = icmp ult i64 %next, %n
  br i1 %again, label %loop, label %exit, !llvm.loop !40
exit:
  ret void
}

; Force cannot invent machinery for computing a future non-affine recurrence.
define void @unsupported_recurrence(ptr %out, i64 %n, i64 %x) {
; CHECK-LABEL: define {{.*}} @unsupported_recurrence(
; CHECK-NOT: call {{.*}}@llvm.riscv.ns.cond
; CHECK: ret void
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %next, %latch ]
  %v = phi i64 [ %x, %entry ], [ %square, %latch ]
  %c = icmp sgt i64 %v, 0
  br i1 %c, label %body, label %latch
body:
  store i64 %v, ptr %out
  br label %latch
latch:
  %square = mul i64 %v, %v
  %next = add nuw i64 %i, 1
  %again = icmp ult i64 %next, %n
  br i1 %again, label %loop, label %exit, !llvm.loop !50
exit:
  ret void
}


; Outer controls do not apply recursively to an unannotated inner loop.
define void @nested(ptr %a, i64 %n, i64 %repeats) {
; DEFAULT-LABEL: define {{.*}} @nested(
; DEFAULT: %swp.long.enough = icmp ugt i64 {{%.*}}, 9
; ZERO-LABEL: define {{.*}} @nested(
; ZERO-NOT: call {{.*}}@llvm.riscv.ns.cond
; ZERO: ret void
entry:
  br label %outer
outer:
  %r = phi i64 [ 0, %entry ], [ %rnext, %outer.latch ]
  br label %inner.ph
inner.ph:
  br label %inner
inner:
  %i = phi i64 [ 0, %inner.ph ], [ %next, %inner.latch ]
  %p = getelementptr i32, ptr %a, i64 %i
  %v = load i32, ptr %p
  %c = icmp sgt i32 %v, 0
  br i1 %c, label %body, label %inner.latch
body:
  store i32 0, ptr %p
  br label %inner.latch
inner.latch:
  %next = add nuw i64 %i, 1
  %again = icmp ult i64 %next, %n
  br i1 %again, label %inner, label %outer.latch
outer.latch:
  %rnext = add nuw i64 %r, 1
  %more = icmp ult i64 %rnext, %repeats
  br i1 %more, label %outer, label %exit, !llvm.loop !30
exit:
  ret void
}

!0 = distinct !{!0, !1, !2, !3, !4}
!1 = !{!"llvm.loop.riscv.ns.swp.enable", i1 true}
!2 = !{!"llvm.loop.riscv.ns.swp.distance", i32 2}
!3 = !{!"llvm.loop.riscv.ns.fission.ir.enable", i1 false}
!4 = !{!"llvm.loop.riscv.ns.fission.machine.enable", i1 false}
!10 = distinct !{!10, !1, !11, !3, !4}
!11 = !{!"llvm.loop.riscv.ns.swp.distance", i32 5}
!20 = distinct !{!20, !1, !21, !22, !23, !24, !25, !3, !4}
!21 = !{!"llvm.loop.riscv.ns.swp.distance", i32 8}
!22 = !{!"llvm.loop.riscv.ns.force", i1 true}
!23 = !{!"llvm.loop.riscv.ns.swp.max_slice_size", i32 1}
!24 = !{!"llvm.loop.riscv.ns.swp.max_loop_size", i32 1}
!25 = !{!"llvm.loop.riscv.ns.swp.max_prologue_size", i32 1}
!30 = distinct !{!30, !31, !22, !3, !4}
!31 = !{!"llvm.loop.riscv.ns.swp.enable", i1 false}
!40 = distinct !{!40, !41, !42, !43, !44, !3, !4}
!41 = !{!"llvm.loop.riscv.ns.swp.distance", i32 3}
!42 = !{!"llvm.loop.riscv.ns.swp.max_slice_size", i32 64}
!43 = !{!"llvm.loop.riscv.ns.swp.max_loop_size", i32 64}
!44 = !{!"llvm.loop.riscv.ns.swp.max_prologue_size", i32 64}
!50 = distinct !{!50, !21, !22, !3, !4}
