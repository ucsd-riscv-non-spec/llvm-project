; RUN: sed 's/"llvm.loop.riscv.ns.force", i1 true/"llvm.loop.riscv.ns.force", i1 false/' %s > %t-unforced.ll
; RUN: llc -mtriple=riscv64 -O2 -stop-after=riscv-branch-condition-pipelining < %t-unforced.ll | FileCheck %s --check-prefix=UNFORCED
; RUN: llc -mtriple=riscv64 -O2 -stop-after=riscv-branch-condition-pipelining < %s > %t
; RUN: FileCheck %s --check-prefix=SWP < %t
; RUN: sed '1d;/^\.\.\./,$d;s/^  //' %t > %t.ll
; RUN: llc -mtriple=riscv64 -O2 -stop-after=riscv-branch-condition-pipelining < %t.ll | FileCheck %s --check-prefix=SWP
; RUN: llc -mtriple=riscv64 -O2 -stop-after=hardware-loops < %s | FileCheck %s --check-prefix=SWP
; RUN: llc -mtriple=riscv64 -O2 -stop-after=finalize-isel < %s | FileCheck %s --check-prefix=SWP
; RUN: llc -mtriple=riscv64 -O2 -stop-after=riscv-ir-loop-fission < %s | FileCheck %s --check-prefix=FISSION

; Each generated loop gets a distinct ID. A drain retains machine policy,
; depth and force; a kernel consumes the requests. Unrelated metadata survives.
define void @pipeline(ptr %a, ptr %out, i64 %n) {
; UNFORCED-LABEL: define {{.*}} @pipeline(
; UNFORCED-NOT: call {{.*}}@llvm.riscv.ns.cond
; UNFORCED: ret void
; SWP-LABEL: define {{.*}} @pipeline(
; SWP: latch.swp:
; SWP: br i1 {{.*}}!llvm.loop ![[KERNEL:[0-9]+]]
; SWP: latch:
; SWP: br i1 {{.*}}, label %loop, label %exit, !llvm.loop ![[DRAIN:[0-9]+]]
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
  store i32 %v, ptr %q
  br label %latch
latch:
  %next = add nuw i64 %i, 1
  %again = icmp ult i64 %next, %n
  br i1 %again, label %loop, label %exit, !llvm.loop !0
exit:
  ret void
}

; IR fission consumes every user request, preserves unrelated properties and
; marks both compute and consume loops, so a repeated invocation skips them.
define void @fission(ptr noalias %a, ptr noalias %out) {
; FISSION-LABEL: define {{.*}} @fission(
; FISSION: br i1 {{.*}}, label %loop.fission, label %fission.exit, !llvm.loop ![[COMPUTE:[0-9]+]]
; FISSION: br i1 {{.*}}, label %loop, label %exit, !llvm.loop ![[CONSUME:[0-9]+]]
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
  store i32 %v, ptr %q
  br label %latch
latch:
  %next = add nuw i64 %i, 1
  %again = icmp ult i64 %next, 32
  br i1 %again, label %loop, label %exit, !llvm.loop !10
exit:
  ret void
}

; SWP-DAG: ![[KERNEL]] = distinct !{![[KERNEL]], ![[UNRELATED:[0-9]+]], ![[KROLE:[0-9]+]]}
; SWP-DAG: ![[KROLE]] = !{!"llvm.loop.riscv.ns.swp.applied", !"kernel"}
; SWP-DAG: ![[UNRELATED]] = !{!"llvm.loop.mustprogress"}
; SWP-DAG: ![[DRAIN]] = distinct !{![[DRAIN]], ![[MACHINE:[0-9]+]], ![[DEPTH:[0-9]+]], ![[FORCE:[0-9]+]], ![[UNRELATED]], ![[DROLE:[0-9]+]]}
; SWP-DAG: ![[MACHINE]] = !{!"llvm.loop.riscv.ns.fission.machine.enable", i1 false}
; SWP-DAG: ![[DEPTH]] = !{!"llvm.loop.riscv.ns.fission.queue_depth", i32 16}
; SWP-DAG: ![[FORCE]] = !{!"llvm.loop.riscv.ns.force", i1 true}
; SWP-DAG: ![[DROLE]] = !{!"llvm.loop.riscv.ns.swp.applied", !"drain"}
; FISSION-DAG: ![[COMPUTE]] = distinct !{![[COMPUTE]], ![[UNRELATED:[0-9]+]], ![[MARKER:[0-9]+]]}
; FISSION-DAG: ![[CONSUME]] = distinct !{![[CONSUME]], ![[UNRELATED]], ![[MARKER]]}
; FISSION-DAG: ![[UNRELATED]] = !{!"llvm.loop.mustprogress"}
; FISSION-DAG: ![[MARKER]] = !{!"riscv.loop-fissioned"}
!0 = distinct !{!0, !1, !2, !3, !4, !5, !6}
!1 = !{!"llvm.loop.riscv.ns.swp.distance", i32 4}
!2 = !{!"llvm.loop.riscv.ns.fission.ir.enable", i1 false}
!3 = !{!"llvm.loop.riscv.ns.fission.machine.enable", i1 false}
!4 = !{!"llvm.loop.riscv.ns.fission.queue_depth", i32 16}
!5 = !{!"llvm.loop.riscv.ns.force", i1 true}
!6 = !{!"llvm.loop.mustprogress"}
!10 = distinct !{!10, !11, !12, !3, !6}
!11 = !{!"llvm.loop.riscv.ns.fission.ir.enable", i1 true}
!12 = !{!"llvm.loop.riscv.ns.swp.enable", i1 false}
