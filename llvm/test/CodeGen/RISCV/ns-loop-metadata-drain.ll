; RUN: llc -mtriple=riscv64 -O2 -save-temp-labels -verify-machineinstrs -loop-fission-queue-depth=0 -stop-after=riscv-loop-fission-decomposed-branch < %s > %t.mir
; RUN: FileCheck %s < %t.mir
; RUN: llc -mtriple=riscv64 -run-pass=riscv-loop-fission-decomposed-branch %t.mir -o - | FileCheck %s
; RUN: sed 's/machine.enable", i1 true/machine.enable", i1 false/' %s > %t-off.ll
; RUN: llc -mtriple=riscv64 -O2 -stop-after=riscv-loop-fission-decomposed-branch < %t-off.ll | FileCheck %s --check-prefix=OFF

; A counted SWP kernel is never machine-fissioned. Its drain inherits machine
; enable, depth and force, so it can fission despite unknown aliases and tile
; eight iterations as 2 + 3 + 3. A local disable prevents this later attempt.
; Generated machine loop provenance also survives a MIR round trip.
; CHECK-LABEL: name: pipeline
; CHECK: nsFissionedBlocks:
; CHECK: bb.{{[0-9]+}}.loop.swp:
; CHECK-NOT: BMOVC_B
; CHECK: bb.{{[0-9]+}}.swp.drain.ph:
; CHECK: ADDI $x0, 2
; CHECK: ADDI $x0, 3
; CHECK: BMOVC_B
; CHECK: PseudoPBC
; OFF-LABEL: name: pipeline
; OFF-NOT: BMOVC_B

define void @pipeline(ptr %a, ptr %out) {
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
  %again = icmp ult i64 %next, 200
  br i1 %again, label %loop, label %exit, !llvm.loop !0
exit:
  ret void
}

!0 = distinct !{!0, !1, !2, !3, !4, !5}
!1 = !{!"llvm.loop.riscv.ns.swp.distance", i32 8}
!2 = !{!"llvm.loop.riscv.ns.fission.ir.enable", i1 false}
!3 = !{!"llvm.loop.riscv.ns.fission.machine.enable", i1 true}
!4 = !{!"llvm.loop.riscv.ns.fission.queue_depth", i32 3}
!5 = !{!"llvm.loop.riscv.ns.force", i1 true}
