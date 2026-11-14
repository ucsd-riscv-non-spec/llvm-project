; RUN: llc -mtriple=riscv64 -O2 -save-temp-labels -stop-after=riscv-loop-fission-decomposed-branch < %s > %t-after.mir
; RUN: llc -mtriple=riscv64 -run-pass=riscv-loop-fission-decomposed-branch %t-after.mir -o - | FileCheck %s --check-prefix=MACHINE
; RUN: llc -mtriple=riscv64 -O2 -stop-after=riscv-ir-loop-fission < %s | FileCheck %s --check-prefix=IR
; RUN: llc -mtriple=riscv64 -O2 -riscv-ns-disable-ir-loop-fission -stop-after=riscv-ir-loop-fission < %s | FileCheck %s --check-prefix=IR-OFF
; RUN: llc -mtriple=riscv64 -O2 -verify-machineinstrs -loop-fission-queue-depth=0 -stop-after=riscv-loop-fission-decomposed-branch < %s | FileCheck %s --check-prefix=MACHINE
; RUN: llc -mtriple=riscv64 -O2 -disable-loop-fission-decomposed-branch -stop-after=riscv-loop-fission-decomposed-branch < %s | FileCheck %s --check-prefix=MACHINE-OFF
; RUN: llc -mtriple=riscv64 -O2 -stop-before=riscv-loop-fission-decomposed-branch < %s > %t.mir
; RUN: sed 's/bb\.\([0-9]*\)\.latch:/bb.\1:/' %t.mir > %t-lost-latch.mir
; RUN: llc -mtriple=riscv64 -run-pass=riscv-loop-fission-decomposed-branch %t-lost-latch.mir -o - | FileCheck %s --check-prefix=MACHINE
; RUN: llc -mtriple=riscv64 -run-pass=riscv-loop-fission-decomposed-branch,riscv-loop-fission-decomposed-branch %t.mir -o - | FileCheck %s --check-prefix=MACHINE
; RUN: sed -E 's/(bb\.[0-9]+)\.[^: ]+:/\1:/' %t.mir > %t-lost-all.mir
; RUN: not llc -mtriple=riscv64 -run-pass=riscv-loop-fission-decomposed-branch %t-lost-all.mir -o /dev/null 2>&1 | FileCheck %s --check-prefix=LOST

; Local depth is a whole-loop cutoff for IR fission and a tile bound for machine
; fission. Force bypasses alias rejection but never hardware depth or disables.
; The MIR tests exercise both missing latch IDs and repeated transformations.
; LOST: error: {{.*}}cannot recover Non-Spec controls for machine loop
; IR-OFF-NOT: call {{.*}}@llvm.riscv.ns.cond

; IR-LABEL: define {{.*}} @ir_depth_cutoff(
; IR-NOT: call {{.*}}@llvm.riscv.ns.cond
; IR: ret void
; MACHINE-LABEL: name: ir_depth_cutoff
; MACHINE-NOT: BMOVC_B
; MACHINE: ...
; MACHINE-OFF-LABEL: name: ir_depth_cutoff
; MACHINE-OFF-NOT: BMOVC_B
; MACHINE-OFF: ...
define void @ir_depth_cutoff(ptr noalias %a, ptr noalias %out) {
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
  br i1 %again, label %loop, label %exit, !llvm.loop !0
exit:
  ret void
}

; IR-LABEL: define {{.*}} @ir_enabled(
; IR: fission.exit:
; IR: call i1 @llvm.riscv.ns.cond.pop
; MACHINE-LABEL: name: ir_enabled
; MACHINE-OFF-LABEL: name: ir_enabled
define void @ir_enabled(ptr noalias %a, ptr noalias %out) {
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
  br i1 %again, label %loop, label %exit, !llvm.loop !20
exit:
  ret void
}

; IR-LABEL: define {{.*}} @ir_alias(
; IR-NOT: call {{.*}}@llvm.riscv.ns.cond
; IR: ret void
; MACHINE-LABEL: name: ir_alias
; MACHINE-NOT: BMOVC_B
; MACHINE: ...
; MACHINE-OFF-LABEL: name: ir_alias
; MACHINE-OFF-NOT: BMOVC_B
; MACHINE-OFF: ...
define void @ir_alias(ptr %a, ptr %out) {
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
  br i1 %again, label %loop, label %exit, !llvm.loop !40
exit:
  ret void
}

; IR-LABEL: define {{.*}} @ir_forced(
; IR: fission.exit:
; IR: call i1 @llvm.riscv.ns.cond.pop
; MACHINE-LABEL: name: ir_forced
; MACHINE-OFF-LABEL: name: ir_forced
define void @ir_forced(ptr %a, ptr %out) {
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
  br i1 %again, label %loop, label %exit, !llvm.loop !60
exit:
  ret void
}

; IR-LABEL: define {{.*}} @machine_enabled(
; IR-NOT: call {{.*}}@llvm.riscv.ns.cond
; IR: ret void
; MACHINE-LABEL: name: machine_enabled
; MACHINE: ADDI $x0, 3
; MACHINE: ADDI $x0, 16
; MACHINE: bb.{{[0-9]+}}:
; MACHINE: BMOVC_B
; MACHINE: PseudoPBC
; MACHINE-OFF-LABEL: name: machine_enabled
; MACHINE-OFF-NOT: BMOVC_B
; MACHINE-OFF: ...
define void @machine_enabled(ptr noalias %a, ptr noalias %out) {
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
  %again = icmp ult i64 %next, 35
  br i1 %again, label %loop, label %exit, !llvm.loop !80
exit:
  ret void
}

; IR-LABEL: define {{.*}} @machine_disabled(
; IR-NOT: call {{.*}}@llvm.riscv.ns.cond
; IR: ret void
; MACHINE-LABEL: name: machine_disabled
; MACHINE-NOT: BMOVC_B
; MACHINE: ...
; MACHINE-OFF-LABEL: name: machine_disabled
; MACHINE-OFF-NOT: BMOVC_B
; MACHINE-OFF: ...
define void @machine_disabled(ptr noalias %a, ptr noalias %out) {
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
  %again = icmp ult i64 %next, 35
  br i1 %again, label %loop, label %exit, !llvm.loop !100
exit:
  ret void
}

; IR-LABEL: define {{.*}} @machine_alias(
; IR-NOT: call {{.*}}@llvm.riscv.ns.cond
; IR: ret void
; MACHINE-LABEL: name: machine_alias
; MACHINE-NOT: BMOVC_B
; MACHINE: ...
; MACHINE-OFF-LABEL: name: machine_alias
; MACHINE-OFF-NOT: BMOVC_B
; MACHINE-OFF: ...
define void @machine_alias(ptr %a, ptr %out) {
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
  %again = icmp ult i64 %next, 35
  br i1 %again, label %loop, label %exit, !llvm.loop !120
exit:
  ret void
}

; IR-LABEL: define {{.*}} @machine_forced(
; IR-NOT: call {{.*}}@llvm.riscv.ns.cond
; IR: ret void
; MACHINE-LABEL: name: machine_forced
; MACHINE: bb.{{[0-9]+}}:
; MACHINE: BMOVC_B
; MACHINE: PseudoPBC
; MACHINE-OFF-LABEL: name: machine_forced
; MACHINE-OFF-NOT: BMOVC_B
; MACHINE-OFF: ...
define void @machine_forced(ptr %a, ptr %out) {
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
  %again = icmp ult i64 %next, 35
  br i1 %again, label %loop, label %exit, !llvm.loop !140
exit:
  ret void
}

!0 = distinct !{!0, !1, !2, !3}
!1 = !{!"llvm.loop.riscv.ns.swp.enable", i1 false}
!2 = !{!"llvm.loop.riscv.ns.fission.queue_depth", i32 16}
!3 = !{!"llvm.loop.riscv.ns.fission.machine.enable", i1 false}

!20 = distinct !{!20, !21, !22, !23, !24}
!21 = !{!"llvm.loop.riscv.ns.swp.enable", i1 false}
!22 = !{!"llvm.loop.riscv.ns.fission.ir.enable", i1 true}
!23 = !{!"llvm.loop.riscv.ns.fission.queue_depth", i32 32}
!24 = !{!"llvm.loop.riscv.ns.fission.machine.enable", i1 false}

!40 = distinct !{!40, !41, !42, !43}
!41 = !{!"llvm.loop.riscv.ns.swp.enable", i1 false}
!42 = !{!"llvm.loop.riscv.ns.fission.ir.enable", i1 true}
!43 = !{!"llvm.loop.riscv.ns.fission.machine.enable", i1 false}

!60 = distinct !{!60, !61, !62, !63, !64}
!61 = !{!"llvm.loop.riscv.ns.swp.enable", i1 false}
!62 = !{!"llvm.loop.riscv.ns.fission.ir.enable", i1 true}
!63 = !{!"llvm.loop.riscv.ns.fission.machine.enable", i1 false}
!64 = !{!"llvm.loop.riscv.ns.force", i1 true}

!80 = distinct !{!80, !81, !82, !83, !84}
!81 = !{!"llvm.loop.riscv.ns.swp.enable", i1 false}
!82 = !{!"llvm.loop.riscv.ns.fission.ir.enable", i1 false}
!83 = !{!"llvm.loop.riscv.ns.fission.machine.enable", i1 true}
!84 = !{!"llvm.loop.riscv.ns.fission.queue_depth", i32 16}

!100 = distinct !{!100, !101, !102, !103, !104, !105}
!101 = !{!"llvm.loop.riscv.ns.swp.enable", i1 false}
!102 = !{!"llvm.loop.riscv.ns.fission.ir.enable", i1 false}
!103 = !{!"llvm.loop.riscv.ns.fission.machine.enable", i1 false}
!104 = !{!"llvm.loop.riscv.ns.fission.queue_depth", i32 16}
!105 = !{!"llvm.loop.riscv.ns.force", i1 true}

!120 = distinct !{!120, !121, !122, !123, !124}
!121 = !{!"llvm.loop.riscv.ns.swp.enable", i1 false}
!122 = !{!"llvm.loop.riscv.ns.fission.ir.enable", i1 false}
!123 = !{!"llvm.loop.riscv.ns.fission.machine.enable", i1 true}
!124 = !{!"llvm.loop.riscv.ns.fission.queue_depth", i32 16}

!140 = distinct !{!140, !141, !142, !143, !144, !145}
!141 = !{!"llvm.loop.riscv.ns.swp.enable", i1 false}
!142 = !{!"llvm.loop.riscv.ns.fission.ir.enable", i1 false}
!143 = !{!"llvm.loop.riscv.ns.fission.machine.enable", i1 true}
!144 = !{!"llvm.loop.riscv.ns.fission.queue_depth", i32 16}
!145 = !{!"llvm.loop.riscv.ns.force", i1 true}
