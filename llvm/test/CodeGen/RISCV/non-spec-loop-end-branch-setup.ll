; RUN: llc -mtriple=riscv64 -O2 -filetype=asm -o - %s | FileCheck %s

define i64 @sum(ptr %a, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %next, %body ]
  %acc = phi i64 [ 0, %entry ], [ %sum, %body ]
  %cond = icmp ult i64 %i, %n
  br i1 %cond, label %body, label %exit

body:
  %p = getelementptr i64, ptr %a, i64 %i
  %v = load i64, ptr %p, align 8
  %sum = add i64 %acc, %v
  %next = add i64 %i, 1
  br label %loop

exit:
  ret i64 %acc
}

; CHECK-LABEL: sum:
; CHECK: bmovs
; CHECK: bmovt
; CHECK: bmovc_loop
; CHECK: .Lns_loop_{{[0-9]+}}:
; CHECK: pb
