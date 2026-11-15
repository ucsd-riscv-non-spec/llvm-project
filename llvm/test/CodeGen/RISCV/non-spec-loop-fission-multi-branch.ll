; RUN: llc -mtriple=riscv64 -O2 -riscv-ns-disable-software-pipelining=true -stop-after=riscv-loop-fission-decomposed-branch %s -o - | FileCheck %s --check-prefix=MIR
; RUN: llc -mtriple=riscv64 -O2 -riscv-ns-disable-software-pipelining=true %s -o - | FileCheck %s --check-prefix=ASM

define void @dynamic_trip_multi_branch(ptr noalias %src, ptr noalias %dst,
                                       i64 %n, i32 %threshold) {
entry:
  %empty = icmp eq i64 %n, 0
  br i1 %empty, label %exit, label %loop.preheader

loop.preheader:
  br label %loop

loop:
  %i = phi i64 [ 0, %loop.preheader ], [ %next, %latch ]
  %src.ptr = getelementptr i32, ptr %src, i64 %i
  %value = load i32, ptr %src.ptr, align 4
  %positive = icmp sgt i32 %threshold, 0
  br i1 %positive, label %store, label %check

check:
  %zero = icmp eq i32 %threshold, 0
  br i1 %zero, label %latch, label %store

store:
  %dst.ptr = getelementptr i32, ptr %dst, i64 %i
  store i32 %value, ptr %dst.ptr, align 4
  br label %latch

latch:
  %next = add nuw i64 %i, 1
  %done = icmp eq i64 %next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

define void @cross_iteration_branch(ptr noalias %src, ptr noalias %dst,
                                    i64 %n) {
entry:
  %empty = icmp eq i64 %n, 0
  br i1 %empty, label %exit, label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %next, %latch ]
  %ptr = getelementptr i32, ptr %src, i64 %i
  %value = load i32, ptr %ptr, align 4
  %positive = icmp sgt i32 %value, 0
  br i1 %positive, label %store, label %latch

store:
  %dst.ptr = getelementptr i32, ptr %dst, i64 %i
  store i32 %value, ptr %dst.ptr, align 4
  br label %latch

latch:
  %next = add nuw i64 %i, 1
  %done = icmp eq i64 %next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

declare i32 @getc(ptr)

define void @call_result_branch(ptr %stream, ptr %dst, i64 %n) {
entry:
  %empty = icmp eq i64 %n, 0
  br i1 %empty, label %exit, label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %next, %latch ]
  %value = call i32 @getc(ptr %stream)
  %is.zero = icmp eq i32 %value, 0
  br i1 %is.zero, label %store, label %latch

store:
  %dst.ptr = getelementptr i32, ptr %dst, i64 %i
  store i32 %value, ptr %dst.ptr, align 4
  br label %latch

latch:
  %next = add nuw i64 %i, 1
  %done = icmp eq i64 %next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

; The pass selects the loop-header branch and tiles the dynamic count.
; MIR-LABEL: name: dynamic_trip_multi_branch
; MIR: PseudoLoopSetup
; MIR: PseudoBRNE
; MIR: SLTIU
; MIR: PseudoPBC
; MIR-LABEL: name: cross_iteration_branch
; MIR-NOT: PseudoPBC
; The call result is copied from a physical return register, so its branch
; condition must not be cloned into the compute loop.
; MIR-LABEL: name: call_result_branch
; MIR-NOT: BMOVC_BNE
; MIR: PseudoLoopSetup

; ASM-LABEL: dynamic_trip_multi_branch:
; ASM: bmovc
; ASM: pb
