; RUN: llc -mtriple=riscv64 -O2 -riscv-ns-susan-corners-interleave2 \
; RUN:   -stop-after=riscv-branch-condition-pipelining < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -O2 -riscv-ns-disable-software-pipelining \
; RUN:   -stop-after=riscv-branch-condition-pipelining < %s \
; RUN:   | FileCheck %s --check-prefix=DISABLED

; The opt-in SUSAN transform computes pixel 1's first condition before
; consuming pixel 0's, then consumes pixel 1's condition after pixel 0's tail.
; The absent-pixel path pushes a false condition so both queues are drained.
; DISABLED-NOT: susan.pixel1.cond
; DISABLED-NOT: llvm.riscv.ns.cond.push

define void @susan_corners(ptr noalias readonly %in, ptr %out,
                           ptr noalias readonly %bp, i32 %max_no, i64 %end) {
entry:
  br label %pre

pre:
  br label %for.body14

for.body14:
  %iv = phi i64 [ 0, %pre ], [ %iv.next, %for.inc ]
  %p = getelementptr inbounds i8, ptr %in, i64 %iv
  %pixel = load i8, ptr %p, align 1
  %score = zext i8 %pixel to i32
  %keep = icmp slt i32 %score, %max_no
  br i1 %keep, label %tail, label %for.inc

tail:
  %dst = getelementptr inbounds i8, ptr %out, i64 %iv
  store i8 %pixel, ptr %dst, align 1
  br label %for.inc

for.inc:
  %iv.next = add nuw i64 %iv, 1
  %done = icmp eq i64 %iv.next, %end
  br i1 %done, label %exit, label %for.body14

exit:
  ret void
}

; CHECK-LABEL: define void @susan_corners(
; CHECK: call void @llvm.riscv.ns.cond.init(i32 0)
; CHECK-NEXT: call void @llvm.riscv.ns.cond.init(i32 1)
; CHECK: call void @llvm.riscv.ns.cond.push.i64(i32 0,
; CHECK: br i1 %pixel1.valid, label %susan.pixel1.cond, label %susan.pixel1.absent
; CHECK: susan.pixel1.cond:
; CHECK: call void @llvm.riscv.ns.cond.push.i64(i32 1,
; CHECK: br label %susan.pixel0.pb
; CHECK: susan.pixel1.absent:
; CHECK: call void @llvm.riscv.ns.cond.push.i64(i32 1,
; CHECK: susan.pixel0.pb:
; CHECK: call i1 @llvm.riscv.ns.cond.pop(i32 0)
; CHECK: br i1 {{.*}}, label %tail, label %susan.after.pixel0
; CHECK: susan.after.pixel0:
; CHECK: br label %susan.pixel1.pb
; CHECK: susan.pixel1.pb:
; CHECK: call i1 @llvm.riscv.ns.cond.pop(i32 1)
; CHECK: %pixel.pair.mode = phi i1
; CHECK: %pixel.step = select i1 %pixel.pair.mode, i64 2, i64 1
