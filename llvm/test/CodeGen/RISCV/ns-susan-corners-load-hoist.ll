; RUN: llc -mtriple=riscv64 -O2 \
; RUN:   -riscv-ns-susan-corners-load-hoist-depth=1 \
; RUN:   -stop-after=riscv-branch-condition-pipelining < %s \
; RUN:   | FileCheck %s --check-prefix=DEPTH1
; RUN: llc -mtriple=riscv64 -O2 \
; RUN:   -riscv-ns-susan-corners-load-hoist-depth=2 \
; RUN:   -stop-after=riscv-branch-condition-pipelining < %s \
; RUN:   | FileCheck %s --check-prefix=DEPTH2
; RUN: llc -mtriple=riscv64 -O2 \
; RUN:   -riscv-ns-susan-corners-load-hoist-depth=3 \
; RUN:   -stop-after=riscv-branch-condition-pipelining < %s \
; RUN:   | FileCheck %s --check-prefix=DEPTH3

define void @susan_corners(ptr noalias readonly %in, ptr %out,
                           ptr noalias readonly %bp, i32 %max_no, i64 %end) {
entry:
  br label %pre

pre:
  br label %for.body14

for.body14:
  %iv = phi i64 [ 0, %pre ], [ %iv.next, %for.inc ]
  %p0 = getelementptr inbounds i8, ptr %in, i64 %iv
  %v0 = load i8, ptr %p0, align 1
  %i0 = zext i8 %v0 to i64
  %q0 = getelementptr inbounds i8, ptr %bp, i64 %i0
  %t0 = load i8, ptr %q0, align 1
  %s0 = zext i8 %t0 to i32
  %cmp0 = icmp ult i32 %s0, %max_no
  br i1 %cmp0, label %if.then0, label %for.inc

if.then0:
  %p1 = getelementptr inbounds i8, ptr %in, i64 %iv
  %v1 = load i8, ptr %p1, align 1
  %i1 = zext i8 %v1 to i64
  %q1 = getelementptr inbounds i8, ptr %bp, i64 %i1
  %t1 = load i8, ptr %q1, align 1
  %c1 = zext i8 %t1 to i32
  %s1 = add i32 %s0, %c1
  %cmp1 = icmp ult i32 %s1, %max_no
  br i1 %cmp1, label %if.then1, label %for.inc

if.then1:
  %p2 = getelementptr inbounds i8, ptr %in, i64 %iv
  %v2 = load i8, ptr %p2, align 1
  %i2 = zext i8 %v2 to i64
  %q2 = getelementptr inbounds i8, ptr %bp, i64 %i2
  %t2 = load i8, ptr %q2, align 1
  %c2 = zext i8 %t2 to i32
  %s2 = add i32 %s1, %c2
  %cmp2 = icmp ult i32 %s2, %max_no
  br i1 %cmp2, label %if.then2, label %for.inc

if.then2:
  %p3 = getelementptr inbounds i8, ptr %in, i64 %iv
  %v3 = load i8, ptr %p3, align 1
  %i3 = zext i8 %v3 to i64
  %q3 = getelementptr inbounds i8, ptr %bp, i64 %i3
  %t3 = load i8, ptr %q3, align 1
  %c3 = zext i8 %t3 to i32
  %s3 = add i32 %s2, %c3
  %cmp3 = icmp ult i32 %s3, %max_no
  br i1 %cmp3, label %if.then3, label %for.inc

if.then3:
  %p4 = getelementptr inbounds i8, ptr %in, i64 %iv
  %v4 = load i8, ptr %p4, align 1
  %i4 = zext i8 %v4 to i64
  %q4 = getelementptr inbounds i8, ptr %bp, i64 %i4
  %t4 = load i8, ptr %q4, align 1
  %c4 = zext i8 %t4 to i32
  %s4 = add i32 %s3, %c4
  br label %for.inc

for.inc:
  %iv.next = add nuw i64 %iv, 1
  %done = icmp eq i64 %iv.next, %end
  br i1 %done, label %exit, label %for.body14

exit:
  ret void
}

; DEPTH1-LABEL: define void @susan_corners(
; DEPTH1: %p1 = getelementptr
; DEPTH1: %t1 = load i8
; DEPTH1: %cmp0 = icmp

; DEPTH2-LABEL: define void @susan_corners(
; DEPTH2: %p2 = getelementptr
; DEPTH2: %t2 = load i8
; DEPTH2: %cmp0 = icmp

; DEPTH3-LABEL: define void @susan_corners(
; DEPTH3: %p3 = getelementptr
; DEPTH3: %t3 = load i8
; DEPTH3: %cmp0 = icmp
