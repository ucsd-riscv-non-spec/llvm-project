; RUN: llc -mtriple=riscv64 -O2 -riscv-ns-disable-software-pipelining \
; RUN:   -riscv-ns-hoist-chain-loads=2 \
; RUN:   -pass-remarks=riscv-chain-load-hoisting \
; RUN:   -pass-remarks-missed=riscv-chain-load-hoisting -o /dev/null %s 2>&1 \
; RUN:   | FileCheck %s --check-prefix=HOIST
; RUN: llc -mtriple=riscv64 -O2 -riscv-ns-disable-software-pipelining \
; RUN:   -riscv-ns-hoist-chain-loads=2 -o - %s \
; RUN:   | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=riscv64 -O2 -riscv-ns-disable-software-pipelining \
; RUN:   -riscv-ns-hoist-chain-loads=8 \
; RUN:   -pass-remarks=riscv-chain-load-hoisting -o /dev/null %s 2>&1 \
; RUN:   | FileCheck %s --check-prefix=LONG
; RUN: llc -mtriple=riscv64 -O2 -riscv-ns-disable-software-pipelining \
; RUN:   -riscv-ns-hoist-chain-loads=2 \
; RUN:   -riscv-ns-hoist-assume-chain-inbounds \
; RUN:   -pass-remarks=riscv-chain-load-hoisting \
; RUN:   -pass-remarks-missed=riscv-chain-load-hoisting -o /dev/null %s 2>&1 \
; RUN:   | FileCheck %s --check-prefix=CONTRACT
; RUN: llc -mtriple=riscv64 -O2 -riscv-ns-disable-software-pipelining \
; RUN:   -riscv-ns-thin-checks=2 -riscv-ns-hoist-assume-chain-inbounds \
; RUN:   -pass-remarks=riscv-chain-check-thinning \
; RUN:   -pass-remarks-missed=riscv-chain-check-thinning -o /dev/null %s 2>&1 \
; RUN:   | FileCheck %s --check-prefix=THIN
; RUN: llc -mtriple=riscv64 -O2 -riscv-ns-disable-software-pipelining \
; RUN:   -riscv-ns-thin-checks=9 -riscv-ns-hoist-assume-chain-inbounds \
; RUN:   -pass-remarks=riscv-chain-check-thinning -o /dev/null %s 2>&1 \
; RUN:   | FileCheck %s --check-prefix=THINSHORT
; RUN: llc -mtriple=riscv64 -O2 -riscv-ns-disable-software-pipelining \
; RUN:   -riscv-ns-thin-checks=2 \
; RUN:   -pass-remarks-missed=riscv-chain-check-thinning -o /dev/null %s 2>&1 \
; RUN:   | FileCheck %s --check-prefix=THINOFF
; RUN: llc -mtriple=riscv64 -O2 -riscv-ns-disable-software-pipelining \
; RUN:   -riscv-ns-hoist-chain-loads=2 \
; RUN:   -pass-remarks-missed=riscv-chain-load-hoisting -o /dev/null %s 2>&1 \
; RUN:   | FileCheck %s --check-prefix=FLAGOFF

define void @hoist_d2(ptr noalias dereferenceable(64) readonly %in,
                      ptr noalias dereferenceable(256) %bp, ptr noalias %out,
                      i32 %limit, i32 %end) {
entry:
  br label %loop
loop:
  %iv = phi i32 [ 0, %entry ], [ %next, %latch ]
  br label %check0
check0:
  %p0 = getelementptr inbounds i8, ptr %in, i64 0
  %v0 = load i8, ptr %p0, align 1
  %i0 = zext i8 %v0 to i64
  %q0 = getelementptr inbounds i8, ptr %bp, i64 %i0
  %t0 = load i8, ptr %q0, align 1
  %s0 = zext i8 %t0 to i32
  %cmp0 = icmp ult i32 %s0, %limit
  br i1 %cmp0, label %check1, label %latch
check1:
  %p1 = getelementptr inbounds i8, ptr %in, i64 1
  %v1 = load i8, ptr %p1, align 1
  %i1 = zext i8 %v1 to i64
  %q1 = getelementptr inbounds i8, ptr %bp, i64 %i1
  %t1 = load i8, ptr %q1, align 1
  %s1 = zext i8 %t1 to i32
  %sum1 = add i32 %s0, %s1
  %cmp1 = icmp ult i32 %sum1, %limit
  br i1 %cmp1, label %check2, label %latch
check2:
  %p2 = getelementptr inbounds i8, ptr %in, i64 2
  %v2 = load i8, ptr %p2, align 1
  %i2 = zext i8 %v2 to i64
  %q2 = getelementptr inbounds i8, ptr %bp, i64 %i2
  %t2 = load i8, ptr %q2, align 1
  %s2 = zext i8 %t2 to i32
  %sum2 = add i32 %sum1, %s2
  %cmp2 = icmp ult i32 %sum2, %limit
  br label %latch
latch:
  %result = phi i32 [ %s0, %check0 ], [ %sum1, %check1 ],
                    [ %sum2, %check2 ]
  store i32 %result, ptr %out, align 4
  %next = add nuw i32 %iv, 1
  %done = icmp eq i32 %next, %end
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

define void @reject_not_dereferenceable(ptr noalias readonly %in,
                                        ptr noalias %bp, ptr noalias %out,
                                        i32 %limit, i32 %end) {
entry:
  br label %loop
loop:
  %iv = phi i32 [ 0, %entry ], [ %next, %latch ]
  br label %check0
check0:
  %p0 = getelementptr inbounds i8, ptr %in, i64 0
  %v0 = load i8, ptr %p0, align 1
  %i0 = zext i8 %v0 to i64
  %q0 = getelementptr inbounds i8, ptr %bp, i64 %i0
  %t0 = load i8, ptr %q0, align 1
  %s0 = zext i8 %t0 to i32
  %cmp0 = icmp ult i32 %s0, %limit
  br i1 %cmp0, label %check1, label %latch
check1:
  %p1 = getelementptr inbounds i8, ptr %in, i64 1
  %v1 = load i8, ptr %p1, align 1
  %i1 = zext i8 %v1 to i64
  %q1 = getelementptr inbounds i8, ptr %bp, i64 %i1
  %t1 = load i8, ptr %q1, align 1
  %s1 = zext i8 %t1 to i32
  %sum1 = add i32 %s0, %s1
  %cmp1 = icmp ult i32 %sum1, %limit
  br i1 %cmp1, label %check2, label %latch
check2:
  %sum2 = add i32 %sum1, 1
  br label %latch
latch:
  %result = phi i32 [ %s0, %check0 ], [ %sum1, %check1 ],
                    [ %sum2, %check2 ]
  store i32 %result, ptr %out, align 4
  %next = add nuw i32 %iv, 1
  %done = icmp eq i32 %next, %end
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

define void @reject_aliasing_store(ptr noalias dereferenceable(64) readonly %in,
                                   ptr dereferenceable(256) %bp,
                                   ptr noalias %out,
                                   ptr dereferenceable(1) %other, i32 %limit,
                                   i32 %end) {
entry:
  br label %loop
loop:
  %iv = phi i32 [ 0, %entry ], [ %next, %latch ]
  br label %check0
check0:
  %p0 = getelementptr inbounds i8, ptr %in, i64 0
  %v0 = load i8, ptr %p0, align 1
  %i0 = zext i8 %v0 to i64
  %q0 = getelementptr inbounds i8, ptr %bp, i64 %i0
  %t0 = load i8, ptr %q0, align 1
  %s0 = zext i8 %t0 to i32
  %cmp0 = icmp ult i32 %s0, %limit
  br i1 %cmp0, label %check1, label %latch
check1:
  %p1 = getelementptr inbounds i8, ptr %in, i64 1
  %v1 = load i8, ptr %p1, align 1
  %i1 = zext i8 %v1 to i64
  %q1 = getelementptr inbounds i8, ptr %bp, i64 %i1
  store i8 1, ptr %other, align 1
  %t1 = load i8, ptr %q1, align 1
  %s1 = zext i8 %t1 to i32
  %sum1 = add i32 %s0, %s1
  %cmp1 = icmp ult i32 %sum1, %limit
  br i1 %cmp1, label %check2, label %latch
check2:
  %sum2 = add i32 %sum1, 1
  br label %latch
latch:
  %result = phi i32 [ %s0, %check0 ], [ %sum1, %check1 ],
                    [ %sum2, %check2 ]
  store i32 %result, ptr %out, align 4
  %next = add nuw i32 %iv, 1
  %done = icmp eq i32 %next, %end
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

define void @reject_loop_exit(ptr noalias dereferenceable(64) readonly %in,
                              ptr noalias dereferenceable(256) readonly %bp,
                              ptr noalias %out, i32 %limit, i32 %end) {
entry:
  br label %loop
loop:
  %iv = phi i32 [ 0, %entry ], [ %next, %latch ]
  br label %check0
check0:
  %p0 = getelementptr inbounds i8, ptr %in, i64 0
  %v0 = load i8, ptr %p0, align 1
  %i0 = zext i8 %v0 to i64
  %q0 = getelementptr inbounds i8, ptr %bp, i64 %i0
  %t0 = load i8, ptr %q0, align 1
  %s0 = zext i8 %t0 to i32
  %cmp0 = icmp ult i32 %s0, %limit
  br i1 %cmp0, label %check1, label %exit
check1:
  %p1 = getelementptr inbounds i8, ptr %in, i64 1
  %v1 = load i8, ptr %p1, align 1
  %i1 = zext i8 %v1 to i64
  %q1 = getelementptr inbounds i8, ptr %bp, i64 %i1
  %t1 = load i8, ptr %q1, align 1
  %s1 = zext i8 %t1 to i32
  %sum1 = add i32 %s0, %s1
  %cmp1 = icmp ult i32 %sum1, %limit
  br i1 %cmp1, label %latch, label %exit
latch:
  %next = add nuw i32 %iv, 1
  %done = icmp eq i32 %next, %end
  br i1 %done, label %exit, label %loop
exit:
  %result = phi i32 [ %s0, %check0 ], [ %sum1, %check1 ],
                    [ %sum1, %latch ]
  store i32 %result, ptr %out, align 4
  ret void
}

define void @contract_d2(ptr %image, ptr %lut, i32 %x, i32 %y,
                         i32 %limit, i32 %end) {
entry:
  br label %loop
loop:
  %iv = phi i32 [ 0, %entry ], [ %next, %latch ]
  %center.ptr = getelementptr i8, ptr %image, i64 0, !riscv.ns.chain.inbounds !0
  %center = load i8, ptr %center.ptr, align 1, !riscv.ns.chain.inbounds !0, !riscv.ns.chain.offset !2
  %center.idx = zext i8 %center to i64, !riscv.ns.chain.inbounds !1
  %lut.base = getelementptr i8, ptr %lut, i64 %center.idx, !riscv.ns.chain.inbounds !1
  br label %check0
check0:
  %p0 = getelementptr i8, ptr %image, i64 1, !riscv.ns.chain.inbounds !0
  %v0 = load i8, ptr %p0, align 1, !riscv.ns.chain.inbounds !0, !riscv.ns.chain.offset !3
  %i0 = zext i8 %v0 to i64, !riscv.ns.chain.inbounds !1
  %n0 = sub i64 0, %i0, !riscv.ns.chain.inbounds !1
  %q0 = getelementptr i8, ptr %lut.base, i64 %n0, !riscv.ns.chain.inbounds !1
  %t0 = load i8, ptr %q0, align 1, !riscv.ns.chain.inbounds !4
  %s0 = zext i8 %t0 to i32
  %cmp0 = icmp ult i32 %s0, %limit
  br i1 %cmp0, label %check1, label %latch
check1:
  %p1 = getelementptr i8, ptr %image, i64 2, !riscv.ns.chain.inbounds !0
  %v1 = load i8, ptr %p1, align 1, !riscv.ns.chain.inbounds !0, !riscv.ns.chain.offset !5
  %i1 = zext i8 %v1 to i64, !riscv.ns.chain.inbounds !1
  %n1 = sub i64 0, %i1, !riscv.ns.chain.inbounds !1
  %q1 = getelementptr i8, ptr %lut.base, i64 %n1, !riscv.ns.chain.inbounds !1
  %t1 = load i8, ptr %q1, align 1, !riscv.ns.chain.inbounds !4
  %s1 = zext i8 %t1 to i32
  %sum1 = add i32 %s0, %s1
  %cmp1 = icmp ult i32 %sum1, %limit
  br i1 %cmp1, label %check2, label %latch
check2:
  %p2 = getelementptr i8, ptr %image, i64 3, !riscv.ns.chain.inbounds !0
  %v2 = load i8, ptr %p2, align 1, !riscv.ns.chain.inbounds !0, !riscv.ns.chain.offset !9
  %i2 = zext i8 %v2 to i64, !riscv.ns.chain.inbounds !1
  %n2 = sub i64 0, %i2, !riscv.ns.chain.inbounds !1
  %q2 = getelementptr i8, ptr %lut.base, i64 %n2, !riscv.ns.chain.inbounds !1
  %t2 = load i8, ptr %q2, align 1, !riscv.ns.chain.inbounds !4
  %s2 = zext i8 %t2 to i32
  %sum2 = add i32 %sum1, %s2
  %cmp2 = icmp ult i32 %sum2, %limit
  br i1 %cmp2, label %check3, label %latch
check3:
  br label %latch
latch:
  %next = add nuw i32 %iv, 1
  %done = icmp eq i32 %next, %end
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

define void @contract_missing_metadata(ptr noalias readonly %in,
                                       ptr noalias %bp, ptr noalias %out,
                                       i32 %limit, i32 %end) {
entry:
  br label %loop
loop:
  %iv = phi i32 [ 0, %entry ], [ %next, %latch ]
  br label %check0
check0:
  %p0 = getelementptr inbounds i8, ptr %in, i64 0
  %v0 = load i8, ptr %p0, align 1
  %i0 = zext i8 %v0 to i64
  %q0 = getelementptr inbounds i8, ptr %bp, i64 %i0
  %t0 = load i8, ptr %q0, align 1
  %s0 = zext i8 %t0 to i32
  %cmp0 = icmp ult i32 %s0, %limit
  br i1 %cmp0, label %check1, label %latch
check1:
  %p1 = getelementptr inbounds i8, ptr %in, i64 1
  %v1 = load i8, ptr %p1, align 1
  %i1 = zext i8 %v1 to i64
  %q1 = getelementptr inbounds i8, ptr %bp, i64 %i1
  %t1 = load i8, ptr %q1, align 1
  %s1 = zext i8 %t1 to i32
  %sum1 = add i32 %s0, %s1
  %cmp1 = icmp ult i32 %sum1, %limit
  br i1 %cmp1, label %check2, label %latch
check2:
  %sum2 = add i32 %sum1, 1
  br label %latch
latch:
  %result = phi i32 [ %s0, %check0 ], [ %sum1, %check1 ],
                    [ %sum2, %check2 ]
  store i32 %result, ptr %out, align 4
  %next = add nuw i32 %iv, 1
  %done = icmp eq i32 %next, %end
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

define void @thin_nonmonotonic(ptr noalias dereferenceable(4) readonly %in,
                               i32 %limit, i32 %end) {
entry:
  br label %loop
loop:
  %iv = phi i32 [ 0, %entry ], [ %next, %latch ]
  br label %check0
check0:
  %p0 = getelementptr inbounds i8, ptr %in, i64 0
  %v0 = load i8, ptr %p0, align 1, !riscv.ns.chain.inbounds !10
  %s0 = zext i8 %v0 to i32
  %cmp0 = icmp ult i32 %s0, %limit
  br i1 %cmp0, label %latch, label %check1
check1:
  %p1 = getelementptr inbounds i8, ptr %in, i64 1
  %v1 = load i8, ptr %p1, align 1, !riscv.ns.chain.inbounds !10
  %s1 = zext i8 %v1 to i32
  %sum1 = add i32 %s0, %s1
  %cmp1 = icmp ult i32 %sum1, %limit
  br i1 %cmp1, label %latch, label %check2
check2:
  %p2 = getelementptr inbounds i8, ptr %in, i64 2
  %v2 = load i8, ptr %p2, align 1, !riscv.ns.chain.inbounds !10
  %s2 = zext i8 %v2 to i32
  %sum2 = add i32 %sum1, %s2
  %cmp2 = icmp ult i32 %sum2, %limit
  br i1 %cmp2, label %latch, label %check3
check3:
  br label %latch
latch:
  %next = add nuw i32 %iv, 1
  %done = icmp eq i32 %next, %end
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

define void @thin_reject_unsafe_load(ptr noalias readonly %in,
                                     i32 %limit, i32 %end) {
entry:
  br label %loop
loop:
  %iv = phi i32 [ 0, %entry ], [ %next, %latch ]
  br label %check0
check0:
  %p0 = getelementptr inbounds i8, ptr %in, i64 0
  %v0 = load i8, ptr %p0, align 1, !riscv.ns.chain.inbounds !10
  %s0 = zext i8 %v0 to i32
  %cmp0 = icmp ult i32 %s0, %limit
  br i1 %cmp0, label %check1, label %latch
check1:
  %p1 = getelementptr inbounds i8, ptr %in, i64 1
  %v1 = load i8, ptr %p1, align 1, !riscv.ns.chain.inbounds !10
  %s1 = zext i8 %v1 to i32
  %sum1 = add i32 %s0, %s1
  %cmp1 = icmp ult i32 %sum1, %limit
  br i1 %cmp1, label %check2, label %latch
check2:
  %p2 = getelementptr inbounds i8, ptr %in, i64 2
  %v2 = load i8, ptr %p2, align 1, !riscv.ns.chain.inbounds !10
  %s2 = zext i8 %v2 to i32
  %sum2 = add i32 %sum1, %s2
  %cmp2 = icmp ult i32 %sum2, %limit
  br i1 %cmp2, label %check3, label %latch
check3:
  br label %latch
latch:
  %next = add nuw i32 %iv, 1
  %done = icmp eq i32 %next, %end
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

define void @thin_reject_exit_liveout(
    ptr noalias dereferenceable(4) readonly %in,
    ptr noalias %out, i32 %limit, i32 %end) {
entry:
  br label %loop
loop:
  %iv = phi i32 [ 0, %entry ], [ %next, %latch ]
  br label %check0
check0:
  %p0 = getelementptr inbounds i8, ptr %in, i64 0
  %v0 = load i8, ptr %p0, align 1, !riscv.ns.chain.inbounds !10
  %s0 = zext i8 %v0 to i32
  %cmp0 = icmp ult i32 %s0, %limit
  br i1 %cmp0, label %check1, label %latch
check1:
  %p1 = getelementptr inbounds i8, ptr %in, i64 1
  %v1 = load i8, ptr %p1, align 1, !riscv.ns.chain.inbounds !10
  %s1 = zext i8 %v1 to i32
  %sum1 = add i32 %s0, %s1
  %cmp1 = icmp ult i32 %sum1, %limit
  br i1 %cmp1, label %check2, label %latch
check2:
  %p2 = getelementptr inbounds i8, ptr %in, i64 2
  %v2 = load i8, ptr %p2, align 1, !riscv.ns.chain.inbounds !10
  %s2 = zext i8 %v2 to i32
  %sum2 = add i32 %sum1, %s2
  %cmp2 = icmp ult i32 %sum2, %limit
  br i1 %cmp2, label %check3, label %latch
check3:
  br label %latch
latch:
  %result = phi i32 [ %s0, %check0 ], [ %sum1, %check1 ],
                    [ %sum2, %check2 ], [ %sum2, %check3 ]
  store i32 %result, ptr %out, align 4
  %next = add nuw i32 %iv, 1
  %done = icmp eq i32 %next, %end
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

define void @thin_reject_memory(ptr noalias dereferenceable(4) readonly %in,
                                ptr noalias %out, i32 %limit, i32 %end) {
entry:
  br label %loop
loop:
  %iv = phi i32 [ 0, %entry ], [ %next, %latch ]
  br label %check0
check0:
  %p0 = getelementptr inbounds i8, ptr %in, i64 0
  %v0 = load i8, ptr %p0, align 1, !riscv.ns.chain.inbounds !10
  %s0 = zext i8 %v0 to i32
  %cmp0 = icmp ult i32 %s0, %limit
  br i1 %cmp0, label %check1, label %latch
check1:
  store i8 0, ptr %out, align 1
  %p1 = getelementptr inbounds i8, ptr %in, i64 1
  %v1 = load i8, ptr %p1, align 1, !riscv.ns.chain.inbounds !10
  %s1 = zext i8 %v1 to i32
  %sum1 = add i32 %s0, %s1
  %cmp1 = icmp ult i32 %sum1, %limit
  br i1 %cmp1, label %check2, label %latch
check2:
  %p2 = getelementptr inbounds i8, ptr %in, i64 2
  %v2 = load i8, ptr %p2, align 1, !riscv.ns.chain.inbounds !10
  %s2 = zext i8 %v2 to i32
  %sum2 = add i32 %sum1, %s2
  %cmp2 = icmp ult i32 %sum2, %limit
  br i1 %cmp2, label %check3, label %latch
check3:
  br label %latch
latch:
  %next = add nuw i32 %iv, 1
  %done = icmp eq i32 %next, %end
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

define void @thin_reject_negative(ptr noalias dereferenceable(4) readonly %in,
                                  i32 %limit, i32 %end) {
entry:
  br label %loop
loop:
  %iv = phi i32 [ 0, %entry ], [ %next, %latch ]
  br label %check0
check0:
  %p0 = getelementptr inbounds i8, ptr %in, i64 0
  %v0 = load i8, ptr %p0, align 1, !riscv.ns.chain.inbounds !10
  %s0 = zext i8 %v0 to i32
  %cmp0 = icmp ult i32 %s0, %limit
  br i1 %cmp0, label %check1, label %latch
check1:
  %p1 = getelementptr inbounds i8, ptr %in, i64 1
  %v1 = load i8, ptr %p1, align 1, !riscv.ns.chain.inbounds !10
  %s1 = sext i8 %v1 to i32
  %sum1 = add i32 %s0, %s1
  %cmp1 = icmp ult i32 %sum1, %limit
  br i1 %cmp1, label %check2, label %latch
check2:
  %p2 = getelementptr inbounds i8, ptr %in, i64 2
  %v2 = load i8, ptr %p2, align 1, !riscv.ns.chain.inbounds !10
  %s2 = zext i8 %v2 to i32
  %sum2 = add i32 %sum1, %s2
  %cmp2 = icmp ult i32 %sum2, %limit
  br i1 %cmp2, label %check3, label %latch
check3:
  br label %latch
latch:
  %next = add nuw i32 %iv, 1
  %done = icmp eq i32 %next, %end
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

define void @thin_reject_overflow(ptr noalias dereferenceable(4) readonly %in,
                                  i9 %limit, i32 %end) {
entry:
  br label %loop
loop:
  %iv = phi i32 [ 0, %entry ], [ %next, %latch ]
  br label %check0
check0:
  %p0 = getelementptr inbounds i8, ptr %in, i64 0
  %v0 = load i8, ptr %p0, align 1, !riscv.ns.chain.inbounds !10
  %s0 = zext i8 %v0 to i9
  %cmp0 = icmp ult i9 %s0, %limit
  br i1 %cmp0, label %check1, label %latch
check1:
  %p1 = getelementptr inbounds i8, ptr %in, i64 1
  %v1 = load i8, ptr %p1, align 1, !riscv.ns.chain.inbounds !10
  %s1 = zext i8 %v1 to i9
  %sum1 = add i9 %s0, %s1
  %cmp1 = icmp ult i9 %sum1, %limit
  br i1 %cmp1, label %check2, label %latch
check2:
  %p2 = getelementptr inbounds i8, ptr %in, i64 2
  %v2 = load i8, ptr %p2, align 1, !riscv.ns.chain.inbounds !10
  %s2 = zext i8 %v2 to i9
  %sum2 = add i9 %sum1, %s2
  %cmp2 = icmp ult i9 %sum2, %limit
  br i1 %cmp2, label %check3, label %latch
check3:
  br label %latch
latch:
  %next = add nuw i32 %iv, 1
  %done = icmp eq i32 %next, %end
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

; HOIST: hoisted 1 early-exit load slice(s) with distance 2
; HOIST: did not hoist a load slice in reject_not_dereferenceable: load{{.*}}not proven dereferenceable at the hoist point
; HOIST: did not hoist a load slice in reject_aliasing_store: a call, fence, or may-alias write lies on the crossed path
; HOIST: load hoisting does not cross a loop-exit or loop-guard branch
; LONG: hoisted 1 early-exit load slice(s) with distance 8
; CONTRACT: hoisted 1 early-exit load slice(s) with distance 2
; CONTRACT: did not hoist a load slice in contract_d2: load 't2'
; CONTRACT: load hoisting does not cross a loop-exit or loop-guard branch
; CONTRACT: did not hoist a load slice in contract_missing_metadata:
; FLAGOFF: did not hoist a load slice in contract_d2: load 't1'
; THIN: will remove 1 early-exit threshold check(s) at spacing 2
; THIN: check thinning does not cross a loop-exit or loop-guard branch
; THIN: kept a threshold check in thin_nonmonotonic: the exit condition is not monotone
; THIN: kept a threshold check in thin_reject_unsafe_load: a load is not safe on the newly reachable exit path
; THIN: kept a threshold check in thin_reject_exit_liveout: an intermediate value is live on the early-exit path
; THIN: kept a threshold check in thin_reject_memory: a store, call, fence, volatile, or atomic access lies between checks
; THIN: kept a threshold check in thin_reject_negative: a contribution is not proven nonnegative
; THIN: kept a threshold check in thin_reject_overflow: accumulator may overflow
; THINSHORT: will remove 1 early-exit threshold check(s) at spacing 9
; THINOFF: kept a threshold check in contract_d2: a load is not safe
; ASM-LABEL: hoist_d2:
; ASM: lbu {{.*}}, 1(a0)
; ASM: lbu {{.*}}, 0({{.*}})
; ASM: bmovc_bgeu

!0 = !{!"v1", !"image", i64 0, i64 0, i64 2, i64 3, !"loop", !"",
        i64 5, i64 5, i64 5, i64 5, !7}
!1 = !{!"v1", !"lut", i64 1, i64 0, i64 2, i64 3, !"loop", !"",
        i64 5, i64 5, i64 5, i64 5, !8}
!2 = !{i64 0, i64 0}
!3 = !{i64 0, i64 1}
!4 = !{!"v1", !"lut", i64 1, i64 0, i64 2, i64 3, !"loop", !"",
        i64 5, i64 5, i64 5, i64 5, !8}
!5 = !{i64 0, i64 2}
!6 = !{i64 0, i64 3}
!7 = !{i64 -3, i64 3, i64 -3, i64 3}
!8 = !{i64 -255, i64 0}
!9 = !{i64 0, i64 4}
!10 = !{!"thin-test"}
