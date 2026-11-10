; RUN: llc -mtriple=riscv64 -O2 -riscv-ns-swp-distance=4 \
; RUN:   -riscv-ns-swp-speculative-distance=4 \
; RUN:   -riscv-ns-swp-speculative -riscv-ns-swp-shift-recurrences \
; RUN:   -stop-after=riscv-branch-condition-pipelining < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -O2 -riscv-ns-swp-distance=16 \
; RUN:   -riscv-ns-swp-speculative-distance=16 \
; RUN:   -riscv-ns-swp-speculative -riscv-ns-swp-shift-recurrences \
; RUN:   -stop-after=riscv-branch-condition-pipelining < %s \
; RUN:   | FileCheck %s --check-prefix=D16
; RUN: llc -mtriple=riscv64 -O2 -riscv-ns-swp-distance=4 \
; RUN:   -riscv-ns-swp-speculative-distance=4 \
; RUN:   -riscv-ns-swp-speculative \
; RUN:   -stop-after=riscv-branch-condition-pipelining < %s \
; RUN:   | FileCheck %s --check-prefix=OFF
; RUN: llc -mtriple=riscv64 -O2 \
; RUN:   -riscv-ns-swp-speculative -riscv-ns-swp-shift-recurrences \
; RUN:   -stop-after=riscv-branch-condition-pipelining < %s \
; RUN:   | FileCheck %s --check-prefix=DEF

; Tests -riscv-ns-swp-shift-recurrences: a value shifted by the same constant
; every iteration is advanced to a later iteration by one shift of the total
; amount, so conditions on it can be computed ahead.

; ntbl_bitcnt in MiBench bitcount, after tail recursion elimination:
; do { cnt += bits[x & 15]; x >>= 4; } while (x >= 16) (signed).
@bits = global [16 x i8] zeroinitializer

; CHECK-LABEL: define {{.*}} @ashr_by_4(
; CHECK:       entry:
; CHECK-NEXT:    call void @llvm.riscv.ns.cond.init(i32 0)
; CHECK-NEXT:    %x.next.swp0 = ashr i64 %x0, 4
; CHECK-NEXT:    call void @llvm.riscv.ns.cond.push.i64(i32 0, i32 36, i64 %x.next.swp0, i64 16)
; CHECK-NEXT:    %x.swp1 = ashr i64 %x0, 4
; CHECK-NEXT:    %x.next.swp1 = ashr i64 %x.swp1, 4
; CHECK:         %x.swp2 = ashr i64 %x0, 8
; CHECK:         %x.swp3 = ashr i64 %x0, 12
; CHECK:       loop:
; CHECK:         %x.next = ashr i64 %x, 4
; CHECK:         %x.ahead = ashr i64 %x, 16
; CHECK-NEXT:    %x.next.ahead = ashr i64 %x.ahead, 4
; CHECK-NEXT:    call void @llvm.riscv.ns.cond.push.i64(i32 0, i32 36, i64 %x.next.ahead, i64 16)
; CHECK-NEXT:    [[POP:%.*]] = call i1 @llvm.riscv.ns.cond.pop(i32 0)
; CHECK-NEXT:    br i1 [[POP]], label %exit, label %loop
; With no distance given, a loop without a trip count is 4 ahead.
; DEF-LABEL:   define {{.*}} @ashr_by_4(
; DEF:         loop:
; DEF:           %x.ahead = ashr i64 %x, 16
; DEF-NEXT:      %x.next.ahead = ashr i64 %x.ahead, 4
; OFF-LABEL:   define {{.*}} @ashr_by_4(
; OFF-NOT:       llvm.riscv.ns.cond
define i32 @ashr_by_4(i64 %x0) {
entry:
  br label %loop

loop:
  %x = phi i64 [ %x0, %entry ], [ %x.next, %loop ]
  %cnt = phi i32 [ 0, %entry ], [ %cnt.next, %loop ]
  %x.next = ashr i64 %x, 4
  %idx = and i64 %x.next, 15
  %q = getelementptr inbounds [16 x i8], ptr @bits, i64 0, i64 %idx
  %v = load i8, ptr %q
  %ve = zext i8 %v to i32
  %cnt.next = add nuw nsw i32 %cnt, %ve
  %c = icmp ult i64 %x.next, 16
  br i1 %c, label %exit, label %loop

exit:
  ret i32 %cnt.next
}

; bit_shifter in MiBench bitcount, with its two exits already merged:
; for (i = n = 0; x && i < 64; ++i, x >>= 1) n += x & 1;
; CHECK-LABEL: define {{.*}} @merged_exits(
; CHECK:       loop:
; CHECK:         %i.ahead = add i32 %i, 4
; CHECK-NEXT:    %x.ahead = ashr i64 %x, 4
; CHECK-NEXT:    %nz.ahead = icmp ugt i64 %x.ahead, 1
; The clone has no samesign.
; CHECK-NEXT:    %lt.ahead = icmp ult i32 %i.ahead, 63
; CHECK-NEXT:    %c.ahead = select i1 %nz.ahead, i1 %lt.ahead, i1 false
; CHECK:         [[POP:%.*]] = call i1 @llvm.riscv.ns.cond.pop(i32 0)
; CHECK-NEXT:    br i1 [[POP]], label %loop, label %exit
; OFF-LABEL:   define {{.*}} @merged_exits(
; OFF-NOT:       llvm.riscv.ns.cond
define i32 @merged_exits(i64 %x0) {
entry:
  br label %loop

loop:
  %n = phi i32 [ 0, %entry ], [ %n.next, %loop ]
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %x = phi i64 [ %x0, %entry ], [ %x.next, %loop ]
  %lo = trunc i64 %x to i32
  %bit = and i32 %lo, 1
  %n.next = add nuw nsw i32 %n, %bit
  %i.next = add nuw nsw i32 %i, 1
  %x.next = ashr i64 %x, 1
  %nz = icmp ugt i64 %x, 1
  %lt = icmp samesign ult i32 %i, 63
  %c = select i1 %nz, i1 %lt, i1 false
  br i1 %c, label %loop, label %exit

exit:
  ret i32 %n.next
}

; while (x != 0) { k++; x >>= 1; } (unsigned)
; CHECK-LABEL: define {{.*}} @lshr_by_1(
; CHECK:       loop:
; The original keeps exact; the clone does not.
; CHECK:         %x.next = lshr exact i64 %x, 1
; CHECK:         %x.ahead = lshr i64 %x, 4
; CHECK-NEXT:    %x.next.ahead = lshr i64 %x.ahead, 1
; CHECK:         call i1 @llvm.riscv.ns.cond.pop(i32 0)
; OFF-LABEL:   define {{.*}} @lshr_by_1(
; OFF-NOT:       llvm.riscv.ns.cond
define i64 @lshr_by_1(i64 %x0) {
entry:
  br label %loop

loop:
  %x = phi i64 [ %x0, %entry ], [ %x.next, %loop ]
  %k = phi i64 [ 0, %entry ], [ %k.next, %loop ]
  %k.next = add nuw nsw i64 %k, 1
  %x.next = lshr exact i64 %x, 1
  %c = icmp ne i64 %x.next, 0
  br i1 %c, label %loop, label %exit

exit:
  ret i64 %k.next
}

; while (x != 0) { k++; x <<= 3; }
; CHECK-LABEL: define {{.*}} @shl_by_3(
; CHECK:         %x.swp3 = shl i64 %x0, 9
; CHECK:       loop:
; CHECK:         %x.next = shl nuw i64 %x, 3
; CHECK:         %x.ahead = shl i64 %x, 12
; CHECK-NEXT:    %x.next.ahead = shl i64 %x.ahead, 3
; CHECK:         call i1 @llvm.riscv.ns.cond.pop(i32 0)
; OFF-LABEL:   define {{.*}} @shl_by_3(
; OFF-NOT:       llvm.riscv.ns.cond
define i64 @shl_by_3(i64 %x0) {
entry:
  br label %loop

loop:
  %x = phi i64 [ %x0, %entry ], [ %x.next, %loop ]
  %k = phi i64 [ 0, %entry ], [ %k.next, %loop ]
  %k.next = add nuw nsw i64 %k, 1
  %x.next = shl nuw i64 %x, 3
  %c = icmp ne i64 %x.next, 0
  br i1 %c, label %loop, label %exit

exit:
  ret i64 %k.next
}

; An i8 shifted by 1: at distance 16, the conditions of iterations 8 and
; later are computed on a value shifted by 8 or more, which is 0 for lshr.
; The condition also depends on k, so that the loop can run longer than 8
; iterations (a loop that never runs more than D is not pipelined).
; D16-LABEL:   define {{.*}} @lshr_past_width(
; D16:           %x.swp7 = lshr i8 %x0, 7
; D16:           %xe.swp8 = zext i8 0 to i64
; D16:           %xe.swp15 = zext i8 0 to i64
; D16:         loop:
; D16:           %xe.ahead = zext i8 0 to i64
; D16:           call i1 @llvm.riscv.ns.cond.pop(i32 0)
define i64 @lshr_past_width(i8 %x0, i64 %n) {
entry:
  br label %loop

loop:
  %x = phi i8 [ %x0, %entry ], [ %x.next, %loop ]
  %k = phi i64 [ 0, %entry ], [ %k.next, %loop ]
  %k.next = add nuw nsw i64 %k, 1
  %x.next = lshr i8 %x, 1
  %xe = zext i8 %x to i64
  %sq = mul i64 %k, %k
  %s = add i64 %xe, %sq
  %c = icmp ult i64 %s, %n
  br i1 %c, label %loop, label %exit

exit:
  ret i64 %k.next
}

; The same with ashr, which keeps only the sign: shifted by 7.
; D16-LABEL:   define {{.*}} @ashr_past_width(
; D16:           %x.swp6 = ashr i8 %x0, 6
; D16:           %x.swp7 = ashr i8 %x0, 7
; D16:           %x.swp8 = ashr i8 %x0, 7
; D16:           %x.swp15 = ashr i8 %x0, 7
; D16:         loop:
; D16:           %x.ahead = ashr i8 %x, 7
; D16:           call i1 @llvm.riscv.ns.cond.pop(i32 0)
define i64 @ashr_past_width(i8 %x0) {
entry:
  br label %loop

loop:
  %x = phi i8 [ %x0, %entry ], [ %x.next, %loop ]
  %k = phi i64 [ 0, %entry ], [ %k.next, %loop ]
  %k.next = add nuw nsw i64 %k, 1
  %x.next = ashr i8 %x, 1
  %c = icmp sgt i8 %x, 0
  br i1 %c, label %loop, label %exit

exit:
  ret i64 %k.next
}

; A loop with a trip count takes the kernel path, and its branch on a shift
; recurrence is computed ahead there too.
; for (i = 0; i < n; i++) { if (x & 1) *p = i; x >>= 1; }
; CHECK-LABEL: define {{.*}} @counted_with_shift(
; CHECK:         %swp.long.enough = icmp ugt i64 %n, 4
; CHECK:       swp.ph:
; CHECK:         %x.swp3 = lshr i64 %x0, 3
; CHECK:         %swp.kernel.iv = phi i64
; CHECK:         %x.ahead = lshr i64 %x.swp, 4
; CHECK:         call i1 @llvm.riscv.ns.cond.pop(i32 0)
; CHECK:       swp.drain.ph:
; CHECK-NEXT:    %{{.*}} = phi i64
; CHECK:         %x.swp.resume = phi i64 [ %x0, %loop.preheader ], [ %x.next.swp, %swp.kernel.exit ]
; With no distance given, a counted loop stays 16 ahead.
; DEF-LABEL:   define {{.*}} @counted_with_shift(
; DEF:           %swp.long.enough = icmp ugt i64 %n, 16
; DEF:           %x.ahead = lshr i64 %x.swp, 16
; OFF-LABEL:   define {{.*}} @counted_with_shift(
; OFF-NOT:       llvm.riscv.ns.cond
define void @counted_with_shift(ptr %p, i64 %n, i64 %x0) {
entry:
  %z = icmp sgt i64 %n, 0
  br i1 %z, label %loop, label %exit

loop:
  %i = phi i64 [ 0, %entry ], [ %inc, %latch ]
  %x = phi i64 [ %x0, %entry ], [ %x.next, %latch ]
  %b = and i64 %x, 1
  %t = icmp ne i64 %b, 0
  br i1 %t, label %store, label %latch

store:
  store i64 %i, ptr %p
  br label %latch

latch:
  %x.next = lshr i64 %x, 1
  %inc = add nuw nsw i64 %i, 1
  %c = icmp slt i64 %inc, %n
  br i1 %c, label %loop, label %exit

exit:
  ret void
}

; Not a constant shift: the total cannot be computed.
; CHECK-LABEL: define {{.*}} @reject_variable_shift(
; CHECK-NOT:     llvm.riscv.ns.cond
define i64 @reject_variable_shift(i64 %x0, i64 %s) {
entry:
  br label %loop

loop:
  %x = phi i64 [ %x0, %entry ], [ %x.next, %loop ]
  %k = phi i64 [ 0, %entry ], [ %k.next, %loop ]
  %k.next = add nuw nsw i64 %k, 1
  %x.next = lshr i64 %x, %s
  %c = icmp ne i64 %x.next, 0
  br i1 %c, label %loop, label %exit

exit:
  ret i64 %k.next
}

; The shift is of another value than the phi.
; CHECK-LABEL: define {{.*}} @reject_not_own_shift(
; CHECK-NOT:     llvm.riscv.ns.cond
define i64 @reject_not_own_shift(i64 %x0) {
entry:
  br label %loop

loop:
  %x = phi i64 [ %x0, %entry ], [ %x.next, %loop ]
  %k = phi i64 [ 0, %entry ], [ %k.next, %loop ]
  %k.next = add nuw nsw i64 %k, 1
  %y = add i64 %x, 1
  %x.next = lshr i64 %y, 1
  %c = icmp ugt i64 %x.next, 1
  br i1 %c, label %loop, label %exit

exit:
  ret i64 %k.next
}

; Two shifts in a row: the update is not one shift of the phi.
; CHECK-LABEL: define {{.*}} @reject_double_shift(
; CHECK-NOT:     llvm.riscv.ns.cond
define i64 @reject_double_shift(i64 %x0) {
entry:
  br label %loop

loop:
  %x = phi i64 [ %x0, %entry ], [ %x.next, %loop ]
  %k = phi i64 [ 0, %entry ], [ %k.next, %loop ]
  %k.next = add nuw nsw i64 %k, 1
  %h = lshr i64 %x, 1
  %x.next = lshr i64 %h, 2
  %c = icmp ne i64 %x.next, 0
  br i1 %c, label %loop, label %exit

exit:
  ret i64 %k.next
}

; bit_count in MiBench bitcount: x &= x - 1 has no closed form.
; CHECK-LABEL: define {{.*}} @reject_clear_lowest(
; CHECK-NOT:     llvm.riscv.ns.cond
; CHECK:         ret i32 %n.next
define i32 @reject_clear_lowest(i64 %x0) {
entry:
  br label %loop

loop:
  %x = phi i64 [ %x0, %entry ], [ %x.next, %loop ]
  %n = phi i32 [ 0, %entry ], [ %n.next, %loop ]
  %n.next = add nuw nsw i32 %n, 1
  %m = add i64 %x, -1
  %x.next = and i64 %m, %x
  %c = icmp eq i64 %x.next, 0
  br i1 %c, label %exit, label %loop

exit:
  ret i32 %n.next
}
