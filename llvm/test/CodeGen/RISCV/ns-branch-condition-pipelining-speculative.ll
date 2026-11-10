; RUN: llc -mtriple=riscv64 -O2 -riscv-ns-swp-distance=4 \
; RUN:   -riscv-ns-swp-speculative-distance=4 \
; RUN:   -riscv-ns-swp-speculative \
; RUN:   -stop-after=riscv-branch-condition-pipelining < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -O2 -riscv-ns-swp-distance=1 \
; RUN:   -riscv-ns-swp-speculative-distance=1 \
; RUN:   -riscv-ns-swp-speculative \
; RUN:   -stop-after=riscv-branch-condition-pipelining < %s \
; RUN:   | FileCheck %s --check-prefix=D1
; RUN: llc -mtriple=riscv64 -O2 -riscv-ns-swp-distance=62 \
; RUN:   -riscv-ns-swp-speculative-distance=62 \
; RUN:   -riscv-ns-swp-speculative \
; RUN:   -stop-after=riscv-branch-condition-pipelining < %s \
; RUN:   | FileCheck %s --check-prefix=D62
; RUN: llc -mtriple=riscv64 -O2 -riscv-ns-swp-speculative \
; RUN:   -stop-after=riscv-branch-condition-pipelining < %s \
; RUN:   | FileCheck %s --check-prefix=DEF
; RUN: llc -mtriple=riscv64 -O2 -riscv-ns-swp-distance=4 \
; RUN:   -stop-after=riscv-branch-condition-pipelining < %s \
; RUN:   | FileCheck %s --check-prefix=OFF

; Tests -riscv-ns-swp-speculative: loops without a computable trip count are
; pipelined in place when their conditions are safe to compute for
; iterations that never run, and are left alone otherwise.

declare void @ext(i64)

; long i = 0; while (i * i < n) i++; return i;
; CHECK-LABEL: define {{.*}} @while_square(
; No guard, kernel copy, counter or drain loop.
; CHECK-NOT:     swp.long.enough
; The preheader queues the conditions of iterations 0-3.
; CHECK:       entry:
; CHECK-NEXT:    call void @llvm.riscv.ns.cond.init(i32 0)
; CHECK-NEXT:    %sq.swp0 = mul i64 0, 0
; CHECK-NEXT:    call void @llvm.riscv.ns.cond.push.i64(i32 0, i32 40, i64 %sq.swp0, i64 %n)
; CHECK-COUNT-3: call void @llvm.riscv.ns.cond.push.i64(i32 0, i32 40,
; CHECK-NEXT:    br label %loop
; The latch queues iteration i + 4 and branches on the condition of i.
; CHECK:       loop:
; CHECK-NEXT:    %i = phi i64 [ 0, %entry ], [ %inc, %loop ]
; CHECK-NEXT:    %inc = add nuw nsw i64 %i, 1
; CHECK-NEXT:    %i.ahead = add i64 %i, 4
; CHECK-NEXT:    %sq.ahead = mul i64 %i.ahead, %i.ahead
; CHECK-NEXT:    call void @llvm.riscv.ns.cond.push.i64(i32 0, i32 40, i64 %sq.ahead, i64 %n)
; CHECK-NEXT:    [[POP:%.*]] = call i1 @llvm.riscv.ns.cond.pop(i32 0)
; CHECK-NEXT:    br i1 [[POP]], label %loop, label %exit
; CHECK:       exit:
; CHECK-NEXT:    ret i64 %i
; CHECK-NOT:     swp.kernel
; CHECK-NOT:     swp.drain
; D1-LABEL:    define {{.*}} @while_square(
; D1:          entry:
; D1-NEXT:       call void @llvm.riscv.ns.cond.init(i32 0)
; D1-NEXT:       %sq.swp0 = mul i64 0, 0
; D1-NEXT:       call void @llvm.riscv.ns.cond.push.i64(i32 0, i32 40, i64 %sq.swp0, i64 %n)
; D1-NEXT:       br label %loop
; D1:            %i.ahead = add i64 %i, 1
; D62-LABEL:   define {{.*}} @while_square(
; D62:         entry:
; D62-NEXT:      call void @llvm.riscv.ns.cond.init(i32 0)
; D62-COUNT-62:  call void @llvm.riscv.ns.cond.push.i64(i32 0, i32 40,
; D62-NEXT:      br label %loop
; D62:           %i.ahead = add i64 %i, 62
; With no distance given, a loop without a trip count is 4 ahead.
; DEF-LABEL:   define {{.*}} @while_square(
; DEF:         entry:
; DEF-NEXT:      call void @llvm.riscv.ns.cond.init(i32 0)
; DEF-NEXT:      %sq.swp0 = mul i64 0, 0
; DEF-NEXT:      call void @llvm.riscv.ns.cond.push.i64(i32 0, i32 40, i64 %sq.swp0, i64 %n)
; DEF-COUNT-3:   call void @llvm.riscv.ns.cond.push.i64(i32 0, i32 40,
; DEF-NEXT:      br label %loop
; DEF:           %i.ahead = add i64 %i, 4
; OFF-LABEL:   define {{.*}} @while_square(
; OFF-NOT:       llvm.riscv.ns.cond
define i64 @while_square(i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %inc, %loop ]
  %inc = add nuw nsw i64 %i, 1
  %sq = mul nsw i64 %i, %i
  %c = icmp slt i64 %sq, %n
  br i1 %c, label %loop, label %exit

exit:
  ret i64 %i
}

; The flags of the original instructions stay; the clones lose them. The
; original values are stored, so that they stay in the loop.
; CHECK-LABEL: define {{.*}} @flags_dropped(
; CHECK:       entry:
; CHECK:         %t.swp0 = mul i64 0, 0
; CHECK-NEXT:    %u.swp0 = shl i64 %t.swp0, 1
; CHECK-NEXT:    %v.swp0 = or i64 %u.swp0, 1
; CHECK-NEXT:    %w.swp0 = lshr i64 %u.swp0, 1
; CHECK-NEXT:    %x.swp0 = add i64 %v.swp0, %w.swp0
; CHECK:       loop:
; CHECK:         %t = mul nsw i64 %i, %i
; CHECK-NEXT:    %u = shl nuw nsw i64 %t, 1
; CHECK-NEXT:    %v = or disjoint i64 %u, 1
; CHECK-NEXT:    %w = lshr exact i64 %u, 1
; CHECK-NEXT:    %x = add nsw i64 %v, %w
; CHECK:         %i.ahead = add i64 %i, 4
; CHECK-NEXT:    %t.ahead = mul i64 %i.ahead, %i.ahead
; CHECK-NEXT:    %u.ahead = shl i64 %t.ahead, 1
; CHECK-NEXT:    %v.ahead = or i64 %u.ahead, 1
; CHECK-NEXT:    %w.ahead = lshr i64 %u.ahead, 1
; CHECK-NEXT:    %x.ahead = add i64 %v.ahead, %w.ahead
; CHECK-NEXT:    call void @llvm.riscv.ns.cond.push.i64(i32 0, i32 40, i64 %x.ahead, i64 %n)
; CHECK-NEXT:    [[POP:%.*]] = call i1 @llvm.riscv.ns.cond.pop(i32 0)
; CHECK-NEXT:    br i1 [[POP]], label %loop, label %exit
; OFF-LABEL:   define {{.*}} @flags_dropped(
; OFF-NOT:       llvm.riscv.ns.cond
define void @flags_dropped(ptr %p, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %inc, %loop ]
  %t = mul nsw i64 %i, %i
  %u = shl nuw nsw i64 %t, 1
  %v = or disjoint i64 %u, 1
  %w = lshr exact i64 %u, 1
  %x = add nsw i64 %v, %w
  store i64 %x, ptr %p
  %inc = add nuw nsw i64 %i, 1
  %c = icmp slt i64 %x, %n
  br i1 %c, label %loop, label %exit

exit:
  ret void
}

; Division by a constant other than 0 or -1 is safe to compute anywhere.
; CHECK-LABEL: define {{.*}} @udiv_const(
; CHECK:         %d.ahead = udiv i64 %i.ahead, 7
; CHECK:         [[POP:%.*]] = call i1 @llvm.riscv.ns.cond.pop(i32 0)
; CHECK-NEXT:    br i1 [[POP]], label %loop, label %exit
; OFF-LABEL:   define {{.*}} @udiv_const(
; OFF-NOT:       llvm.riscv.ns.cond
define i64 @udiv_const(i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %inc, %loop ]
  %inc = add nuw nsw i64 %i, 1
  %d = udiv i64 %i, 7
  %sq = mul i64 %d, %i
  %c = icmp ult i64 %sq, %n
  br i1 %c, label %loop, label %exit

exit:
  ret i64 %i
}

; A branch in the body and the latch both run every iteration and get
; queues of their own. They share one condition, which is deleted only once.
; CHECK-LABEL: define {{.*}} @shared_condition(
; CHECK:       entry:
; CHECK-NEXT:    call void @llvm.riscv.ns.cond.init(i32 0)
; CHECK-NEXT:    call void @llvm.riscv.ns.cond.init(i32 1)
; CHECK:       loop:
; CHECK-NOT:     icmp
; CHECK:         [[P0:%.*]] = call i1 @llvm.riscv.ns.cond.pop(i32 0)
; CHECK-NEXT:    br i1 [[P0]], label %then, label %else
; CHECK:       latch:
; CHECK-NOT:     icmp
; CHECK:         [[P1:%.*]] = call i1 @llvm.riscv.ns.cond.pop(i32 1)
; CHECK-NEXT:    br i1 [[P1]], label %loop, label %exit
; OFF-LABEL:   define {{.*}} @shared_condition(
; OFF-NOT:       llvm.riscv.ns.cond
define void @shared_condition(ptr %p, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %inc, %latch ]
  %sq = mul i64 %i, %i
  %c = icmp slt i64 %sq, %n
  br i1 %c, label %then, label %else

then:
  store i64 1, ptr %p
  br label %latch

else:
  store i64 2, ptr %p
  br label %latch

latch:
  %inc = add nuw nsw i64 %i, 1
  br i1 %c, label %loop, label %exit

exit:
  ret void
}

; The inner loop is entered once per outer iteration; init empties its queue
; each time, in the inner loop's preheader.
; CHECK-LABEL: define {{.*}} @reentry(
; CHECK:       outer:
; CHECK-NEXT:    %j = phi i64
; CHECK-NEXT:    call void @llvm.riscv.ns.cond.init(i32 0)
; CHECK:         br label %inner
; CHECK:       inner:
; CHECK:         [[POP:%.*]] = call i1 @llvm.riscv.ns.cond.pop(i32 0)
; CHECK-NEXT:    br i1 [[POP]], label %inner, label %outer.latch
; The outer loop has an inner loop, so it is not pipelined.
; CHECK:       outer.latch:
; CHECK:         %oc = icmp ult i64 %j.next, %m
; CHECK-NEXT:    br i1 %oc, label %outer, label %exit
; OFF-LABEL:   define {{.*}} @reentry(
; OFF-NOT:       llvm.riscv.ns.cond
define void @reentry(ptr %p, i64 %m, i64 %n) {
entry:
  br label %outer

outer:
  %j = phi i64 [ 0, %entry ], [ %j.next, %outer.latch ]
  br label %inner

inner:
  %i = phi i64 [ 0, %outer ], [ %inc, %inner ]
  %inc = add nuw nsw i64 %i, 1
  %sq = mul i64 %i, %i
  %lim = add i64 %n, %j
  %c = icmp slt i64 %sq, %lim
  br i1 %c, label %inner, label %outer.latch

outer.latch:
  store i64 %i, ptr %p
  %j.next = add nuw nsw i64 %j, 1
  %oc = icmp ult i64 %j.next, %m
  br i1 %oc, label %outer, label %exit

exit:
  ret void
}

; A load could read an invalid address in an iteration that never runs.
; long i = 0; while (a[i] != 0) i++; return i;
; CHECK-LABEL: define {{.*}} @reject_load(
; CHECK-NOT:     llvm.riscv.ns.cond
define i64 @reject_load(ptr %a) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %inc, %loop ]
  %inc = add nuw nsw i64 %i, 1
  %q = getelementptr inbounds i64, ptr %a, i64 %i
  %v = load i64, ptr %q
  %c = icmp ne i64 %v, 0
  br i1 %c, label %loop, label %exit

exit:
  ret i64 %i
}

; Division by a value that may be 0 is undefined behavior.
; CHECK-LABEL: define {{.*}} @reject_udiv_var(
; CHECK-NOT:     llvm.riscv.ns.cond
define i64 @reject_udiv_var(i64 %n, i64 %d) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %inc, %loop ]
  %inc = add nuw nsw i64 %i, 1
  %q = udiv i64 %i, %d
  %sq = mul i64 %q, %i
  %c = icmp ult i64 %sq, %n
  br i1 %c, label %loop, label %exit

exit:
  ret i64 %i
}

; INT_MIN / -1 overflows, which is undefined behavior.
; CHECK-LABEL: define {{.*}} @reject_sdiv_minus_one(
; CHECK-NOT:     llvm.riscv.ns.cond
define i64 @reject_sdiv_minus_one(i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %inc, %loop ]
  %inc = add nuw nsw i64 %i, 1
  %q = sdiv i64 %i, -1
  %sq = mul i64 %q, %i
  %c = icmp slt i64 %sq, %n
  br i1 %c, label %loop, label %exit

exit:
  ret i64 %i
}

; A call could write anything.
; CHECK-LABEL: define {{.*}} @reject_call(
; CHECK-NOT:     llvm.riscv.ns.cond
define i64 @reject_call(i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %inc, %loop ]
  call void @ext(i64 %i)
  %inc = add nuw nsw i64 %i, 1
  %sq = mul i64 %i, %i
  %c = icmp slt i64 %sq, %n
  br i1 %c, label %loop, label %exit

exit:
  ret i64 %i
}

; A recurrence that is not affine cannot be advanced to a later iteration.
; while (x != 0) { n++; x >>= 1; }
; CHECK-LABEL: define {{.*}} @reject_shift_recurrence(
; CHECK-NOT:     llvm.riscv.ns.cond
define i64 @reject_shift_recurrence(i64 %x0) {
entry:
  br label %loop

loop:
  %x = phi i64 [ %x0, %entry ], [ %x.next, %loop ]
  %k = phi i64 [ 0, %entry ], [ %k.next, %loop ]
  %k.next = add nuw nsw i64 %k, 1
  %x.next = lshr i64 %x, 1
  %c = icmp ne i64 %x.next, 0
  br i1 %c, label %loop, label %exit

exit:
  ret i64 %k.next
}

; A second exit would leave the loop with conditions for its own iterations.
; CHECK-LABEL: define {{.*}} @reject_two_exits(
; CHECK-NOT:     llvm.riscv.ns.cond
define i64 @reject_two_exits(i64 %n, i64 %m) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %inc, %latch ]
  %sq = mul i64 %i, %i
  %e = icmp eq i64 %sq, %m
  br i1 %e, label %early, label %latch

latch:
  %inc = add nuw nsw i64 %i, 1
  %c = icmp slt i64 %sq, %n
  br i1 %c, label %loop, label %exit

early:
  ret i64 -1

exit:
  ret i64 %i
}

; The branch in %maybe does not run every iteration, so only the header's
; branch and the latch are pipelined.
; CHECK-LABEL: define {{.*}} @partial_branch(
; CHECK:       loop:
; CHECK:         [[P0:%.*]] = call i1 @llvm.riscv.ns.cond.pop(i32 0)
; CHECK-NEXT:    br i1 [[P0]], label %maybe, label %latch
; CHECK:       maybe:
; CHECK-NEXT:    %b = icmp ugt i64 %sq, 100
; CHECK-NEXT:    br i1 %b, label %store, label %latch
; CHECK:       latch:
; CHECK:         [[P1:%.*]] = call i1 @llvm.riscv.ns.cond.pop(i32 1)
; CHECK-NEXT:    br i1 [[P1]], label %loop, label %exit
; OFF-LABEL:   define {{.*}} @partial_branch(
; OFF-NOT:       llvm.riscv.ns.cond
define void @partial_branch(ptr %p, i64 %n, i64 %m) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %inc, %latch ]
  %sq = mul i64 %i, %i
  %a = icmp ult i64 %sq, %m
  br i1 %a, label %maybe, label %latch

maybe:
  %b = icmp ugt i64 %sq, 100
  br i1 %b, label %store, label %latch

store:
  store i64 %i, ptr %p
  br label %latch

latch:
  %inc = add nuw nsw i64 %i, 1
  %c = icmp slt i64 %sq, %n
  br i1 %c, label %loop, label %exit

exit:
  ret void
}

; A loop with a trip count still takes the guarded kernel and drain path.
; for (i = 0; i < n; i++) if (i * i > m) *p = i;
; CHECK-LABEL: define {{.*}} @counted_loop_unchanged(
; CHECK:         %swp.long.enough = icmp ugt i64 %n, 4
; CHECK:         %swp.kernel.count = sub nuw i64 %n, 4
; CHECK:         %swp.kernel.iv = phi i64
; CHECK:       swp.drain.ph:
; OFF-LABEL:   define {{.*}} @counted_loop_unchanged(
; OFF:           %swp.long.enough = icmp ugt i64 %n, 4
define void @counted_loop_unchanged(ptr %p, i64 %n, i64 %m) {
entry:
  %z = icmp sgt i64 %n, 0
  br i1 %z, label %loop, label %exit

loop:
  %i = phi i64 [ 0, %entry ], [ %inc, %latch ]
  %sq = mul i64 %i, %i
  %b = icmp ugt i64 %sq, %m
  br i1 %b, label %store, label %latch

store:
  store i64 %i, ptr %p
  br label %latch

latch:
  %inc = add nuw nsw i64 %i, 1
  %c = icmp slt i64 %inc, %n
  br i1 %c, label %loop, label %exit

exit:
  ret void
}
