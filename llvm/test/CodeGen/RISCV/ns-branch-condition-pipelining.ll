; RUN: llc -mtriple=riscv64 -O2 -riscv-ns-swp-distance=4 \
; RUN:   -stop-after=riscv-branch-condition-pipelining < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -O2 -riscv-ns-disable-software-pipelining \
; RUN:   -stop-after=riscv-branch-condition-pipelining < %s \
; RUN:   | FileCheck %s --check-prefix=DISABLED

; Tests which loop branches RISCVBranchConditionPipelining computes ahead,
; and the loop structure it builds for them.

; DISABLED-NOT: llvm.riscv.ns.cond

; for (i = 0; i < n; ++i) if (a[i] > 0) a[i] += 1;
define void @inc_pos(ptr %a, i64 %n) {
; CHECK-LABEL: define {{.*}} @inc_pos(
; CHECK:       ph:
; CHECK:         %swp.long.enough = icmp ugt i64 %n, 4
; CHECK-NEXT:    br i1 %swp.long.enough, label %swp.ph, label %swp.drain.ph
; The prologue queues the conditions of iterations 0-3.
; CHECK:       swp.ph:
; CHECK-NEXT:    call void @llvm.riscv.ns.cond.init(i32 0)
; CHECK-NEXT:    %p.swp0 = getelementptr inbounds i32, ptr %a, i64 0
; CHECK-NEXT:    %v.swp0 = load i32, ptr %p.swp0
; CHECK-NEXT:    [[X0:%.*]] = sext i32 %v.swp0 to i64
; CHECK-NEXT:    call void @llvm.riscv.ns.cond.push.i64(i32 0, i32 38, i64 [[X0]], i64 0)
; CHECK:         %p.swp3 = getelementptr inbounds i32, ptr %a, i64 3
; CHECK-NEXT:    %v.swp3 = load i32, ptr %p.swp3
; CHECK-NEXT:    [[X3:%.*]] = sext i32 %v.swp3 to i64
; CHECK-NEXT:    call void @llvm.riscv.ns.cond.push.i64(i32 0, i32 38, i64 [[X3]], i64 0)
; CHECK-NEXT:    %swp.kernel.count = sub nuw i64 %n, 4
; CHECK-NEXT:    br label %loop.swp
; The kernel queues the condition four iterations ahead and branches on
; the one queued four iterations ago.
; CHECK:       loop.swp:
; CHECK-NEXT:    %swp.kernel.iv = phi i64 [ 0, %swp.ph ], [ %swp.kernel.iv.next, %latch.swp ]
; CHECK-NEXT:    %i.swp = phi i64 [ 0, %swp.ph ], [ %i.next.swp, %latch.swp ]
; CHECK-NEXT:    %i.ahead = add i64 %i.swp, 4
; CHECK-NEXT:    %p.ahead = getelementptr inbounds i32, ptr %a, i64 %i.ahead
; CHECK-NEXT:    %v.ahead = load i32, ptr %p.ahead
; CHECK-NEXT:    [[XA:%.*]] = sext i32 %v.ahead to i64
; CHECK-NEXT:    call void @llvm.riscv.ns.cond.push.i64(i32 0, i32 38, i64 [[XA]], i64 0)
; CHECK-NEXT:    %p.swp = getelementptr inbounds i32, ptr %a, i64 %i.swp
; CHECK-NEXT:    %v.swp = load i32, ptr %p.swp
; CHECK-NEXT:    [[POP:%.*]] = call i1 @llvm.riscv.ns.cond.pop(i32 0)
; CHECK-NEXT:    br i1 [[POP]], label %then.swp, label %latch.swp
; The kernel stops four iterations early.
; CHECK:       latch.swp:
; CHECK-NEXT:    %i.next.swp = add nuw i64 %i.swp, 1
; CHECK-NEXT:    %swp.kernel.iv.next = add nuw i64 %swp.kernel.iv, 1
; CHECK-NEXT:    %swp.kernel.done = icmp eq i64 %swp.kernel.iv.next, %swp.kernel.count
; CHECK-NEXT:    br i1 %swp.kernel.done, label %swp.kernel.exit, label %loop.swp
; CHECK:       swp.kernel.exit:
; CHECK-NEXT:    br label %swp.drain.ph
; The original loop finishes the remaining iterations, or all of them.
; CHECK:       swp.drain.ph:
; CHECK-NEXT:    %i.swp.resume = phi i64 [ 0, %ph ], [ %i.next.swp, %swp.kernel.exit ]
; CHECK-NEXT:    br label %loop
; CHECK:       loop:
; CHECK-NEXT:    %i = phi i64 [ %i.swp.resume, %swp.drain.ph ], [ %i.next, %latch ]
; CHECK-NOT:     llvm.riscv.ns.cond
; CHECK:         br i1 %c, label %then, label %latch
entry:
  %guard = icmp sgt i64 %n, 0
  br i1 %guard, label %ph, label %exit

ph:
  br label %loop

loop:
  %i = phi i64 [ 0, %ph ], [ %i.next, %latch ]
  %p = getelementptr inbounds i32, ptr %a, i64 %i
  %v = load i32, ptr %p
  %c = icmp sgt i32 %v, 0
  br i1 %c, label %then, label %latch

then:
  %v1 = add i32 %v, 1
  store i32 %v1, ptr %p
  br label %latch

latch:
  %i.next = add nuw i64 %i, 1
  %done = icmp eq i64 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

; The future condition is queued before independent work after its load.
define void @schedule_ahead(ptr noalias %a, ptr noalias %out, i64 %n) {
; CHECK-LABEL: define {{.*}} @schedule_ahead(
; CHECK:       loop.swp:
; CHECK:         %v.ahead = load i32, ptr %p.ahead
; CHECK:         call void @llvm.riscv.ns.cond.push.i64
; CHECK:         %v.swp = load i32, ptr %p.swp
; CHECK:         %out.index.swp = add i64 %i.swp, 123
entry:
  %guard = icmp sgt i64 %n, 0
  br i1 %guard, label %ph, label %exit

ph:
  br label %loop

loop:
  %i = phi i64 [ 0, %ph ], [ %i.next, %latch ]
  %p = getelementptr inbounds i32, ptr %a, i64 %i
  %v = load i32, ptr %p
  %c = icmp sgt i32 %v, 0
  %out.index = add i64 %i, 123
  %out.p = getelementptr inbounds i32, ptr %out, i64 %out.index
  br i1 %c, label %then, label %else

then:
  %v1 = add i32 %v, 1
  store i32 %v1, ptr %out.p
  br label %latch

else:
  store i32 0, ptr %out.p
  br label %latch

latch:
  %i.next = add nuw i64 %i, 1
  %done = icmp eq i64 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

; A dynamic affine step is multiplied by the pipeline distance once, not once
; per iteration.
define void @dynamic_step(ptr noalias %a, ptr noalias %out, i64 %n,
                          i64 %step) {
; CHECK-LABEL: define {{.*}} @dynamic_step(
; CHECK:       ph:
; CHECK:         [[OFFSET:%.*]] = mul i64 %step, 4
; CHECK:       swp.ph:
; CHECK:       loop.swp:
; CHECK:         %i.ahead = add i64 %i.swp, [[OFFSET]]
; CHECK-NOT:     mul i64 %step, 4
entry:
  %guard = icmp sgt i64 %n, 0
  br i1 %guard, label %ph, label %exit

ph:
  br label %loop

loop:
  %i = phi i64 [ 0, %ph ], [ %i.next, %latch ]
  %k = phi i64 [ 0, %ph ], [ %k.next, %latch ]
  %p = getelementptr inbounds i32, ptr %a, i64 %i
  %v = load i32, ptr %p
  %c = icmp sgt i32 %v, 0
  %out.index = add i64 %k, 123
  %out.p = getelementptr inbounds i32, ptr %out, i64 %out.index
  br i1 %c, label %then, label %else

then:
  %v1 = add i32 %v, 1
  store i32 %v1, ptr %out.p
  br label %latch

else:
  store i32 0, ptr %out.p
  br label %latch

latch:
  %i.next = add i64 %i, %step
  %k.next = add nuw i64 %k, 1
  %done = icmp eq i64 %k.next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

; A store three elements ahead overlaps the load of the iteration three
; ahead, so the condition can only be computed two iterations early.
define void @store_ahead(ptr %a, i64 %n) {
; CHECK-LABEL: define {{.*}} @store_ahead(
; CHECK:         %swp.long.enough = icmp ugt i64 %n, 2
; CHECK:       swp.ph:
; CHECK-COUNT-2: call void @llvm.riscv.ns.cond.push.i64
; CHECK-NOT:     call void @llvm.riscv.ns.cond.push.i64
; CHECK:         %swp.kernel.count = sub nuw i64 %n, 2
; CHECK:         %i.ahead = add i64 %i.swp, 2
entry:
  %guard = icmp sgt i64 %n, 0
  br i1 %guard, label %ph, label %exit

ph:
  br label %loop

loop:
  %i = phi i64 [ 0, %ph ], [ %i.next, %latch ]
  %p = getelementptr inbounds i32, ptr %a, i64 %i
  %v = load i32, ptr %p
  %c = icmp sgt i32 %v, 0
  br i1 %c, label %then, label %latch

then:
  %q = getelementptr inbounds i8, ptr %p, i64 12
  store i32 0, ptr %q
  br label %latch

latch:
  %i.next = add nuw i64 %i, 1
  %done = icmp eq i64 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

; The condition reads what the previous iteration stored: no distance works.
define void @prefix(ptr %a, i64 %n) {
; CHECK-LABEL: define {{.*}} @prefix(
; CHECK-NOT:     llvm.riscv.ns.cond
; CHECK:       ret void
entry:
  %guard = icmp sgt i64 %n, 0
  br i1 %guard, label %ph, label %exit

ph:
  br label %loop

loop:
  %i = phi i64 [ 0, %ph ], [ %i.next, %latch ]
  %p = getelementptr inbounds i32, ptr %a, i64 %i
  %v = load i32, ptr %p
  %c = icmp sgt i32 %v, 0
  br i1 %c, label %then, label %latch

then:
  %q = getelementptr inbounds i8, ptr %p, i64 4
  store i32 %v, ptr %q
  br label %latch

latch:
  %i.next = add nuw i64 %i, 1
  %done = icmp eq i64 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

; a and b may overlap at any distance.
define void @may_alias(ptr %a, ptr %b, i64 %n) {
; CHECK-LABEL: define {{.*}} @may_alias(
; CHECK-NOT:     llvm.riscv.ns.cond
; CHECK:       ret void
entry:
  %guard = icmp sgt i64 %n, 0
  br i1 %guard, label %ph, label %exit

ph:
  br label %loop

loop:
  %i = phi i64 [ 0, %ph ], [ %i.next, %latch ]
  %p = getelementptr inbounds i32, ptr %a, i64 %i
  %v = load i32, ptr %p
  %c = icmp sgt i32 %v, 0
  br i1 %c, label %then, label %latch

then:
  %q = getelementptr inbounds i32, ptr %b, i64 %i
  store i32 1, ptr %q
  br label %latch

latch:
  %i.next = add nuw i64 %i, 1
  %done = icmp eq i64 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

; ... unless they cannot.
define void @no_alias(ptr noalias %a, ptr noalias %b, i64 %n) {
; CHECK-LABEL: define {{.*}} @no_alias(
; CHECK:       swp.ph:
; CHECK-COUNT-4: call void @llvm.riscv.ns.cond.push.i64
entry:
  %guard = icmp sgt i64 %n, 0
  br i1 %guard, label %ph, label %exit

ph:
  br label %loop

loop:
  %i = phi i64 [ 0, %ph ], [ %i.next, %latch ]
  %p = getelementptr inbounds i32, ptr %a, i64 %i
  %v = load i32, ptr %p
  %c = icmp sgt i32 %v, 0
  br i1 %c, label %then, label %latch

then:
  %q = getelementptr inbounds i32, ptr %b, i64 %i
  store i32 1, ptr %q
  br label %latch

latch:
  %i.next = add nuw i64 %i, 1
  %done = icmp eq i64 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

; Partitioning around a pivot. k advances by at most one element per
; iteration, so the swap's a[k] never passes a[i], and stays behind every
; element the condition reads ahead. The kernel carries k into the original
; loop like any other phi.
; for (i = 0; i < n; ++i) if (a[i] < pivot) swap(&a[i], &a[k++]);
define i32 @partition(ptr %a, i32 %n, i32 %pivot) {
; CHECK-LABEL: define {{.*}} @partition(
; CHECK:       swp.ph:
; CHECK-COUNT-4: call void @llvm.riscv.ns.cond.push.i64
; CHECK-NOT:     call void @llvm.riscv.ns.cond.push.i64
; CHECK:         %swp.kernel.count = sub nuw i64 %tc, 4
; CHECK:       loop.swp:
; CHECK-NEXT:    %swp.kernel.iv = phi i64
; CHECK-NEXT:    %i.swp = phi i64
; CHECK-NEXT:    %k.swp = phi i32 [ 0, %swp.ph ], [ %k.next.swp, %latch.swp ]
; CHECK:         %i.ahead = add i64 %i.swp, 4
; CHECK:         [[POP:%.*]] = call i1 @llvm.riscv.ns.cond.pop(i32 0)
; CHECK-NEXT:    br i1 [[POP]], label %then.swp, label %latch.swp
; CHECK:       swp.drain.ph:
; CHECK-NEXT:    %i.swp.resume = phi i64
; CHECK-NEXT:    %k.swp.resume = phi i32 [ 0, %ph ], [ %k.next.swp, %swp.kernel.exit ]
entry:
  %guard = icmp sgt i32 %n, 0
  br i1 %guard, label %ph, label %exit

ph:
  %tc = zext nneg i32 %n to i64
  br label %loop

loop:
  %i = phi i64 [ 0, %ph ], [ %i.next, %latch ]
  %k = phi i32 [ 0, %ph ], [ %k.next, %latch ]
  %p = getelementptr inbounds i32, ptr %a, i64 %i
  %v = load i32, ptr %p
  %c = icmp slt i32 %v, %pivot
  br i1 %c, label %then, label %latch

then:
  %k.inc = add nsw i32 %k, 1
  %k.ext = sext i32 %k to i64
  %q = getelementptr inbounds i32, ptr %a, i64 %k.ext
  %w = load i32, ptr %q
  store i32 %v, ptr %q
  store i32 %w, ptr %p
  br label %latch

latch:
  %k.next = phi i32 [ %k.inc, %then ], [ %k, %loop ]
  %i.next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %i.next, %tc
  br i1 %done, label %exit, label %loop

exit:
  %r = phi i32 [ 0, %entry ], [ %k.next, %latch ]
  ret i32 %r
}

; The same with an output pointer, which advances by a byte offset.
; for (i = 0; i < n; ++i) if (a[i] != 0) *out++ = a[i];  // out = a
define ptr @compact(ptr %a, i32 %n) {
; CHECK-LABEL: define {{.*}} @compact(
; CHECK:       swp.ph:
; CHECK-COUNT-4: call void @llvm.riscv.ns.cond.push.i64
; CHECK:       loop.swp:
; CHECK:         %out.swp = phi ptr [ %a, %swp.ph ], [ %out.next.swp, %latch.swp ]
; CHECK:         call i1 @llvm.riscv.ns.cond.pop(i32 0)
entry:
  %guard = icmp sgt i32 %n, 0
  br i1 %guard, label %ph, label %exit

ph:
  %tc = zext nneg i32 %n to i64
  br label %loop

loop:
  %i = phi i64 [ 0, %ph ], [ %i.next, %latch ]
  %out = phi ptr [ %a, %ph ], [ %out.next, %latch ]
  %p = getelementptr inbounds i32, ptr %a, i64 %i
  %v = load i32, ptr %p
  %c = icmp ne i32 %v, 0
  br i1 %c, label %then, label %latch

then:
  store i32 %v, ptr %out
  %out.inc = getelementptr inbounds i8, ptr %out, i64 4
  br label %latch

latch:
  %out.next = phi ptr [ %out.inc, %then ], [ %out, %loop ]
  %i.next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %i.next, %tc
  br i1 %done, label %exit, label %loop

exit:
  %r = phi ptr [ %a, %entry ], [ %out.next, %latch ]
  ret ptr %r
}

; The counter can also advance through a select.
; for (i = 0; i < n; ++i) if (a[i] < pivot) a[k++] = a[i];
define void @filter_select(ptr %a, i32 %n, i32 %pivot) {
; CHECK-LABEL: define {{.*}} @filter_select(
; CHECK:       swp.ph:
; CHECK-COUNT-4: call void @llvm.riscv.ns.cond.push.i64
; CHECK:       loop.swp:
; CHECK:         call i1 @llvm.riscv.ns.cond.pop(i32 0)
entry:
  %guard = icmp sgt i32 %n, 0
  br i1 %guard, label %ph, label %exit

ph:
  %tc = zext nneg i32 %n to i64
  br label %loop

loop:
  %i = phi i64 [ 0, %ph ], [ %i.next, %latch ]
  %k = phi i32 [ 0, %ph ], [ %k.next, %latch ]
  %p = getelementptr inbounds i32, ptr %a, i64 %i
  %v = load i32, ptr %p
  %c = icmp slt i32 %v, %pivot
  %k.inc = add nsw i32 %k, 1
  %k.next = select i1 %c, i32 %k.inc, i32 %k
  br i1 %c, label %then, label %latch

then:
  %k.ext = sext i32 %k to i64
  %q = getelementptr inbounds i32, ptr %a, i64 %k.ext
  store i32 %v, ptr %q
  br label %latch

latch:
  %i.next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %i.next, %tc
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

; A store to a fixed element only meets the load that reads that element, in
; iteration 0, and the prologue still runs that load before the store.
define void @fixed_store(ptr %a, i32 %n) {
; CHECK-LABEL: define {{.*}} @fixed_store(
; CHECK:       swp.ph:
; CHECK-COUNT-4: call void @llvm.riscv.ns.cond.push.i64
entry:
  %guard = icmp sgt i32 %n, 0
  br i1 %guard, label %ph, label %exit

ph:
  %tc = zext nneg i32 %n to i64
  br label %loop

loop:
  %i = phi i64 [ 0, %ph ], [ %i.next, %latch ]
  %p = getelementptr inbounds i32, ptr %a, i64 %i
  %v = load i32, ptr %p
  %c = icmp slt i32 %v, 0
  br i1 %c, label %then, label %latch

then:
  store i32 0, ptr %a
  br label %latch

latch:
  %i.next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %i.next, %tc
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

; k can advance by two elements in an iteration, and so overtake i.
define void @counter_overtakes(ptr %a, i32 %n) {
; CHECK-LABEL: define {{.*}} @counter_overtakes(
; CHECK-NOT:     llvm.riscv.ns.cond
; CHECK:       ret void
entry:
  %guard = icmp sgt i32 %n, 0
  br i1 %guard, label %ph, label %exit

ph:
  %tc = zext nneg i32 %n to i64
  br label %loop

loop:
  %i = phi i64 [ 0, %ph ], [ %i.next, %latch ]
  %k = phi i32 [ 0, %ph ], [ %k.next, %latch ]
  %p = getelementptr inbounds i32, ptr %a, i64 %i
  %v = load i32, ptr %p
  %c = icmp slt i32 %v, 0
  br i1 %c, label %then, label %latch

then:
  %k.inc = add nsw i32 %k, 2
  %k.ext = sext i32 %k to i64
  %q = getelementptr inbounds i32, ptr %a, i64 %k.ext
  store i32 0, ptr %q
  br label %latch

latch:
  %k.next = phi i32 [ %k.inc, %then ], [ %k, %loop ]
  %i.next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %i.next, %tc
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

; The store to a[k] runs before the load of a[i] in its iteration, and k can
; equal i.
define void @counter_store_first(ptr %a, i32 %n) {
; CHECK-LABEL: define {{.*}} @counter_store_first(
; CHECK-NOT:     llvm.riscv.ns.cond
; CHECK:       ret void
entry:
  %guard = icmp sgt i32 %n, 0
  br i1 %guard, label %ph, label %exit

ph:
  %tc = zext nneg i32 %n to i64
  br label %loop

loop:
  %i = phi i64 [ 0, %ph ], [ %i.next, %latch ]
  %k = phi i32 [ 0, %ph ], [ %k.next, %latch ]
  %k.ext = sext i32 %k to i64
  %q = getelementptr inbounds i32, ptr %a, i64 %k.ext
  store i32 0, ptr %q
  %p = getelementptr inbounds i32, ptr %a, i64 %i
  %v = load i32, ptr %p
  %c = icmp slt i32 %v, 0
  br i1 %c, label %then, label %latch

then:
  %k.inc = add nsw i32 %k, 1
  br label %latch

latch:
  %k.next = phi i32 [ %k.inc, %then ], [ %k, %loop ]
  %i.next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %i.next, %tc
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

; k only moves down, away from a[i], but without nsw it could wrap around to
; the top of its range and so land ahead of a[i].
define void @counter_may_wrap(ptr %a, i32 %n) {
; CHECK-LABEL: define {{.*}} @counter_may_wrap(
; CHECK-NOT:     llvm.riscv.ns.cond
; CHECK:       ret void
entry:
  %guard = icmp sgt i32 %n, 0
  br i1 %guard, label %ph, label %exit

ph:
  %tc = zext nneg i32 %n to i64
  br label %loop

loop:
  %i = phi i64 [ 0, %ph ], [ %i.next, %latch ]
  %k = phi i32 [ 0, %ph ], [ %k.next, %latch ]
  %p = getelementptr inbounds i32, ptr %a, i64 %i
  %v = load i32, ptr %p
  %c = icmp slt i32 %v, 0
  br i1 %c, label %then, label %latch

then:
  %k.dec = add i32 %k, -2
  %k.ext = sext i32 %k to i64
  %q = getelementptr inbounds i32, ptr %a, i64 %k.ext
  store i32 0, ptr %q
  br label %latch

latch:
  %k.next = phi i32 [ %k.dec, %then ], [ %k, %loop ]
  %i.next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %i.next, %tc
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

; Like @partition, but nothing bounds the trip count, and so nothing bounds
; how far a[i] gets ahead of a[k]. That gap could, in principle, wrap around
; the address space.
define i32 @partition_unbounded(ptr %a, i64 %n, i32 %pivot) {
; CHECK-LABEL: define {{.*}} @partition_unbounded(
; CHECK-NOT:     llvm.riscv.ns.cond
; CHECK:       ret i32
entry:
  %guard = icmp sgt i64 %n, 0
  br i1 %guard, label %ph, label %exit

ph:
  br label %loop

loop:
  %i = phi i64 [ 0, %ph ], [ %i.next, %latch ]
  %k = phi i32 [ 0, %ph ], [ %k.next, %latch ]
  %p = getelementptr inbounds i32, ptr %a, i64 %i
  %v = load i32, ptr %p
  %c = icmp slt i32 %v, %pivot
  br i1 %c, label %then, label %latch

then:
  %k.inc = add nsw i32 %k, 1
  %k.ext = sext i32 %k to i64
  %q = getelementptr inbounds i32, ptr %a, i64 %k.ext
  %w = load i32, ptr %q
  store i32 %v, ptr %q
  store i32 %w, ptr %p
  br label %latch

latch:
  %k.next = phi i32 [ %k.inc, %then ], [ %k, %loop ]
  %i.next = add nuw i64 %i, 1
  %done = icmp eq i64 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  %r = phi i32 [ 0, %entry ], [ %k.next, %latch ]
  ret i32 %r
}

declare void @ext()

; The call could write anything the condition reads.
define void @with_call(ptr %a, i64 %n) {
; CHECK-LABEL: define {{.*}} @with_call(
; CHECK-NOT:     llvm.riscv.ns.cond
; CHECK:       ret void
entry:
  %guard = icmp sgt i64 %n, 0
  br i1 %guard, label %ph, label %exit

ph:
  br label %loop

loop:
  %i = phi i64 [ 0, %ph ], [ %i.next, %latch ]
  %p = getelementptr inbounds i32, ptr %a, i64 %i
  %v = load i32, ptr %p
  %c = icmp sgt i32 %v, 0
  br i1 %c, label %then, label %latch

then:
  call void @ext()
  br label %latch

latch:
  %i.next = add nuw i64 %i, 1
  %done = icmp eq i64 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

; Never more iterations than the distance.
define void @short_loop(ptr %a) {
; CHECK-LABEL: define {{.*}} @short_loop(
; CHECK-NOT:     llvm.riscv.ns.cond
; CHECK:       ret void
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
  %p = getelementptr inbounds i32, ptr %a, i64 %i
  %v = load i32, ptr %p
  %c = icmp sgt i32 %v, 0
  br i1 %c, label %then, label %latch

then:
  store i32 0, ptr %p
  br label %latch

latch:
  %i.next = add nuw i64 %i, 1
  %done = icmp eq i64 %i.next, 4
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

; A running sum cannot be computed for a later iteration.
define i32 @running_sum(ptr %a, i64 %n) {
; CHECK-LABEL: define {{.*}} @running_sum(
; CHECK-NOT:     llvm.riscv.ns.cond
; CHECK:       ret i32
entry:
  %guard = icmp sgt i64 %n, 0
  br i1 %guard, label %ph, label %exit

ph:
  br label %loop

loop:
  %i = phi i64 [ 0, %ph ], [ %i.next, %latch ]
  %s = phi i32 [ 0, %ph ], [ %s.next, %latch ]
  %cnt = phi i32 [ 0, %ph ], [ %cnt.next, %latch ]
  %p = getelementptr inbounds i32, ptr %a, i64 %i
  %v = load i32, ptr %p
  %s.next = add i32 %s, %v
  %c = icmp sgt i32 %s.next, 100
  br i1 %c, label %then, label %latch

then:
  %cnt.inc = add i32 %cnt, 1
  br label %latch

latch:
  %cnt.next = phi i32 [ %cnt.inc, %then ], [ %cnt, %loop ]
  %i.next = add nuw i64 %i, 1
  %done = icmp eq i64 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  %r = phi i32 [ 0, %entry ], [ %cnt.next, %latch ]
  ret i32 %r
}

; Only the outer branch runs in every iteration. The resume phi carries the
; kernel's reduction into the original loop.
define i32 @nested(ptr noalias %a, ptr noalias %b, i64 %n) {
; CHECK-LABEL: define {{.*}} @nested(
; CHECK:       swp.ph:
; CHECK-NEXT:    call void @llvm.riscv.ns.cond.init(i32 0)
; CHECK-NOT:     call void @llvm.riscv.ns.cond.init
; CHECK:       loop.swp:
; CHECK:         [[POP:%.*]] = call i1 @llvm.riscv.ns.cond.pop(i32 0)
; CHECK-NEXT:    br i1 [[POP]], label %outer.swp, label %latch.swp
; CHECK:       outer.swp:
; CHECK-NEXT:    %c2.swp = icmp sgt i32 %v.swp, 50
; CHECK-NEXT:    br i1 %c2.swp, label %inner.swp, label %latch.swp
; CHECK:       swp.drain.ph:
; CHECK-NEXT:    %i.swp.resume = phi i64
; CHECK-NEXT:    %sum.swp.resume = phi i32 [ 0, %ph ], [ %sum.next.swp, %swp.kernel.exit ]
entry:
  %guard = icmp sgt i64 %n, 0
  br i1 %guard, label %ph, label %exit

ph:
  br label %loop

loop:
  %i = phi i64 [ 0, %ph ], [ %i.next, %latch ]
  %sum = phi i32 [ 0, %ph ], [ %sum.next, %latch ]
  %p = getelementptr inbounds i32, ptr %a, i64 %i
  %v = load i32, ptr %p
  %c = icmp sgt i32 %v, 0
  br i1 %c, label %outer, label %latch

outer:
  %c2 = icmp sgt i32 %v, 50
  br i1 %c2, label %inner, label %latch

inner:
  %q = getelementptr inbounds i32, ptr %b, i64 %i
  store i32 2, ptr %q
  br label %latch

latch:
  %add = phi i32 [ 0, %loop ], [ 1, %outer ], [ 2, %inner ]
  %sum.next = add i32 %sum, %add
  %i.next = add nuw i64 %i, 1
  %done = icmp eq i64 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  %r = phi i32 [ 0, %entry ], [ %sum.next, %latch ]
  ret i32 %r
}

; A pointer induction variable is advanced by a byte offset, and an unsigned
; comparison of narrow values is sign-extended like a signed one.
define void @ptr_loop(ptr %p0, ptr %end) {
; CHECK-LABEL: define {{.*}} @ptr_loop(
; CHECK:       swp.ph:
; CHECK:         %p.swp1 = getelementptr i8, ptr %p0, i64 2
; CHECK-NEXT:    %v.swp1 = load i16, ptr %p.swp1
; CHECK-NEXT:    [[X1:%.*]] = sext i16 %v.swp1 to i64
; CHECK-NEXT:    call void @llvm.riscv.ns.cond.push.i64(i32 0, i32 36, i64 [[X1]], i64 7)
; CHECK:       loop.swp:
; CHECK:         %p.ahead = getelementptr i8, ptr %p.swp, i64 8
; CHECK-NEXT:    %v.ahead = load i16, ptr %p.ahead
entry:
  %guard = icmp ne ptr %p0, %end
  br i1 %guard, label %ph, label %exit

ph:
  br label %loop

loop:
  %p = phi ptr [ %p0, %ph ], [ %p.next, %latch ]
  %v = load i16, ptr %p
  %c = icmp ult i16 %v, 7
  br i1 %c, label %then, label %latch

then:
  store i16 7, ptr %p
  br label %latch

latch:
  %p.next = getelementptr inbounds i8, ptr %p, i64 2
  %done = icmp eq ptr %p.next, %end
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

; Each branch gets its own queue. A condition that is not an integer
; comparison is queued as (c != 0).
define void @two_branches(ptr noalias %a, ptr noalias %f, i64 %n) {
; CHECK-LABEL: define {{.*}} @two_branches(
; CHECK:       swp.ph:
; CHECK-NEXT:    call void @llvm.riscv.ns.cond.init(i32 0)
; CHECK-NEXT:    call void @llvm.riscv.ns.cond.init(i32 1)
; CHECK:         [[Z:%.*]] = zext i1 %c2.swp0 to i64
; CHECK-NEXT:    call void @llvm.riscv.ns.cond.push.i64(i32 1, i32 33, i64 [[Z]], i64 0)
; CHECK:       loop.swp:
; CHECK:         call i1 @llvm.riscv.ns.cond.pop(i32 0)
; CHECK:       mid.swp:
; CHECK:         call i1 @llvm.riscv.ns.cond.pop(i32 1)
entry:
  %guard = icmp sgt i64 %n, 0
  br i1 %guard, label %ph, label %exit

ph:
  br label %loop

loop:
  %i = phi i64 [ 0, %ph ], [ %i.next, %latch ]
  %p = getelementptr inbounds i32, ptr %a, i64 %i
  %v = load i32, ptr %p
  %c = icmp slt i32 %v, 0
  br i1 %c, label %then, label %mid

then:
  store i32 0, ptr %p
  br label %mid

mid:
  %q = getelementptr inbounds float, ptr %f, i64 %i
  %x = load float, ptr %q
  %c2 = fcmp ogt float %x, 1.0
  br i1 %c2, label %then2, label %latch

then2:
  store float 1.0, ptr %q
  br label %latch

latch:
  %i.next = add nuw i64 %i, 1
  %done = icmp eq i64 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

; Growing the code of a function optimized for size is not worth it.
define void @opt_size(ptr %a, i64 %n) optsize {
; CHECK-LABEL: define {{.*}} @opt_size(
; CHECK-NOT:     llvm.riscv.ns.cond
; CHECK:       ret void
entry:
  %guard = icmp sgt i64 %n, 0
  br i1 %guard, label %ph, label %exit

ph:
  br label %loop

loop:
  %i = phi i64 [ 0, %ph ], [ %i.next, %latch ]
  %p = getelementptr inbounds i32, ptr %a, i64 %i
  %v = load i32, ptr %p
  %c = icmp sgt i32 %v, 0
  br i1 %c, label %then, label %latch

then:
  store i32 0, ptr %p
  br label %latch

latch:
  %i.next = add nuw i64 %i, 1
  %done = icmp eq i64 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

declare void @llvm.lifetime.start.p0(i64 immarg, ptr)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr)

; A stack slot whose lifetime restarts every iteration could be reused under
; a load that now reads it iterations early.
define void @lifetime(ptr %a, i64 %n) {
; CHECK-LABEL: define {{.*}} @lifetime(
; CHECK-NOT:     llvm.riscv.ns.cond
; CHECK:       ret void
entry:
  %tmp = alloca i32
  %guard = icmp sgt i64 %n, 0
  br i1 %guard, label %ph, label %exit

ph:
  br label %loop

loop:
  %i = phi i64 [ 0, %ph ], [ %i.next, %latch ]
  call void @llvm.lifetime.start.p0(i64 4, ptr %tmp)
  %p = getelementptr inbounds i32, ptr %a, i64 %i
  %v = load i32, ptr %p
  store i32 %v, ptr %tmp
  %c = icmp sgt i32 %v, 0
  br i1 %c, label %then, label %latch

then:
  store i32 0, ptr %p
  br label %latch

latch:
  call void @llvm.lifetime.end.p0(i64 4, ptr %tmp)
  %i.next = add nuw i64 %i, 1
  %done = icmp eq i64 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

; An exit straight into undefined behavior, like the default of a switch that
; covers every value, is never taken.
define void @unreachable_exit(ptr noalias %a, ptr noalias %b, i64 %n) {
; CHECK-LABEL: define {{.*}} @unreachable_exit(
; CHECK:       swp.ph:
; CHECK:       loop.swp:
; CHECK:         call i1 @llvm.riscv.ns.cond.pop(i32 0)
; CHECK:       sw.swp:
; CHECK:         switch i32 %k.swp, label %bad [
; CHECK:       bad:
; CHECK-SAME:    preds = %sw.swp, %sw
entry:
  %guard = icmp sgt i64 %n, 0
  br i1 %guard, label %ph, label %exit

ph:
  br label %loop

loop:
  %i = phi i64 [ 0, %ph ], [ %i.next, %latch ]
  %p = getelementptr inbounds i32, ptr %a, i64 %i
  %v = load i32, ptr %p
  %c = icmp slt i32 %v, 0
  br i1 %c, label %then, label %sw

then:
  store i32 0, ptr %p
  br label %sw

sw:
  %q = getelementptr inbounds i32, ptr %b, i64 %i
  %w = load i32, ptr %q
  %k = and i32 %w, 1
  switch i32 %k, label %bad [
    i32 0, label %latch
    i32 1, label %odd
  ]

odd:
  store i32 1, ptr %q
  br label %latch

bad:
  unreachable

latch:
  %i.next = add nuw i64 %i, 1
  %done = icmp eq i64 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

declare void @abort() noreturn

; An exit that does something before it reaches undefined behavior is real.
define void @abort_exit(ptr %a, i64 %n) {
; CHECK-LABEL: define {{.*}} @abort_exit(
; CHECK-NOT:     llvm.riscv.ns.cond
; CHECK:       ret void
entry:
  %guard = icmp sgt i64 %n, 0
  br i1 %guard, label %ph, label %exit

ph:
  br label %loop

loop:
  %i = phi i64 [ 0, %ph ], [ %i.next, %latch ]
  %p = getelementptr inbounds i32, ptr %a, i64 %i
  %v = load i32, ptr %p
  %c = icmp slt i32 %v, 0
  br i1 %c, label %then, label %mid

then:
  store i32 0, ptr %p
  br label %mid

mid:
  %big = icmp sgt i32 %v, 1000
  br i1 %big, label %fail, label %latch

fail:
  call void @abort()
  unreachable

latch:
  %i.next = add nuw i64 %i, 1
  %done = icmp eq i64 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}
