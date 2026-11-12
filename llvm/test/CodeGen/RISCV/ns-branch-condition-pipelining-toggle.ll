; RUN: llc -mtriple=riscv64 -O2 -riscv-ns-swp-distance=4 \
; RUN:   -riscv-ns-swp-speculative-distance=4 -riscv-ns-swp-toggles \
; RUN:   -stop-after=riscv-branch-condition-pipelining < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -O2 -riscv-ns-swp-distance=4 \
; RUN:   -riscv-ns-swp-speculative-distance=3 -riscv-ns-swp-toggles \
; RUN:   -stop-after=riscv-branch-condition-pipelining < %s \
; RUN:   | FileCheck %s --check-prefix=D3
; RUN: llc -mtriple=riscv64 -O2 -riscv-ns-swp-toggles \
; RUN:   -stop-after=riscv-branch-condition-pipelining < %s \
; RUN:   | FileCheck %s --check-prefix=DEF
; RUN: llc -mtriple=riscv64 -O2 -riscv-ns-swp-distance=4 \
; RUN:   -riscv-ns-swp-speculative-distance=4 -riscv-ns-swp-toggles \
; RUN:   -riscv-ns-swp-speculative \
; RUN:   -stop-after=riscv-branch-condition-pipelining < %s \
; RUN:   | FileCheck %s --check-prefix=SPEC
; RUN: llc -mtriple=riscv64 -O2 -riscv-ns-swp-distance=4 \
; RUN:   -riscv-ns-swp-speculative-distance=4 \
; RUN:   -stop-after=riscv-branch-condition-pipelining < %s \
; RUN:   | FileCheck %s --check-prefix=OFF

; Tests -riscv-ns-swp-toggles: a value flipped every iteration is advanced to
; a later iteration by flipping it once or not at all, a branch may test it
; directly, and a loop whose conditions read nothing else is pipelined in
; place even though it has a trip count.

; The shape of bufferstep in MiBench ADPCM:
; for (i = 0; i < n; i++) { v = in[i]; if (t) *out++ = v; t = !t; }
; CHECK-LABEL: define {{.*}} @toggle_i1(
; No guard, kernel copy, counter or drain loop.
; CHECK-NOT:     swp.long.enough
; The preheader queues the conditions of iterations 0-3.
; CHECK:       entry:
; CHECK-NEXT:    call void @llvm.riscv.ns.cond.init(i32 0)
; CHECK-NEXT:    call void @llvm.riscv.ns.cond.push.i64(i32 0, i32 33, i64 0, i64 0)
; CHECK-NEXT:    call void @llvm.riscv.ns.cond.push.i64(i32 0, i32 33, i64 1, i64 0)
; CHECK-NEXT:    call void @llvm.riscv.ns.cond.push.i64(i32 0, i32 33, i64 0, i64 0)
; CHECK-NEXT:    call void @llvm.riscv.ns.cond.push.i64(i32 0, i32 33, i64 1, i64 0)
; CHECK-NEXT:    br label %loop
; Four iterations on, t is what it is now.
; CHECK:       loop:
; CHECK:         %v = load i8, ptr %p
; CHECK-NEXT:    [[T:%.*]] = zext i1 %t to i64
; CHECK-NEXT:    call void @llvm.riscv.ns.cond.push.i64(i32 0, i32 33, i64 [[T]], i64 0)
; CHECK-NEXT:    [[POP:%.*]] = call i1 @llvm.riscv.ns.cond.pop(i32 0)
; CHECK-NEXT:    br i1 [[POP]], label %odd, label %latch
; The latch still counts the loop.
; CHECK:       latch:
; CHECK:         %t.next = xor i1 %t, true
; CHECK:         %c = icmp ult i64 %i.next, %n
; CHECK-NEXT:    br i1 %c, label %loop, label %exit
; CHECK-NOT:     swp.kernel
; CHECK-NOT:     swp.drain
; Three iterations on, t is flipped.
; D3-LABEL:    define {{.*}} @toggle_i1(
; D3:          entry:
; D3-NEXT:       call void @llvm.riscv.ns.cond.init(i32 0)
; D3-NEXT:       call void @llvm.riscv.ns.cond.push.i64(i32 0, i32 33, i64 0, i64 0)
; D3-NEXT:       call void @llvm.riscv.ns.cond.push.i64(i32 0, i32 33, i64 1, i64 0)
; D3-NEXT:       call void @llvm.riscv.ns.cond.push.i64(i32 0, i32 33, i64 0, i64 0)
; D3-NEXT:       br label %loop
; D3:            %t.ahead = xor i1 %t, true
; D3-NEXT:       [[T:%.*]] = zext i1 %t.ahead to i64
; D3-NEXT:       call void @llvm.riscv.ns.cond.push.i64(i32 0, i32 33, i64 [[T]], i64 0)
; With no distance given, a loop pipelined in place is 4 ahead.
; DEF-LABEL:   define {{.*}} @toggle_i1(
; DEF:         entry:
; DEF-NEXT:      call void @llvm.riscv.ns.cond.init(i32 0)
; DEF-COUNT-4:   call void @llvm.riscv.ns.cond.push.i64(i32 0, i32 33,
; DEF-NEXT:      br label %loop
; DEF:         loop:
; DEF:           [[T:%.*]] = zext i1 %t to i64
; DEF-NEXT:      call void @llvm.riscv.ns.cond.push.i64(i32 0, i32 33, i64 [[T]], i64 0)
; OFF-LABEL:   define {{.*}} @toggle_i1(
; OFF-NOT:       llvm.riscv.ns.cond
define void @toggle_i1(ptr %in, ptr %out, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
  %t = phi i1 [ false, %entry ], [ %t.next, %latch ]
  %o = phi ptr [ %out, %entry ], [ %o.next, %latch ]
  %p = getelementptr inbounds i8, ptr %in, i64 %i
  %v = load i8, ptr %p
  br i1 %t, label %odd, label %latch

odd:
  store i8 %v, ptr %o
  %o.inc = getelementptr inbounds i8, ptr %o, i64 1
  br label %latch

latch:
  %o.next = phi ptr [ %o, %loop ], [ %o.inc, %odd ]
  %t.next = xor i1 %t, true
  %i.next = add nuw nsw i64 %i, 1
  %c = icmp ult i64 %i.next, %n
  br i1 %c, label %loop, label %exit

exit:
  ret void
}

; A wider toggle, tested through a comparison:
; for (i = 0; i < n; i++) { if (t) out[i] = i; t ^= 1; }
; CHECK-LABEL: define {{.*}} @toggle_i32(
; CHECK:       entry:
; CHECK-NEXT:    call void @llvm.riscv.ns.cond.init(i32 0)
; CHECK-NEXT:    [[S0:%.*]] = sext i32 %t0 to i64
; CHECK-NEXT:    call void @llvm.riscv.ns.cond.push.i64(i32 0, i32 33, i64 [[S0]], i64 0)
; CHECK-NEXT:    %t.swp1 = xor i32 %t0, 1
; CHECK-NEXT:    [[S1:%.*]] = sext i32 %t.swp1 to i64
; CHECK-NEXT:    call void @llvm.riscv.ns.cond.push.i64(i32 0, i32 33, i64 [[S1]], i64 0)
; CHECK:         %t.swp3 = xor i32 %t0, 1
; CHECK:       loop:
; CHECK:         [[S:%.*]] = sext i32 %t to i64
; CHECK-NEXT:    call void @llvm.riscv.ns.cond.push.i64(i32 0, i32 33, i64 [[S]], i64 0)
; CHECK-NEXT:    [[POP:%.*]] = call i1 @llvm.riscv.ns.cond.pop(i32 0)
; CHECK-NEXT:    br i1 [[POP]], label %odd, label %latch
; CHECK-NOT:     swp.kernel
; OFF-LABEL:   define {{.*}} @toggle_i32(
; OFF-NOT:       llvm.riscv.ns.cond
define void @toggle_i32(ptr %out, i64 %n, i32 %t0) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
  %t = phi i32 [ %t0, %entry ], [ %t.next, %latch ]
  %set = icmp ne i32 %t, 0
  br i1 %set, label %odd, label %latch

odd:
  %p = getelementptr inbounds i64, ptr %out, i64 %i
  store i64 %i, ptr %p
  br label %latch

latch:
  %t.next = xor i32 %t, 1
  %i.next = add nuw nsw i64 %i, 1
  %c = icmp ult i64 %i.next, %n
  br i1 %c, label %loop, label %exit

exit:
  ret void
}

; Flipping an invariant set of bits: t ^= m, tested as t & 4.
; CHECK-LABEL: define {{.*}} @toggle_mask(
; CHECK:       entry:
; CHECK-NEXT:    call void @llvm.riscv.ns.cond.init(i32 0)
; CHECK-NEXT:    %bit.swp0 = and i64 0, 4
; CHECK-NEXT:    call void @llvm.riscv.ns.cond.push.i64(i32 0, i32 32, i64 %bit.swp0, i64 0)
; CHECK-NEXT:    %t.swp1 = xor i64 0, %m
; CHECK-NEXT:    %bit.swp1 = and i64 %t.swp1, 4
; CHECK-NEXT:    call void @llvm.riscv.ns.cond.push.i64(i32 0, i32 32, i64 %bit.swp1, i64 0)
; CHECK:       loop:
; CHECK:         %bit.ahead = and i64 %t, 4
; CHECK-NEXT:    call void @llvm.riscv.ns.cond.push.i64(i32 0, i32 32, i64 %bit.ahead, i64 0)
; CHECK-NEXT:    [[POP:%.*]] = call i1 @llvm.riscv.ns.cond.pop(i32 0)
; CHECK-NEXT:    br i1 [[POP]], label %latch, label %odd
; D3-LABEL:    define {{.*}} @toggle_mask(
; D3:          loop:
; D3:            %t.ahead = xor i64 %t, %m
; D3-NEXT:       %bit.ahead = and i64 %t.ahead, 4
; OFF-LABEL:   define {{.*}} @toggle_mask(
; OFF-NOT:       llvm.riscv.ns.cond
define void @toggle_mask(ptr %out, i64 %n, i64 %m) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
  %t = phi i64 [ 0, %entry ], [ %t.next, %latch ]
  %bit = and i64 %t, 4
  %set = icmp eq i64 %bit, 0
  br i1 %set, label %latch, label %odd

odd:
  %p = getelementptr inbounds i64, ptr %out, i64 %i
  store i64 %i, ptr %p
  br label %latch

latch:
  %t.next = xor i64 %m, %t
  %i.next = add nuw nsw i64 %i, 1
  %c = icmp ult i64 %i.next, %n
  br i1 %c, label %loop, label %exit

exit:
  ret void
}

; A loop that also has a condition on a load cannot compute that one for
; iterations that never run, so it gets a kernel and a drain loop as usual,
; and the toggle is advanced like any other induction variable.
; CHECK-LABEL: define {{.*}} @toggle_and_load(
; CHECK:         %swp.long.enough = icmp ugt i64 %umax, 4
; CHECK:       swp.ph:
; CHECK-NEXT:    call void @llvm.riscv.ns.cond.init(i32 0)
; CHECK-NEXT:    call void @llvm.riscv.ns.cond.init(i32 1)
; CHECK-NEXT:    call void @llvm.riscv.ns.cond.push.i64(i32 0, i32 33, i64 1, i64 0)
; CHECK:         call void @llvm.riscv.ns.cond.push.i64(i32 1, i32 40, i64 %v.swp0, i64 0)
; CHECK-NEXT:    call void @llvm.riscv.ns.cond.push.i64(i32 0, i32 33, i64 0, i64 0)
; CHECK:       loop.swp:
; CHECK:         [[T:%.*]] = zext i1 %t.swp to i64
; CHECK-NEXT:    call void @llvm.riscv.ns.cond.push.i64(i32 0, i32 33, i64 [[T]], i64 0)
; CHECK-NEXT:    [[POP0:%.*]] = call i1 @llvm.riscv.ns.cond.pop(i32 0)
; CHECK-NEXT:    br i1 [[POP0]], label %even.swp, label %mid.swp
; CHECK:       mid.swp:
; CHECK:         call void @llvm.riscv.ns.cond.push.i64(i32 1, i32 40, i64 %v.ahead, i64 0)
; CHECK-NEXT:    [[POP1:%.*]] = call i1 @llvm.riscv.ns.cond.pop(i32 1)
; The drain loop resumes the toggle and branches on it as before.
; CHECK:       swp.drain.ph:
; CHECK:         %t.swp.resume = phi i1 [ true, %entry ], [ %t.next.swp, %swp.kernel.exit ]
; CHECK:       loop:
; CHECK:         br i1 %t, label %even, label %mid
; Without the flag only the condition on the load is pipelined.
; OFF-LABEL:   define {{.*}} @toggle_and_load(
; OFF:         loop.swp:
; OFF-NOT:       llvm.riscv.ns.cond
; OFF:           br i1 %t.swp, label %even.swp, label %mid.swp
; OFF:           call i1 @llvm.riscv.ns.cond.pop(i32 0)
define i64 @toggle_and_load(ptr %in, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
  %t = phi i1 [ true, %entry ], [ %t.next, %latch ]
  %s = phi i64 [ 0, %entry ], [ %s.next, %latch ]
  br i1 %t, label %even, label %mid

even:
  %s.even = add i64 %s, 3
  br label %mid

mid:
  %s.mid = phi i64 [ %s, %loop ], [ %s.even, %even ]
  %p = getelementptr inbounds i64, ptr %in, i64 %i
  %v = load i64, ptr %p
  %neg = icmp slt i64 %v, 0
  br i1 %neg, label %latch, label %pos

pos:
  %s.pos = add i64 %s.mid, %v
  br label %latch

latch:
  %s.next = phi i64 [ %s.mid, %mid ], [ %s.pos, %pos ]
  %t.next = xor i1 %t, true
  %i.next = add nuw nsw i64 %i, 1
  %c = icmp ult i64 %i.next, %n
  br i1 %c, label %loop, label %exit

exit:
  ret i64 %s.next
}

; t ^= i flips different bits every iteration, so it is not a toggle.
; CHECK-LABEL: define {{.*}} @varying_mask(
; CHECK-NOT:     llvm.riscv.ns.cond
define void @varying_mask(ptr %out, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
  %t = phi i64 [ 0, %entry ], [ %t.next, %latch ]
  %set = icmp eq i64 %t, 0
  br i1 %set, label %latch, label %odd

odd:
  %p = getelementptr inbounds i64, ptr %out, i64 %i
  store i64 %i, ptr %p
  br label %latch

latch:
  %t.next = xor i64 %t, %i
  %i.next = add nuw nsw i64 %i, 1
  %c = icmp ult i64 %i.next, %n
  br i1 %c, label %loop, label %exit

exit:
  ret void
}

; A loop without a trip count is only pipelined with
; -riscv-ns-swp-speculative, which can then use the toggle too.
; do { v = in[i++]; if (t) s += v; t = !t; } while (v != 0);
; CHECK-LABEL: define {{.*}} @no_trip_count(
; CHECK-NOT:     call {{.*}}@llvm.riscv.ns.cond
; SPEC-LABEL:  define {{.*}} @no_trip_count(
; SPEC:        entry:
; SPEC-NEXT:     call void @llvm.riscv.ns.cond.init(i32 0)
; SPEC-COUNT-4:  call void @llvm.riscv.ns.cond.push.i64(i32 0, i32 33,
; SPEC-NEXT:     br label %loop
; SPEC:          [[T:%.*]] = zext i1 %t to i64
; SPEC-NEXT:     call void @llvm.riscv.ns.cond.push.i64(i32 0, i32 33, i64 [[T]], i64 0)
; SPEC-NEXT:     [[POP:%.*]] = call i1 @llvm.riscv.ns.cond.pop(i32 0)
; SPEC-NEXT:     br i1 [[POP]], label %odd, label %latch
define i64 @no_trip_count(ptr %in) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
  %t = phi i1 [ false, %entry ], [ %t.next, %latch ]
  %s = phi i64 [ 0, %entry ], [ %s.next, %latch ]
  %p = getelementptr inbounds i64, ptr %in, i64 %i
  %v = load i64, ptr %p
  br i1 %t, label %odd, label %latch

odd:
  %s.odd = add i64 %s, %v
  br label %latch

latch:
  %s.next = phi i64 [ %s, %loop ], [ %s.odd, %odd ]
  %t.next = xor i1 %t, true
  %i.next = add nuw nsw i64 %i, 1
  %c = icmp eq i64 %v, 0
  br i1 %c, label %exit, label %loop

exit:
  ret i64 %s.next
}
