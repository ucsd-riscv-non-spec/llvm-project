; RUN: llc -mtriple=riscv64 -O2 < %s | FileCheck %s
; RUN: not llc -mtriple=riscv64 -mattr=+e -target-abi=lp64e -O2 < %s 2>&1 \
; RUN:   | FileCheck %s --check-prefix=RVE

; A hardware loop's B-register is live across every call in the loop, so it
; has to be callee-saved (B24-B31), and the function using it saves it.

declare void @use(i64, i64)

; Branches, but keeps no B-register live across a call, so it must neither
; touch nor save a callee-saved one.
define i64 @callee(i64 %x) noinline {
; CHECK-LABEL: callee:
; CHECK-NOT:     {{bpush|bpop|b2[4-9]|b3[01]}}
; CHECK:       .Lfunc_end0:
entry:
  %c = icmp sgt i64 %x, 10
  br i1 %c, label %big, label %small
big:
  %a = mul i64 %x, 3
  br label %done
small:
  %b = add i64 %x, 7
  br label %done
done:
  %r = phi i64 [ %a, %big ], [ %b, %small ]
  ret i64 %r
}

; RVE: function 'one_loop' keeps 1 branch register(s) live across calls, but only 0 callee-saved branch register(s) are available
define i64 @one_loop(ptr %p, i64 %n) {
; CHECK-LABEL: one_loop:
; CHECK:         bpush b24, [[SLOT:[0-9]+]](sp)
; The setup cannot move above the save.
; CHECK-NOT:     b24
; CHECK:         bmovs b24, [[LOOP:.Lns_loop_[0-9]+]]
; CHECK:         bmovc_loop b24,
; CHECK:         pbal {{b[0-9]+}}, ra
; CHECK:       [[LOOP]]:
; CHECK-NEXT:    pb b24
; CHECK:         bpop b24, [[SLOT]](sp)
; CHECK-NOT:     b24
; CHECK:       .Lns_return_{{[0-9]+}}:
; CHECK-NEXT:    pb {{b([0-9]|1[0-9]|2[0-3])$}}
entry:
  %guard = icmp sgt i64 %n, 0
  br i1 %guard, label %loop, label %exit
loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %acc = phi i64 [ 0, %entry ], [ %acc.next, %loop ]
  %gep = getelementptr inbounds i64, ptr %p, i64 %i
  %v = load i64, ptr %gep
  %r = call i64 @callee(i64 %v)
  %acc.next = add i64 %acc, %r
  %i.next = add nuw nsw i64 %i, 1
  %cmp = icmp eq i64 %i.next, %n
  br i1 %cmp, label %exit, label %loop
exit:
  %res = phi i64 [ 0, %entry ], [ %acc.next, %loop ]
  ret i64 %res
}

; Both loops' B-registers are live across the call.
define void @nested(i64 %n, i64 %m) {
; CHECK-LABEL: nested:
; CHECK-DAG:     bpush b24,
; CHECK-DAG:     bpush b25,
; CHECK:         bmovc_loop [[OUTER:b2[45]]],
; CHECK:         bmovc_loop [[INNER:b2[45]]],
; CHECK:         pbal {{b[0-9]+}}, ra
; CHECK:         pb [[INNER]]
; CHECK:         pb [[OUTER]]
; CHECK-DAG:     bpop b24,
; CHECK-DAG:     bpop b25,
; CHECK:       .Lfunc_end2:
entry:
  %gn = icmp sgt i64 %n, 0
  %gm = icmp sgt i64 %m, 0
  %g = and i1 %gn, %gm
  br i1 %g, label %outer, label %exit
outer:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  br label %inner
inner:
  %j = phi i64 [ 0, %outer ], [ %j.next, %inner ]
  call void @use(i64 %i, i64 %j)
  %j.next = add nuw nsw i64 %j, 1
  %jc = icmp eq i64 %j.next, %m
  br i1 %jc, label %outer.latch, label %inner
outer.latch:
  %i.next = add nuw nsw i64 %i, 1
  %ic = icmp eq i64 %i.next, %n
  br i1 %ic, label %exit, label %outer
exit:
  ret void
}

; Without a call the loop's B-register need not be callee-saved.
define void @no_call(ptr %p, i64 %n) {
; CHECK-LABEL: no_call:
; CHECK-NOT:     {{bpush|bpop|b2[4-9]|b3[01]}}
; CHECK:         bmovc_loop
; CHECK-NOT:     {{bpush|bpop|b2[4-9]|b3[01]}}
; CHECK:       .Lfunc_end3:
entry:
  %guard = icmp sgt i64 %n, 0
  br i1 %guard, label %loop, label %exit
loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %gep = getelementptr inbounds i64, ptr %p, i64 %i
  store i64 %i, ptr %gep
  %i.next = add nuw nsw i64 %i, 1
  %cmp = icmp eq i64 %i.next, %n
  br i1 %cmp, label %exit, label %loop
exit:
  ret void
}

@flag = external global i8

declare signext i32 @f1(i8 signext)
declare signext i32 @f2(i8 signext)
declare signext i32 @f3(i8 signext)

; RISCVLoopFission splits this loop in two: the first queues every
; iteration's condition in a B-register, and the second consumes them around
; its calls. The main register allocator gives that B-register the first
; callee-saved one, so the second loop's own B-register needs the next.
define signext i32 @split_loop() nounwind {
; CHECK-LABEL: split_loop:
; CHECK-DAG:     bpush b24,
; CHECK-DAG:     bpush b25,
; CHECK:         bmovc_beq b24,
; CHECK:         bmovc_loop b25,
; CHECK:         pbal {{b[0-9]+}}, ra
; CHECK:         pb b25
; CHECK-DAG:     bpop b24,
; CHECK-DAG:     bpop b25,
; CHECK:       .Lfunc_end4:
entry:
  br label %loop

loop:
  %i = phi i32 [ 16, %entry ], [ %i.next, %latch ]
  %v = load i8, ptr @flag, align 1
  %c = icmp eq i8 %v, 0
  br i1 %c, label %zero, label %nonzero

nonzero:
  %r1 = tail call signext i32 @f1(i8 signext %v)
  %r2 = tail call signext i32 @f2(i8 signext %v)
  %r3 = tail call signext i32 @f3(i8 signext %v)
  br label %latch

zero:
  %r4 = tail call signext i32 @f2(i8 signext %v)
  br label %latch

latch:
  %i.next = add nsw i32 %i, -1
  %done = icmp eq i32 %i.next, 0
  br i1 %done, label %exit, label %loop

exit:
  ret i32 0
}
