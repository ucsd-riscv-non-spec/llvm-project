; RUN: llc -mtriple=riscv64 -mattr=+m -O2 < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -mattr=+m -O2 \
; RUN:   -riscv-ns-disable-hoist-freq-from-best < %s \
; RUN:   | FileCheck %s --check-prefix=OFF

; The setups of a branch inside a loop are hoisted no further than the
; preheader when the loop is rarely entered: the hoisting pass compares the
; entry's frequency with the preheader's, not with the loop body's, which
; includes the trip count. With the change off, they go to the entry and run
; on every call.

; CHECK-LABEL: rare_loop:
; CHECK-NOT:     .Lns_bge_
; CHECK:         pb
; CHECK:       # %loop.preheader
; CHECK:         bmovs {{b[0-9]+}}, .Lns_bge_0
; CHECK:       .Lns_bge_0:
; CHECK-LABEL: .Lfunc_end0:
; OFF-LABEL:   rare_loop:
; OFF:           bmovs {{b[0-9]+}}, .Lns_bge_0
; OFF:         .Lns_bne_0:
; OFF-NEXT:      pb
; OFF:         # %loop.preheader
; OFF-NOT:       bmovs {{b[0-9]+}}, .Lns_bge_0
; OFF:         .Lns_bge_0:
; OFF-LABEL:   .Lfunc_end0:

; Entered 3 times in 4: the entry is within twice the preheader, so the
; setups go to the entry either way.
; CHECK-LABEL: common_loop:
; CHECK:         bmovs {{b[0-9]+}}, .Lns_bge_1
; CHECK:       .Lns_beq_0:
; CHECK-NEXT:    pb
; CHECK:       # %loop.preheader
; CHECK-NOT:     .Lns_bge_
; CHECK:       .Lns_bge_1:
; OFF-LABEL:   common_loop:
; OFF:           bmovs {{b[0-9]+}}, .Lns_bge_1
; OFF:         .Lns_beq_0:
; OFF-NEXT:      pb

define void @rare_loop(ptr %p, i32 %n, i1 %c) {
entry:
  br i1 %c, label %loop, label %exit, !prof !0

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %latch ]
  %a = getelementptr inbounds i32, ptr %p, i32 %i
  %v = load i32, ptr %a
  %neg = icmp slt i32 %v, 0
  br i1 %neg, label %clear, label %latch, !prof !3

clear:
  store i32 0, ptr %a
  br label %latch

latch:
  %i.next = add i32 %i, 1
  %done = icmp eq i32 %i.next, %n
  br i1 %done, label %exit, label %loop, !prof !1

exit:
  ret void
}

define void @common_loop(ptr %p, i32 %n, i1 %c) {
entry:
  br i1 %c, label %loop, label %exit, !prof !2

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %latch ]
  %a = getelementptr inbounds i32, ptr %p, i32 %i
  %v = load i32, ptr %a
  %neg = icmp slt i32 %v, 0
  br i1 %neg, label %clear, label %latch, !prof !3

clear:
  store i32 0, ptr %a
  br label %latch

latch:
  %i.next = add i32 %i, 1
  %done = icmp eq i32 %i.next, %n
  br i1 %done, label %exit, label %loop, !prof !1

exit:
  ret void
}

!0 = !{!"branch_weights", i32 1, i32 15}
!1 = !{!"branch_weights", i32 1, i32 31}
!2 = !{!"branch_weights", i32 3, i32 1}
!3 = !{!"branch_weights", i32 1, i32 1}
