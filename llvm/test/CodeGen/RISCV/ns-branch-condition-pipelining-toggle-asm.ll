; RUN: llc -mtriple=riscv64 -mattr=+m -O2 -riscv-ns-swp-toggles < %s \
; RUN:   | FileCheck %s

; Tests that a counted loop pipelined in place by -riscv-ns-swp-toggles
; stays a hardware loop, and that RISCVBranchSetupHoisting leaves the setup
; of its queue in the loop's preheader, as for
; ns-branch-condition-pipelining-speculative-asm.ll: the loop leaves four
; conditions queued when it exits, which only that setup discards before the
; loop is entered again.

; for (j = 0; j < m; j++) { t = 0; for (i = 0; i < n; i++) { if (t) s += a[i]; t = !t; } }
define i64 @reentry_toggle(ptr %a, i64 %m, i64 %n) {
; CHECK-LABEL: reentry_toggle:
; The other setups are hoisted to the entry, but not the queue's.
; CHECK-NOT:     .Lns_q_
; CHECK:       # %outer
; CHECK:         li [[T:a[0-9]+]], 0
; CHECK-NEXT:    bmovs [[Q:b[0-9]+]], .Lns_q_
; CHECK-NEXT:    bmovt [[Q]],
; The prologue queues iterations 0-3: t is 0, 1, 0, 1.
; CHECK-NEXT:    bmovc_beq [[Q]], zero, zero
; CHECK-NEXT:    bmovc_beq [[Q]], [[ONE:a[0-9]+]], zero
; CHECK-NEXT:    bmovc_beq [[Q]], zero, zero
; CHECK-NEXT:    bmovc_beq [[Q]], [[ONE]], zero
; The inner loop is still counted by a hardware loop.
; CHECK-NEXT:    bmovc_loop [[LOOP:b[0-9]+]],
; CHECK:       # %latch
; CHECK-NEXT:    #
; CHECK-NEXT:    xori [[T]], [[T]], 1
; CHECK:         pb [[LOOP]]
; Each iteration queues t for four iterations on, which is t itself.
; CHECK:       # %inner
; CHECK-NOT:     bmovs [[Q]],
; CHECK:         andi [[C:a[0-9]+]], [[T]], 1
; CHECK-NEXT:    bmovc_beq [[Q]], [[C]], zero
; CHECK-NEXT:  .Lns_q_
; CHECK-NEXT:    pb [[Q]]
entry:
  br label %outer

outer:
  %j = phi i64 [ 0, %entry ], [ %j.next, %outer.latch ]
  %acc = phi i64 [ 0, %entry ], [ %s.next, %outer.latch ]
  br label %inner

inner:
  %i = phi i64 [ 0, %outer ], [ %i.next, %latch ]
  %t = phi i1 [ false, %outer ], [ %t.next, %latch ]
  %s = phi i64 [ %acc, %outer ], [ %s.next, %latch ]
  br i1 %t, label %odd, label %latch

odd:
  %q = getelementptr inbounds i64, ptr %a, i64 %i
  %v = load i64, ptr %q
  %s.odd = add i64 %s, %v
  br label %latch

latch:
  %s.next = phi i64 [ %s, %inner ], [ %s.odd, %odd ]
  %t.next = xor i1 %t, true
  %i.next = add nuw nsw i64 %i, 1
  %c = icmp ult i64 %i.next, %n
  br i1 %c, label %inner, label %outer.latch

outer.latch:
  %j.next = add nuw nsw i64 %j, 1
  %oc = icmp ult i64 %j.next, %m
  br i1 %oc, label %outer, label %exit

exit:
  ret i64 %s.next
}
