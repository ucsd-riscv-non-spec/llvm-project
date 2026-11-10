; RUN: llc -mtriple=riscv64 -mattr=+m -O2 -riscv-ns-swp-distance=4 \
; RUN:   -riscv-ns-swp-speculative -riscv-ns-swp-shift-recurrences < %s \
; RUN:   | FileCheck %s

; Tests that RISCVBranchSetupHoisting leaves the setup of a queue whose
; condition is on a shift recurrence in the loop's preheader, as for
; ns-branch-condition-pipelining-speculative-asm.ll: only that setup discards
; the conditions a previous entry left queued.

; for (j = 0; j < m; j++) { x = a[j]; do { c += x & 15; x >>= 4; } while (x >= 16); }
define i64 @reentry_shift(ptr %a, i64 %m) {
; CHECK-LABEL: reentry_shift:
; The other setups are hoisted to the entry, but not the queue's.
; CHECK-NOT:     .Lns_q_
; The setup is in the inner loop's preheader, after the load of the new x.
; CHECK:       # %outer
; CHECK:         ld [[X:a[0-9]+]],
; CHECK:         bmovs [[Q:b[0-9]+]], .Lns_q_
; CHECK-NEXT:    bmovt [[Q]],
; The prologue queues iterations 0-3, on x >> 4, 8, 12 and 16.
; CHECK-NEXT:    srai [[T:a[0-9]+]], [[X]], 4
; CHECK-NEXT:    bmovc_bgeu [[Q]], [[T]],
; CHECK:         srai [[T]], [[X]], 16
; CHECK-NEXT:    bmovc_bgeu [[Q]], [[T]],
; CHECK:       # %inner
; CHECK-NOT:     bmovs [[Q]],
; The kernel queues iteration i + 4: its x >> 4, which is x >> 20.
; CHECK:         srai {{a[0-9]+}}, [[X]], 20
; CHECK:       .Lns_q_
; CHECK-NEXT:    pb [[Q]]
entry:
  br label %outer

outer:
  %j = phi i64 [ 0, %entry ], [ %j.next, %outer.latch ]
  %acc = phi i64 [ 0, %entry ], [ %c.next, %outer.latch ]
  %q = getelementptr inbounds i64, ptr %a, i64 %j
  %x0 = load i64, ptr %q
  br label %inner

inner:
  %x = phi i64 [ %x0, %outer ], [ %x.next, %inner ]
  %c = phi i64 [ %acc, %outer ], [ %c.next, %inner ]
  %lo = and i64 %x, 15
  %c.next = add i64 %c, %lo
  %x.next = ashr i64 %x, 4
  %done = icmp ult i64 %x.next, 16
  br i1 %done, label %outer.latch, label %inner

outer.latch:
  %j.next = add nuw nsw i64 %j, 1
  %oc = icmp ult i64 %j.next, %m
  br i1 %oc, label %outer, label %exit

exit:
  ret i64 %c.next
}
