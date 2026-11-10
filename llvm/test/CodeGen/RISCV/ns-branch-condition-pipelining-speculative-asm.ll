; RUN: llc -mtriple=riscv64 -mattr=+m -O2 -riscv-ns-swp-distance=4 \
; RUN:   -riscv-ns-swp-speculative-distance=4 \
; RUN:   -riscv-ns-swp-speculative < %s | FileCheck %s

; Tests that RISCVBranchSetupHoisting leaves the setup of a queue from
; -riscv-ns-swp-speculative in the loop's preheader. A speculative loop
; leaves up to D conditions queued when it exits, which only that setup
; (its BMOVS and BMOVT, which empty the queue) discards before the loop is
; entered again. Hoisted out of an enclosing loop, it would run once, and the
; second entry would consume the first entry's leftover conditions.

; for (j = 0; j < m; j++) { i = 0; while (i * i < n + j) i++; *p = i; }
define void @reentry(ptr %p, i64 %m, i64 %n) {
; CHECK-LABEL: reentry:
; The other setups are hoisted to the entry, but not the queue's.
; CHECK-NOT:     .Lns_q_
; CHECK:       # %outer
; The setup is the first instruction of the inner loop's preheader.
; CHECK-NOT:     {{^	[a-z]}}
; CHECK:         bmovs [[Q:b[0-9]+]], .Lns_q_
; CHECK-NEXT:    bmovt [[Q]], .LBB0_2
; CHECK-COUNT-4: bmovc_blt [[Q]],
; CHECK:       # %inner
; CHECK-NOT:     bmovs [[Q]],
; CHECK:         bmovc_blt [[Q]],
; CHECK:       .Lns_q_
; CHECK-NEXT:    pb [[Q]]
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

; The same with a loop-invariant limit, which hoisting would otherwise be
; free to move with the setup.
; for (j = 0; j < m; j++) { i = 0; while (i * i < n) i++; p[j] = i; }
define void @reentry_invariant(ptr %p, i64 %m, i64 %n) {
; CHECK-LABEL: reentry_invariant:
; CHECK-NOT:     .Lns_q_
; CHECK:       # %outer
; The setup is the first instruction of the inner loop's preheader.
; CHECK-NOT:     {{^	[a-z]}}
; CHECK:         bmovs [[Q:b[0-9]+]], .Lns_q_
; CHECK-NEXT:    bmovt [[Q]],
; CHECK:       # %inner
; CHECK-NOT:     bmovs [[Q]],
; CHECK:       .Lns_q_
; CHECK-NEXT:    pb [[Q]]
entry:
  br label %outer

outer:
  %j = phi i64 [ 0, %entry ], [ %j.next, %outer.latch ]
  br label %inner

inner:
  %i = phi i64 [ 0, %outer ], [ %inc, %inner ]
  %inc = add nuw nsw i64 %i, 1
  %sq = mul i64 %i, %i
  %c = icmp slt i64 %sq, %n
  br i1 %c, label %inner, label %outer.latch

outer.latch:
  %q = getelementptr inbounds i64, ptr %p, i64 %j
  store i64 %i, ptr %q
  %j.next = add nuw nsw i64 %j, 1
  %oc = icmp ult i64 %j.next, %m
  br i1 %oc, label %outer, label %exit

exit:
  ret void
}
