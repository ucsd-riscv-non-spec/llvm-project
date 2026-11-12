; RUN: llc -mtriple=riscv64 -mattr=+m -O2 < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -mattr=+m -O2 -riscv-ns-disable-split-exit-jumps \
; RUN:   < %s | FileCheck %s --check-prefix=OFF

; Tests where the setup of a loop's exit jump ends up. When the loop's PB
; does not branch to the block after it, the loop exits through a PB to its
; exit; RISCVExpandPseudo gives that jump and its setup a block of their own,
; after the loop's PB, so that the setup runs when the loop exits, not on
; every iteration.

declare i64 @f(i64)

; for (j = 0; j < m; j++) { s = 0; for (i = 0; i < n; i++) s += f(i); p[j] = s; }
; The call keeps every setup in a loop from leaving it. Each loop's exit jump
; follows the loop's PB, and its setup runs on the way out, once, not in the
; loop on every iteration.
define void @call_loop(ptr %p, i64 %m, i64 %n) {
; CHECK-LABEL: call_loop:
; The outer latch: its loop PB, then the exit jump's setup and the jump.
; CHECK:       # %outer.latch
; CHECK-NOT:     .Lns_j_
; CHECK:       .Lns_loop_{{[0-9]+}}:
; CHECK-NEXT:    pb [[OUTER:b[0-9]+]]
; CHECK-NEXT:  # %bb.{{[0-9]+}}:
; CHECK-NEXT:    bmovs [[OEXIT:b[0-9]+]], .Lns_j_
; CHECK-NEXT:    bmovt [[OEXIT]], .LBB0_
; CHECK-NEXT:  .Lns_j_
; CHECK-NEXT:    pb [[OEXIT]]
; The inner loop: the call, the loop PB, then the exit jump's setup.
; CHECK:       # %inner
; CHECK-NOT:     .Lns_j_
; CHECK:         pbal
; CHECK-NOT:     .Lns_j_
; CHECK:       .Lns_loop_{{[0-9]+}}:
; CHECK-NEXT:    pb [[INNER:b[0-9]+]]
; CHECK-NEXT:  # %bb.{{[0-9]+}}:
; CHECK-NEXT:  # in Loop:
; CHECK-NEXT:    bmovs [[IEXIT:b[0-9]+]], .Lns_j_
; CHECK-NEXT:    bmovt [[IEXIT]], .LBB0_
; CHECK-NEXT:  .Lns_j_
; CHECK-NEXT:    pb [[IEXIT]]
; Without the change, both setups run before their loop's PB.
; OFF-LABEL:   call_loop:
; OFF:         # %outer.latch
; OFF-NEXT:    # in Loop:
; OFF-NEXT:      bmovs [[OEXIT:b[0-9]+]], .Lns_j_
; OFF:         .Lns_loop_{{[0-9]+}}:
; OFF-NEXT:      pb
; OFF-NEXT:    .Lns_j_
; OFF-NEXT:      pb [[OEXIT]]
; OFF:           pbal
; OFF:           bmovs [[IEXIT:b[0-9]+]], .Lns_j_
; OFF:         .Lns_loop_{{[0-9]+}}:
; OFF-NEXT:      pb
; OFF-NEXT:    .Lns_j_
; OFF-NEXT:      pb [[IEXIT]]
entry:
  %cm = icmp sgt i64 %m, 0
  br i1 %cm, label %outer, label %exit
outer:
  %j = phi i64 [ 0, %entry ], [ %j.next, %outer.latch ]
  %cn = icmp sgt i64 %n, 0
  br i1 %cn, label %inner, label %outer.latch
inner:
  %i = phi i64 [ 0, %outer ], [ %i.next, %inner ]
  %s = phi i64 [ 0, %outer ], [ %s.next, %inner ]
  %v = call i64 @f(i64 %i)
  %s.next = add i64 %s, %v
  %i.next = add nuw nsw i64 %i, 1
  %ci = icmp eq i64 %i.next, %n
  br i1 %ci, label %outer.latch, label %inner
outer.latch:
  %r = phi i64 [ 0, %outer ], [ %s.next, %inner ]
  %a = getelementptr i64, ptr %p, i64 %j
  store i64 %r, ptr %a
  %j.next = add nuw nsw i64 %j, 1
  %cj = icmp eq i64 %j.next, %m
  br i1 %cj, label %exit, label %outer
exit:
  ret void
}

; The same without the call: for (...) { s = 0; for (...) s += q[i] * j; p[j] = s; }
; Nothing keeps the exit jumps' setups in their loops: they are hoisted out,
; and the jumps' own blocks are left with only the PB.
define void @nocall_loop(ptr %p, ptr %q, i64 %m, i64 %n) {
; CHECK-LABEL: nocall_loop:
; CHECK:       # %outer.preheader
; CHECK:       # %outer.latch
; CHECK-NOT:     bmovs
; CHECK:       .Lns_loop_{{[0-9]+}}:
; CHECK-NEXT:    pb
; CHECK-NEXT:  # %bb.{{[0-9]+}}:
; CHECK-NEXT:  .Lns_j_
; CHECK-NEXT:    pb
; CHECK:       # %outer
; CHECK-NOT:     bmovs {{b[0-9]+}}, .Lns_j_
; CHECK:       # %inner
; CHECK-NOT:     bmovs
; CHECK:       .Lns_loop_{{[0-9]+}}:
; CHECK-NEXT:    pb
; CHECK-NEXT:  # %bb.{{[0-9]+}}:
; CHECK-NEXT:  # in Loop:
; CHECK-NEXT:  .Lns_j_
; CHECK-NEXT:    pb
entry:
  %cm = icmp sgt i64 %m, 0
  br i1 %cm, label %outer, label %exit
outer:
  %j = phi i64 [ 0, %entry ], [ %j.next, %outer.latch ]
  %cn = icmp sgt i64 %n, 0
  br i1 %cn, label %inner, label %outer.latch
inner:
  %i = phi i64 [ 0, %outer ], [ %i.next, %inner ]
  %s = phi i64 [ 0, %outer ], [ %s.next, %inner ]
  %qa = getelementptr i64, ptr %q, i64 %i
  %qv = load i64, ptr %qa
  %v = mul i64 %qv, %j
  %s.next = add i64 %s, %v
  %i.next = add nuw nsw i64 %i, 1
  %ci = icmp eq i64 %i.next, %n
  br i1 %ci, label %outer.latch, label %inner
outer.latch:
  %r = phi i64 [ 0, %outer ], [ %s.next, %inner ]
  %a = getelementptr i64, ptr %p, i64 %j
  store i64 %r, ptr %a
  %j.next = add nuw nsw i64 %j, 1
  %cj = icmp eq i64 %j.next, %m
  br i1 %cj, label %exit, label %outer
exit:
  ret void
}
