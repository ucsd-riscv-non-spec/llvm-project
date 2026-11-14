; RUN: split-file %s %t
; RUN: opt -mtriple=riscv64 -mattr=+v -S -passes='default<O2>' %t/input.ll -o %t/optimized.ll
; RUN: %python %t/check.py %t/optimized.ll
; RUN: llc -O2 -stop-after=riscv-ns-loop-metadata %t/optimized.ll -o /dev/null
; RUN: opt -mtriple=riscv64 -mattr=+v -S -passes='lto-pre-link<O2>' %t/input.ll -o %t/prelink.ll
; RUN: opt -mtriple=riscv64 -mattr=+v -S -passes='lto<O2>' %t/prelink.ll -o %t/lto.ll
; RUN: %python %t/check.py %t/lto.ll
; RUN: llc -O2 -stop-after=riscv-ns-loop-metadata %t/lto.ll -o /dev/null
; RUN: opt -S -passes='loop-simplify,lcssa,loop-rotate,loop-unroll' %t/input.ll -o %t/unrolled.ll
; RUN: %python %t/check.py %t/unrolled.ll
; RUN: opt -S -passes='loop-simplify,lcssa,loop-unroll' -unroll-peel-count=2 %t/input.ll -o %t/peeled.ll
; RUN: %python %t/check.py %t/peeled.ll
; RUN: opt -S -passes='loop-simplify,lcssa,loop-distribute' %t/input.ll -o %t/distributed.ll
; RUN: %python %t/check.py %t/distributed.ll
; RUN: FileCheck %s --check-prefix=DISTRIBUTE < %t/distributed.ll

; Audit surviving descendant loops, including vector and scalar remainders,
; partial/runtime unrolling, rotation and peeling, in ordinary and LTO pipelines.
; DISTRIBUTE: define void @distribute(
; DISTRIBUTE: loop.ldist1:
; DISTRIBUTE: loop:
; These properties remain requests until the Non-Spec passes consume them.

;--- check.py
import re
import sys

text = open(sys.argv[1]).read()
metadata = dict(re.findall(r"^!(\d+) = (.+)$", text, re.M))
loop_ids = re.findall(r"!llvm.loop !(\d+)", text)
assert loop_ids, "expected surviving loops"
for loop_id in loop_ids:
    properties = " ".join(metadata.get(ref, "") for ref in
                          re.findall(r"!(\d+)", metadata[loop_id])[1:])
    for key in ("swp.enable", "fission.ir.enable", "fission.machine.enable",
                "force", "swp.distance", "fission.queue_depth"):
        assert '"llvm.loop.riscv.ns.' + key + '"' in properties, (loop_id, key)

;--- input.ll
define void @copy(ptr noalias %a, ptr noalias %b, i64 %n) {
entry:
  %positive = icmp ugt i64 %n, 0
  br i1 %positive, label %ph, label %exit
ph:
  br label %loop
loop:
  %i = phi i64 [ 0, %ph ], [ %next, %loop ]
  %p = getelementptr i32, ptr %a, i64 %i
  %v = load i32, ptr %p
  %q = getelementptr i32, ptr %b, i64 %i
  %w = add i32 %v, 1
  store i32 %w, ptr %q
  %next = add nuw i64 %i, 1
  %again = icmp ult i64 %next, %n
  br i1 %again, label %loop, label %exit, !llvm.loop !0
exit:
  ret void
}

define void @partial(ptr noalias %a, ptr noalias %b, i64 %n) {
entry:
  %positive = icmp ugt i64 %n, 0
  br i1 %positive, label %ph, label %exit
ph:
  br label %loop
loop:
  %i = phi i64 [ 0, %ph ], [ %next, %loop ]
  %p = getelementptr i32, ptr %a, i64 %i
  %v = load i32, ptr %p
  %q = getelementptr i32, ptr %b, i64 %i
  store i32 %v, ptr %q
  %next = add nuw i64 %i, 1
  %again = icmp ult i64 %next, %n
  br i1 %again, label %loop, label %exit, !llvm.loop !10
exit:
  ret void
}


define void @distribute(ptr noalias %a, ptr noalias %b, ptr noalias %c,
                        ptr noalias %d, ptr noalias %e) {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %next, %loop ]
  %ap = getelementptr i32, ptr %a, i64 %i
  %av = load i32, ptr %ap
  %bp = getelementptr i32, ptr %b, i64 %i
  %bv = load i32, ptr %bp
  %product = mul i32 %av, %bv
  %next = add nuw nsw i64 %i, 1
  %anext = getelementptr i32, ptr %a, i64 %next
  store i32 %product, ptr %anext
  %dp = getelementptr i32, ptr %d, i64 %i
  %dv = load i32, ptr %dp
  %ep = getelementptr i32, ptr %e, i64 %i
  %ev = load i32, ptr %ep
  %independent = mul i32 %dv, %ev
  %cp = getelementptr i32, ptr %c, i64 %i
  store i32 %independent, ptr %cp
  %done = icmp eq i64 %next, 20
  br i1 %done, label %exit, label %loop, !llvm.loop !20
exit:
  ret void
}

!0 = distinct !{!0, !1, !2, !3, !4, !5, !6}
!1 = !{!"llvm.loop.riscv.ns.swp.enable", i1 false}
!2 = !{!"llvm.loop.riscv.ns.fission.ir.enable", i1 false}
!3 = !{!"llvm.loop.riscv.ns.fission.machine.enable", i1 false}
!4 = !{!"llvm.loop.riscv.ns.force", i1 false}
!5 = !{!"llvm.loop.riscv.ns.swp.distance", i32 3}
!6 = !{!"llvm.loop.riscv.ns.fission.queue_depth", i32 16}
!10 = distinct !{!10, !1, !2, !3, !4, !5, !6, !11, !12}
!11 = !{!"llvm.loop.unroll.count", i32 4}
!12 = !{!"llvm.loop.vectorize.enable", i1 false}

!20 = distinct !{!20, !1, !2, !3, !4, !5, !6, !21}
!21 = !{!"llvm.loop.distribute.enable", i1 true}
