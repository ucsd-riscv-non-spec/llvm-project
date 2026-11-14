; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv64 -O2 -stop-after=riscv-ns-loop-metadata %t/outer.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=OUTER
; RUN: not llc -mtriple=riscv64 -O2 -stop-after=riscv-ns-loop-metadata %t/location.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=LOC
; RUN: llc -mtriple=riscv64 -O2 -stop-after=riscv-ns-loop-metadata %t/valid.ll -o /dev/null
; RUN: not llc -mtriple=riscv64 -O2 -riscv-ns-disable-software-pipelining -riscv-ns-disable-ir-loop-fission -disable-loop-fission-decomposed-branch -stop-after=riscv-ns-loop-metadata %t/arity.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=ARITY
; RUN: not llc -mtriple=riscv64 -O2 -stop-after=riscv-ns-loop-metadata %t/bool.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=BOOL
; RUN: not llc -mtriple=riscv64 -O2 -stop-after=riscv-ns-loop-metadata %t/type.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=TYPE
; RUN: not llc -mtriple=riscv64 -O2 -stop-after=riscv-ns-loop-metadata %t/zero.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=RANGE
; RUN: not llc -mtriple=riscv64 -O2 -stop-after=riscv-ns-loop-metadata %t/large.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=RANGE
; RUN: not llc -mtriple=riscv64 -O2 -stop-after=riscv-ns-loop-metadata %t/duplicate.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=DUP
; RUN: not llc -mtriple=riscv64 -O2 -stop-after=riscv-ns-loop-metadata %t/force.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=FORCE
; RUN: not llc -mtriple=riscv64 -O2 -stop-after=riscv-ns-loop-metadata %t/depth.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=DEPTH
; RUN: not llc -mtriple=riscv64 -O2 -stop-after=riscv-ns-loop-metadata %t/size.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=SIZE

; None of these loops is a transformation candidate. Unknown keys and malformed
; unknown values are accepted. Equal-value duplicates are nevertheless errors.
; OUTER: error: {{.*}}invalid Non-Spec loop metadata 'llvm.loop.riscv.ns.swp.enable'{{.*}}expected a distinct, self-referencing !llvm.loop ID
; LOC: error: loop.c:17:5: {{.*}}invalid Non-Spec loop metadata 'llvm.loop.riscv.ns.force'
; ARITY: error: {{.*}}invalid Non-Spec loop metadata 'llvm.loop.riscv.ns.swp.enable' in function 'arity', loop latch %loop: expected !{!"llvm.loop.riscv.ns.swp.enable", i1 <boolean>}
; BOOL: error: {{.*}}'llvm.loop.riscv.ns.fission.ir.enable'{{.*}}expected {{.*}}i1 <boolean>
; TYPE: error: {{.*}}'llvm.loop.riscv.ns.swp.distance'{{.*}}expected {{.*}}i32 <integer in [1, 62]>
; RANGE: error: {{.*}}'llvm.loop.riscv.ns.swp.distance'{{.*}}expected {{.*}}i32 <integer in [1, 62]>
; DUP: error: {{.*}}'llvm.loop.riscv.ns.swp.toggles'{{.*}}duplicate key; expected {{.*}}i1 <boolean>
; FORCE: error: {{.*}}'llvm.loop.riscv.ns.force'{{.*}}expected {{.*}}i1 <boolean>
; DEPTH: error: {{.*}}'llvm.loop.riscv.ns.fission.queue_depth'{{.*}}expected {{.*}}i32 <integer in [1, 64]>
; SIZE: error: {{.*}}'llvm.loop.riscv.ns.swp.max_loop_size'{{.*}}expected {{.*}}i32 <integer in [1, 2147483647]>

;--- valid.ll
define void @unknown(ptr %p) {
entry:
  br label %loop
loop:
  %v = load volatile i1, ptr %p
  br i1 %v, label %loop, label %exit, !llvm.loop !0
exit:
  ret void
}
!0 = distinct !{!0, !1, !2}
!1 = !{!"llvm.loop.riscv.ns.future", !"anything", i64 -1}
!2 = !{!"llvm.loop.riscv.ns.swp.distance", i32 62}

;--- arity.ll
define void @arity(ptr %p) {
entry:
  br label %loop
loop:
  %v = load volatile i1, ptr %p
  br i1 %v, label %loop, label %exit, !llvm.loop !0
exit:
  ret void
}
!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.riscv.ns.swp.enable"}

;--- bool.ll
define void @bad_bool(ptr %p) {
entry:
  br label %loop
loop:
  %v = load volatile i1, ptr %p
  br i1 %v, label %loop, label %exit, !llvm.loop !0
exit:
  ret void
}
!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.riscv.ns.fission.ir.enable", i32 1}

;--- type.ll
define void @bad_type(ptr %p) {
entry:
  br label %loop
loop:
  %v = load volatile i1, ptr %p
  br i1 %v, label %loop, label %exit, !llvm.loop !0
exit:
  ret void
}
!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.riscv.ns.swp.distance", !"eight"}

;--- zero.ll
define void @zero(ptr %p) {
entry:
  br label %loop
loop:
  %v = load volatile i1, ptr %p
  br i1 %v, label %loop, label %exit, !llvm.loop !0
exit:
  ret void
}
!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.riscv.ns.swp.distance", i32 0}

;--- large.ll
define void @large(ptr %p) {
entry:
  br label %loop
loop:
  %v = load volatile i1, ptr %p
  br i1 %v, label %loop, label %exit, !llvm.loop !0
exit:
  ret void
}
!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.riscv.ns.swp.distance", i32 63}

;--- duplicate.ll
define void @duplicate(ptr %p) {
entry:
  br label %loop
loop:
  %v = load volatile i1, ptr %p
  br i1 %v, label %loop, label %exit, !llvm.loop !0
exit:
  ret void
}
!0 = distinct !{!0, !1, !1}
!1 = !{!"llvm.loop.riscv.ns.swp.toggles", i1 true}

;--- force.ll
define void @force(ptr %p) {
entry:
  br label %loop
loop:
  %v = load volatile i1, ptr %p
  br i1 %v, label %loop, label %exit, !llvm.loop !0
exit:
  ret void
}
!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.riscv.ns.force", i1 true, i1 false}

;--- depth.ll
define void @depth(ptr %p) {
entry:
  br label %loop
loop:
  %v = load volatile i1, ptr %p
  br i1 %v, label %loop, label %exit, !llvm.loop !0
exit:
  ret void
}
!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.riscv.ns.fission.queue_depth", i32 65}

;--- size.ll
define void @size(ptr %p) {
entry:
  br label %loop
loop:
  %v = load volatile i1, ptr %p
  br i1 %v, label %loop, label %exit, !llvm.loop !0
exit:
  ret void
}
!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.riscv.ns.swp.max_loop_size", i32 -1}

;--- location.ll
source_filename = "loop.c"
define void @located(ptr %p) !dbg !4 {
entry:
  br label %loop
loop:
  %v = load volatile i1, ptr %p
  br i1 %v, label %loop, label %exit, !llvm.loop !10
exit:
  ret void
}
!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!1}
!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !2, producer: "test", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = !DIFile(filename: "loop.c", directory: "/tmp")
!3 = !DISubroutineType(types: !{})
!4 = distinct !DISubprogram(name: "located", scope: !2, file: !2, line: 1, type: !3, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
!5 = !DILocation(line: 17, column: 5, scope: !4)
!10 = distinct !{!10, !5, !11}
!11 = !{!"llvm.loop.riscv.ns.force", i8 1}

;--- outer.ll
define void @outer(ptr %p) {
entry:
  br label %loop
loop:
  %v = load volatile i1, ptr %p
  br i1 %v, label %loop, label %exit, !llvm.loop !0
exit:
  ret void
}
!0 = !{!1}
!1 = !{!"llvm.loop.riscv.ns.swp.enable", i1 false}
