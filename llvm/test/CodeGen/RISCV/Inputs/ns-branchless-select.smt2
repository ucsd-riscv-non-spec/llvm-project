; Local refinement proof for the scalar mask blend.
; Run: z3 ns-branchless-select.smt2
; Expected: unsat, unsat. See llvm/docs/RISCV/NonSpecBranchlessSelect.md.

; RV32: arbitrary values, poison flags, and masks of defined bits.
(reset)
(set-logic QF_BV)
(define-sort Word () (_ BitVec 32))
(declare-const c Word)
(declare-const t Word)
(declare-const f Word)
(declare-const t-defined Word)
(declare-const f-defined Word)
(declare-const t-poison Bool)
(declare-const f-poison Bool)
(declare-const frozen-t Word)
(declare-const frozen-f Word)

; A defined SELECT condition obeys ZeroOrOneBooleanContent. A poison
; condition imposes no result constraint; undef may choose either boolean.
(assert (or (= c (_ bv0 32)) (= c (_ bv1 32))))
; Freeze preserves all defined bits of nonpoison inputs. Its arbitrary
; choices for other bits are shared across all uses of that freeze.
(assert (=> (not t-poison)
            (= (bvand frozen-t t-defined) (bvand t t-defined))))
(assert (=> (not f-poison)
            (= (bvand frozen-f f-defined) (bvand f f-defined))))

(define-fun normalized-c () Word
  (ite (= c (_ bv0 32)) (_ bv0 32) (_ bv1 32)))
(define-fun mask () Word (bvneg normalized-c))
(define-fun result () Word
  (bvxor frozen-f (bvand (bvxor frozen-t frozen-f) mask)))
(define-fun selected () Word (ite (= c (_ bv1 32)) t f))
(define-fun selected-defined () Word
  (ite (= c (_ bv1 32)) t-defined f-defined))
(define-fun selected-poison () Bool
  (ite (= c (_ bv1 32)) t-poison f-poison))

; A counterexample would change a defined bit of a nonpoison chosen arm.
; The unchosen arm's poison flag and defined-bit mask remain unconstrained.
(assert (not selected-poison))
(assert (distinct (bvand result selected-defined)
                  (bvand selected selected-defined)))
(check-sat)

; RV64: arbitrary values, poison flags, and masks of defined bits.
(reset)
(set-logic QF_BV)
(define-sort Word () (_ BitVec 64))
(declare-const c Word)
(declare-const t Word)
(declare-const f Word)
(declare-const t-defined Word)
(declare-const f-defined Word)
(declare-const t-poison Bool)
(declare-const f-poison Bool)
(declare-const frozen-t Word)
(declare-const frozen-f Word)

; A defined SELECT condition obeys ZeroOrOneBooleanContent. A poison
; condition imposes no result constraint; undef may choose either boolean.
(assert (or (= c (_ bv0 64)) (= c (_ bv1 64))))
; Freeze preserves all defined bits of nonpoison inputs. Its arbitrary
; choices for other bits are shared across all uses of that freeze.
(assert (=> (not t-poison)
            (= (bvand frozen-t t-defined) (bvand t t-defined))))
(assert (=> (not f-poison)
            (= (bvand frozen-f f-defined) (bvand f f-defined))))

(define-fun normalized-c () Word
  (ite (= c (_ bv0 64)) (_ bv0 64) (_ bv1 64)))
(define-fun mask () Word (bvneg normalized-c))
(define-fun result () Word
  (bvxor frozen-f (bvand (bvxor frozen-t frozen-f) mask)))
(define-fun selected () Word (ite (= c (_ bv1 64)) t f))
(define-fun selected-defined () Word
  (ite (= c (_ bv1 64)) t-defined f-defined))
(define-fun selected-poison () Bool
  (ite (= c (_ bv1 64)) t-poison f-poison))

; A counterexample would change a defined bit of a nonpoison chosen arm.
; The unchosen arm's poison flag and defined-bit mask remain unconstrained.
(assert (not selected-poison))
(assert (distinct (bvand result selected-defined)
                  (bvand selected selected-defined)))
(check-sat)
