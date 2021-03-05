#lang racket/base

(require (submod bazaar/order test)
         racket/list
         racket/match
         rackunit
         satore/clause
         satore/unification)

(define (subst/#false->imsubst s)
  (cond [(subst? s)
         (subst->list s)]
        [(list? s)
         (sort (map (λ (p) (cons (Var-name (Varify (car p)))
                                 (Varify (cdr p))))
                    s)
               Var-name<? #:key car)]
        [else s]))


(check-eq? (Var-name->symbol (symbol->Var-name 'C)) 'C)
(check-eq? (Var-name->symbol (symbol->Var-name 'X0)) 'X0)
(check-eq? (Var-name->symbol (symbol->Var-name 'X1353)) 'X1353)


(check-equal?
 (find-var-names (Varify '(p D C A B)))
 (map Var-name (Varify '(D C A B))))
(check-equal?
 (find-var-names (Varify '(p (q D E) C A B E D D A)))
 (map Var-name (Varify '(D E C A B))))

(let ()
  (define-check (check/fail-var-occs<=> p q c)
    (let ([res (var-occs<=> p q)])
      (unless (eq? res c)
        (fail-check (format "Params: ~a ~a \nExpected: ~a\nGot: ~a\n" p q c res)))))
  (define (check-var-occs<=> p q c)
    (let ([p (Varify p)] [q (Varify q)])
      (check/fail-var-occs<=> p q c)
      (case c
        [(<) (check/fail-var-occs<=> q p '>)]
        [(>) (check/fail-var-occs<=> q p '<)]
        [(= #false) (check/fail-var-occs<=> q p c)])))

  (check-var-occs<=> '(p) '(q) '=)
  (check-var-occs<=> '(p X) '(q) '>)
  (check-var-occs<=> '(p X) '(q X) '=)
  (check-var-occs<=> '(p X X) '(q X) '>)
  (check-var-occs<=> '(p X Y) '(q X) '>)
  (check-var-occs<=> '(p X Y) '(q X Z) #false)
  (check-var-occs<=> '(p X X Y) '(q X Z) #false)
  (check-var-occs<=> '(p X X Y) '(q X Y) '>)
  (check-var-occs<=> '(p X X Y) '(q X Y Y) #false))

(let ()
  (check equal? (lnot 'auie) `(not auie))
  (check equal? (lnot (lnot 'auie)) 'auie)
  (check equal? (lnot lfalse) ltrue)
  (check equal? (lnot `(not ,lfalse)) lfalse) ; to fix non-reduced values
  (check equal? (lnot `(not ,ltrue)) ltrue) ; to fix non-reduced values
  (check equal? (lnot ltrue) lfalse)
  (check equal? (lnot (lnot ltrue)) ltrue)
  (check equal? (lnot (lnot lfalse)) lfalse)
  (check<=> polarity<=> 'a '(not a) '<))


(let ()
  (define-simple-check (check-atom1<=> a b res)
    (check<=> atom1<=> (Varify a) (Varify b) res))

  (check-atom1<=> lfalse 'a '<)
  (check-atom1<=> lfalse lfalse '=)
  (check-atom1<=> '() '() '=)
  (check-atom1<=> '(eq b a) '(eq a b) '>) ; lexicographical order
  (check-atom1<=> '(p X Y) '(p Y X) '=) ; no lex order between variables
  (check-atom1<=> '(p a X) '(p X a) '=) ; no lex order between variable and symbol
  (check-atom1<=> '(p A (q B))
                 '(p (q A) B)
                 '=) ; no lex order when variable is involved
  (check-atom1<=> '(p A (q b))
                 '(p (q A) b)
                 '=) ; ????
  (check-atom1<=> '(not (eq X0 X1 X1)) ; var-occs=
                 '(not (eq X0 X1 X1))
                 '=)

  (check-atom1<=> '(eq X0 X1 X1)
                 '(not (eq X0 X1 X1))
                 '=) ; negation should NOT count
  ;;; This is very important, otherwise the following problem can end with 'saturated:
  ;;; ((notp A B) (p A B)) ; axiom, binary clause
  ;;; ((not (notp A B)) (not (p A B))) ; axiom, converse binary clause
  ;;; ((p a A)) ; these two clauses should resolve to '() immediately
  ;;; ((not (p A a))) ; Note that 'a A' is to prevent unit-clause rewrites


  (check-atom1<=> 'p 'q '<)
  (check-atom1<=> '(not p) '(not q) '<)

  (check-atom1<=> '(not (eq X0 X1 X1))
                 '(not (eq X1 X0))
                 '>)
  (check-atom1<=> '(p X Y Z)
                 '(p X Y one)
                 '>)
  (check-atom1<=> '(p X A one)
                 '(p X Y one)
                 '#false)
  (check-atom1<=> '(p X one one)
                 '(p X one)
                 '>)
  (check-atom1<=> '(p X one (q one))
                 '(p X one one)
                 '>)


  )


;; Tests for KBO
(let ()
  (define-simple-check (check-KBO1lex<=> a b res)
    (check<=> KBO1lex<=> (Varify a) (Varify b) res))


  (check-KBO1lex<=> lfalse 'a '<)
  (check-KBO1lex<=> lfalse lfalse '=)


  ;(check-KBO1lex<=> '() '() '=) ; not a term
  (check-KBO1lex<=> '(eq b a) '(eq a b) '>) ; lexicographical order
  (check-KBO1lex<=> '(p X Y) '(p Y X) #false) ; commutativity cannot be oriented
  (check-KBO1lex<=> '(p a X) '(p X a) #false) ; left->right order: a <=> X -> #false
  (check-KBO1lex<=> '(p A (q B))
                    '(p (q A) B)
                    '<) ; left->right order: A < (q A)
  (check-KBO1lex<=> '(p A (q b))
                    '(p (q A) b)
                    '<) ; left->right order: A < (q A)
  (check-KBO1lex<=> '(not (eq X0 X1 X1)) ; var-occs=
                    '(not (eq X0 X1 X1))
                    '=)

  (check-KBO1lex<=> '(eq X0 X1 X1)
                    '(not (eq X0 X1 X1))
                    '=) ; negation should NOT count
  ;;; This is very important, otherwise the following problem can end with 'saturated:
  ;;; ((notp A B) (p A B)) ; axiom, binary clause
  ;;; ((not (notp A B)) (not (p A B))) ; axiom, converse binary clause
  ;;; ((p a A)) ; these two clauses should resolve to '() immediately
  ;;; ((not (p A a))) ; Note that 'a A' is to prevent unit-clause rewrites


  (check-KBO1lex<=> 'p 'q '<) ; lex
  (check-KBO1lex<=> '(not p) '(not q) '<) ; lex

  (check-KBO1lex<=> '(not (eq X0 X1 X1))
                    '(not (eq X1 X0))
                    '>) ; by var-occs and weight
  (check-KBO1lex<=> '(p X Y Z)
                    '(p X Y one)
                    #false) ; var-occs incomparable
  (check-KBO1lex<=> '(p X Y (f Z))
                    '(p X (f Y) one)
                    #false) ; var-occs incomparable
  (check-KBO1lex<=> '(p X Y (f Z))
                    '(p X (f Y) Z)
                    '<) ; same weight, Y < (f Y)
  (check-KBO1lex<=> '(p X A one)
                    '(p X Y one)
                    #false)
  (check-KBO1lex<=> '(p X one one)
                    '(p X one)
                    '>)
  (check-KBO1lex<=> '(p X one (q one))
                    '(p X one one)
                    '>)

  (check-KBO1lex<=> '(p A (p B C))
                    '(p (p A B) C)
                    '<) ; associativity ok: A < (p A B)
  )

(let ()
  (check-equal? (term-lex2<=> '(p a) '(p b)) '<))

(let ()
  (define-simple-check (check-term-lex<=> a b res)
    (let-values ([(a b) (apply values (fresh (Varify (list a b))))])
      (check<=> term-lex<=> (Varify a) (Varify b) res)))
  (check-term-lex<=> 'a (Var 'X) '>)
  (check-term-lex<=> (Var 'X) (Var 'X) '=)
  (check-term-lex<=> 'a 'a '=)
  (check-term-lex<=> 'a 'b '<)

  (define-simple-check (check-literal<=> a b res)
    (let-values ([(a b) (apply values (fresh (Varify (list a b))))])
      (check<=> literal<=> (Varify a) (Varify b) res)))
  (check-literal<=> 'a '(not a) '<)
  (check-literal<=> 'a 'b '<)
  (check-literal<=> 'z '(not a) '<)
  (check-literal<=> '(z b) '(not a) '<)
  (check-literal<=> 'a 'a '=)
  (check-literal<=> 'z '(a a) '>)
  (check-literal<=> '(z z) '(z (a a)) '<))

(let ()
  (check-true (literal==? 'a 'a))
  (check-true (literal==? (Var 'X) (Var 'X)))
  (check-true (literal==? (Var 'X) #s(Var X))) ; prefab
  (check-false (literal==? (Var 'X) (Var 'Y)))
  (check-false (literal==? (fresh (Var 'X)) (Var 'X)))
  (check-false (literal==? 'X (Var 'X))) ; not considered the same??
  (check-true (literal==? `(p (f ,(Var 'X) ,(Var 'X)) y) `(p (f ,(Var 'X) ,(Var 'X)) y))))


(let ()
  (define-check (test-unify t1 t2 subst)
    (let ([t1 (Varify t1)] [t2 (Varify t2)])
      (set! subst (subst/#false->imsubst subst))
      (define sh (unify t1 t2))
      (define sl (subst/#false->imsubst
                  (and sh
                       (for/list ([(k v) (in-subst sh)])
                         (cons (Var k)
                               (if (already-substed? v)
                                   (already-substed-term v)
                                   v))))))
      (unless (equal? sl subst)
        (fail-check (format "Expected ~a. Got: ~a\nt1 = ~a\nt2 = ~a\n"
                            subst sl
                            t1 t2)))
      (when sh
        (define r1 (substitute t1 sh))
        (define r2 (substitute t2 sh))
        (unless (equal? r1 r2)
          (fail-check "r1≠r2" sh r1 r2)))))

  (test-unify '(p X)
              '(p X)
              '())
  (test-unify '(p (f X) X)
              '(p (f a) a)
              '((X . a)))
  (test-unify '(p (f c) (g X))
              '(p Y Y)
              #false)
  (test-unify '(p X (f X))
              '(p a Y)
              '((X . a) (Y . (f a))))
  (test-unify '(p (f X Y) (f Y Z))
              '(p (f (f a) (f b)) (f (f b) c))
              '((X . (f a)) (Y . (f b)) (Z . c)))
  (test-unify '(p X (p X) a)
              '(p Y (p (p Z)) Z)
              (if reduce-mgu?
                  '((Z . a) (X . (p a)) (Y . (p a)))
                  '((X . Y) (Y . (p Z)) (Z . a))))
  (test-unify '(p X (p X) (p (p X)))
              '(p a Y Z)
              '((X . a) (Y . (p a)) (Z . (p (p a)))))
  (test-unify '(p X (p X) (p (p X)))
              '(p a (p Y) (p (p Z)))
              '((X . a) (Y . a) (Z . a)))
  (test-unify '(p (p X) (p X) a)
              '(p Y (p (p Z)) Z)
              (if reduce-mgu?
                  '((Z . a) (X . (p a)) (Y . (p (p a))))
                  '((Y . (p X)) (X . (p Z)) (Z . a))))
  (test-unify '(p X X)
              '(p a Y)
              '((X . a) (Y . a)))
  (test-unify '(p X X)
              '(p (f Y) Z)
              '((X . (f Y)) (Z . (f Y))))
  (test-unify '(p X X) '(p (f Y) Y) #false)
  (test-unify '(p (f X       Y)  (g Z Z))
              '(p (f (f W U) V)  W)
              (if reduce-mgu?
                  '((W . (g Z Z)) (Y . V) (X . (f (g Z Z) U)))
                  '((X . (f W U)) (Y . V) (W . (g Z Z)))))

  (test-unify '(eq X30 (mul X31 (mul X32 (inv (mul X31 X32)))))
              '(eq (mul X25 one) X26)
              `((X26 . (mul X31 (mul X32 (inv (mul X31 X32)))))
                (X30 . (mul X25 one))))
  (test-unify '(p A B)
              '(p B A)
              '((A . B)))
  )

(let ()
  (define (test-suite-left-unify left-unify)
    (define-simple-check (test-left-unify t1 t2 subst)
      (let ([t1 (Varify t1)] [t2 (Varify t2)])
        (set! subst (subst/#false->imsubst subst))
        (define sh (left-unify t1 t2))
        (define sl (subst/#false->imsubst sh))
        (check-equal? sl subst
                      (format "Expected ~a. Got: ~at1 = ~a\nt2 = ~a\n"
                              subst sl
                              t1 t2))
        (when sh
          (define r1 (left-substitute t1 sh))
          (check-equal? r1 t2 (format "r1≠t2\nsh=~a\nr1=~a\nt2=~a\n" sh r1 t2)))))

    (test-left-unify '(p (f X) X)
                     '(p (f a) a)
                     '((X . a)))
    (test-left-unify '(p (f c) (g X))
                     '(p Y Y)
                     #false)
    (test-left-unify '(p X (f X)) '(p a Y) #false)
    (test-left-unify '(p (f X Y) (f Y Z))
                     '(p (f (f a) (f b)) (f (f b) c))
                     '((Z . c) (Y . (f b)) (X . (f a))))
    (test-left-unify '(p X X) '(p a Y) #false)
    (test-left-unify '(p X X) '(p (f Y) Z) #false)
    (test-left-unify '(p X X) '(p (f Y) Y) #false)
    (test-left-unify '(p (f X       Y)  (g Z Z))
                     '(p (f (f W U) V)  W)
                     #false)
    (test-left-unify '(p X X)
                     '(p A B)
                     #false)
    ; This MUST return false because of the circularity.
    ; The found substitution must be specializing, that is C2σ = C2 (and C1σ = C2),
    ; otherwise safe factoring can fail, in particular.
    ; Hence we must ensure that vars(C2) ∩ dom(σ) = ø.
    (test-left-unify '(p A B)
                     '(p B A)
                     #false)
    (test-left-unify '(p B A)
                     '(p A B)
                     #false)
    (test-left-unify '(p A A)
                     '(p B B)
                     '((A . B)))
    (test-left-unify '(p A)
                     '(p A)
                     '())
    (test-left-unify '(p a)
                     '(p a)
                     '())
    (test-left-unify '(p A X)
                     '(p X Y)
                     #false))

  (test-suite-left-unify left-unify)
  (test-suite-left-unify (λ (t1 t2) (define subst-assoc (left-unify/assoc t1 t2))
                           (and subst-assoc (make-subst subst-assoc)))))


(let ([pat '(_not_ (_not_ #s(Var A)))]
      [t (fresh (Varify '(q (p (_not_ (_not_ (f A B)))))))])
  (define s
    (left-unify-anywhere pat t))
  (check-equal? (left-substitute pat s)
                (cadadr t)))

(let ([t '(q (p (_not_ (_not_ (f A B)))))])
  (check-equal?
   (match-anywhere (match-lambda [`(_not_ (_not_ ,x)) `([x . ,x])] [else #false])
                   t)
   '([x . (f A B)])))

(let ([c1 (clausify '((theorem (or A B)) (not (theorem (or A (or (or A B) B))))))]
      [c2 (clausify '((theorem (or (_not_ A) (or B A)))))])
  (define s1 (unify (first c2) (lnot (second c1))))
  (define s2 (unify (lnot (second c1)) (first c2)))
  (list c1
        c2
        s1
        (substitute (first c1) s1)
        s2
        (substitute (first c1) s2))
  (check clause-equivalence?
         (substitute (list (first c1)) s1)
         (substitute (list (first c1)) s2)))

;; Stress test for unification
;; This should take 0ms
;; See https://en.wikipedia.org/wiki/Unification_(computer_science)
;;       #Examples_of_syntactic_unification_of_first-order_terms
(let ()
  (define last-var? #true)
  (define (stress-unify n)
    (define A
      (let left ([d 1])
        (if (>= d n)
            (list '* (Var d) (if last-var? (Var (+ d 1)) 'a))
            (list '* (left (+ d 1)) (Var d)))))
    (define B
      (let right ([d 1])
        (if (>= d n)
            (list '* (if last-var? (Var (+ d 1)) 'a) (Var d))
            (list '* (Var d) (right (+ d 1))))))
    (define subst (time (unify A B)))
    ; Verify that there's only 1 variable in the each rhs
    (when (and reduce-mgu? last-var?)
      (check-equal? (length (Vars (map cdr (subst->list subst))))
                    1
                    subst))
    (time (substitute A subst)))
  (for ([n (in-range 30 50)])
    (printf "~a: " n)
    (stress-unify n)))
