#lang racket/base

;***************************************************************************************;
;****                             Operations on clauses                             ****;
;***************************************************************************************;

(require bazaar/cond-else
         bazaar/list
         bazaar/loop
         bazaar/mutation
         (except-in bazaar/order atom<=>)
         define2
         global
         racket/file
         racket/list
         satore/misc
         satore/trie
         satore/unification
         syntax/parse/define)

(provide (all-defined-out))

(define-global *subsumes-iter-limit* 0
  '("Number of iterations in the θ-subsumption loop before failing."
    "May help in cases where subsumption take far too long."
    "0 = no limit.")
  exact-nonnegative-integer?
  string->number)

(define-counter n-tautologies 0)

;; Returns a new clause where the literals have been sorted according to `literal<?`.
;;
;; (listof literal?) -> (listof literal?)
(define (sort-clause cl)
  (sort cl literal<?))

;; 'Normalizes' a clause by sorting the literals, safely factoring it (removes duplicate literals),
;; and 'freshing' the variables.
;; cl is assumed to be already Varified, but possibly not freshed.
;;
;; (listof literal?) -> (listof literal?)
(define (clause-normalize cl)
   ; fresh the variables just to make sure
  (fresh (safe-factoring (sort-clause cl))))

;; Takes a tree of symbols and returns a clause, after turning symbol variables into `Var`s.
;; Used to turn human-readable clauses into computer-friendly clauses.
;;
;; tree? -> clause?
(define (clausify l)
  (clause-normalize (Varify l)))

;; clause? -> boolean?
(define (empty-clause? cl)
  (empty? cl))

;; Returns whether the clause `cl` is a tautologie.
;; cl is a tautology if it contains the literals `l` and `(not l)`.
;; Assumes that the clause cl is sorted according to `sort-clause`.
;;
;; clause? -> boolean?
(define (clause-tautology? cl)
  (define-values (neg pos) (partition lnot? cl))
  (define pneg (map lnot neg))
  (and
   (or
    (memq ltrue pos)
    (memq lfalse pneg)
    (let loop ([pos pos] [pneg pneg])
      (cond/else
       [(or (empty? pos) (empty? pneg)) #false]
       #:else
       (define p (first pos))
       (define n (first pneg))
       (define c (literal<=> p n))
       #:cond
       [(order<? c) (loop (rest pos) pneg)]
       [(order>? c) (loop pos (rest pneg))]
       [(literal==? p n)]
       #:else (error "uh?"))))
   (begin (++n-tautologies) #true)))

;; Returns the converse clause of `cl`.
;; Notice: This does *not* rename the variables.
;;
;; clause? -> clause?
(define (clause-converse cl)
  (sort-clause (map lnot cl)))

;; Returns the pair of (predicate-symbol . arity) of the literal.
;;
;; literal? -> (cons/c symbol? exact-nonnegative-integer?)
(define (predicate.arity lit)
  (let ([lit (depolarize lit)])
    (cond [(list? lit) (cons (first lit) (length lit))]
          [else (cons lit 0)])))

;; Several counters to keep track of statistics.
(define-counter n-subsumes-checks 0)
(define-counter n-subsumes-steps 0)
(define-counter n-subsumes-breaks 0)
(define (reset-subsumes-stats!)
  (reset-n-subsumes-checks!)
  (reset-n-subsumes-steps!)
  (reset-n-subsumes-breaks!))


;; θ-subsumption. Returns a (unreduced) most-general unifier θ such that caθ ⊆ cb, in the sense
;; of set inclusion.
;; Assumes vars(ca) ∩ vars(cb) = ∅.
;; Note that this function does not check for multiset inclusion. A length check is performed in
;; Clause-subsumes?.
;;
;; clause? clause? -> subst?
(define (clause-subsumes ca cb)
  (++n-subsumes-checks)
  ; For every each la of ca  with current substitution β, we need to find a literal lb of cb
  ; such that we can extend β to β' so that la β' = lb.

  (define cbtrie (make-trie #:variable? Var?))
  (for ([litb (in-list cb)])
    ; the key must be a list, but a literal may be just a constant, so we need to `list` it.
    (trie-insert! cbtrie (list litb) litb))

  ;; Each literal lita of ca is paired with a list of potential literals in cb that lita matches,
  ;; for subsequent left-unification.
  ;; We sort the groups by smallest size first, to fail fast.
  (define groups
    (sort
     (for/list ([lita (in-list ca)])
       ; lita must match litb, hence inverse-ref
       (cons lita (append* (trie-inverse-ref cbtrie (list lita)))))
     < #:key length #:cache-keys? #true))

  ;; Depth-first search while trying to find a substitution that works for all literals of ca.
  (define n-iter-max (*subsumes-iter-limit*))
  (define n-iter 0)

  (let/ec return
    (let loop ([groups groups] [subst '()])
      (++ n-iter)
      ; Abort when we have reached the step limit
      (when (= n-iter n-iter-max) ; if n-iter-max = 0 then no limit
        (++n-subsumes-breaks)
        (return #false))
      (++n-subsumes-steps)
      (cond
        [(empty? groups) subst]
        [else
         (define gp (first groups))
         (define lita (car gp))
         (define litbs (cdr gp))
         (for/or ([litb (in-list litbs)])
           ; We use a immutable substitution to let racket handle copies when needed.
           (define new-subst (left-unify/assoc lita litb subst))
           (and new-subst (loop (rest groups) new-subst)))]))))

;; Returns the shortest clause `cl2` such that `cl2` subsumes `cl`.
;; Since `cl` subsumes each of its factors (safe or unsafe, and in the sense of
;; non-multiset subsumption above), this means that `cl2` is equivalent to `cl`
;; (hence no information is lost in `cl2`, it's a 'safe' factor).
;; Assumes that the clause cl is sorted according to `sort-clause`.
;; - The return value is eq? to the argument cl if no safe-factoring is possible.
;; - Applies safe-factoring as much as possible.
;;
;; clause? -> clause?
(define (safe-factoring cl)
  (let/ec return
    (zip-loop ([(l x r) cl])
      (define pax (predicate.arity x))
      (zip-loop ([(l2 y r2) r] #:break (not (equal? pax (predicate.arity y))))
        ; To avoid code duplication:
        (define-simple-macro (attempt a b)
          (begin
            (define s (left-unify a b))
            (when s
              (define new-cl
                (sort-clause
                 (fresh ; required for clause-subsumes below
                  (left-substitute (rev-append l (rev-append l2 (cons a r2))) ; remove b
                                   s))))
              (when (clause-subsumes new-cl cl)
                ; Try one more time with new-cl.
                (return (safe-factoring new-cl))))))

        (attempt x y)
        (attempt y x)))
    cl))

;; Returns whether the two clauses subsume each other,
;; in the sense of (non-multiset) subsumption above.
;;
;; clause? clause? -> boolean?
(define (clause-equivalence? cl1 cl2)
  (and (clause-subsumes cl1 cl2)
       (clause-subsumes cl2 cl1)))

;=================;
;=== Save/load ===;
;=================;

;; Save the clauses `cls` to the file `f`.
;;
;; cls : (listof clause?)
;; f : file?
;; exists : symbol? ; See `with-output-to-file`.
(define (save-clauses! cls f #:? [exists 'replace])
  (with-output-to-file f #:exists exists
    (λ () (for-each writeln cls))))

;; Returns the list of clauses loaded from the file `f`.
;;
;; file? -> (listof clause?)
(define (load-clauses f)
  (map clausify (file->list f)))
