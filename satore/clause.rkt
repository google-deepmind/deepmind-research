#lang racket/base

;***************************************************************************************;
;****                             Operations on clauses                             ****;
;***************************************************************************************;

(require bazaar/cond-else
         bazaar/debug
         bazaar/list
         bazaar/loop
         bazaar/mutation
         (except-in bazaar/order atom<=>)
         define2
         global
         racket/file
         racket/format
         racket/list
         racket/pretty
         satore/misc
         satore/trie
         satore/unification
         syntax/parse/define
         text-table)

(provide (all-defined-out))

(define-global *subsumes-iter-limit* 0
  '("Number of iterations in the θ-subsumption loop before failing."
    "May help in cases where subsumption take far too long."
    "0 = no limit.")
  exact-nonnegative-integer?
  string->number)

(define-counter n-tautologies 0)

(define (sort-clause cl)
  (sort cl literal<?))

;; cl is assumed to be already Varified, but possibly not freshed.
;; Notice: Does not do rewriting.
(define (clause-normalize cl)
   ; fresh the variables just to make sure
  (fresh (safe-factoring (sort-clause cl))))

;; Used to turn human-readable clauses into computer-friendly clauses.
(define (clausify cl)
  (clause-normalize (Varify cl)))

(define (empty-clause? cl)
  (empty? cl))

;; Definition of tautology:
;; cl is a tautology if all ground instances of cl contain an atom and its negation.
;; Assumes that the clause cl is sorted according to `sort-clause`.
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

;; NOTICE: This does *not* rename the variables.
(define (clause-converse cl)
  (sort-clause (map lnot cl)))

;; Assumes that cl is sorted according to `sort-clause`.
(define (remove-duplicate-literals cl)
  (zip-loop ([(x r) cl] [res '()] #:result (reverse res))
    (cond/else
     [(empty? r) (cons x res)]
     #:else
     (define y (first r))
     #:cond
     [(literal==? x y) res]
     #:else (cons x res))))

(define (predicate.arity lit)
  (cond [(list? lit) (cons (first lit) (length lit))]
        [else (cons lit 0)]))


(define-counter n-subsumes-checks 0)
(define-counter n-subsumes-steps 0)
(define-counter n-subsumes-breaks 0)
(define (reset-subsumes-stats!)
  (reset-n-subsumes-checks!)
  (reset-n-subsumes-steps!)
  (reset-n-subsumes-breaks!))


;; θ-subsumption.
;; ca θ-subsumes cb if there exists a substitution α such that ca[α] ⊆ cb
;; (requires removing duplicate literals as in FOL clauses are assumed to be sets of literals).
;; Assumes vars(ca) ∩ vars(cb) = ∅.

(define (clause-subsumes ca cb)
  (++n-subsumes-checks)
  ; For every each la of ca  with current substitution β, we need to find a literal lb of cb
  ; such that we can extend β to β' so that la[β'] = lb.
  ; TODO: order the groups by smallest size for cb.
  ; TODO: need to split by polarity first, or sort by (polarity predicate arity)
  ; For each literal of ca, obtain the list of literals of cb that unify with it.
  ; place cb in a trie
  ; then retrieve

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
  ;; TODO: if number of iterations is larger than threshold, abort (use let/ec)
  (define n-iter-max (*subsumes-iter-limit*))
  (define n-iter 0)

  (let/ec return
    (let loop ([groups groups] [subst '()])
      (++ n-iter)
      (when (= n-iter n-iter-max) ; if n-iter-max = 0 then no limit
        (++n-subsumes-breaks)
        (return #false))
      (++n-subsumes-steps)
      (cond
        [(empty? groups) subst]
        [else
         (define-values (lita litbs) (car+cdr (first groups)))
         (for/or ([litb (in-list litbs)])
           ; We use a immutable substitution to let racket handle copies when needed.
           (define new-subst (left-unify/assoc lita litb subst))
           (and new-subst (loop (rest groups) new-subst)))]))))

;; Assumes that the clause cl is sorted according to `sort-clause`.
;; A safe factor f of cl is such that f[α] ⊆ cl for some subst α, that is,
;; f θ-subsumes cl. But since cl necessarily θ-subsumes all of its factors XXX
;; - The return value is eq? to the argument cl if no safe-factoring is possible.
;; - Applies safe-factoring as much as possible.
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

(define (clause-equivalence? A B)
  (and (clause-subsumes A B)
       (clause-subsumes B A)))

;==============;
;=== Saving ===;
;==============;

(define (save-clauses! cls f #:? [exists 'replace])
  (with-output-to-file f #:exists exists
    (λ () (for-each writeln cls))))

(define (load-clauses f)
  (map clausify (file->list f)))
