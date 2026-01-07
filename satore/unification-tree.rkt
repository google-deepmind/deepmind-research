#lang racket/base

;**************************************************************************************;
;****                               Unification Tree                               ****;
;**************************************************************************************;

;;; An imperfect discrimination tree specialized for unifying literals and other
;;; operations. This is *different* from "substitution trees"
;;; (https://link.springer.com/chapter/10.1007%2F3-540-59200-8_52)
;;; Note: This should be probably named a Clause-trie instead, since the
;;;   major difference with the trie is that we are dealing with Clauses, which
;;;   are lists of literals, and the same Clause can appear in different leaves
;;;   of the trie. Unification is only one of the operations performed on Clauses.

(require bazaar/list
         bazaar/loop
         bazaar/mutation
         define2
         global
         racket/list
         satore/clause-struct
         satore/clause
         satore/misc
         satore/trie
         satore/unification)

(provide (all-defined-out)
         (all-from-out satore/trie))

(module+ test
  (require rackunit))

;; Experimental.
;; Based on "A Unifying Principle for Clause Elimination in First-Order Logic",
;; by Kiesl, Benjamin and Suda, Martin, CADE 26, 2017.
;;
;; Notice: This cannot be applied to input clauses.
;; Notice: To pass Russell's problem, we must
;; do 1-to-N resolution (non-binary resolution), OR, maybe,
;; binary resolution + unsafe factoring, but the 'resolutions'
;; for factoring must be taken into account too.
(define-global:boolean *L-resolvent-pruning?* #false
  '("Discard clauses for which a literal leads to 0 resolvents."
    "Currently doesn't apply to input clauses."))

(define-counter n-L-resolvent-pruning 0)

;========================;
;=== Unification Tree ===;
;========================;

;; A unification tree is simply a discrimination tree where trie-node-values
;; are lists of `utree-leaf`.
(struct unification-tree trie () #:transparent)

;; A leaf value.
;; Several leaves may have the same clause-idx but different clauses——well, the same clauses
;; but ordered differently. It's named `uclause` to make it clear it's not a well-formed clause
;; (stands for unordered-clause).
(struct utree-leaf (Clause uclause) #:transparent)

;; Returns a new unification tree (or a substruct).
;;
;; constructor : procedure?
;; other-args : list?
;; -> unification-tree?
(define (make-unification-tree #:constructor [constructor unification-tree]
                               . other-args)
  (apply make-trie
         #:constructor constructor
         #:variable? Var?
         other-args))

;; Inserts a Clause into the unification tree.
;; Each literal of the clause cl is added to the tree, and the leaf value at each literal lite is the
;; clause, but where the first literal is lit.
;; Notice: Thus the clause is *not* sorted according to `sort-clause`.
;; Note: We could also keep the clause unchanged and cons the index of the literal,
;; that would avoid using up new cons cells, while keeping the clause intact.
;;
;; unification-tree? Clause? -> void?
(define (add-Clause! utree C)
  (define cl (Clause-clause C))
  (zip-loop ([(left lit right) cl])
    (define reordered-clause (cons lit (rev-append left right)))
    (trie-insert! utree lit (utree-leaf C reordered-clause))))

;; Returns the list of unique Clauses present in the utree.
;;
;; unification-tree? -> (listof Clause?)
(define (unification-tree-Clauses utree)
  (remove-duplicates (map utree-leaf-Clause (append* (trie-values utree))) eq?))

;; Calls `on-unified` for each literal of each clause of utree that unifies with `lit`.
;; If a clause `cl` has n literals that unify with `lit`, then `on-unified` is called n times.
;;
;; utree : unification-tree?
;; lit : literal?
;; on-unified : utree-leaf? subst lit1 lit2 other-lit2s -> void?
;; -> void?
(define (find-unifiers utree lit on-unified)
  (trie-both-find utree lit
                  (λ (nd)
                    (define val (trie-node-value nd))
                    (when (list? val)
                      (for ([lf (in-list val)])
                        (define cl (utree-leaf-uclause lf))
                        ; Unify only with the first literal, assuming clauses in node-values
                        ; are so that the first literal corresponds to the key
                        ; (the path from the root)
                        (define lit2 (first cl))
                        (define subst (unify lit2 lit))
                        (when subst
                          (on-unified lf subst lit lit2 (rest cl))))))))

;; Returns the set of Clauses that *may* left-unify with lit.
;; The returned clauses are sorted according to `sort-clause` and duplicate clauses are removed.
;;
;; unification-tree? literal? -> (listof Clause?)
(define (unification-tree-ref utree lit)
  ; Node values are lists of rules, and trie-ref returns a list of node-values,
  ; hence the append*.
  (remove-duplicates (append* (map utree-leaf-Clause (trie-ref utree lit))) eq?))

;; Helper for the resolve/factors functions below.
;; Defines a new set of Clauses, and a helper function that creates new Clauses,
;; rewrites them, checks for tautologies and add them to the new-Clauses.
(define-syntax-rule (define-add-Clause! C new-Clauses add-Clause! rewriter)
  (begin
    (define new-Clauses '())
    (define (add-Clause! lits subst type parents)
      (define cl (clause-normalize (substitute lits subst)))
      (define new-C (make-Clause cl (cons C parents) #:type type))
      ; Rewrite
      (let ([new-C (rewriter new-C)])
        (unless (Clause-tautology? new-C)
          (cons! new-C new-Clauses))))))

;; Resolves the Clause `C` with all the Clauses in `utree` and returns the list of resolvents.
;; Uses negative literal selection based on `literal-cost`:
;; If `C` has at least one negative literal, the costliest one is selected using `literal-cost`,
;; and only this literal is used for resolution with other Clauses.
;; The resolvents are immediately rewritten with `rewriter`.
;;
;; utree : unification-tree?
;; C : Clause?
;; rewriter : Clause? -> Clause?
;; literal-cost : literal? -> number?
;; -> (listof Clause?)
(define (utree-resolve/select-literal utree C
                                      #:? [rewriter (λ (C) C)]
                                      #:? [literal-cost literal-size])

  (define cl (Clause-clause C))
  ;; Choose the costliest negative literal if any (for elimination)
  (define selected-idx
    (for/fold ([best-idx #false]
               [best-cost -inf.0]
               #:result best-idx)
              ([lit (in-list cl)]
               [idx (in-naturals)]
               #:when (lnot? lit)) ; negative literals only
      (define c (literal-cost lit))
      (if (> c best-cost)
          (values idx c)
          (values best-idx best-cost))))

  (zip-loop ([(left lit right) cl]
             [resolvents '()]
             [lit-idx 0]
             #:result (or resolvents '()))
    (cond
      [(or (not selected-idx)
           (= lit-idx selected-idx))

       (define-add-Clause! C new-Clauses add-Clause! rewriter)

       ; Find resolvents
       (find-unifiers utree
                      (lnot lit)
                      (λ (lf subst nlit lit2 rcl2)
                        (add-Clause! (rev-append left (rev-append right rcl2))
                                     subst
                                     'res
                                     (list (utree-leaf-Clause lf)))))
       (values (rev-append new-Clauses resolvents)
               (+ 1 lit-idx))]
      [else
       (values resolvents
               (+ 1 lit-idx))])))

;; Returns the list of unsafe factors of `C`.
;; Factors are immediately rewritten with `rewriter`.
;;
;; C : Clause?
;; rewriter : Clause? -> Clause?
;; -> (listof Clause?)
(define (unsafe-factors C #:? [rewriter (λ (C) C)])
  (define-add-Clause! C factors add-Clause! rewriter)
  (define cl (Clause-clause C))

  (zip-loop ([(left lit1 right) cl])
    (define pax (predicate.arity lit1))
    (zip-loop ([(left2 lit2 right2) right]
               ; Literals are sorted, so no need to go further.
               #:break (not (equal? pax (predicate.arity lit2))))
      (define subst (unify lit1 lit2))
      ; We could do left-unify instead, but then we need to do both sides,
      ; at the risk of generating twice as many clauses, so may not be worth it.
      (when subst
        (add-Clause! (rev-append left right) ; remove lit1
                     subst
                     'fac
                     '()))))
  factors)

;; Appends and returns the results of `unsafe-factors` and `utree-resolve/select-literal`.
;;
;; utree : unification-tree?
;; C : Clause?
;; rewriter : Clause? -> Clause?
;; literal-cost : literal? -> number?
;; -> (listof Clause?)
(define (utree-resolve+unsafe-factors/select utree C #:? rewriter #:? literal-cost)
  (rev-append
   (unsafe-factors C #:rewriter rewriter)
   (utree-resolve/select-literal utree C
                                 #:rewriter rewriter
                                 #:literal-cost literal-cost)))

;; Like `utree-resolve+unsafe-factors/select` but without negative literal selection,
;; and with L-resolvent pruning.
;; Returns the set of Clauses from resolutions between cl and the clauses in utree,
;; as well as the factors.
;; Clauses are immediately rewritten with `rewriter`.
;;
;; utree : unification-tree?
;; C : Clause?
;; rewriter : Clause? -> Clause?
;; L-resolvent-pruning? : boolean?
;; -> (listof Clause?)
(define (utree-resolve+unsafe-factors utree C
                                      #:? [rewriter (λ (C) C)]
                                      #:! L-resolvent-pruning?)
  ;; Used to prevent pruning by L-resolvent-discard.
  ;; This is used to mark the second literals in unsafe factors.
  (define lit-marks (make-vector (Clause-n-literals C) #false))
  (define (mark-literal! idx) (vector-set! lit-marks idx #true))
  (define (literal-marked? idx) (vector-ref lit-marks idx))


  (zip-loop ([(left lit right) (Clause-clause C)]
             [resolvents+factors '()]
             [lit-idx 0]
             #:break (not resolvents+factors) ; shortcut
             #:result (or resolvents+factors '()))

    (define-add-Clause! C new-Clauses add-Clause! rewriter)

    ;; Resolutions
    (find-unifiers utree
                   (lnot lit)
                   (λ (lf subst nlit lit2 rcl2)
                     (add-Clause! (rev-append left (rev-append right rcl2))
                                  subst
                                  'res
                                  (list (utree-leaf-Clause lf)))))
    ;; Unsafe binary factors
    ;; Somewhat efficient implementation since the literals are sorted by predicate.arity.
    (define pax (predicate.arity lit))
    (zip-loop ([(left2 lit2 right2) right]
               [lit2-idx (+ 1 lit-idx)]
               #:break (not (equal? pax (predicate.arity lit2))))
      (define subst (unify lit lit2))
      (when subst
        (mark-literal! lit2-idx) ; prevents pruning
        (add-Clause! (rev-append left right) ; remove lit
                     subst
                     'fac
                     '()))
      (+ 1 lit2-idx))

    ;; L-resolvent 'pruning'
    ;; See the principle of implication modulo resolution:
    ;;   "A unifying principle for clause elimination in first-order logic", CADE 26.
    ;;   which contains other techniques and short proofs of their soundness.
    ;; We return the empty set of resolution, meaning that the selected clause
    ;; can (must) be discarded, i.e., not added to the active set.
    (cond [(and L-resolvent-pruning?
                (empty? new-Clauses)
                (not (literal-marked? lit-idx)))
           (++n-L-resolvent-pruning)
           (values #false (+ 1 lit-idx))]
          [else
           (values (rev-append new-Clauses resolvents+factors)
                   (+ 1 lit-idx))])))

;; Returns the first (in any order) Clause C2 such that
;; there is a literal of C2 that imperfectly matches a literal of C,
;; and (pred C C2).
;;
;; utree : unification-tree?
;; C2 : Clause?
;; pred : Clause? Clause? -> boolean?
(define (utree-find/any utree C2 pred)
  (define tested (make-hasheq)) ; don't test the same C2 twice
  (define cl2 (Clause-clause C2))
  (let/ec return
    (for ([lit (in-list cl2)])
      (trie-find utree lit
                 (λ (nd)
                   (define val (trie-node-value nd))
                   (when (list? val)
                     (for ([lf (in-list val)])
                       (define C (utree-leaf-Clause lf))
                       (hash-ref! tested
                                  C
                                  (λ ()
                                    (when (pred C C2)
                                      (return C))
                                    #true)))))))
    #false))

;; Return all Clauses C that left-subunify on at least one literal and for which (pred C C2).
;;
;; utree : unification-tree?
;; C2 : Clause?
;; pred : Clause? Clause? -> boolean?
(define (utree-find/all utree C2 pred)
  (define tested (make-hasheq)) ; don't test the same C2 twice
  (define cl2 (Clause-clause C2))
  (define res '())
  (for ([lit (in-list cl2)])
    (trie-find utree lit
               (λ (nd)
                 (define val (trie-node-value nd))
                 (when (list? val)
                   (for ([lf (in-list val)])
                     (define C (utree-leaf-Clause lf))
                     (hash-ref! tested
                                C
                                (λ ()
                                  (when (pred C C2)
                                    (set! res (cons C res)))
                                  #true)))))))
  res)

;; Removes the Clause C from the utree.
;;
;; utree : unification-tree?
;; C : Clause?
(define (utree-remove-Clause! utree C)
  (define cl (Clause-clause C))
  (for ([lit (in-list cl)])
    (trie-find utree lit
               (λ (nd)
                 (define val (trie-node-value nd))
                 (when (list? val)
                   (set-trie-node-value! nd
                     (filter-not (λ (lf2) (eq? C (utree-leaf-Clause lf2)))
                                 val)))))))

;; Finds the leaves for which C loosely left-unifies on some literal and remove those which clause C2
;; where (pred C C2).
;; Returns the set of Clauses that have been removed.
;;
;; utree : unification-tree?
;; C2 : Clause?
;; pred: Clause? Clause? -> boolean
(define (utree-inverse-find/remove! utree C pred)
  ; Since the same Clause may match multiple times,
  ; We use a hash to remember which clauses have already been tested (and if the result
  ; was #true or #false).
  ; Then remove all the leaves of each clause to remove.
  (define tested (make-hasheq))
  (define Clauses-to-remove '())
  (define cl (Clause-clause C))
  (for ([lit (in-list cl)])
    (trie-inverse-find utree lit
                       (λ (nd)
                         (define val (trie-node-value nd))
                         (when (list? val)
                           (for ([lf (in-list (trie-node-value nd))])
                             (define C2 (utree-leaf-Clause lf))
                             (hash-ref! tested
                                        C2
                                        (λ ()
                                          (cond [(pred C C2)
                                                 (cons! C2 Clauses-to-remove)
                                                 #true]
                                                [else #false]))))))))
  (for ([C2 (in-list Clauses-to-remove)])
    (utree-remove-Clause! utree C2))
  Clauses-to-remove)
