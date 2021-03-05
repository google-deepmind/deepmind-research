#lang racket/base

;***************************************************************************************;
;****                              Binary Rewrite Tree                              ****;
;***************************************************************************************;

(require bazaar/cond-else
         bazaar/mutation
         (except-in bazaar/order atom<=>)
         define2
         define2/define-wrapper
         global
         racket/file
         racket/list
         racket/string
         satore/Clause
         satore/clause-format
         satore/clause
         satore/misc
         satore/trie
         satore/unification)

(provide (all-defined-out)
         (all-from-out satore/trie))

;===============;
;=== Globals ===;
;===============;

(define-counter n-rule-added 0)
(define-counter n-binary-rewrites 0)

(define-global:boolean *bounded-confluence?* #true
  '("When performing confluence, should the size of the critical pairs "
    "be bounded by the size of parent rules?"))

(define-global *confluence-max-steps* 10
  "Maximum number of confluence steps"
  number?
  string->number)

;====================;
;=== Rewrite tree ===;
;====================;

;; atom<=> : comparator?
;; dynamic-ok? : boolean? ; whether we keep dynamic rules
;; rules-file : file? ; if not #false, loads rules from the given file
(struct rewrite-tree trie (atom<=> dynamic?))

;; rewrite-tree constructor.
;;
;; constructor: procedure?
;; atom<=> : comparator?
;; dynamic-ok? : boolean?
;; rules-file : file?
;; other-args : list?
;; -> rewrite-tree?
(define (make-rewrite-tree #:? [constructor rewrite-tree]
                           #:! atom<=>
                           #:? [dynamic-ok? #true]
                           #:? [rules-file #false]
                           . other-args)
  (define rwtree
    (apply make-trie
           #:constructor constructor
           #:variable? Var?
           atom<=>
           dynamic-ok?
           other-args))
  (when rules-file
    (load-rules! rwtree #:rules-file rules-file #:rewrite? #true))
  rwtree)

;; Returns a new rewrite-tree.
;; Duplicates the structure of rwtree, but the Clauses and rules are shared.
;; This means that the Clause ancestors of the rules are preserved.
;;
;; rwtree : rewrite-tree?
;; -> rewrite-tree?
(define (rewrite-tree-shallow-copy rwtree)
  (define new-rwtree (make-rewrite-tree #:atom<=> (rewrite-tree-atom<=> rwtree)
                                        #:dynamic-ok? (rewrite-tree-dynamic? rwtree)))
  (for ([rl (in-list (rewrite-tree-rules rwtree))])
    (add-rule! new-rwtree rl))
  new-rwtree)

;; Returns the list of values that match the literal t.
;;
;; rwtree : rewrite-tree?
;; t : any/c
;; -> list?
(define (rewrite-tree-ref rwtree t)
  ; Node values are lists of rules, and trie-ref returns a list of node-values,
  ; hence the append*.
  (append* (trie-ref rwtree t)))

;; Clause : The Clause from which this rule originates
;;   NOTICE: Clause may subsume the rule, and can be *more* general if the rule is asymmetric.
;; from-literal : heavy-side
;; to-literal : light side
;; to-fresh : variables of to-literal not in from-literal that need to be freshed
;;   after applying the rule
;;
;; Clause : Clause?
;; rule-group : rule-group?
;; from-literal? : literal?
;; to-literal? : literal?
;; to-fresh : unused
;; static? : boolean?
(struct rule (Clause rule-group from-literal to-literal to-fresh static?)
  #:prefab)

;; A rule group holds all the rules that have been created by a single call to
;; Clause->rules.
;; A rule also has a reference to its rule group, so it makes it easy to find
;; the other rules of the same group from a given group.
;; A single group can have 2 or 4 rules: 2 for static rules or dynamic self-converse rules,
;; and 4 for dynamic non-self-converse rules.
;;
;; Clause : (or/c false/c Clause?)
;; Converse : Clause?
;; rules: (listof rule?)
(struct rule-group (Clause Converse [rules #:mutable]) #:prefab)

;; Rule constructor.
;;
;; C : Clause?
;; rule-gp : rule-group?
;; from : literal?
;; to : literal?
;; static? : boolean?
;; -> rule?
(define (make-rule C rule-gp from to static?)
  (define-values (lit1 lit2) (apply values (fresh (list from to))))
  (rule C rule-gp lit1 lit2 (variables-minus lit2 lit1) static?))

;; Returns whether rule is a unit rule, that is, if the `to` literal
;; is either ltrue or lfalse.
;;
;; rl : rule?
;; -> any/c
(define (unit-rule? rl)
  (memq (rule-to-literal rl) (list lfalse ltrue)))

;; Returns whether rl is a 'tautology', that is, if its `to` and `from`
;; literals are syntactically equal.
;;
;; rl : rule?
;; -> boolean?
(define (rule-tautology? rl)
  (equal? (rule-from-literal rl) (rule-to-literal rl)))

;; Returns whether the rule is left linear, that is, if each variable
;; occurs at most once in the `from` literal.
;;
;; rl : rule?
;; -> boolean?
(define (left-linear? rl)
  (define occs (var-occs (rule-from-literal rl)))
  (for/and ([(v n) (in-hash occs)])
    (<= n 1)))

;; Returns whether rl1 subsumes rl2.
;;
;; rl1 : rule?
;; rl2 : rule?
;; -> any/c
(define (rule-subsumes rl1 rl2)
  (define subst (left-unify (rule-from-literal rl1) (rule-from-literal rl2)))
  (and subst (left-unify (rule-to-literal rl1) (rule-to-literal rl2) subst)))

;; A binary Clause [lit1 lit2] is treated as an *equivalence* (4 implications)
;; rather than two implications,
;; which means that a converse Clause or one that subsumes the latter MUST have been proven too
;; before adding the rules.
;; Thus: (Clause->rules C <=>) == (Clause->rules (make-converse-Clause C) <=>).
;; That is, if C = ~q | p, the converse C' = q | ~p MUST have been proven.
;; We MUST add the four rules at the same time, because the if we add just the (max 2) rules
;; corresponding to the implications of C only, and then add the rules for C',
;; then C' is going to be rewritten to a tautology before it has a chance to propose its rules
;; (since resolution(C, converse(C)) = tautology).
;; The number of returned rules is either 2 or 4.
;; Conv is either the converse Clause of C, or a Clause that subsumes it, and is used
;; as a parent reference (for proofs), even if converse(C) is more specific than Conv.
;; Conv is *not* used to generate the rules themselves (only C), so if Conv is more general than C
;; (as for asymmetric rules) the generated rules will *not* be more general than those of C.
;;
;; C : Clause?
;; Conv : Clause?
;; atom<=> : comparator?
;; dynamic-ok? : boolean?
;; -> (listof rule?)
(define (Clause->rules C Conv #:! atom<=> #:? [dynamic-ok? #true])
  (define cl (Clause-clause C))
  (define unit? (unit-Clause? C))
  (define lit1 (first cl))
  (define lit2 (if unit? lfalse (second cl)))

  (define rg (rule-group C Conv #false))
  ; Not the most efficient. The last 2 cases can be deduced from the first 2,
  ; as long as atom<=> does not count polarity (which SHOULD be the case).
  (define rules
    (for/list ([parent
                (in-list (cond [unit?
                                (list C           #false      C)]
                               [(eq? C Conv) ; self-converse, no need to duplicate the rules
                                (list C           C)]
                               [else
                                (list C           C           Conv        Conv)]))]
               [from (in-list   (list (lnot lit1) (lnot lit2) lit1        lit2))]
               [to   (in-list   (list lit2        lit1        (lnot lit2) (lnot lit1)))]
               ; if Conv is #false for example, skip it (see force-add-binary-Clause!)
               ; also useful for unit?
               #:when parent
               [c (in-value (atom<=> from to))]
               #:when (and (not (order<? c)) ; wrong direction
                           ; either we keep dynamic rules or the rule is static
                           ; TODO: Tests
                           (or dynamic-ok? (order>? c))))
      ; rule cannot be oriented to → from,
      ; so the rule from → to is valid.
      ; It is 'static' if it can be oriented from → to, 'dynamic' otherwise.
      (make-rule parent rg from to (order>? c))))
  (set-rule-group-rules! rg rules)
  rules)

;; a is left-unify< to b if a left-unifies with b.
;;
;; comparator?
;; literal? literal? -> (one-of '< '> '= #false)
(define left-unify<=> (make<=> left-unify) ; left-unify is <=?
  ; equivalent to:
  #;(if (left-unify c1 c2)
      (if (left-unify c2 c1)
          '=
          '<)
      (if (left-unify c2 c1)
          '>
          #false)))

;;; What we want to assess is whether
;;; when applying rl1 to any clause c to get c1,
;;; c1 is always necessarily better than c2 when applying rl2 to c.

;; Compares 2 rules. May help decide if one should be discarded.
;; rl1 <= rl2 if the heavyside of rl1 subsumes that of rl2,
;;   and its light side is no heavier than that of rl2.
;; NOTICE: A return value of '= does not mean that rl1 and rl2 are equal or equivalent;
;; it only means that either can be used. It does mean that their heavy sides are equivalent though.
;;
;; rl1 : rule?
;; rl2 : rule?
;; atom<=> : comparator?
;; -> (one-of '< '> '= #false)
(define (rule<=> rl1 rl2 atom<=>)
  (chain-comparisons
   (left-unify<=> (rule-from-literal rl1) (rule-from-literal rl2))
   (atom<=> (rule-to-literal rl1) (rule-to-literal rl2))))

;; Returns a rule of rwtree that subsumes rl if any.
;;
;; rwtree : rewrite-tree?
;; rl : rule?
;; -> (or/c #false rule?)
(define (find-subsuming-rule rwtree rl)
  (for/or ([rl2 (in-list (rewrite-tree-ref rwtree (rule-from-literal rl)))])
    (and (rule-subsumes rl2 rl) rl2)))

;; If there is a substitution α such that rule-from α = lit, then rule-to α is returned,
;; otherwise #false is returned.
;; If the rule introduces variables, these are freshed.
;;
;; rl : rule?
;; lit : literal?
;; -> literal?
(define (rule-rewrite-literal rl lit)
  (define subst (left-unify (rule-from-literal rl) lit))
  (cond [subst
         ; NOTICE: We need to fresh the variables
         ; that are introduced by the rule.
         ; Ex:
         ; clause: [(p a) (p b)] rule: (p X) -> (q Y)
         ; Then this must be rewritten to:
         ; [(q A) (q B)] and NOT [(q Y) (q Y)]
         (for ([v (in-list (rule-to-fresh rl))])
           (subst-set!/name subst v (new-Var)))
         (left-substitute (rule-to-literal rl) subst)]
        [else #false]))

;; Recursively rewrites the given literal lit from the rules in rwtree.
;; Returns the new literal and the sequence of rules used to rewrite it
;; (may contain duplicate rules).
;; lit: literal?
;; C: The Clause containing the literal lit. Used to avoid backward rewriting C to a tautology.
;;
;; rwtree : rewrite-tree?
;; lit : literal?
;; C : Clause?
(define (binary-rewrite-literal rwtree lit C)
  (define atom<=> (rewrite-tree-atom<=> rwtree))
  (let loop ([lit lit] [rules '()])
    (define candidate-rules (rewrite-tree-ref rwtree lit))
    ; Find the best rewrite of lit according to atom<=>.
    ; Q: Can we rewrite more greedily, that is, applied each rewrite asap?
    ; A: if the set of rules is confluent, yes, since they are all equivalent (although
    ; choosing the one that leads to the smallest literal may still be faster)
    ;   But if the rules are not confluent then we may have a problem.
    (for/fold ([best-lit lit]
               [best-rule #false]
               #:result (if best-rule
                            (loop best-lit (cons best-rule rules)) ; try once more
                            (values best-lit (reverse rules)))) ; no more rewrites
              ([r (in-list candidate-rules)]
               #:unless (let ([g (rule-rule-group r)])
                          ; Don't rewrite yourself or your converse!
                          (or (eq? C (rule-group-Clause g))
                              (eq? C (rule-group-Converse g)))))
      (define new-lit (rule-rewrite-literal r lit))
      (if (and new-lit (order<? (atom<=> new-lit best-lit)))
          (values new-lit r)
          (values best-lit best-rule)))))

;; Returns a rewritten clause and the set (without duplicates) of rules used for rewriting.
;; Rewriting a clause is a simple as rewriting each literal, because
;; only left-unification is used, so the substitution cannot apply to the rest of the clause.
;; NOTICE: The variables of the resulting clause are not freshed.
;;
;; rwtree : rewrite-tree?
;; cl : clause?
;; C : Clause?
(define (binary-rewrite-clause rwtree cl C)
  (let/ec return
    (for/fold ([lits '()]
               [rules '()]
               #:result (values (sort-clause lits)
                                (remove-duplicates rules eq?)))
              ([lit (in-list cl)])
      (define-values (new-lit new-rules) (binary-rewrite-literal rwtree lit C))
      (values (cond [(ltrue? new-lit) (return (list new-lit) new-rules)] ; tautology shortcut
                    [(lfalse? new-lit) lits]
                    [else (cons new-lit lits)])
              (append new-rules rules)))))

(define Clause-type-rw 'rw)

;; Clause? -> boolean?
(define (Clause-type-rw? C)
  (eq? (Clause-type C) Clause-type-rw))

;; Returns a new Clause if C can be rewritten, C otherwise.
;; The parents of the new Clause are C followed by the set of clauses from which the rules
;; used for rewriting C originate.
;;
;; rwtree : rewrite-tree?
;; C : Clause?
;; -> Clause?
(define (binary-rewrite-Clause rwtree C)
  (define-values (new-cl rules) (binary-rewrite-clause rwtree (Clause-clause C) C))
  (cond [(empty? rules)
         C]
        [else
         (++n-binary-rewrites)
         (make-Clause (clause-normalize new-cl)
                      (cons C (remove-duplicates (map rule-Clause rules) eq?))
                      #:type Clause-type-rw)]))

;; Returns whether the clause cl would be rewritten. Does not perform the rewriting.
;;
;; rwtree : rewrite-tree?
;; cl : clause?
;; C : Clause?
;; -> boolean?
(define (binary-rewrite-clause? rwtree cl C)
  (for/or ([lit (in-list cl)])
    ; We need to perform the literal rewriting anyway, because
    ; we need to check if the result is order<?
    (define-values (new-lit new-rules) (binary-rewrite-literal rwtree lit C))
    (not (empty? new-rules))))

;; Returns whether the Clause C would be rewritten. Does not perform the rewriting.
;;
;; rwtree : rewrite-tree?
;; C : Clause?
;; -> boolean?
(define (binary-rewrite-Clause? rwtree C)
  (binary-rewrite-clause? rwtree (Clause-clause C) C))

;; Unconditionally adds the rule rl to the rewrite-tree rwtree,
;; and removes from rwtree the rules that are subsumed by rl
;;
;; rwtree : rewrite-tree?
;; rl : rule?
;; -> void?
(define (add-rule! rwtree rl)
  (define atom<=> (rewrite-tree-atom<=> rwtree))
  (unless (or (rule-tautology? rl)
              (find-subsuming-rule rwtree rl))
    ;; Remove existing rules that are subsumed by rl.
    (define from-lit (rule-from-literal rl))
    (trie-inverse-find rwtree from-lit
                       (λ (nd)
                         (define matches (trie-node-value nd))
                         (when (list? matches) ; o.w. no-value
                           (define new-value
                             (for/list ([rl2 (in-list matches)]
                                        #:unless (rule-subsumes rl rl2))
                               rl2))
                           ;; TODO: If new-value is '(), the node should be removed from the trie,
                           ;; TODO: along with any similar parent. (this could be made automatic?)
                           (set-trie-node-value! nd new-value))))

    ;; NOTICE: No need to backward-rewrite the rules.
    ;; Resolution (in the saturation loop) will take care of generating new rules
    ;; and rewriting both their heavy sides and the light sides,
    ;; while add-rule! takes care of subsumption on the heavy side.
    ;; Letting the resolution loop take care of this is much safer, as it ensures
    ;; that rwtree and utree are in sync, and that rwtree is not going to rewrite
    ;; clauses before utree has a chance to generate new candidates via resolution.
    ;; That is, if we only *remove* stuff from rwtree, and not do the job of resolution
    ;; (apart from forward rewriting), then we should be fine?
    (++n-rule-added)
    (trie-insert! rwtree from-lit rl)))

;; Unconditionally removes a single (oriented) rule from the tree.
;; Use with caution and see instead remove-rule-group!.
;;
;; rwtree : rewrite-tree?
;; rl : rule?
;; -> void?
(define (remove-rule! rwtree rl)
  (trie-find rwtree (rule-from-literal rl)
             (λ (nd)
               (define old-value (trie-node-value nd))
               (when (list? old-value) ; o.w. no-value
                 ;; TODO: If new-value is '(), the node should be removed from the trie,
                 ;; TODO: along with any similar parent. (this could be made automatic?)
                 (set-trie-node-value! nd (remove rl old-value eq?))))))

;; Removes a group of rules that were added at the same time via Clause->rules
;; (via add-binary-Clause!).
;;
;; rwtree : rewrite-tree?
;; gp : rule-group?
;; -> void?
(define (remove-rule-group! rwtree gp)
  (for ([rl (in-list (rule-group-rules gp))])
    (remove-rule! rwtree rl)))

;; Turn the unit Clause into a binary Clause before adding it.
;; As for self-converses, we say that C is its own converse but that's a lie;
;; This is to avoid problems and specific cases when loading/saving rules.
;; A self-converse rule is necessarily dynamic (unless commutativity can be handled statically?),.
;; A unit rule is necessarily static (since $false is the bottom element).
;; Hence a 'self-converse' static rule is necessarily a unit rule (for now).
;;
;; rwtree : rewrite-tree?
;; C : Clause?
;; rewrite? : boolean?
;; -> void?
(define (rewrite-tree-add-unit-Clause! rwtree C #:? rewrite?)
  (unless (unit-Clause? C) (error "Non-unit Clause v" C))
  (rewrite-tree-add-binary-Clause! rwtree C C #:rewrite? rewrite?))

;; Rewriting of the clause C must be done prior to calling this function.
;; Conv: Converse of C. See Clause->rules.
;; Returns the new rules (use these to obtain the rewritten Clauses).
;;
;; rwtree : rewrite-tree?
;; C : Clause?
;; Conv : Clause?
;; rewrite? : boolean?
;; -> void?
(define (rewrite-tree-add-binary-Clause! rwtree C Conv #:? [rewrite? #true])
  (cond
    [(Clause-binary-rewrite-rule? C) ; already added as a rule in the past?
     (when-debug>= steps
                   (displayln "Clause has already been added before. Skipping."))
     '()]
    [else
     (let-values
         ([(C Conv)
           (cond/else
            [(not rewrite?) (values C Conv)]
            #:else ; Rewriting
            (define C2 (binary-rewrite-Clause rwtree C))
            (when-debug>= steps
                          (displayln "Considering binary Clause for a rule (before rewriting):")
                          (display-Clause-ancestor-graph C #:depth 0)
                          (displayln "After rewriting:")
                          (display-Clause-ancestor-graph C2)
                          (when (Clause-tautology? C2)
                            (displayln "...but tautology.")))
            #:cond
            [(Clause-tautology? C2)
             (values #false #false)]
            #:else
            ; Rewritten rule can still be used, rewrite the converse too. Not very efficient.
            (define Conv2 (if (eq? C Conv) Conv (binary-rewrite-Clause rwtree Conv)))
            (values C2 Conv2))])

       (cond
         [(not C) '()]
         [(empty-clause? (Clause-clause C))
          ; Refutation found!
          (when-debug>= steps
                        (displayln "Refutation found while adding rules!")
                        (display-Clause-ancestor-graph C))
          ; TODO: Let's just discard it for now. A refutation will probably be found very early
          ; TODO: at the next saturation iteration.
          '()]
         [else
          (define atom<=> (rewrite-tree-atom<=> rwtree))
          (define dynamic-ok? (rewrite-tree-dynamic? rwtree))
          (define rls (Clause->rules C Conv #:atom<=> atom<=> #:dynamic-ok? dynamic-ok?))
          (when-debug>= steps
                        (displayln "Adding the following rules:")
                        (display-rules rls))
          (for ([rl (in-list rls)])
            (add-rule! rwtree rl))
          ; We set the bit to #true, *even if* no rule has been added,
          ; because the purpose of this bit is to avoid considering
          ; C later again to save time.
          ; TODO: We could set Conv as a rewrite-rule too but ONLY if C also subsumes the converse
          ; TODO: of conv.
          (set-Clause-binary-rewrite-rule?! C #true)
          rls]))]))

;; Adds the binary Clauses Cs as rules and returns the corresponding rules.
;; If rewrite? is not #false, the clauses are rewritten beforehand using the rules in the tree
;; (the order of the rules in Cs does matter as of now)
;; and tautologies are filtered out.
;;   The default rewrite = #true is 'safe' because when considering A->B, Clause->rules also considers
;;   B->A because we provide it with the converse equivalence (that is, the proof that B->A is valid).
;;   Hence even converse implications can safely be rewritten to tautologies without losing rules.
;; Conv: MUST Subsumes the converse clause of each of Cs.
;;
;; rwtree : rewrite-tree?
;; Cs : (listof Clause?)
;; Conv : Clause?
;; rewrite? : boolean?
;; -> void?
(define (rewrite-tree-add-binary-Clauses! rwtree Cs Conv #:? rewrite?)
  (for/fold ([rules '()])
            ([C (in-list Cs)])
    (define rls (rewrite-tree-add-binary-Clause! rwtree C Conv #:rewrite? rewrite?))
    (append rls rules)))

;; Returns the list of rules (without duplicates) of rwtree.
;;
;; rwtree : rewrite-tree?
;; -> (listof rule?)
(define (rewrite-tree-rules rwtree)
  (remove-duplicates (append* (trie-values rwtree)) eq?))

;; Returns the list of rule groups (without duplicates) of rwtree.
;;
;; rwtree : rewrite-tree?
;; -> (listof rule-group?)
(define (rewrite-tree-rule-groups rwtree)
  (remove-duplicates (map rule-rule-group (append* (trie-values rwtree))) eq?))

;; Returns the list of unique Clauses that have been used to create the rules
;; held by the given rule groups.
;;
;; groups : (listof rule-group?)
;; -> (listof Clause?)
(define (rule-groups-original-Clauses groups)
  (remove-duplicates (append (map rule-group-Clause groups)
                             (map rule-group-Converse groups))
                     eq?))

;; Returns the list of unique Clauses that have been used to create the rules.
;;
;; rules : (listof rule?)
;; -> (listof Clause?)
(define (rules-original-Clauses rules)
  (rule-groups-original-Clauses (map rule-rule-group rules)))

;; Returns the list of unique Clauses that have been used to create the rules
;; of the rewrite tree.
;;
;; rwtree : rewrite-tree?
;; -> (listof Clause?)
(define (rewrite-tree-original-Clauses rwtree)
  (rule-groups-original-Clauses (rewrite-tree-rule-groups rwtree)))

;; Returns the number of rules in the rewrite tree.
;;
;; rwtree : rewrite-tree?
;; -> exact-nonnegative-integer?
(define (rewrite-tree-count rwtree)
  (length (rewrite-tree-rules rwtree))) ; not efficient

;; Returns a dictionary of statistics about the rewrite tree.
;;
;; rwtree : rewrite-tree?
;; -> list?
(define (rewrite-tree-stats rwtree)
  (define rules (rewrite-tree-rules rwtree))
  (define n-rules (length rules))
  (define n-dyn (count (λ (r) (not (rule-static? r))) rules))
  (define n-unit (count unit-rule? rules))
  (define n-bin (- n-rules n-unit))
  `((rules . ,n-rules)
    (unit-rules . ,n-unit)
    (binary-rules . ,n-bin)
    (binary-rules-static . ,(- n-bin n-dyn))
    (binary-rules-dynamic . ,n-dyn)))

;; Attempts to simplify the rules by successively removing each rule,
;; rewrite its underlying Clause and add the rule again—checking for subsumption.
;; Since all rules are processed, the new set of rules can be obtained via rewrite-tree-rules.
;; Notice: Not (currently) suitable for use *inside* the saturation loop because the new Clauses
;; (as per `eq?` are not part of the active set or candidates.
;;
;; rwtree : rewrite-tree?
;; -> void?
(define (re-add-rules! rwtree)
  (define groups (rewrite-tree-rule-groups rwtree))
  (for ([gp (in-list groups)])
    (remove-rule-group! rwtree gp)
    (define C (rule-group-Clause gp))
    (set-Clause-binary-rewrite-rule?! C #false) ; otherwise will not be added
    (rewrite-tree-add-binary-Clause! rwtree C (rule-group-Converse gp) #:rewrite? #true)))

;===================;
;=== Save / load ===;
;===================;

;; Save the rules to file f (as clauses).
;;
;; rwtree : rewrite-tree?
;; rules-file : file?
;; exists : symbol? ; see `display-to-file`.
;; -> void?
(define (save-rules! rwtree #:! rules-file #:? [exists 'replace])
  (define groups (rewrite-tree-rule-groups rwtree))
  ; We use hash to avoid duplicating clauses, so that we also avoid loading
  ; the same clause under different names.
  (define-counter idx 0)
  (define h (make-hasheq))
  (define (get-clause C)
    (if (hash-has-key? h C)
        (hash-ref h C)
        (begin0 (Clause-clause C)
                (++idx)
                (hash-set! h C idx))))

  (make-parent-directory* rules-file) ; ensure the path exists
  (with-output-to-file rules-file #:exists exists
    (λ () (for ([gp (in-list groups)])
            (define C (rule-group-Clause gp))
            (define Conv (rule-group-Converse gp))
            (writeln (if (eq? C Conv)
                         (list (get-clause C))  ; self-converse
                         (list (get-clause C)
                               (get-clause Conv))))))))

;; Private.
;;
;; rules-file : file?
;; -> (listof Clause?)
(define (load-rule-Clause-lists #:! rules-file)
  (define-counter idx 0)
  ; Each clause has a number (local to the file) and if a number appears while reading
  ; then we must load the saved clause of the same number.
  (define h (make-hasheqv))
  (define (get-Clause! x)
    (cond [(number? x)
           (hash-ref h x)]
          [else
           (++idx)
           (define C (make-Clause x #:type 'load))
           (hash-set! h idx C)
           C]))

  (for/list ([cls (in-list (file->list rules-file))])
    (map get-Clause! cls)))

;; Load the rules from file into the rewrite-tree.
;; Optionally: rewrites the rules before adding them
;; Returns the set of new rules.
;;
;; rwtree : rewrite-tree?
;; rules-file : file?
;; rewrite? : boolean?
;; -> (listof rule?)
(define (load-rules! rwtree #:! rules-file #:? [rewrite? #true])
  (define Crules (load-rule-Clause-lists #:rules-file rules-file))
  (for/fold ([rules '()])
            ([Cs (in-list Crules)])
    (define C (first Cs))
    (define Conv
      (if (= (length Cs) 1)
          C ; self-converse
          (second Cs)))
    (define new-rules (rewrite-tree-add-binary-Clause! rwtree C Conv #:rewrite? rewrite?))
    (append new-rules rules)))

;=======================;
;=== Filtering rules ===;
;=======================;

;; Returns a new rwtree containing only the filtered rules
;;
;; rwtree : rewrite-tree?
;; proc : rule-group? -> boolean?
;; -> rewrite-tree?
(define (filter-rule-groups rwtree proc)
  (define rwtree2 (make-rewrite-tree #:atom<=> (rewrite-tree-atom<=> rwtree)
                                     #:dynamic-ok? (rewrite-tree-dynamic? rwtree)))
  (define groups (rewrite-tree-rule-groups rwtree))
  (for ([gp (in-list groups)])
    (when (proc gp)
      ; Copy the clauses (shallow)
      (define C    (make-Clause (Clause-clause (rule-group-Clause   gp)) #:type 'filter-rule-groups))
      (define Conv (make-Clause (Clause-clause (rule-group-Converse gp)) #:type 'filter-rule-groups))
      (rewrite-tree-add-binary-Clause! rwtree2 C Conv #:rewrite? #true)))
  rwtree2)

;==================;
;=== Confluence ===;
;==================;

;; Unifies (resolves) only with the lhs of rules and returns
;; the new set of Clauses (and converse Clauses).
;; If bounded? is not #false, then Clauses which have a literal
;; that is heavier (in literal-size) than a `from' literal of its parent rules
;; are discarded.
;;
;; rwtree : rewrite-tree?
;; rl : rule?
;; bounded? : boolean?
;; -> (listof Clause?)
(define (find-critical-pairs rwtree rl #:? [bounded? (*bounded-confluence?*)])
  (define lnot-from-lit1 (lnot (rule-from-literal rl)))
  (define to-lit1 (rule-to-literal rl))
  (define new-Clauses '())
  ; Resolution (like unification with the converse)
  ; to ensure that the Clause parents are correct
  (trie-both-find rwtree lnot-from-lit1
                  (λ (nd)
                    (define rls (trie-node-value nd))
                    (when (list? rls)
                      (for ([rl2 (in-list rls)]
                            #:unless (eq? rl rl2))
                        (define from-lit2 (rule-from-literal rl2))
                        (define to-lit2 (rule-to-literal rl2))
                        (define s (unify lnot-from-lit1 from-lit2))
                        (when s
                          (define cl (clause-normalize (substitute (list to-lit1 to-lit2) s)))
                          (define C (make-Clause cl (list (rule-Clause rl) (rule-Clause rl2))
                                                 #:type 'c-p)) ; critical-pair
                          (define max-size (max (literal-size (rule-from-literal rl))
                                                (literal-size from-lit2)))
                          (unless (or (Clause-tautology? C)
                                      (and bounded?
                                           (> (max (literal-size (first cl))
                                                   (literal-size (second cl)))
                                              max-size)))
                            (cons! C new-Clauses)))))))
  new-Clauses)

;; This is a bad fix. Instead we should find the correct converse Clause (via rule groups)
;; Returns the list of rules that were added, or '() if none.
;;
;; rwtree : rewrite-tree?
;; C : Clause?
;; -> void?
(define (force-add-binary-Clause! rwtree C)
  (define Conv (make-converse-Clause C))
  (rewrite-tree-add-binary-Clause! rwtree C (if (Clause-equivalence? C Conv) C Conv)))

;; Add all critical pairs between existing rules.
;; This procedure may need to be called several times, but such a loop may not terminate
;; as some rules can be inductive.
;; It is recommended to call `simplify-rewrite-tree!` before and after calling this procedure.
;; Returns #true if any rules has been added.
;;
;; rwtree : rewrite-tree?
;; bounded? : boolean?
;; -> boolean?
(define (rewrite-tree-confluence-step! rwtree #:? bounded?)
  (define rules (rewrite-tree-rules rwtree))
  (for/fold ([any-change? #false])
            ([rl (in-list rules)])
    (define pairs (find-critical-pairs rwtree rl #:bounded? bounded?))
    (for/fold ([any-change? any-change?])
              ([C (in-list pairs)])
      (define change? (force-add-binary-Clause! rwtree C))
      (or any-change? (and change? #true)))))

;; Performs several steps of confluence until no more change happens.
;; Returns whether any change has been made.
;; Notice: if bounded? is #false, this may loop forever.
;; If it does halt however, the system is confluent, but may not be minimal;
;; call re-add-rules! to remove unnecessary rules and simplify rhs of rules
;; (this should help make rewriting faster).
;; It is advised to call re-add-rules! once before and once after rewrite-tree-confluence!.
;;
;; rwtree : rewrite-tree?
;; bounded? : boolean?
;; max-steps : exact-nonnegative-integer?
;; -> boolean?
(define (rewrite-tree-confluence! rwtree #:? bounded? #:? [max-steps (*confluence-max-steps*)])
  (let loop ([step 1] [any-change? #false])
    (define changed? (rewrite-tree-confluence-step! rwtree #:bounded? bounded?))
    (cond [(or (>= step max-steps)
               (not changed?))
           any-change?]
          [else (loop (+ step 1) #true)])))

;================;
;=== Printing ===;
;================;

;; Returns a list representing the rule.
;;
;; rule? -> list?
(define (rule->list rl)
  (cons (Clause-idx (rule-Clause rl))
        (Vars->symbols (list (rule-from-literal rl)
                             (rule-to-literal rl)))))

;; A comparator to sort the rules for printing
(define display-rule<=>
  (make-chain<=> boolean<=> rule-static?
                 boolean<=> unit-rule?
                 boolean<=> (λ (r) (lnot? (rule-from-literal r)))
                 number<=> (λ (r) (+ (tree-size (rule-from-literal r))
                                     (tree-size (rule-to-literal r))))
                 number<=> (λ (r) (tree-size (rule-to-literal r)))
                 length<=> rule-to-fresh))

;; (listof rule?) -> (listof rule?)
(define (sort-rules rls)
  (sort rls (λ (a b) (order>? (display-rule<=> a b)))))

;; Human-readable output
;; Notice: Since unit rules are asymmetric (only one rule per unit clause),
;; if positive-only is not #false it may not display some unit rules.
;;
;; rls : (listof rule?)
;; sort? : boolean?
;; positive-only? : boolean?
;; unit? : boolean?
(define (rules->string rls #:? [sort? #true] #:? [positive-only? #false] #:? [unit? #true])
  (string-join
   (for/list ([rl (in-list (if sort? (sort-rules rls) rls))]
              #:unless (or (and (not unit?) (unit-rule? rl))
                           (and positive-only? (lnot? (rule-from-literal rl)))))
     ; abusing clause->string:
     (clause->string (list (if (rule-static? rl) "static " "dynamic")
                           (map Var (rule-to-fresh rl))
                           ':
                           (rule-from-literal rl)
                           '→
                           (rule-to-literal rl))))
   "\n"))

;; Like rules->string but directly displays the result.
;;
;; rls : (listof rule?)
;; sort? : boolean?
;; positive-only? : boolean?
;; unit? : boolean?
(define-wrapper (display-rules (rules->string rls #:? sort? #:? positive-only? #:? unit?))
  #:call-wrapped call
  (displayln (call)))
