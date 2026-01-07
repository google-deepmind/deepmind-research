#lang racket/base

;**************************************************************************************;
;****                             Saturation Algorithm                             ****;
;**************************************************************************************;

(require bazaar/cond-else
         bazaar/date
         bazaar/debug
         bazaar/dict
         bazaar/mutation
         data/heap/unsafe
         data/queue
         define2
         global
         racket/block
         racket/dict
         racket/format
         racket/list
         racket/math
         racket/pretty
         racket/string
         satore/clause-struct
         satore/clause-format
         satore/clause
         satore/interact
         satore/json-output
         satore/misc
         satore/rewrite-tree
         satore/tptp
         satore/unification-tree
         satore/unification)

(provide (all-defined-out))

;===============;
;=== Globals ===;
;===============;

(define-global:boolean *quiet-json?* #false
  '("JSON output format and silent mode. Deactivates some verbose options."))

(define-global *memory-limit* 4096
  "Memory limit in MB, including the Racket VM."
  exact-positive-integer?
  string->number)

(define-global:boolean *find-unit-rules-in-candidates?* #false
  '("Search for unit rewrite rules in the condidate set?"
    "This may speed up the search significantly, or slow it down significantly,"
    "depending on how many unit rules can be generated."))

(define-global:boolean *backward-rw?* #true
  '("Use binary-clause rewriting? (aka θ-equivalences)."
    "The iterative mode does not use backward rewrites."))

(define-global:boolean *dynamic-rules?* #false
  "Use dynamic rules? Experimental.")

(define-global:boolean *proof?* #false
  "Display the proof when found?")

(define-global *age:cost* '(1 9)
  "Age:cost ratio. Format: <age-freq>:<cost-freq>, e.g., '1:3'"
  (λ (p) (and (list? p) (= 2 (length p)) (andmap exact-nonnegative-integer? p)))
  (λ (s) (map string->number (string-split s ":"))))

(define-global:category *cost-type* 'weight-fair
  '(weight-fair weight)
  "Cost type.")

(define-global *cost-depth-factor* 1.5
  "Cost = weight + cost-depth-factor * depth"
  (λ (x) #true)
  string->number)

(define-global *cost-noise* 0.
  "Noise factor to add to costs"
  (λ (x) (and (real? x) (>= x 0)))
  string->number)

(define-global:boolean *parent-discard?* #false
  "Discard clauses when at least one parent has been discarded?")

(define-global:boolean *discover-online?* #true
  '("Use rewrite rules as soon as they are discovered?"
    "The rules will not be made confluence until the next restart"))

(define-global:boolean *negative-literal-selection?* #true
  "When resolving, if a clause has negative literals, only select one of them.")

(define-global *cpu-limit* +inf.0
  "CPU limit in seconds per problem."
  exact-positive-integer?
  string->number)

(define-global *cpu-first-limit* +inf.0
  "CPU limit in seconds per problem."
  exact-positive-integer?
  string->number)

(define-global *cpu-limit-factor* 3
  '("Increase of cpu limit of cpu-first-limit once an iteration has failed."
    "Set to 0 to avoid restarting."
    "Assumes --cpu-limit.")
  (λ (x) (and (real? x) (>= x 0)))
  string->number)

(define-global *input-rules* #false
  '("File to read rewrite rules from.")
  file-exists?
  values)

(define-global *output-rules* #false
  '("File to write rewrite rules to. If 'auto', a unique name is chosen automatically.")
  (λ (x) #true)
  (λ (str)
    (if (string=? str "auto")
        (build-path "rules" (string-append "rules-" (date-iso-file) ".txt"))
        str)))

;======================;
;=== Rule discovery ===;
;======================;

;; Finds new binary equivalences between `C` and the clauses of `utree`,
;; and adds and returns the set of resulting new rules that can be added to `rwtree`.
;;
;; rwtree-out : rewrite-tree?
;; C : Clause?
;; utree : unification-tree?
(define (discover-new-rules! rwtree-out C utree)
  (cond/else
   [(not rwtree-out) '()]

   ;; FIND UNIT REWRITE RULES
   ;; Unit rewrite using the binary rewrite tree
   [(unit-Clause? C)
    (when-debug>= steps (displayln "Found unit clause"))
    ;; Add the new unit clause to the set of unit rewrite rules
    (rewrite-tree-add-unit-Clause! rwtree-out C #:rewrite? #false)]

   [(not (binary-Clause? C))
    '()]

   ;; FIND BINARY REWRITE RULES
   #:else
   (when-debug>= steps (displayln "Found binary clause"))

   ;; We search for the converse implication in the active set.
   ;; This takes care of clauses that have converse clauses but not the converse.
   ;; Ex: C1 = p(X) | q(X)  ;   C2 = ~p(a) | ~q(a)
   ;; C2 has a converse clause, but not C1. Hence C2 can be added as a binary rewrite rule
   ;; but not C1.
   (define Conv (make-converse-Clause C))
   (define self-conv? (Clause-subsumes C Conv))
   #:cond
   [self-conv?
    ; Self-converse Clause, so only one Clause to add (but will lead to 2 rules).
    (when-debug>= steps (displayln "Self-converse"))
    (rewrite-tree-add-binary-Clause! rwtree-out C C #:rewrite? #false)]
   #:else
   (define Subs (utree-find/any utree Conv (λ (a Conv) (and (binary-Clause? a)
                                                            (Clause-subsumes a Conv)))))
   #:cond
   [Subs
    ; We found a converse Clause in the active set, or a Clause that subsumes the
    ; converse Clause, so we can add selected-Clause as a rule and also its
    ; converse Clause.
    ; We can't add Subs, since Subs may not itself have a converse
    ; clause in the active set. Ex:
    ; selected-Clause = p(a) | q(a), so Conv = ~p(a) | ~q(a) and Subs = ~p(X) | ~q(X)
    ; Subs cannot be added as a rule because (supposedly) we haven't found a converse
    ; clause yet, but we can still add Conv as a rule.
    ; If Subs is aleardy a rule, Conv will be rewritten to a tautology and discarded.
    (when-debug>= steps (printf "Found converse subsuming clause: ~a\n"
                                (Clause->string Subs)))
    ; Since C has already been rewritten, there is no need to do it again.
    (rewrite-tree-add-binary-Clause! rwtree-out C Subs #:rewrite? #false)]
   #:else
   ; Asymmetric rules.
   ; Even when selected-Clause is not a rule (yet?), it may still enable
   ; other more specific Clauses of the active set to be added as rules too.
   ; Eg, selected-Clause = p(X) | q(X) and some other active clause is ~p(a) | ~q(a).
   (define Subsd (utree-find/all utree Conv (λ (a Conv) (and (binary-Clause? a)
                                                             (Clause-subsumes Conv a)))))
   #:cond
   [(not (empty? Subsd))
    (when-debug>= steps
                  (displayln "Found converse subsumed clauses:")
                  (print-Clauses Subsd))
    ; rewrite=#true is ok here because Clause->rules already considers both directions
    ; of the implication. Hence if a potential rule is rewritten to a tautology,
    ; we know it's already redundant anyway.
    ; TODO: Rename this to add-asymmetric-rules ?
    (rewrite-tree-add-binary-Clauses! rwtree-out Subsd C #:rewrite? #true)]
   #:else '()))

;====================;
;=== Clause costs ===;
;====================;

;; Clause comparison for the cost queue.
;;
;; Clause? Clause? -> boolean?
(define (Candidate<= C1 C2)
  (<= (Clause-cost C1) (Clause-cost C2)))

;; Sets the cost of a list of Clauses that all have the same parent `parent`.
;; Some noise can be added to the cost via *cost-noise*.
;;
;; Cs : (listof Clause?)
;; cost-type: symbol?
;; parent : Clause?
;; cost-depth-factor : number?
;; -> void?
(define (Clauses-calculate-cost! Cs
                                 cost-type
                                 parent
                                 #:! cost-depth-factor)
  (case cost-type
    ;; Very simple cost function that uses the weight of the Clause. May not be fair.
    [(weight)
     (for ([C (in-list Cs)])
       (set-Clause-cost! C
         (if (empty? (Clause-clause C))
             -inf.0 ; empty clause should always be top of the list
             (Clause-size C))))]

    ;; Very simple cost function that is fair
    [(weight-fair)
     (for ([C (in-list Cs)])
       (set-Clause-cost! C
         (if (empty? (Clause-clause C))
             -inf.0 ; empty clause should always be top of the list
             (+ (* (Clause-depth C) cost-depth-factor)
                (Clause-size C)))))])

  ;; Add noise to the cost so as to potentially solve more later.
  ;; To combine with a slowly increasing step-limit-factor in iterative saturation
  (unless (zero? (*cost-noise*))
    (define ε (*cost-noise*))
    (for ([C (in-list Cs)])
      (set-Clause-cost! C
        (* (+ (- 1. ε) (* ε (random)))
           (Clause-cost C))))))

;==================;
;=== Saturation ===;
;==================;

;; List of possible status values. Used to prevent mistakes.
(define statuses '(refuted saturated time memory steps running))

;; Returns whether the result dictionary `res` has the status `status`.
;;
;; res : dict?
;; status : symbol?
(define (check-status res status)
  (define res-status (dict-ref res 'status #false))

  ; To avoid silent typo bugs.
  (assert (memq status statuses) status)
  (assert (memq res-status statuses) res-status)

  (eq? status res-status))

;; The main algorithm. Saturates the formula given by the input clauses
;; by adding new clauses (either resolutions or factors) until either
;; the empty clause is produced, or a resource limit is reached (steps, time, memory).
;;
;; input-clauses : (listof clause?)
;; step-limit : number?
;; memory-limit : number?
;; cpu-limit : number?
;; rwtree : (or/c #false rewrite-tree?)
;; rwtree-out : (or/c #false rewrite-tree?)
;; backward-rewrite? : boolean?
;; parent-discard? : boolean?
;; age:cost : (list/c exact-nonnegative-integer? exact-nonnegative-integer?)
;; cost-type : symbol?
;; disp-proof? : boolean?
;; L-resolvent-pruning : boolean?
;; find-unit-rules-in-candidates? : boolean?
;; negative-literal-selection? : boolean?
;; -> dict?
(define (saturation input-clauses
                    #:? [step-limit +inf.0]
                    #:? [memory-limit (*memory-limit*)] ; in MB
                    #:? [cpu-limit +inf.0] ; in seconds
                    ; The rewrite tree holding the binary rules.
                    ; Set it to #false to deactivate it.
                    #:? [rwtree (make-rewrite-tree #:atom<=> (get-atom<=>)
                                                   #:dynamic-ok? (*dynamic-rules?*)
                                                   #:rules-file (*input-rules*))]
                    ; rewrite-tree where new rules found during saturation are stored.
                    ; If rwtree-out is different from rwtree, new rules are not used but only stored,
                    ; and backward rewriting is deactivated.
                    #:? [rwtree-out rwtree] ; or #false if don't care
                    ; Only effective if (eq? rwtree rwtree-out)
                    #:? [backward-rewrite? (*backward-rw?*)]
                    #:? [parent-discard? (*parent-discard?*)]
                    #:? [age:cost (*age:cost*)] ; chooses age if (< (modulo t (+ age cost)) age)
                    #:? [cost-type (*cost-type*)]
                    #:? [disp-proof? (*proof?*)]
                    #:? [L-resolvent-pruning? (*L-resolvent-pruning?*)]
                    #:? [find-unit-rules-in-candidates? (*find-unit-rules-in-candidates?*)]
                    #:? [negative-literal-selection? (*negative-literal-selection?*)])
  ;; Do NOT reset the clause-index, in particular if rwtree is kept over several calls to saturation.
  #;(reset-clause-index!)

  ;; Tree containing the active Clauses
  (define utree (make-unification-tree))
  ;; Clauses are pulled from priority first, unless empty.
  ;; INVARIANT: active-Clauses U priority is (should be!) equisatisfiable with input-Clauses.
  ;; In other words, if priority is empty, then the set of active-Clauses
  ;; is equisatisfiable with the input-Clauses.
  ;; Some active clauses may be removed from utree and pushed back into priority for further
  ;; processing like 'backward' rewriting. In that case, the active Clauses (utree) is not
  ;; 'complete'.
  (define priority (make-queue))

  ;; Both heaps contain the candidate clauses (there may be duplicates between the two,
  ;; but this is checked when extracting Clauses from either heap).
  (define candidates (make-heap Candidate<=))
  (define age-queue (make-heap Clause-age>=))
  ;; Frequency of extracting Clauses from either heap.
  (define age-freq (first age:cost))
  (define cost-freq (second age:cost))
  (define age+cost-freq (+ age-freq cost-freq))

  ;; Add the Clauses Cs to the priority queue for priority processing.
  (define (add-priority-Clauses! Cs)
    (for ([C (in-list Cs)])
      ; Need to set candidate? to #true otherwise it may be skipped.
      ; (or maybe we should not skip clauses loaded from `priority`?)
      (set-Clause-candidate?! C #true)
      (enqueue! priority C)))

  (define cost-depth-factor (*cost-depth-factor*)) ; immutable value

  (define (add-candidates! parent Cs)
    ;; Calculate costs and add to candidate heap.
    (unless (empty? Cs)
      (Clauses-calculate-cost! Cs cost-type parent #:cost-depth-factor cost-depth-factor)

      (for ([C (in-list Cs)])
        (set-Clause-candidate?! C #true))

      (unless (= 0 cost-freq) (heap-add-all! candidates Cs))
      (unless (= 0 age-freq)  (heap-add-all! age-queue  Cs))
      (when-debug>= steps
                    (printf "#new candidates: ~a #candidates: ~a\n"
                            (length Cs)
                            (heap-count candidates)))))

  (define input-Clauses
    (map (λ (c) (make-Clause c '() #:type 'in)) input-clauses))

  ;; This maintains the invariant: If priority is empty, then the set of active-Clauses
  ;; is equisatisfiable with the input-Clauses.
  ;; In other words active-Clauses U priority is (should be!) equisatisfiable with input-Clauses.
  (add-priority-Clauses! input-Clauses)

  ;; We add the Clauses of the binary rules as candidates, so as to not cluter the active set
  ;; in case there are many rules.
  ;; Another option is to add them to the priority queue because they can be seen as possibly
  ;; useful lemmas.
  (when rwtree
    ; A mock root clause parent of all input rules
    (define C0rules (make-Clause (list ltrue) '() #:type 'rules-root))
    ; rewrite=#false should not be necessary (since rewriting checks if a clause is from the original
    (add-candidates! C0rules (rewrite-tree-original-Clauses rwtree)))

  (define step 0)
  (reset-n-tautologies!)
  (define n-parent-discard 0)
  (define n-forward-subsumed 0)
  (define n-backward-subsumed 0)
  (reset-n-binary-rewrites!)
  (reset-n-rule-added!)
  (reset-subsumes-stats!)
  (reset-n-L-resolvent-pruning!)
  (define start-time (current-milliseconds))

  ;; TODO: Some calls are very slow...
  (define (make-return-dict status [other '()])
    (assert (memq status statuses) status)
    (define stop-time (current-milliseconds))
    `((status . ,status)
      (steps . ,step)
      (generated . ,clause-index) ; includes all input clauses and rules and intermediate steps
      (actives . ,(length (unification-tree-Clauses utree)))
      (candidates . ,(heap-count candidates))
      (priority-remaining . ,(queue-length priority))
      (tautologies . ,n-tautologies) ; counted in generated, (mostly) not in candidates
      ,@(rewrite-tree-stats rwtree)
      (binary-rewrites . ,n-binary-rewrites)
      (forward-subsumed . ,n-forward-subsumed)
      (backward-subsumed . ,n-backward-subsumed)
      (subsumes-checks . ,n-subsumes-checks)
      (subsumes-steps . ,n-subsumes-steps)
      (subsumes-breaks . ,n-subsumes-breaks)
      (parent-discard . ,n-parent-discard)
      (L-resolvent-pruning . ,n-L-resolvent-pruning)
      (memory . ,(current-memory-use)) ; doesn't account for GC---this would take too much time
      (time . ,(- stop-time start-time))
      . ,other))

  (define (make-refuted-dict C)
    (define proof (Clause-ancestor-graph C)) ; no duplicates
    (define flat-proof (flatten proof))
    (define type-occs (occurrences flat-proof #:key Clause-type))
    (when disp-proof?
      (displayln "#| begin-proof")
      (display-Clause-ancestor-graph C #:tab " ")
      (displayln "end-proof |#"))
    (make-return-dict 'refuted
                      `((proof-length . ,(length flat-proof)) ; doesn't account for compound rewrites
                        (proof-steps . ,(for/sum ([C2 (in-list flat-proof)])
                                          (define n (length (Clause-parents C2)))
                                          (if (< n 2) n (- n 1))))
                        (proof-inferences . ,(count (λ (C2) (not (empty? (Clause-parents C2))))
                                                    flat-proof))
                        ,@(for/list ([(t o) (in-dict type-occs)])
                            (cons (string->symbol (format "proof-type:~a" t)) o)))))

  ;:::::::::::::::::::::;
  ;:: Saturation Loop ::;
  ;:::::::::::::::::::::;

  (define result
    (let loop ()
      (++ step)
      (define time-passed (- (current-milliseconds) start-time)) ; this is fast
      (define mem (current-memory-use-MB)) ; mflatt says it's fast

      (when-debug>= steps
                    (printf "\nstep: ~a  generated: ~a  processed/s: ~a  generated/s: ~a\n"
                            step
                            clause-index
                            (quotient (* 1000 step) (+ 1 time-passed))
                            (quotient (* 1000 clause-index) (+ 1 time-passed))))
      (cond/else
       [(and (= 0 (heap-count candidates))
             (= 0 (heap-count age-queue))
             (= 0 (queue-length priority)))
        (when-debug>= steps (displayln "Saturated"))
        (make-return-dict 'saturated)]
       [(> step step-limit) (make-return-dict 'steps)]
       [(> time-passed (* 1000 cpu-limit)) (make-return-dict 'time)]
       [(and (> mem memory-limit)
             (block
              (define pre (current-milliseconds))
              ;; Memory is full, but try to collect garbage first.
              (unless (*quiet-json?*)
                (printf "; before GC: memory-limit: ~a memory-use: ~a\n" memory-limit mem))
              (collect-garbage)
              (collect-garbage)
              (define mem2 (current-memory-use-MB))
              (define post (current-milliseconds))
              (unless (*quiet-json?*)
                (printf "; after  GC: memory-limit: ~a memory-use: ~a gc-time: ~a\n"
                        memory-limit mem2 (* 0.001 (- post pre))))
              (> mem2 memory-limit)))
        ; mem is full even after GC, so exit
        (make-return-dict 'memory)]
       #:else
       ;; Choose a queue/heap to extract the selected-Clause from.
       (define queue
         (cond [(> (queue-length priority) 0)
                ; Always has priority.
                priority]
               [(or (= 0 (heap-count candidates))
                    (and (> (heap-count age-queue) 0)
                         (< (modulo step age+cost-freq) age-freq)))
                ; TODO: This is somewhat defeated by the `priority` queue.
                age-queue]
               [else candidates]))
       (when-debug>= steps
                     (printf "Selected queue: ~a\n" (cond [(eq? queue priority) "priority"]
                                                          [(eq? queue candidates) "candidates"]
                                                          [else "age queue"])))
       (define selected-Clause
         (if (heap? queue)
             (begin0 (heap-min queue)
                     (heap-remove-min! queue))
             (dequeue! queue)))

       #:cond
       ;; ALREADY PROCESSED
       [(not (Clause-candidate? selected-Clause))
        (when-debug>= steps (displayln "Clause already processed. Skipping."))
        (-- step) ; don't count this as a step
        (loop)]
       ;; ONE PARENT DISCARDED
       [(and parent-discard? (ormap Clause-discarded? (Clause-parents selected-Clause)))
        (when-debug>= steps (displayln "At least one parent has been discarded. Discard too."))
        (discard-Clause! selected-Clause)
        (++ n-parent-discard)
        (loop)]
       #:else
       (set-Clause-candidate?! selected-Clause #false)

       ;; FORWARD REWRITE
       ;; BINARY CLAUSE REWRITE OF SELECTED
       ;; NOTICE: We do binary rewrites first because if we did unit then binary
       ;; we would need to attempt a second unit-rewrite after that.
       ;; (This may lead to unnecessary binary rewrites, but it's cleaner this way.)
       (define selected-Clause-brw
         (if rwtree
             (binary-rewrite-Clause rwtree selected-Clause)
             selected-Clause))

       (when-debug>= steps
                     (printf "|\nstep ~a: selected: ~a\n"
                             step (Clause->string/alone selected-Clause 'all))
                     (define binary-rewritten? (not (eq? selected-Clause-brw selected-Clause)))
                     (when binary-rewritten?
                       (displayln "Binary rewritten:")
                       (display-Clause-ancestor-graph selected-Clause-brw #:depth 1))
                     (unless (eq? selected-Clause-brw selected-Clause-brw)
                       (displayln "Unit rewritten:")
                       (display-Clause-ancestor-graph selected-Clause-brw #:depth 1))
                     (when-debug>= interact
                                   (interact-saturation
                                    (priority utree rwtree selected-Clause make-return-dict)
                                    selected-Clause-brw selected-Clause-brw)))

       (set! selected-Clause selected-Clause-brw)
       ;;; From now on, only selected-Clause should be used

       (define selected-clause (Clause-clause selected-Clause))
       #:cond
       ;; REFUTED?
       [(empty-clause? selected-clause)
        (make-refuted-dict selected-Clause)]
       ;; TAUTOLOGY?
       [(clause-tautology? selected-clause)
        (when-debug>= steps (displayln "Tautology."))
        (discard-Clause! selected-Clause)
        (loop)] ; skip clause
       ;; FORWARD SUBSUMPTION
       [(utree-find/any utree selected-Clause Clause-subsumes)
        ;; TODO: Tests
        =>
        (λ (C2)
          (++ n-forward-subsumed)
          (when-debug>= steps (printf "Subsumed by ~a\n" (Clause->string C2 'all)))
          (discard-Clause! selected-Clause)
          (loop))] ; skip clause
       #:else
       ;; Clause is being processed.

       ;; BACKWARD SUBSUMPTION
       (define removed (utree-inverse-find/remove! utree selected-Clause Clause-subsumes))
       (for-each discard-Clause! removed)
       (+= n-backward-subsumed (length removed))

       (when-debug>= steps
                     (define n-removed (length removed))
                     (when (> n-removed 0)
                       (printf "#backward subsumed: ~a\n" n-removed)
                       (when-debug>= interact
                                     (print-Clauses removed 'all))))

       ;; FIND NEW REWRITE RULES
       (define clause-index-before-discover clause-index)
       (define new-rules (discover-new-rules! rwtree-out selected-Clause utree))
       (define new-rule-Clauses (rules-original-Clauses new-rules))
       ;; NOTICE: We MUST add Clauses that are newly generated to the set of active rules
       ;; (via priority) otherwise we may miss some resolutions.
       ;; Only the Clauses that have been created during the discovery process need to be added.
       ;; Notice: To prevent the clauses from which the rules have originated to be rewritten to
       ;; tautologies, a test is performed in binary-rewrite-literal.
       ;; But this applies *only* to the `eq?`-Clause of the rule, hence beware of copies or
       ;; rewrites.
       (add-priority-Clauses!
        (filter (λ (C) (> (Clause-idx C) clause-index-before-discover))
                new-rule-Clauses))

       ;; BACKWARD BINARY REWRITING
       ;; We don't need to backward rewrite if the new rules are not stored in rwtree,
       ;; as this means the set of used rules does not change during the whole saturation.
       (when (and backward-rewrite?
                  rwtree
                  (eq? rwtree rwtree-out) ; not storing new rules in a different rwtree
                  (not (empty? new-rules)))
         ; Remove active Clauses that can be rewritten, and push them into priority.
         ; We must check whether the clauses we remove will be rewritten,
         ; otherwise we might add all the same candidates again when the removed Clause
         ; is popped from priority.
         (define removed-active-Clauses
           ;; TODO: This is inefficient. We should modify utree-inverse-find/remove!
           ;; TODO: to handle multiple rule-C so as to take advantage of its hash/cache.
           (remove-duplicates
            (flatten
             (for/list ([rule-C (in-list new-rule-Clauses)])
               (utree-inverse-find/remove! utree rule-C
                                           (λ (_rule-C C2)
                                             (binary-rewrite-Clause? rwtree C2)))))
            eq?))
         (unless (empty? removed-active-Clauses)
           (when-debug>= steps
                         (displayln "Some active Clauses can be backward binary rewritten:")
                         (print-Clauses removed-active-Clauses))
           (add-priority-Clauses! removed-active-Clauses)))
       ;; Note that backward-rewritable Clauses are not yet discarded. They may be discarded
       ;; when they are pulled from priority and deemed discardable.

       ;;; Even if the selected Clause is a unit/binary rewrite rule, we must continue processing it
       ;;; and generate resolutions (because rewriting is only left-unification, not full unification)

       ;;; NEW CANDIDATES

       (define L-resolvent-pruning-allowed?
         (and L-resolvent-pruning?
              ; As per the invariant, if no Clause is in the priority queue,
              ; then the set of active Clauses of utree is equisatisfiable with the input clauses.
              (= 0 (queue-length priority))))

       (define new-Candidates
         (if negative-literal-selection?
             (utree-resolve+unsafe-factors/select utree selected-Clause
                                                  #:rewriter (λ (C) (binary-rewrite-Clause rwtree C)))
             (utree-resolve+unsafe-factors utree selected-Clause
                                           #:rewriter (λ (C) (binary-rewrite-Clause rwtree C))
                                           #:L-resolvent-pruning? L-resolvent-pruning-allowed?)))

       (when-debug>= interact
                      (displayln "New candidates:")
                      (print-Clauses new-Candidates))

       ;; If a clause has no resolvent with the active set (when complete)
       ;; then it will never resolve with anything and can thus be discarded.
       #:cond
       [(and L-resolvent-pruning-allowed?
             (empty? new-Candidates))
        (discard-Clause! selected-Clause)
        (when-debug>= steps
                      (printf "No resolvent (L-resolvent-pruning?=~a). Clause discarded. \n"
                              L-resolvent-pruning?))
        (loop)]
       #:else
       ;; ADD CLAUSE TO ACTIVES
       ;; Rewrite the candidates with unit and binary rules, filter out tautologies,
       ;; calculate their costs and add them to the queues.
       (add-candidates! selected-Clause new-Candidates)

       ;; UNIT RULE DISCOVERY IN CANDIDATES
       ;; Look for unit rewrite rules in the candidate set.
       ;; (Looking for binary rules would be too costly here)
       (when find-unit-rules-in-candidates?
         (when rwtree-out
           (for ([C (in-list new-Candidates)])
             (when (unit-Clause? C)
               ;; TODO: Should be calling/merged with discover-rules! to avoid inconsistencies
               (rewrite-tree-add-unit-Clause! rwtree-out C #:rewrite? #false)))))

       (add-Clause! utree selected-Clause)
       (when-debug>= steps
                     (displayln "Adding clause.")
                     (print-active-Clauses utree #false))

       (loop))))

  (when-debug>= interact
                (displayln "Saturation loop finished.")
                (pretty-print result)
                (define selected-Clause #false) ; mock up
                (interact-saturation
                 (priority utree rwtree selected-Clause make-return-dict)))
  result)

;========================;
;=== User interaction ===;
;========================;

;; Some commands to use with '--debug interact'. Type 'help' for information.

(define interact-commands '())

(define-namespace-anchor ns-anchor)
(define-syntax-rule (interact-saturation
                     (priority utree rwtree selected-Clause make-return-dict)
                     more ...)
  (begin
    (define what '(idx parents clause-pretty))
    (interact
     #:command (and (not (empty? interact-commands))
                    (begin0 (first interact-commands)
                            (rest! interact-commands)))
     #:variables (priority utree rwtree what more ...)
     #:namespace-anchor ns-anchor
     #:readline? #true
     [(list 'steps (? number? n))
      "skips n steps"
      (when (> n 0)
        (cons! "" interact-commands)
        (cons! (format "steps ~a" (- n 1)) interact-commands))]
     [(list (or 'ancestors 'ancestor-graph 'graph))
      "display the ancestor graph of the selected Clause."
      (display-Clause-ancestor-graph selected-Clause)]
     [(list (or 'ancestors 'ancestor-graph 'graph) (? number? depth))
      "display the ancestor graph of the selected Clause down to the given depth."
      (display-Clause-ancestor-graph selected-Clause #:depth depth)]
     [(list 'what-fields)
      (string-append
       "Prints which fields are available for 'what,\n"
       "which is used for printing clause information.")
      (displayln Clause->string-all-fields)]
     [(list 'selected)
      "Selected clause"
      (print-Clauses (list selected-Clause) what)]
     [(list 'active)
      "Active clauses"
      (print-active-Clauses utree #true what)]
     [(list (or 'binary 'rules))
      "Found binary rules"
      (print-binary-rules rwtree #true)]
     [(list 'stats)
      "Return-dictionary-like stats"
      (pretty-print (make-return-dict 'running))]
     [(list 'save-rules)
      "Save the binary rules from the default rules-file"
      (save-rules! rwtree #:rules-file (*output-rules*))])))

;; Prints the set of active Clauses (held in utree).
;;
;; utree : unification-tree?
;; long? : boolean?
;; what : (or/c 'all (listof symbol?))
;; -> void?
(define (print-active-Clauses utree long? [what 'all])
  (define actives (unification-tree-Clauses utree))
  (printf "#active clauses: ~a\n" (length actives))
  (when long?
    (displayln "Active clauses:")
    (print-Clauses (sort actives < #:key Clause-idx) what)))

;; Prints the set of binary rules.
;;
;; rewrite-tree? boolean? -> void?
(define (print-binary-rules rwtree long?)
  (define rules (rewrite-tree-rules rwtree))
  (printf "#binary rules: ~a #original clauses: ~a\n"
          (length rules)
          (length (remove-duplicates (map rule-Clause rules) eq?)))
  (when long?
    (display-rules (rewrite-tree-rules rwtree))))

;============================;
;=== Iterative saturation ===;
;============================;

;; A struct holding information about a given input formula.
(struct problem (file name clauses [time-used #:mutable] [last-time #:mutable]))

;; file? (or/ #false string?) (listof clause?) -> problem?
(define (make-problem file name clauses)
  (problem file name clauses 0 0))

;; Returns the same values as body ... but evaluates time-body ... before returning.
;; The result of time-body ... is discarded.
(define-syntax-rule (with-time-result [(cpu real gc) time-body ...] body ...)
  (let-values ([(res cpu real gc) (time-apply (λ () body ...) '())])
    time-body ...
    (apply values res)))

;; Calls saturation for a set of problems in a loop.
;; Tries again each unsolved problem after multiplying the step-limit by step-limit-factor
;; and so on untill all problems are solved.
;; Loading time from files is *not* taken into account.
;;
;; problems : (listof problem?)
;; saturate : procedure?
;; memory-limit : number?
;; cpu-limit : number?
;; cpu-first-limit : number?
;; cpu-limit-factor? : number?
;; -> void?
(define (iterative-saturation/problem-set problems
                                          saturate
                                          #:! memory-limit
                                          #:! cpu-limit         ; in second
                                          #:! cpu-first-limit   ; in seconds
                                          #:! cpu-limit-factor) ; in seconds
  (define n-problems (length problems))
  (define n-attempted 0)
  (define n-solved 0)
  (with-time-result [(cpu real gc)
                     (unless (*quiet-json?*)
                       (printf "; Total time: cpu: ~a real: ~a gc: ~a\n" cpu real gc))]
    (let loop ([problems problems] [iter 0])
      (define n-unsolved (length problems))
      (define n-solved-iter 0)
      (define n-attempted-iter 0)
      (define new-unsolved
        (for/fold ([unsolved '()]
                   [cumu-time 0]
                   #:result (reverse unsolved))
                  ([prob (in-list problems)])
          (define input-clauses (problem-clauses prob))
          (when-debug>= init (for-each (compose displayln clause->string) input-clauses))

          ;; Collecting garbage can take time even when there's nothing to collect,
          ;; and can take a significant proportion of the time when solving is fast,
          ;; hence it's better to trigger GC only if needed.
          (when (>= (current-memory-use-MB) (* 0.8 memory-limit))
            (collect-garbage)
            (collect-garbage))

          ;; Main call
          (define cpu-limit-problem (min (max cpu-first-limit
                                              (* cpu-limit-factor (problem-last-time prob)))
                                         (- cpu-limit (problem-time-used prob))))
          (define res (saturate input-clauses cpu-limit-problem))

          (set! res (append `((name . ,(problem-name prob))
                              (file . ,(problem-file prob)))
                            res))
          (++ n-attempted-iter)
          (when (= 0 iter) (++ n-attempted))
          (define solved? (or (check-status res 'refuted)
                              (check-status res 'saturated)))
          (when solved?
            (++ n-solved-iter)
            (++ n-solved))

          (define last-time (* 0.001 (dict-ref res 'time)))
          (set-problem-last-time! prob last-time)
          (set-problem-time-used! prob (+ (problem-time-used prob) last-time))
          (set! res (dict-set res 'cumulative-time (exact-ceiling (* 1000 (problem-time-used prob)))))

          (define remove-problem?
            (or solved?
                (check-status res 'memory) ; more time won't help if status=memory
                (>= (problem-time-used prob) cpu-limit))) ; cpu exhausted for this problem

          ; Don't pretty-print to keep it on a single line which is simpler for parsing.
          ; Only print the last iteration of a problem for borg.
          (cond
            [(*quiet-json?*)
             (when remove-problem? (displayln (saturation-result->json res)))]
            [else
             (pretty-write res)
             (printf "; ~a/~a solved (iter: ~a/~a/~a success: ~a% avg-time: ~as ETA: ~as)\n"
                     n-solved n-attempted
                     n-solved-iter n-attempted-iter n-unsolved
                     (~r (* 100 (/ n-solved-iter n-attempted-iter)) #:precision '(= 1))
                     (~r (/ cumu-time n-attempted-iter 1000.) #:precision '(= 3))
                     (~r (* (/ cumu-time n-attempted-iter 1000.)
                            (- n-unsolved n-attempted-iter)) #:precision '(= 2)))])
          (flush-output)

          (values
            (if remove-problem?
                unsolved
                (cons prob unsolved))
            (+ cumu-time last-time))))
      (unless (or (empty? new-unsolved)
                  (= cpu-limit-factor 0))
        (loop new-unsolved (+ iter 1))))))

;; Calls saturate on a single set of clauses, first with a time limit of cpu-first-limit,
;; then restarts and doubles it until the cumulative time reaches cpu-limit.
;; Loading time is taken into account.
;; During a call to saturate, the new rewrite rules are saved in a separate tree,
;; which means that no new rule is introduced until the next restart—and thus the first
;; call to saturate uses no rewrite rule.
;;
;; NOTICE: In this mode the unit rewrites are gathered only for the next round, but this is
;; likely not necessary!
;;
;; saturate : procedure?
;; tptp-program : string?
;; rwtree-in : rewrite-tree?
;; discover-online? : boolean?
;; cpu-limit : number?
;; cpu-first-limit : number?
;; cpu-limit-factor? : number?
;; -> void?
(define (iterative-saturation saturate
                              #:! tptp-program
                              #:! rwtree-in
                              #:? [discover-online? (*discover-online?*)]
                              #:? [cpu-limit (*cpu-limit*)]
                              #:? [cpu-first-limit (*cpu-first-limit*)]
                              #:? [cpu-limit-factor (*cpu-limit-factor*)])
  (define cpu-start (current-inexact-seconds))
  ; Don't make new Clauses here, they need to be created at each `saturation` call.
  (define clauses (tptp-prog->clauses tptp-program))
  (define quiet? (*quiet-json?*))

  (define n-rules-init (rewrite-tree-count rwtree-in))

  (let loop ([iter 1] [uncapped-current-cpu-limit cpu-first-limit] [rwtree-in rwtree-in])
    (define remaining-cpu (- cpu-limit (- (current-inexact-seconds) cpu-start)))
    (define current-cpu-limit (min remaining-cpu uncapped-current-cpu-limit))
    (unless quiet?
      (printf "; iter: ~a remaining-cpu: ~a current-cpu-limit: ~a\n"
              iter
              remaining-cpu
              current-cpu-limit))

    ; Simplify the set of rules (only once)
    (unless (and (= 1 iter)
                 (= 0 n-rules-init)) ; don't do this if no restarting
      ; Note that these steps destroy the Clause ancestry, and proofs will be incomplete.
      (unless quiet?
        (printf "; Rules stats: ~v\n" (rewrite-tree-stats rwtree-in))
        (displayln "; Simplifying the rules via re-add-rules!"))

      ;; Rewrite lhs and rhs of rules, remove subsumed and tautologies.
      (re-add-rules! rwtree-in)
      (unless quiet?
        (printf "; Rules stats: ~v\n" (rewrite-tree-stats rwtree-in))
        (printf "; Confluence! bounded? = ~a\n" (*bounded-confluence?*)))

      ;; Unify rhs of rules to produce new rules.
      (rewrite-tree-confluence! rwtree-in)
      (unless quiet?
        (printf "; Rules stats: ~v\n" (rewrite-tree-stats rwtree-in))
        (displayln "; Simplifying the rules via re-add-rules! (again)"))

      ;; Rewrite and simplify again.
      (re-add-rules! rwtree-in)
      (unless quiet? (printf "; Rules stats: ~v\n" (rewrite-tree-stats rwtree-in))))
    (flush-output)

    (define rwtree-out (if discover-online? rwtree-in (rewrite-tree-shallow-copy rwtree-in)))

    (define res (saturate #:clauses clauses
                          #:cpu-limit current-cpu-limit
                          #:rwtree-in rwtree-in
                          #:rwtree-out rwtree-out))

    (define new-cumulative-cpu (- (current-inexact-seconds) cpu-start))
    (set! res (dict-set res 'cumulative-time (exact-ceiling (* 1000. new-cumulative-cpu)))) ; ms
    (set! res (dict-set res 'saturation-iter iter))

    (define solved? (or (check-status res 'refuted)
                        (check-status res 'saturated)))

    ;; We exit also if memory limit has been reached, but we could instead restart
    ;; if new rules have been found.
    (define finished? (or solved?
                          (check-status res 'memory)
                          (> new-cumulative-cpu cpu-limit)))

    (cond
      [(*quiet-json?*)
       (when finished? (displayln (saturation-result->json res)))]
      [else
       (pretty-write res)])
    (flush-output)

    (cond
      [finished?
       (when (*output-rules*)
         (unless quiet?
           (printf "Saving rules to ~a\n"
                   (if (string? (*output-rules*))
                       (*output-rules*)
                       (path->string (*output-rules*)))))
         (save-rules! rwtree-out #:rules-file (*output-rules*)))]
      [else
       (loop (+ iter 1)
             (* uncapped-current-cpu-limit cpu-limit-factor)
             rwtree-out)])))
