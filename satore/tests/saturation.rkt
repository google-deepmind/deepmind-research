#lang racket/base

(require (for-syntax racket/base)
         define2
         define2/define-wrapper
         global
         racket/dict
         racket/pretty
         rackunit
         (prefix-in sat: satore/saturation)
         satore/clause
         satore/misc
         satore/rewrite-tree
         satore/unification
         syntax/parse/define)

(define-global *cpu-limit* 10
  "Time limit in seconds for tests"
  number?
  string->number)

(define (Vars+clausify-list l)
  (map clausify
       (symbol-variables->Vars l)))

(define current-saturation-args #false)

(define-syntax (for-in-list* stx)
  (syntax-parse stx
      [(_ ([var x ...] ... clauses ...) body ...)
       #:with (name ...) (generate-temporaries #'(var ...))
       #'(for ((~@ [var (in-list (list x ...))]
                   [name (in-list '(x ...))]
                   #:when #true)
               ...)
           (set! current-saturation-args (list (cons 'var name) ...))
           body ...)]))

;; Print additional information
(define old-check-handler (current-check-handler))
(current-check-handler
 (λ (e)
   (eprintf (pretty-format current-saturation-args))
   (eprintf "\n")
   (old-check-handler e)))

;; USE THIS FOR DEBUGGING
(define-simple-macro (replay-on-failure body ...)
  (let ([old-check-handler (current-check-handler)])
    (parameterize ([current-check-handler
                    (λ (e)
                      (old-check-handler e)
                      (eprintf
                       "Some checks have failed. Replaying in interactive mode for debugging.\n")
                      (*debug-level* 3)
                      (*cpu-limit* +inf.0)
                      (let () body ...))])
      ; encapsulated to avoid collisions
      (let () body ...))))

(for-in-list* ([neg-lit-select? #true #false]
               [l-res-pruning? #true #false]
               [atom<=> #false KBO1lex<=> atom1<=>]
               [dynamic-ok? #;#true #false]
               [parent-discard? #false]
               ; notice: if parent-discard? is #true, then can't have both rewriting and
               ; neg-lit-select.
               )

  (define-wrapper (saturation
                   (sat:saturation input-clauses
                                   #:? [step-limit 200]
                                   #:? [memory-limit 4096] ; in MB
                                   #:? [cpu-limit (*cpu-limit*)] ; in seconds
                                   #:? [rwtree (make-rewrite-tree #:atom<=> atom<=>
                                                                  #:dynamic-ok? dynamic-ok?)]
                                   #:? [rwtree-out (and atom<=> rwtree)]
                                   #:? backward-rewrite?
                                   #:? age:cost
                                   #:? cost-type
                                   #:? [parent-discard? parent-discard?]
                                   #:? [disp-proof? #false]
                                   #:? [L-resolvent-pruning? l-res-pruning?]
                                   #:? [negative-literal-selection? neg-lit-select?]))
    #:call-wrapped call
    (define res (call))
    (unless l-res-pruning?
      (check-equal? (dict-ref res 'L-resolvent-pruning) 0))
    (unless dynamic-ok?
      (check-equal? (dict-ref res 'binary-rules-dynamic) 0))
    (unless atom<=>
      (check-equal? (dict-ref res 'binary-rules) 0)
      (check-true (= (dict-ref res 'binary-rewrites) 0)))
    res)


  ;; Some refutation tests
  (check-equal?
   (dict-ref (saturation (Vars+clausify-list '( [] )))
             'status)
   'refuted)
  (check-equal?
   (dict-ref (saturation (Vars+clausify-list '( [p] )))
             'status)
   'saturated)
  (check-equal?
   (dict-ref (saturation (Vars+clausify-list '( [p]
                                                [(not p)])))
             'status)
   'refuted)
  (check-equal?
   (dict-ref (saturation (Vars+clausify-list '( [p]
                                                [(not q)])))
             'status)
   'saturated)

  ;; To do: If L-resolvents-pruning applied to input clauses too,
  ;; it would discard the 2nd clause immediately and would saturate.
  (replay-on-failure
   (check-equal?
    (dict-ref (saturation (Vars+clausify-list '( [(p z)]
                                                 [(not (p X)) (p (s X))]
                                                 [(not q)])))
              'status)
    'steps))



  ;; Russell's 'paradox', requires factoring:
  (check-equal?
   (dict-ref (saturation (Vars+clausify-list
                          '( [(s X X) (s b X)]
                             [(not (s X X)) (not (s b X))])))
             'status)
   'refuted)

  ;; Second version
  (check-equal?
   (dict-ref (saturation (Vars+clausify-list
                          '( [(s X b) (s b X)]
                             [(not (s X b)) (not (s b X))])))
             'status)
   'refuted)



  (check-equal?
   (dict-ref (saturation (Vars+clausify-list
                          '( [(big_f T0_0 T0_1) (big_g T0_0 T0_2)]
                             [(big_f T1_0 T1_1) (not (big_g T1_0 T1_0))]
                             [(big_g T2_0 T2_1) (not (big_f T2_0 T2_2))]
                             [(not (big_f T3_0 T3_1)) (not (big_g T3_0 (esk1_1 T3_0)))])))
             'status)
   'refuted)


  (check-equal?
   (dict-ref (saturation (Vars+clausify-list '( [p1 p2]
                                                [p1 (not p2)]
                                                [p2 (not p1)]
                                                [(not p1) (not p2)])))
             'status)
   'refuted)

  (check-equal?
   (dict-ref (saturation
              '((p1 p2 p3)
                (p1 p3 (not p2))
                (p2 p3 (not p1))
                (p1 p2 (not p3))
                (p1 (not p2) (not p3))
                (p2 (not p1) (not p3))
                (p3 (not p1) (not p2))
                ((not p1) (not p2) (not p3))))
             'status)
   'refuted)


  (check-equal?
   (dict-ref (saturation
              (Vars+clausify-list
               '( [(big_f T0_0 T0_1 (esk3_2 T0_0 T0_1))]
                  [(big_f esk1_0 esk2_0 esk2_0) (not (big_f esk1_0 esk1_0 esk2_0))]
                  [(big_f esk1_0 esk1_0 esk2_0)
                   (big_f esk1_0 esk2_0 esk2_0)
                   (not (big_f esk2_0 esk2_0 T2_0))]
                  [(big_f esk1_0 esk1_0 esk2_0)
                   (big_f esk2_0 T3_0 T3_1)
                   (not (big_f esk1_0 esk2_0 esk2_0))]
                  [(not (big_f esk1_0 esk1_0 esk2_0))
                   (not (big_f esk1_0 esk2_0 esk2_0))
                   (not (big_f esk2_0 esk2_0 T4_0))]
                  [(big_f esk1_0 esk2_0 esk2_0)
                   (not (big_f T5_0 T5_1 (esk3_2 T5_1 T5_0)))
                   (not (big_f esk1_0 esk1_0 esk2_0))]
                  [(big_f esk1_0 esk1_0 esk2_0)
                   (not (big_f T6_0 (esk3_2 T6_0 T6_1) (esk3_2 T6_0 T6_1)))
                   (not (big_f esk1_0 esk2_0 esk2_0))] )))
             'status)
   'refuted)

  (check-equal?
   (dict-ref
    (saturation
     (Vars+clausify-list
      '([(p X) (not (p (p X)))]
        [(not (p a))]
        [(not (q a))]
        [(q X) (not (q (q (q X))))]
        [(q (q (q (q (q (q (q (q (q (q (q (q (q (q (q a)))))))))))))))])))
    'status)
   'refuted)
  ;; This problem shows there may be some loops with implication-removal and factoring!
  (check-equal?
   (dict-ref
    (saturation
     (Vars+clausify-list
      '([(not (p X Y)) (p X Z) (p Z Y)]
        [(p x x)]
        [(not (q a a a a b b b b c c c c))]
        [(q A A A A B B B B C C C C) (not (q (q A A A A) (q B B B B) (q C C C C)))]
        [(q (q a a a a) (q b b b b) (q c c c c))])))
    'status)
   'refuted)

  ;; Binary rewrite
  ;; This example shows that *not* backward rewriting rules can be a problem:
  ;; Around step 19, there should be immediate resolution to '() with an active clause.
  ;; But because [(not (p a A))] has not been rewritten to [(notp a A)],
  ;; it cannot unify to '() immediately, and must wait for a *resolution* between
  ;; the rule and the clause to pop up from the queue.
  (replay-on-failure
    (define res
      (saturation
       (map clausify
            '(((notp A B) (p A B)) ; axiom, binary clause
              ((not (notp A B)) (not (p A B))) ; axiom, converse binary clause
              ((p a A) (q b B) (r c C) (s d D)) ; these two clauses should resolve to '() immediately
              ((not (p A a))) ; Note that 'a A' is to prevent unit-clause rewrites
              ((not (q B b)))
              ((not (r C c)))
              ((not (s D d)))
              ))))
    (check-equal? (dict-ref res 'status) 'refuted)
    (when atom<=>
      (check > (dict-ref res 'unit-rules) 0)
      (check-equal? (dict-ref res 'binary-rules) 2)
      (check > (dict-ref res 'binary-rewrites) 0)))

  ;; 'Asymmetric' rules
  (replay-on-failure
    (define res
      (saturation
       (map clausify
            '([(not (p A A)) (q A)] ; Not a rule in itself (too general), but enables the next ones
              [(p a a) (not (q a))] ; rule (p a a) <-> (q a)
              [(p b b) (not (q b))] ; rule (p b b) <-> (q b)
              [(p a a) (p b b) (p c c)]
              [(not (q a)) (remove-me x Y)]
              [(not (q b)) (remove-me x Y)]
              [(not (p c c)) (remove-me x Y)]
              [(not (remove-me X y))] ; defeats urw
              ))))
    (check-equal? (dict-ref res 'status) 'refuted)
    (when atom<=>
      (check-equal? (dict-ref res 'binary-rules) 4)
      (check-true (> (dict-ref res 'binary-rewrites) 0))))
  ;; TODO: Same test but with rules loaded from a file

  ;; Greedy selection of binary rewrites can lead to failure
  (replay-on-failure
    (define res
      (saturation
       (map clausify
            '(; equivalences
              [(not (q A B C D)) (p A B C)] ; (q A B C D) <=> (p A B C)
              [(q A B C D) (not (p A B C))]
              [(not (p A b C)) (t a)]       ; (p A b C) <=> (t a)
              [(p A b C) (not (t a))]
              [(not (q A B c D)) (s b c)]   ; (q A b c D) <=> (s b c)
              [(q A B c D) (not (s b c))]
              ; inputs
              ; may be rewritten to (s b c)
              [(q a b c d) (remove-me x Y) (remove-me y Y) (remove-me z Y)]
              [(not (t a)) (remove-me x Y) (remove-me y Y) (remove-me z Y)]
              ;
              [(not (remove-me X y))] ; defeats urw
              ))))
    (check-equal? (dict-ref res 'status) 'refuted)
    (when atom<=>
      (check-equal? (dict-ref res 'binary-rules) 6)
      (check-true (> (dict-ref res 'binary-rewrites) 0))))

  ;; Overlapping rewrites can lead to failures (but not without rewrites)
  (replay-on-failure
    (define res
      (saturation
       (map clausify
            '(; equivalences
              [(not (q A B C D)) (p A B C)] ; (q A B C D) <=> (p A B C)
              [(q A B C D) (not(p A B C))]
              [(not (p A b C)) (t a)]       ; (p A b C) <=> (t a)
              [(p A b C) (not (t a))]
              [(not (q A b c D)) (s b c)]   ; (q A b c D) <=> (s b c)
              [(q A b c D) (not (s b c))]
              ; inputs
              [(s b c) (remove-me x Y) (remove-me y Y) (remove-me z Y)]
              [(not (t a)) (remove-me x Y) (remove-me y Y) (remove-me z Y)]
              ;
              [(not (remove-me X y))] ; defeats urw
              ))))
    (check-equal? (dict-ref res 'status) 'refuted)
    (when atom<=>
      (check-equal? (dict-ref res 'binary-rules) 6)
      (check-true (> (dict-ref res 'binary-rewrites) 0))))

  ;; Greedy selection of overlapping rewrites can lead to failures
  (replay-on-failure
    (define res
      (saturation
       (map clausify
            '(; equivalences
              [(not (q A B C D)) (p A B C)] ; (q A B C D) <=> (p A B C)
              [(q A B C D) (not(p A B C))]
              [(not (p A b C)) (r A C)]     ; (p A b C) <=> (r A C)
              [(p A b C) (not (r A C))]
              [(not (r A c)) (t a)]         ; (r A c) <=> (t a)
              [(r A c) (not (t a))]
              [(not (q A b c D)) (s b c)]   ; (q A b c D) <=> (s b c)
              [(q A b c D) (not (s b c))]
              ; inputs
              [(s b c) (remove-me x Y) (remove-me y Y) (remove-me z Y)]
              [(not (t a)) (remove-me x Y) (remove-me y Y) (remove-me z Y)]
              ;
              [(not (remove-me X y))] ; defeats urw
              ))))
    (check-equal? (dict-ref res 'status) 'refuted)
    (when atom<=>
      (check-equal? (dict-ref res 'binary-rules) 8)
      (check-true (> (dict-ref res 'binary-rewrites) 0)))))
