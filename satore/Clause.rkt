#lang racket/base

;**************************************************************************************;
;****            Clause: Clauses With Additional Properties In A Struct            ****;
;**************************************************************************************;

(require define2
         define2/define-wrapper
         global
         racket/format
         racket/list
         racket/string
         satore/clause
         satore/clause-format
         satore/misc
         satore/unification
         text-table)

(provide (all-defined-out))

;==============;
;=== Clause ===;
;==============;

;; TODO: A lot of space is wasted in Clause (boolean flags?)
;; What's the best way to gain space without losing time or readability?

;; parents : (listof Clause?) ; The first parent is the 'mother'.
;; binary-rewrite-rule? : Initiually #false, set to #true if the clause has been added (at some point)
;;   to the binary rewrite rules (but may not be in the set anymore if subsumed).
;; size: tree-size of the clause.
;; depth: Maternal-path depth.
;; cost: Maternal-path cost.
(struct Clause (idx
                parents
                clause
                type
                [binary-rewrite-rule? #:mutable]
                [candidate? #:mutable]
                [discarded? #:mutable]
                n-literals
                size
                depth
                [cost #:mutable]
                [g-cost #:mutable])
  #:prefab)

(define-counter clause-index 0)

(define (make-Clause cl
                     [parents '()]
                     #:type [type '?]
                     #:candidate? [candidate? #false]
                     #:n-literals [n-literals (length cl)]
                     #:size [size (clause-size cl)]
                     #:depth [depth (if (empty? parents) 1 (+ 1 (Clause-depth (first parents))))])
  (++clause-index)
  (when-debug>= steps
                (define cl2 (clause-normalize cl))  ; costly, hence done only in debug mode
                (unless (= (tree-size cl) (tree-size cl2))
                  (displayln "Assertion failed: clause is in normal form")
                  (printf "Clause (type: ~a):\n~a\n" type (clause->string cl))
                  (displayln "Parents:")
                  (print-Clauses parents)
                  (error (format "Assertion failed: (= (tree-size cl) (tree-size cl2)): ~a ~a"
                                 (tree-size cl) (tree-size cl2)))))
  ; Notice: Variables are ASSUMED freshed. Freshing is not performed here.
  (Clause clause-index
          parents
          cl
          type
          #false ; binary-rewrite-rule
          candidate?
          #false ; discarded?
          n-literals
          size
          depth ; depth (C0 is of depth 0, axioms are of depth 1)
          0. ; cost
          0. ; g-cost
          ))

(define (discard-Clause! C) (set-Clause-discarded?! C #true))

(define true-Clause (make-Clause (list ltrue)))

;; For temporary converse Clauses for binary clauses.
(define (make-converse-Clause C #:candidate? [candidate? #false])
  (if (unit-Clause? C)
      true-Clause ; If C has 1 literal A, then C = A | false, and converse is ~A | true = true
      (make-Clause (fresh (clause-converse (Clause-clause C)))
                   (list C)
                   #:type 'converse
                   #:candidate? candidate?
                   )))

(define Clause->string-all-fields '(idx parents clause type binary-rw? depth size cost))

;; If what is a list, each element is printed (possibly multiple times).
;; If what is 'all, all fields are printed.
(define (Clause->list C [what '(idx parents clause)])
  (when (eq? what 'all)
    (set! what Clause->string-all-fields))
  (for/list ([w (in-list what)])
    (case w
      [(idx)            (~a (Clause-idx C))]
      [(parents)        (~a (map Clause-idx (Clause-parents C)))]
      [(clause)         (clause->string (Clause-clause C))]
      [(clause-pretty)  (clause->string/pretty (Clause-clause C))]
      [(type)           (~a (Clause-type C))]
      [(binary-rw?)     (~a (Clause-binary-rewrite-rule? C))]
      [(depth)          (~r (Clause-depth C))]
      [(size)           (~r (Clause-size C))]
      [(cost)           (~r2 (Clause-cost C))])))

(define (Clause->string C [what '(idx parents clause)])
  (string-join (Clause->list C what) " "))

(define (Clause->string/alone C [what '(idx parents clause)])
  (when (eq? what 'all)
    (set! what Clause->string-all-fields))
  (string-join (map (λ (f w) (format "~a: ~a " w f))
                    (Clause->list C what)
                    what)
               " "))

(define (print-Clauses Cs [what '(idx parents clause)])
  (when (eq? what 'all)
    (set! what Clause->string-all-fields))
  (print-simple-table
   (cons what
         (map (λ (C) (Clause->list C what)) Cs))))

;; <=> to avoid hard-to-debug mistakes where Clause-subsumes is used instead of Clause<-subsumes
;; for example.
;; Notice: This is an approximation of the correct subsumption based on multisets, and may not
;; be confluent.
(define (Clause<=>-subsumes C1 C2)
  (clause-subsumes (Clause-clause C1) (Clause-clause C2)))

;; Use atom<=> ?
(define Clause-cmp-key Clause-size)
(define (Clause<= C1 C2) (<= (Clause-cmp-key C1) (Clause-cmp-key C2)))
(define (Clause<  C1 C2) (<  (Clause-cmp-key C1) (Clause-cmp-key C2)))

(define (Clause<=-subsumes C1 C2)
  (and (Clause<= C1 C2)
       (Clause<=>-subsumes C1 C2)))

(define (Clause<-subsumes C1 C2)
  (and (Clause< C1 C2)
       (Clause<=>-subsumes C1 C2)))

;; Useful for rewrite rules
(define (Clause<=>-converse-subsumes C1 C2)
  (clause-subsumes (clause-converse (Clause-clause C1))
                   (Clause-clause C2)))

(define (unit-Clause? C)
  (= 1 (Clause-n-literals C)))

(define (binary-Clause? C)
  (= 2 (Clause-n-literals C)))

(define (Clause-tautology? C)
  (clause-tautology? (Clause-clause C)))

;; Returns the tree of ancestor Clauses of C up to init Clauses,
;; but each Clause appears only once in the tree.
;; (The full tree can be further retrieved from the Clause-parents.)
;; Used for proofs.
(define (Clause-ancestor-graph C #:depth [dmax +inf.0])
  (define h (make-hasheq))
  (let loop ([C C] [depth 0])
    (cond
      [(or (> depth dmax)
           (hash-has-key? h C))
       #false]
      [else
       (hash-set! h C #true)
       (cons C (filter-map (λ (C2) (loop C2 (+ depth 1)))
                           (Clause-parents C)))])))

(define (Clause-ancestor-graph-string C
                                      #:? [depth +inf.0]
                                      #:? [prefix ""]
                                      #:? [tab " "]
                                      #:? [what '(idx parents type clause)])
  (define h (make-hasheq))
  (define str-out "")
  (let loop ([C C] [d 0])
    (unless (or (> d depth)
                (hash-has-key? h C))
      (set! str-out (string-append str-out
                                   prefix
                                   (string-append* (make-list d tab))
                                   (Clause->string C what)
                                   "\n"))
      (hash-set! h C #true)
      (for ([P (in-list (Clause-parents C))])
        (loop P (+ d 1)))))
  str-out)

(define-wrapper (display-Clause-ancestor-graph
                 (Clause-ancestor-graph-string C #:? depth #:? prefix #:? tab #:? what))
  #:call-wrapped call
  (display (call)))

(define (Clause-age<= C1 C2)
  (<= (Clause-idx C1) (Clause-idx C2)))

(define (save-Clauses! Cs f #:? exists)
  (save-clauses! (map Clause-clause Cs) f #:exists exists))

(define (load-Clauses f #:? [sort? #true] #:? [type 'load])
  (define Cs (map (λ (c) (make-Clause c #:type type))
                  (load-clauses f)))
  (if sort?
      (sort Cs Clause<=)
      Cs))

(define (Clause-equivalence? C1 C2)
  (and (Clause<=>-subsumes C1 C2)
       (Clause<=>-subsumes C2 C1)))

;; Provides testing utilities. Use with `(require (submod "Clause.rkt" test))`.
(module+ test
  (require rackunit)
  (provide Clausify
           check-Clause-set-equivalent?)

  (define Clausify (compose make-Clause clausify))

  (define-check (check-Clause-set-equivalent? Cs1 Cs2)
    (unless (= (length Cs1) (length Cs2))
      (fail-check "not ="))
    (for/fold ([Cs2 Cs2])
              ([C1 (in-list Cs1)])
      (define C1b
        (for/first ([C2 (in-list Cs2)] #:when (Clause-equivalence? C1 C2))
          C2))
      (unless C1b
        (printf "Cannot find equivalence Clause for ~a\n" (Clause->string C1))
        (print-Clauses Cs1)
        (print-Clauses Cs2)
        (fail-check))
      (remq C1b Cs2))))
