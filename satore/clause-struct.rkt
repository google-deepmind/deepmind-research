#lang racket/base

;**************************************************************************************;
;****            Clause: Clauses With Additional Properties In A Struct            ****;
;**************************************************************************************;

(require define2
         define2/define-wrapper
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
;; TODO: What's the best way to gain space without losing time or readability?

;; idx : exact-nonnegative-integer? ; unique id of the Clause.
;; parents : (listof Clause?) ; The first parent is the 'mother'.
;; clause : clause? ; the list of literals.
;; type : symbol? ; How the Clause was generated (loaded from file, input clause, rewrite, resolution,
;;   factor, etc.)
;; binary-rewrite-rule? : boolean? ; Initially #false, set to #true if the clause has been added
;;   (at some point) to the binary rewrite rules (but may not be in the set anymore if subsumed).
;; candidate? : boolean? ; Whether the clause is currently a candidate (see `saturation` in
;;   saturation.rkt).
;; discarded? : boolean? ; whether the Clause has been discarded (see `saturation` in saturation.rkt).
;; n-literals : exact-nonnegative-integer? ; number of literals in the clause.
;; size : number? ; tree-size of the clause.
;; depth : exact-nonnegative-integer? : Number of parents up to the input clauses, when following
;;   resolutions and factorings.
;; cost : number? ; Used to sort Clauses in `saturation` (in saturation.rkt).
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
                [cost #:mutable])
  #:prefab)

;; Unique clause index. No two Clauses should have the same index.
(define-counter clause-index 0)

;; Clause constructor. See the struct Clause for more information.
;;
;; parents : (listof Clause?)
;; candidate? : boolean?
;; n-literals : exact-nonnegative-integer?
;; size : number?
;; depth : exact-nonnegative-integer?
(define (make-Clause cl
                     [parents '()]
                     #:? [type '?]
                     #:? [candidate? #false]
                     #:? [n-literals (length cl)]
                     #:? [size (clause-size cl)]
                     #:? [depth (if (empty? parents) 1 (+ 1 (Clause-depth (first parents))))])
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
          ))

;; Sets the Clause as discarded. Used in `saturation`.
;;
;; Clause? -> void?
(define (discard-Clause! C) (set-Clause-discarded?! C #true))

;; A tautological clause used for as parent of the converse of a unit Clause.
(define true-Clause (make-Clause (list ltrue)))

;; Returns a  converse Clause of a unit or binary Clause.
;; These are meant to be temporary.
;;
;; C : Clause?
;; candidate? : boolean?
;; -> Clause?
(define (make-converse-Clause C #:? [candidate? #false])
  (if (unit-Clause? C)
      true-Clause ; If C has 1 literal A, then C = A | false, and converse is ~A | true = true
      (make-Clause (fresh (clause-converse (Clause-clause C)))
                   (list C)
                   #:type 'converse
                   #:candidate? candidate?)))

;; List of possible fields for output formatting.
(define Clause->string-all-fields '(idx parents clause type binary-rw? depth size cost))

;; Returns a tree representation of the Clause, for human reading.
;; If what is a list, each element is printed (possibly multiple times).
;; If what is 'all, all fields are printed.
;;
;; Clause? (or/c 'all (listof symbol?)) -> list?
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

;; Returns a string representation of a Clause.
;;
;; Clause? (or/c 'all (listof symbol?)) -> string?
(define (Clause->string C [what '(idx parents clause)])
  (string-join (Clause->list C what) " "))

;; Returns a string representation of a Clause, for displaying a single Clause.
;;
;; Clause? (listof symbol?) -> string?
(define (Clause->string/alone C [what '(idx parents clause)])
  (when (eq? what 'all)
    (set! what Clause->string-all-fields))
  (string-join (map (λ (f w) (format "~a: ~a " w f))
                    (Clause->list C what)
                    what)
               " "))

;; Outputs the Clauses `Cs` in a table for human reading.
;;
;; (listof Clause?) (or/c 'all (listof symbol?)) -> void?
(define (print-Clauses Cs [what '(idx parents clause)])
  (when (eq? what 'all)
    (set! what Clause->string-all-fields))
  (print-simple-table
   (cons what
         (map (λ (C) (Clause->list C what)) Cs))))

;; Returns a substitution if C1 subsumes C2 and the number of literals of C1 is no larger
;; than that of C2, #false otherwise.
;; Indeed, even when the clauses are safely factored, there can still be issues, for example,
;; this prevents cases infinite chains such as:
;; p(A, A) subsumed by p(A, B) | p(B, A) subsumed by p(A, B) | p(B, C) | p(C, A) subsumed by…
;; Notice: This is an approximation of the correct subsumption based on multisets.
;;
;; Clause? Clause? -> (or/c #false subst?)
(define (Clause-subsumes C1 C2)
  (and (<= (Clause-n-literals C1) (Clause-n-literals C2))
       (clause-subsumes (Clause-clause C1) (Clause-clause C2))))

;; Like Clause-subsumes but first takes the converse of C1.
;; Useful for rewrite rules.
;;
;; Clause? Clause? -> (or/c #false subst?)
(define (Clause-converse-subsumes C1 C2)
  (and (<= (Clause-n-literals C1) (Clause-n-literals C2))
       (clause-subsumes (clause-converse (Clause-clause C1))
                        (Clause-clause C2))))

;; Clause? -> boolean?
(define (unit-Clause? C)
  (= 1 (Clause-n-literals C)))

;; Clause? -> boolean?
(define (binary-Clause? C)
  (= 2 (Clause-n-literals C)))

;; Clause? -> boolean?
(define (Clause-tautology? C)
  (clause-tautology? (Clause-clause C)))

;; Returns whether C1 and C2 are α-equivalences, that is,
;; if there exists a renaming substitution α such that C1α = C2
;; and C2α⁻¹ = C1.
;;
;; Clause? Clause? -> boolean?
(define (Clause-equivalence? C1 C2)
  (and (Clause-subsumes C1 C2)
       (Clause-subsumes C2 C1)))

;================;
;=== Printing ===;
;================;

;; Returns the tree of ancestor Clauses of C up to init Clauses,
;; but each Clause appears only once in the tree.
;; (The full tree can be further retrieved from the Clause-parents.)
;; Used for proofs.
;;
;; C : Clause?
;; dmax : number?
;; -> (treeof Clause?)
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

;; Like `Clause-ancestor-graph` but represented as a string for printing.
;;
;; C : Clause?
;; prefix : string? ; a prefix before each line
;; tab : string? ; tabulation string to show the tree-like structure
;; what : (or/c 'all (listof symbol?)) ; see `Clause->string`
;; -> string?
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

;; Like `Clause-ancestor-graph-string` but directly outputs it.
(define-wrapper (display-Clause-ancestor-graph
                 (Clause-ancestor-graph-string C #:? depth #:? prefix #:? tab #:? what))
  #:call-wrapped call
  (display (call)))

;; Returns #true if C1 was generated before C2
;;
;; Clause? Clause? -> boolean?
(define (Clause-age>= C1 C2)
  (<= (Clause-idx C1) (Clause-idx C2)))


;=================;
;=== Save/load ===;
;=================;

;; Saves the Clauses `Cs` to the file `f`.
;;
;; Cs : (listof Clause?)
;; f : file?
;; exists : symbol? ; see `with-output-to-file`
;; -> void?
(define (save-Clauses! Cs f #:? exists)
  (save-clauses! (map Clause-clause Cs) f #:exists exists))

;; Loads Clauses from a file. If `sort?` is not #false, Clauses are sorted by Clause-size.
;; The type defaults to `'load` and can be changed with `type`.
;;
;; f : file?
;; sort? : boolean?
;; type : symbol?
;; -> (listof Clause?)
(define (load-Clauses f #:? [sort? #true] #:? [type 'load])
  (define Cs (map (λ (c) (make-Clause c #:type type))
                  (load-clauses f)))
  (if sort?
      (sort Cs <= #:key Clause-size)
      Cs))

;======================;
;=== Test utilities ===;
;======================;

;; Provides testing utilities. Use with `(require (submod satore/Clause test))`.
(module+ test
  (require rackunit)
  (provide Clausify
           check-Clause-set-equivalent?)

  ;; Takes a symbol tree, turns symbol variables into actual `Var`s, freshes them,
  ;; sorts the literals and makes a new Clause.
  ;;
  ;; tree? -> Clause?
  (define Clausify (compose make-Clause clausify))

  ;; Returns whether for every clause of Cs1 there is an α-equivalent clause in Cs2.
  ;;
  ;; (listof Clause?)  (listof Clause?) -> any/c
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
