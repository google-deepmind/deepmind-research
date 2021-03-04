#lang racket/base

;***************************************************************************************;
;****                      Trie: Imperfect Discrimination Tree                      ****;
;***************************************************************************************;

;;; A discrimination tree is like a hashtable where the key is a list of elements.
;;; The keys are organized in a tree structure so that to retrieving an object
;;; takes at most O(A×l) steps, where l is the length of the key and A is the size of
;;; the alphabet. In practice it will be closer to O(l) since a hash table is used
;;; at each node to store the branches.
;;;
;;; A key is a actually tree (a list of lists of ...), which is flattened to a list
;;; where parenthesis are replaced with symbols.
;;; Variables are considered to be unnamed and there is no unification/matching.
;;; The only dependency on first-order logic specifics is `variable?`.
;;;
;;; An imperfect discrimination tree does not differentiate variable names.
;;; Hence p(X Y) is stored in the same node as p(A A). An additional tests
;;; is required to tell them apart.

(require bazaar/cond-else
         define2
         racket/list
         racket/match
         satore/misc)

(provide (except-out (all-defined-out) no-value)
         (rename-out [no-value trie-no-value]))

;; Default value at the leaves. Should not be visible to the user.
(define no-value       (string->uninterned-symbol "no-value"))
; Tokens used in the keys of the tree
(define anyvar         (string->uninterned-symbol "¿"))
(define sublist-begin  (string->uninterned-symbol "<<"))
(define sublist-end    (string->uninterned-symbol ">>"))

;; edges : hasheq(key . node?)
;; value : any/c
(struct trie-node (edges value)
  #:transparent
  #:mutable)
(define (make-node)
  (trie-node (make-hasheq) no-value))

;; Trie structure with variables.
;;
;; root : trie-node?
;; variable? : any/c -> boolean?
(struct trie (root variable?))

;; Trie constructor.
;;
;; constructor : procedure?
;; variable? : any/c -> boolean?
;; other-args  : (listof any/c)
;; -> trie?
(define (make-trie #:? [constructor trie]
                   #:? [variable? (λ (x) #false)]
                   . other-args)
  (apply constructor (make-node) variable? other-args))

;; Updates the value of the node for the given key (or sets one if none exists).
;; If default-val/proc is a procedure of arity 0, then it is applied to produce the
;; default value when requested, otherwise default-val/proc is used itself as the
;; default value.
;;
;; atrie : trie?
;; key : any/c
;; update : any/c -> any/c
;; default-val/proc : (or/c thunk? any/c)
;; -> void?
(define (trie-update! atrie key update default-val/proc)
  (match-define (trie root variable?) atrie)
  ; The key is `list`ed because we need a list, and this allows the given key to not be a list.
  (let node-insert! ([nd root] [key (list key)])
    (cond/else
     [(empty? key)
      ; Stop here.
      (define old-value (trie-node-value nd))
      (set-trie-node-value! nd (update (if (eq? old-value no-value)
                                           (if (thunk? default-val/proc)
                                               (default-val/proc)
                                               default-val/proc)
                                           old-value)))]
     #:else ; key is a list
     (define k (car key))
     (define edges (trie-node-edges nd))
     #:cond
     [(pair? k)
      ; Linearize the tree structure of the key.
      (define key2 (cons sublist-begin (append k (cons sublist-end (cdr key)))))
      (node-insert! nd key2)]
     #:else ; nil, atom, variable
     (let ([k (if (variable? k) anyvar k)])
       (define nd2 (hash-ref! edges k make-node))
       (node-insert! nd2 (cdr key))))))

;; Keeps a list of values at the leaves.
;; If `trie-insert!` is used, any use of `trie-update!` should be consistent with values being lists.
;;
;; atrie : trie?
;; key : any/c
;; val : any/c
;; -> void?
(define (trie-insert! atrie key val)
  (trie-update! atrie key (λ (old) (cons val old)) '()))

;; Replacing the current value (if any) for key with val.
;;
;; atrie : trie?
;; key : any/c
;; val : any/C
(define (trie-set! atrie key val)
  (trie-update! atrie key (λ _ val) #false))

;; Applies on-leaf at each node that match with key.
;; The matching keys of the trie are necessarily no less general than the given key.
;; `on-leaf` may be effectful.
;;
;; atrie : trie?
;; key : any/c
;; on-leaf : trie-node? -> any
;; -> void?
(define (trie-find atrie key on-leaf)
  (define variable? (trie-variable? atrie))
  (let node-ref ([nd (trie-root atrie)] [key (list key)])
    (cond/else
     [(empty? key)
      ; Leaf found.
      (unless (eq? no-value (trie-node-value nd))
        (on-leaf nd))]
     #:else
     (define k (car key))
     (define var-nd (hash-ref (trie-node-edges nd) anyvar #false))
     #:cond
     [(variable? k)
      (when var-nd
        ; both the key and the node are variables
        (node-ref var-nd (cdr key)))]
     #:else
     (when var-nd
       ; If a variable matches, consider the two paths.
       (node-ref var-nd (cdr key)))
     #:cond
     [(pair? k)
      ; Linearize the tree structure of the key.
      (define key2 (cons sublist-begin (append k (cons sublist-end (cdr key)))))
      (node-ref nd key2)]
     #:else
     (define nd2 (hash-ref (trie-node-edges nd) k #false))
     (when nd2
       (node-ref nd2 (cdr key))))))

;; Applies the procedure `on-leaf` to any node for which the key is matched by the given key.
;; The matching keys of the trie are necessarily no more general than the given key.
;; `on-leaf` may be effectful.
;;
;; atrie : trie?
;; key : any/c
;; on-leaf : trie-node -> any/c
;; -> void?
(define (trie-inverse-find atrie key on-leaf)
  (define variable? (trie-variable? atrie))
  (let node-find ([nd (trie-root atrie)] [key (list key)] [depth 0])
    (define edges (trie-node-edges nd))
    (cond/else
     [(> depth 0)
      ; If the depth is positive, that means we are currently matching a variable.
      ; We need to continue through every branch and decrease the depth only if we encounter
      ; a sublist-end, and increase the counter if we encounter a sublist-begin.
      ; Note that key can be empty while depth > 0.
      (for([(k2 nd2) (in-hash edges)])
        (node-find nd2 key
                  (cond [(eq? k2 sublist-begin) (+ depth 1)]
                        [(eq? k2 sublist-end) (- depth 1)]
                        [else depth])))]
     [(empty? key)
      ; Leaf found.
      (unless (eq? no-value (trie-node-value nd))
        (on-leaf nd))]
     #:else
     (define k (car key))
     #:cond
     [(variable? k)
      ;; Anything matches. For sublist we need to keep track of the depth.
      ;; Note that variables in the tree can only be matched if k is a variable.
      (for ([(k2 nd2) (in-hash edges)])
        (node-find nd2 (cdr key) (if (eq? k2 sublist-begin) 1 0)))]
     [(pair? k)
      ; Linearize the tree structure of the key.
      (define key2 (cons sublist-begin (append k (cons sublist-end (cdr key)))))
      (node-find nd key2 0)]
     #:else
     (define nd2 (hash-ref edges k #false))
     (when nd2
       (node-find nd2 (cdr key) 0)))))

;; Both find and inverse-find at the same time.
;; Useful when (full) unification must be performed afterwards.
;; `on-leaf` may be effectful.
;;
;; atrie : trie?
;; key : any/c
;; on-leaf : trie-node? -> any
;; -> void?
(define (trie-both-find atrie key on-leaf)
  (define variable? (trie-variable? atrie))
  (let node-find ([nd (trie-root atrie)] [key (list key)] [depth 0])
    (define edges (trie-node-edges nd))
    (cond/else
     [(> depth 0)
      ; If the depth is positive, that means we are currently matching a variable.
      ; Consume everything until we find a sublist-end at depth 1.
      ; We need to continue through every branch and decrease the depth only if we encounter
      ; a sublist-end, and increase the counter if we encounter a sublist-begin.
      ; Note that key can be empty while depth > 0.
      (for([(k2 nd2) (in-hash edges)])
        (node-find nd2 key
                  (cond [(eq? k2 sublist-begin) (+ depth 1)]
                        [(eq? k2 sublist-end) (- depth 1)]
                        [else depth])))]
     [(empty? key)
      ; Leaf found.
      (unless (eq? no-value (trie-node-value nd))
        (on-leaf nd))]
     #:else
     (define k (car key))
     (define var-nd (hash-ref (trie-node-edges nd) anyvar #false))
     #:cond
     [(variable? k)
      ;; Anything matches. For sublist we need to keep track of the depth.
      ;; Note that variables in the tree can only be matched if k is a variable.
      (for ([(k2 nd2) (in-hash edges)])
        (node-find nd2 (cdr key) (if (eq? k2 sublist-begin) 1 0)))]
     #:else
     (when var-nd
      ; The node contains a variable, which thus matches the key.
       (node-find var-nd (cdr key) 0))
     #:cond
     [(pair? k)
      ; Linearize the tree structure of the key.
      (define key2 (cons sublist-begin (append k (cons sublist-end (cdr key)))))
      (node-find nd key2 0)]
     #:else
     (define nd2 (hash-ref edges k #false))
     (when nd2
       (node-find nd2 (cdr key) 0)))))

;; Helper function
(define ((make-proc-tree-ref proc) atrie key)
  (define res '())
  (proc atrie
        key
        (λ (nd) (set! res (cons (trie-node-value nd) res))))
  res)

;; Returns a list of values which keys are matched by the given key.
;; The matching keys of the trie are necessarily no more general than the given key.
;; These functions do not have side effects.
;;
;; Each function takes as input:
;; atrie : trie?
;; key : any/c
;; -> list?
(define trie-inverse-ref (make-proc-tree-ref trie-inverse-find))
(define trie-ref         (make-proc-tree-ref trie-find))
(define trie-both-ref    (make-proc-tree-ref trie-both-find))

;; Returns the list of all values in all nodes.
;;
;; atrie : trie?
;; -> list?
(define (trie-values atrie)
  (let loop ([nd (trie-root atrie)] [res '()])
    (define edges (trie-node-edges nd))
    (define val (trie-node-value nd))
    (for/fold ([res (if (eq? val no-value) res (cons val res))])
              ([(key nd2) (in-hash edges)])
      (loop nd2 res))))
