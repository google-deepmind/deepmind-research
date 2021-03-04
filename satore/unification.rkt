#lang racket/base

;***************************************************************************************;
;****                Operations On Literals: Unification And Friends                ****;
;***************************************************************************************;

(require bazaar/cond-else
         bazaar/debug
         bazaar/list
         bazaar/mutation
         (except-in bazaar/order atom<=>)
         define2
         global
         racket/dict
         racket/list
         racket/match
         (submod racket/performance-hint begin-encourage-inline)
         satore/misc)

(provide (all-defined-out))

;===============;
;=== Globals ===;
;===============;

(define-global:category *atom-order* 'atom1
  '(atom1 KBO1lex)
  "Atom comparison function for rewrite rules.")

(define (get-atom<=> #:? [atom-order (*atom-order*)])
  (case atom-order
    [(KBO1lex) KBO1lex<=>]
    [(atom1)     atom1<=>]
    [else (error "Unknown atom order: ~a" (*atom-order*))]))

;=================;
;=== Variables ===;
;=================;

(struct Var (name)
  #:prefab)

(begin-encourage-inline
  (define Var-name<?  <)
  (define Var-name=?  eqv?)
  (define Var-name<=> number<=>)

  (define (Var=? v1 v2) (Var-name=? (Var-name v1) (Var-name v2)))
  (define (Var<? v1 v2) (Var-name<? (Var-name v1) (Var-name v2))))
(define (Var<=> v1 v2) (Var-name<=> (Var-name v1) (Var-name v2)))
; (order=? (Var<=> v1 v2)) = (Vars=? v1 v2)

;:::::::::::::::::::::::::::::::::::;
;:: Basic operations on Variables ::;
;:::::::::::::::::::::::::::::::::::;

;; All symbols starting with a capitale letter are considered as variables.
(define (symbol-variable? t)
  (and (symbol? t)
       (char<=? #\A (string-ref (symbol->string t) 0) #\Z)))

;; The same symbol is always mapped to the same Var-name, globally.
(define (symbol->Var-name s)
  (define str (symbol->string s))
  (cond [(regexp-match #px"^X(\\d+)$" str)
         => (λ (m) (+ 26 (string->number (second m))))]
        [(regexp-match #px"^[A-Z]$" str)
         => (λ (m) (- (char->integer (string-ref str 0))
                      (char->integer #\A)))]
        [else
         (error 'Varify "Unknown variable format: ~a" s)]))

;; The same Var-name is always mapped to the same symbol, globally.
(define (Var-name->symbol n)
  (cond [(symbol-variable? n) n]
        [(number? n)
         (if (< n 26)
             (string->symbol (string (integer->char (+ (char->integer #\A) n))))
             (string->symbol (format "X~a" (- n 26))))]
        [else (error 'Var-name->symbol "Don't know what to do with ~a" n)]))

;; Returns a new atom like t where all symbol-variables have been turned into `Var?`s.
(define (Varify t)
  (cond [(pair? t)
         ; Works also in assocs
         (cons (Varify (car t))
               (Varify (cdr t)))]
        [(symbol-variable? t)
         (Var (symbol->Var-name t))]
        [else t]))

;; Returns a new atom like t where all `Var`s are replaced with a symbol.
(define (unVarify t)
  (cond [(pair? t)
         ; Works also in assocs
         (cons (unVarify (car t))
               (unVarify (cdr t)))]
        [(Var? t)
         (Var-name->symbol (Var-name t))]
        [else t]))

(define (rule->string rule)
  (format "~a -> ~a" (first rule) (second rule)))


;====================================;
;=== Substitutions data structure ===;
;====================================;

(define (make-make-hash =?)
  (cond [(eq? =? eq?) make-hasheq]
        [(or (eq? =? =) (eq? =? eqv?)) make-hasheqv]
        [(eq? =? equal?) make-hash]
        [else (error 'make-make-hash "Unknown hash type: ~a" =?)]))

(begin-encourage-inline
  (define make-subst       (make-make-hash Var-name=?))
  (define subst?           hash?)
  (define in-subst         in-hash)
  (define subst-count      hash-count)
  (define subst-ref/name   hash-ref) ; for when the name is retrieved from the subst
  (define subst-set!/name  hash-set!)
  (define subst-copy       hash-copy)

  (define (subst-set! subst var t)
    (hash-set! subst (Var-name var) t)
    subst) ; return the substitution to mimick the immutable update behaviour

  (define (subst-ref subst var [default #false])
    (hash-ref subst (Var-name var) default))

  (define (subst-ref! subst var default)
    (hash-ref! subst (Var-name var) default))

  (define (subst-update! subst var update default)
    (hash-update! subst (Var-name var) update default)
    subst)

  (define (subst->list s)
    (sort (hash->list s) Var-name<? #:key car))
  )

;::::::::::::::::::::::::::::;
;:: Immutable substitution ::;
;::::::::::::::::::::::::::::;

(begin-encourage-inline
  (define (make-imsubst [pairs '()]) pairs)

  (define (imsubst-ref subst var default)
    (define p (assoc (Var-name var) subst Var-name=?))
    (if p (cdr p) default))
  )

;; like dict-set, but possibly faster and with fewer checks
(define (imsubst-set subst var val)
  (define name (Var-name var))
  (let loop ([s subst] [left '()])
    (cond/else
     [(empty? s)
      (cons (cons name val) subst)]
     #:else
     (define p (car s))
     #:cond
     [(Var-name=? (car p) name)
      (rev-append left (cons (cons name val) (cdr s)))]
     #:else
     (loop (cdr s) (cons p left)))))

;===============================;
;=== Operations on Variables ===;
;===============================;

(define fresh-idx 0)
(define (new-Var)
  (++ fresh-idx)
  (Var fresh-idx))

(define-syntax-rule (define-Vars v ...)
  (begin (define v (new-Var)) ...))

;; Renames all variables with fresh names to avoid collisions.
(define (fresh C)
  (define h (make-subst))
  (let loop ([t C])
    (cond [(pair? t)
           (cons (loop (car t)) (loop (cdr t)))]
          [(Var? t)
           (subst-ref! h t new-Var)]
          [else t])))

;; Variables names are mapped to a unique symbol, but the resulting Var-name is unpredictable,
;; and this mapping is guaranteed to be consistent only locally to the term t.
;; Used mostly to turn human-readable expressions into terms, without needing to worry about
;; the actual names of the variables.
(define (symbol-variables->Vars t)
  (define h (make-hasheq))
  (let loop ([t t])
    (cond [(pair? t)
           (cons (loop (car t)) (loop (cdr t)))]
          [(symbol-variable? t)
           (hash-ref! h t new-Var)]
          [else t])))

;; Variables are replaced with symbols by order of appearence. Mostly for ease of reading by humans.
(define (Vars->symbols t)
  (define h (make-subst))
  (define idx -1)
  (let loop ([t t])
    (cond [(pair? t)
           (cons (loop (car t)) (loop (cdr t)))]
          [(Var? t)
           (subst-ref! h t (λ () (++ idx) (Var-name->symbol idx)))]
          [else t])))

;; Returns a subst of the number of occurrences of the variables *names* in the term t.
;; (-> term? subst?)
(define (var-occs t)
  (define h (make-subst))
  (let loop ([t t])
    (cond [(pair? t)
           (loop (car t))
           (loop (cdr t))]
          [(Var? t)
           (subst-update! h t add1 0)]))
  h)

;; Returns the variable names of the term t.
(define (vars t)
  (map car (subst->list (var-occs t))))

;; Returns the variables of the term t.
(define (Vars t)
  (map Var (vars t)))

;; Useful for debugging that two literals have different fresh variables.
(define (common-variables t1 t2)
  (let ([h (var-occs t1)])
    (for/list ([(v n) (in-hash (var-occs t2))]
               #:when (hash-has-key? h v))
      v)))

;; Returns the set of variables *names* that appear in t1 but not in t2.
(define (variables-minus t1 t2)
  (define h2 (var-occs t2))
  (for/list ([(v n) (in-hash (var-occs t1))]
             #:unless (hash-has-key? h2 v))
    v))

;; Returns the lexicographical index of each occurrence of the variables, with a depth-first search.
(define (find-var-names t)
  (define h (make-subst))
  (let loop ([t t] [idx 0])
    (cond [(pair? t)
           (loop (cdr t) (loop (car t) idx))]
          [(Var? t)
           (subst-update! h t min idx)
           (+ idx 1)]
          [else idx]))
  (map car (sort (subst->list h) < #:key cdr)))

;; Returns '< if each variable of t1 appears no more times in t1
;; than the same variable in t2,
;; and at least one variable appears strictly fewer times.
;; Returns '= if the occurrences are equal.
;; Returns #false otherwise.
;; This can be seen as a kind of Pareto dominance.
;; This is used for KBO in particular.
;; Note: (var-occs<=> t1 t2) == (var-occs<=> t2 t1)
;; Note: t1 and t2 may have variables in common if they are two subterms of the same clause.
(define (var-occs<=> t1 t2)
  (define h1 (var-occs t1)) ; assumes does not contain 0s
  (define h2 (var-occs t2)) ; assumes does not contain 0s
  (define n-common 0)
  (define cmp
    (for/fold ([cmp '=])
              ([(v1 n1) (in-subst h1)])
      (define n2 (subst-ref/name h2 v1 0))
      (cond
        [(> n2 0)
         (++ n-common)
         (define c (number<=> n1 n2))
         (cond [(eq? cmp '=) c]
               [(eq? c   '=) cmp]
               [(eq? cmp c)  c]
               [else #false])] ; incomparable
        [else cmp])))

  (define n1 (subst-count h1))
  (define n2 (subst-count h2))
  (cond [(and (< n-common n1)
              (< n-common n2))
         #false]
        [(< n-common n2)
         (case cmp [(< =) '<] [else #false])]
        [(< n-common n1)
         (case cmp [(> =) '>] [else #false])]
        [else cmp]))

;=====================;
;=== Boolean logic ===;
;=====================;

(begin-encourage-inline
  ;; Logical false
  (define lfalse '$false)
  (define (lfalse? x) (eq? lfalse x))
  ;; lfalse must be the bottom element for the various atom orders.
  (define (lfalse<=> a b)
    (define afalse? (lfalse? a))
    (define bfalse? (lfalse? b))
    (cond [(and afalse? bfalse?) '=]
          [afalse? '<]
          [bfalse? '>]
          [else #false]))

  (define ltrue '$true)
  (define (ltrue? x) (eq? x ltrue))

  (define (lnot? lit)
    (and (pair? lit)
         (eq? 'not (car lit))))

  ;; Inverses the polarity of the atom.
  ;; NOTICE: Always use `lnot`, do not construct negated atoms yourself.
  (define (lnot x)
    (cond [(lnot? x) (cadr x)]
          [(lfalse? x) ltrue]
          [(ltrue? x) lfalse]
          [else (list 'not x)]))

  (define (polarity<=> lit1 lit2)
    (boolean<=> (lnot? lit1) (lnot? lit2)))

  )

;; Converse implication clause. Invert polarities if binary.
(define (converse cl)
  (case (length cl)
    [(1) ltrue] ; If A is a fact, then true => A, and thus converse is A => true == true
    [(2) (map lnot cl)]
    [else (error "Undefined converse for ~v" cl)]))

;=================================;
;=== Literals, atoms, terms, … ===;
;=================================;

#|
literal   = atom | (not atom)
atom      = constant | (predicate term ...)
term      = (funtion term ...) | variable | constant
predicate = symbol?
function  = symbol?
constant  = symbol?
variable  = (Var number?)

For simplicity, we sometimes use 'term' to mean 'atom or term', or even
'literal, atom or tem'.
|#

;; Returns the number of nodes in the tree representing the term t (or literal, atom).
(define (tree-size t)
  (let loop ([t t] [s 0])
    (cond [(Var? t) (+ s 1)]
          [(pair? t)
           (loop (cdr t) (loop (car t) s))]
          [else (+ s 1)])))

;; The literals are depolarized first, because negation should not count.
(define (literal-size lit)
  (tree-size (depolarize lit)))

;; In particular, it should be as easy to prove A | B as ~A | ~B, otherwise finding equivalences
;; can be more difficult.
(define (clause-size cl)
  (for/sum ([lit (in-list cl)])
    (literal-size lit)))

;; Returns < if for every substitution α, (atom1<=> t1α t2α) returns <.
;; (Can this be calculated given a base atom1<=> ?)
;; - Rk: variables of t2 that don't appear in t1 are not a problem since they are not instanciated
;;   in t2α.
;; - Equality is loose and is based only on *some* properties of the atoms.
;; - This is a good first comparator, but not good enough (e.g., does not associativity)
;; Notice: (order=? (atom<=> t1 t2)) does NOT necessarily mean that t1 and t2 are syntactically equal.
(define (atom1<=> t1 t2)
  (let ([t1 (depolarize t1)]
        [t2 (depolarize t2)])
    (cond/else
     [(lfalse<=> t1 t2)] ; continue if neither is lfalse
     #:else
     (define size (number<=> (tree-size t1) (tree-size t2)))
     (define vs (var-occs<=> t1 t2))
     #:cond
     [(and (order=? vs) (order=? size)) (or (term-lex2<=> t1 t2) '=)] ; for commutativity
     [(and (order≤? vs) (order≤? size)) '<] ; one is necessarily '<
     [(and (order≥? vs) (order≥? size)) '>]
     #:else #false)))

;; For KBO
;; fun-weight is also for constants, hence it's more like symbol-weight
;; (but the name 'function' is commonly used for constants too).
(define (term-weight t #:? [var-weight 1] #:? [fun-weight (λ (f) 1)])
  (let loop ([t t])
    (cond [(Var? t) var-weight]
          [(symbol? t) (fun-weight t)]
          [(list? t) (for/sum ([s (in-list t)]) (loop s))]
          [else (error "Unknown term ~a" t)])))

;; Knuth-Bendix Ordering, naive version.
;; See "Things to know when implementing KB", Löchner, 2006.
;; var-weight MUST be ≤ to all fun-weights of constants.
;; Simple version for clarity and proximity to the specifications.
;; TODO: Implement a faster version
(define (make-KBO<=> #:? var-weight #:? fun-weight #:? [fun<=> symbol<=>])
  (define (weight t)
    (term-weight t #:var-weight var-weight #:fun-weight fun-weight))

  (define (KBO<=> t1 t2)
    (cond
      [(and (Var? t1) (Var? t2)) (and (Var=? t1 t2) '=)] ; not specified, but surely right?
      [(Var? t1) (and (occurs? t1 t2) '<)]
      [(Var? t2) (and (occurs? t2 t1) '>)]
      [else ; both are fun apps or constants
       (define v (var-occs<=> t1 t2))
       (and v
            (let ([t-cmp (sub-KBO<=> (if (list? t1) t1 (list t1)) ; turn constants into fun apps.
                                     (if (list? t2) t2 (list t2)))])
              (case v
                [(<) (and (order<=? t-cmp) t-cmp)]
                [(>) (and (order>=? t-cmp) t-cmp)]
                [(=) t-cmp])))]))

  ;; t1 and t2 MUST be lists.
  (define (sub-KBO<=> t1 t2)
    (chain-comparisons
     (number<=> (weight t1) (weight t2))
     (fun<=>     (first t1)  (first t2))
     ;; Chain on subterms.
     (<=>map KBO<=> (rest t1) (rest t2))))

  (λ (t1 t2)
    (let ([t1 (depolarize t1)]
          [t2 (depolarize t2)])
      (or (lfalse<=> t1 t2)
          (KBO<=> t1 t2)))))

(define KBO1lex<=> (make-KBO<=>))

;; Returns a literal like lit, but without negation if lit was negative.
(define (depolarize lit)
  (match lit
    [`(not ,x) x]
    [else lit]))

;; Returns the number of arguments of the predicate of the literal lit, after depolarizing it.
(define (literal-arity lit)
  (let ([lit (depolarize lit)])
    (if (list? lit)
        (length lit)
        0)))

;; Returns the name of the predicate (or constant) of the literal.
(define (literal-symbol lit)
  (match lit
    [`(not (,p . ,r)) p]
    [`(not ,a) a]
    [`(,p . ,r) p]
    [else lit]))

;; Lexicographical comparison.
;; Guarantees: (order=? (term-lex<=> t1 t2)) = (term==? t1 t2) (but maybe a slightly slower?)
;; Warning: Doesn't handle variables that are not Var? properly
(define (term-lex<=> t1 t2)
  (cond [(eq? t1 t2) '=] ; takes care of '()
        [(and (pair? t1) (pair? t2))
         (chain-comparisons (term-lex<=> (car t1) (car t2))
                            (term-lex<=> (cdr t1) (cdr t2)))]
        [(pair? t1) '>]
        [(pair? t2) '<]
        [(and (Var? t1) (Var? t2))
         (Var<=> t1 t2)]
        [(Var? t1) '<]
        [(Var? t2) '>]
        [(and (symbol? t1) (symbol? t2))
         (symbol<=> t1 t2)]
        [else
         (error 'term-lex<=> "Unknown term kind for: ~a, ~a" t1 t2)]))

;; Can't compare vars with symbols, or vars with vars. Can only compare ground symbols:
;; A binary rule can't be oriented with variables
(define (term-lex2<=> t1 t2)
  (cond [(eq? t1 t2) '=] ; takes care of '()
        [(and (Var? t1) (Var? t2) (Var=? t1 t2)) '=]
        [(or  (Var? t1) (Var? t2)) #false] ; incomparable, cannot be oriented
        [(and (pair? t1) (pair? t2))
         (chain-comparisons (term-lex2<=> (car t1) (car t2))
                            (term-lex2<=> (cdr t1) (cdr t2)))]
        [(pair? t1) '>]
        [(pair? t2) '<]
        [(and (symbol? t1) (symbol? t2))
         (symbol<=> t1 t2)]
        [else
         (error 'term-lex2<=> "Unknown term kind for: ~a, ~a" t1 t2)]))

;; Depth-first lexicographical order (df-lex)
;; Guarantees: (order=? (literal<=> lit1 lit2)) = (literal==? lit1 lit2). (or it's a bug)
(define (literal<=> lit1 lit2)
  (chain-comparisons
   (polarity<=> lit1 lit2)
   (symbol<=> (literal-symbol lit1) (literal-symbol lit2)) ; A literal cannot be a variable
   (cond [(and (list? lit1) (list? lit2))
          ; this also checks arity
          (<=>map term-lex<=> (rest lit1) (rest lit2))]
         [(list? lit2) '<]
         [(list? lit1) '>]
         [else '=])))

(define (literal<? lit1 lit2)
  (order<? (literal<=> lit1 lit2)))

;; This works because variables are transparent (prefab), hence equal? traverses the struct too.
;; We use `==` to denote syntactic equivalence.
(define term==? equal?)
(define literal==? equal?)

;==================================;
;=== Substitution / Unification ===;
;==================================;

;; Notice: Setting this to #true forces the mgu substitutions to ensure
;; dom(σ)\cap vran(σ) = ø
;; but can be exponentially slow in some rare cases.
;; Also, it's not necessary.
(define reduce-mgu? #false)

(struct already-substed (term) #:prefab)

;; Returns a term where the substitution s is applied to the term t.
(define (substitute t s)
  (define t-orig t)
  (let loop ([t t])
    (cond
      [(null? t) t]
      [(pair? t)
       (cons (loop (car t))
             (loop (cdr t)))]
      [(and (Var? t)
            (subst-ref s t #false))
       ; Recur into the substitution.
       ; This avoids recurring many times inside the same substitution.
       =>
       (λ (rhs)
         (cond [(already-substed? rhs)
                ; No need to loop inside the new term.
                (already-substed-term rhs)]
               [else
                (define new-rhs (loop rhs))
                (subst-set! s t (already-substed new-rhs))
                new-rhs]))]
      [else t])))

;; v : variable name (symbol)
;; t : term
(define (occurs? V t)
  (cond [(Var? t) (Var=? V t)]
        [(pair? t)
         (or (occurs? V (car t))
             (occurs? V (cdr t)))]
        [else #false]))

(define (occurs?/extend var t2 subst)
  (define t2c (substitute t2 subst))
  (if (occurs? var t2c)
    #false
    (begin
      (subst-set! subst var t2c)
      subst)))

;; Returns one most general unifier α such that t1α = t2α.
;; Assumes that the variables of t1 and t2 are disjoint,
;; so that occur-check is not needed. (really?)
;; TODO: In case left-unification is possible, does this return a left-unifier?
;;   Can we also return a boolean saying whether this is the case?
(define (unify t1 t2 [subst (make-subst)])
  (define success?
    (let loop ([t1 t1] [t2 t2])
      (cond/else
       [(eq? t1 t2) ; takes care of both null?
        subst]
       [(and (pair? t1) (pair? t2))
        (and (loop (car t1) (car t2))
             (loop (cdr t1) (cdr t2)))]
       #:else
       (define v1? (Var? t1))
       (define v2? (Var? t2))
       #:cond
       [(and (not v1?) (not v2?)) ; since they are not `eq?`
        #false]
       [(and v1? v2? (Var=? t1 t2)) ; since at least one is a Var
        ; Same variable, no need to substitute, and should not fail occurs?/extend.
        subst]
       #:else
       (define t1b (and v1? (subst-ref subst t1 #false)))
       (define t2b (and v2? (subst-ref subst t2 #false)))
       #:cond
       [(or t1b t2b)
        ; rec
        (loop (or t1b t1) (or t2b t2))]
       [v1? ; t2 may also be a variable
        (occurs?/extend t1 t2 subst)]
       [v2? ; v2? but not v1?
        (occurs?/extend t2 t1 subst)]
       #:else (void))))
  ; Make sure we return a most general unifier
  ; NOTICE: This can take a lot of time (see strest tests), but may prevent issues too.
  (and success?
       (if reduce-mgu?
           (let ([s2 (make-subst)])
             (for ([(k v) (in-subst subst)])
               (subst-set!/name s2 k (substitute v subst)))
             s2)
           subst)))

(define (subst/#false->imsubst s)
  (cond [(subst? s)
         (subst->list s)]
        [(list? s)
         (sort (map (λ (p) (cons (Var-name (Varify (car p)))
                                 (Varify (cdr p))))
                    s)
               Var-name<? #:key car)]
        [else s]))

;; Creates a procedure that returns the substitution α such that t1α = t2, of #false if none exists.
;; t2 is assumed to not contain any variable of t1.
;; Also known as matching
;; - The optional argument is useful to chain left-unify over several literals, say.
;; - Works with both mutable and immutable substitutions.
;; NOTICE:
;; The found substitution must be specializing, that is C2σ = C2 (and C1σ = C2),
;; otherwise safe factoring can fail, in particular.
;; Hence we must ensure that vars(C2) ∩ dom(σ) = ø.
;; TODO: Find a literature reference for these definitions!
(define-syntax-rule
  (define-left-subst+unify left-substitute left-unify make-subst subst-ref subst-set)
  (begin
    ;; t: term?
    ;; s: subst?
    (define (left-substitute t s)
      (let loop ([t t])
        (cond
          [(null? t) t]
          [(pair? t)
           (cons (loop (car t))
                 (loop (cdr t)))]
          [(and (Var? t)
                (subst-ref s t #false))]
          [else t])))
    ;; t1: term?
    ;; t2: term?
    ;; subst: subst?
    (define (left-unify t1 t2 [subst (make-subst)])
      (cond/else
       [(eq? t1 t2) ; takes care of both null?
        subst]
       [(and (pair? t1) (pair? t2))
        (define new-subst (left-unify (car t1) (car t2) subst))
        (and new-subst
             (left-unify (cdr t1) (cdr t2) new-subst))]
       [(term==? t1 t2) subst] ; *** WARNING: This is costly
       [(not (Var? t1)) #false]
       #:else
       (define t1b (subst-ref subst t1 #false))
       #:cond
       [t1b (and (term==? t1b t2) subst)]
       ; This ensures that vars(C2) ∩ dom(σ) = ø:
       ; if var, t1 must not occur in rhs of subst
       ; and any lhs of subst and t1 must not occur in t2
       [(or (occurs? t1 t2)
            (for/or ([(var-name val) (in-dict subst)])
              (or (occurs? t1 val)
                  (occurs? (Var var-name) t2))))
        #false]
       #:else
       (subst-set subst t1 t2)))))


(define-left-subst+unify left-substitute left-unify make-subst subst-ref subst-set!)
;; Immutable substitutions
(define-left-subst+unify left-substitute/assoc left-unify/assoc make-imsubst imsubst-ref imsubst-set)

;; Like traditional pattern matching, but using left-unify
(define (left-unify-anywhere pat t)
  (let loop ([t t])
    (cond [(left-unify pat t)]
          [(list? t) (ormap loop t)]
          [else #false])))

; Could also use match:
(define (match-anywhere filt term)
  (let loop ([t term])
    (cond [(filt t)]
          [(list? t) (ormap loop (rest t))]
          [else #false])))
