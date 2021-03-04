#lang racket/base

;**************************************************************************************;
;****                           Tptp Input/Output Format                           ****;
;**************************************************************************************;

(require bazaar/debug
         bazaar/string
         global
         racket/dict
         racket/file
         racket/format
         racket/list
         racket/match
         racket/port
         racket/string
         satore/clause
         satore/unification)

(provide (all-defined-out))

(define-global:boolean *tptp-out?* #false
  "Output is in TPTP format?")

#|
File formats:
  .rktd: Racket data, one Racket clause per line.
  .p: Prolog format, with Prolog clauses that contain tptp (FOL) clauses.
  .tptp: only the tptp clauses, one per line.

|#

;; Reads a .p file and returns a list of clauses.
;;
;; program-file : file?
;; -> (listof clause?)
(define (tptp-program-file->clauses program-file)
  ; Not efficient: Loads the whole program as a string then parses it.
  ; It would be more efficient to read it as a stream with an actual parser.
  ; Another possibility is to read it line by line and parse each line as a cnf(…)
  ; but that will file if the cnf(…) is multiline.
  (tptp-prog->clauses (file->string program-file)))

;; Helper function
(define (tptp-pre-clauses->clauses pre-clauses)
  (define clauses
    (for/list ([cl (in-list pre-clauses)])
      (let loop ([t cl])
        (match t
          [(? symbol? x) x]
          [(? string? x)
           (string->symbol (string-append "_str_" x))] ; to avoid being interpreted as a variable
          ['() '()]
          [(list '~ (? symbol? pred) (list a ...) r ...)
           (cons (list 'not (cons (loop pred) (loop a)))
                 (loop r))]
          [(list  (? symbol? pred) (list a ...) r ...)
           (cons (cons (loop pred) (loop a))
                 (loop r))]
          [(list '~ x r ...)
           (cons (list 'not (loop x))
                 (loop r))]
          [(list x a ...)
           (cons (loop x) (loop a))]
          [else (error "Unrecognized token: " t)]))))
  (map (compose clausify symbol-variables->Vars) clauses))

;; Reads the .p program given as a string and returns a list of clauses.
;;
;; str : string?
;; -> (listof clause?)
(define (tptp-prog->clauses str)

  ; hardly tested and not strict enough
  ; It should be mostly robust to line breaking though.
  ; Doesn't parse strings properly (will remove lines that look like comments in multiline strings)
  (define l
    (filter
     (λ (x)
       (if (list? x)
         x
         (begin
           (assert (memq x '(cnf end_cnf))
                   x)
           #false)))
     ; Ensure operators are surrounded with spaces
     ; turn racket special symbols (| and ,) into normal symbols.
     ; then use racket's reader to parse it like an s-expression
     (string->data
      (regexp-replaces
       str
       (list*
        '[#px"(?:^|\n)\\s*[%#][^\n]*" "\n"] ; prolog and shell/python/eprover full-line comments
        '[#px"\\bnot\\b" "_not_"] ;; To do: Use $not for `lnot` instead? (as in TPTP)
        (map (λ (p) (list (regexp-quote (first p))
                          (string-append " " (regexp-replace-quote (second p)) " ")))
             '(["|" "" ]
               ["&" "" ]
               ["," "" ]
               ["$false" ""] ; empty literal
               ["~" "~"]
               ["." "end_cnf"]
               ["'" "\""])))))))
  ; first is name, second is type, third is clause, rest is comments(?)
  (define pre-clauses (map third l))
  (tptp-pre-clauses->clauses pre-clauses))


;; Simple parser for the proposer output into s-exp clauses.
;; The format is expected to be in cnf.
;;
;; str : string?
;; -> (listof clause?)
(define (tptp-string->clauses str)
  ; To do: Optimize. This can be very slow for large conjectures.
  (define pre-clauses
    (append*
     ; split first to avoid regenerating the whole string after each substitution?
     (for/list ([str (in-list (string-split str #px"&|\n"))]) ; & and \n play the same role
       (with-handlers ([exn? (λ (e) (displayln str) (raise e))])
         (string->data
          ; Ensure operators are surrounded with spaces
          ; turn racket special symbols (| and ,) into normal symbols
          (regexp-replaces
           str
           (list*
            '[#px"\\bnot\\b" "_not_"] ;; To do: use $not for `lnot` instead? (as TPTP)
            (map (λ (p) (list (regexp-quote (first p))
                              (string-append " " (regexp-replace-quote (second p)) " ")))
                 '(["|" ""]
                   ["," ""]
                   ["~" "~"]
                   ["'" "\""])))))))))
  (tptp-pre-clauses->clauses pre-clauses))

;; Returns a string representing the literal lit.
;;
;; lit : literal?
;; -> string?
(define (literal->tptp-string lit)
    (cond
      [(lnot? lit)
       (string-append "~ " (literal->tptp-string (second lit)))]
      [(empty? lit)
       "$false"]
      [(list? lit)
       (string-append (literal->tptp-string (first lit))
                      "("
                      (string-join (map literal->tptp-string (rest lit)) ", ")
                      ")")]
      [(Var? lit) (symbol->string (Var-name->symbol lit))]
      [else (format "~a" lit)]))

;; Returns a string representing the clause cl.
;;
;; cl : clause?
;; ->string?
(define (clause->tptp-string cl)
  (string-join
   (map literal->tptp-string (Vars->symbols cl))
   " | "))

;; Returns a string representing the clauses cls.
;;
;; cls : (listof clause?)
;; -> string?
(define (clauses->tptp-string cls)
  (string-join (map clause->tptp-string cls) "\n"))

;; String replacement of tptp names with shorter ones to improve readability
;;
;; str : string?
;; -> string?
(define (tptp-shortener str)
  (define substs
    (sort
     (map (λ (p) (cons (~a (car p)) (~a (cdr p))))
          ; fld_1
          (append
          '((multiplicative_identity . _1)
            (additive_identity . _0)
            (less_or_equal . ≤)
            (additive_inverse . –)
            (multiplicative_inverse . /)
            (equalish . ≃)
            (multiply . ×)
            (product . ×=)
            (inverse . /)
            (add . +)
            )
          ;grp_5
          '((equalish . ≃)
            (multiply . ×)
            (product . ×=)
            (inverse . /)
            (identity . _1)
            )
          ; geo
          '((convergent_lines . /\\)
            (unorthogonal_lines . ¬⊥)
            (orthogonal_through_point . ⊥_thru_pt)
            (parallel_through_point . //_thru_pt)
            (distinct_lines . ≠_ln)
            (apart_point_and_line . ≠_pt_ln)
            (orthogonal_lines . ⊥)
            (distinct_points . ≠_pt)
            (parallel_lines . //)
            (equal_lines . =_ln)
            (equal_points . =_pt)))
          )
     ; forces prefixes to appear later to match longer strings first:
     > #:key (compose string-length car)))


  (string-join
   (for/list ([line (in-lines (open-input-string str))])
     (for/fold ([line line])
               ([(from to) (in-dict substs)])
       (string-replace line from to #:all? #true)))
   "\n"))

;; Helper: Surround any printing operation with this macro
;; to automatically replace the output with shortened names.
(define-syntax-rule (with-tptp-shortener body ...)
  (let ([str (with-output-to-string (λ () body ...))])
    (displayln (tptp-shortener str))))

;============;
;=== Main ===;
;============;

(module+ main
  (require global
           racket/file)

  (define-global *rktd-file* #false
    "file in rktd format to output in tptp format"
    file-exists?
    values)

  (void (globals->command-line))

  (when (*rktd-file*)
    (displayln (clauses->tptp-string (file->list (*rktd-file*))))))
