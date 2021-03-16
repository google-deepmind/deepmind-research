#lang racket/base

;***************************************************************************************;
;****                               Various Utilities                               ****;
;***************************************************************************************;

(require (for-syntax racket/base racket/port racket/syntax)
         global
         racket/format
         racket/port)

(provide (all-defined-out))

;; Prints #true and #false
(print-boolean-long-form #true)

(define-syntax-rule (begin-for-both e)
  (begin
    e
    (begin-for-syntax e)))

(begin-for-both
  ;; (or/c string? symbol? number?) -> number?
  (define (debug-level->number lev)
    (cond
      [(number? lev) lev]
      [(string? lev) (debug-level->number (with-input-from-string lev read))]
      [else
       (case lev
         [(none) 0]
         [(init) 1]
         [(step steps) 2]
         [(interact) 3]
         [else (error "unknown debug level" lev)])])))

(define-global *debug-level* 0
  "Number or one of (none=0 init=1 steps interact)."
  exact-nonnegative-integer?
  debug-level->number
  '("--debug"))

;; ;; (or/c string? symbol? number?) -> boolean?
(define (debug>= lev)
  (>= (*debug-level*) (debug-level->number lev)))

;; Do a sequence of actions only when debug-level is greater than a given level.
(define-syntax (when-debug>= stx)
  (syntax-case stx ()
    [(_ lev body ...)
     (with-syntax ([levv (debug-level->number (syntax-e #'lev))])
       #'(when (>= (*debug-level*) levv)
           body ...))]))

;; any/c -> boolean?
(define (thunk? p)
  (and (procedure? p)
       (procedure-arity-includes? p 0)))

;; Defines a counter with a reset function and an increment function.
;; Ex:
;; (define-counter num 0)
;; (++num)
;; (++num 3)
;; (reset-num!)
(define-syntax (define-counter stx)
  (syntax-case stx ()
    [(_ name init)
     (with-syntax ([reset (format-id stx #:source stx "reset-~a!" (syntax-e #'name))]
                   [++ (format-id stx #:source stx "++~a" (syntax-e #'name))])
       #'(begin
           (define name init)
           (define (reset)
             (set! name init))
           (define (++ [n 1])
             (set! name (+ name n)))))]))

;; -> number?
(define (current-inexact-seconds)
  (* 0.001 (current-inexact-milliseconds)))

;; -> exact-nonnegative-integere?
(define (current-memory-use-MB)
  (arithmetic-shift (current-memory-use) -20))

;; Prints `x` with a given precision if it is a rational, otherwise formants it with `~a`.
;;
;; x : any/c
;; precision : (or/c exact-nonnegative-integer? (list/c '= exact-nonnegative-integer?))
(define (~r2 x #:precision [precision 2])
  (if (rational? x)
      (~r x #:precision precision)
      (~a x)))
