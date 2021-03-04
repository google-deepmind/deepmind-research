#lang racket/base

;***************************************************************************************;
;****                         Clause <-> String Conversions                         ****;
;***************************************************************************************;

;;; In a separate file because of cyclic dependencies with "tptp.rkt" if in "clause.rkt"

(require racket/format
         racket/list
         racket/pretty
         satore/clause
         satore/tptp
         satore/unification
         text-table)

(provide (all-defined-out))

(define (clause->string cl)
  ((if (*tptp-out?*)
       clause->tptp-string
       ~a)
   (Vars->symbols cl)))

(define (clause->string/pretty cl)
  (pretty-format (Vars->symbols cl)))

(define (print-clause cl)
  (displayln (clause->string cl)))

(define (print-clauses cls #:sort? [sort? #false])
  (unless (empty? cls)
    (print-table
     (for/list ([cl (in-list (if sort?
                                 (sort cls < #:key tree-size #:cache-keys? #true)
                                 cls))]
                [i (in-naturals)])
       (cons i (Vars->symbols cl)))
     #:border-style 'space
     #:row-sep? #false
     #:framed? #false)))
