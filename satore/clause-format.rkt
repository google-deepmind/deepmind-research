#lang racket/base

;***************************************************************************************;
;****                         Clause <-> String Conversions                         ****;
;***************************************************************************************;

;;; In a separate file because of cyclic dependencies with "tptp.rkt" if in "clause.rkt"

(require define2
         racket/format
         racket/list
         racket/pretty
         satore/tptp
         satore/unification
         text-table)

(provide (all-defined-out))

;; Returns a string representation of the clause.
;;
;; clause? -> string?
(define (clause->string cl)
  ((if (*tptp-out?*)
       clause->tptp-string
       ~a)
   (Vars->symbols cl)))

;; Same as clause->string but pretty prints the result for better reading.
;;
;; clause? -> string?
(define (clause->string/pretty cl)
  (pretty-format (Vars->symbols cl)))

;; clause? -> void?
(define (print-clause cl)
  (displayln (clause->string cl)))

;; Prints the list of clauses in a table, possibly sothing them first.
;;
;; cls : (listof clause?)
;; sort? : boolean?
(define (print-clauses cls #:? [sort? #false])
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
