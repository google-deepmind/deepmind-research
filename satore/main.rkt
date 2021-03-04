#!/usr/bin/env racket
#lang racket/base

;**************************************************************************************;
;****                                    Satore                                    ****;
;**************************************************************************************;

;;; Try:
;;;   racket -l- satore --help
;;; to see all the available flags.

(module+ main
  (require global
           racket/file
           racket/port
           satore/misc
           satore/rewrite-tree
           satore/saturation
           satore/unification)

  (define-global *prog* #false
    '("Data file containing a single TPTP program."
      "If not provided, reads from the input port.")
    file-exists?
    values
    '("-p"))

  ;; If -p is not specified, reads from current-input-port
  (void (globals->command-line #:program "satore"))

  ;; No validation here yet.
  (define program
    (if (*prog*)
        (file->string (*prog*))
        (port->string)))

  (iterative-saturation
   (λ (#:clauses input-clauses #:cpu-limit cpu-limit #:rwtree-in rwtree-in #:rwtree-out rwtree-out)
     (saturation input-clauses
                 #:cpu-limit cpu-limit
                 #:rwtree rwtree-in
                 #:rwtree-out rwtree-out))
   #:tptp-program program
   #:rwtree-in (make-rewrite-tree #:atom<=> (get-atom<=>)
                                  #:dynamic-ok? (*dynamic-rules?*)
                                  #:rules-file (*input-rules*))))
