#lang racket/base

(require racket/list
         racket/port
         rackunit
         satore/interact)

(define-syntax-rule (check-interact in out args ...)
  (check-equal?
   (with-output-to-string
     (λ ()
       (with-input-from-string (string-append in "\n\n") ; ensure no read loop
         (λ () (interact args ...)))))
   out))

(define-namespace-anchor ns-anchor) ; optional, to use the eval command

(let ([x 2] [y 'a])
  (check-interact
   "x\ny\nx 3\nx"
   "2\n'a\n3\n"
   #:prompt ""
   #:variables (x y)))

(let ([x 3] [y 'a])
  (check-interact
   "yo\nyo 4\nx\nx 2\nx"
   "yo\n(yo yo yo yo)\n3\n2\n"
   #:prompt ""
   #:namespace-anchor ns-anchor
   #:variables (x y)
   ;; All patterns must be of the form (list ....)
   [(list 'yo) "prints yo" (displayln "yo")]
   [(list 'yo (? number? n)) "prints multiple yos" (displayln (make-list n 'yo))]))
