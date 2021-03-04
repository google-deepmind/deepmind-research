#lang racket/base

(require rackunit
         satore/misc)

(define-counter num 0)
(check-equal? num 0)
(++num)
(check-equal? num 1)
(++num 3)
(check-equal? num 4)
(reset-num!)
(check-equal? num 0)
