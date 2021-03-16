#lang racket/base

(require rackunit
         satore/trie
         (only-in satore/unification symbol-variable?))

(let ([atrie (make-trie #:variable? symbol-variable?)])
  (trie-set! atrie
             '(a X (f Y) c)
             'A)
  (trie-set! atrie
             '(a b (f Y) c)
             'B)
  (trie-set! atrie
             '(a b (f Y) E)
             'C)
  (check-equal?
   (sort (trie-ref atrie '(a b (f (g e)) c)) symbol<?)
   '(A B C))
  (check-equal? (trie-ref atrie '(a Y (f (g e)) c))
                '(A))
  (check-equal? (trie-ref atrie '(a Y (f (g Y)) c))
                '(A))
  (check-equal? (trie-ref atrie '(a Y (f (g Y)) Z))
                '())
  (check-equal? (trie-ref atrie '(a b (f Y) (g e)))
                '(C))
  (check-equal? (trie-ref atrie '(a (f (g Y)) (f Y) c))
                '(A))

  (check-equal? (sort (trie-values atrie) symbol<?)
                '(A B C))
  (check-equal? (trie-ref atrie '(X X X X)) '())
  (check-equal? (sort (trie-inverse-ref atrie '(X X X X)) symbol<?)
                '(A B C))
  (check-equal? (trie-inverse-ref atrie '(a b (f e) c)) '())

  (check-equal? (sort (trie-both-ref atrie '(a Y (f c) c)) symbol<?)
                '(A B C))
  (check-equal? (sort (trie-both-ref atrie '(a e (f (g X)) c)) symbol<?)
                '(A))
  (check-equal? (sort (trie-both-ref atrie '(a b (f c) d)) symbol<?)
                '(C)))

(let ([atrie (make-trie #:variable? symbol-variable?)])
  (trie-set! atrie
             '(eq X0 X1)
             'A)
  (trie-set! atrie
             '(eq X0 (mul X1 one))
             'B)
  (trie-set! atrie
             '(eq X0 (mul X1 X0))
             'C)
  (check-equal? (trie-ref atrie '(eq X Y))
                '(A))
  (check-equal?
   (sort (trie-ref atrie '(eq Y (mul Y one))) symbol<?)
   '(A B C)))

;; Trie traversal
(let ([atrie (make-trie #:variable? symbol-variable?)])
  (trie-set! atrie
             '(a X (f Y) c)
             'A)
  (trie-set! atrie
             '(a b (f Y) c)
             'B)
  (trie-set! atrie
             '(a b (f Y) E)
             'C)
  (trie-set! atrie 'X 'D)
  (trie-set! atrie 'abc 'E)
  (trie-set! atrie '(a B c) 'F)
  (trie-set! atrie '(a B c d) 'G)
  (trie-set! atrie '(a B) 'H)
  (trie-set! atrie '() 'I)
  (check-equal? (sort (trie-inverse-ref atrie 'A) symbol<?)
                '(A B C D E F G H I)))
