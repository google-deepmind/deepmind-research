#lang racket/base

(require racket/dict
         rackunit
         satore/clause
         satore/misc
         satore/unification)

(*subsumes-iter-limit* 0)

(begin
  (define-simple-check (check-tautology cl res)
    (check-equal? (clause-tautology? (sort-clause (Varify cl))) res))

  (check-tautology '[] #false)
  (check-tautology `[,ltrue] #true)
  (check-tautology `[,(lnot lfalse)] #true)
  (check-tautology '[a] #false)
  (check-tautology '[a a] #false)
  (check-tautology '[a (not a)] #true)
  (check-tautology '[a b (not c)] #false)
  (check-tautology '[a b (not a)] #true)
  (check-tautology '[a (not (a a)) (a b) (not (a (not a)))] #false)
  (check-tautology '[a (a a) b c (not (a a))] #true)
  (check-tautology `[(a b) b (not (b a)) (not (b b)) (not (a c)) (not (a ,(Var 'b)))] #false)
  )

(begin
  ;; Equivalences
  (for ([(A B) (in-dict '(([] . [] ) ; if empty clause #true, everything is #true
                          ([p] . [p] )
                          ([(p X)] . [(p X)] )
                          ([(p X)] . [(p Y)] )
                          ([(not (p X))] . [(not (p X))] )
                          ([(p X) (q X)] . [(p X) (q X) (q Y)] )
                          ))])
    (define cl1 (sort-clause (Varify A)))
    (define cl2 (sort-clause (fresh (Varify B))))
    (check-not-false (clause-subsumes cl1 cl2)
                     (format "cl1: ~a\ncl2: ~a" cl1 cl2))
    (check-not-false (clause-subsumes cl2 cl1)
                     (format "cl1: ~a\ncl2: ~a" cl1 cl2))
    )

  ;; One-way implication (not equivalence)
  (for ([(A B) (in-dict '(([] . [p] ) ; if empty clause #true, everything is #true
                          ([p] . [p q] )
                          ([(p X)] . [(p c)] )
                          ([(p X) (p X) (p Y)] . [(p c)] )
                          ([(p X)] . [(p X) (q X)] )
                          ([(p X)] . [(p X) (q Y)] )
                          ([(p X Y)] . [(p X X)] )
                          ([(p X) (q Y)] . [(p X) (p Y) (q Y)] )
                          ([(p X) (p Y) (q Y)] . [(p Y) (q Y) c] )
                          ([(p X Y) (p Y X)] . [(p X X)] )
                          ([(q X X) (q X Y) (q Y Z)]  . [(q a a) (q b b)])
                          ([(f (q X)) (p X)] . [(p c) (f (q c))])
                          ; A θ-subsumes B, but does not θ-subsume it 'strictly'
                          ([(p X Y) (p Y X)] . [(p X X) (r)])
                          ))])
    (define cl1 (sort-clause (Varify A)))
    (define cl2 (sort-clause (fresh (Varify B))))
    (check-not-false (clause-subsumes cl1 cl2))
    (check-false (clause-subsumes cl2 cl1)))

  ; Not implications, both ways. Actually, this is independence
  (for ([(A B) (in-dict '(([p] . [q])
                          ([(p X)] . [(q X)])
                          ([p] . [(not p)])
                          ([(p X c)] . [(p d Y)])
                          ([(p X) (q X)] . [(p c)])
                          ([(p X) (f (q X))] . [(p c)])
                          ([(eq X X)] . [(eq (mul X0 X1) (mul X2 X3))
                                         (not (eq X0 X2)) (not (eq X1 X3))])
                          ; A implies B, but there is no θ-subsumption
                          ; https://www.doc.ic.ac.uk/~kb/MACTHINGS/SLIDES/2013Notes/6LSub4up13.pdf
                          ([(p (f X)) (not (p X))] . [(p (f (f Y))) (not (p Y))])
                          ))])
    (define cl1 (sort-clause (Varify A)))
    (define cl2 (sort-clause (fresh (Varify B))))
    (check-false (clause-subsumes cl1 cl2)
                 (list (list 'A= A) (list 'B= B)))
    (check-false (clause-subsumes cl2 cl1)
                 (list A B)))

  (let* ()
    (define cl
      (Varify
       `((not (incident X Y))
         (not (incident ab Y))
         (not (incident ab Z))
         (not (incident ab Z))
         (not (incident ac Y))
         (not (incident ac Z))
         (not (incident ac Z))
         (not (incident bc a1b1))
         (not (line_equal Z Z))
         (not (point_equal bc X)))))
    (define cl2
      (sort-clause (fresh (left-substitute cl (hasheq (symbol->Var-name 'X) 'bc
                                                      (symbol->Var-name 'Y) 'a1b1)))))
    (check-not-false (clause-subsumes cl cl2))))

#;
(begin
  ; This case SHOULD pass, according to the standard definition of clause subsumption based on
  ; multisets, but our current definition of subsumption is more general (not necessarily in a
  ; good way.)
  ; Our definition is based on sets, with a constraint on the number of literals (in
  ; Clause-subsumes).
  ; This makes it more general, but also not well-founded (though I'm not sure yet whether this is
  ; really bad).
  (check-false (clause-subsumes (clausify '[(p A A) (q X Y) (q Y Z)])
                                (clausify '[(p a a) (p b b) (q C C)]))))


(begin

  (*debug-level* (debug-level->number 'steps))

  (define-simple-check (check-safe-factoring cl res)
    (define got (safe-factoring (sort-clause (Varify cl))))
    (set! res (sort-clause (Varify res)))
    ; Check equivalence
    (check-not-false (clause-subsumes res got))
    (check-not-false (clause-subsumes got res)))

  (check-safe-factoring '[(p a b) (p A B)]
                        '[(p a b)]) ; Note that [(p a b) (p A B)] ≠> (p A B)
  (check-safe-factoring '[(p X) (p Y)]
                        '[(p Y)])
  (check-safe-factoring '[(p Y) (p Y)]
                        '[(p Y)])
  (check-safe-factoring '[(p X) (q X) (p Y) (q Y)]
                        '[(p Y) (q Y)])
  (check-safe-factoring '[(p X Y) (p A X)]
                        '[(p X Y) (p A X)])
  (check-safe-factoring '[(p X Y) (p X X)]
                        '[(p X X)]) ; is a subset of above, so necessarily no less general
  (check-safe-factoring '[(p X Y) (p A X) (p Y A)]
                        '[(p X Y) (p A X) (p Y A)]) ; cannot be safely factored?
  (check-safe-factoring '[(p X) (p Y) (q X Y)]
                        '[(p X) (p Y) (q X Y)]) ; Cannot be safely factored (proven)
  (check-safe-factoring '[(leq B A) (leq A B) (not (def B)) (not (def A))]
                        '[(leq B A) (leq A B) (not (def B)) (not (def A))]) ; no safe factor
  (check-safe-factoring '[(p X) (p (f X))]
                        '[(p X) (p (f X))])

  (check-safe-factoring
   (fresh '((not (incident #s(Var 5343) #s(Var 5344)))
            (not (incident ab #s(Var 5344)))
            (not (incident ab #s(Var 5345)))
            (not (incident ab #s(Var 5345)))
            (not (incident ac #s(Var 5344)))
            (not (incident ac #s(Var 5345)))
            (not (incident ac #s(Var 5345)))
            (not (incident bc a1b1))
            (not (line_equal #s(Var 5345) #s(Var 5345)))
            (not (point_equal bc #s(Var 5343)))))
   (fresh
    '((not (incident #s(Var 148) #s(Var 149)))
      (not (incident ab #s(Var 149)))
      (not (incident ab #s(Var 150)))
      (not (incident ac #s(Var 149)))
      (not (incident ac #s(Var 150)))
      (not (incident bc a1b1))
      (not (line_equal #s(Var 150) #s(Var 150)))
      (not (point_equal bc #s(Var 148))))))

  (check-not-exn (λ () (safe-factoring
                        (fresh '((not (incident #s(Var 5343) #s(Var 5344)))
                                 (not (incident ab #s(Var 5344)))
                                 (not (incident ab #s(Var 5345)))
                                 (not (incident ab #s(Var 5345)))
                                 (not (incident ac #s(Var 5344)))
                                 (not (incident ac #s(Var 5345)))
                                 (not (incident ac #s(Var 5345)))
                                 (not (incident bc a1b1))
                                 (not (line_equal #s(Var 5345) #s(Var 5345)))
                                 (not (point_equal bc #s(Var 5343))))))))
  )
