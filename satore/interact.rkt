#lang racket/base

;***************************************************************************************;
;****                           User Interaction Commands                           ****;
;***************************************************************************************;

(require (for-syntax racket/base syntax/parse)
         racket/format
         racket/list
         racket/match
         racket/port)

(provide (all-defined-out))

;; Notice: variables set via eval or only set locally, in the local namespace,
;; and not in the main namespace.
;; variables set via the (list 'var val) pattern are set in the main namespace.
;; Even though the namespace is at the module level, the variables
;; are set in the namespace with their value so they can be used with eval.
(define-syntax (interact stx)
  (syntax-parse stx
    #:literals (list)
    [(_ (~alt (~optional (~seq #:prompt prompt:expr)) ; must evaluate to a string, default "> "
              (~optional (~seq #:command command:expr))
              (~optional (~seq #:namespace-anchor ns-anchor:expr)) ; default #false
              (~optional (~seq #:variables (var:id ...))) ; must be bound identifiers
              (~optional (~seq #:readline? readline?:expr))) ; start with readline enabled? (#false)
        ...
        [(list pat ...) help-string body ...+] ...) ; match patterns
     (with-syntax ([(var ...) #'(~? (var ...) ())])
       #'(begin
           (define names (list 'var ...))
           (define nsa (~? ns-anchor #false))
           (define ns (and nsa (namespace-anchor->namespace nsa)))
           (when (~? readline? #false) (eval '(require readline) ns))
           (when ns
             (namespace-set-variable-value! 'var var #false ns) ...
             (void)) ; to avoid bad 'when' form if no variable
           (define the-prompt (~? prompt "> "))
           (let loop ()
             (with-handlers ([exn:fail? (λ (e)
                                          (displayln (exn-message e))
                                          (loop))])
               (define cmd (~? command #false))
               (when (and cmd (not (string? cmd)))
                 (error "command must be a string"))
               (unless cmd (display the-prompt))
               (define cmd-str (or cmd (read-line)))
               (unless (eof-object? cmd-str)
                 (define cmd (with-input-from-string (string-append "(" cmd-str ")") read))
                 (match cmd
                   ['() (void)]
                   ['(help)
                    (unless (empty? names)
                      (printf "Available variables: ~a\n" names))
                    (displayln "Other commands:")
                    (parameterize ([print-reader-abbreviations #true]
                                   [print-as-expression #false])
                      (void)
                      (begin
                        (displayln (string-append "  " (apply ~v '(pat ...) #:separator " ")))
                        (displayln (string-append "    " help-string)))
                      ...)
                    (when ns
                      (displayln "  eval expr")
                      (displayln
                       "    Evaluate expr in a namespace that is local to this interaction loop."))
                    (loop)]
                   [(list 'eval cmd)
                    (if ns
                        (call-with-values (λ () (eval cmd ns))
                                          (λ l (if (= 1 (length l))
                                                   (unless (void? (first l))
                                                     (displayln (first l)))
                                                   (for-each displayln l))))
                        (displayln "Cannot use eval without a namespace-anchor"))
                    (loop)]
                   ['(var) (println var) (loop)] ...
                   [(list 'var val) (set! var val) (loop)] ...
                   [(list pat ...) body ... (loop)] ...
                   [else (printf "Unknown command: ~a\n" cmd)
                         (loop)]))))))]))

;; For manual testing in DrRacket
(module+ drracket
  (define-namespace-anchor ns-anchor) ; optional, to use the eval command

  (let ([x 3] [y 'a])
    (interact
     #:prompt ">> "
     #:namespace-anchor ns-anchor
     #:variables (x y)
     ;; All patterns must be of the form (list ....)
     [(list 'yo) "prints yo" (displayln "yo")]
     [(list 'yo (? number? n)) "prints multiple yos" (displayln (make-list n 'yo))])))
