#lang racket/base

;***************************************************************************************;
;****                                  Json Output                                  ****;
;***************************************************************************************;

(require racket/dict
         racket/string)

(provide (all-defined-out))

;; Correspondance with Eprover output values
(define status-dict
  '((running   . UNSPECIFIED_PROOF_STATUS)
    (refuted   . REFUTATION_FOUND)
    (time      . TIME_LIMIT_REACHED)
    (memory    . MEMORY_LIMIT_REACHED)
    (steps     . STEP_LIMIT_REACHED)
    (saturated . COUNTER_SATISFIABLE)))

;; Take a result dictionary from `saturation` and returns a JSON string representation of it.
;;
;; dict? -> string?
(define (saturation-result->json res)
  (define d
    (let* ([res (dict-remove res 'name)]
           [res (dict-remove res 'file)]
           [res (dict-update res 'status (λ (v) (dict-ref status-dict v)))])
      res))
  (string-join
   #:before-first "{\n  "
   (for/list ([(k v) (in-dict d)])
     (define kstr (regexp-replace* #px"-|:" (symbol->string k) "_"))
     (format "~s: ~s" kstr (if (symbol? v) (symbol->string v) v)))
   ",\n  "
   #:after-last "\n}"))

;; Simple visual test.
(module+ drracket
  (define res
  '((name . "GEO170+1.p")
    (file . "data/tptp_geo/GEO170+1.p")
    (status . refuted)
    (steps . 205)
    (generated . 3186)
    (actives . 106)
    (candidates . 2651)
    (priority-remaining . 0)
    (tautologies . 156)
    (rules . 30)
    (unit-rules . 24)
    (binary-rules . 6)
    (binary-rules-static . 0)
    (binary-rules-dynamic . 6)
    (binary-rewrites . 164)
    (forward-subsumed . 96)
    (backward-subsumed . 0)
    (subsumes-checks . 7654)
    (subsumes-steps . 13268)
    (subsumes-breaks . 0)
    (L-resolvent-pruning . 0)
    (memory . 181509744)
    (time . 196)
    (proof-length . 12)
    (proof-inferences . 5)
    (proof-type:in . 7)
    (proof-type:res . 4)
    (proof-type:rw . 1)))
  (displayln (saturation-result->json res)))
