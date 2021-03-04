#lang racket/base

;***************************************************************************************;
;****             Logging To File With Consistent Debugging Information             ****;
;***************************************************************************************;

(require bazaar/date
         bazaar/debug
         define2
         global
         racket/file
         racket/port
         racket/pretty
         racket/string
         racket/system)

(provide call-with-log
         *log*)

(define-global:boolean *log* #false
  "Output to a log file?")

(define-global:boolean *git?* #false
  "Commit to git if needed and include the last git commit hash in the globals.")

;; Calls thunk. Outputs to a log file if `log?` is not #false.
;; When `git?` is not #false, also commits to git to ensure consistency of the code base
;; with the experiment, and adds the git commit number to the global variables.
;;
;; thunk : thunk?
;; dir : path-string?
;; filename : string?
;; filepath : path-string?
;; log? : boolean?
;; git? : boolean?
;; quiet? : boolean?
(define (call-with-log thunk
                       #:? [dir "logs"]
                       #:? [filename (string-append "log-" (date-iso-file) ".txt")]
                       ; if given, dir and filename have no effect:
                       #:? [filepath (build-path dir filename)]
                       #:? [log? (*log*)]
                       #:? [git? (*git?*)]
                       #:? [quiet? #false])

  (when git?
    (define cmd "git commit -am \".\" ")
    (displayln cmd)
    (system cmd))

  ;; Non-quiet mode.
  (define (thunk2)
    ; Also write the last commit hash for easy retrieval.
    (pretty-write
     (list* `(cmd-line . ,(current-command-line-arguments))
            `(git-commit . ,(and git?
                                 (string-normalize-spaces
                                  (with-output-to-string (λ () (system "git rev-parse HEAD"))))))
            (globals->assoc)))
    (thunk))

  (cond [log?
         (make-parent-directory* filepath)
         (assert (not (file-exists? filepath)) filepath)
         (printf "Logging to: ~a\n" filepath)
         (pretty-write (globals->assoc))
         (with-output-to-file filepath thunk2)]
        [quiet? (thunk)]
        [else (thunk2)]))
