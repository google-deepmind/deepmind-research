#lang racket/base

(require "../timeplus.rkt"
         rackunit)

(check-equal? (string-drop-common-prefix '("auiebépo" "auiensrt" "au"))
              '("iebépo" "iensrt" ""))
(check-equal? (string-drop-common-prefix '("auiebépo" "auiensrt" ))
              '("bépo" "nsrt"))
(check-equal? (string-drop-common-prefix '("auiebépo" ))
              '(""))
(check-equal? (string-drop-common-prefix '("" ))
              '(""))
