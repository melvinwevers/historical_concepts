(TeX-add-style-hook
 "acl2019"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "11pt" "a4paper")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("acl2019" "hyperref" "nohyperref")))
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "url")
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art11"
    "times"
    "latexsym"
    "graphicx"
    "url"
    "etoolbox")
   (TeX-add-symbols
    "BibTeX"
    "aclpaperid")
   (LaTeX-add-labels
    "fig:vocab-size"
    "sect:pdf"
    "ssec:layout"
    "font-table"
    "ssec:first"
    "tab:accents"
    "ssec:accessibility"
    "sec:length"
    "sec:appendix"
    "sec:supplemental")
   (LaTeX-add-bibliographies))
 :latex)

