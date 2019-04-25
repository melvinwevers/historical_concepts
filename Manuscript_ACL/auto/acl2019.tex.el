(TeX-add-style-hook
 "acl2019.tex"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "11pt" "a4paper")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("inputenc" "utf8") ("fontenc" "T1") ("ulem" "normalem") ("acl2019" "hyperref" "nohyperref")))
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art11"
    "inputenc"
    "fontenc"
    "graphicx"
    "grffile"
    "longtable"
    "wrapfig"
    "rotating"
    "ulem"
    "amsmath"
    "textcomp"
    "amssymb"
    "capt-of"
    "hyperref"
    "acl2019"
    "times"
    "latexsym"
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
   (LaTeX-add-bibliographies
    "acl2019"))
 :latex)

