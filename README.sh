#!/bin/bash

# dump pandoc template for latex
pandoc -D latex > README.tex

# adjust template
sed -i \
  -e 's/\(\\documentclass\[.\+\]{.\+}\)/\1\n\\pagenumbering{gobble}/' \
  -e 's/\\usepackage{listings}/\\usepackage{listings}\\lstset{basicstyle=\\footnotesize\\ttfamily}/' \
  README.tex

# cleanup markup and pipe into pandoc using the template
sed \
  -e 's/\[\[.\+\](.\+)\]//' \
  -e '/!\[.\+\](.\+)/{n;d}' \
  README.md | \
pandoc \
  --template=README.tex --listings \
  -f markdown_github+implicit_figures \
  -V documentclass=scrartcl \
  -V classoption=DIV=26 \
  -V linkcolor=black \
  -V citecolor=black \
  -V urlcolor=black \
  -o documentation/libxstream.pdf
