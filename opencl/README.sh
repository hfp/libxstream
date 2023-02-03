#!/usr/bin/env bash

HERE=$(cd $(dirname $0); pwd -P)

# output directory
if [ "" != "$1" ]; then
  DOCDIR=$1
  shift
else
  DOCDIR=.
fi

# temporary file
TMPFILE=$(mktemp fileXXXXXX)
mv ${TMPFILE} ${TMPFILE}.tex

# dump pandoc template for latex, and adjust the template
pandoc -D latex \
| sed \
  -e 's/\(\\documentclass\[..*\]{..*}\)/\1\n\\pagenumbering{gobble}\n\\RedeclareSectionCommands[beforeskip=-1pt,afterskip=1pt]{subsection,subsubsection}/' \
  -e 's/\\usepackage{listings}/\\usepackage{listings}\\lstset{basicstyle=\\footnotesize\\ttfamily,showstringspaces=false}/' > \
  ${TMPFILE}.tex

# cleanup markup and pipe into pandoc using the template
( iconv -t utf-8 README.md && echo && \
  iconv -t utf-8 smm/README.md
) | sed \
  -e 's/<sub>/~/g' -e 's/<\/sub>/~/g' \
  -e 's/<sup>/^/g' -e 's/<\/sup>/^/g' \
  -e 's/\[\[..*\](..*)\]//g' \
  -e 's/\[!\[..*\](..*)\](..*)//g' \
| tee >( pandoc \
  --template=${TMPFILE}.tex --listings \
  -f gfm+implicit_figures+subscript+superscript \
  -V documentclass=scrartcl \
  -V title-meta="XCONFIGURE Documentation" \
  -V author-meta="Hans Pabst" \
  -V classoption=DIV=45 \
  -V linkcolor=black \
  -V citecolor=black \
  -V urlcolor=black \
  -o ${DOCDIR}/README.pdf) \
| tee >( pandoc \
  -f gfm+implicit_figures+subscript+superscript \
  -o ${DOCDIR}/README.html) \
| pandoc \
  -f gfm+implicit_figures+subscript+superscript \
  -o ${DOCDIR}/README.docx

# remove temporary file
rm ${TMPFILE}.tex
