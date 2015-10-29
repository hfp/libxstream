# export all variables to sub-make processes
#.EXPORT_ALL_VARIABLES: #export
# Set MAKE_PARALLEL=0 for issues with parallel make (older GNU Make)
ifeq (0,$(MAKE_PARALLEL))
.NOTPARALLEL:
else ifneq (3.82,$(firstword $(sort $(MAKE_VERSION) 3.82)))
.NOTPARALLEL:
endif

# Linux cut has features we use that do not work elsewhere
# Mac, etc. users should install GNU coreutils and use cut from there.
#
# For example, if you use Homebrew, run "brew install coreutils" once
# and then invoke the LIBXSMM make command with
# CUT=/usr/local/Cellar/coreutils/8.24/libexec/gnubin/cut
CUT ?= cut

ARCH = intel64

ROOTDIR = $(abspath $(dir $(word $(words $(MAKEFILE_LIST)),$(MAKEFILE_LIST))))
INCDIR = $(ROOTDIR)/include
SRCDIR = $(ROOTDIR)/src
BLDDIR = build/$(ARCH)
OUTDIR = lib/$(ARCH)
DOCDIR = documentation

CXXFLAGS = $(NULL)
CFLAGS = $(NULL)
DFLAGS = -D__extern_always_inline=inline -DLIBXSTREAM_EXPORTED
IFLAGS = -I$(INCDIR)

# Request strongest code conformance
PEDANTIC ?= 0

OFFLOAD ?= 1
STATIC ?= 1
OMP ?= 0
SYM ?= 0
DBG ?= 0
IPO ?= 0
EXP ?= 0
SSE ?= 0
AVX ?= 0

OUTNAME = $(shell basename $(ROOTDIR))
HEADERS = $(shell ls -1 $(INCDIR)/*.h   2> /dev/null | tr "\n" " ") \
          $(shell ls -1 $(SRCDIR)/*.hpp 2> /dev/null | tr "\n" " ") \
          $(shell ls -1 $(SRCDIR)/*.hxx 2> /dev/null | tr "\n" " ") \
          $(shell ls -1 $(SRCDIR)/*.hh  2> /dev/null | tr "\n" " ")
CPPSRCS = $(shell ls -1 $(SRCDIR)/*.cpp 2> /dev/null | tr "\n" " ")
CXXSRCS = $(shell ls -1 $(SRCDIR)/*.cxx 2> /dev/null | tr "\n" " ")
CCXSRCS = $(shell ls -1 $(SRCDIR)/*.cc  2> /dev/null | tr "\n" " ")
CSOURCS = $(shell ls -1 $(SRCDIR)/*.c   2> /dev/null | tr "\n" " ")
FTNSRCS = $(shell ls -1 $(SRCDIR)/*.f   2> /dev/null | tr "\n" " ")
F77SRCS = $(shell ls -1 $(SRCDIR)/*.F   2> /dev/null | tr "\n" " ")
F90SRCS = $(shell ls -1 $(SRCDIR)/*.f90 2> /dev/null | tr "\n" " ")
FTNINCS = $(shell ls -1 $(DEPDIR)/include/*.f   2> /dev/null | tr "\n" " ")
F77INCS = $(shell ls -1 $(DEPDIR)/include/*.F   2> /dev/null | tr "\n" " ")
F90INCS = $(shell ls -1 $(DEPDIR)/include/*.f90 2> /dev/null | tr "\n" " ")
FTNMODS = $(patsubst %,$(BLDDIR)/%,$(notdir $(FTNINCS:.f=-mod.o)))
F77MODS = $(patsubst %,$(BLDDIR)/%,$(notdir $(F77INCS:.F=-mod77.o)))
F90MODS = $(patsubst %,$(BLDDIR)/%,$(notdir $(F90INCS:.f90=-mod90.o)))
MODULES = $(FTNMODS) $(F77MODS) $(F90MODS)
SOURCES = $(CPPSRCS) $(CXXSRCS) $(CCXSRCS) $(CSOURCS) $(FTNSRCS) $(F77SRCS) $(F90SRCS)
CPPOBJS = $(patsubst %,$(BLDDIR)/%,$(notdir $(CPPSRCS:.cpp=-cpp.o)))
CXXOBJS = $(patsubst %,$(BLDDIR)/%,$(notdir $(CXXSRCS:.cxx=-cxx.o)))
CCXOBJS = $(patsubst %,$(BLDDIR)/%,$(notdir $(CCXSRCS:.cc=-cc.o)))
COBJCTS = $(patsubst %,$(BLDDIR)/%,$(notdir $(CSOURCS:.c=-c.o)))
FTNOBJS = $(patsubst %,$(BLDDIR)/%,$(notdir $(FTNSRCS:.f=-f.o)))
F77OBJS = $(patsubst %,$(BLDDIR)/%,$(notdir $(F77SRCS:.F=-f77.o)))
F90OBJS = $(patsubst %,$(BLDDIR)/%,$(notdir $(F90SRCS:.f90=-f90.o)))
OBJECTS = $(CPPOBJS) $(CXXOBJS) $(CCXOBJS) $(COBJCTS) $(FTNOBJS) $(F77OBJS) $(F90OBJS)

ICPC    = $(notdir $(shell which icpc     2> /dev/null))
ICC     = $(notdir $(shell which icc      2> /dev/null))
IFORT   = $(notdir $(shell which ifort    2> /dev/null))
GPP     = $(notdir $(shell which g++      2> /dev/null))
GCC     = $(notdir $(shell which gcc      2> /dev/null))
GFC     = $(notdir $(shell which gfortran 2> /dev/null))

CXX_CHECK = $(notdir $(shell which $(CXX) 2> /dev/null))
CC_CHECK  = $(notdir $(shell which $(CC)  2> /dev/null))
FC_CHECK  = $(notdir $(shell which $(FC)  2> /dev/null))

# prefer Intel Compiler (if available)
CXX = $(ICPC)
FC = $(IFORT)
CC = $(ICC)

INTEL = $(shell echo $$((3==$(words $(filter icc icpc ifort,$(CC) $(CXX) $(FC))))))

ifneq (0,$(INTEL))
	AR = xiar
	CXXFLAGS += -fPIC -Wall -std=c++0x
	CFLAGS += -fPIC -Wall -std=c99
	FCMTFLAGS += -threads
	FCFLAGS += -fPIC
	LDFLAGS += -fPIC -lrt
	ifeq (1,$(PEDANTIC))
		CFLAGS += -std=c89 -Wcheck
	else ifneq (0,$(PEDANTIC))
		CFLAGS += -std=c89 -Wcheck -Wremarks
	endif
	ifeq (0,$(DBG))
		CXXFLAGS += -fno-alias -ansi-alias -O2
		CFLAGS += -fno-alias -ansi-alias -O2
		FCFLAGS += -O2
		DFLAGS += -DNDEBUG
		ifneq (0,$(IPO))
			CXXFLAGS += -ipo
			CFLAGS += -ipo
			FCFLAGS += -ipo
		endif
	else
		CXXFLAGS += -O0
		CFLAGS += -O0
		FCFLAGS += -O0
		SYM = $(DBG)
	endif
	ifeq (1,$(shell echo $$((2 > $(DBG)))))
		ifeq (1,$(AVX))
			TARGET = -xAVX
		else ifeq (2,$(AVX))
			TARGET = -xCORE-AVX2
		else ifeq (3,$(AVX))
			ifeq (0,$(MIC))
				TARGET = -xCOMMON-AVX512
			else
				TARGET = -xMIC-AVX512
			endif
		else ifeq (1,$(shell echo $$((2 <= $(SSE)))))
			TARGET = -xSSE$(SSE)
		else ifeq (1,$(SSE))
			TARGET = -xSSE3
		else
			TARGET = -xHost
		endif
	endif
	ifneq (0,$(SYM))
		ifneq (1,$(SYM))
			CXXFLAGS := -g3 -gdwarf-2 -debug inline-debug-info $(CXXFLAGS)
			CFLAGS := -g3 -gdwarf-2 -debug inline-debug-info $(CFLAGS)
			FCFLAGS := -g $(FCFLAGS)
		else
			CXXFLAGS := -g $(CXXFLAGS)
			CFLAGS := -g $(CFLAGS)
			FCFLAGS := -g $(FCFLAGS)
		endif
	endif
	ifeq (0,$(EXP))
		CXXFLAGS += -fno-exceptions
	endif
	ifneq (0,$(OMP))
		CXXFLAGS += -openmp
		CFLAGS += -openmp
		FCFLAGS += -openmp
		LDFLAGS += -openmp
	endif
	ifeq (0,$(OFFLOAD))
		CXXFLAGS += -no-offload
		CFLAGS += -no-offload
		FCFLAGS += -no-offload
	endif
	ifneq (0,$(STATIC))
		SLDFLAGS += -no-intel-extensions -static-intel
	endif
	FCMODDIRFLAG = -module
else # GCC assumed
	ifeq (,$(CXX_CHECK))
		CXX = $(GPP)
	endif
	ifeq (,$(CC_CHECK))
		CC = $(GCC)
	endif
	ifeq (,$(FC_CHECK))
		FC = $(GFC)
	endif
	VERSION = $(shell $(CC) --version | grep "gcc (GCC)" | sed "s/gcc (GCC) \([0-9]\+\.[0-9]\+\.[0-9]\+\).*$$/\1/")
	VERSION_MAJOR = $(shell echo "$(VERSION)" | $(CUT) -d"." -f1)
	VERSION_MINOR = $(shell echo "$(VERSION)" | $(CUT) -d"." -f2)
	VERSION_PATCH = $(shell echo "$(VERSION)" | $(CUT) -d"." -f3)
	MIC = 0
	CXXFLAGS += -Wall -std=c++0x -Wno-unused-function
	CFLAGS += -Wall -Wno-unused-function
	LDFLAGS += -lrt
	ifneq (Windows_NT,$(OS))
		CXXFLAGS += -fPIC
		CFLAGS += -fPIC
		FCFLAGS += -fPIC
		LDFLAGS += -fPIC
	endif
	ifneq (0,$(PEDANTIC))
		CFLAGS += -std=c89 -pedantic -Wno-variadic-macros -Wno-long-long -Wno-overlength-strings
	endif
	ifeq (0,$(DBG))
		CXXFLAGS += -O2 -ftree-vectorize -ffast-math -funroll-loops
		CFLAGS += -O2 -ftree-vectorize -ffast-math -funroll-loops
		FCFLAGS += -O2 -ftree-vectorize -ffast-math -funroll-loops
		DFLAGS += -DNDEBUG
		ifneq (0,$(IPO))
			CXXFLAGS += -flto -ffat-lto-objects
			CFLAGS += -flto -ffat-lto-objects
			FCFLAGS += -flto -ffat-lto-objects
			LDFLAGS += -flto
		endif
	else
		CXXFLAGS += -O0
		CFLAGS += -O0
		FCFLAGS += -O0
		SYM = $(DBG)
	endif
	ifeq (1,$(shell echo $$((2 > $(DBG)))))
		ifeq (1,$(AVX))
			TARGET = -mavx
		else ifeq (2,$(AVX))
			TARGET = -mavx2
		else ifeq (3,$(AVX))
			TARGET = -mavx512f -mavx512cd
			ifneq (0,$(MIC))
				TARGET += -mavx512er -mavx512pf
			endif
		else ifeq (1,$(shell echo $$((2 <= $(SSE)))))
			TARGET = -msse$(SSE)
		else ifeq (1,$(SSE))
			TARGET = -msse3
		else
			TARGET = -march=native
		endif
	endif
	ifneq (0,$(SYM))
		ifneq (1,$(SYM))
			CXXFLAGS := -g3 -gdwarf-2 -debug inline-debug-info $(CXXFLAGS)
			CFLAGS := -g3 -gdwarf-2 -debug inline-debug-info $(CFLAGS)
			FCFLAGS := -g $(FCFLAGS)
		else
			CXXFLAGS := -g $(CXXFLAGS)
			CFLAGS := -g $(CFLAGS)
			FCFLAGS := -g $(FCFLAGS)
		endif
	endif
	ifeq (0,$(EXP))
		CXXFLAGS += -fno-exceptions
	endif
	ifneq (0,$(OMP))
		CXXFLAGS += -fopenmp
		CFLAGS += -fopenmp
		FCFLAGS += -fopenmp
		LDFLAGS += -fopenmp
	endif
	ifneq (0,$(STATIC))
		SLDFLAGS += -static
	endif
	FCMODDIRFLAG = -J
endif

ifneq (,$(CXX))
	LD = $(CXX)
endif
ifeq (,$(LD))
	LD = $(CC)
endif
ifeq (,$(LD))
	LD = $(FC)
endif

ifeq (,$(CXXFLAGS))
	CXXFLAGS = $(CFLAGS)
endif
ifeq (,$(CFLAGS))
	CFLAGS = $(CXXFLAGS)
endif
ifeq (,$(FCFLAGS))
	FCFLAGS = $(CFLAGS)
endif
ifeq (,$(LDFLAGS))
	LDFLAGS = $(CFLAGS)
endif

ifneq (0,$(STATIC))
	LIBEXT = a
else
	LIBEXT = so
endif

parent = $(subst ?, ,$(firstword $(subst /, ,$(subst $(NULL) ,?,$(patsubst ./%,%,$1)))))

.PHONY: all
all: $(OUTDIR)/$(OUTNAME).$(LIBEXT)

$(OUTDIR)/$(OUTNAME).$(LIBEXT): $(OBJECTS)
	@mkdir -p $(dir $@)
ifeq ($(STATIC),0)
	$(LD) -shared -o $@ $(LDFLAGS) $^
else
	$(AR) -rs $@ $^
endif

$(BLDDIR)/%-mod.o: $(DEPDIR)/include/%.f $(ROOTDIR)/Makefile
	@mkdir -p $(dir $@)
	$(FC) $(FCFLAGS) $(FCMTFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@ $(FCMODDIRFLAG) $(dir $@)

$(BLDDIR)/%-mod90.o: $(DEPDIR)/include/%.f90 $(ROOTDIR)/Makefile
	@mkdir -p $(dir $@)
	$(FC) $(FCFLAGS) $(FCMTFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@ $(FCMODDIRFLAG) $(dir $@)

$(BLDDIR)/%-mod77.o: $(DEPDIR)/include/%.F $(ROOTDIR)/Makefile
	@mkdir -p $(dir $@)
	$(FC) $(FCFLAGS) $(FCMTFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@ $(FCMODDIRFLAG) $(dir $@)

$(BLDDIR)/%-cpp.o: $(SRCDIR)/%.cpp $(HEADERS) $(ROOTDIR)/Makefile
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@

$(BLDDIR)/%-c.o: $(SRCDIR)/%.c $(HEADERS) $(ROOTDIR)/Makefile
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@

$(BLDDIR)/%-f.o: $(SRCDIR)/%.f $(MODULES) $(ROOTDIR)/Makefile
	@mkdir -p $(dir $@)
	$(FC) $(FCFLAGS) $(FCMTFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@

$(BLDDIR)/%-f90.o: $(SRCDIR)/%.f90 $(MODULES) $(ROOTDIR)/Makefile
	@mkdir -p $(dir $@)
	$(FC) $(FCFLAGS) $(FCMTFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@

$(BLDDIR)/%-f77.o: $(SRCDIR)/%.F $(MODULES) $(ROOTDIR)/Makefile
	@mkdir -p $(dir $@)
	$(FC) $(FCFLAGS) $(FCMTFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@

$(DOCDIR)/libxstream.pdf: $(ROOTDIR)/README.md
	@mkdir -p $(dir $@)
	$(eval TEMPLATE := $(shell mktemp --tmpdir=. --suffix=.tex))
	@pandoc -D latex > $(TEMPLATE)
	@TMPFILE=`mktemp`
	@sed -i ${TMPFILE} \
		-e 's/\(\\documentclass\[.\+\]{.\+}\)/\1\n\\pagenumbering{gobble}\n\\RedeclareSectionCommands[beforeskip=-1pt,afterskip=1pt]{subsection,subsubsection}/' \
		-e 's/\\usepackage{listings}/\\usepackage{listings}\\lstset{basicstyle=\\footnotesize\\ttfamily}/' \
		$(TEMPLATE)
	@rm -f ${TMPFILE}
	@sed \
		-e 's/https:\/\/raw\.githubusercontent\.com\/hfp\/libxstream\/master\///' \
		-e 's/\[!\[.\+\](https:\/\/travis-ci.org\/hfp\/libxstream.svg?branch=.\+)\](.\+)//' \
		-e 's/\[\[.\+\](.\+)\]//' \
		-e '/!\[.\+\](.\+)/{n;d}' \
		$(ROOTDIR)/README.md | \
	pandoc \
		--latex-engine=xelatex --template=$(TEMPLATE) --listings \
		-f markdown_github+implicit_figures+all_symbols_escapable \
		-V documentclass=scrartcl \
		-V title-meta="LIBXSTREAM Documentation" \
		-V author-meta="Hans Pabst" \
		-V classoption=DIV=45 \
		-V linkcolor=black \
		-V citecolor=black \
		-V urlcolor=black \
		-o $@
	@rm $(TEMPLATE)

$(DOCDIR)/cp2k.pdf: $(ROOTDIR)/documentation/cp2k.md
	@mkdir -p $(dir $@)
	$(eval TEMPLATE := $(shell mktemp --tmpdir=. --suffix=.tex))
	@pandoc -D latex > $(TEMPLATE)
	@TMPFILE=`mktemp`
	@sed -i ${TMPFILE} \
		-e 's/\(\\documentclass\[.\+\]{.\+}\)/\1\n\\pagenumbering{gobble}\n\\RedeclareSectionCommands[beforeskip=-1pt,afterskip=1pt]{subsection,subsubsection}/' \
		-e 's/\\usepackage{listings}/\\usepackage{listings}\\lstset{basicstyle=\\footnotesize\\ttfamily}/' \
		$(TEMPLATE)
	@rm -f ${TMPFILE}
	@sed \
		-e 's/https:\/\/raw\.githubusercontent\.com\/hfp\/libxstream\/master\///' \
		-e 's/\[!\[.\+\](https:\/\/travis-ci.org\/hfp\/libxstream.svg?branch=.\+)\](.\+)//' \
		-e 's/\[\[.\+\](.\+)\]//' \
		-e '/!\[.\+\](.\+)/{n;d}' \
		$(ROOTDIR)/documentation/cp2k.md | \
	pandoc \
		--latex-engine=xelatex --template=$(TEMPLATE) --listings \
		-f markdown_github+implicit_figures+all_symbols_escapable \
		-V documentclass=scrartcl \
		-V title-meta="CP2K with LIBXSTREAM" \
		-V author-meta="Hans Pabst" \
		-V classoption=DIV=45 \
		-V linkcolor=black \
		-V citecolor=black \
		-V urlcolor=black \
		-o $@
	@rm $(TEMPLATE)

.PHONY: documentation
documentation: $(DOCDIR)/libxstream.pdf $(DOCDIR)/cp2k.pdf

.PHONY: clean
clean:
ifneq ($(abspath $(call parent,$(BLDDIR))),$(ROOTDIR))
ifneq ($(abspath $(call parent,$(BLDDIR))),$(abspath .))
	@rm -rf $(call parent,$(BLDDIR)) *.mod
else
	@rm -f $(OBJECTS) $(BLDDIR)/*.mod
endif
else
	@rm -f $(OBJECTS) $(BLDDIR)/*.mod
endif

.PHONY: realclean
realclean: clean
ifneq ($(abspath $(call parent,$(OUTDIR))),$(ROOTDIR))
ifneq ($(abspath $(call parent,$(OUTDIR))),$(abspath .))
	@rm -rf $(call parent,$(OUTDIR))
else
	@rm -f $(OUTDIR)/$(OUTNAME)
endif
else
	@rm -f $(OUTDIR)/$(OUTNAME)
endif

install: all clean
	@cp -r $(INCDIR) . 2> /dev/null || true

