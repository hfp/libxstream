# ROOTDIR avoid abspath to match Makefile targets
ROOTDIR := $(subst //,,$(dir $(firstword $(MAKEFILE_LIST)))/)
PROJECT := libxstream
# Source and scripts locations
ROOTINC := $(ROOTDIR)/include
ROOTSCR := $(ROOTDIR)/scripts
ROOTSRC := $(ROOTDIR)/src
# Project directory structure
INCDIR := include
SCRDIR := scripts
TSTDIR := tests
BLDDIR := obj
SRCDIR := src
OUTDIR := lib
BINDIR := bin
SPLDIR := samples
DOCDIR := documentation

# subdirectories (relative) to PREFIX (install targets)
PINCDIR ?= $(INCDIR)
PSRCDIR ?= $(PROJECT)
POUTDIR ?= $(OUTDIR)
PPKGDIR ?= $(OUTDIR)
PMODDIR ?= $(OUTDIR)
PBINDIR ?= $(BINDIR)
PTSTDIR ?= $(TSTDIR)
PSHRDIR ?= share/$(PROJECT)
PDOCDIR ?= $(PSHRDIR)
LICFDIR ?= $(PDOCDIR)
LICFILE ?= LICENSE.md

# initial default flags: RPM_OPT_FLAGS are usually NULL
CFLAGS := $(RPM_OPT_FLAGS)
CXXFLAGS := $(RPM_OPT_FLAGS)

# Enable thread-local cache (registry)
# 0: "disable", 1: "enable", or small power-of-two number.
CACHE ?= 1

# Specify the size of a cacheline (Bytes)
CACHELINE ?= 64

# Determines if the library is thread-safe
THREADS ?= 1

# 0: link all dependencies as specified for the target
# 1: attempt to avoid dependencies if not referenced
ASNEEDED ?= 0

# project needs OpenCL by default
OCL ?= 2

# OpenMP is disabled by default and the library
# is agnostic wrt the threading runtime
OMP ?= 0

# There is no reliance on Fortran
FORTRAN ?= 0

# Kind of documentation (internal key)
DOCEXT := pdf

# Timeout when downloading documentation parts
TIMEOUT := 30

# fixed .state file directory (included by source)
DIRSTATE := $(OUTDIR)/..

# avoid to link with C++ standard library
FORCE_CXX := 0

# enable additional/compile-time warnings
WCHECK := 1

# state to be excluded from tracking the (re-)build state
EXCLUDE_STATE := \
  DESTDIR PREFIX BINDIR CURDIR DOCDIR DOCEXT INCDIR LICFDIR OUTDIR TSTDIR TIMEOUT \
  PBINDIR PINCDIR POUTDIR PPKGDIR PMODDIR PSRCDIR PTSTDIR PSHRDIR PDOCDIR SCRDIR \
  SPLDIR SRCDIR TEST VERSION_STRING ALIAS_% %_TARGET %ROOT

# root directory of LIBXS
LIBXSROOT := $(wildcard $(ROOTDIR)/../libxs)

# include common Makefile artifacts
include $(LIBXSROOT)/Makefile.inc

# setup LIBXS
ifneq (,$(LIBXSROOT))
  LIBXS_SL := $(wildcard $(LIBXSROOT)/lib/libxs.$(SLIBEXT))
  LIBXS_DL := $(wildcard $(LIBXSROOT)/lib/libxs.$(DLIBEXT))
  LIBXS := $(wildcard $(LIBXSROOT)/lib/libxs.$(LIBEXT))
  LIBXS := $(strip $(if $(LIBXS),$(LIBXS), \
    $(if $(LIBXS_SL),$(LIBXS_SL),$(LIBXS_DL))))
  IFLAGS += -I$(call quote,$(LIBXSROOT)/include)
endif
ifneq (,$(LIBXS))
  DFLAGS += -D__LIBXS
endif

# 0: static, 1: shared, 2: static and shared
ifneq (,$(filter-out file,$(origin STATIC)))
  ifneq (0,$(STATIC))
    BUILD := 0
  else # shared
    BUILD := 1
  endif
else # default
  BUILD := 2
endif

# target library for a broad range of systems
SSE ?= 1

# include directories
IFLAGS += -I$(call quote,$(INCDIR))

# Version numbers according to interface (version.txt)
VERSION_MAJOR ?= $(shell $(ROOTSCR)/tool_version.sh $(PROJECT) 1)
VERSION_MINOR ?= $(shell $(ROOTSCR)/tool_version.sh $(PROJECT) 2)
VERSION_UPDATE ?= $(shell $(ROOTSCR)/tool_version.sh $(PROJECT) 3)
VERSION_STRING ?= $(VERSION_MAJOR).$(VERSION_MINOR).$(VERSION_UPDATE)
VERSION_ALL ?= $(shell $(ROOTSCR)/tool_version.sh $(PROJECT) 0)
VERSION_API ?= $(VERSION_MAJOR)
VERSION_RELEASED ?= $(if $(shell $(ROOTSCR)/tool_version.sh $(PROJECT) 4),0,1)
VERSION_RELEASE ?= HEAD
VERSION_PACKAGE ?= 1

# Link shared library with correct version stamp
solink_version = $(call solink,$1,$(VERSION_MAJOR),$(VERSION_MINOR),$(VERSION_UPDATE),$(VERSION_API))

# target library for a broad range of systems
ifeq (file,$(origin AVX))
  AVX_STATIC := 0
endif
AVX_STATIC ?= $(AVX)

HEADERS_MAIN := $(ROOTINC)/libxstream_dbcsr.h $(ROOTINC)/libxstream_opencl.h $(ROOTINC)/libxstream.h
HEADERS_SRC := $(wildcard $(ROOTSRC)/*.h)
HEADERS := $(HEADERS_SRC) $(HEADERS_MAIN)
SRCFILES := $(patsubst %,$(ROOTSRC)/%,libxstream_dbcsr.c libxstream.c libxstream_event.c libxstream_mem.c libxstream_stream.c)
OBJFILES := $(patsubst %,$(BLDDIR)/intel64/%.o,$(basename $(notdir $(SRCFILES))))

# no warning conversion for released versions
ifneq (0,$(VERSION_RELEASED))
  WERROR := 0
endif
# no warning conversion for non-x86
#ifneq (x86_64,$(MNAME))
#  WERROR := 0
#endif
# no warning conversion
ifneq (,$(filter-out 0 1,$(INTEL)))
  WERROR := 0
endif

information = \
  $(info ================================================================================) \
  $(info $(PROJUPP)$(if $(VERSION_ALL), $(VERSION_ALL)) ($(UNAME)$(if \
         $(filter-out 0,$(LIBXS_TARGET_HIDDEN)),$(NULL),$(if $(HOSTNAME),@$(HOSTNAME))))) \
  $(info --------------------------------------------------------------------------------) \
  $(info $(GINFO)) \
  $(info $(CINFO)) \
  $(info --------------------------------------------------------------------------------) \
  $(if $(ENVSTATE),$(info Environment: $(ENVSTATE)) \
  $(info --------------------------------------------------------------------------------))

ifneq (,$(strip $(TEST)))
.PHONY: run-tests
run-tests: tests
endif

.PHONY: $(PROJECT)
$(PROJECT): lib
	$(information)

.PHONY: lib
lib: libs

.PHONY: all
all: $(PROJECT)

.PHONY: realall
realall: all samples

.PHONY: winterface
winterface: headers sources

.PHONY: config
config: $(INCDIR)/$(PROJECT)_version.h

$(INCDIR)/$(PROJECT)_version.h: $(INCDIR)/.make $(DIRSTATE)/.state $(ROOTSCR)/tool_version.sh
	$(information)
	$(info --- $(PROJUPP) build log)
	@$(CP) -r $(ROOTSCR) . 2>/dev/null || true
	@$(CP) $(ROOTDIR)/Makefile.inc . 2>/dev/null || true
	@$(CP) $(ROOTDIR)/.mktmp.sh . 2>/dev/null || true
	@$(CP) $(ROOTDIR)/.flock.sh . 2>/dev/null || true
	@$(CP) $(ROOTDIR)/.state.sh . 2>/dev/null || true
	@$(CP) $(HEADERS_MAIN) $(INCDIR) 2>/dev/null || true
	@$(CP) $(SRCFILES) $(HEADERS_SRC) $(SRCDIR) 2>/dev/null || true
	@$(ROOTSCR)/tool_version.sh $(PROJECT) -1 >$@
	@$(SED) 's/^\(  VERSION \)[0-9][0-9.]*/\1$(VERSION_STRING)/' $(ROOTDIR)/CMakeLists.txt >$(ROOTDIR)/CMakeLists.tmp 2>/dev/null && mv $(ROOTDIR)/CMakeLists.tmp $(ROOTDIR)/CMakeLists.txt || true

define DEFINE_COMPILE_RULE
$(1): $(2) $(3) $(dir $(1))/.make
# @-rm -f $(1)
	-$(CC) $(if $(filter 0,$(WERROR)),$(4),$(filter-out $(WERROR_CFLAG),$(4)) $(WERROR_CFLAG)) -c $(2) -o $(1)
endef

$(foreach OBJ,$(OBJFILES),$(eval $(call DEFINE_COMPILE_RULE, \
  $(OBJ),$(patsubst %.o,$(ROOTSRC)/%.c,$(notdir $(OBJ))), \
  $(HEADERS_MAIN) $(INCDIR)/$(PROJECT)_version.h, \
  $(DFLAGS) $(IFLAGS) $(CTARGET) $(CFLAGS))))

.PHONY: libs
libs: $(OUTDIR)/$(PROJECT)-static.pc $(OUTDIR)/$(PROJECT)-shared.pc
ifeq (,$(filter-out 0 2,$(BUILD)))
$(OUTDIR)/$(PROJECT).$(SLIBEXT): $(OUTDIR)/.make $(OBJFILES) $(FTNOBJS)
	$(MAKE_AR) $(OUTDIR)/$(PROJECT).$(SLIBEXT) $(call tailwords,$^)
else
.PHONY: $(OUTDIR)/$(PROJECT).$(SLIBEXT)
endif
ifeq (0,$(filter-out 1 2,$(BUILD))$(ANALYZE))
$(OUTDIR)/$(PROJECT).$(DLIBEXT): $(OUTDIR)/.make $(OBJFILES) $(FTNOBJS)
	$(LIB_SOLD) $(call solink_version,$(OUTDIR)/$(PROJECT).$(DLIBEXT)) \
		$(call tailwords,$^) $(call cleanld,$(LDFLAGS) $(CLDFLAGS))
else
.PHONY: $(OUTDIR)/$(PROJECT).$(DLIBEXT)
endif

# use dir not qdir to avoid quotes
SAMPLES := $(dir $(shell $(if $(GIT),$(GIT) ls-files,ls -1) $(SPLDIR)/*/Makefile 2>/dev/null))
SPLMDS := $(addprefix $(ABSDIR)/,$(shell $(if $(GIT),$(GIT) ls-files,ls -1) \
  $(SPLDIR)/*/README.md 2>/dev/null))
DOCMDS := $(addprefix $(ABSDIR)/,$(filter-out \
    $(DOCDIR)/$(PROJECT)_samples.md \
    $(DOCDIR)/$(PROJECT)_scripts.md, \
  $(shell $(if $(GIT),$(GIT) ls-files,ls -1) \
    $(DOCDIR)/$(PROJECT)_*.md 2>/dev/null)))

.PHONY: samples $(DIRS_SAMPLES)
samples: $(DIRS_SAMPLES)
$(DIRS_SAMPLES): libs
	@$(FLOCK) $@ "$(MAKE)"

.PHONY: test-all
test-all: tests

.PHONY: test
test: tests

.PHONY: drytest
drytest: build-tests

.PHONY: build-tests
build-tests: libs
	@$(FLOCK) $(ROOTDIR)/$(TSTDIR) "$(MAKE) --no-print-directory"

.PHONY: tests
tests: libs
	@$(FLOCK) $(ROOTDIR)/$(TSTDIR) "$(MAKE) --no-print-directory test"

$(DOCDIR)/index.md: $(DOCDIR)/.make $(ROOTDIR)/Makefile $(ROOTDIR)/README.md
	@$(SED) $(ROOTDIR)/README.md \
		-e 's/\[!\[..*\](..*)\](..*)//g' \
		-e 's/\[\[..*\](..*)\]//g' \
		-e "s/](${DOCDIR}\//](/g" \
		-e 'N;/^\n$$/d;P;D' \
		>$@
	@$(CP) $(ROOTDIR)/LICENSE.md $(DOCDIR)/LICENSE.md

$(DOCDIR)/$(PROJECT)_scripts.md: $(DOCDIR)/.make $(ROOTDIR)/Makefile $(ROOTSCR)/README.md
	@$(SED) $(ROOTSCR)/README.md \
		-e 's/\[!\[..*\](..*)\](..*)//g' \
		-e 's/\[\[..*\](..*)\]//g' \
		-e "s/](${DOCDIR}\//](/g" \
		-e 'N;/^\n$$/d;P;D' \
		>$@

$(DOCDIR)/$(PROJECT)_samples.md: $(DOCDIR)/.make $(DOCDIR)/$(SPLDIR)/.make $(ROOTDIR)/Makefile $(SPLMDS)
	@for MD in $(SPLMDS); do \
		$(SED) $${MD} \
			-e 's/\[!\[..*\](..*)\](..*)//g' \
			-e 's/\[\[..*\](..*)\]//g' \
			-e "s/](${DOCDIR}\//](/g" \
			-e 'N;/^\n$$/d;P;D' \
			>$(DOCDIR)/$(SPLDIR)/$(PROJECT)_$$(basename $$(dirname $${MD})).md; \
	done
	@$(SED) $(SPLMDS) \
		-e 's/^#/##/' \
		-e 's/<sub>/~/g' -e 's/<\/sub>/~/g' \
		-e 's/<sup>/^/g' -e 's/<\/sup>/^/g' \
		-e 's/----*//g' \
		-e '1s/^/# [$(PROJUPP) Samples](https:\/\/github.com\/hfp\/$(PROJECT)\/raw\/main\/$(DOCDIR)\/$(PROJECT)_samples.pdf)\n\n/' \
		>$@

$(DOCDIR)/$(PROJECT).$(DOCEXT): $(DOCDIR)/.make $(ABSDIR)/$(DOCDIR)/index.md $(ABSDIR)/$(DOCDIR)/$(PROJECT)_scripts.md $(DOCMDS)
	$(eval TMPFILE = $(shell $(MKTEMP) $(ROOTDIR)/$(DOCDIR)/.$(PROJECT)_XXXXXX.tex))
	@pandoc -D latex \
	| $(SED) \
		-e 's/\(\\documentclass\[..*\]{..*}\)/\1\n\\pagenumbering{gobble}\n\\RedeclareSectionCommands[beforeskip=-1pt,afterskip=1pt]{subsection,subsubsection}/' \
		-e 's/\\usepackage{listings}/\\usepackage{listings}\\lstset{basicstyle=\\footnotesize\\ttfamily,showstringspaces=false}/' \
		-e 's/\(\\usepackage.*{hyperref}\)/\\usepackage[hyphens]{url}\n\1/' \
		>$(TMPFILE)
	@cd $(ROOTDIR)/$(DOCDIR) && ( \
		iconv -t utf-8 $(ABSDIR)/$(DOCDIR)/index.md && echo && \
		echo "# $(PROJUPP) Domains" && \
		for DOC in $(DOCMDS); do \
			$(SED) "s/^\(##*\) /#\1 /" $${DOC} | iconv -t utf-8 && echo; \
		done && \
		echo "# Appendix" && \
		$(SED) "s/^\(##*\) /#\1 /" $(ABSDIR)/$(DOCDIR)/$(PROJECT)_scripts.md | iconv -t utf-8; ) \
	| $(SED) \
		-e 's/<sub>/~/g' -e 's/<\/sub>/~/g' \
		-e 's/<sup>/^/g' -e 's/<\/sup>/^/g' \
		-e 's/----*//g' \
	| pandoc \
		--template=$(call qndir,$(TMPFILE)) --listings \
		-f gfm+subscript+superscript \
		-V documentclass=scrartcl \
		-V title-meta="$(PROJUPP) Documentation" \
		-V author-meta="Hans Pabst" \
		-V classoption=DIV=45 \
		-V linkcolor=black \
		-V citecolor=black \
		-V urlcolor=black \
		-o $(call qndir,$@)
	@rm $(TMPFILE)

$(DOCDIR)/$(PROJECT)_samples.$(DOCEXT): $(ROOTDIR)/$(DOCDIR)/$(PROJECT)_samples.md
	$(eval TMPFILE = $(shell $(MKTEMP) .$(PROJECT)_XXXXXX.tex))
	@pandoc -D latex \
	| $(SED) \
		-e 's/\(\\documentclass\[..*\]{..*}\)/\1\n\\pagenumbering{gobble}\n\\RedeclareSectionCommands[beforeskip=-1pt,afterskip=1pt]{subsection,subsubsection}/' \
		-e 's/\\usepackage{listings}/\\usepackage{listings}\\lstset{basicstyle=\\footnotesize\\ttfamily,showstringspaces=false}/' \
		-e 's/\(\\usepackage.*{hyperref}\)/\\usepackage[hyphens]{url}\n\1/' \
		>$(TMPFILE)
	@iconv -t utf-8 $(ROOTDIR)/$(DOCDIR)/$(PROJECT)_samples.md \
	| $(SED) \
		-e 's/\xe2\x88\x92/-/g' \
		-e 's/\xe2\x89\xa4/<=/g' \
		-e 's/\xc2\xb7/*/g' \
		-e 's/\xc3\x97/x/g' \
		-e 's/\xe2\x80\x93/--/g' \
		-e 's/\xe2\x80\x94/---/g' \
		-e 's/\xe2\x80\xa6/.../g' \
		-e 's/\xe2\x86\x92/->/g' \
		-e 's/\xc2\xb2/2/g' \
		-e 's/\xc2\xa0/ /g' \
		-e 's/\xe2\x80\xaf/ /g' \
		-e 's/----*//g' \
	| pandoc \
		--template=$(TMPFILE) --listings \
		-f gfm+subscript+superscript \
		-V documentclass=scrartcl \
		-V title-meta="$(PROJUPP) Sample Code Summary" \
		-V classoption=DIV=45 \
		-V linkcolor=black \
		-V citecolor=black \
		-V urlcolor=black \
		-o $@
	@rm $(TMPFILE)

.PHONY: documentation
documentation: $(DOCDIR)/$(PROJECT).$(DOCEXT) $(DOCDIR)/$(PROJECT)_samples.$(DOCEXT)

.PHONY: mkdocs
mkdocs: $(ROOTDIR)/$(DOCDIR)/index.md
	@mkdocs build --clean
	@mkdocs serve

.PHONY: clean
clean:
ifneq ($(call qapath,$(BLDDIR)),$(ROOTDIR))
ifneq ($(call qapath,$(BLDDIR)),$(HEREDIR))
	@-rm -rf $(BLDDIR)
endif
endif
ifneq (,$(wildcard $(BLDDIR))) # still exists
	@-rm -f $(OBJFILES) $(FTNOBJS)
	@-rm -f $(BLDDIR)/*.gcno $(BLDDIR)/*.gcda $(BLDDIR)/*.gcov
endif

.PHONY: realclean
realclean: clean
ifneq ($(call qapath,$(OUTDIR)),$(ROOTDIR))
ifneq ($(call qapath,$(OUTDIR)),$(HEREDIR))
	@-rm -rf $(OUTDIR)
endif
endif
ifneq (,$(wildcard $(OUTDIR))) # still exists
	@-rm -f $(OUTDIR)/$(PROJECT)*.$(SLIBEXT) $(OUTDIR)/$(PROJECT)*.$(DLIBEXT)*
	@-rm -f $(OUTDIR)/$(PROJECT)*.pc
endif
ifneq ($(call qapath,$(BINDIR)),$(ROOTDIR))
ifneq ($(call qapath,$(BINDIR)),$(HEREDIR))
	@-rm -rf $(BINDIR)
endif
endif
	@-rm -f $(INCDIR)/$(PROJECT)_version.h

.PHONY: deepclean
deepclean: realclean
	@find . -type f \( -name .make -or -name .state \) -exec rm {} \;

.PHONY: distclean
distclean: deepclean
	@find $(ROOTDIR)/$(SPLDIR) $(ROOTDIR)/$(TSTDIR) -type f -name Makefile -exec $(FLOCK) {} \
		"$(MAKE) --no-print-directory deepclean" \; 2>/dev/null || true
	@-rm -rf $(PROJECT)*

# keep original prefix (:)
ALIAS_PREFIX := $(PREFIX)

# DESTDIR is used as prefix of PREFIX
ifneq (,$(strip $(DESTDIR)))
  override PREFIX := $(call qapath,$(DESTDIR)/$(PREFIX))
endif
# fall-back
ifeq (,$(strip $(PREFIX)))
  override PREFIX := $(HEREDIR)
endif

# setup maintainer-layout
ifeq (,$(strip $(ALIAS_PREFIX)))
  override ALIAS_PREFIX := $(PREFIX)
endif
ifneq ($(ALIAS_PREFIX),$(PREFIX))
  PPKGDIR := libdata/pkgconfig
  PMODDIR := $(PSHRDIR)
endif

# remove existing PREFIX
CLEAN ?= 0

.PHONY: install-minimal
install-minimal: $(PROJECT)
ifneq ($(PREFIX),$(ABSDIR))
	@echo
ifneq (0,$(CLEAN))
#ifneq (,$(findstring ?$(HOMEDIR),?$(call qapath,$(PREFIX))))
	@if [ -d $(PREFIX) ]; then echo "$(PROJUPP) removing $(PREFIX)..." && rm -rf $(PREFIX) || true; fi
#endif
endif
	@echo "$(PROJUPP) installing libraries..."
	@$(MKDIR) -p $(PREFIX)/$(POUTDIR)
	@$(CP) -va $(OUTDIR)/$(PROJECT)*.$(DLIBEXT)* $(PREFIX)/$(POUTDIR) 2>/dev/null || true
	@$(CP) -v  $(OUTDIR)/$(PROJECT).$(SLIBEXT)  $(PREFIX)/$(POUTDIR) 2>/dev/null || true
	@echo
	@echo "$(PROJUPP) installing pkg-config files..."
	@$(MKDIR) -p $(PREFIX)/$(PPKGDIR)
	@$(CP) -va $(OUTDIR)/*.pc $(PREFIX)/$(PPKGDIR) 2>/dev/null || true
	@if [ ! -e $(PREFIX)/$(PMODDIR)/$(PROJECT).env ]; then \
		$(MKDIR) -p $(PREFIX)/$(PMODDIR); \
		$(CP) -v $(OUTDIR)/$(PROJECT).env $(PREFIX)/$(PMODDIR) 2>/dev/null || true; \
	fi
	@echo
	@echo "$(PROJUPP) installing interface..."
	@$(CP) -v  $(HEADERS_MAIN) $(PREFIX)/$(PINCDIR) 2>/dev/null || true
	@$(CP) -v  $(INCDIR)/$(PROJECT)_version.h $(PREFIX)/$(PINCDIR) 2>/dev/null || true
	@$(CP) -v  $(INCDIR)/$(PROJECT).h $(PREFIX)/$(PINCDIR) 2>/dev/null || true
	@$(CP) -va $(INCDIR)/*.mod* $(PREFIX)/$(PINCDIR) 2>/dev/null || true
endif

.PHONY: install
install: install-minimal
ifneq ($(PREFIX),$(ABSDIR))
	@echo
	@echo "$(PROJUPP) installing documentation..."
	@$(MKDIR) -p $(PREFIX)/$(PDOCDIR)
	@$(CP) -va $(ROOTDIR)/$(DOCDIR)/*.pdf $(PREFIX)/$(PDOCDIR)
	@$(CP) -va $(ROOTDIR)/$(DOCDIR)/*.md $(PREFIX)/$(PDOCDIR)
	@$(CP) -v  $(ROOTDIR)/SECURITY.md $(PREFIX)/$(PDOCDIR)
	@$(CP) -v  $(ROOTDIR)/version.txt $(PREFIX)/$(PDOCDIR)
	@$(SED) "s/^\"//;s/\\\n\"$$//;/STATIC=/d" $(DIRSTATE)/.state >$(PREFIX)/$(PDOCDIR)/build.txt 2>/dev/null || true
	@$(MKDIR) -p $(PREFIX)/$(LICFDIR)
ifneq ($(call qapath,$(PREFIX)/$(PDOCDIR)/LICENSE.md),$(call qapath,$(PREFIX)/$(LICFDIR)/$(LICFILE)))
	@$(MV) $(PREFIX)/$(PDOCDIR)/LICENSE.md $(PREFIX)/$(LICFDIR)/$(LICFILE)
endif
endif

.PHONY: install-all
install-all: install build-tests
ifneq ($(PREFIX),$(ABSDIR))
	@echo
	@echo "$(PROJUPP) installing tests..."
	@$(MKDIR) -p $(PREFIX)/$(PSHRDIR)/$(PTSTDIR)
	@$(CP) -v $(basename $(wildcard $(ROOTDIR)/$(TSTDIR)/*.c)) $(PREFIX)/$(PSHRDIR)/$(PTSTDIR) 2>/dev/null || true
endif

.PHONY: install-dev
install-dev: install
ifneq ($(PREFIX),$(ABSDIR))
	@if test -t 0; then \
		echo; \
		echo "================================================================================"; \
		echo "Installing development tools does not respect a common PREFIX, e.g., /usr/local."; \
		echo "For development, consider checking out https://github.com/hfp/$(PROJECT),"; \
		echo "or perform plain \"install\" (or \"install-all\")."; \
		echo "Hit CTRL-C to abort, or wait $(WAIT) seconds to continue."; \
		echo "--------------------------------------------------------------------------------"; \
		sleep $(WAIT); \
	fi
	@echo
	@echo "$(PROJUPP) installing utilities..."
	@$(MKDIR) -p $(PREFIX)
	@$(CP) -v $(ROOTDIR)/Makefile.inc $(PREFIX) 2>/dev/null || true
	@$(CP) -v $(ROOTDIR)/.mktmp.sh $(PREFIX) 2>/dev/null || true
	@$(CP) -v $(ROOTDIR)/.flock.sh $(PREFIX) 2>/dev/null || true
	@$(CP) -v $(ROOTDIR)/.state.sh $(PREFIX) 2>/dev/null || true
	@echo
	@echo "$(PROJUPP) tool scripts..."
	@$(MKDIR) -p $(PREFIX)/$(SCRDIR)
	@$(CP) -v $(ROOTSCR)/tool_getenvars.sh $(PREFIX)/$(SCRDIR) 2>/dev/null || true
endif

.PHONY: install-realall
install-realall: install-all install-dev samples
ifneq ($(PREFIX),$(ABSDIR))
	@echo
	@echo "$(PROJUPP) installing samples..."
	@$(MKDIR) -p $(PREFIX)/$(PSHRDIR)/$(SPLDIR)
	@for SAMPLE in $(SAMPLES); do \
		$(CP) -v $(addprefix $(ROOTDIR)/$(SPLDIR)/$${SAMPLE}/,*.x) $(PREFIX)/$(PSHRDIR)/$(SPLDIR) 2>/dev/null || true; \
	done
endif

ALIAS_INCDIR := $(subst $$$$,$(if $(findstring $$$$/,$$$$$(PINCDIR)),,\$${prefix}/),$(subst $$$$$(ALIAS_PREFIX),\$${prefix},$$$$$(PINCDIR)))
ALIAS_LIBDIR := $(subst $$$$,$(if $(findstring $$$$/,$$$$$(POUTDIR)),,\$${prefix}/),$(subst $$$$$(ALIAS_PREFIX),\$${prefix},$$$$$(POUTDIR)))

ifeq (,$(filter-out 0 2,$(BUILD)))
$(OUTDIR)/$(PROJECT)-static.pc: $(OUTDIR)/$(PROJECT).$(SLIBEXT)
	@echo "Name: $(PROJECT)" >$@
	@echo "Description: Specialized tensor operations" >>$@
	@echo "URL: https://github.com/hfp/$(PROJECT)/" >>$@
	@echo "Version: $(VERSION_STRING)" >>$@
	@echo >>$@
	@echo "prefix=$(ALIAS_PREFIX)" >>$@
	@echo "includedir=$(ALIAS_INCDIR)" >>$@
	@echo "libdir=$(ALIAS_LIBDIR)" >>$@
	@echo >>$@
	@echo "Cflags: -I\$${includedir}" >>$@
  ifneq (Windows_NT,$(UNAME))
	@echo "Libs: -L\$${libdir} -l:$(PROJECT).$(SLIBEXT)" >>$@
  else
	@echo "Libs: -L\$${libdir} -lxsmm" >>$@
  endif
  ifneq (,$(LIBXS))
	@echo "Requires.private: libxs-static" >>$@
  endif
  ifeq (,$(filter-out 0 2,$(BUILD)))
	@ln -fs $(notdir $@) $(OUTDIR)/$(PROJECT).pc
  endif
else
.PHONY: $(OUTDIR)/$(PROJECT)-static.pc
endif

ifeq (,$(filter-out 1 2,$(BUILD)))
$(OUTDIR)/$(PROJECT)-shared.pc: $(OUTDIR)/$(PROJECT).$(DLIBEXT)
	@echo "Name: $(PROJECT)" >$@
	@echo "Description: Specialized tensor operations" >>$@
	@echo "URL: https://github.com/hfp/$(PROJECT)/" >>$@
	@echo "Version: $(VERSION_STRING)" >>$@
	@echo >>$@
	@echo "prefix=$(ALIAS_PREFIX)" >>$@
	@echo "includedir=$(ALIAS_INCDIR)" >>$@
	@echo "libdir=$(ALIAS_LIBDIR)" >>$@
	@echo >>$@
	@echo "Cflags: -I\$${includedir}" >>$@
	@echo "Libs: -L\$${libdir} -lxsmm" >>$@
  ifneq (,$(LIBXS))
	@echo "Requires.private: libxs" >>$@
  endif
  ifeq (,$(filter-out 1,$(BUILD)))
	@ln -fs $(notdir $@) $(OUTDIR)/$(PROJECT).pc
  endif
else
.PHONY: $(OUTDIR)/$(PROJECT)-shared.pc
endif

$(OUTDIR)/$(PROJECT).env: $(OUTDIR)/.make $(INCDIR)/$(PROJECT).h
	@echo "#%Module1.0" >$@
	@echo >>$@
	@echo "module-whatis \"$(PROJUPP) $(VERSION_STRING)\"" >>$@
	@echo >>$@
	@echo "set PREFIX \"$(ALIAS_PREFIX)\"" >>$@
	@echo "prepend-path PATH \"\$$PREFIX/bin\"" >>$@
	@echo "prepend-path LD_LIBRARY_PATH \"\$$PREFIX/lib\"" >>$@
	@echo >>$@
	@echo "prepend-path PKG_CONFIG_PATH \"\$$PREFIX/lib\"" >>$@
	@echo "prepend-path LIBRARY_PATH \"\$$PREFIX/lib\"" >>$@
	@echo "prepend-path CPATH \"\$$PREFIX/include\"" >>$@

.PHONY: deb
deb:
	@if [ "$$(command -v git)" ]; then \
		VERSION_ARCHIVE_SONAME=$$($(ROOTSCR)/tool_version.sh $(PROJECT) 1); \
		VERSION_ARCHIVE=$$($(ROOTSCR)/tool_version.sh $(PROJECT) 5); \
	fi; \
	if [ "$${VERSION_ARCHIVE}" ] && [ "$${VERSION_ARCHIVE_SONAME}" ]; then \
		ARCHIVE_AUTHOR_NAME="$$(git config user.name)"; \
		ARCHIVE_AUTHOR_MAIL="$$(git config user.email)"; \
		ARCHIVE_NAME=$(PROJECT)$${VERSION_ARCHIVE_SONAME}; \
		ARCHIVE_DATE="$$(LANG=C date -R)"; \
		if [ "$${ARCHIVE_AUTHOR_NAME}" ] && [ "$${ARCHIVE_AUTHOR_MAIL}" ]; then \
			ARCHIVE_AUTHOR="$${ARCHIVE_AUTHOR_NAME} <$${ARCHIVE_AUTHOR_MAIL}>"; \
		else \
			echo "Warning: Please git-config user.name and user.email!"; \
			if [ "$${ARCHIVE_AUTHOR_NAME}" ] || [ "$${ARCHIVE_AUTHOR_MAIL}" ]; then \
				ARCHIVE_AUTHOR="$${ARCHIVE_AUTHOR_NAME}$${ARCHIVE_AUTHOR_MAIL}"; \
			fi \
		fi; \
		if ! [ -e $${ARCHIVE_NAME}_$${VERSION_ARCHIVE}.orig.tar.gz ]; then \
			git archive --prefix $${ARCHIVE_NAME}-$${VERSION_ARCHIVE}/ \
				-o $${ARCHIVE_NAME}_$${VERSION_ARCHIVE}.orig.tar.gz $(VERSION_RELEASE); \
		fi; \
		tar xf $${ARCHIVE_NAME}_$${VERSION_ARCHIVE}.orig.tar.gz; \
		cd $${ARCHIVE_NAME}-$${VERSION_ARCHIVE}; \
		$(MKDIR) -p debian/source; cd debian/source; \
		echo "3.0 (quilt)" >format; \
		cd ..; \
		echo "Source: $${ARCHIVE_NAME}" >control; \
		echo "Section: libs" >>control; \
		echo "Homepage: https://github.com/hfp/$(PROJECT)/" >>control; \
		echo "Vcs-Git: https://github.com/hfp/$(PROJECT)/$(PROJECT).git" >>control; \
		echo "Maintainer: $${ARCHIVE_AUTHOR}" >>control; \
		echo "Priority: optional" >>control; \
		echo "Build-Depends: debhelper (>= 13)" >>control; \
		echo "Standards-Version: 3.9.8" >>control; \
		echo >>control; \
		echo "Package: $${ARCHIVE_NAME}" >>control; \
		echo "Section: libs" >>control; \
		echo "Architecture: amd64" >>control; \
		echo "Depends: \$${shlibs:Depends}, \$${misc:Depends}" >>control; \
		echo "Description: Specialized tensor operations" >>control; \
		wget -T $(TIMEOUT) -qO- "https://api.github.com/repos/$(PROJECT)/$(PROJECT)" \
		| $(SED) -n 's/ *\"description\": \"\(..*\)\".*/\1/p' \
		| fold -s -w 79 | $(SED) -e 's/^/ /' -e 's/[[:space:]][[:space:]]*$$//' >>control; \
		echo "$${ARCHIVE_NAME} ($${VERSION_ARCHIVE}-$(VERSION_PACKAGE)) UNRELEASED; urgency=low" >changelog; \
		echo >>changelog; \
		wget -T $(TIMEOUT) -qO- "https://api.github.com/repos/$(PROJECT)/$(PROJECT)/releases/tags/$${VERSION_ARCHIVE}" \
		| $(SED) -n 's/ *\"body\": \"\(..*\)\".*/\1/p' \
		| $(SED) -e 's/\\r\\n/\n/g' -e 's/\\"/"/g' -e 's/\[\([^]]*\)\]([^)]*)/\1/g' \
		| $(SED) -n 's/^\* \(..*\)/\* \1/p' \
		| fold -s -w 78 | $(SED) -e 's/^/  /g' -e 's/^  \* /\* /' -e 's/^/  /' -e 's/[[:space:]][[:space:]]*$$//' >>changelog; \
		echo >>changelog; \
		echo " -- $${ARCHIVE_AUTHOR}  $${ARCHIVE_DATE}" >>changelog; \
		echo "#!/usr/bin/make -f" >rules; \
		echo "export DH_VERBOSE = 1" >>rules; \
		echo >>rules; \
		echo "%:" >>rules; \
		$$(which echo) -e "\tdh \$$@" >>rules; \
		echo >>rules; \
		echo "override_dh_auto_install:" >>rules; \
		$$(which echo) -e "\tdh_auto_install -- prefix=/usr" >>rules; \
		echo >>rules; \
		echo "13" >compat; \
		$(CP) ../LICENSE.md copyright; \
		rm -f ../$(TSTDIR)/mhd_test.mhd; \
		chmod +x rules; \
		debuild \
			-e PREFIX=debian/$${ARCHIVE_NAME}/usr \
			-e PDOCDIR=share/doc/$${ARCHIVE_NAME} \
			-e LICFILE=copyright \
			-e LICFDIR=../.. \
			-e SONAMELNK=1 \
			-e SYM=1 \
			-us -uc; \
	else \
		echo "Error: Git is unavailable or make-deb runs outside of cloned repository!"; \
	fi
