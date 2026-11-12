ROOTDIR = $(abspath $(dir $(firstword $(MAKEFILE_LIST))))
PROJECT := libxstream
DEPDIR = $(ROOTDIR)/../..
SRCDIR = .
INCDIR = .
BLDDIR = obj
OUTDIR = .

CXXFLAGS = $(NULL)
CFLAGS = -Wno-format-nonliteral
DFLAGS = $(if $(ELEM_TYPE),-DELEM_TYPE=$(ELEM_TYPE))

BLAS = 0
OMP = 1
SYM = 0
OCL = 2

# arguments of mixes.x for the test target: stream elements, GEMM order, atomic slots, timed launches
TESTARG ?= $(NULL)

# root directory of LIBXS
LIBXSROOT := $(wildcard $(DEPDIR)/../libxs)

# include common Makefile artifacts
include $(LIBXSROOT)/Makefile.inc

# setup LIBXS (library or header-only)
ifneq (,$(LIBXSROOT))
  LIBXS_LIBDIR := $(call archlibdir,$(LIBXSROOT),libxs)
  LIBXS_SL := $(call libexists,$(wildcard $(LIBXS_LIBDIR)/libxs.$(SLIBEXT)))
  LIBXS_DL := $(call libexists,$(wildcard $(LIBXS_LIBDIR)/libxs.$(DLIBEXT)))
  LIBXS := $(call libexists,$(wildcard $(LIBXS_LIBDIR)/libxs.$(LIBEXT)))
  LIBXS := $(strip $(if $(LIBXS),$(LIBXS), \
    $(if $(LIBXS_SL),$(LIBXS_SL),$(LIBXS_DL))))
  IFLAGS += -I$(call quote,$(LIBXSROOT))
  ifneq (,$(LIBXS))
    DFLAGS += -D__LIBXS
  else ifneq (,$(wildcard $(LIBXSROOT)/libxs/libxs_source.h))
    DFLAGS += -D__LIBXS -DLIBXS_SOURCE
    LIBXS_SRCDEP := $(wildcard $(LIBXSROOT)/src/*.c) $(wildcard $(LIBXSROOT)/src/*.h)
  endif
endif

# setup LIBXSTREAM (library or header-only)
ifneq (,$(XLIB))
ifneq (,$(LIBXS))
  DFLAGS += -D__LIBXSTREAM
else
  DFLAGS += -D__LIBXSTREAM -DLIBXSTREAM_SOURCE
  XSTREAM_SRCDEP := $(wildcard $(DEPDIR)/src/*.c) $(wildcard $(DEPDIR)/src/*.h)
endif
else
  XSTREAM_SOURCE := $(wildcard $(DEPDIR)/libxstream/libxstream_source.h)
  ifneq (,$(XSTREAM_SOURCE))
    DFLAGS += -D__LIBXSTREAM -DLIBXSTREAM_SOURCE
    XSTREAM_SRCDEP := $(wildcard $(DEPDIR)/src/*.c) $(wildcard $(DEPDIR)/src/*.h)
  endif
endif

# include directories
IFLAGS += -I$(call quote,$(DEPDIR))
IFLAGS += -I$(call quote,$(INCDIR))

# kernel source and generated header
SCRDIR := $(DEPDIR)/scripts
KRNELS := $(SRCDIR)/kernels/mixes.cl
GENHDR := $(SRCDIR)/mixes_kernels.h
GENSCR := $(SCRDIR)/tool_opencl.sh
GENARG := -p "" -I $(call quote,$(DEPDIR)/libxstream)

ifneq (0,$(SYM))
  GENARG += -d
endif

OUTNAME := mixes
# header-only builds also depend on the sources libxs_source.h and libxstream_source.h include
HEADERS := $(wildcard $(INCDIR)/*.h) $(GENHDR) \
           $(wildcard $(DEPDIR)/libxstream/*.h) \
           $(wildcard $(DEPDIR)/libxstream/opencl/*.h) \
           $(wildcard $(LIBXSROOT)/libxs/*.h) \
           $(LIBXS_SRCDEP) $(XSTREAM_SRCDEP)
CSOURCS := $(SRCDIR)/mixes.c
COBJCTS := $(patsubst %,$(BLDDIR)/%,$(notdir $(CSOURCS:.c=-c.o)))
OBJECTS := $(COBJCTS)
XFILES := $(OUTDIR)/$(OUTNAME).x

.PHONY: all
all: $(XFILES)

.PHONY: compile
compile: $(OBJECTS)

# generate embedded kernel header from .cl sources
$(GENHDR): $(KRNELS) $(wildcard $(DEPDIR)/libxstream/opencl/*.h) $(GENSCR) Makefile $(LIBXSROOT)/Makefile.inc
	$(GENSCR) $(GENARG) $(KRNELS) $@

$(OUTDIR)/$(OUTNAME).x: $(OUTDIR)/.make $(OBJECTS) $(XLIB) $(LIBXS)
	$(LD) -o $@ $(OBJECTS) $(XLIB) $(LIBXS) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS) $(LIBS)

$(BLDDIR)/%-c.o: $(SRCDIR)/%.c .state $(BLDDIR)/.make $(HEADERS) Makefile $(LIBXSROOT)/Makefile.inc
	$(CC) $(DFLAGS) $(IFLAGS) $(CFLAGS) $(CTARGET) -c $< -o $@

# every mix validates its result (exit status)
.PHONY: test
test: $(XFILES)
	$(OUTDIR)/$(OUTNAME).x $(TESTARG)

.PHONY: clean
clean:
ifneq ($(call qapath,$(BLDDIR)),$(ROOTDIR))
ifneq ($(call qapath,$(BLDDIR)),$(call qapath,.))
	@-rm -rf $(BLDDIR)
endif
endif
ifneq (,$(wildcard $(BLDDIR))) # still exists
	@-rm -f $(OBJECTS)
endif

.PHONY: realclean
realclean: clean
ifneq ($(call qapath,$(OUTDIR)),$(ROOTDIR))
ifneq ($(call qapath,$(OUTDIR)),$(call qapath,.))
	@-rm -rf $(OUTDIR)
endif
endif
ifneq (,$(wildcard $(OUTDIR))) # still exists
	@-rm -f $(XFILES)
endif
	@-rm -f $(GENHDR)

.PHONY: deepclean
deepclean: realclean
	@-rm -f .make .state

# Relink when only the link flags changed: those are tracked apart from .state so
# that a link-only change no longer recompiles every object, which leaves the link
# as the one step that still has to see them.
$(XFILES): $(LNKSTATE)
