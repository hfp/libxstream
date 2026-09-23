ROOTDIR = $(abspath $(dir $(firstword $(MAKEFILE_LIST))))
PROJECT := libxstream
DEPDIR = $(ROOTDIR)/../..
SRCDIR = .
INCDIR = .
BLDDIR = obj
OUTDIR = .

CXXFLAGS = $(NULL)
CFLAGS = -Wno-format-nonliteral
DFLAGS = $(NULL)

BLAS = 2
OMP = 1
SYM = 0
OCL = 2

# root directory of LIBXS
LIBXSROOT := $(wildcard $(DEPDIR)/../libxs)
# CP2K source tree (miniapp target only)
CP2K_ROOT ?= $(wildcard $(DEPDIR)/../cp2k)

# include common Makefile artifacts
include $(LIBXSROOT)/Makefile.inc

# setup LIBXS (library or header-only)
ifneq (,$(LIBXSROOT))
  LIBXS_SL := $(call libexists,$(wildcard $(LIBXSROOT)/lib/libxs.$(SLIBEXT)))
  LIBXS_DL := $(call libexists,$(wildcard $(LIBXSROOT)/lib/libxs.$(DLIBEXT)))
  LIBXS := $(call libexists,$(wildcard $(LIBXSROOT)/lib/libxs.$(LIBEXT)))
  LIBXS := $(strip $(if $(LIBXS),$(LIBXS), \
    $(if $(LIBXS_SL),$(LIBXS_SL),$(LIBXS_DL))))
  IFLAGS += -I$(call quote,$(LIBXSROOT))
  ifneq (,$(LIBXS))
    DFLAGS += -D__LIBXS
  else ifneq (,$(wildcard $(LIBXSROOT)/libxs/libxs_source.h))
    DFLAGS += -D__LIBXS -DLIBXS_SOURCE
  endif
endif

# setup LIBXSTREAM (library or header-only)
ifneq (,$(XLIB))
ifneq (,$(LIBXS))
  DFLAGS += -D__LIBXSTREAM
else
  DFLAGS += -D__LIBXSTREAM -DLIBXSTREAM_SOURCE
endif
else
  XSTREAM_SOURCE := $(wildcard $(DEPDIR)/libxstream/libxstream_source.h)
  ifneq (,$(XSTREAM_SOURCE))
    DFLAGS += -D__LIBXSTREAM -DLIBXSTREAM_SOURCE
  endif
endif

# include directories
IFLAGS += -I$(call quote,$(DEPDIR))
IFLAGS += -I$(call quote,$(INCDIR))

# kernel source and generated header
SCRDIR := $(DEPDIR)/scripts
KRNELS := $(SRCDIR)/kernels/dbm_multiply.cl
GENHDR := $(SRCDIR)/dbm_kernels.h
GENSCR := $(SCRDIR)/tool_opencl.sh
GENARG := -p "" -I $(call quote,$(DEPDIR)/libxstream)

ifneq (0,$(SYM))
  GENARG += -d
endif

# acc_bench and LIBSMM come from the SMM sample, which owns their build knobs
SMMDIR := $(ROOTDIR)/../smm
SMMOBJ := $(addprefix $(SMMDIR)/obj/,acc_bench-c.o smm_acc-c.o smm_trans-c.o smm_params-c.o smm_kernel-c.o)

OUTNAME := dbm_bench
HEADERS := $(wildcard $(INCDIR)/*.h) $(GENHDR) \
           $(wildcard $(DEPDIR)/libxstream/*.h) \
           $(wildcard $(DEPDIR)/libxstream/opencl/*.h) \
           $(wildcard $(LIBXSROOT)/libxs/*.h)
CSOURCS := $(SRCDIR)/dbm_opencl.c
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

.PHONY: smm
smm:
	@$(MAKE) -C $(SMMDIR) compile

$(SMMOBJ): smm

$(OUTDIR)/$(OUTNAME).x: $(OUTDIR)/.make $(OBJECTS) $(SMMOBJ) $(XLIB) $(LIBXS)
	$(LD) -o $@ $(OBJECTS) $(SMMOBJ) $(XLIB) $(LIBXS) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS) $(LIBS)

$(BLDDIR)/%-c.o: $(SRCDIR)/%.c .state $(BLDDIR)/.make $(HEADERS) Makefile $(LIBXSROOT)/Makefile.inc
	$(CC) $(DFLAGS) $(IFLAGS) $(CFLAGS) $(CTARGET) -c $< -o $@

# a negative DBM_MULTIPLY_SMM routes the homogeneous LIBSMM stacks into the DBM kernel
.PHONY: test
test: $(XFILES)
	DBM_MULTIPLY_SMM=-1 $(OUTDIR)/$(OUTNAME).x

# CP2K builds dbm_miniapp.x, taking the DBM backend from this directory
.PHONY: miniapp
miniapp:
	$(if $(CP2K_ROOT),,$(error CP2K_ROOT is not set))
	@$(MAKE) -C $(CP2K_ROOT)/src/dbm OPENCL=1 \
		LIBXSTREAMROOT=$(call quote,$(abspath $(DEPDIR))) \
		LIBXSROOT=$(call quote,$(abspath $(LIBXSROOT)))

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
