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

BLAS = 2
OMP = 1
SYM = 0
OCL = 2

# oneMKL's SYCL interface (see below) calls the *_64 CBLAS names, which MKL's GNU Fortran
# interface lacks: this is a C program, hence MKL's Intel interface serves it either way
SYCL ?= 1
ifneq (0,$(SYCL))
  override MKL_FCRTL := intel
endif

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

# oneMKL's SYCL interface takes stacks too large for the SMM kernels, compiled by the C++
# compiler at hand (no SYCL compiler): on where MKL is used and the SYCL runtime is in reach
# (CMPLR_ROOT, icx, or next to MKLROOT), SYCL=0 disables it. Libraries are linked by path,
# since oneAPI's lib directory also holds an OpenCL loader that must not be picked up.
ifneq (0,$(SYCL))
ifneq (0,$(MKL))
  SYCLROOT ?= $(firstword $(foreach DIR,$(CMPLR_ROOT) $(ICXROOT) $(MKLROOT)/../../compiler/latest, \
    $(if $(wildcard $(DIR)/include/sycl/sycl.hpp),$(if $(wildcard $(DIR)/lib/libsycl.$(DLIBEXT)),$(call qapath,$(DIR))))))
  SYCL_MKLLIB := $(firstword $(wildcard $(MKLROOT)/lib/$(MKL_ARCH)/libmkl_sycl_blas.$(DLIBEXT) \
    $(MKLROOT)/lib/libmkl_sycl_blas.$(DLIBEXT)))
  ifneq (,$(and $(SYCLROOT),$(SYCL_MKLLIB),$(call which,$(CXX))))
    DFLAGS += -DOPENCL_LIBSMM_SYCL
    CXXSOURCS := $(SRCDIR)/smm_sycl.cpp
    # RPATH rather than RUNPATH: the SYCL runtime loads its backend by name
    SYCL_LIBS := $(SYCL_MKLLIB) $(SYCLROOT)/lib/libsycl.$(DLIBEXT) -Wl,--disable-new-dtags \
      -Wl,-rpath,$(call qapath,$(dir $(SYCL_MKLLIB))) -Wl,-rpath,$(SYCLROOT)/lib
  endif
endif
endif

# include directories
IFLAGS += -I$(call quote,$(DEPDIR))
IFLAGS += -I$(call quote,$(INCDIR))

# GPU-specific tuning parameters
WITH_GPU := $(if $(WITH_GPU),$(WITH_GPU),$(GPUVER))
PARAMS_WITHGPU := $(SRCDIR)/params/tune_multiply_$(WITH_GPU).csv
PARAMS_DEFAULT := $(SRCDIR)/tune_multiply.csv
PARAMS := $(if $(wildcard $(PARAMS_WITHGPU)),$(PARAMS_WITHGPU),$(wildcard $(PARAMS_DEFAULT)))

PARAMDIR ?= $(SRCDIR)/params
ifeq (,$(PARAMS))
ifneq (command line,$(origin WITH_GPU))
  PARAMS := $(wildcard $(PARAMDIR)/*.csv)
endif
endif
ifneq (command line,$(origin WITH_GPU))
  PARAMS_BIN := $(wildcard $(PARAMDIR)/*.bin)
endif
PARAMS_PREDICT := $(sort $(PARAMS) $(patsubst %.bin,%.csv,$(PARAMS_BIN)))

# kernel source and generated header
SCRDIR := $(DEPDIR)/scripts
KRNELS := $(wildcard $(SRCDIR)/kernels/*.cl)
GENHDR := $(SRCDIR)/smm_kernels.h
GENSCR := $(SCRDIR)/tool_opencl.sh

ifneq (0,$(SYM))
  GENARG += -d
endif

OUTNAME := acc_bench
# header-only builds also depend on the sources libxs_source.h and libxstream_source.h include
HEADERS := $(wildcard $(INCDIR)/*.h) $(GENHDR) \
           $(wildcard $(DEPDIR)/libxstream/*.h) \
           $(wildcard $(DEPDIR)/libxstream/opencl/*.h) \
           $(wildcard $(LIBXSROOT)/libxs/*.h) \
           $(LIBXS_SRCDEP) $(XSTREAM_SRCDEP)
CSOURCS := $(SRCDIR)/acc_bench.c $(SRCDIR)/smm_acc.c $(SRCDIR)/smm_trans.c $(SRCDIR)/smm_params.c $(SRCDIR)/smm_kernel.c
COBJCTS := $(patsubst %,$(BLDDIR)/%,$(notdir $(CSOURCS:.c=-c.o)))
XOBJCTS := $(patsubst %,$(BLDDIR)/%,$(notdir $(CXXSOURCS:.cpp=-cpp.o)))
OBJECTS := $(COBJCTS) $(XOBJCTS)
XFILES := $(OUTDIR)/$(OUTNAME).x

.PHONY: all
all: $(XFILES)

.PHONY: compile
compile: $(OBJECTS)

# generate embedded kernel header from .cl sources
$(GENHDR): $(KRNELS) $(PARAMS_PREDICT) $(PARAMS_BIN) $(GENSCR) Makefile $(LIBXSROOT)/Makefile.inc
	$(GENSCR) $(GENARG) $(KRNELS) $(PARAMS_PREDICT) $@

$(OUTDIR)/$(OUTNAME).x: $(OUTDIR)/.make $(OBJECTS) $(XLIB) $(LIBXS)
	$(LD) -o $@ $(OBJECTS) $(SYCL_LIBS) $(XLIB) $(LIBXS) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS) $(LIBS)

.PHONY: match
match: $(OUTDIR)/acc_match.x

$(OUTDIR)/acc_match.x: $(OUTDIR)/.make $(BLDDIR)/acc_match-c.o $(XLIB) $(LIBXS)
	$(LD) -o $@ $(BLDDIR)/acc_match-c.o $(XLIB) $(LIBXS) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)

$(BLDDIR)/%-c.o: $(SRCDIR)/%.c .state $(BLDDIR)/.make $(HEADERS) Makefile $(LIBXSROOT)/Makefile.inc
	$(CC) $(DFLAGS) $(IFLAGS) $(CFLAGS) $(CTARGET) -c $< -o $@

# the SYCL headers as system headers: warnings stay with this file, its own CL headers with it too
$(BLDDIR)/%-cpp.o: $(SRCDIR)/%.cpp .state $(BLDDIR)/.make $(SRCDIR)/smm_sycl.h Makefile $(LIBXSROOT)/Makefile.inc
	$(CXX) $(DFLAGS) -isystem $(call quote,$(SYCLROOT)/include) $(CXXFLAGS) $(CTARGET) -std=c++17 -c $< -o $@

.PHONY: test
test: $(XFILES)
	$(OUTDIR)/$(OUTNAME).x

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
	@-rm -f $(XFILES) $(OUTDIR)/acc_match.x
endif
	@-rm -f $(GENHDR)

.PHONY: reformat
reformat: $(wildcard $(SRCDIR)/acc_.c) $(wildcard $(INCDIR)/acc_*.h)
	$(SCRDIR)/tool_clangformat.sh -style=file:$(SRCDIR)/.clang-format -i $^

.PHONY: deepclean
deepclean: realclean
	@-rm -f .make .state

# Relink when only the link flags changed: those are tracked apart from .state so
# that a link-only change no longer recompiles every object, which leaves the link
# as the one step that still has to see them.
$(XFILES): $(LNKSTATE)
