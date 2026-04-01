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

# root directory of LIBXS
LIBXSROOT := $(wildcard $(DEPDIR)/../libxs)

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

# include directories
IFLAGS += -I$(call quote,$(DEPDIR)/include)
IFLAGS += -I$(call quote,$(INCDIR))

# GPU-specific tuning parameters
WITH_GPU := $(if $(WITH_GPU),$(WITH_GPU),$(GPUVER))
PARAMS_WITHGPU := $(SRCDIR)/params/tune_multiply_$(WITH_GPU).csv
PARAMS_DEFAULT := $(SRCDIR)/tune_multiply.csv
PARAMS := $(if $(wildcard $(PARAMS_WITHGPU)),$(PARAMS_WITHGPU),$(wildcard $(PARAMS_DEFAULT)))

PARAMDIR ?= $(SRCDIR)/params
ifeq (,$(PARAMS))
  PARAMS := $(wildcard $(PARAMDIR)/*.csv)
endif

# kernel source and generated header
SCRDIR := $(DEPDIR)/scripts
KRNELS := $(wildcard $(SRCDIR)/kernels/*.cl)
GENHDR := $(SRCDIR)/opencl_kernels.h

OUTNAME := acc_bench
HEADERS := $(wildcard $(INCDIR)/*.h) $(GENHDR) \
           $(wildcard $(DEPDIR)/include/*.h) \
           $(wildcard $(DEPDIR)/include/opencl/*.h)
CSOURCS := $(SRCDIR)/acc_bench.c $(SRCDIR)/opencl_libsmm.c
COBJCTS := $(patsubst %,$(BLDDIR)/%,$(notdir $(CSOURCS:.c=-c.o)))
OBJECTS := $(COBJCTS)
XFILES := $(OUTDIR)/$(OUTNAME).x

.PHONY: all
all: $(XFILES)

.PHONY: compile
compile: $(OBJECTS)

# generate embedded kernel header from .cl sources
$(GENHDR): $(KRNELS) $(PARAMS) $(SCRDIR)/tool_opencl.sh Makefile $(LIBXSROOT)/Makefile.inc
	$(SCRDIR)/tool_opencl.sh $(KRNELS) $(PARAMS) $@

$(OUTDIR)/$(OUTNAME).x: $(OUTDIR)/.make $(OBJECTS) $(XLIB) $(LIBXS)
	$(LD) -o $@ $(OBJECTS) $(XLIB) $(LIBXS) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS) $(LIBS)

$(BLDDIR)/%-c.o: $(SRCDIR)/%.c .state $(BLDDIR)/.make $(HEADERS) Makefile $(LIBXSROOT)/Makefile.inc
	$(CC) $(DFLAGS) $(IFLAGS) $(CFLAGS) $(CTARGET) -c $< -o $@

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
	@-rm -f $(XFILES)
endif
	@-rm -f $(GENHDR)

.PHONY: reformat
reformat: $(CSOURCS) $(wildcard $(INCDIR)/*.h)
	$(SCRDIR)/tool_clangformat.sh -style=file:$(SRCDIR)/.clang-format -i $^

.PHONY: deepclean
deepclean: realclean
	@-rm -f .make .state
