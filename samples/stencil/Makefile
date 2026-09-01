ROOTDIR = $(abspath $(dir $(firstword $(MAKEFILE_LIST))))
PROJECT := libxstream
DEPDIR = $(ROOTDIR)/../..
SRCDIR = .
INCDIR = .
BLDDIR = obj
OUTDIR = .

CXXFLAGS = $(NULL)
CFLAGS = $(NULL)
DFLAGS = $(NULL)

BLAS = 0
OMP = 1
SYM = 0

# CPU=1 runs the stencil on the host: the C compiler translates
# kernels/stencil_fp32.cl and neither OpenCL nor LIBXSTREAM is needed.
# The kernel is specialized at build time, so CPUDEF carries its
# configuration, e.g. CPUDEF="-DSTENCIL_NX=512 -DSTENCIL_NY=512 -DSTENCIL_NZ=512".
CPU ?= 0
CPUDEF ?= $(NULL)
ifeq (0,$(CPU))
  OCL = 2
  TESTARG ?= -n 32 -t 4 -d 3
else
  OCL = 0
  TESTARG ?= -n 256 -t 4 -d 3
endif

# root directory of LIBXS
LIBXSROOT := $(wildcard $(DEPDIR)/../libxs)

# include common Makefile artifacts
include $(LIBXSROOT)/Makefile.inc

# setup LIBXS (library or header-only)
ifneq (,$(LIBXSROOT))
  LIBXS_SL := $(wildcard $(LIBXSROOT)/lib/libxs.$(SLIBEXT))
  LIBXS_DL := $(wildcard $(LIBXSROOT)/lib/libxs.$(DLIBEXT))
  LIBXS := $(wildcard $(LIBXSROOT)/lib/libxs.$(LIBEXT))
  LIBXS := $(strip $(if $(LIBXS),$(LIBXS), \
    $(if $(LIBXS_SL),$(LIBXS_SL),$(LIBXS_DL))))
  IFLAGS += -I$(call quote,$(LIBXSROOT))
  IFLAGS += -I$(call quote,$(LIBXSROOT)/samples/stencil)
  ifneq (,$(LIBXS))
    DFLAGS += -D__LIBXS
  else ifneq (,$(wildcard $(LIBXSROOT)/libxs/libxs_source.h))
    DFLAGS += -D__LIBXS -DLIBXS_SOURCE
  endif
endif

# setup LIBXSTREAM (library or header-only), or select the host path
ifneq (0,$(CPU))
  DFLAGS += -DSTENCIL_CPU=1 $(CPUDEF)
  XLIB := $(NULL)
else ifneq (,$(XLIB))
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

SCRDIR := $(DEPDIR)/scripts
KRNELS := $(SRCDIR)/kernels/stencil_bf16.cl \
          $(SRCDIR)/kernels/stencil_fp32.cl \
          $(SRCDIR)/kernels/stencil_int8.cl \
          $(NULL)
KRNDEP := $(KRNELS) \
          $(SRCDIR)/kernels/stencil_common.cl \
          $(NULL)
GENHDR := $(SRCDIR)/stencil_kernels.h
GENSCR := $(SCRDIR)/tool_opencl.sh
GENARG := -p ""

ifneq (0,$(SYM))
  GENARG += -d
endif

OUTNAME := $(shell basename "$(ROOTDIR)")
# The host path includes the kernels directly instead of embedding them.
HEADERS := $(wildcard $(INCDIR)/*.h) \
           $(if $(filter 0,$(CPU)),$(GENHDR),$(KRNDEP)) \
           $(wildcard $(DEPDIR)/libxstream/*.h) \
           $(wildcard $(DEPDIR)/libxstream/opencl/*.h) \
           $(wildcard $(LIBXSROOT)/libxs/*.h)
CSOURCS := $(wildcard $(SRCDIR)/*.c)
ifneq (0,$(CPU))
  CSOURCS := $(filter-out $(SRCDIR)/stencil_opencl.c,$(CSOURCS))
endif
COBJCTS := $(patsubst %,$(BLDDIR)/%,$(notdir $(CSOURCS:.c=-c.o)))
SOURCES := $(CSOURCS)
OBJECTS := $(COBJCTS)
XFILES := $(OUTDIR)/$(OUTNAME).x

.PHONY: all
all: $(XFILES)

.PHONY: compile
compile: $(OBJECTS)

# Generate embedded kernel header from .cl sources (resolves #include directives)
$(GENHDR): $(KRNDEP) $(wildcard $(DEPDIR)/libxstream/opencl/*.h) $(GENSCR) Makefile $(LIBXSROOT)/Makefile.inc
	$(GENSCR) $(GENARG) $(KRNELS) $@

$(OUTDIR)/$(OUTNAME).x: $(OUTDIR)/.make $(OBJECTS) $(XLIB) $(LIBXS)
	$(LD) -o $@ $(OBJECTS) $(XLIB) $(LIBXS) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS) $(LIBS)

# The host build of the kernel keeps the device signature, which leaves the
# runtime extents unused wherever the layout supplies them at compile time.
$(BLDDIR)/stencil_cpu-c.o: CFLAGS += $(call ctest,$(CC),-Wno-unused-parameter)

$(BLDDIR)/%-c.o: $(SRCDIR)/%.c .state $(BLDDIR)/.make $(HEADERS) Makefile $(LIBXSROOT)/Makefile.inc
	$(CC) $(DFLAGS) $(IFLAGS) $(CFLAGS) $(CTARGET) -c $< -o $@

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
