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
  IFLAGS += -I$(call quote,$(LIBXSROOT)/samples/ozaki)
  ifneq (,$(LIBXS))
    DFLAGS += -D__LIBXS
  endif
endif

# setup LIBXSTREAM (header-only)
XSTREAM_SOURCE := $(wildcard $(DEPDIR)/include/libxstream_source.h)
ifneq (,$(XSTREAM_SOURCE))
  DFLAGS += -D__LIBXSTREAM -DLIBXSTREAM_SOURCE
endif

# Embed precompiled TinyTC kernels if .clx files are present.
# Filenames encode BM x BN, e.g. ozaki1_f64_256x128.clx.
# Exclude production variants (*_tri.clx, *_sq.clx) loaded at runtime.
OZAKI_CLX_F64 := $(filter-out %_tri.clx %_sq.clx, \
  $(wildcard $(SRCDIR)/kernels/ozaki1_f64_*x*.clx))
OZAKI_CLX_F32 := $(filter-out %_tri.clx %_sq.clx, \
  $(wildcard $(SRCDIR)/kernels/ozaki1_f32_*x*.clx))
ifneq (,$(OZAKI_CLX_F64))
ifneq (,$(OZAKI_CLX_F32))
  OZAKI_BMBN := $(lastword $(subst _, ,$(basename $(notdir $(OZAKI_CLX_F64)))))
  DFLAGS += -DOZAKI_TINYTC_EMBED \
    -DOZAKI_TINYTC_BM=$(word 1,$(subst x, ,$(OZAKI_BMBN))) \
    -DOZAKI_TINYTC_BN=$(word 2,$(subst x, ,$(OZAKI_BMBN)))
endif
endif

# Production .clx files (specialized per ndecomp/trim/scheme)
OZAKI_CLX_PROD := $(wildcard $(SRCDIR)/kernels/ozaki1_*_n*t*_tri.clx \
                             $(SRCDIR)/kernels/ozaki1_*_n*t*_sq.clx)
GENHDR_PROD := $(SRCDIR)/ozaki_tinytc.h

# include directories
IFLAGS += -I$(call quote,$(DEPDIR)/include)
IFLAGS += -I$(call quote,$(INCDIR))

SCRDIR := $(DEPDIR)/scripts
KRNELS := $(SRCDIR)/kernels/ozaki1_int8.cl \
          $(SRCDIR)/kernels/ozaki2_int8.cl \
          $(SRCDIR)/kernels/gemm3m.cl \
          $(NULL)
GENHDR := $(SRCDIR)/ozaki_kernels.h
GENSCR := $(SCRDIR)/tool_opencl.sh
GENARG := -p ""

ifneq (0,$(SYM))
  GENARG += -d
endif

OUTNAME := $(shell basename "$(ROOTDIR)")
HEADERS := $(wildcard $(INCDIR)/*.h) \
           $(GENHDR) $(GENHDR_PROD) \
           $(wildcard $(DEPDIR)/include/common/*.h)
CSOURCS := $(wildcard $(SRCDIR)/*.c)
COBJCTS := $(patsubst %,$(BLDDIR)/%,$(notdir $(CSOURCS:.c=-c.o)))
SOURCES := $(CSOURCS)
OBJECTS := $(COBJCTS)
XFILES := $(OUTDIR)/$(OUTNAME).x

.PHONY: all
all: $(XFILES)

.PHONY: compile
compile: $(OBJECTS)

# Generate embedded kernel header from .cl sources (resolves #include directives)
$(GENHDR): $(KRNELS) $(wildcard $(DEPDIR)/include/common/*.h) $(GENSCR) Makefile $(LIBXSROOT)/Makefile.inc
	$(GENSCR) $(GENARG) $(KRNELS) $@

# Generate production TinyTC kernel header (INCBIN + lookup table).
# Depends on the prod .clx files; if none exist the header is a no-op stub.
$(GENHDR_PROD): $(OZAKI_CLX_PROD) $(SRCDIR)/kernels/ozaki1.sh
	$(SRCDIR)/kernels/ozaki1.sh -h $@

$(OUTDIR)/$(OUTNAME).x: $(OUTDIR)/.make $(OBJECTS) $(XLIB) $(LIBXS)
	$(LD) -o $@ $(OBJECTS) $(XLIB) $(LIBXS) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS) $(LIBS)

$(BLDDIR)/%-c.o: $(SRCDIR)/%.c .state $(BLDDIR)/.make $(HEADERS) Makefile $(LIBXSROOT)/Makefile.inc
	$(CC) $(DFLAGS) $(IFLAGS) $(CFLAGS) $(CTARGET) -c $< -o $@

.PHONY: test
test: $(XFILES)
	$(OUTDIR)/$(OUTNAME).x 64 64 64

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
	@-rm -f $(GENHDR_PROD) $(GENHDR)

.PHONY: deepclean
deepclean: realclean
	@-rm -f .make .state
