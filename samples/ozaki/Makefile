ROOTDIR = $(abspath $(dir $(firstword $(MAKEFILE_LIST))))
DEPDIR = ../..
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

# include common Makefile artifacts
include $(DEPDIR)/Makefile.inc

# root directory of LIBXS
LIBXSROOT := $(DEPDIR)/../libxs

# necessary include directories
IFLAGS += -I$(call quote,$(LIBXSROOT)/include)
IFLAGS += -I$(call quote,$(LIBXSROOT)/samples/ozaki)
IFLAGS += -I$(call quote,$(DEPDIR)/include)
IFLAGS += -I$(call quote,$(INCDIR))

# ACC/OpenCL backend library (libxstream)
XSTLIB := $(DEPDIR)/lib/libxstream.$(SLIBEXT)

SCRDIR := $(DEPDIR)/scripts
KERNEL := $(SRCDIR)/kernels/ozaki1_int8.cl
GENHDR := $(SRCDIR)/ozaki_kernels.h

OUTNAME := $(shell basename "$(ROOTDIR)")
HEADERS := $(wildcard $(INCDIR)/*.h) \
           $(GENHDR) \
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

# Generate embedded kernel header from .cl source (resolves #include directives)
$(GENHDR): $(KERNEL) $(wildcard $(DEPDIR)/include/common/*.h) $(SCRDIR)/acc_opencl.sh Makefile $(DEPDIR)/Makefile.inc
	$(SCRDIR)/acc_opencl.sh -p "" $(KERNEL) $@

$(OUTDIR)/$(OUTNAME).x: $(OUTDIR)/.make $(OBJECTS) $(XSTLIB)
	$(LD) -o $@ $(OBJECTS) $(XSTLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS) $(LIBS)

$(BLDDIR)/%-c.o: $(SRCDIR)/%.c .state $(BLDDIR)/.make $(HEADERS) Makefile $(DEPDIR)/Makefile.inc
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
	@-rm -f $(GENHDR)

.PHONY: deepclean
deepclean: realclean
	@-rm -f .make .state
