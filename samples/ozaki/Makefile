MAKDIR := $(subst //,,$(dir $(firstword $(MAKEFILE_LIST)))/)
ROOTDIR := $(MAKDIR)/../..
SCRDIR := $(ROOTDIR)/scripts

# Kernel source and include path
KERNEL := $(MAKDIR)/kernels/ozaki.cl
GENHDR := $(MAKDIR)/ozaki_kernels.h
INCDIR := $(ROOTDIR)/include

# Output
OUTNAME := ozaki_bench

CC ?= gcc
AR ?= ar

UNAME := $(shell uname)

CFLAGS := -fPIC \
  -Wall -Wextra -Wcast-qual \
  -Wno-overlength-strings \
  -Wno-variadic-macros \
  -Wno-long-long \
  $(NULL)

ifneq (0,$(DBG))
  ifeq (,$(DBG))
    CFLAGS += -O2 -DNDEBUG
  else
    ifneq (1,$(DBG))
      CFLAGS += -D_DEBUG
    endif
    CFLAGS += -O0 -g
  endif
else
  CFLAGS += -O2 -DNDEBUG
endif

ifneq (0,$(OMP))
  ifneq (Darwin,$(UNAME))
    CFLAGS += -fopenmp
    LDFLAGS += -fopenmp
  endif
endif

# OpenCL linkage
ifeq (Darwin,$(UNAME))
  LDFLAGS += -framework OpenCL
else
  OPENCL_LIB := $(shell ldconfig -p 2>/dev/null | grep -m1 OpenCL | rev | cut -d' ' -f1 | rev)
  ifeq (,$(OPENCL_LIB))
    OPENCL_LIB := $(wildcard /usr/lib/x86_64-linux-gnu/libOpenCL.so.1)
  endif
  # Check for oneAPI/CUDA toolkit OpenCL includes
  ifneq (,$(wildcard $(OPENCL_INC)))
    CFLAGS += -I$(OPENCL_INC)
  else
    ICX := $(shell which icx 2>/dev/null)
    ifneq (,$(ICX))
      OPENCL_ROOT := $(abspath $(dir $(ICX))/..)
      CLINC := $(wildcard $(OPENCL_ROOT)/include/sycl/CL/cl.h $(OPENCL_ROOT)/include/CL/cl.h)
      ifneq (,$(CLINC))
        CFLAGS += -I$(abspath $(dir $(firstword $(CLINC)))/..)
      endif
    endif
  endif
  ifneq (,$(wildcard $(OPENCL_LIB)))
    LDFLAGS += $(OPENCL_LIB)
  else
    LDFLAGS += -l:libOpenCL.so.1
  endif
endif

LDFLAGS += -lm

# Collect LD_LIBRARY_PATH
LD_LIBRARY_DIRS := $(wildcard $(subst :, ,$(LD_LIBRARY_PATH)))
LD_LIBSTUB_PATH := $(wildcard $(patsubst %,%/stubs,$(LD_LIBRARY_DIRS)))
LIBPATHS := $(foreach DIR,$(LD_LIBRARY_DIRS),$(if $(filter -L$(DIR),$(LDFLAGS)),$(NULL),-L$(DIR)))
LIBSTUBS := $(foreach DIR,$(LD_LIBSTUB_PATH),$(if $(filter -L$(DIR),$(LDFLAGS)),$(NULL),-L$(DIR)))
LDFLAGS += $(LIBPATHS) $(LIBSTUBS)

.PHONY: all
all: $(MAKDIR)/$(OUTNAME)

# Generate embedded kernel header from .cl source (resolves #include directives)
$(GENHDR): $(KERNEL) $(wildcard $(INCDIR)/common/*.h) $(SCRDIR)/acc_opencl.sh $(MAKDIR)/Makefile
	@echo "Generating $@ from $<"
	$(SCRDIR)/acc_opencl.sh -p "" $(KERNEL) $@

$(MAKDIR)/ozaki_bench.o: $(MAKDIR)/ozaki_bench.c $(MAKDIR)/ozaki_opencl.h $(GENHDR) $(MAKDIR)/Makefile
	$(CC) $(CFLAGS) -I$(MAKDIR) -c $< -o $@

$(MAKDIR)/$(OUTNAME): $(MAKDIR)/ozaki_bench.o
	$(CC) $^ $(LDFLAGS) -o $@

.PHONY: test
test: $(MAKDIR)/$(OUTNAME)
	$(MAKDIR)/$(OUTNAME) 64 64 64

.PHONY: clean
clean:
	@rm -f $(MAKDIR)/*.o
	@rm -f $(GENHDR)

.PHONY: realclean
realclean: clean
	@rm -f $(MAKDIR)/$(OUTNAME)
