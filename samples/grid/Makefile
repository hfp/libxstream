ROOTDIR = $(abspath $(dir $(firstword $(MAKEFILE_LIST))))
DEPDIR = $(ROOTDIR)/../..
BLDDIR = obj
OUTDIR = .

# CP2K source tree, which provides the grid library and its miniapp
CP2K_ROOT ?= $(wildcard $(abspath $(DEPDIR)/../cp2k))
# BLAS for the DGEMM backend
BLAS_LIBS ?= -lblas
# CUDA backend if nvcc is found (CUDA=0 disables it)
NVCC ?= $(shell which nvcc 2>/dev/null)
CUDA ?= $(if $(NVCC),1,0)
CUDA_ARCH ?= sm_70
# OpenCL backend (grid_opencl.c) as the GPU backend unless CUDA is used (OPENCL=0 disables it)
LIBXSROOT ?= $(wildcard $(abspath $(DEPDIR)/../libxs))
OPENCL ?= $(if $(and $(LIBXSROOT),$(filter 0,$(CUDA))),1,0)
# arguments of the test target: cycles of each replay
TESTARG ?= 3

CC ?= gcc
CFLAGS ?= -O3 -march=native
OMPFLAG ?= -fopenmp
GRIDFLAGS := -std=c11 $(OMPFLAG) -Wall -Wextra -Wno-vla-parameter
LIBS := $(BLAS_LIBS) -lm

GRIDDIR := $(CP2K_ROOT)/src/grid
SRCDIR := $(CP2K_ROOT)/src
SOURCES := $(filter-out %/grid_miniapp.c %/grid_unittest.c, \
  $(wildcard $(GRIDDIR)/*.c $(GRIDDIR)/common/*.c $(GRIDDIR)/ref/*.c \
             $(GRIDDIR)/cpu/*.c $(GRIDDIR)/dgemm/*.c)) \
  $(SRCDIR)/offload/offload_buffer.c $(SRCDIR)/offload/offload_library.c \
  $(SRCDIR)/offload/offload_mempool.c $(SRCDIR)/mpiwrap/cp_mpi.c
ifneq (0,$(CUDA))
  CUSOURCES := $(wildcard $(GRIDDIR)/gpu/*.cu)
  GRIDFLAGS += -D__OFFLOAD_CUDA $(if $(CUDA_PATH),-I$(CUDA_PATH)/include)
  LIBS += $(if $(CUDA_PATH),-L$(CUDA_PATH)/lib64) -lcudart -lcuda -lcublas
endif
ifneq (0,$(OPENCL))
  # header-only LIBXS and LIBXSTREAM, which need POSIX beyond C11
  GRIDFLAGS := $(patsubst -std=c11,-std=gnu11,$(GRIDFLAGS)) -D__LIBXSTREAM -DLIBXSTREAM_SOURCE \
    -D__OFFLOAD_OPENCL_GRID -I$(DEPDIR) -I$(LIBXSROOT) -I$(SRCDIR) -I$(abspath $(BLDDIR))
  LIBS += -lOpenCL -ldl -lpthread
  CLOBJECTS := $(BLDDIR)/grid_opencl.o
  CLKERNELS := $(wildcard $(ROOTDIR)/kernels/*.cl)
  CLHEADER := $(BLDDIR)/grid_kernels.h
endif
OBJECTS := $(patsubst $(SRCDIR)/%.c,$(BLDDIR)/%.o,$(SOURCES)) \
  $(patsubst $(SRCDIR)/%.cu,$(BLDDIR)/%-cu.o,$(CUSOURCES)) $(CLOBJECTS)
HEADERS := $(wildcard $(GRIDDIR)/*.h $(GRIDDIR)/*/*.h $(SRCDIR)/offload/*.h \
  $(SRCDIR)/mpiwrap/*.h)
MINIAPP := $(OUTDIR)/grid_miniapp.x
UNITTEST := $(OUTDIR)/grid_unittest.x

.PHONY: all
ifneq (,$(CP2K_ROOT))
all: miniapp
else
all:
	@echo "INFO: CP2K not found (CP2K_ROOT), the grid miniapp is not built"
endif

# CP2K's grid miniapp and unit test, built here rather than in CP2K's tree
.PHONY: miniapp
miniapp: $(MINIAPP) $(UNITTEST)

$(OUTDIR)/grid_%.x: $(BLDDIR)/grid/grid_%.o $(OBJECTS)
	$(if $(CP2K_ROOT),,$(error CP2K_ROOT is not set))
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(OMPFLAG) -o $@ $^ $(LIBS)

$(BLDDIR)/%.o: $(SRCDIR)/%.c $(HEADERS) $(ROOTDIR)/Makefile
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(GRIDFLAGS) -c $< -o $@

$(BLDDIR)/%-cu.o: $(SRCDIR)/%.cu $(HEADERS) $(ROOTDIR)/Makefile
	@mkdir -p $(dir $@)
	$(NVCC) -O3 -arch $(CUDA_ARCH) -Xcompiler "$(CFLAGS) $(OMPFLAG)" \
		-D__OFFLOAD_CUDA -c $< -o $@

# kernels with CP2K's headers inlined, whose include guards are kept since headers repeat
$(CLHEADER): $(CLKERNELS) $(ROOTDIR)/grid_opencl.h $(wildcard $(ROOTDIR)/kernels/*.h $(DEPDIR)/libxstream/opencl/*.h) \
            $(DEPDIR)/scripts/tool_opencl.sh $(HEADERS)
	@mkdir -p $(dir $@)
	$(DEPDIR)/scripts/tool_opencl.sh -k -I $(DEPDIR)/libxstream -I $(GRIDDIR)/common $(CLKERNELS) $@

$(BLDDIR)/grid_opencl.o: $(ROOTDIR)/grid_opencl.c $(ROOTDIR)/grid_opencl.h $(CLHEADER) $(HEADERS) $(ROOTDIR)/Makefile
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(GRIDFLAGS) -c $< -o $@

# CP2K's unit test, then each sample task recorded as task list (collocate and
# integrate) by the reference backend and replayed by every other backend
TESTBACKENDS ?= ref cpu $(if $(filter-out 0,$(CUDA) $(OPENCL)),gpu)
TESTDIR := $(BLDDIR)/tasklists
.PHONY: test
test: miniapp
	@$(abspath $(UNITTEST)) $(CP2K_ROOT)/
	@rm -rf $(TESTDIR) && mkdir -p $(TESTDIR)
	@cd $(TESTDIR) && for TASK in $(GRIDDIR)/sample_tasks/*.task; do \
		NAME=$$(basename $${TASK} .task); \
		for KIND in "" --integrate; do \
			$(abspath $(MINIAPP)) --backend ref --dump 1 $${KIND} --batch 50 50 \
				$${TASK} >/dev/null || exit 1; \
		done; \
		for FILE in grid_*_0_0.tasklist; do mv $${FILE} $${NAME}_$${FILE}; done; \
	done
	@cd $(TESTDIR) && for FILE in *.tasklist; do \
		for BACKEND in $(TESTBACKENDS); do \
			OUTPUT=$$($(abspath $(MINIAPP)) --backend $${BACKEND} $(TESTARG) \
				$${FILE} 2>&1) || { echo "$${OUTPUT}"; exit 1; }; \
			echo "$${OUTPUT}" | grep "^Task list"; \
		done; \
	done

.PHONY: clean
clean:
	@-rm -rf $(BLDDIR)

.PHONY: realclean
realclean: clean
	@-rm -f $(MINIAPP) $(UNITTEST)
