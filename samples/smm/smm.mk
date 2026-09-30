# Makefile fragment building LIBSMM (this sample's backend) into a consumer's
# objects, e.g., CP2K's src/dbm/Makefile for the DBM miniapp. The files are
# enumerated here, hence renaming them is private to LIBXSTREAM. Optionally set
# LIBXSTREAM_SMM_GENDIR (directory of the generated kernel header, default:
# current directory) and LIBXSTREAM_SMM_GENARG (e.g., -d) before or after
# including this file, then use:
#
#   CFLAGS += $(LIBXSTREAM_SMM_IFLAGS)
#   ALL_OBJECTS += $(LIBXSTREAM_SMM_OBJS)
#   $(LIBXSTREAM_SMM_OBJS): %.o: $(LIBXSTREAM_SMM_DIR)/%.c $(LIBXSTREAM_SMM_HEADERS)
#
# LIBXSTREAM_SMM_IFLAGS puts the generated header ahead of one installed with the
# sources. LIBXSTREAM_SMM_INCDIR is the directory holding opencl/*.h.

LIBXSTREAM_SMM_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
LIBXSTREAM_SMM_ROOT := $(abspath $(LIBXSTREAM_SMM_DIR)/../..)
LIBXSTREAM_SMM_INCDIR ?= $(LIBXSTREAM_SMM_ROOT)/libxstream
LIBXSTREAM_SMM_GENDIR ?= $(CURDIR)
LIBXSTREAM_SMM_GENARG ?=
LIBXSTREAM_SMM_SCRIPT := $(LIBXSTREAM_SMM_ROOT)/scripts/tool_opencl.sh

LIBXSTREAM_SMM_SRCS := $(addprefix $(LIBXSTREAM_SMM_DIR)/, \
  smm_acc.c smm_kernel.c smm_params.c smm_trans.c)
LIBXSTREAM_SMM_OBJS := $(notdir $(LIBXSTREAM_SMM_SRCS:.c=.o))
LIBXSTREAM_SMM_KERNELS := $(wildcard $(LIBXSTREAM_SMM_DIR)/kernels/*.cl)
# a prediction model (.bin) is embedded along with the CSV file of its name
LIBXSTREAM_SMM_PARAMS := $(wildcard $(LIBXSTREAM_SMM_DIR)/params/*.csv)
LIBXSTREAM_SMM_MODELS := $(wildcard $(LIBXSTREAM_SMM_DIR)/params/*.bin)
LIBXSTREAM_SMM_GENHDR := $(LIBXSTREAM_SMM_GENDIR)/smm_kernels.h
LIBXSTREAM_SMM_HEADERS := $(LIBXSTREAM_SMM_GENHDR) $(wildcard $(LIBXSTREAM_SMM_DIR)/*.h)
LIBXSTREAM_SMM_IFLAGS := -I$(LIBXSTREAM_SMM_GENDIR) -I$(LIBXSTREAM_SMM_DIR)

# the rule below must not become the includer's default goal
LIBXSTREAM_SMM_GOAL := $(.DEFAULT_GOAL)

$(LIBXSTREAM_SMM_GENHDR): $(LIBXSTREAM_SMM_KERNELS) $(LIBXSTREAM_SMM_PARAMS) \
                          $(LIBXSTREAM_SMM_MODELS) $(LIBXSTREAM_SMM_SCRIPT) \
                          $(wildcard $(LIBXSTREAM_SMM_INCDIR)/opencl/*.h)
	$(LIBXSTREAM_SMM_SCRIPT) $(LIBXSTREAM_SMM_GENARG) -I $(LIBXSTREAM_SMM_INCDIR) \
		$(LIBXSTREAM_SMM_KERNELS) $(LIBXSTREAM_SMM_PARAMS) $@

.DEFAULT_GOAL := $(LIBXSTREAM_SMM_GOAL)
