# Makefile fragment building the DBM backend into a consumer's objects, e.g.,
# CP2K's src/dbm/Makefile. The files are enumerated here, hence renaming them is
# private to LIBXSTREAM. Optionally set LIBXSTREAM_DBM_GENDIR (directory of the
# generated kernel header, default: current directory) and LIBXSTREAM_DBM_GENARG
# (e.g., -d) before or after including this file, then use:
#
#   CFLAGS += $(LIBXSTREAM_DBM_IFLAGS)
#   ALL_OBJECTS += $(LIBXSTREAM_DBM_OBJS)
#   $(LIBXSTREAM_DBM_OBJS): %.o: $(LIBXSTREAM_DBM_DIR)/%.c $(LIBXSTREAM_DBM_HEADERS)
#
# LIBXSTREAM_DBM_IFLAGS puts the generated header ahead of one installed with the
# sources. LIBXSTREAM_DBM_INCDIR is the directory holding opencl/*.h.

LIBXSTREAM_DBM_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
LIBXSTREAM_DBM_ROOT := $(abspath $(LIBXSTREAM_DBM_DIR)/../..)
LIBXSTREAM_DBM_INCDIR ?= $(LIBXSTREAM_DBM_ROOT)/libxstream
LIBXSTREAM_DBM_GENDIR ?= $(CURDIR)
LIBXSTREAM_DBM_GENARG ?=
LIBXSTREAM_DBM_SCRIPT := $(LIBXSTREAM_DBM_ROOT)/scripts/tool_opencl.sh

LIBXSTREAM_DBM_SRCS := $(LIBXSTREAM_DBM_DIR)/dbm_opencl.c
LIBXSTREAM_DBM_OBJS := $(notdir $(LIBXSTREAM_DBM_SRCS:.c=.o))
LIBXSTREAM_DBM_KERNELS := $(wildcard $(LIBXSTREAM_DBM_DIR)/kernels/*.cl)
LIBXSTREAM_DBM_GENHDR := $(LIBXSTREAM_DBM_GENDIR)/dbm_kernels.h
LIBXSTREAM_DBM_HEADERS := $(LIBXSTREAM_DBM_GENHDR) $(wildcard $(LIBXSTREAM_DBM_DIR)/*.h)
LIBXSTREAM_DBM_IFLAGS := -I$(LIBXSTREAM_DBM_GENDIR) -I$(LIBXSTREAM_DBM_DIR)

# the rule below must not become the includer's default goal
LIBXSTREAM_DBM_GOAL := $(.DEFAULT_GOAL)

$(LIBXSTREAM_DBM_GENHDR): $(LIBXSTREAM_DBM_KERNELS) $(LIBXSTREAM_DBM_SCRIPT) \
                          $(wildcard $(LIBXSTREAM_DBM_INCDIR)/opencl/*.h)
	$(LIBXSTREAM_DBM_SCRIPT) $(LIBXSTREAM_DBM_GENARG) -p "$(LIBXSTREAM_DBM_DIR)/params" \
		-I $(LIBXSTREAM_DBM_INCDIR) $(LIBXSTREAM_DBM_KERNELS) $@

.DEFAULT_GOAL := $(LIBXSTREAM_DBM_GOAL)
