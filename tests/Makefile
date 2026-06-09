ROOTDIR = $(abspath $(dir $(firstword $(MAKEFILE_LIST))))
PROJECT := libxstream
DEPDIR = $(ROOTDIR)/..
SRCDIR = .
INCDIR = .
BLDDIR = obj
OUTDIR = .

CXXFLAGS = $(NULL)
CFLAGS = $(NULL)
DFLAGS = $(NULL)

OMP = 1
SYM = 1

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
  ifneq (,$(LIBXS))
    DFLAGS += -D__LIBXS
  else ifneq (,$(wildcard $(LIBXSROOT)/libxs/libxs_source.h))
    DFLAGS += -D__LIBXS -DLIBXS_SOURCE
  endif
endif

# include directories
IFLAGS += -I$(call quote,$(DEPDIR))

OUTNAME := $(shell basename "$(ROOTDIR)")
HEADERS := $(wildcard $(INCDIR)/*.h) $(wildcard $(SRCDIR)/*.h) \
           $(wildcard $(LIBXSROOT)/libxs/*.h)
CSOURCX := $(wildcard $(SRCDIR)/*.c)
OBJECTS := $(NULL)
OBJECTX := $(call objname,$(CSOURCX))
XFILES  := $(addsuffix .x,$(addprefix $(OUTDIR)/,$(basename $(notdir $(CSOURCX)))))

.PHONY: all
all: $(XFILES)

.PHONY: tests
tests: test

.PHONY: test
test: $(OUTDIR)/.make $(OUTDIR)/test.sh $(XFILES)
	@$(OUTDIR)/test.sh

define DEFINE_LINK_RULE
$(1).x: $(2) $(dir $(1))/.make Makefile $(LIBXS)
	$(LD) $(SLDFLAGS) -o $(1).x $(2) $(LIBXS) \
		$(call cleanld,$(LDFLAGS) $(CLDFLAGS))
endef

$(foreach SRC, $(CSOURCX), \
  $(eval $(call DEFINE_LINK_RULE, $(basename $(notdir $(SRC))), $(call objname,$(SRC)))))

$(BLDDIR)/%-c.o: $(SRCDIR)/%.c .state $(BLDDIR)/.make $(HEADERS) Makefile
	$(CC) $(DFLAGS) $(IFLAGS) $(CFLAGS) $(CTARGET) -c $< -o $@

.PHONY: clean
clean:
ifneq ($(call qapath,$(BLDDIR)),$(ROOTDIR))
ifneq ($(call qapath,$(BLDDIR)),$(call qapath,.))
	@-rm -rf $(BLDDIR)
endif
endif
	@-rm -f $(OBJECTX) $(XFILES)

.PHONY: realclean
realclean: clean
