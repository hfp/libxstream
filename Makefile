# export all variables to sub-make processes
.EXPORT_ALL_VARIABLES: #export

ARCH = intel64

ROOTDIR = $(abspath $(dir $(word $(words $(MAKEFILE_LIST)),$(MAKEFILE_LIST))))
INCDIR = $(ROOTDIR)/include
SRCDIR = $(ROOTDIR)/src
BLDDIR = build/$(ARCH)
LIBDIR = lib/$(ARCH)

LIBNAME = libxstream
SOURCES = $(shell ls -1 $(SRCDIR)/*.cpp | tr "\n" " ")
HEADERS = $(shell ls -1 $(INCDIR)/*.h   | tr "\n" " ") \
          $(shell ls -1 $(SRCDIR)/*.hpp | tr "\n" " ")
OBJECTS = $(patsubst %,./$(BLDDIR)%,$(notdir $(SOURCES:.cpp=-cpp.o)))

CXXFLAGS = $(NULL)
CFLAGS = $(NULL)
DFLAGS = -DLIBXSTREAM_EXPORTED
IFLAGS = -I$(INCDIR) -I$(SRCDIR)

STATIC ?= 1
DBG ?= 0

parent = $(subst ?, ,$(firstword $(subst /, ,$(subst $(NULL) ,?,$(patsubst ./%,%,$1)))))

ICPC = $(notdir $(shell which icpc 2> /dev/null))
ICC = $(notdir $(shell which icc 2> /dev/null))

ifneq (,$(ICPC))
	CXX = $(ICPC)
	ifeq (,$(ICC))
		CC = $(CXX)
	endif
	AR = xiar
endif
ifneq (,$(ICC))
	CC = $(ICC)
	ifeq (,$(ICPC))
		CXX = $(CC)
	endif
	AR = xiar
endif

ifneq ($(CXX),)
	LD = $(CXX)
endif
ifeq ($(LD),)
	LD = $(CC)
endif

ifneq (,$(filter $(CXX),icpc))
	CXXFLAGS += -fPIC -Wall -std=c++0x
	DFLAGS += -DUSE_MKL
	ifeq (0,$(DBG))
		CXXFLAGS += -fno-alias -ansi-alias -O3 -ipo
		DFLAGS += -DNDEBUG
		ifeq ($(AVX),1)
			CXXFLAGS += -xAVX
		else ifeq ($(AVX),2)
			CXXFLAGS += -xCORE-AVX2
		else ifeq ($(AVX),3)
			CXXFLAGS += -xCOMMON-AVX512
		else
			CXXFLAGS += -xHost
		endif
	else ifneq (1,$(DBG))
		CXXFLAGS += -O0 -g3 -gdwarf-2 -debug inline-debug-info
	else
		CXXFLAGS += -O0 -g
	endif
	ifeq (0,$(OFFLOAD))
		CXXFLAGS += -no-offload
	else
		#CXXFLAGS += -offload-option,mic,compiler,"-O2 -opt-assume-safe-padding"
	endif
	LDFLAGS += -fPIC
	ifneq ($(STATIC),0)
		ifneq ($(STATIC),)
			LDFLAGS += -no-intel-extensions -static-intel
		endif
	endif
else ifneq (,$(filter $(CC),icc))
	CFLAGS += -fPIC -Wall
	DFLAGS += -DUSE_MKL
	ifeq (0,$(DBG))
		CFLAGS += -fno-alias -ansi-alias -O3 -ipo
		DFLAGS += -DNDEBUG
		ifeq ($(AVX),1)
			CFLAGS += -xAVX
		else ifeq ($(AVX),2)
			CFLAGS += -xCORE-AVX2
		else ifeq ($(AVX),3)
			CFLAGS += -xCOMMON-AVX512
		else
			CFLAGS += -xHost
		endif
	else ifneq (1,$(DBG))
		CFLAGS += -O0 -g3 -gdwarf-2 -debug inline-debug-info
	else
		CFLAGS += -O0 -g
	endif
	ifeq (0,$(OFFLOAD))
		CFLAGS += -no-offload
	else
		#CFLAGS += -offload-option,mic,compiler,"-O2 -opt-assume-safe-padding"
	endif
	LDFLAGS += -fPIC
	ifneq ($(STATIC),0)
		ifneq ($(STATIC),)
			LDFLAGS += -no-intel-extensions -static-intel
		endif
	endif
else # GCC assumed
	CXXFLAGS += -Wall -std=c++0x
	ifeq (0,$(DBG))
		CXXFLAGS += -O3
		DFLAGS += -DNDEBUG
		ifeq ($(AVX),1)
			CXXFLAGS += -mavx
		else ifeq ($(AVX),2)
			CXXFLAGS += -mavx2
		else ifeq ($(AVX),3)
			CXXFLAGS += -mavx512f
		else
			CXXFLAGS += -march=native
		endif
	else ifneq (1,$(DBG))
		CXXFLAGS += -O0 -g3 -gdwarf-2
	else
		CXXFLAGS += -O0 -g
	endif
	ifneq ($(OS),Windows_NT)
		LDFLAGS += -fPIC
		CXXFLAGS += -fPIC
	endif
	ifneq ($(STATIC),0)
		ifneq ($(STATIC),)
			LDFLAGS += -static
		endif
	endif
endif

ifeq (,$(CXXFLAGS))
	CXXFLAGS = $(CFLAGS)
endif
ifeq (,$(CFLAGS))
	CFLAGS = $(CXXFLAGS)
endif

ifneq ($(STATIC),0)
	LIBEXT := a
else
	LIBEXT := so
endif

.PHONY: all
all: ./$(LIBDIR)/$(LIBNAME).$(LIBEXT)

./$(LIBDIR)/$(LIBNAME).$(LIBEXT): $(OBJECTS)
	@mkdir -p ./$(LIBDIR)
ifeq ($(STATIC),0)
	$(LD) -shared -o $@ $(LDFLAGS) $^
else
	$(AR) -rs $@ $^
endif

./$(BLDDIR)%-c.o: $(SRCDIR)/%.c $(HEADERS) $(ROOTDIR)/Makefile
	@mkdir -p ./$(BLDDIR)
	$(CC) $(CFLAGS) $(DFLAGS) $(IFLAGS) -c $< -o $@

./$(BLDDIR)%-cpp.o: $(SRCDIR)/%.cpp $(HEADERS) $(ROOTDIR)/Makefile
	@mkdir -p ./$(BLDDIR)
	$(CXX) $(CXXFLAGS) $(DFLAGS) $(IFLAGS) -c $< -o $@

DUMMY := $(shell echo "DEBUG: |$(call parent,$(BLDDIR))|" >&2)

.PHONY: clean
clean:
	@rm -rf ./$(call parent,$(BLDDIR))

.PHONY: realclean
realclean: clean
	@rm -rf ./$(call parent,$(LIBDIR))

install: all clean
	@cp -r $(INCDIR) .

