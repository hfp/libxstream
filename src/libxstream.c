/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#if defined(__OPENCL)
# include <libxstream/libxstream_opencl.h>
# include <libxs/libxs_hash.h>
# include <libxs/libxs_str.h>
# if defined(_WIN32)
#   include <windows.h>
#   include <process.h>
# else
#   include <unistd.h>
#   include <errno.h>
#   include <glob.h>
# endif
# include <fcntl.h>
# include <sys/stat.h>
# if !defined(S_ISDIR) && defined(S_IFMT) && defined(S_IFDIR)
#   define S_ISDIR(A) ((S_IFMT & (A)) == S_IFDIR)
# endif
# if !defined(S_IREAD)
#   define S_IREAD S_IRUSR
# endif
# if !defined(S_IWRITE)
#   define S_IWRITE S_IWUSR
# endif

# if !defined(LIBXSTREAM_NLOCKS)
#   define LIBXSTREAM_NLOCKS 4
# endif
# if !defined(LIBXSTREAM_TEMPDIR) && 1
#   define LIBXSTREAM_TEMPDIR "/tmp"
# endif
# if !defined(LIBXSTREAM_CACHE_DID) && 1
#   define LIBXSTREAM_CACHE_DID
# endif
# if !defined(LIBXSTREAM_CACHE_DIR) && 0
#   define LIBXSTREAM_CACHE_DIR ".cl_cache"
# endif
# if !defined(LIBXSTREAM_CPPBIN) && 1
#   define LIBXSTREAM_CPPBIN "/usr/bin/cpp"
# endif
# if !defined(LIBXSTREAM_SEDBIN) && 1
#   define LIBXSTREAM_SEDBIN "/usr/bin/sed"
# endif
/* disabled: let MPI runtime come up before */
# if !defined(LIBXSTREAM_PREINIT) && 0
#   define LIBXSTREAM_PREINIT
# endif
/* attempt to enable command aggregation */
# if !defined(LIBXSTREAM_CMDAGR) && 1
#   define LIBXSTREAM_CMDAGR
# endif
# if !defined(LIBXSTREAM_NCCS) && 1
#   define LIBXSTREAM_NCCS 0
# endif


LIBXSTREAM_APIVAR_DEFINE(char internal_libxstream_opencl_locks[LIBXS_CACHELINE * LIBXSTREAM_NLOCKS]);
/* global configuration discovered during initialization */
LIBXSTREAM_APIVAR_PUBLIC_DEF(libxstream_opencl_config_t libxstream_opencl_config);
/**
 * Explicit configuration requested by libxstream_init_config. Zero-initialized
 * like any APIVAR, which is why the sentinel cannot be the -1 the API documents:
 * zero is a meaningful request (explicitly disable). The companion flag states
 * whether the struct carries a request at all.
 */
LIBXSTREAM_APIVAR_DEFINE(libxstream_init_config_t internal_libxstream_init_cfg);
LIBXSTREAM_APIVAR_DEFINE(int internal_libxstream_init_cfg_valid);

# if defined(LIBXSTREAM_CACHE_DID)
LIBXSTREAM_APIVAR_DEFINE(int internal_libxstream_opencl_active_id);
# endif


LIBXSTREAM_API_INTERN void libxstream_opencl_notify(
  const char /*errinfo*/[], const void* /*private_info*/, size_t /*cb*/, void* /*user_data*/);
LIBXSTREAM_API_INTERN void libxstream_opencl_notify(const char errinfo[], const void* private_info, size_t cb, void* user_data)
{
  LIBXS_UNUSED(private_info);
  LIBXS_UNUSED(cb);
  LIBXS_UNUSED(user_data);
  fprintf(stderr, "ERROR ACC/OpenCL: %s\n", errinfo);
}


/**
 * Comparator used with qsort; stabilized by tail condition (a < b ? -1 : 1).
 * Brings GPUs with local memory in front, followed by (potentially) integrated GPUs,
 * and further orders by memory capacity.
 */
LIBXSTREAM_API_INTERN int libxstream_opencl_order_devices(const void* /*dev_a*/, const void* /*dev_b*/);
LIBXSTREAM_API_INTERN int libxstream_opencl_order_devices(const void* dev_a, const void* dev_b)
{
  const cl_device_id* const a = (const cl_device_id*)dev_a;
  const cl_device_id* const b = (const cl_device_id*)dev_b;
  cl_device_type type_a = 0, type_b = 0;
  assert(NULL != a && NULL != b && a != b);
  LIBXS_EXPECT_DEBUG(EXIT_SUCCESS == clGetDeviceInfo(*a, CL_DEVICE_TYPE, sizeof(cl_device_type), &type_a, NULL));
  LIBXS_EXPECT_DEBUG(EXIT_SUCCESS == clGetDeviceInfo(*b, CL_DEVICE_TYPE, sizeof(cl_device_type), &type_b, NULL));
  if (CL_DEVICE_TYPE_DEFAULT & type_a) {
    return -1;
  }
  else if (CL_DEVICE_TYPE_DEFAULT & type_b) {
    return 1;
  }
  else {
    if (CL_DEVICE_TYPE_GPU & type_a) {
      if (CL_DEVICE_TYPE_GPU & type_b) {
        int unified_a, unified_b;
        size_t size_a, size_b;
        LIBXS_EXPECT_DEBUG(EXIT_SUCCESS == libxstream_opencl_info_devmem(*a, NULL, &size_a, NULL, &unified_a));
        LIBXS_EXPECT_DEBUG(EXIT_SUCCESS == libxstream_opencl_info_devmem(*b, NULL, &size_b, NULL, &unified_b));
        if ((0 == unified_a && 0 == unified_b) || (0 != unified_a && 0 != unified_b)) {
          if (size_a != size_b) return (size_a < size_b ? 1 : -1);
          if (0 != (64 & libxstream_opencl_config.xhints)) {
            cl_uint bus_a = 0, bus_b = 0;
            struct { cl_uint domain, bus, device, function; } pci_a, pci_b;
            if (EXIT_SUCCESS == clGetDeviceInfo(*a, 0x420F /*CL_DEVICE_PCI_BUS_INFO_INTEL*/, sizeof(pci_a), &pci_a, NULL) &&
                EXIT_SUCCESS == clGetDeviceInfo(*b, 0x420F /*CL_DEVICE_PCI_BUS_INFO_INTEL*/, sizeof(pci_b), &pci_b, NULL))
            {
              bus_a = pci_a.bus;
              bus_b = pci_b.bus;
            }
            if (bus_a != bus_b) return (bus_a < bus_b ? -1 : 1);
          }
          return (a < b ? -1 : 1);
        }
        /* discrete GPU goes in front */
        else if (0 == unified_b) return 1;
        else return -1;
      }
      else return -1;
    }
    else if (CL_DEVICE_TYPE_GPU & type_b) {
      return 1;
    }
    else {
      if (CL_DEVICE_TYPE_CPU & type_a) {
        if (CL_DEVICE_TYPE_CPU & type_b) {
          size_t size_a, size_b;
          LIBXS_EXPECT_DEBUG(EXIT_SUCCESS == libxstream_opencl_info_devmem(*a, NULL, &size_a, NULL, NULL));
          LIBXS_EXPECT_DEBUG(EXIT_SUCCESS == libxstream_opencl_info_devmem(*b, NULL, &size_b, NULL, NULL));
          return (size_a < size_b ? 1 : (size_a != size_b ? -1 : (a < b ? -1 : 1)));
        }
        else return -1;
      }
      else if (CL_DEVICE_TYPE_CPU & type_b) {
        return 1;
      }
      else {
        size_t size_a = 0, size_b = 0;
        LIBXS_EXPECT_DEBUG(EXIT_SUCCESS == libxstream_opencl_info_devmem(*a, NULL, &size_a, NULL, NULL));
        LIBXS_EXPECT_DEBUG(EXIT_SUCCESS == libxstream_opencl_info_devmem(*b, NULL, &size_b, NULL, NULL));
        return (size_a < size_b ? 1 : (size_a != size_b ? -1 : (a < b ? -1 : 1)));
      }
    }
  }
}


/**
 * Resolve the settings a caller can state explicitly: the request wins over the
 * environment, which wins over the default. Separate from libxstream_opencl_setup
 * and idempotent because that setup is one-shot and a constructor
 * (libxstream_opencl_init) may run it at load time, i.e. before any caller could
 * state a request -- re-resolving here is what lets a later libxstream_init_config
 * still take effect. Settings read lazily at their point of use (usm, device) need
 * no equivalent.
 */
LIBXSTREAM_API_INTERN void libxstream_opencl_configure(void);
LIBXSTREAM_API_INTERN void libxstream_opencl_configure(void)
{
  const char* const env_verbose = getenv("LIBXSTREAM_VERBOSE");
  const char* const env_subbuffer = getenv("LIBXSTREAM_SUBBUFFER");
  const int verbosity = (0 != internal_libxstream_init_cfg_valid ? internal_libxstream_init_cfg.verbosity : -1);
  const int subbuffer = (0 != internal_libxstream_init_cfg_valid ? internal_libxstream_init_cfg.subbuffer : -1);
  libxstream_opencl_config.verbosity = (0 <= verbosity ? verbosity
    : (NULL == env_verbose ? /*default*/ 0 : atoi(env_verbose)));
  /* opt-in: sub-buffers for offset kernel-arguments (see the config field) */
  libxstream_opencl_config.subbuffer = (0 <= subbuffer ? subbuffer
    : (NULL == env_subbuffer ? /*default*/ 0 : atoi(env_subbuffer)));
}


/** Setup to run prior to touching OpenCL runtime. */
LIBXSTREAM_API_INTERN void libxstream_opencl_setup(void);
LIBXSTREAM_API_INTERN void libxstream_opencl_setup(void)
{
  const char *const env_devsplit = getenv("LIBXSTREAM_DEVSPLIT"), *const env_nlocks = getenv("LIBXSTREAM_NLOCKS");
  const char* const env_dump_acc = getenv("LIBXSTREAM_DUMP");
  const char *const env_debug = getenv("LIBXSTREAM_DEBUG"), *const env_profile = getenv("LIBXSTREAM_PROFILE");
  const char* const env_profile_mem = getenv("LIBXSTREAM_PROFILE_MEM");
  const char* const env_dump = (NULL != env_dump_acc ? env_dump_acc : getenv("IGC_ShaderDumpEnable"));
  const char *const env_neo = getenv("NEOReadDebugKeys"), *const env_wa = getenv("LIBXSTREAM_WA");
  static char neo_enable_debug_keys[] = "NEOReadDebugKeys=1";
# if defined(LIBXS_INTERCEPT_DYNAMIC)
  const char* const env_cuda_pin = getenv("LIBXSTREAM_CUDA_PIN");
# endif
# if defined(LIBXSTREAM_STREAM_PRIORITIES)
  const char* const env_priority = getenv("LIBXSTREAM_PRIORITY");
# endif
# if defined(LIBXSTREAM_NCCS)
  const char* const env_nccs = getenv("LIBXSTREAM_NCCS");
  const int nccs = (NULL == env_nccs ? LIBXSTREAM_NCCS : atoi(env_nccs));
# endif
# if defined(LIBXSTREAM_XHINTS)
  const char* const env_xhints = (LIBXSTREAM_XHINTS);
  const int xhints_default = 1 + 2 + 4 + 8 + 64;
# else
  const char* const env_xhints = NULL;
  const int xhints_default = 0;
# endif
# if defined(LIBXSTREAM_ASYNC)
  const char* const env_async = (LIBXSTREAM_ASYNC);
  const int async_default = 1 + 2 + 4 + 8;
# else
  const char* const env_async = NULL;
  const int async_default = 0;
# endif
  const int wa_default = 2 + 4 + 8 + 16;
  const int nlocks = (NULL == env_nlocks ? 1 /*default*/ : atoi(env_nlocks));
  const int neo = (NULL == env_neo ? 1 : atoi(env_neo));
  int i;
# if defined(_OPENMP)
  const int max_threads = omp_get_max_threads(), num_threads = omp_get_num_threads();
  memset(&libxstream_opencl_config, 0, sizeof(libxstream_opencl_config));
  libxstream_opencl_config.nthreads = (num_threads < max_threads ? max_threads : num_threads);
# else
  memset(&libxstream_opencl_config, 0, sizeof(libxstream_opencl_config));
  libxstream_opencl_config.nthreads = 1;
# endif
  assert(NULL == libxstream_opencl_config.lock_main); /* test condition to avoid initializing multiple times */
  libxs_init(); /* before using LIBXSMM's functionality */
  assert(sizeof(libxs_lock_t) <= LIBXS_CACHELINE);
  for (i = 0; i < LIBXSTREAM_NLOCKS; ++i) {
    LIBXS_LOCK_ATTR_TYPE(LIBXS_LOCK) acc_opencl_attr_;
    LIBXS_LOCK_ATTR_INIT(LIBXS_LOCK, &acc_opencl_attr_);
    LIBXS_LOCK_INIT(LIBXS_LOCK, (libxs_lock_t*)(internal_libxstream_opencl_locks + LIBXS_CACHELINE * i), &acc_opencl_attr_);
    LIBXS_LOCK_ATTR_DESTROY(LIBXS_LOCK, &acc_opencl_attr_);
  }
  libxstream_opencl_config.lock_main = (libxs_lock_t*)internal_libxstream_opencl_locks;
  libxstream_opencl_config.lock_memory = /* 2nd lock-domain */
    (1 < LIBXS_MIN(nlocks, LIBXSTREAM_NLOCKS) ? ((libxs_lock_t*)(internal_libxstream_opencl_locks + LIBXS_CACHELINE * 1))
                                              : libxstream_opencl_config.lock_main);
  libxstream_opencl_config.lock_stream = /* 3rd lock-domain */
    (2 < LIBXS_MIN(nlocks, LIBXSTREAM_NLOCKS) ? ((libxs_lock_t*)(internal_libxstream_opencl_locks + LIBXS_CACHELINE * 2))
                                              : libxstream_opencl_config.lock_main);
  libxstream_opencl_config.lock_event = /* 4th lock-domain */
    (3 < LIBXS_MIN(nlocks, LIBXSTREAM_NLOCKS) ? ((libxs_lock_t*)(internal_libxstream_opencl_locks + LIBXS_CACHELINE * 3))
                                              : libxstream_opencl_config.lock_main);
  libxstream_opencl_configure(); /* verbosity is used below */
  libxstream_opencl_config.devsplit = (NULL == env_devsplit ? (/*1 < libxs_nranks() ? -1 :*/ 0) : atoi(env_devsplit));
# if defined(LIBXSTREAM_STREAM_PRIORITIES)
  libxstream_opencl_config.priority = (NULL == env_priority ? /*default*/ 3 : atoi(env_priority));
# endif
  libxstream_opencl_config.profile = (NULL == env_profile ? /*default*/ 0 : atoi(env_profile));
  libxstream_opencl_config.profile_mem = (NULL == env_profile_mem ? /*default*/ 0 : atoi(env_profile_mem));
  libxstream_opencl_config.xhints = (NULL == env_xhints ? xhints_default : atoi(env_xhints));
  libxstream_opencl_config.async = (NULL == env_async ? async_default : atoi(env_async));
  libxstream_opencl_config.dump = (NULL == env_dump ? /*default*/ 0 : atoi(env_dump));
  libxstream_opencl_config.debug = (NULL == env_debug ? libxstream_opencl_config.dump : atoi(env_debug));
  { /**
     * NEOReadDebugKeys gates only the bits that populate a NEO debug key (1-8).
     * The higher bits select library-side behavior with no such dependency, and
     * bit 16 disables USM to work around unified-memory stacks: losing a
     * correctness workaround by unsetting an unrelated variable is a trap.
     */
    const int wa = (NULL == env_wa ? ((1 != libxstream_opencl_config.devsplit ? 0 : 1) + wa_default) : atoi(env_wa));
    libxstream_opencl_config.wa = (0 != neo ? wa : (wa & ~0xF));
  }
# if defined(LIBXSTREAM_CACHE_DIR)
  { /* environment is populated before touching the compute runtime */
    const char *const env_cache = getenv("LIBXSTREAM_CACHE"), *env_cachedir = getenv("NEO_CACHE_DIR");
    int cache = (NULL == env_cache ? 0 : atoi(env_cache));
    struct stat cachedir;
    if (0 == cache) {
      if (stat(LIBXSTREAM_CACHE_DIR, &cachedir) == 0 && S_ISDIR(cachedir.st_mode)) cache = 1;
      else if (stat(LIBXSTREAM_TEMPDIR "/" LIBXSTREAM_CACHE_DIR, &cachedir) == 0 && S_ISDIR(cachedir.st_mode)) cache = 2;
    }
    if (1 == cache) {
      static char neo_cachedir[] = "NEO_CACHE_DIR=" LIBXSTREAM_CACHE_DIR;
      static char ocl_cachedir[] = "cl_cache_dir=" LIBXSTREAM_CACHE_DIR;
      LIBXS_EXPECT(0 == LIBXS_PUTENV(neo_cachedir)); /* putenv before entering OpenCL */
      LIBXS_EXPECT(0 == LIBXS_PUTENV(ocl_cachedir)); /* putenv before entering OpenCL */
      env_cachedir = LIBXSTREAM_CACHE_DIR;
    }
#   if defined(LIBXSTREAM_TEMPDIR)
    else if (NULL == env_cachedir) { /* code-path entered by default */
      if (NULL == env_cache || 0 != cache) { /* customize NEO_CACHE_DIR unless LIBXSTREAM_CACHE=0 */
        static char neo_cachedir[] = "NEO_CACHE_DIR=" LIBXSTREAM_TEMPDIR "/" LIBXSTREAM_CACHE_DIR;
        LIBXS_EXPECT(0 == LIBXS_PUTENV(neo_cachedir)); /* putenv before entering OpenCL */
        env_cachedir = LIBXSTREAM_TEMPDIR "/" LIBXSTREAM_CACHE_DIR;
      }
      if (0 != cache) { /* legacy-NEO is treated with explicit opt-in */
        static char ocl_cachedir[] = "cl_cache_dir=" LIBXSTREAM_TEMPDIR "/" LIBXSTREAM_CACHE_DIR;
        LIBXS_EXPECT(0 == LIBXS_PUTENV(ocl_cachedir)); /* putenv before entering OpenCL */
      }
    }
#   endif
    if (NULL != env_cachedir) {
#   if defined(_WIN32)
      LIBXS_UNUSED(env_cachedir);
#   else
#     if defined(S_IRWXU) && defined(S_IRGRP) && defined(S_IXGRP) && defined(S_IROTH) && defined(S_IXOTH)
      const int mode = S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH;
#     else
      const int mode = 0xFFFFFFFF;
#     endif
      LIBXS_EXPECT(0 == mkdir(env_cachedir, mode) || EEXIST == errno); /* soft-error */
#   endif
    }
  }
# endif
# if defined(LIBXSTREAM_NCCS)
  if (0 != nccs && NULL == getenv("ZEX_NUMBER_OF_CCS")) {
    static char zex_nccs[LIBXSTREAM_MAXNDEVS * 8 + 32] = "ZEX_NUMBER_OF_CCS=";
    const int mode = ((1 == nccs || 2 == nccs) ? nccs : 4);
    int j = strlen(zex_nccs);
    for (i = 0; i < LIBXSTREAM_MAXNDEVS; ++i) {
      const int n = (0 < i ? LIBXS_SNPRINTF(zex_nccs + j, sizeof(zex_nccs) - j, ",%u:%i", i, mode)
                           : LIBXS_SNPRINTF(zex_nccs + j, sizeof(zex_nccs) - j, "%u:%i", i, mode));
      if (0 < n) j += n;
      else {
        j = 0;
        break;
      }
    }
    if (0 < j && 0 == LIBXS_PUTENV(zex_nccs) && /* populate before touching the compute runtime */
        (2 <= libxstream_opencl_config.verbosity || 0 > libxstream_opencl_config.verbosity))
    {
      fprintf(stderr, "INFO ACC/OpenCL: directly map to Compute Command Streamers (%i-CCS mode)\n", mode);
    }
  }
# endif
  if (0 != neo && (NULL != env_neo || 0 == LIBXS_PUTENV(neo_enable_debug_keys))) {
    static char a[] = "ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE", b[] = "EnableRecoverablePageFaults=0";
    static char c[] = "DirectSubmissionOverrideBlitterSupport=0", d[] = "UCX_RCACHE_ENABLE=0";
    static char e[] = "DisableScratchPages=1", *const apply[] = {a, b, c, d, e};
    if ((1 & libxstream_opencl_config.wa) && NULL == getenv("ZE_FLAT_DEVICE_HIERARCHY")) {
      LIBXS_EXPECT(0 == LIBXS_PUTENV(apply[0]));
    }
# if (1 >= LIBXSTREAM_USM)
    if ((2 & libxstream_opencl_config.wa) && NULL == getenv("EnableRecoverablePageFaults")) {
      LIBXS_EXPECT(0 == LIBXS_PUTENV(apply[1]));
    }
# endif
    if ((4 & libxstream_opencl_config.wa) && NULL == getenv("DirectSubmissionOverrideBlitterSupport")) {
      LIBXS_EXPECT(0 == LIBXS_PUTENV(apply[2]));
    }
    if ((8 & libxstream_opencl_config.wa) && NULL == getenv("UCX_RCACHE_ENABLE")) {
      LIBXS_EXPECT(0 == LIBXS_PUTENV(apply[3]));
    }
    if (0 != libxstream_opencl_config.debug && NULL == getenv("DisableScratchPages")) {
      LIBXS_EXPECT(0 == LIBXS_PUTENV(apply[4]));
    }
  }
# if defined(LIBXS_INTERCEPT_DYNAMIC)
  /**
   * Host memory is registered with CUDA if the application linked its runtime.
   * LIBXSTREAM_CUDA_PIN=0 leaves it unregistered, which is what measures the
   * pageable transport (nothing else about the library changes).
   */
  if (NULL == env_cuda_pin || 0 != atoi(env_cuda_pin)) {
    union { const void* dlsym; int (*ptr)(void*, size_t, unsigned int); } reg;
    union { const void* dlsym; int (*ptr)(void*); } unreg;
    dlerror(); /* clear an eventual error status */
    reg.dlsym = dlsym(LIBXS_RTLD_DEFAULT, "cudaHostRegister");
    unreg.dlsym = dlsym(LIBXS_RTLD_DEFAULT, "cudaHostUnregister");
    /* both or neither: a registration that cannot be undone outlives the mapping */
    if (NULL != reg.dlsym && NULL != unreg.dlsym) {
      libxstream_opencl_config.cudaHostRegister = reg.ptr;
      libxstream_opencl_config.cudaHostUnregister = unreg.ptr;
    }
  }
# endif
}


LIBXSTREAM_API void libxstream_init_config_default(libxstream_init_config_t* cfg)
{
  if (NULL != cfg) {
    cfg->usm = -1;
    cfg->device = -1;
    cfg->verbosity = -1;
    cfg->subbuffer = -1;
  }
}


LIBXSTREAM_API int libxstream_init_config(const libxstream_init_config_t* cfg)
{
  /* NULL resets to sentinels rather than keeping a prior call's request */
  if (NULL != cfg) internal_libxstream_init_cfg = *cfg;
  else libxstream_init_config_default(&internal_libxstream_init_cfg);
  internal_libxstream_init_cfg_valid = 1;
  return libxstream_init();
}


LIBXSTREAM_API int libxstream_init(void)
{
# if defined(_OPENMP) && 0 /* TODO */
  /* initialization/finalization is not meant to be thread-safe */
  int result = ((0 == omp_in_parallel() || /*main*/ 0 == omp_get_thread_num()) ? EXIT_SUCCESS : EXIT_FAILURE);
# else
  int result = EXIT_SUCCESS;
# endif
  if (NULL == libxstream_opencl_config.lock_main) { /* avoid to configure multiple times */
    libxstream_opencl_setup();
  }
  else { /* setup already ran (constructor): pick up an explicit configuration */
    libxstream_opencl_configure();
  }
  /* eventually touch OpenCL/compute runtime after configure */
  if (0 == libxstream_opencl_config.ndevices && EXIT_SUCCESS == result) { /* avoid to initialize multiple times */
    const unsigned int mxcsr_saved = LIBXS_MXCSR_GET();
    char buffer[LIBXSTREAM_BUFFERSIZE];
    cl_platform_id platforms[LIBXSTREAM_MAXNDEVS] = {NULL};
    cl_device_id devices[LIBXSTREAM_MAXNDEVS];
    cl_device_type type = CL_DEVICE_TYPE_ALL;
    cl_uint nplatforms = 0, ndevices = 0, i;
    const char* const env_devmatch = getenv("LIBXSTREAM_DEVMATCH");
    const char* const env_devtype = getenv("LIBXSTREAM_DEVTYPE");
    const char* const env_device = getenv("LIBXSTREAM_DEVICE");
    char* const env_devids = getenv("LIBXSTREAM_DEVIDS");
    const int cfg_device = (0 != internal_libxstream_init_cfg_valid ? internal_libxstream_init_cfg.device : -1);
    int device_id = (0 <= cfg_device) ? cfg_device : (NULL == env_device ? 0 : atoi(env_device));
# if defined(LIBXSTREAM_CACHE_DID)
    assert(0 == internal_libxstream_opencl_active_id);
# endif
    if (EXIT_SUCCESS != libxstream_opencl_device_uid(NULL /*device*/, env_devmatch, &libxstream_opencl_config.devmatch)) {
      libxstream_opencl_config.devmatch = 1;
    }
    if (EXIT_SUCCESS == clGetPlatformIDs(0, NULL, &nplatforms) && 0 < nplatforms) {
      CL_CHECK(result, clGetPlatformIDs(nplatforms <= LIBXSTREAM_MAXNDEVS ? nplatforms : LIBXSTREAM_MAXNDEVS, platforms, 0));
    }
    if (EXIT_SUCCESS == result) {
      if (NULL != env_devtype && '\0' != *env_devtype) {
        if (NULL != libxs_stristr(env_devtype, "gpu")) {
          type = CL_DEVICE_TYPE_GPU;
        }
        else if (NULL != libxs_stristr(env_devtype, "cpu")) {
          type = CL_DEVICE_TYPE_CPU;
        }
        else if (NULL != libxs_stristr(env_devtype, "acc") || NULL != libxs_stristr(env_devtype, "other")) {
          type = CL_DEVICE_TYPE_ACCELERATOR;
        }
        else {
          type = CL_DEVICE_TYPE_ALL;
        }
      }
      libxstream_opencl_config.ndevices = 0;
      for (i = 0; i < nplatforms; ++i) {
        if (EXIT_SUCCESS == clGetDeviceIDs(platforms[i], type, 0, NULL, &ndevices) && 0 < ndevices) {
          CL_CHECK(result, clGetDeviceIDs(platforms[i], type, ndevices, devices, NULL));
          if (EXIT_SUCCESS == result) {
            cl_uint j = 0;
            for (; j < ndevices; ++j) {
# if defined(CL_VERSION_1_2)
              cl_device_partition_property properties[] = {
                CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN, CL_DEVICE_AFFINITY_DOMAIN_NUMA, /*terminator*/ 0};
              cl_uint nunits = 0, n = 0;
              if ((1 < libxstream_opencl_config.devsplit || 0 > libxstream_opencl_config.devsplit) &&
                  /* Intel CPU (e.g., out of two sockets) yields thread-count of both sockets */
                  EXIT_SUCCESS == clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &nunits, NULL) &&
                  1 < nunits)
              {
                n = LIBXS_MIN(
                  1 < libxstream_opencl_config.devsplit ? (cl_uint)libxstream_opencl_config.devsplit : nunits, LIBXSTREAM_MAXNDEVS);
                properties[0] = CL_DEVICE_PARTITION_EQUALLY;
                properties[1] = LIBXS_UPDIV(nunits, n);
              }
              if (0 == libxstream_opencl_config.devsplit || 1 == libxstream_opencl_config.devsplit ||
                  (libxstream_opencl_config.ndevices + 1) == LIBXSTREAM_MAXNDEVS ||
                  EXIT_SUCCESS != clCreateSubDevices(devices[j], properties, 0, NULL, &n))
# endif
              {
                libxstream_opencl_config.devices[libxstream_opencl_config.ndevices] = devices[j];
                ++libxstream_opencl_config.ndevices;
              }
# if defined(CL_VERSION_1_2)
              else if (1 < n) { /* create subdevices */
                if (LIBXSTREAM_MAXNDEVS < (libxstream_opencl_config.ndevices + n)) {
                  n = (cl_uint)LIBXSTREAM_MAXNDEVS - libxstream_opencl_config.ndevices;
                }
                if (EXIT_SUCCESS == clCreateSubDevices(devices[j], properties, n,
                                      libxstream_opencl_config.devices + libxstream_opencl_config.ndevices, NULL))
                {
                  CL_CHECK(result, clReleaseDevice(devices[j]));
                  libxstream_opencl_config.ndevices += n;
                }
                else break;
              }
              else {
                libxstream_opencl_config.devices[libxstream_opencl_config.ndevices] = devices[j];
                ++libxstream_opencl_config.ndevices;
              }
# endif
            }
          } /*else break;*/
        }
      }
    }
    if (EXIT_SUCCESS == result && 0 < libxstream_opencl_config.ndevices) {
      const char* const env_vendor = getenv("LIBXSTREAM_VENDOR");
      /* filter device by vendor (if requested) */
      if (NULL != env_vendor && '\0' != *env_vendor) {
        for (i = 0; LIBXS_CAST_INT(i) < libxstream_opencl_config.ndevices;) {
          if (EXIT_SUCCESS ==
              clGetDeviceInfo(libxstream_opencl_config.devices[i], CL_DEVICE_VENDOR, LIBXSTREAM_BUFFERSIZE, buffer, NULL))
          {
            if (NULL == libxs_stristr(buffer, env_vendor)) {
# if defined(CL_VERSION_1_2)
              LIBXS_EXPECT_DEBUG(EXIT_SUCCESS == clReleaseDevice(libxstream_opencl_config.devices[i]));
# endif
              --libxstream_opencl_config.ndevices;
              if (LIBXS_CAST_INT(i) < libxstream_opencl_config.ndevices) { /* keep original order (stable) */
                memmove(&libxstream_opencl_config.devices[i], &libxstream_opencl_config.devices[i + 1],
                  sizeof(cl_device_id) * (libxstream_opencl_config.ndevices - i));
              }
            }
            else ++i;
          }
          else break; /* error: retrieving device vendor */
        }
      }
      /* reorder devices according to libxstream_opencl_order_devices */
      if (EXIT_SUCCESS == result && 1 < libxstream_opencl_config.ndevices) {
        qsort(libxstream_opencl_config.devices, libxstream_opencl_config.ndevices, sizeof(cl_device_id),
          libxstream_opencl_order_devices);
      }
      /* LIBXSTREAM_DEVIDS is parsed as a list of devices (whitelist) */
      if (EXIT_SUCCESS == result && NULL != env_devids && '\0' != *env_devids) {
        cl_uint devids[LIBXSTREAM_MAXNDEVS], ndevids = 0;
        int di = 0, dlen = 0;
        const char* did = libxs_strtoken(env_devids, LIBXS_DELIMS " ", di, &dlen);
        for (; NULL != did && ndevids < LIBXSTREAM_MAXNDEVS;
          did = libxs_strtoken(env_devids, LIBXS_DELIMS " ", ++di, &dlen))
        {
          const int id = (int)strtol(did, NULL, 10);
          if (0 <= id && id < libxstream_opencl_config.ndevices) devids[ndevids++] = id;
        }
        if (0 < ndevids) {
          ndevices = (cl_uint)libxstream_opencl_config.ndevices;
          for (i = 0; i < ndevices; ++i) {
            cl_uint match = 0, j = 0;
            do
              if (i == devids[j]) {
                match = 1;
                break;
              }
            while (++j < ndevids);
            if (0 == match) {
# if defined(CL_VERSION_1_2)
              LIBXS_EXPECT_DEBUG(EXIT_SUCCESS == clReleaseDevice(libxstream_opencl_config.devices[i]));
# endif
              libxstream_opencl_config.devices[i] = NULL;
            }
          }
          for (i = libxstream_opencl_config.ndevices - 1;; --i) {
            if (NULL == libxstream_opencl_config.devices[i]) { /* keep original order (stable) */
              const cl_uint nmove = libxstream_opencl_config.ndevices - (i + 1);
              if (0 < nmove) {
                memmove(
                  &libxstream_opencl_config.devices[i], &libxstream_opencl_config.devices[i + 1], sizeof(cl_device_id) * nmove);
              }
              --libxstream_opencl_config.ndevices;
            }
            if (0 == i) break;
          }
        }
      }
    }
    if (EXIT_SUCCESS == result && 0 < libxstream_opencl_config.ndevices) {
      /* preselect any default device or prune to homogeneous set of devices */
      if (NULL == env_device || '\0' == *env_device) {
        char tmp[LIBXSTREAM_BUFFERSIZE] = "";
        ndevices = (cl_uint)libxstream_opencl_config.ndevices;
        for (i = 0; i < ndevices; ++i) {
          cl_device_type itype;
          result = clGetDeviceInfo(libxstream_opencl_config.devices[i], CL_DEVICE_TYPE, sizeof(cl_device_type), &itype, NULL);
          if (EXIT_SUCCESS == result) {
            if (0 != (CL_DEVICE_TYPE_DEFAULT & itype)) {
              if (0 < i) {
                libxstream_opencl_config.devices[0] = libxstream_opencl_config.devices[i];
              }
              libxstream_opencl_config.ndevices = 1;
              device_id = 0;
              break;
            }
            else if (CL_DEVICE_TYPE_ALL == type && NULL == env_devtype /*&& CL_DEVICE_TYPE_GPU == itype*/ &&
                     device_id <= LIBXS_CAST_INT(i))
            {
              result = clGetDeviceInfo(libxstream_opencl_config.devices[i], CL_DEVICE_NAME, LIBXSTREAM_BUFFERSIZE, buffer, NULL);
              if (EXIT_SUCCESS == result /* prune for homogeneous set of devices */
                  && ('\0' == *tmp || 0 == strncmp(buffer, tmp, LIBXSTREAM_BUFFERSIZE)))
              {
                libxstream_opencl_config.ndevices = i + 1;
                strncpy(tmp, buffer, LIBXSTREAM_BUFFERSIZE);
                tmp[LIBXSTREAM_BUFFERSIZE - 1] = '\0';
              }
              else break; /* error: retrieving device name */
            }
          }
          else break; /* error: retrieving device type */
        }
      }
      else { /* prune number of devices to only expose requested ID */
        if (1 < libxstream_opencl_config.ndevices) {
          if (0 < device_id) {
            libxstream_opencl_config.devices[0] = libxstream_opencl_config.devices[device_id % libxstream_opencl_config.ndevices];
          }
          libxstream_opencl_config.ndevices = 1;
        }
        device_id = 0;
      }
    }
    if (device_id < libxstream_opencl_config.ndevices) {
      if (EXIT_SUCCESS == result) {
        const size_t nhandles = LIBXSTREAM_MAXNITEMS * libxstream_opencl_config.nthreads;
        assert(0 < libxstream_opencl_config.ndevices);
        assert(libxstream_opencl_config.ndevices < LIBXSTREAM_MAXNDEVS);
        assert(NULL == libxstream_opencl_config.memptrs);
        assert(NULL == libxstream_opencl_config.memptr_data);
        assert(0 == libxstream_opencl_config.nmemptrs);
        assert(NULL == libxstream_opencl_config.streams);
        assert(NULL == libxstream_opencl_config.events);
        assert(NULL == libxstream_opencl_config.stream_data);
        assert(NULL == libxstream_opencl_config.event_data);
        assert(0 == libxstream_opencl_config.nstreams);
        assert(0 == libxstream_opencl_config.nevents);
        /* allocate and initialize memptr registry */
        libxstream_opencl_config.nmemptrs = nhandles;
        libxstream_opencl_config.memptrs = (libxstream_opencl_info_memptr_t**)malloc(
          sizeof(libxstream_opencl_info_memptr_t*) * nhandles);
        libxstream_opencl_config.memptr_data = (libxstream_opencl_info_memptr_t*)malloc(
          sizeof(libxstream_opencl_info_memptr_t) * nhandles);
        if (NULL != libxstream_opencl_config.memptrs && NULL != libxstream_opencl_config.memptr_data) {
          libxs_pmalloc_init(sizeof(libxstream_opencl_info_memptr_t), &libxstream_opencl_config.nmemptrs,
            (void**)libxstream_opencl_config.memptrs, libxstream_opencl_config.memptr_data);
        }
        else {
          free(libxstream_opencl_config.memptrs);
          free(libxstream_opencl_config.memptr_data);
          libxstream_opencl_config.memptr_data = NULL;
          libxstream_opencl_config.memptrs = NULL;
          libxstream_opencl_config.nmemptrs = 0;
          result = EXIT_FAILURE;
        }
        /* allocate and initialize streams registry */
        libxstream_opencl_config.nstreams = nhandles;
        libxstream_opencl_config.streams = (libxstream_opencl_stream_t**)malloc(sizeof(libxstream_opencl_stream_t*) * nhandles);
        libxstream_opencl_config.stream_data = (libxstream_opencl_stream_t*)malloc(sizeof(libxstream_opencl_stream_t) * nhandles);
        if (NULL != libxstream_opencl_config.streams && NULL != libxstream_opencl_config.stream_data) {
          libxs_pmalloc_init(sizeof(libxstream_opencl_stream_t), &libxstream_opencl_config.nstreams,
            (void**)libxstream_opencl_config.streams, libxstream_opencl_config.stream_data);
        }
        else {
          free(libxstream_opencl_config.streams);
          free(libxstream_opencl_config.stream_data);
          libxstream_opencl_config.stream_data = NULL;
          libxstream_opencl_config.streams = NULL;
          libxstream_opencl_config.nstreams = 0;
          result = EXIT_FAILURE;
        }
        /* allocate and initialize events registry */
        libxstream_opencl_config.nevents = nhandles;
        libxstream_opencl_config.events = (cl_event**)malloc(sizeof(cl_event*) * nhandles);
        libxstream_opencl_config.event_data = (cl_event*)malloc(sizeof(cl_event) * nhandles);
        if (NULL != libxstream_opencl_config.events && NULL != libxstream_opencl_config.event_data) {
          libxs_pmalloc_init(sizeof(cl_event*), &libxstream_opencl_config.nevents, (void**)libxstream_opencl_config.events,
            libxstream_opencl_config.event_data);
        }
        else {
          free(libxstream_opencl_config.events);
          free(libxstream_opencl_config.event_data);
          libxstream_opencl_config.event_data = NULL;
          libxstream_opencl_config.events = NULL;
          libxstream_opencl_config.nevents = 0;
          result = EXIT_FAILURE;
        }
        /* allocate and initialize per-launch profile records (only if profiling) */
        if (EXIT_SUCCESS == result && 0 != libxstream_opencl_config.profile) {
          libxstream_opencl_config.nlaunch_infos = nhandles;
          libxstream_opencl_config.launch_infos = (libxstream_opencl_launch_info_t**)malloc(
            sizeof(libxstream_opencl_launch_info_t*) * nhandles);
          libxstream_opencl_config.launch_info_data = (libxstream_opencl_launch_info_t*)malloc(
            sizeof(libxstream_opencl_launch_info_t) * nhandles);
          if (NULL != libxstream_opencl_config.launch_infos && NULL != libxstream_opencl_config.launch_info_data) {
            libxs_pmalloc_init(sizeof(libxstream_opencl_launch_info_t), &libxstream_opencl_config.nlaunch_infos,
              (void**)libxstream_opencl_config.launch_infos, libxstream_opencl_config.launch_info_data);
          }
          else { /* profiling is optional: proceed without it rather than failing */
            free(libxstream_opencl_config.launch_infos);
            free(libxstream_opencl_config.launch_info_data);
            libxstream_opencl_config.launch_info_data = NULL;
            libxstream_opencl_config.launch_infos = NULL;
            libxstream_opencl_config.nlaunch_infos = 0;
          }
        }
        /* create host memory pool (USM/SVM-aware custom allocator) */
        if (EXIT_SUCCESS == result) {
          libxstream_opencl_config.pool_hst = libxs_malloc_xpool(
            (libxs_malloc_xfn)libxstream_mem_hst_xmalloc,
            (libxs_free_xfn)libxstream_mem_hst_xfree, libxstream_opencl_config.nthreads);
          if (NULL == libxstream_opencl_config.pool_hst) result = EXIT_FAILURE;
        }
        if (0 != libxstream_opencl_config.profile_mem) {
          const int profile = LIBXS_MAX(LIBXS_ABS(libxstream_opencl_config.profile_mem), 2);
          /**
           * {size, size, duration}, all averaged. Averaged rather than
           * accumulated for the reason spelled out at the kernel histogram: a
           * summed duration divided a per-sample amount by a bucket total, which
           * understated the rate by roughly the number of samples sharing a
           * bucket. Three values rather than two because the rate must not use
           * vals[0]: that is the binning key, which query_percentile derives from
           * the bucket's axis position instead of from the samples, so it equals
           * the transferred amount only while every sample has the same size.
           * Mixed sizes (a panelled upload, or zero-fills covering both a small
           * exponent array and a large slice plane) otherwise paired an
           * interpolated size with an unrelated duration, making the reported
           * rate depend on the bucket count.
           */
          const libxs_hist_update_t update[] = {libxs_hist_update_avg, libxs_hist_update_avg, libxs_hist_update_avg};
          libxstream_opencl_config.hist_h2d = libxs_hist_create(profile + 1, 3, update);
          libxstream_opencl_config.hist_d2h = libxs_hist_create(profile + 1, 3, update);
          libxstream_opencl_config.hist_d2d = libxs_hist_create(profile + 1, 3, update);
          libxstream_opencl_config.hist_zero = libxs_hist_create(profile + 1, 3, update);
        }
        else {
          assert(NULL == libxstream_opencl_config.hist_h2d);
          assert(NULL == libxstream_opencl_config.hist_d2h);
          assert(NULL == libxstream_opencl_config.hist_d2d);
          assert(NULL == libxstream_opencl_config.hist_zero);
        }
        if (EXIT_SUCCESS == result) { /* lastly, print list of devices and actived device */
          const unsigned int nrank = libxs_nrank();
# if defined(LIBXSTREAM_ACTIVATE) && (0 <= LIBXSTREAM_ACTIVATE)
          if (LIBXSTREAM_ACTIVATE < libxstream_opencl_config.ndevices) {
            result = libxstream_opencl_set_active_device(NULL /*lock*/, LIBXSTREAM_ACTIVATE);
          }
          else
# endif
          { /* auto-select initial device */
            if (0 < nrank && 1 < libxstream_opencl_config.ndevices) {
              device_id = nrank % libxstream_opencl_config.ndevices;
            }
            result = libxstream_opencl_set_active_device(NULL /*lock*/, device_id);
          }
          if (EXIT_SUCCESS == result && (
# if (1 >= LIBXSTREAM_USM)
              NULL != libxstream_opencl_config.device.clDeviceMemAllocINTEL ||
              NULL != libxstream_opencl_config.device.clSharedMemAllocINTEL ||
# endif
# if (0 != LIBXSTREAM_USM)
              0 != libxstream_opencl_config.device.usm ||
# endif
              0 /*sentinel*/))
          {
            libxstream_opencl_config.pool_dev = libxs_malloc_xpool(
              (libxs_malloc_xfn)libxstream_mem_dev_xmalloc,
              (libxs_free_xfn)libxstream_mem_dev_xfree, libxstream_opencl_config.nthreads);
          }
          if ((2 <= libxstream_opencl_config.verbosity || 0 > libxstream_opencl_config.verbosity) && (0 == nrank)) {
            char platform_name[LIBXSTREAM_BUFFERSIZE];
            for (i = 0; i < (cl_uint)libxstream_opencl_config.ndevices; ++i) {
              if (EXIT_SUCCESS == libxstream_opencl_device_name(libxstream_opencl_config.devices[i], buffer, LIBXSTREAM_BUFFERSIZE,
                                    platform_name, LIBXSTREAM_BUFFERSIZE, /*cleanup*/ 0))
              {
                fprintf(stderr, "INFO ACC/OpenCL: DEVICE -> \"%s : %s\" (%u)\n", platform_name, buffer, i);
              }
            }
          }
        }
      }
    }
    else { /* mark as initialized */
      libxstream_opencl_config.ndevices = -1;
    }
    LIBXS_MXCSR_SET(mxcsr_saved);
  }
  CL_RETURN(result, "");
}


/* attempt to automatically initialize backend */
LIBXSTREAM_API_INTERN LIBXS_ATTRIBUTE_CTOR void libxstream_opencl_init(void)
{
  if (NULL == libxstream_opencl_config.lock_main) { /* avoid to configure multiple times */
    libxstream_opencl_setup();
  }
# if defined(LIBXSTREAM_PREINIT)
  LIBXS_EXPECT_DEBUG(EXIT_SUCCESS == libxstream_init());
# endif
}


/**
 * Print the identifying prefix shared by every PROF row. Under Slurm the job-ID
 * is included so rows from concurrent jobs remain attributable.
 */
LIBXSTREAM_API_INTERN void libxstream_opencl_print_id(FILE* ostream, const char name[]);
LIBXSTREAM_API_INTERN void libxstream_opencl_print_id(FILE* ostream, const char name[])
{
  const char* const env_slurm = getenv("SLURM_JOBID");
  const int slurm = (NULL == env_slurm ? -1 : atoi(env_slurm));
  if (0 > slurm) fprintf(ostream, "\nPROF ACC/OpenCL: ID=%i %s", libxs_rid(), name);
  else fprintf(ostream, "\nPROF ACC/OpenCL: ID=%i.%i %s", slurm, libxs_rid(), name);
}


/**
 * Print one histogram whose samples are {amount, amount, us} (transfers,
 * amount_first non-zero) or {ms, gflop, mb} (kernels). Both are the same
 * measurement -- an amount of work over the time it took -- so both are reported
 * here rather than in separate routines: a transfer yields a rate in the caller's
 * unit, a kernel yields its duration plus whatever rates the stated work
 * supports.
 *
 * A rate is always derived from a single sample's amount and that same sample's
 * duration, never from independently aggregated totals, and never from vals[0]
 * of a transfer: that slot is the binning key, reconstructed from the bucket's
 * axis position rather than from the samples it holds. Returns 1 if a row was
 * printed, 0 if the histogram held no usable sample.
 */
LIBXSTREAM_API_INTERN int libxstream_opencl_print_hist(
  FILE* ostream, const libxs_hist_t* hist, const char name[], int amount_first, const char unit[], double scale);
LIBXSTREAM_API_INTERN int libxstream_opencl_print_hist(
  FILE* ostream, const libxs_hist_t* hist, const char name[], int amount_first, const char unit[], double scale)
{
  int result = 0;
  if (NULL != hist) {
    /**
     * An empty histogram leaves vals untouched, so it must be cleared: reading
     * it uninitialized (or inheriting a previous kind's median) reported a bogus
     * rate for kinds that never recorded a sample.
     */
    double vals[3];
    vals[0] = 0;
    vals[1] = 0;
    vals[2] = 0;
    libxs_hist_query_median(NULL /*lock*/, hist, vals);
    if (0 != amount_first) { /* transfers: {amount, amount, us}, amount per time */
      /**
       * The mode rather than the median: transfer sizes are commonly multi-modal
       * (a whole operand alongside per-panel blocks, or a zero-fill covering both
       * a small exponent array and a large slice plane), and a median can fall
       * between the clusters and describe no observed transfer. The mode always
       * names a bucket that samples landed in. For a single-sized workload the two
       * agree, so nothing is lost where the median was already right.
       */
      libxs_hist_query_mode(NULL /*lock*/, hist, vals);
      if (0 < vals[2]) {
        /**
         * prec[0] also formats the bucket bound itself and a negative value
         * suppresses the whole line, so the key column stays enabled; it repeats
         * the amount, which is what the bound already conveys.
         */
        const int precision[] = {1, 1, 1};
        libxstream_opencl_print_id(ostream, name);
        /* one decimal: a whole-number GB/s would quantize slow transfers away */
        fprintf(ostream, "=%.1f %s", scale * vals[1] / vals[2], unit);
        libxs_hist_print(ostream, hist, precision, "\n");
        result = 1;
      }
    }
    else if (0 < vals[0]) { /* kernels: {ms, gflop, mb}, time always, rates if stated */
      /**
       * 3 decimals: kernels span microseconds to hundreds of milliseconds, and a
       * coarser format would print every short kernel as 0.000. A negative
       * precision suppresses a column, which is how the work amounts a caller
       * did not state are kept out of the per-bucket detail rather than shown
       * as a column of zeros.
       */
      int precision[3];
      precision[0] = 3;
      precision[1] = (0 < vals[1] ? 1 : -1);
      precision[2] = (0 < vals[2] ? 1 : -1);
      libxstream_opencl_print_id(ostream, name);
      fprintf(ostream, "=%.3f %s", vals[0], unit);
      if (0 < vals[1]) fprintf(ostream, " %.1f GFLOPS/s", 1E3 * vals[1] / vals[0]);
      if (0 < vals[2]) fprintf(ostream, " %.1f GB/s", vals[2] / vals[0]);
      libxs_hist_print(ostream, hist, precision, "\n");
      result = 1;
    }
  }
  return result;
}


/** Print the transfer histograms. Caller holds the stdio lock. */
LIBXSTREAM_API_INTERN int libxstream_opencl_print_transfers(FILE* ostream, const libxs_hist_t* hist[], int nhist);
LIBXSTREAM_API_INTERN int libxstream_opencl_print_transfers(FILE* ostream, const libxs_hist_t* hist[], int nhist)
{
  const char *const kind[] = { "H2D", "D2H", "D2D", "ZERO" };
  int nrows = 0, i;
  assert(nhist <= (int)(sizeof(kind) / sizeof(*kind)));
  for (i = 0; i < nhist; ++i) {
    /**
     * GB/s: samples carry megabytes over microseconds, so the ratio is MB/us and
     * 1E3 converts it. Matches the unit the kernel rows already report, and
     * device-attached memory reaches five digits of MB/s where GB/s stays
     * legible. The per-bucket columns keep their own units (MB, microseconds).
     */
    nrows += libxstream_opencl_print_hist(ostream, hist[i], kind[i], 1 /*amount_first*/, "GB/s", 1E3);
  }
  return nrows;
}


/** Print the per-kernel histograms. Caller holds the stdio lock. */
LIBXSTREAM_API_INTERN int libxstream_opencl_print_kernels(FILE* ostream);
LIBXSTREAM_API_INTERN int libxstream_opencl_print_kernels(FILE* ostream)
{
  int nrows = 0;
  size_t i;
  for (i = 0; i < libxstream_opencl_config.nkernels; ++i) {
    nrows += libxstream_opencl_print_hist(ostream, libxstream_opencl_config.hist_kernel[i],
      libxstream_opencl_config.name_kernel[i], 0 /*amount_first*/, "ms", 1.0);
  }
  if (0 != libxstream_opencl_config.nprofile_kernel_lost) {
    /* a silent cap would read as "these are all the kernels" */
    fprintf(ostream, "\nPROF ACC/OpenCL: kernels=%i (%lu not profiled, raise LIBXSTREAM_MAXNKERNELS)",
      (LIBXSTREAM_MAXNKERNELS), (unsigned long)libxstream_opencl_config.nprofile_kernel_lost);
    ++nrows;
  }
  return nrows;
}


/**
 * Print the sample floor and what it cost. Kept apart from the rate rows: this
 * describes the clock rather than a transfer, carries no byte count, and is only
 * meaningful where samples were actually dropped -- alongside complete figures
 * it is noise. Returns the number of rows printed. Caller holds the stdio lock.
 */
LIBXSTREAM_API_INTERN int libxstream_opencl_print_floor(FILE* ostream);
LIBXSTREAM_API_INTERN int libxstream_opencl_print_floor(FILE* ostream)
{
  const unsigned long ndiscarded = (unsigned long)libxstream_opencl_config.nprofile_short;
  int nrows = 0;
  if (0 != ndiscarded) {
    const unsigned long timer_ns = (unsigned long)libxstream_opencl_config.device.timer_ns;
    fprintf(ostream, "\nPROF ACC/OpenCL: discarded=%lu", ndiscarded);
    if (0 != timer_ns) { /* floor is device-derived: ticks x granularity */
      fprintf(ostream, " timer=%luns", timer_ns);
      if (1 < (LIBXSTREAM_PROFILE_TICKS)) {
        fprintf(ostream, " floor=%luns", (unsigned long)(LIBXSTREAM_PROFILE_TICKS) * timer_ns);
      }
    }
    nrows = 1;
  }
  /**
   * A profile that collected nothing whatsoever is reported, because silence
   * there is indistinguishable from a run that simply transferred nothing. This
   * is the shape the command-type attribution defect took: every sample dropped,
   * no output, and no indication that anything had gone missing.
   */
  else if (0 == libxstream_opencl_config.nprofile) {
    fprintf(ostream, "\nPROF ACC/OpenCL: no samples recorded");
    nrows = 1;
  }
  return nrows;
}


/* attempt to automatically finalize backend */
LIBXSTREAM_API_INTERN LIBXS_ATTRIBUTE_DTOR void libxstream_opencl_finalize(void)
{
  assert(libxstream_opencl_config.ndevices < LIBXSTREAM_MAXNDEVS);
  if (0 != libxstream_opencl_config.ndevices) {
    const libxs_hist_t* hist[] = { NULL, NULL, NULL, NULL };
    const int nhist = (int)(sizeof(hist) / sizeof(*hist));
    int i;
    hist[0] = libxstream_opencl_config.hist_h2d;
    hist[1] = libxstream_opencl_config.hist_d2h;
    hist[2] = libxstream_opencl_config.hist_d2d;
    hist[3] = libxstream_opencl_config.hist_zero;
    /**
     * Print only what was requested: kernel rows for LIBXSTREAM_PROFILE and
     * transfer rows for LIBXSTREAM_PROFILE_MEM. The two mix only when both are
     * given, which is what keeps a rate row from being read as a duration.
     */
    if (0 != libxstream_opencl_config.profile || 0 != libxstream_opencl_config.profile_mem) {
      int nrows = 0;
      LIBXS_STDIO_ACQUIRE();
      if (0 != libxstream_opencl_config.profile) nrows += libxstream_opencl_print_kernels(stderr);
      if (0 != libxstream_opencl_config.profile_mem) nrows += libxstream_opencl_print_transfers(stderr, hist, nhist);
      nrows += libxstream_opencl_print_floor(stderr);
      if (0 != nrows) fprintf(stderr, "\n\n");
      LIBXS_STDIO_RELEASE();
    }
    /**
     * Both counts, because the interesting outcome is a partial one: memory the
     * CUDA runtime refused stays pageable for its transfers, which reads as a
     * slow GEMM rather than as a slow copy.
     */
    if (0 != libxstream_opencl_config.nhostreg &&
        (2 <= libxstream_opencl_config.verbosity || 0 > libxstream_opencl_config.verbosity))
    {
      fprintf(stderr, "INFO ACC/OpenCL: %lu of %lu host allocations registered with the CUDA runtime\n",
        (unsigned long)libxstream_opencl_config.nhostreg_ok, (unsigned long)libxstream_opencl_config.nhostreg);
    }
    for (i = 0; i < LIBXSTREAM_MAXNDEVS; ++i) {
      const cl_device_id device_id = libxstream_opencl_config.devices[i];
      if (NULL != device_id) {
# if defined(CL_VERSION_1_2) && 0 /* avoid potential segfault */
        LIBXS_EXPECT_DEBUG(EXIT_SUCCESS == clReleaseDevice(device_id));
# endif
      }
    }
    /* release/reset buffers */
    libxs_hist_destroy(libxstream_opencl_config.hist_h2d);
    libxs_hist_destroy(libxstream_opencl_config.hist_d2h);
    libxs_hist_destroy(libxstream_opencl_config.hist_d2d);
    libxs_hist_destroy(libxstream_opencl_config.hist_zero);
    for (i = 0; i < LIBXS_CAST_INT(libxstream_opencl_config.nkernels); ++i) {
      void* name;
      libxs_hist_destroy(libxstream_opencl_config.hist_kernel[i]);
      LIBXS_UNION_ASSIGN(void*, name, const char*, libxstream_opencl_config.name_kernel[i]);
      free(name); /* strdup'ed from CL_KERNEL_FUNCTION_NAME */
      libxstream_opencl_config.hist_kernel[i] = NULL;
      libxstream_opencl_config.name_kernel[i] = NULL;
    }
    libxstream_opencl_config.nkernels = 0;
    free(libxstream_opencl_config.launch_infos);
    free(libxstream_opencl_config.launch_info_data);
    libxstream_opencl_config.launch_info_data = NULL;
    libxstream_opencl_config.launch_infos = NULL;
    libxstream_opencl_config.nlaunch_infos = 0;
    libxs_free_pool(libxstream_opencl_config.pool_dev);
    libxs_free_pool(libxstream_opencl_config.pool_hst);
    if (NULL != libxstream_opencl_config.pool_hst_queue) {
      clReleaseCommandQueue(libxstream_opencl_config.pool_hst_queue); /* ignore return code */
    }
    if (NULL != libxstream_opencl_config.pool_hst_context) {
      clReleaseContext(libxstream_opencl_config.pool_hst_context); /* ignore return code */
    }
    if (NULL != libxstream_opencl_config.device.stream.queue) { /* release private stream */
      clReleaseCommandQueue(libxstream_opencl_config.device.stream.queue); /* ignore return code */
    }
    if (NULL != libxstream_opencl_config.device.memptr_kernel) {
      clReleaseKernel(libxstream_opencl_config.device.memptr_kernel); /* ignore return code */
      libxstream_opencl_config.device.memptr_kernel = NULL;
      libxstream_opencl_config.device.memptr_context = NULL;
    }
    if (NULL != libxstream_opencl_config.device.context) {
      const cl_context context = libxstream_opencl_config.device.context;
      libxstream_opencl_config.device.context = NULL;
      clReleaseContext(context); /* ignore return code */
    }
    for (i = 0; i < LIBXSTREAM_NLOCKS; ++i) { /* destroy locks */
      LIBXS_LOCK_DESTROY(LIBXS_LOCK, (libxs_lock_t*)(internal_libxstream_opencl_locks + LIBXS_CACHELINE * i));
    }
    /**
     * NOTE: registered streams/events are not individually released here;
     * the OpenCL runtime reclaims resources at process exit (atexit context).
     */
    free(libxstream_opencl_config.memptrs);
    free(libxstream_opencl_config.memptr_data);
    free(libxstream_opencl_config.subs);
    free(libxstream_opencl_config.streams);
    free(libxstream_opencl_config.stream_data);
    free(libxstream_opencl_config.events);
    free(libxstream_opencl_config.event_data);
    /* clear entire configuration structure */
    memset(&libxstream_opencl_config, 0, sizeof(libxstream_opencl_config));
# if defined(LIBXSTREAM_CACHE_DID)
    internal_libxstream_opencl_active_id = 0; /* reset cached active device-ID */
# endif
    libxs_finalize();
  }
}


LIBXSTREAM_API int libxstream_finalize(void)
{
# if defined(_OPENMP)
  /* initialization/finalization is not meant to be thread-safe */
  int result = ((0 == omp_in_parallel() || /*main*/ 0 == omp_get_thread_num()) ? EXIT_SUCCESS : EXIT_FAILURE);
# else
  int result = EXIT_SUCCESS;
# endif
  static void (*cleanup)(void) = libxstream_opencl_finalize;
  assert(libxstream_opencl_config.ndevices < LIBXSTREAM_MAXNDEVS);
  if (0 != libxstream_opencl_config.ndevices && NULL != cleanup) {
    if (EXIT_SUCCESS == result) result = atexit(cleanup);
    cleanup = NULL;
  }
  CL_RETURN(result, "");
}


LIBXSTREAM_API int libxstream_opencl_use_cmem_size(const libxstream_opencl_device_t* devinfo, size_t size)
{
# if defined(LIBXSTREAM_CMEM)
  const size_t needed = (0 != size) ? size : devinfo->size_maxalloc;
  return (0 != devinfo->size_maxcmem && needed <= devinfo->size_maxcmem) ? EXIT_SUCCESS : EXIT_FAILURE;
# else
  LIBXS_UNUSED(size);
  return EXIT_FAILURE;
# endif
}


LIBXSTREAM_API int libxstream_opencl_use_cmem(const libxstream_opencl_device_t* devinfo)
{
  return libxstream_opencl_use_cmem_size(devinfo, 0);
}


LIBXSTREAM_API int libxstream_device_count(int* ndevices)
{
  int result;
  result = libxstream_init();
  if (EXIT_SUCCESS == result)
  {
    if (NULL != ndevices) {
      *ndevices = (0 < libxstream_opencl_config.ndevices ? libxstream_opencl_config.ndevices : 0);
      result = EXIT_SUCCESS;
    }
    else result = EXIT_FAILURE;
  }
  CL_RETURN(result, "");
}


LIBXSTREAM_API int libxstream_opencl_device_id(cl_device_id device, int* device_id, int* global_id)
{
  int result = EXIT_SUCCESS, i;
  assert(libxstream_opencl_config.ndevices < LIBXSTREAM_MAXNDEVS);
  assert(NULL != device_id || NULL != global_id);
  for (i = 0; i < libxstream_opencl_config.ndevices; ++i) {
    if (device == libxstream_opencl_config.devices[i]) break;
  }
  if (i < libxstream_opencl_config.ndevices) {
    if (NULL != device_id) *device_id = i;
    if (NULL != global_id) {
      *global_id = i;
      for (++i; i < LIBXSTREAM_MAXNDEVS; ++i) {
        if (NULL != libxstream_opencl_config.devices[i]) {
          if (device == libxstream_opencl_config.devices[i]) {
            *global_id = i;
            break;
          }
        }
        else break;
      }
    }
  }
  else {
    if (NULL != device_id) *device_id = -1;
    if (NULL != global_id) *global_id = -1;
    if (NULL != device) result = EXIT_FAILURE;
  }
  return result;
}


LIBXSTREAM_API int libxstream_opencl_device_vendor(cl_device_id device, const char vendor[], int use_platform_name)
{
  char buffer[LIBXSTREAM_BUFFERSIZE];
  int result = EXIT_SUCCESS;
  assert(NULL != device && NULL != vendor);
  if (0 == use_platform_name) {
    result = clGetDeviceInfo(device, CL_DEVICE_VENDOR, LIBXSTREAM_BUFFERSIZE, buffer, NULL);
  }
  else {
    cl_platform_id platform;
    result = clGetDeviceInfo(device, CL_DEVICE_PLATFORM, sizeof(cl_platform_id), &platform, NULL);
    if (EXIT_SUCCESS == result) {
      result = clGetPlatformInfo(
        platform, 1 == use_platform_name ? CL_PLATFORM_NAME : CL_PLATFORM_VENDOR, LIBXSTREAM_BUFFERSIZE, buffer, NULL);
    }
  }
  if (EXIT_SUCCESS == result) {
    result = (NULL != libxs_stristr(buffer, vendor) ? EXIT_SUCCESS : EXIT_FAILURE);
  }
  return result;
}


LIBXSTREAM_API int libxstream_opencl_device_uid(cl_device_id device, const char devname[], unsigned int* uid)
{
  int result;
  if (NULL != uid) {
    if (NULL != device && EXIT_SUCCESS == libxstream_opencl_device_vendor(device, "intel", 0 /*use_platform_name*/)) {
      result = clGetDeviceInfo(device, 0x4251 /*CL_DEVICE_ID_INTEL*/, sizeof(unsigned int), uid, NULL);
    }
    else result = EXIT_FAILURE;
    if (EXIT_SUCCESS != result) {
      if (NULL != devname && '\0' != *devname) {
        *uid = LIBXS_CAST_UINT(strtoul(devname, NULL, 0));
        if (0 == *uid) {
          const char *const begin = strrchr(devname, '['), *const end = strrchr(devname, ']');
          if (NULL != begin && begin < end) {
            *uid = LIBXS_CAST_UINT(strtoul(begin + 1, NULL, 0));
          }
          if (0 == *uid) {
            const size_t size = strlen(devname);
            const unsigned int hash = libxs_hash(devname, LIBXS_CAST_UINT(size), 25071975 /*seed*/);
            *uid = libxs_hash(&hash, 4 /*size*/, hash >> 16 /*seed*/) & 0xFFFF;
          }
        }
        result = EXIT_SUCCESS;
      }
      else {
        result = EXIT_FAILURE;
        *uid = 0;
      }
    }
  }
  else result = EXIT_FAILURE;
  return result;
}


LIBXSTREAM_API int libxstream_opencl_device_name(
  cl_device_id device, char name[], size_t name_maxlen, char platform[], size_t platform_maxlen, int cleanup)
{
  int result_name = 0, result_platform = 0;
  assert(NULL != name || NULL != platform);
  if (NULL == device && 0 < libxstream_opencl_config.ndevices) {
    device = libxstream_opencl_config.devices[0]; /* NULL-device refers to device 0 */
  }
  if (NULL != name && 0 != name_maxlen) {
    result_name = clGetDeviceInfo(device, CL_DEVICE_NAME, name_maxlen, name, NULL);
    if (0 != cleanup && EXIT_SUCCESS == result_name) {
      char* const part = strchr(name, ':');
      if (NULL != part) *part = '\0';
    }
  }
  if (NULL != platform && 0 != platform_maxlen) {
    cl_platform_id platform_id;
    result_platform = clGetDeviceInfo(device, CL_DEVICE_PLATFORM, sizeof(cl_platform_id), &platform_id, NULL);
    if (EXIT_SUCCESS == result_platform) {
      result_platform = clGetPlatformInfo(platform_id, CL_PLATFORM_NAME, platform_maxlen, platform, NULL);
    }
  }
  return result_name | result_platform;
}


LIBXSTREAM_API void libxstream_opencl_device_name_cleanup(char name[])
{
  char* dst = name;
  char* src = name;
  if (NULL == name) return;
  while ('\0' != *src) {
    if ('[' == *src && '0' == src[1] && 'x' == src[2]) {
      while ('\0' != *src && ']' != *src) ++src;
      if (']' == *src) ++src;
    }
    else if ('(' == *src && (('R' == src[1] && ')' == src[2]) ||
      ('T' == src[1] && 'M' == src[2] && ')' == src[3])))
    {
      src += ('T' == src[1]) ? 4 : 3;
    }
    else if ((' ' == *src || src == name) && 0 != isdigit((unsigned char)src[' ' == *src ? 1 : 0])) {
      char* p = src + (' ' == *src ? 1 : 0);
      while (0 != isdigit((unsigned char)*p)) ++p;
      if (('G' == *p || 'g' == *p) && ('B' == p[1] || 'b' == p[1]) &&
          ('\0' == p[2] || ' ' == p[2] || '-' == p[2]))
      {
        src = p + 2;
      }
      else {
        *dst++ = *src++;
      }
    }
    else if ('-' == *src && 0 != isdigit((unsigned char)src[1])) {
      char* p = src + 1;
      while (0 != isdigit((unsigned char)*p)) ++p;
      if (('G' == *p || 'g' == *p) && ('B' == p[1] || 'b' == p[1]) &&
          ('\0' == p[2] || ' ' == p[2]))
      {
        src = p + 2;
      }
      else {
        *dst++ = *src++;
      }
    }
    else {
      *dst++ = *src++;
    }
  }
  *dst = '\0';
  while (dst > name && ' ' == dst[-1]) *--dst = '\0';
  dst = name;
  src = name;
  while ('\0' != *src) {
    if (' ' == *src && ' ' == src[1]) {
      ++src;
    }
    else {
      *dst++ = *src++;
    }
  }
  *dst = '\0';
  if (' ' == *name) {
    src = name + 1;
    dst = name;
    while ('\0' != *src) *dst++ = *src++;
    *dst = '\0';
  }
}


LIBXSTREAM_API int libxstream_opencl_device_level(
  cl_device_id device, int std_clevel[2], int std_level[2], char std_flag[32], cl_device_type* type)
{
  char buffer[LIBXSTREAM_BUFFERSIZE];
  unsigned int std_clevel_uint[2] = {0}, std_level_uint[2] = {0};
  int result = EXIT_SUCCESS;
  assert(NULL != device && (NULL != std_clevel || NULL != std_level || NULL != std_flag || NULL != type));
  result = clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, LIBXSTREAM_BUFFERSIZE / 2, buffer, NULL);
  if (EXIT_SUCCESS == result && (NULL != std_clevel || NULL != std_flag)) {
    if (2 == sscanf(buffer, "OpenCL C %u.%u", std_clevel_uint, std_clevel_uint + 1)) {
      if (NULL != std_clevel) {
        std_clevel[0] = LIBXS_CAST_INT(std_clevel_uint[0]);
        std_clevel[1] = LIBXS_CAST_INT(std_clevel_uint[1]);
      }
    }
    else result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result && (NULL != std_level || NULL != std_flag)) {
    result = clGetDeviceInfo(
      device, CL_DEVICE_VERSION, LIBXSTREAM_BUFFERSIZE - LIBXSTREAM_BUFFERSIZE / 2, buffer + LIBXSTREAM_BUFFERSIZE / 2, NULL);
    if (EXIT_SUCCESS == result) {
      if (2 == sscanf(buffer + LIBXSTREAM_BUFFERSIZE / 2, "OpenCL %u.%u", std_level_uint, std_level_uint + 1)) {
        if (NULL != std_level) {
          std_level[0] = LIBXS_CAST_INT(std_level_uint[0]);
          std_level[1] = LIBXS_CAST_INT(std_level_uint[1]);
        }
      }
      else result = EXIT_FAILURE;
    }
  }
  if (EXIT_SUCCESS == result && NULL != std_flag) {
    if (2 <= std_level_uint[0]) {
      const int nchar = LIBXS_SNPRINTF(std_flag, 32, "-cl-std=CL%u.0", std_level_uint[0]);
      if (0 >= nchar || 32 <= nchar) result = EXIT_FAILURE;
    }
    else if (1 <= std_level_uint[0]) {
      if (1 <= std_level_uint[1]) {
        const int nchar = LIBXS_SNPRINTF(std_flag, 32, "-cl-std=CL%u.%u", std_level_uint[0], std_level_uint[1]);
        if (0 >= nchar || 32 <= nchar) result = EXIT_FAILURE;
      }
      else if (1 <= std_clevel_uint[0]) { /* fallback */
        const int nchar = LIBXS_SNPRINTF(std_flag, 32, "-cl-std=CL%u.%u", std_clevel_uint[0], std_clevel_uint[1]);
        if (0 >= nchar || 32 <= nchar) result = EXIT_FAILURE;
      }
      else *std_flag = '\0'; /* not an error */
    }
    else *std_flag = '\0'; /* not an error */
  }
  if (EXIT_SUCCESS == result && NULL != type) {
    result = clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(cl_device_type), type, NULL);
  }
  if (EXIT_SUCCESS != result) {
    if (NULL != std_clevel) std_clevel[0] = std_clevel[1] = 0;
    if (NULL != std_level) std_level[0] = std_level[1] = 0;
    if (NULL != std_flag) *std_flag = '\0';
    if (NULL != type) *type = 0;
  }
  return result;
}


LIBXSTREAM_API int libxstream_opencl_device_ext(cl_device_id device, const char* const extnames[], int num_exts)
{
  int result = ((NULL != extnames && 0 < num_exts) ? EXIT_SUCCESS : EXIT_FAILURE);
  char extensions[LIBXSTREAM_BUFFERSIZE], buffer[LIBXSTREAM_BUFFERSIZE];
  assert(NULL != device);
  CL_CHECK(result, clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, LIBXSTREAM_BUFFERSIZE, extensions, NULL));
  if (EXIT_SUCCESS == result) {
    do {
      if (NULL != extnames[--num_exts]) {
        const char* const end = buffer + strlen(extnames[num_exts]); /* before strtok */
        char* ext;
        strncpy(buffer, extnames[num_exts], LIBXSTREAM_BUFFERSIZE - 1);
        buffer[LIBXSTREAM_BUFFERSIZE - 1] = '\0';
        ext = strtok(buffer, LIBXS_DELIMS " \t");
        for (; NULL != ext; ext = ((ext + 1) < end ? strtok((ext + 1) + strlen(ext), LIBXS_DELIMS " \t") : NULL)) {
          if (NULL == strstr(extensions, ext)) {
            return EXIT_FAILURE;
          }
        }
      }
    } while (0 < num_exts);
  }
  return result;
}


LIBXSTREAM_API int libxstream_opencl_create_context(cl_device_id active_id, cl_context* context)
{
  cl_platform_id platform = NULL;
  int result;
  assert(0 < libxstream_opencl_config.ndevices);
  assert(NULL != active_id && NULL != context);
  result = clGetDeviceInfo(active_id, CL_DEVICE_PLATFORM, sizeof(cl_platform_id), &platform, NULL);
  assert(EXIT_SUCCESS != result || NULL != platform);
  if (EXIT_SUCCESS == result) {
    void (*const notify)(
      const char*, const void*, size_t, void*) = (0 != libxstream_opencl_config.verbosity ? libxstream_opencl_notify : NULL);
    cl_context_properties properties[] = {
      CL_CONTEXT_PLATFORM, 0 /*placeholder*/, 0 /* end of properties */
    };
    cl_context ctx = NULL;
    properties[1] = (cl_context_properties)platform;
    ctx = clCreateContext(properties, 1 /*num_devices*/, &active_id, notify, NULL /* user_data*/, &result);
    if (EXIT_SUCCESS != result && CL_INVALID_DEVICE != result) { /* retry */
      ctx = clCreateContext(NULL /*properties*/, 1 /*num_devices*/, &active_id, notify, NULL /* user_data*/, &result);
    }
    if (EXIT_SUCCESS == result) {
      assert(NULL != ctx);
      *context = ctx;
      if (0 != libxstream_opencl_config.verbosity) {
        char buffer[LIBXSTREAM_BUFFERSIZE];
        int global_id = 0;
        if (EXIT_SUCCESS == libxstream_opencl_device_name(
                              active_id, buffer, LIBXSTREAM_BUFFERSIZE, NULL /*platform*/, 0 /*platform_maxlen*/, /*cleanup*/ 1) &&
            EXIT_SUCCESS == libxstream_opencl_device_id(active_id, NULL /*devid*/, &global_id))
        {
          const size_t size = strlen(buffer);
          unsigned int uid[] = {0, 0};
          if ((EXIT_SUCCESS == libxstream_opencl_device_uid(NULL /*device*/, buffer, uid + 1)) &&
              (EXIT_SUCCESS == libxstream_opencl_device_uid(active_id, NULL /*devname*/, uid) || 0 != uid[1]) && uid[0] != uid[1])
          {
            LIBXS_EXPECT(0 < LIBXS_SNPRINTF(buffer + size, LIBXS_MAX(0, LIBXSTREAM_BUFFERSIZE - size), " [0x%04x]",
                               0 != uid[0] ? uid[0] : uid[1]));
          }
          fprintf(stderr, "INFO ACC/OpenCL: ndevices=%i device%i=\"%s\" context=%p pid=%u nthreads=%i\n",
            libxstream_opencl_config.ndevices, global_id, buffer, (void*)ctx, libxs_pid(), libxstream_opencl_config.nthreads);
        }
      }
    }
    else {
      if (CL_INVALID_DEVICE == result &&
          EXIT_SUCCESS == libxstream_opencl_device_vendor(active_id, "nvidia", 0 /*use_platform_name*/))
      {
        fprintf(stderr, "WARN ACC/OpenCL: if MPI-ranks target the same device in exclusive mode,\n"
                        "                    SMI must be used to enable sharing the device.\n");
      }
      *context = NULL;
    }
  }
  return result;
}


LIBXSTREAM_API int libxstream_opencl_set_active_device(libxs_lock_t* lock, int device_id)
{
  libxstream_opencl_device_t* const devinfo = &libxstream_opencl_config.device;
  int result = EXIT_SUCCESS;
  assert(libxstream_opencl_config.ndevices < LIBXSTREAM_MAXNDEVS);
  if (0 <= device_id && device_id < libxstream_opencl_config.ndevices) {
    /* accessing devices is thread-safe (array is fixed after initialization) */
    const cl_device_id active_id = libxstream_opencl_config.devices[device_id];
    if (NULL != active_id) {
      cl_context context = NULL;
      if (NULL != lock) LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, lock);
      context = devinfo->context;
      if (NULL != context) {
        if (device_id != libxstream_opencl_config.device_id) {
          const cl_device_id context_id = libxstream_opencl_config.devices[libxstream_opencl_config.device_id];
          assert(NULL != context_id);
# if defined(CL_VERSION_1_2)
          LIBXS_EXPECT_DEBUG(EXIT_SUCCESS == clReleaseDevice(context_id));
# endif
          if (NULL != devinfo->memptr_kernel) {
            LIBXS_EXPECT_DEBUG(EXIT_SUCCESS == clReleaseKernel(devinfo->memptr_kernel));
            devinfo->memptr_kernel = NULL;
            devinfo->memptr_context = NULL;
          }
          result = clReleaseContext(context);
          if (EXIT_SUCCESS == result) devinfo->context = NULL;
          context = NULL;
        }
      }
      if (EXIT_SUCCESS == result && (NULL == devinfo->context || device_id != libxstream_opencl_config.device_id)) {
        result = libxstream_opencl_create_context(active_id, &context);
        assert(NULL != context || EXIT_SUCCESS != result);
      }
      /* update/cache device-specific information */
      if (EXIT_SUCCESS == result && (NULL == devinfo->context || device_id != libxstream_opencl_config.device_id)) {
        if (NULL != devinfo->stream.queue) { /* release private stream */
          LIBXS_EXPECT_DEBUG(EXIT_SUCCESS == clReleaseCommandQueue(devinfo->stream.queue));
        }
        memset(devinfo, 0, sizeof(*devinfo));
        result = libxstream_opencl_device_level(
          active_id, devinfo->std_clevel, devinfo->std_level, devinfo->std_flag, &devinfo->type);
        if (EXIT_SUCCESS == result) {
          char devname[LIBXSTREAM_BUFFERSIZE] = "";
          const char* const sgexts[] = {
            "cl_khr_subgroups", "cl_intel_required_subgroup_size",
            "cl_intel_subgroups", "cl_intel_subgroups_long", 
          };
          size_t sgsizes[16], nbytes = 0, i;
          LIBXSTREAM_STREAM_PROPERTIES_TYPE properties[4] = {
            CL_QUEUE_PROPERTIES, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, 0 /* terminator */
          };
          devinfo->intel = (EXIT_SUCCESS == libxstream_opencl_device_vendor(active_id, "intel", 0 /*use_platform_name*/));
          if (0 != devinfo->intel) { /* intel: 1=GPU, 2=GPU with XMX (DPAS + 2D block I/O) */
            const char* const xmx_exts[] = {
              "cl_intel_subgroup_matrix_multiply_accumulate", "cl_intel_subgroup_2d_block_io"
            };
            if (EXIT_SUCCESS == libxstream_opencl_device_ext(active_id, xmx_exts, 2)) {
              devinfo->intel = 2;
            }
          }
          { const char* const env_intel = getenv("LIBXSTREAM_INTEL");
            if (NULL != env_intel) devinfo->intel = atoi(env_intel);
          }
          devinfo->nv = (EXIT_SUCCESS == libxstream_opencl_device_vendor(active_id, "nvidia", 0 /*use_platform_name*/));
          if (0 != devinfo->nv) { /* nv: 1=generic, 2=SM>=7.5 (dp4a/mma.m8n8k16), 3=SM>=8.0 (mma.m16n8k32), 4=SM>=9.0 (TMA/wgmma) */
            cl_uint sm_major = 0, sm_minor = 0;
            if (EXIT_SUCCESS == clGetDeviceInfo(active_id, 0x4000 /*CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV*/, sizeof(cl_uint),
                                  &sm_major, NULL) &&
                EXIT_SUCCESS == clGetDeviceInfo(active_id, 0x4001 /*CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV*/, sizeof(cl_uint),
                                  &sm_minor, NULL))
            {
              const int sm = (int)(sm_major * 10 + sm_minor);
              if (90 <= sm) devinfo->nv = 4;
              else if (80 <= sm) devinfo->nv = 3;
              else if (75 <= sm) devinfo->nv = 2;
            }
          }
          { const char* const env_nv = getenv("LIBXSTREAM_NV");
            if (NULL != env_nv) devinfo->nv = atoi(env_nv);
          }
          if (EXIT_SUCCESS != libxstream_opencl_device_name(active_id, devname, LIBXSTREAM_BUFFERSIZE, NULL /*platform*/,
                                0 /*platform_maxlen*/, /*cleanup*/ 1) ||
              EXIT_SUCCESS != libxstream_opencl_device_uid(active_id, devname, &devinfo->uid))
          {
            devinfo->uid = (cl_uint)-1;
          }
          if (EXIT_SUCCESS == libxstream_opencl_device_vendor(active_id, "amd", 0 /*use_platform_name*/) ||
              EXIT_SUCCESS == libxstream_opencl_device_vendor(active_id, "amd", 1 /*use_platform_name*/))
          {
            devinfo->amd = 1;
            if ('\0' != *devname) {
              const char* const gfxname = libxs_stristr(devname, "gfx");
              if (NULL != gfxname && 90 <= atoi(gfxname + 3)) {
                devinfo->amd = 2;
              }
            }
          }
          if (EXIT_SUCCESS !=
              clGetDeviceInfo(active_id, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(cl_bool) /*cl_int*/, &devinfo->unified, NULL))
          {
            devinfo->unified = CL_FALSE;
          }
          if (EXIT_SUCCESS !=
              clGetDeviceInfo(active_id, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &devinfo->size_maxalloc, NULL))
          {
            devinfo->size_maxalloc = 0;
          }
          if (EXIT_SUCCESS !=
              clGetDeviceInfo(active_id, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(cl_ulong), &devinfo->size_maxcmem, NULL))
          {
            devinfo->size_maxcmem = 0;
          }
          if (EXIT_SUCCESS != clGetDeviceInfo(active_id, CL_DEVICE_PROFILING_TIMER_RESOLUTION, sizeof(size_t),
                                &devinfo->timer_ns, NULL))
          {
            devinfo->timer_ns = 0; /* unknown: no sample is rejected */
          }
          if (EXIT_SUCCESS != clGetDeviceInfo(active_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), devinfo->wgsize, NULL)) {
            devinfo->wgsize[0] = 1;
          }
          if (EXIT_SUCCESS != clGetDeviceInfo(active_id, 4199 /*CL_DEVICE_PREFERRED_WORK_GROUP_SIZE_MULTIPLE*/, sizeof(size_t),
                                devinfo->wgsize + 1, NULL)) /* CL_VERSION_3_0 */
          {
            devinfo->wgsize[1] = 1;
          }
          assert(0 == devinfo->wgsize[2]);
          if (EXIT_SUCCESS == libxstream_opencl_device_ext(active_id, sgexts, 1) && 1 < devinfo->wgsize[1]) {
            if (0 != devinfo->intel &&
                EXIT_SUCCESS == libxstream_opencl_device_ext(active_id, sgexts + 1, 1) &&
                EXIT_SUCCESS ==
                  clGetDeviceInfo(active_id, 0x4108 /*CL_DEVICE_SUB_GROUP_SIZES_INTEL*/, sizeof(sgsizes), sgsizes, &nbytes))
            {
              for (i = 0; (i * sizeof(size_t)) < nbytes; ++i) {
                const size_t sgsize = sgsizes[i];
                if (devinfo->wgsize[2] < sgsize && (0 == (sgsize % devinfo->wgsize[1]) || 0 == (devinfo->wgsize[1] % sgsize))) {
                  if (devinfo->wgsize[1] < sgsize) devinfo->wgsize[1] = sgsize;
                  devinfo->wgsize[2] = sgsize;
                }
              }
            }
            else devinfo->wgsize[2] = devinfo->wgsize[1]; /* KHR-only */
          }
          if (0 != devinfo->intel) {
            const char* const env_biggrf = getenv("LIBXSTREAM_BIGGRF");
            devinfo->biggrf = (NULL != env_biggrf && 0 != atoi(env_biggrf));
          }
          /**
           * LIBXSTREAM_USM runtime levels:
           *   not set: OpenCL 2.0 SVM coarse-grain (same as level 2)
           *   0: disable all USM, force clCreateBuffer path
           *   1: Intel USM ext
           *   2: OpenCL 2.0 SVM coarse-grain only (skip Intel ext)
           *   3: OpenCL 2.0 SVM with device-reported caps (skip Intel ext)
           *
           * Levels 1 and 3 are opt-in, never reached by the default. Both are
           * faster in a microbenchmark -- the Intel extension gives an actual
           * asynchronous transfer (45.3 vs 9.1 GB/s for a 128 MB H2D on a GPU
           * Max 1550) -- and both are nevertheless excluded here, because the
           * default has to stay predictable across drivers rather than fastest
           * on one. The extension path carries the known issues that motivated
           * gating it behind an explicit request, and level 3 takes whatever the
           * driver reports, which on Xe reaches fine-grain *system* allocations:
           * a far broader contract than coarse-grain buffers, and not something
           * to acquire implicitly from a capability bit.
           *
           * Coarse-grain costs nothing against fine-grain anyway: it adds a
           * SVMMap/SVMUnmap pair that fine-grain omits, but that is bookkeeping
           * and the host memcpy both paths end in dominates -- 15.2 vs 15.1 GB/s
           * on a Xeon 8480+, the device here offering both grains.
           */
          {
            const char* const env_usm = getenv("LIBXSTREAM_USM");
            const int cfg_usm = (0 != internal_libxstream_init_cfg_valid ? internal_libxstream_init_cfg.usm : -1);
# if (0 != LIBXSTREAM_USM)
            /**
             * WA-16 disables USM on unified-memory Intel GPUs (iGPU), where some
             * software stacks are not correct with SVM allocations or transfers.
             * It only supplies the default level, i.e. an explicit request still
             * wins, or level 2 would be unreachable on such a device without also
             * clearing the bit. Restricted to a GPU: an Intel CPU device reports
             * unified memory as well but is correct, and the buffer-based path
             * would cost it a real copy where coarse-grain SVM costs nothing.
             * The same bit additionally forces synchronous transfers, because the
             * buffer-based path is not correct asynchronously there either.
             */
            const int usm_default = (LIBXSTREAM_WA_UNIFIED(devinfo) ? 0 : -1 /*as level 2*/);
# else
            const int usm_default = -1;
# endif
            const int usm_level = (0 <= cfg_usm) ? cfg_usm : (NULL != env_usm ? atoi(env_usm) : usm_default);
# if (0 != LIBXSTREAM_USM)
            /* the only symptom is a different transfer path, hence worth stating */
            if (0 == usm_default && NULL == env_usm && 0 > cfg_usm &&
                (2 <= libxstream_opencl_config.verbosity || 0 > libxstream_opencl_config.verbosity))
            {
              fprintf(stderr, "WARN ACC/OpenCL: USM disabled on unified memory (LIBXSTREAM_WA=%i).\n",
                libxstream_opencl_config.wa);
            }
# endif
# if defined(LIBXSTREAM_XHINTS) && (1 >= LIBXSTREAM_USM)
            /* Intel USM extensions: enabled only by explicit level 1 */
            if (1 == usm_level && 2 <= *devinfo->std_level && 0 != devinfo->intel &&
                (0 == devinfo->unified || NULL != (LIBXSTREAM_XHINTS)))
            {
              cl_platform_id platform = NULL;
              cl_bitfield bitfield = 0;
              if (EXIT_SUCCESS == clGetDeviceInfo(active_id, CL_DEVICE_PLATFORM, sizeof(cl_platform_id), &platform, NULL) &&
                  EXIT_SUCCESS == libxstream_opencl_device_vendor(active_id, "intel", 2 /*platform vendor*/) &&
                  EXIT_SUCCESS == clGetDeviceInfo(active_id, 0x4191 /*CL_DEVICE_DEVICE_MEM_CAPABILITIES_INTEL*/, sizeof(cl_bitfield),
                                    &bitfield, NULL) &&
                  0 != bitfield)
              {
                void* ptr[8] = {NULL};
                int ii = 0, n = 0;
                ptr[0] = clGetExtensionFunctionAddressForPlatform(platform, "clSetKernelArgMemPointerINTEL");
                ptr[1] = clGetExtensionFunctionAddressForPlatform(platform, "clEnqueueMemFillINTEL");
                ptr[2] = clGetExtensionFunctionAddressForPlatform(platform, "clEnqueueMemcpyINTEL");
                ptr[3] = clGetExtensionFunctionAddressForPlatform(platform, "clDeviceMemAllocINTEL");
                ptr[4] = clGetExtensionFunctionAddressForPlatform(platform, "clSharedMemAllocINTEL");
                ptr[5] = clGetExtensionFunctionAddressForPlatform(platform, "clHostMemAllocINTEL");
                ptr[6] = clGetExtensionFunctionAddressForPlatform(platform, "clMemFreeINTEL");
                for (; ii < (int)(sizeof(ptr) / sizeof(*ptr)); ++ii) {
                  if (NULL != ptr[ii]) ++n;
                }
                if (7 == n) {
                  LIBXS_ASSIGN(&devinfo->clSetKernelArgMemPointerINTEL, ptr + 0);
                  LIBXS_ASSIGN(&devinfo->clEnqueueMemFillINTEL, ptr + 1);
                  LIBXS_ASSIGN(&devinfo->clEnqueueMemcpyINTEL, ptr + 2);
                  LIBXS_ASSIGN(&devinfo->clDeviceMemAllocINTEL, ptr + 3);
                  LIBXS_ASSIGN(&devinfo->clSharedMemAllocINTEL, ptr + 4);
                  LIBXS_ASSIGN(&devinfo->clHostMemAllocINTEL, ptr + 5);
                  LIBXS_ASSIGN(&devinfo->clMemFreeINTEL, ptr + 6);
                }
                else if (0 != n) {
                  fprintf(stderr, "WARN ACC/OpenCL: inconsistent state discovered!\n");
                }
              }
            }
# endif
# if (0 != LIBXSTREAM_USM)
            /**
             * OpenCL 2.0 SVM: the default and levels 2-3, or the fallback when an
             * explicit level 1 could not load the Intel extensions. Only level 3
             * keeps the reported capabilities (see the coarse-grain mask below);
             * everything else is restricted to coarse-grain buffers, so a driver
             * advertising fine-grain system allocations does not silently widen
             * what the default relies on.
             */
            if (0 > usm_level || 2 <= usm_level || (1 == usm_level && NULL == devinfo->clMemFreeINTEL))
            {
              cl_device_svm_capabilities svmcaps = 0;
              cl_int query_result = EXIT_SUCCESS;
              if (0 == devinfo->nv) { /* vendor workaround */
                query_result = clGetDeviceInfo(active_id, CL_DEVICE_SVM_CAPABILITIES, sizeof(cl_device_svm_capabilities), &svmcaps, NULL);
                assert(EXIT_SUCCESS == query_result || 0 == svmcaps);
                if (EXIT_SUCCESS != query_result && 0 <= usm_level) result = query_result;
              }
              if (EXIT_SUCCESS == query_result && 3 != usm_level) { /* coarse-grain only */
                svmcaps &= CL_DEVICE_SVM_COARSE_GRAIN_BUFFER;
              }
              if (EXIT_SUCCESS == query_result) devinfo->usm = (cl_int)svmcaps;
            }
# endif
# if (0 != LIBXSTREAM_USM)
            /**
             * Only an unsatisfied explicit request warrants a warning: SVM is the
             * intended default, so reporting its synchronous transfers on every
             * run would be noise. Level 1 asked for the asynchronous path and did
             * not get it, which is worth stating because the symptom is merely a
             * slower run.
             */
            if (1 == usm_level && NULL == devinfo->clMemFreeINTEL &&
                (2 <= libxstream_opencl_config.verbosity || 0 > libxstream_opencl_config.verbosity))
            {
              fprintf(stderr, "WARN ACC/OpenCL: no Intel USM extensions, falling back to SVM.\n");
            }
# endif
          }
# if defined(LIBXSTREAM_CMDAGR)
          if (0 != devinfo->intel) { /* device vendor (above) can now be used */
            int result_cmdagr = EXIT_SUCCESS;
            const cl_command_queue q = LIBXSTREAM_CREATE_COMMAND_QUEUE(context, active_id, properties, &result_cmdagr);
            if (EXIT_SUCCESS == result_cmdagr) {
              assert(NULL != q);
              clReleaseCommandQueue(q);
            }
          }
# endif
          properties[1] = 0;
          if (EXIT_SUCCESS == result) {
            devinfo->stream.queue = LIBXSTREAM_CREATE_COMMAND_QUEUE(context, active_id, properties, &result);
          }
        }
        if (EXIT_SUCCESS == result) {
          if (NULL == devinfo->context || device_id != libxstream_opencl_config.device_id) {
            libxstream_opencl_config.device_id = device_id;
            devinfo->context = context;
          }
        }
        else memset(devinfo, 0, sizeof(*devinfo));
      }
      if (NULL != lock) LIBXS_LOCK_RELEASE(LIBXS_LOCK, lock);
    }
    else result = EXIT_FAILURE;
  }
  else result = EXIT_FAILURE;
  assert(EXIT_SUCCESS == result || NULL == devinfo->context);
  return result;
}


LIBXSTREAM_API int libxstream_device_set_active(int device_id)
{
  int result = EXIT_SUCCESS;
  if (0 <= device_id) {
    if (0 == libxstream_opencl_config.ndevices) { /* not initialized */
      result = libxstream_init();
    }
  }
  else result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result) {
    if (device_id < libxstream_opencl_config.ndevices) {
# if defined(LIBXSTREAM_CACHE_DID)
      if (internal_libxstream_opencl_active_id != (device_id + 1))
# endif
      {
        result = libxstream_opencl_set_active_device(libxstream_opencl_config.lock_main, device_id);
# if defined(LIBXSTREAM_CACHE_DID)
        if (EXIT_SUCCESS == result) internal_libxstream_opencl_active_id = device_id + 1;
# endif
      }
    }
    else result = EXIT_FAILURE;
  }
  CL_RETURN(result, "");
}


LIBXSTREAM_API int libxstream_opencl_flags_atomics(const libxstream_opencl_device_t* devinfo, libxstream_opencl_atomic_fp_t kind,
  const char* exts[], size_t* exts_maxlen, char flags[], size_t flags_maxlen)
{
  size_t ext1, ext2;
  int result = 0;
  for (ext1 = 0; ext1 < (NULL != exts_maxlen ? *exts_maxlen : 0); ++ext1) {
    if (NULL == exts[ext1] || '\0' == *exts[ext1]) break;
  }
  for (ext2 = ext1 + 1; ext2 < (NULL != exts_maxlen ? *exts_maxlen : 0); ++ext2) {
    if (NULL == exts[ext2] || '\0' == *exts[ext2]) break;
  }
  if (NULL != devinfo && NULL != exts_maxlen && ext2 < *exts_maxlen) {
    const cl_device_id device_id = libxstream_opencl_config.devices[libxstream_opencl_config.device_id];
    const char* atomic_type = "";
    switch (kind) {
      case libxstream_opencl_atomic_fp_64: {
        exts[ext1] = "cl_khr_fp64 cl_khr_int64_base_atomics cl_khr_int64_extended_atomics";
        if (2 <= *devinfo->std_level && EXIT_SUCCESS == libxstream_opencl_device_ext(device_id, exts, ext2)) {
          atomic_type = "-DTA=long -DTA2=atomic_long -DTF=atomic_double";
        }
        else {
          exts[ext1] = "cl_khr_fp64 cl_khr_int64_base_atomics";
          if (EXIT_SUCCESS == libxstream_opencl_device_ext(device_id, exts, ext2)) {
            atomic_type = "-DTA=long";
          }
          else { /* fallback */
            exts[ext1] = "cl_khr_fp64 cl_khr_global_int32_base_atomics cl_khr_global_int32_extended_atomics";
            if (2 <= *devinfo->std_level && EXIT_SUCCESS == libxstream_opencl_device_ext(device_id, exts, ext2)) {
              atomic_type = "-DATOMIC32_ADD64 -DTA=int -DTA2=atomic_int -DTF=atomic_double";
            }
            else {
              exts[ext1] = "cl_khr_fp64 cl_khr_global_int32_base_atomics";
              if (EXIT_SUCCESS == libxstream_opencl_device_ext(device_id, exts, ext2)) {
                atomic_type = "-DATOMIC32_ADD64 -DTA=int";
              }
              else kind = libxstream_opencl_atomic_fp_no;
            }
          }
        }
      } break;
      case libxstream_opencl_atomic_fp_32: {
        exts[ext1] = "cl_khr_global_int32_base_atomics cl_khr_global_int32_extended_atomics";
        if (2 <= *devinfo->std_level && EXIT_SUCCESS == libxstream_opencl_device_ext(device_id, exts, ext2)) {
          exts[ext2] = "cl_khr_int64_base_atomics cl_khr_int64_extended_atomics";
          atomic_type = "-DTA=int -DTA2=atomic_int -DTF=atomic_float";
        }
        else {
          exts[ext1] = "cl_khr_global_int32_base_atomics";
          if (EXIT_SUCCESS == libxstream_opencl_device_ext(device_id, exts, ext2)) {
            exts[ext2] = "cl_khr_int64_base_atomics";
            atomic_type = "-DTA=int";
          }
          else kind = libxstream_opencl_atomic_fp_no;
        }
      } break;
      default: assert(libxstream_opencl_atomic_fp_no == kind);
    }
    if (libxstream_opencl_atomic_fp_no != kind) {
      const char *barrier_expr = NULL, *atomic_exp = NULL, *atomic_ops = "";
      const char* const env_barrier = getenv("LIBXSTREAM_BARRIER");
      const char* const env_atomics = getenv("LIBXSTREAM_ATOMICS");
      if (NULL == env_barrier || '0' != *env_barrier) {
        barrier_expr = ((2 <= *devinfo->std_level && (0 == devinfo->intel || (CL_DEVICE_TYPE_CPU != devinfo->type)))
                          ? "-D\"BARRIER(A)=work_group_barrier(A,memory_scope_work_group)\""
                          : "-D\"BARRIER(A)=barrier(A)\"");
      }
      else barrier_expr = ""; /* no barrier */
      assert(NULL != barrier_expr);
      if (NULL == env_atomics || '0' != *env_atomics) {
        /* can signal/force atomics without confirmation */
        const int force_atomics = ((NULL == env_atomics || '\0' == *env_atomics) ? 0 : atoi(env_atomics));
        if (NULL == env_atomics || '\0' == *env_atomics || 0 != force_atomics) {
          cl_bitfield fp_atomics = 0;
          if (EXIT_SUCCESS == clGetDeviceInfo(device_id, (cl_device_info)(libxstream_opencl_atomic_fp_64 == kind ? 0x4232 : 0x4231),
                                sizeof(cl_bitfield), &fp_atomics, NULL) &&
              0 != (/*add*/ (1 << 1) & fp_atomics))
          {
            exts[ext2] = "cl_ext_float_atomics";
# if 1 /* enabling this permitted extension in source code causes compiler warning */
            *exts_maxlen = ext2; /* quietly report extension by reducing exts_maxlen */
# endif
            atomic_exp = (libxstream_opencl_atomic_fp_64 == kind ? "atomic_fetch_add_explicit((GLOBAL_VOLATILE(atomic_double)*)A,B,"
                                                                   "memory_order_relaxed,memory_scope_work_group)"
                                                                 : "atomic_fetch_add_explicit((GLOBAL_VOLATILE(atomic_float)*)A,B,"
                                                                   "memory_order_relaxed,memory_scope_work_group)");
          }
          else if (0 != force_atomics || (0 != devinfo->intel && ((0x4905 != devinfo->uid && 0 == devinfo->unified)))) {
            if ((((0 != force_atomics || (0 != devinfo->intel && ((0x0bd0 <= devinfo->uid && 0x0bdb >= devinfo->uid) ||
                                                                   libxstream_opencl_atomic_fp_32 == kind))))))
            {
              if (0 == force_atomics && (0 == devinfo->intel || 0x0bd0 > devinfo->uid || 0x0bdb < devinfo->uid)) {
                exts[ext2] = "cl_intel_global_float_atomics";
                atomic_ops = "-Dcl_intel_global_float_atomics";
              }
              else {
                atomic_ops = ((2 > *devinfo->std_level && 2 > force_atomics)
                                ? "-DATOMIC_PROTOTYPES=1"
                                : (3 > force_atomics ? "-DATOMIC_PROTOTYPES=2" : "-DATOMIC_PROTOTYPES=3"));
              }
              atomic_exp = ((2 > *devinfo->std_level && 2 > force_atomics) ? "atomic_add(A,B)"
                                                                           : "atomic_fetch_add_explicit((GLOBAL_VOLATILE(TF)*)A,B,"
                                                                             "memory_order_relaxed,memory_scope_work_group)");
            }
            else {
              atomic_exp = "atomic_add_global_cmpxchg(A,B)";
              atomic_ops = "-DCMPXCHG=atom_cmpxchg";
            }
          }
          else if (0 == devinfo->nv) {
            if (1 >= devinfo->amd) {
              atomic_ops = (libxstream_opencl_atomic_fp_32 == kind ? "-DCMPXCHG=atomic_cmpxchg" : "-DCMPXCHG=atom_cmpxchg");
              atomic_exp = "atomic_add_global_cmpxchg(A,B)";
              exts[ext2] = NULL;
            }
            else { /* GCN */
              atomic_exp = (libxstream_opencl_atomic_fp_64 == kind
                              ? "__builtin_amdgcn_global_atomic_fadd_f64(A,B,__ATOMIC_RELAXED,__OPENCL_MEMORY_SCOPE_WORK_GROUP)"
                              : "__builtin_amdgcn_global_atomic_fadd_f32(A,B,__ATOMIC_RELAXED,__OPENCL_MEMORY_SCOPE_WORK_GROUP)");
            }
          }
          else { /* xchg */
            assert(NULL != atomic_ops && '\0' == *atomic_ops);
            atomic_exp = "atomic_add_global_xchg(A,B)";
          }
        }
        else if (NULL != libxs_stristr(env_atomics, "cmpxchg")) {
          atomic_ops = (libxstream_opencl_atomic_fp_32 == kind ? "-DCMPXCHG=atomic_cmpxchg" : "-DCMPXCHG=atom_cmpxchg");
          atomic_exp = "atomic_add_global_cmpxchg(A,B)";
          exts[ext2] = NULL;
        }
        else { /* xchg */
          atomic_exp = "atomic_add_global_xchg(A,B)";
          atomic_ops = (libxstream_opencl_atomic_fp_32 == kind ? "-DXCHG=atomic_xchg" : "-DXCHG=atom_xchg");
        }
      }
      else { /* unsynchronized */
        atomic_exp = "*(A)+=(B)"; /* non-atomic update */
      }
      assert(NULL != atomic_exp);
      /* compose build parameters and flags */
      result = LIBXS_SNPRINTF(flags, flags_maxlen, " -DTAN=%i %s %s -D\"ATOMIC_ADD_GLOBAL(A,B)=%s\" %s", kind, atomic_type,
        atomic_ops, atomic_exp, barrier_expr);
    }
  }
  return result;
}


LIBXSTREAM_API int libxstream_opencl_defines(const char defines[], char buffer[], size_t buffer_size, int cleanup)
{
  const libxstream_opencl_device_t* const devinfo = &libxstream_opencl_config.device;
  int result = 0;
  if (NULL != buffer && NULL != devinfo->context) {
    const int std_clevel = 100 * devinfo->std_clevel[0] + 10 * devinfo->std_clevel[1];
    const int std_level = 100 * devinfo->std_level[0] + 10 * devinfo->std_level[1];
    result = LIBXS_SNPRINTF(buffer, buffer_size, " -DLIBXSTREAM_OCLVER=%u -DLIBXSTREAM_OCLVER_C=%u%s", std_level, std_clevel,
      0 == libxstream_opencl_config.debug ? " -DNDEBUG" : "");
    if (0 < result && LIBXS_CAST_INT(buffer_size) > result) {
      const int n = LIBXS_SNPRINTF(
        buffer + result, buffer_size - result, ' ' != buffer[result - 1] ? " %s" : "%s", NULL != defines ? defines : "");
      if (0 <= n) {
        if (LIBXS_CAST_INT(buffer_size) > (result += n) && 0 != cleanup) {
          char* replace = strpbrk(buffer + result - n, "\""); /* more portable (system/cpp needs quotes to protect braces) */
          for (; NULL != replace; replace = strpbrk(replace + 1, "\"")) *replace = ' ';
        }
      }
      else result = -1;
    }
  }
  else result = -1;
  return result;
}


LIBXSTREAM_API int libxstream_opencl_kernel_flags(const char build_params[], const char build_options[], const char try_options[],
  cl_program program, char buffer[], size_t buffer_size)
{
  const libxstream_opencl_device_t* const devinfo = &libxstream_opencl_config.device;
  int result = EXIT_SUCCESS, nchar = 0;
  assert(NULL != program && (NULL != buffer || 0 == buffer_size));
  nchar = libxstream_opencl_defines(build_params, buffer, buffer_size, 1 /*cleanup*/);
  if (0 <= nchar && LIBXS_CAST_INT(buffer_size) > nchar) {
    const int debug = (0 != libxstream_opencl_config.debug && 0 != devinfo->intel && CL_DEVICE_TYPE_CPU != devinfo->type);
    int n = LIBXS_SNPRINTF(buffer + nchar, buffer_size - nchar, " %s%s %s%s", 0 == debug ? "" : "-gline-tables-only ",
      devinfo->std_flag, NULL != build_options ? build_options : "", 0 != devinfo->biggrf ? " -cl-intel-256-GRF-per-thread" : "");
    if (0 <= n) {
      nchar += n;
      if (NULL != try_options && '\0' != *try_options) { /* length is not reported in result */
        n = LIBXS_SNPRINTF(buffer + nchar, buffer_size - nchar, " %s", try_options);
        if (0 > n || LIBXS_CAST_INT(buffer_size) <= (nchar + n)) buffer[nchar] = '\0';
      }
    }
    else nchar = n;
  }
  if (0 <= nchar && LIBXS_CAST_INT(buffer_size) > nchar) { /* check if internal flags apply */
    const cl_device_id device_id = libxstream_opencl_config.devices[libxstream_opencl_config.device_id];
    result = clBuildProgram(program, 1 /*num_devices*/, &device_id, buffer, NULL /*callback*/, NULL /*user_data*/);
    if (EXIT_SUCCESS != result) { /* failed to apply internal flags */
      LIBXS_EXPECT_DEBUG(EXIT_SUCCESS == clReleaseProgram(program)); /* avoid unclean state */
      buffer[nchar] = '\0'; /* remove internal flags */
    }
  }
  else result = EXIT_FAILURE;
  return result;
}


LIBXSTREAM_API int libxstream_opencl_program(size_t source_kind, const char source[], const char name[], const char build_params[],
  const char build_options[], const char try_options[], int* try_ok, const char* const extnames[], size_t num_exts,
  cl_program* program)
{
  char buffer[LIBXSTREAM_BUFFERSIZE] = "", buffer_name[LIBXSTREAM_MAXSTRLEN * 2];
  const cl_device_id device_id = libxstream_opencl_config.devices[libxstream_opencl_config.device_id];
  const libxstream_opencl_device_t* const devinfo = &libxstream_opencl_config.device;
  int result = ((NULL != source && NULL != name && '\0' != *name) ? EXIT_SUCCESS : EXIT_FAILURE);
  int ok = EXIT_SUCCESS, source_is_cl = (2 > source_kind), nchar = 0;
  size_t size_src = 0, size = 0;
  FILE* file_src = NULL;
  assert(NULL != devinfo->context);
  assert(NULL != program);
  *program = NULL;
  if (EXIT_SUCCESS == result && (1 == source_kind)) file_src = fopen(source, "rb");
  if (NULL != file_src) {
    if (EXIT_SUCCESS == result) {
      const char* const file_ext = strrchr(source, '.');
      char* src = NULL;
      source_is_cl = ((NULL != file_ext && NULL != libxs_stristr(file_ext + 1, "cl")) ? 1 : 0);
      size_src = (EXIT_SUCCESS == fseek(file_src, 0 /*offset*/, SEEK_END) ? ftell(file_src) : 0);
      src = (char*)((0 != size_src && EXIT_SUCCESS == fseek(file_src, 0 /*offset*/, SEEK_SET))
              ? libxs_malloc(NULL, size_src + source_is_cl /*terminator?*/, 0 /*auto-align*/)
              : NULL);
      if (NULL != src) {
        if (size_src == fread(src, 1 /*sizeof(char)*/, size_src /*count*/, file_src)) {
          if (0 != source_is_cl) src[size_src] = '\0'; /* terminator */
          source = src;
        }
        else {
          result = EXIT_FAILURE;
          libxs_free(src);
        }
      }
      else result = EXIT_FAILURE;
    }
    fclose(file_src);
  }
  else size_src = source_kind;
  if (EXIT_SUCCESS == result && 0 != source_is_cl) {
    const char* ext_source = source;
    size_src = strlen(ext_source);
    if (NULL != extnames) {
      int n = num_exts, nflat = 0;
      size_t size_ext = 0;
      for (; 0 < n; --n) {
        if (NULL != extnames[n - 1]) {
          const char* const end = buffer + strlen(extnames[n - 1]); /* before strtok */
          char* ext = strtok(strncpy(buffer, extnames[n - 1], LIBXSTREAM_BUFFERSIZE - 1), LIBXS_DELIMS " \t");
          for (; NULL != ext; ext = ((ext + 1) < end ? strtok((ext + 1) + strlen(ext), LIBXS_DELIMS " \t") : NULL), ++nflat) {
            size_ext += strlen(ext);
          }
        }
      }
      if (0 < size_ext && 0 < nflat) {
        const char* const enable_ext = "#pragma OPENCL EXTENSION %s : enable\n";
        const size_t size_src_ext = size_src + size_ext + nflat * (strlen(enable_ext) - 2 /*%s*/);
        char* const ext_source_buffer = (char*)libxs_malloc(
          NULL, size_src_ext + 1 /*terminator*/, 0 /*auto-align*/);
        if (NULL != ext_source_buffer) {
          for (n = 0; 0 < num_exts; --num_exts) {
            if (NULL != extnames[num_exts - 1]) {
              const char* const end = buffer_name + strlen(extnames[num_exts - 1]); /* before strtok */
              char* ext;
              strncpy(buffer_name, extnames[num_exts - 1], LIBXSTREAM_MAXSTRLEN * 2 - 1);
              buffer_name[LIBXSTREAM_MAXSTRLEN * 2 - 1] = '\0';
              ext = strtok(buffer_name, LIBXS_DELIMS " \t");
              for (; NULL != ext; ext = ((ext + 1) < end ? strtok((ext + 1) + strlen(ext), LIBXS_DELIMS " \t") : NULL)) {
                const char* line = source;
                for (;;) {
                  if (2 != sscanf(line, "#pragma OPENCL EXTENSION %[^: ]%*[: ]%[^\n]", buffer, buffer + LIBXSTREAM_BUFFERSIZE / 2))
                  {
                    line = NULL;
                    break;
                  }
                  else if (0 == strncmp(buffer, ext, LIBXSTREAM_BUFFERSIZE / 2) &&
                           0 == strncmp(buffer + LIBXSTREAM_BUFFERSIZE / 2, "enable", LIBXSTREAM_BUFFERSIZE / 2))
                  {
                    break;
                  }
                  line = strchr(line, '\n');
                  if (NULL != line) {
                    ++line;
                  }
                  else break;
                }
# if !defined(NDEBUG)
                if (EXIT_SUCCESS == libxstream_opencl_device_ext(device_id, (const char* const*)&ext, 1))
# endif
                { /* NDEBUG: assume given extension is supported (confirmed upfront) */
                  if (NULL == line) { /* extension is not already part of source */
                    n += LIBXS_SNPRINTF(
                      ext_source_buffer + n, size_src_ext + 1 /*terminator*/ - n, "#pragma OPENCL EXTENSION %s : enable\n", ext);
                  }
                }
# if !defined(NDEBUG)
                else if (0 != strcmp("cl_intel_global_float_atomics", ext)) {
                  fprintf(stderr, "WARN ACC/OpenCL: extension \"%s\" is not supported.\n", ext);
                }
# endif
              }
            }
          }
          memcpy(ext_source_buffer + n, source, size_src);
          size_src += n; /* according to given/permitted extensions */
          assert(size_src <= size_src_ext);
          ext_source_buffer[size_src] = '\0';
          ext_source = ext_source_buffer;
        }
      }
      buffer[0] = '\0'; /* reset to empty */
    }
    /* cpp: consider to preprocess kernel (failure does not impact result code) */
    if (0 != libxstream_opencl_config.dump && NULL == file_src) {
      char dump_filename[LIBXSTREAM_MAXSTRLEN];
      nchar = LIBXS_SNPRINTF(dump_filename, sizeof(dump_filename), "%s.cl", name);
      if (0 < nchar && (int)sizeof(dump_filename) > nchar) {
        const int std_flag_len = LIBXS_CAST_INT(strlen(devinfo->std_flag));
        const char* const env_cpp = getenv("LIBXSTREAM_CPP");
        const int cpp = (NULL == env_cpp ? (3 <= libxstream_opencl_config.dump) : atoi(env_cpp));
# if defined(LIBXSTREAM_CPPBIN)
        FILE* const file_cpp = (0 != cpp ? fopen(LIBXSTREAM_CPPBIN, "rb") : NULL);
# else
        FILE* const file_cpp = NULL;
# endif
        int file_dmp = -1;
        if (NULL != file_cpp) {
          nchar = LIBXS_SNPRINTF(buffer_name, sizeof(buffer_name), LIBXSTREAM_TEMPDIR "/.%s.XXXXXX", name);
          if (0 < nchar && (int)sizeof(buffer_name) > nchar) file_dmp = mkstemp(buffer_name);
          fclose(file_cpp); /* existence-check */
        }
        else file_dmp = open(dump_filename, O_CREAT | O_TRUNC | O_RDWR, S_IREAD | S_IWRITE);
        if (0 <= file_dmp) {
          if ((0 != std_flag_len &&
                (3 != write(file_dmp, "/*\n", 3) || std_flag_len != write(file_dmp, devinfo->std_flag, std_flag_len) ||
                  4 != write(file_dmp, "\n*/\n", 4))) ||
              size_src != (size_t)write(file_dmp, ext_source, size_src))
          {
            file_dmp = -1;
          }
          LIBXS_EXPECT_DEBUG(EXIT_SUCCESS == close(file_dmp));
        }
# if defined(LIBXSTREAM_CPPBIN)
        if (NULL != file_cpp && 0 <= file_dmp) { /* preprocess source-code */
          const char* sed_pattern = "";
#   if defined(LIBXSTREAM_SEDBIN)
          FILE* const file_sed = fopen(LIBXSTREAM_SEDBIN, "rb");
          if (NULL != file_sed) {
            sed_pattern = "| " LIBXSTREAM_SEDBIN " '/^[[:space:]]*\\(\\/\\/.*\\)*$/d'";
            fclose(file_sed); /* existence-check */
          }
#   endif
          nchar = LIBXS_SNPRINTF(
            buffer, LIBXSTREAM_BUFFERSIZE, LIBXSTREAM_CPPBIN " -P -C -nostdinc %s", 0 == devinfo->nv ? "" : "-D__NV_CL_C_VERSION ");
          if (0 < nchar && LIBXSTREAM_BUFFERSIZE > nchar) {
            int n = libxstream_opencl_defines(build_params, buffer + nchar, LIBXSTREAM_BUFFERSIZE - nchar, 0 /*cleanup*/);
            if (0 <= n && LIBXSTREAM_BUFFERSIZE > (nchar += n)) {
              n = LIBXS_SNPRINTF(buffer + nchar, LIBXSTREAM_BUFFERSIZE - nchar,
                ' ' != buffer[nchar - 1] ? " %s %s >%s" : "%s %s >%s", buffer_name, sed_pattern, dump_filename);
            }
            nchar = (0 <= n ? nchar : 0) + n;
          }
          if (0 < nchar && LIBXSTREAM_BUFFERSIZE > nchar && EXIT_SUCCESS == system(buffer)) {
            FILE* const file = fopen(dump_filename, "r");
            if (NULL != file) {
              const long int size_file = (EXIT_SUCCESS == fseek(file, 0 /*offset*/, SEEK_END) ? ftell(file) : 0);
              char* const src = (char*)(EXIT_SUCCESS == fseek(file, 0 /*offset*/, SEEK_SET)
                                          ? libxs_malloc(
                                              NULL, size_file + 1 /*terminator*/, 0 /*auto-align*/)
                                          : NULL);
              if (NULL != src) {
                if ((size_t)size_file == fread(src, 1 /*sizeof(char)*/, size_file /*count*/, file)) {
                  if (source != ext_source) {
                    void* p = NULL;
                    LIBXS_ASSIGN(&p, &ext_source);
                    libxs_free(p);
                  }
                  src[size_file] = '\0';
                  ext_source = src;
                }
                else libxs_free(src);
              }
              LIBXS_EXPECT_DEBUG(EXIT_SUCCESS == fclose(file));
            }
          }
          LIBXS_EXPECT_DEBUG(EXIT_SUCCESS == unlink(buffer_name)); /* remove temporary file */
          buffer[0] = '\0'; /* reset to empty */
        }
# endif
      }
    }
    *program = clCreateProgramWithSource(devinfo->context, 1 /*nlines*/, &ext_source, NULL, &result);
    assert(EXIT_SUCCESS != result || NULL != *program);
    if (EXIT_SUCCESS == result) {
      ok = libxstream_opencl_kernel_flags(build_params, build_options, try_options, *program, buffer, LIBXSTREAM_BUFFERSIZE);
      if (EXIT_SUCCESS != ok) {
        *program = clCreateProgramWithSource(devinfo->context, 1 /*nlines*/, &ext_source, NULL, &result);
        assert(EXIT_SUCCESS != result || NULL != *program);
        if (EXIT_SUCCESS == result) {
          result = clBuildProgram(*program, 1 /*num_devices*/, &device_id, buffer, NULL /*callback*/, NULL /*user_data*/);
        }
      }
      else if (0 != libxstream_opencl_config.verbosity && 0 != libxstream_opencl_config.debug) {
        fprintf(stderr, "INFO ACC/OpenCL [%s]: %s\n", name, buffer);
      }
    }
    if (EXIT_SUCCESS == result) {
      if (source != ext_source) {
        void* p = NULL;
        LIBXS_ASSIGN(&p, &ext_source);
        libxs_free(p);
      }
      buffer[0] = '\0'; /* reset to empty */
      if (EXIT_SUCCESS == result && NULL == file_src && (2 <= libxstream_opencl_config.dump || 0 > libxstream_opencl_config.dump)) {
        unsigned char* binary = NULL;
        binary = (unsigned char*)(EXIT_SUCCESS == clGetProgramInfo(*program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &size, NULL)
                                    ? libxs_malloc(NULL, size, 0 /*auto-align*/)
                                    : NULL);
        if (NULL != binary) {
          result = clGetProgramInfo(*program, CL_PROGRAM_BINARIES, sizeof(unsigned char*), &binary, NULL);
          if (EXIT_SUCCESS == result) { /* successfully queried program binary */
            FILE* file;
            nchar = LIBXS_SNPRINTF(buffer, LIBXSTREAM_BUFFERSIZE, "%s.dump", name);
            file = ((0 < nchar && LIBXSTREAM_BUFFERSIZE > nchar) ? fopen(buffer, "wb") : NULL);
            buffer[0] = '\0'; /* reset to empty */
            if (NULL != file) {
              if (size != fwrite(binary, 1, size, file)) result = EXIT_FAILURE;
              fclose(file);
            }
            else result = EXIT_FAILURE;
          }
          libxs_free(binary);
        }
        else result = EXIT_FAILURE;
      }
    }
    else if (source != ext_source) { /* error: creating program */
      void* p = NULL;
      LIBXS_ASSIGN(&p, &ext_source);
      libxs_free(p);
    }
  }
  else if (EXIT_SUCCESS == result) { /* binary representation */
    assert(1 < size_src || 0 == size_src);
# if defined(CL_VERSION_2_1)
    if (0 != libxstream_opencl_config.dump) *program = clCreateProgramWithIL(devinfo->context, source, size_src, &result);
    else
# endif
    {
      *program = clCreateProgramWithBinary(
        devinfo->context, 1, &device_id, &size_src, (const unsigned char**)&source, NULL /*binary_status*/, &result);
    }
    assert(EXIT_SUCCESS != result || NULL != *program);
    if (EXIT_SUCCESS == result) {
      ok = libxstream_opencl_kernel_flags(build_params, build_options, try_options, *program, buffer, LIBXSTREAM_BUFFERSIZE);
      if (EXIT_SUCCESS != ok) {
# if defined(CL_VERSION_2_1)
        if (0 != libxstream_opencl_config.dump) *program = clCreateProgramWithIL(devinfo->context, source, size_src, &result);
        else
# endif
        {
          *program = clCreateProgramWithBinary(
            devinfo->context, 1, &device_id, &size_src, (const unsigned char**)&source, NULL /*binary_status*/, &result);
        }
        assert(EXIT_SUCCESS != result || NULL != *program);
        if (EXIT_SUCCESS == result) {
          result = clBuildProgram(*program, 1 /*num_devices*/, &device_id, buffer, NULL /*callback*/, NULL /*user_data*/);
        }
      }
      else if (0 != libxstream_opencl_config.verbosity && 0 != libxstream_opencl_config.debug) {
        fprintf(stderr, "INFO ACC/OpenCL [%s]: %s\n", name, buffer);
      }
    }
  }
  if (NULL != file_src) {
    void* p = NULL;
    LIBXS_ASSIGN(&p, (const void**)&source);
    assert(1 == source_kind);
    libxs_free(p);
  }
  if (NULL != *program) {
    if (2 <= libxstream_opencl_config.verbosity || 0 > libxstream_opencl_config.verbosity) {
      if (EXIT_SUCCESS == clGetProgramBuildInfo(*program, device_id, CL_PROGRAM_BUILD_LOG, LIBXSTREAM_BUFFERSIZE, buffer, &size)) {
        const char* info = buffer;
        while ('\0' != *info && NULL != strchr("\n\r\t ", *info)) ++info; /* remove preceding newline etc. */
        assert(NULL != name && '\0' != *name);
        if ('\0' != *info) fprintf(stderr, "INFO ACC/OpenCL: %s -> %s\n", name, info);
      }
      else buffer[0] = '\0'; /* reset to empty */
    }
    if (EXIT_SUCCESS != result) {
      LIBXS_EXPECT_DEBUG(EXIT_SUCCESS == clReleaseProgram(*program));
      *program = NULL;
    }
  }
  if (NULL != try_ok) *try_ok = result | ok;
  CL_RETURN(result, buffer);
}


LIBXSTREAM_API int libxstream_opencl_kernel_query(cl_program program, const char kernel_name[], cl_kernel* kernel)
{
  int result;
  assert(NULL != kernel);
  *kernel = NULL;
  if (NULL != program && NULL != kernel_name && '\0' != *kernel_name) {
    *kernel = clCreateKernel(program, kernel_name, &result);
# if defined(CL_VERSION_1_2)
    if (EXIT_SUCCESS != result) { /* discover available kernels in program, and adopt the last kernel listed */
      char kbuf[LIBXSTREAM_BUFFERSIZE];
      if (EXIT_SUCCESS == clGetProgramInfo(program, CL_PROGRAM_KERNEL_NAMES, sizeof(kbuf), kbuf, NULL) && '\0' != *kbuf) {
        const char *const semicolon = strrchr(kbuf, ';'), *const kname = (NULL == semicolon ? kbuf : (semicolon + 1));
        *kernel = clCreateKernel(program, kname, &result);
      }
    }
# endif
    assert(EXIT_SUCCESS != result || NULL != *kernel);
  }
  else result = EXIT_FAILURE;
  return result;
}


LIBXSTREAM_API int libxstream_opencl_kernel(size_t source_kind, const char source[], const char kernel_name[],
  const char build_params[], const char build_options[], const char try_options[], int* try_ok, const char* const extnames[],
  size_t num_exts, cl_kernel* kernel)
{
  cl_program program = NULL;
  int result;
  assert(NULL != kernel);
  *kernel = NULL;
  result = libxstream_opencl_program(
    source_kind, source, kernel_name, build_params, build_options, try_options, try_ok, extnames, num_exts, &program);
  if (EXIT_SUCCESS == result) {
    result = libxstream_opencl_kernel_query(program, kernel_name, kernel);
  }
  if (NULL != program) {
    if (EXIT_SUCCESS != result && NULL != *kernel) {
      LIBXS_EXPECT_DEBUG(EXIT_SUCCESS == clReleaseKernel(*kernel));
      *kernel = NULL;
    }
    LIBXS_EXPECT_DEBUG(EXIT_SUCCESS == clReleaseProgram(program));
  }
  return result;
}


/**
 * Sub-buffer covering info->memory from offset onward, recorded so that the
 * parent's deallocation releases it.
 *
 * Releasing right after clSetKernelArg is what this replaces: that call does not
 * retain the cl_mem, so dropping the last reference left the pending launch with
 * a freed handle and the Intel driver aborted inside clEnqueueNDRangeKernel (no
 * error code, SIGABRT from within the driver).
 *
 * A fresh sub-buffer per argument rather than one cached per offset, even though
 * the offset determines the region: reusing one handle across launches computed
 * wrong results (Ozaki-2 panelled GEMM, rsq 0.71 against 1.0 for fresh handles),
 * while the region, size and parent of the reused handle all verified correct.
 * The reason is not established -- suspected aliasing of a region written by one
 * kernel and read by the next through the same sub-buffer -- so correctness wins
 * over economy here. Consequently the list grows per launch, and the release at
 * deallocation is what bounds it: buffers whose lifetime spans very many
 * launches will hold correspondingly many handles.
 *
 * The caller must hold lock_memory: the list is shared with the deallocator.
 */
LIBXSTREAM_API_INTERN cl_mem libxstream_opencl_subbuffer(cl_mem /*parent*/, size_t /*offset*/, int* /*result*/);
LIBXSTREAM_API_INTERN cl_mem libxstream_opencl_subbuffer(cl_mem parent, size_t offset, int* result)
{
  cl_mem sub = NULL;
  size_t total = 0;
  assert(NULL != parent && NULL != result && 0 != offset);
  *result = clGetMemObjectInfo(parent, CL_MEM_SIZE, sizeof(size_t), &total, NULL);
  if (EXIT_SUCCESS == *result && offset < total) {
    cl_buffer_region region;
    region.origin = offset;
    region.size = total - offset;
    sub = clCreateSubBuffer(parent, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, result);
    if (EXIT_SUCCESS == *result && NULL != sub) {
      size_t i = 0;
      for (; i < libxstream_opencl_config.nsubs; ++i) { /* reuse a free entry */
        if (NULL == libxstream_opencl_config.subs[i].parent) break;
      }
      if (i == libxstream_opencl_config.nsubs) { /* grow */
        const size_t n = (0 != libxstream_opencl_config.nsubs ? (2 * libxstream_opencl_config.nsubs) : 32);
        libxstream_opencl_info_sub_t* const subs = (libxstream_opencl_info_sub_t*)realloc(
          libxstream_opencl_config.subs, sizeof(libxstream_opencl_info_sub_t) * n);
        if (NULL != subs) {
          memset(subs + libxstream_opencl_config.nsubs, 0,
            sizeof(libxstream_opencl_info_sub_t) * (n - libxstream_opencl_config.nsubs));
          libxstream_opencl_config.subs = subs;
          libxstream_opencl_config.nsubs = n;
        }
      }
      if (i < libxstream_opencl_config.nsubs) {
        libxstream_opencl_config.subs[i].memory = sub;
        libxstream_opencl_config.subs[i].parent = parent;
      }
      else { /* cannot track it, hence cannot keep it alive */
        LIBXS_EXPECT_DEBUG(EXIT_SUCCESS == clReleaseMemObject(sub));
        sub = NULL;
        *result = EXIT_FAILURE;
      }
    }
    else sub = NULL;
  }
  else if (EXIT_SUCCESS == *result) *result = EXIT_FAILURE;
  return sub;
}


LIBXSTREAM_API int libxstream_opencl_set_kernel_ptr(cl_kernel kernel, cl_uint arg_index, const void* arg_value)
{
  const libxstream_opencl_device_t* const devinfo = &libxstream_opencl_config.device;
  int result = EXIT_FAILURE;
  assert(NULL != devinfo->context);
# if (1 >= LIBXSTREAM_USM)
  if (NULL != devinfo->clSetKernelArgMemPointerINTEL) {
    result = devinfo->clSetKernelArgMemPointerINTEL(kernel, arg_index, arg_value);
  }
  else
# endif
# if (0 != LIBXSTREAM_USM)
    if (0 != devinfo->usm)
  {
    result = clSetKernelArgSVMPointer(kernel, arg_index, arg_value);
  }
  else
# elif defined(NDEBUG)
  LIBXS_UNUSED(devinfo);
# endif
  {
    size_t offset = 0;
    void* nc;
    libxstream_opencl_info_memptr_t* info;
    LIBXS_UNION_ASSIGN(void*, nc, const void*, arg_value);
    /**
     * The lock spans the sub-buffer table as well, not just the lookup: the
     * table is state shared with the deallocator, so releasing earlier would let
     * a concurrent free tear it down mid-use.
     */
    LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, libxstream_opencl_config.lock_memory);
    info = libxstream_opencl_info_devptr_modify(NULL, nc, 1 /*elsize*/, NULL /*amount*/, &offset);
    if (NULL != info) {
      if (0 == offset) {
        result = clSetKernelArg(kernel, arg_index, sizeof(cl_mem), &info->memory);
      }
      /**
       * A non-zero offset needs a sub-buffer, because clSetKernelArg takes a
       * cl_mem and cannot express one. Off by default: the kernel is the better
       * place to apply the offset, so a caller that can pass it as a separate
       * index argument should do that instead. Opting in costs a driver object
       * per distinct offset whose lifetime must outlive the launch, and releasing
       * such an object was observed to fault inside the NVIDIA driver even for a
       * handle that clRetainMemObject and clGetMemObjectInfo both accept.
       */
      else if (0 != libxstream_opencl_config.subbuffer) {
        cl_mem sub = libxstream_opencl_subbuffer(info->memory, offset, &result);
        if (EXIT_SUCCESS == result) {
          /**
           * No release here: clSetKernelArg does not retain the cl_mem, so
           * dropping the last reference would enqueue the kernel with a dangling
           * handle. The table owns it until the parent buffer is deallocated.
           */
          result = clSetKernelArg(kernel, arg_index, sizeof(cl_mem), &sub);
        }
      }
      else { /* the caller must offset inside the kernel, or opt into sub-buffers */
        if (0 != libxstream_opencl_config.verbosity) {
          fprintf(stderr, "ERROR ACC/OpenCL: offset kernel-argument requires LIBXSTREAM_SUBBUFFER=1"
                          " or libxstream_init_config_t::subbuffer=1.\n");
        }
        result = EXIT_FAILURE;
      }
    }
    LIBXS_LOCK_RELEASE(LIBXS_LOCK, libxstream_opencl_config.lock_memory);
  }
  CL_RETURN(result, "");
}


/** Return a per-launch record to the pool. */
LIBXSTREAM_API_INTERN void libxstream_launch_info_free(libxstream_opencl_launch_info_t* info);
LIBXSTREAM_API_INTERN void libxstream_launch_info_free(libxstream_opencl_launch_info_t* info)
{
  if (NULL != info) {
    libxs_pfree_lock(info, (void**)libxstream_opencl_config.launch_infos, &libxstream_opencl_config.nlaunch_infos,
      libxstream_opencl_config.lock_event);
  }
}


/**
 * Record a completed kernel launch as {ms, gflop, mb}. The work amounts come from
 * the per-launch record rather than from the table, so two launches of the same
 * kernel in flight at once cannot overwrite each other's counts. The callback
 * resolves no names and takes no lock beyond the histogram's and the pool's.
 */
LIBXSTREAM_API_INTERN void CL_CALLBACK libxstream_kernel_notify(cl_event /*event*/, cl_int /*event_status*/, void* /*data*/);
LIBXSTREAM_API_INTERN void CL_CALLBACK libxstream_kernel_notify(cl_event event, cl_int event_status, void* data)
{
  libxstream_opencl_launch_info_t* const info = (libxstream_opencl_launch_info_t*)data;
  cl_command_type type = 0;
  int result = EXIT_SUCCESS;
  double vals[3];
  vals[0] = libxstream_opencl_duration(event, &result) * 1E3; /* Milliseconds */
  LIBXS_UNUSED(event_status);
  assert(CL_COMPLETE == event_status && NULL != info);
  if (EXIT_SUCCESS == result && NULL != info
    && EXIT_SUCCESS == clGetEventInfo(event, CL_EVENT_COMMAND_TYPE, sizeof(type), &type, NULL)
    && CL_COMMAND_NDRANGE_KERNEL == type)
  {
    const size_t i = info->slot;
    libxs_hist_t* const hist = (i < libxstream_opencl_config.nkernels ? libxstream_opencl_config.hist_kernel[i] : NULL);
    if (NULL != hist) {
      /* same floor as the transfer path: a duration spanning too few ticks of the
         device timer is quantization noise rather than a measurement */
      const double floor_ms = 1E-6 * (double)(LIBXSTREAM_PROFILE_TICKS * libxstream_opencl_config.device.timer_ns);
      vals[1] = info->gflop;
      vals[2] = info->mb;
      if (vals[0] >= floor_ms) {
        libxs_hist_push(libxstream_opencl_config.lock_event, hist, vals);
        LIBXS_ATOMIC_ADD_FETCH(&libxstream_opencl_config.nprofile, 1, LIBXS_ATOMIC_RELAXED);
        if (0 > libxstream_opencl_config.profile) {
          fprintf(stderr, "PROF ACC/OpenCL: %s ms=%.3f\n", libxstream_opencl_config.name_kernel[i], vals[0]);
        }
      }
      else {
        LIBXS_ATOMIC_ADD_FETCH(&libxstream_opencl_config.nprofile_short, 1, LIBXS_ATOMIC_RELAXED);
        if (0 > libxstream_opencl_config.profile) {
          fprintf(stderr, "PROF ACC/OpenCL: %s ms=%.3f (below %.3f ms, discarded)\n",
            libxstream_opencl_config.name_kernel[i], vals[0], floor_ms);
        }
      }
    }
  }
  libxstream_launch_info_free(info);
  if (NULL != event) LIBXS_EXPECT_DEBUG(EXIT_SUCCESS == clReleaseEvent(event));
}


/**
 * Map a kernel handle to its hist_kernel slot, creating the entry on first sight.
 * The name comes from the handle (CL_KERNEL_FUNCTION_NAME) rather than from the
 * caller, so a launch site needs no identifier argument. Returns the number of
 * slots when the table is full or the kernel cannot be identified, which the
 * caller treats as "do not profile this launch".
 */
LIBXSTREAM_API_INTERN size_t libxstream_kernel_slot(cl_kernel kernel);
LIBXSTREAM_API_INTERN size_t libxstream_kernel_slot(cl_kernel kernel)
{
  size_t result = LIBXSTREAM_MAXNKERNELS, i;
  LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, libxstream_opencl_config.lock_event);
  for (i = 0; i < libxstream_opencl_config.nkernels; ++i) {
    if (kernel == libxstream_opencl_config.kernels[i]) {
      result = i;
      break;
    }
  }
  if (LIBXSTREAM_MAXNKERNELS == result) {
    i = libxstream_opencl_config.nkernels;
    if (i < LIBXSTREAM_MAXNKERNELS) {
      char* const name = (char*)malloc(LIBXSTREAM_MAXSTRLEN);
      if (NULL != name) {
        if (EXIT_SUCCESS == clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, LIBXSTREAM_MAXSTRLEN, name, NULL)) {
          const int nbuckets = LIBXS_MAX(LIBXS_ABS(libxstream_opencl_config.profile), 2) + 1;
          /**
           * All three averaged, not accumulated: libxs_hist_query_percentile
           * reports vals[0] as an interpolated per-sample duration, so the work
           * amounts must be per-sample too. Summing them would divide a bucket
           * total by a single-sample time -- the same numerator/denominator
           * mismatch that let the old facility report more work in less time.
           */
          const libxs_hist_update_t update[] = {libxs_hist_update_avg, libxs_hist_update_avg, libxs_hist_update_avg};
          libxs_hist_t* const hist = libxs_hist_create(nbuckets, 3, update);
          if (NULL != hist) {
            libxstream_opencl_config.hist_kernel[i] = hist;
            libxstream_opencl_config.name_kernel[i] = name;
            libxstream_opencl_config.kernels[i] = kernel;
            /* publish the entry only once it is complete: the callback reads it
               without the lock, bounded by nkernels */
            LIBXS_ATOMIC_ADD_FETCH(&libxstream_opencl_config.nkernels, 1, LIBXS_ATOMIC_SEQ_CST);
            result = i;
          }
          else free(name);
        }
        else free(name);
      }
    }
    else LIBXS_ATOMIC_ADD_FETCH(&libxstream_opencl_config.nprofile_kernel_lost, 1, LIBXS_ATOMIC_RELAXED);
  }
  LIBXS_LOCK_RELEASE(LIBXS_LOCK, libxstream_opencl_config.lock_event);
  return result;
}


LIBXSTREAM_API int libxstream_opencl_launch_work(libxstream_stream_t* stream, cl_kernel kernel, cl_uint work_dim,
  const size_t* global_work_offset, const size_t* global_work_size, const size_t* local_work_size,
  cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event, size_t nflops, size_t nbytes)
{
  libxstream_opencl_stream_t* const str = (libxstream_opencl_stream_t*)stream;
  int result = EXIT_SUCCESS;
  if (NULL != str && NULL != kernel) {
    /**
     * Profile only if requested. The record is taken from the pool before the
     * launch and returned by the callback, so an exhausted pool skips profiling
     * rather than the launch. The pool bounds launches *in flight*, not total
     * launches: a record lives only until its completion callback runs.
     */
    libxstream_opencl_launch_info_t* info = NULL;
    cl_event evt = NULL;
    if (0 != libxstream_opencl_config.profile && NULL != libxstream_opencl_config.launch_infos) {
      const size_t slot = libxstream_kernel_slot(kernel);
      if (slot < LIBXSTREAM_MAXNKERNELS) {
        info = (libxstream_opencl_launch_info_t*)libxs_pmalloc_lock(
          (void**)libxstream_opencl_config.launch_infos, &libxstream_opencl_config.nlaunch_infos,
          libxstream_opencl_config.lock_event);
        if (NULL != info) {
          info->slot = slot;
          info->gflop = 1E-9 * (double)nflops;
          info->mb = 1E-6 * (double)nbytes;
        }
      }
    }
    result = clEnqueueNDRangeKernel(str->queue, kernel, work_dim, global_work_offset, global_work_size, local_work_size,
      num_events_in_wait_list, event_wait_list, (NULL != info || NULL != event) ? &evt : NULL);
    if (EXIT_SUCCESS == result && NULL != info) {
      /* retain when the caller keeps the event: the callback releases its own
         reference, and the caller releases theirs */
      if (NULL == event || EXIT_SUCCESS == clRetainEvent(evt)) {
        if (EXIT_SUCCESS != clSetEventCallback(evt, CL_COMPLETE, libxstream_kernel_notify, info)) {
          /* the launch succeeded: a profile that cannot be taken is not an error */
          if (NULL != event) LIBXS_EXPECT_DEBUG(EXIT_SUCCESS == clReleaseEvent(evt));
          libxstream_launch_info_free(info);
          info = NULL;
        }
      }
      else {
        libxstream_launch_info_free(info);
        info = NULL;
      }
    }
    else if (NULL != info) { /* launch failed: no callback will run */
      libxstream_launch_info_free(info);
      info = NULL;
    }
    if (NULL != event) *event = evt;
    else if (NULL != evt && NULL == info) LIBXS_EXPECT_DEBUG(EXIT_SUCCESS == clReleaseEvent(evt));
  }
  else result = EXIT_FAILURE;
  CL_RETURN(result, "");
}


LIBXSTREAM_API int libxstream_opencl_launch(libxstream_stream_t* stream, cl_kernel kernel, cl_uint work_dim,
  const size_t* global_work_offset, const size_t* global_work_size, const size_t* local_work_size,
  cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event)
{
  return libxstream_opencl_launch_work(stream, kernel, work_dim, global_work_offset, global_work_size, local_work_size,
    num_events_in_wait_list, event_wait_list, event, 0 /*nflops*/, 0 /*nbytes*/);
}


LIBXSTREAM_API double libxstream_opencl_duration(cl_event event, int* result_code)
{
  cl_ulong begin = 0, end = 0;
  int r = EXIT_FAILURE;
  double result = 0;
  if (NULL != event) {
    r = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &begin, NULL);
    if (EXIT_SUCCESS == r) {
      r = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
      if (EXIT_SUCCESS == r) {
        result = 1E-9 * LIBXS_DELTA(begin, end); /* Nanoseconds->seconds */
      }
    }
  }
  if (NULL != result_code) *result_code = r;
  return result;
}


LIBXSTREAM_API int libxstream_opencl_error_consume(void)
{
  const int code = libxstream_opencl_config.device.error.code;
  libxstream_opencl_config.device.error.name = NULL;
  libxstream_opencl_config.device.error.code = EXIT_SUCCESS;
  return code;
}


LIBXSTREAM_API const char* libxstream_opencl_strerror(cl_int err)
{
  switch (err) {
    case 0: return "CL_SUCCESS";
    case -1: return "CL_DEVICE_NOT_FOUND";
    case -2: return "CL_DEVICE_NOT_AVAILABLE";
    case -3: return "CL_COMPILER_NOT_AVAILABLE";
    case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case -5: return "CL_OUT_OF_RESOURCES";
    case -6: return "CL_OUT_OF_HOST_MEMORY";
    case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case -8: return "CL_MEM_COPY_OVERLAP";
    case -9: return "CL_IMAGE_FORMAT_MISMATCH";
    case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case -11: return "CL_BUILD_PROGRAM_FAILURE";
    case -12: return "CL_MAP_FAILURE";
    case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case -15: return "CL_COMPILE_PROGRAM_FAILURE";
    case -16: return "CL_LINKER_NOT_AVAILABLE";
    case -17: return "CL_LINK_PROGRAM_FAILURE";
    case -18: return "CL_DEVICE_PARTITION_FAILED";
    case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
    case -30: return "CL_INVALID_VALUE";
    case -31: return "CL_INVALID_DEVICE_TYPE";
    case -32: return "CL_INVALID_PLATFORM";
    case -33: return "CL_INVALID_DEVICE";
    case -34: return "CL_INVALID_CONTEXT";
    case -35: return "CL_INVALID_QUEUE_PROPERTIES";
    case -36: return "CL_INVALID_COMMAND_QUEUE";
    case -37: return "CL_INVALID_HOST_PTR";
    case -38: return "CL_INVALID_MEM_OBJECT";
    case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case -40: return "CL_INVALID_IMAGE_SIZE";
    case -41: return "CL_INVALID_SAMPLER";
    case -42: return "CL_INVALID_BINARY";
    case -43: return "CL_INVALID_BUILD_OPTIONS";
    case -44: return "CL_INVALID_PROGRAM";
    case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case -46: return "CL_INVALID_KERNEL_NAME";
    case -47: return "CL_INVALID_KERNEL_DEFINITION";
    case -48: return "CL_INVALID_KERNEL";
    case -49: return "CL_INVALID_ARG_INDEX";
    case -50: return "CL_INVALID_ARG_VALUE";
    case -51: return "CL_INVALID_ARG_SIZE";
    case -52: return "CL_INVALID_KERNEL_ARGS";
    case -53: return "CL_INVALID_WORK_DIMENSION";
    case -54: return "CL_INVALID_WORK_GROUP_SIZE";
    case -55: return "CL_INVALID_WORK_ITEM_SIZE";
    case -56: return "CL_INVALID_GLOBAL_OFFSET";
    case -57: return "CL_INVALID_EVENT_WAIT_LIST";
    case -58: return "CL_INVALID_EVENT";
    case -59: return "CL_INVALID_OPERATION";
    case -60: return "CL_INVALID_GL_OBJECT";
    case -61: return "CL_INVALID_BUFFER_SIZE";
    case -62: return "CL_INVALID_MIP_LEVEL";
    case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
    case -64: return "CL_INVALID_PROPERTY";
    case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
    case -66: return "CL_INVALID_COMPILER_OPTIONS";
    case -67: return "CL_INVALID_LINKER_OPTIONS";
    case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";
    case -69: return "CL_INVALID_PIPE_SIZE";
    case -70: return "CL_INVALID_DEVICE_QUEUE";
    default: return "CL_UNKNOWN_ERROR";
  }
}

#endif /*__OPENCL*/
