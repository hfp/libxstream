/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#if defined(__OPENCL)
#  include <libxstream_opencl.h>
#  include <string.h>
#  include <limits.h>
#  include <ctype.h>
#  include <math.h>
#  if defined(_WIN32)
#    include <windows.h>
#    include <process.h>
#  else
#    include <unistd.h>
#    include <errno.h>
#    include <glob.h>
#  endif
#  if defined(__DBCSR_ACC)
#    include "../acc_libsmm.h"
#  endif
#  include <fcntl.h>
#  include <sys/stat.h>
#  if !defined(S_ISDIR) && defined(S_IFMT) && defined(S_IFDIR)
#    define S_ISDIR(A) ((S_IFMT & (A)) == S_IFDIR)
#  endif
#  if !defined(S_IREAD)
#    define S_IREAD S_IRUSR
#  endif
#  if !defined(S_IWRITE)
#    define S_IWRITE S_IWUSR
#  endif

#  if !defined(LIBXSTREAM_NLOCKS)
#    define LIBXSTREAM_NLOCKS 4
#  endif
#  if !defined(LIBXSTREAM_TEMPDIR) && 1
#    define LIBXSTREAM_TEMPDIR "/tmp"
#  endif
#  if !defined(LIBXSTREAM_CACHE_DID) && 1
#    define LIBXSTREAM_CACHE_DID
#  endif
#  if !defined(LIBXSTREAM_CACHE_DIR) && 0
#    define LIBXSTREAM_CACHE_DIR ".cl_cache"
#  endif
#  if !defined(LIBXSTREAM_CPPBIN) && 1
#    define LIBXSTREAM_CPPBIN "/usr/bin/cpp"
#  endif
#  if !defined(LIBXSTREAM_SEDBIN) && 1
#    define LIBXSTREAM_SEDBIN "/usr/bin/sed"
#  endif
/* disabled: let MPI runtime come up before */
#  if !defined(LIBXSTREAM_PREINIT) && 0
#    define LIBXSTREAM_PREINIT
#  endif
/* attempt to enable command aggregation */
#  if !defined(LIBXSTREAM_CMDAGR) && 1
#    define LIBXSTREAM_CMDAGR
#  endif
#  if !defined(LIBXSTREAM_NCCS) && 1
#    define LIBXSTREAM_NCCS 0
#  endif


#  if defined(__cplusplus)
extern "C" {
#  endif

char libxstream_opencl_locks[LIBXS_CACHELINE * LIBXSTREAM_NLOCKS];
/* global configuration discovered during initialization */
libxstream_opencl_config_t libxstream_opencl_config;

#  if defined(LIBXSTREAM_CACHE_DID)
int libxstream_opencl_active_id;
#  endif


void libxstream_opencl_notify(const char /*errinfo*/[], const void* /*private_info*/, size_t /*cb*/, void* /*user_data*/);
void libxstream_opencl_notify(const char errinfo[], const void* private_info, size_t cb, void* user_data) {
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
int libxstream_opencl_order_devices(const void* /*dev_a*/, const void* /*dev_b*/);
int libxstream_opencl_order_devices(const void* dev_a, const void* dev_b) {
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
          return (size_a < size_b ? 1 : (size_a != size_b ? -1 : (a < b ? -1 : 1)));
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


/** Setup to run prior to touching OpenCL runtime. */
void libxstream_opencl_configure(void);
void libxstream_opencl_configure(void) {
  const char* const env_rank = (NULL != getenv("PMI_RANK") ? getenv("PMI_RANK") : getenv("OMPI_COMM_WORLD_LOCAL_RANK"));
  const char* const env_nranks = getenv("MPI_LOCALNRANKS"); /* TODO */
  const char *const env_devsplit = getenv("LIBXSTREAM_DEVSPLIT"), *const env_nlocks = getenv("LIBXSTREAM_NLOCKS");
  const char *const env_verbose = getenv("LIBXSTREAM_VERBOSE"), *const env_dump_acc = getenv("LIBXSTREAM_DUMP");
  const char *const env_debug = getenv("LIBXSTREAM_DEBUG"), *const env_profile = getenv("LIBXSTREAM_PROFILE");
  const char* const env_dump = (NULL != env_dump_acc ? env_dump_acc : getenv("IGC_ShaderDumpEnable"));
  const char *const env_neo = getenv("NEOReadDebugKeys"), *const env_wa = getenv("LIBXSTREAM_WA");
  static char neo_enable_debug_keys[] = "NEOReadDebugKeys=1";
#  if defined(LIBXSTREAM_STREAM_PRIORITIES)
  const char* const env_priority = getenv("LIBXSTREAM_PRIORITY");
#  endif
#  if defined(LIBXSTREAM_NCCS)
  const char* const env_nccs = getenv("LIBXSTREAM_NCCS");
  const int nccs = (NULL == env_nccs ? LIBXSTREAM_NCCS : atoi(env_nccs));
#  endif
#  if defined(LIBXSTREAM_XHINTS)
  const char* const env_xhints = (LIBXSTREAM_XHINTS);
  const int xhints_default = 1 + 2 + 4 + 8 + 16;
#  else
  const char* const env_xhints = NULL;
  const int xhints_default = 0;
#  endif
#  if defined(LIBXSTREAM_ASYNC)
  const char* const env_async = (LIBXSTREAM_ASYNC);
  const int async_default = 1 + 2 + 4 + 8;
#  else
  const char* const env_async = NULL;
  const int async_default = 0;
#  endif
  const int nlocks = (NULL == env_nlocks ? 1 /*default*/ : atoi(env_nlocks));
  const int neo = (NULL == env_neo ? 1 : atoi(env_neo));
  int i;
#  if defined(_OPENMP)
  const int max_threads = omp_get_max_threads(), num_threads = omp_get_num_threads();
  memset(&libxstream_opencl_config, 0, sizeof(libxstream_opencl_config));
  libxstream_opencl_config.nthreads = (num_threads < max_threads ? max_threads : num_threads);
#  else
  memset(&libxstream_opencl_config, 0, sizeof(libxstream_opencl_config));
  libxstream_opencl_config.nthreads = 1;
#  endif
  assert(NULL == libxstream_opencl_config.lock_main); /* test condition to avoid initializing multiple times */
  libxs_init(); /* before using LIBXSMM's functionality */
  libxstream_opencl_config.nranks = LIBXS_MAX(NULL != env_nranks ? atoi(env_nranks) : 1, 1);
  libxstream_opencl_config.nrank = (NULL != env_rank ? atoi(env_rank) : 0) % libxstream_opencl_config.nranks;
  assert(sizeof(libxs_lock_t) <= LIBXS_CACHELINE);
  for (i = 0; i < LIBXSTREAM_NLOCKS; ++i) {
    LIBXS_LOCK_ATTR_TYPE(LIBXS_LOCK) acc_opencl_attr_;
    LIBXS_LOCK_ATTR_INIT(LIBXS_LOCK, &acc_opencl_attr_);
    LIBXS_LOCK_INIT(LIBXS_LOCK, (libxs_lock_t*)(libxstream_opencl_locks + LIBXS_CACHELINE * i), &acc_opencl_attr_);
    LIBXS_LOCK_ATTR_DESTROY(LIBXS_LOCK, &acc_opencl_attr_);
  }
  libxstream_opencl_config.lock_main = (libxs_lock_t*)libxstream_opencl_locks;
  libxstream_opencl_config.lock_memory = /* 2nd lock-domain */
    (1 < LIBXS_MIN(nlocks, LIBXSTREAM_NLOCKS) ? ((libxs_lock_t*)(libxstream_opencl_locks + LIBXS_CACHELINE * 1))
                                                : libxstream_opencl_config.lock_main);
  libxstream_opencl_config.lock_stream = /* 3rd lock-domain */
    (2 < LIBXS_MIN(nlocks, LIBXSTREAM_NLOCKS) ? ((libxs_lock_t*)(libxstream_opencl_locks + LIBXS_CACHELINE * 2))
                                                : libxstream_opencl_config.lock_main);
  libxstream_opencl_config.lock_event = /* 4th lock-domain */
    (3 < LIBXS_MIN(nlocks, LIBXSTREAM_NLOCKS) ? ((libxs_lock_t*)(libxstream_opencl_locks + LIBXS_CACHELINE * 3))
                                                : libxstream_opencl_config.lock_main);
  libxstream_opencl_config.verbosity = (NULL == env_verbose ? 0 : atoi(env_verbose));
  libxstream_opencl_config.devsplit = (NULL == env_devsplit ? (/*1 < libxstream_opencl_config.nranks ? -1 :*/ 0)
                                                             : atoi(env_devsplit));
#  if defined(LIBXSTREAM_STREAM_PRIORITIES)
  libxstream_opencl_config.priority = (NULL == env_priority ? /*default*/ 3 : atoi(env_priority));
#  endif
  libxstream_opencl_config.profile = (NULL == env_profile ? /*default*/ 0 : atoi(env_profile));
  libxstream_opencl_config.xhints = (NULL == env_xhints ? xhints_default : atoi(env_xhints));
  libxstream_opencl_config.async = (NULL == env_async ? async_default : atoi(env_async));
  libxstream_opencl_config.dump = (NULL == env_dump ? /*default*/ 0 : atoi(env_dump));
  libxstream_opencl_config.debug = (NULL == env_debug ? libxstream_opencl_config.dump : atoi(env_debug));
  libxstream_opencl_config.wa = neo * (NULL == env_wa ? ((1 != libxstream_opencl_config.devsplit ? 0 : 1) + (2 + 4 + 8))
                                                       : atoi(env_wa));
#  if defined(LIBXSTREAM_CACHE_DIR)
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
#    if defined(LIBXSTREAM_TEMPDIR)
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
#    endif
    if (NULL != env_cachedir) {
#    if defined(_WIN32)
      LIBXS_UNUSED(env_cachedir);
#    else
#      if defined(S_IRWXU) && defined(S_IRGRP) && defined(S_IXGRP) && defined(S_IROTH) && defined(S_IXOTH)
      const int mode = S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH;
#      else
      const int mode = 0xFFFFFFFF;
#      endif
      LIBXS_EXPECT(0 == mkdir(env_cachedir, mode) || EEXIST == errno); /* soft-error */
#    endif
    }
  }
#  endif
#  if defined(LIBXSTREAM_NCCS)
  if (0 != nccs && NULL == getenv("ZEX_NUMBER_OF_CCS")) {
    static char zex_nccs[LIBXSTREAM_MAXNDEVS * 8 + 32] = "ZEX_NUMBER_OF_CCS=";
    const int mode = ((1 == nccs || 2 == nccs) ? nccs : 4);
    int j = strlen(zex_nccs);
    for (i = 0; i < LIBXSTREAM_MAXNDEVS; ++i) {
      const int n = (0 < i
        ? LIBXS_SNPRINTF(zex_nccs + j, sizeof(zex_nccs) - j, ",%u:%i", i, mode)
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
      fprintf(stderr, "INFO ACC/OpenCL: support multiple separate compute command streamers (%i-CCS mode)\n", mode);
    }
  }
#  endif
  if (0 != neo && (NULL != env_neo || 0 == LIBXS_PUTENV(neo_enable_debug_keys))) {
    if ((1 + 2 + 4) & libxstream_opencl_config.wa) {
      static char a[] = "ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE", b[] = "EnableRecoverablePageFaults=0";
      static char c[] = "DirectSubmissionOverrideBlitterSupport=0", *const apply[] = {a, b, c};
      if ((1 & libxstream_opencl_config.wa) && NULL == getenv("ZE_FLAT_DEVICE_HIERARCHY")) {
        LIBXS_EXPECT(0 == LIBXS_PUTENV(apply[0]));
      }
#  if (1 >= LIBXSTREAM_USM)
      if ((2 & libxstream_opencl_config.wa) && NULL == getenv("EnableRecoverablePageFaults")) {
        LIBXS_EXPECT(0 == LIBXS_PUTENV(apply[1]));
      }
#  endif
      if ((4 & libxstream_opencl_config.wa) && NULL == getenv("DirectSubmissionOverrideBlitterSupport")) {
        LIBXS_EXPECT(0 == LIBXS_PUTENV(apply[2]));
      }
    }
    if (0 != libxstream_opencl_config.debug && NULL == getenv("DisableScratchPages")) {
      static char a[] = "DisableScratchPages=1", *const apply[] = {a};
      LIBXS_EXPECT(0 == LIBXS_PUTENV(apply[0]));
    }
  }
}


int libxstream_init(void) {
#  if defined(_OPENMP)
  /* initialization/finalization is not meant to be thread-safe */
  int result = ((0 == omp_in_parallel() || /*main*/ 0 == omp_get_thread_num()) ? EXIT_SUCCESS : EXIT_FAILURE);
#  else
  int result = EXIT_SUCCESS;
#  endif
  if (NULL == libxstream_opencl_config.lock_main) { /* avoid to configure multiple times */
    libxstream_opencl_configure();
  }
  /* eventually touch OpenCL/compute runtime after configure */
  if (0 == libxstream_opencl_config.ndevices && EXIT_SUCCESS == result) { /* avoid to initialize multiple times */
    char buffer[LIBXSTREAM_BUFFERSIZE];
    cl_platform_id platforms[LIBXSTREAM_MAXNDEVS] = {NULL};
    cl_device_id devices[LIBXSTREAM_MAXNDEVS];
    cl_device_type type = CL_DEVICE_TYPE_ALL;
    cl_uint nplatforms = 0, ndevices = 0, i;
    const char* const env_devmatch = getenv("LIBXSTREAM_DEVMATCH");
    const char* const env_devtype = getenv("LIBXSTREAM_DEVTYPE");
    const char* const env_device = getenv("LIBXSTREAM_DEVICE");
    char* const env_devids = getenv("LIBXSTREAM_DEVIDS");
    int device_id = (NULL == env_device ? 0 : atoi(env_device));
#  if defined(LIBXSTREAM_CACHE_DID)
    assert(0 == libxstream_opencl_active_id);
#  endif
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
#  if defined(CL_VERSION_1_2)
              cl_device_partition_property properties[] = {
                CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN, CL_DEVICE_AFFINITY_DOMAIN_NUMA, /*terminator*/ 0};
              cl_uint nunits = 0, n = 0;
              if ((1 < libxstream_opencl_config.devsplit || 0 > libxstream_opencl_config.devsplit) &&
                  /* Intel CPU (e.g., out of two sockets) yields thread-count of both sockets */
                  EXIT_SUCCESS == clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &nunits, NULL) &&
                  1 < nunits)
              {
                n = LIBXS_MIN(1 < libxstream_opencl_config.devsplit ? (cl_uint)libxstream_opencl_config.devsplit : nunits,
                  LIBXSTREAM_MAXNDEVS);
                properties[0] = CL_DEVICE_PARTITION_EQUALLY;
                properties[1] = (nunits + n - 1) / n;
              }
              if (0 == libxstream_opencl_config.devsplit || 1 == libxstream_opencl_config.devsplit ||
                  (libxstream_opencl_config.ndevices + 1) == LIBXSTREAM_MAXNDEVS ||
                  EXIT_SUCCESS != clCreateSubDevices(devices[j], properties, 0, NULL, &n))
#  endif
              {
                libxstream_opencl_config.devices[libxstream_opencl_config.ndevices] = devices[j];
                ++libxstream_opencl_config.ndevices;
              }
#  if defined(CL_VERSION_1_2)
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
#  endif
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
#  if defined(CL_VERSION_1_2)
              LIBXS_EXPECT_DEBUG(EXIT_SUCCESS == clReleaseDevice(libxstream_opencl_config.devices[i]));
#  endif
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
        char* did = strtok(env_devids, LIBXSTREAM_DELIMS " ");
        for (; NULL != did && ndevids < LIBXSTREAM_MAXNDEVS; did = strtok(NULL, LIBXSTREAM_DELIMS " ")) {
          const int id = atoi(did);
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
#  if defined(CL_VERSION_1_2)
              LIBXS_EXPECT_DEBUG(EXIT_SUCCESS == clReleaseDevice(libxstream_opencl_config.devices[i]));
#  endif
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
            else if (CL_DEVICE_TYPE_ALL == type && NULL == env_devtype /*&& CL_DEVICE_TYPE_GPU == itype*/ && device_id <= LIBXS_CAST_INT(i)) {
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
            libxstream_opencl_config.devices[0] =
              libxstream_opencl_config.devices[device_id % libxstream_opencl_config.ndevices];
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
        libxstream_opencl_config.stream_data = (libxstream_opencl_stream_t*)malloc(
          sizeof(libxstream_opencl_stream_t) * nhandles);
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
          libxs_pmalloc_init(sizeof(cl_event*), &libxstream_opencl_config.nevents,
            (void**)libxstream_opencl_config.events, libxstream_opencl_config.event_data);
        }
        else {
          free(libxstream_opencl_config.events);
          free(libxstream_opencl_config.event_data);
          libxstream_opencl_config.event_data = NULL;
          libxstream_opencl_config.events = NULL;
          libxstream_opencl_config.nevents = 0;
          result = EXIT_FAILURE;
        }
        /* create host memory pool (3-arg libxs_malloc) */
        if (EXIT_SUCCESS == result) {
          libxstream_opencl_config.pool_hst = libxs_malloc_pool(NULL /*malloc*/, NULL /*free*/);
          if (NULL == libxstream_opencl_config.pool_hst) result = EXIT_FAILURE;
        }
        if (
          1 <= libxstream_opencl_config.profile ||
          0 > libxstream_opencl_config.profile)
        {
          const int profile = LIBXS_MAX(LIBXS_ABS(libxstream_opencl_config.profile), 2);
          const libxs_hist_update_t update[] = {libxs_hist_update_avg, libxs_hist_update_add};
          libxstream_opencl_config.hist_h2d = libxs_hist_create(profile + 1, 2, update);
          libxstream_opencl_config.hist_d2h = libxs_hist_create(profile + 1, 2, update);
          libxstream_opencl_config.hist_d2d = libxs_hist_create(profile + 1, 2, update);
        }
        else {
          assert(NULL == libxstream_opencl_config.hist_h2d);
          assert(NULL == libxstream_opencl_config.hist_d2h);
          assert(NULL == libxstream_opencl_config.hist_d2d);
        }
        if (EXIT_SUCCESS == result) { /* lastly, print active device and list of devices */
#  if defined(LIBXSTREAM_ACTIVATE)
          if (0 <= LIBXSTREAM_ACTIVATE && LIBXSTREAM_ACTIVATE < libxstream_opencl_config.ndevices) {
            result = libxstream_opencl_set_active_device(NULL /*lock*/, LIBXSTREAM_ACTIVATE);
          }
          else {
            if (0 < libxstream_opencl_config.nrank && 1 < libxstream_opencl_config.ndevices) {
              device_id = libxstream_opencl_config.nrank % libxstream_opencl_config.ndevices;
            }
            result = libxstream_opencl_set_active_device(NULL /*lock*/, device_id);
          }
#  else
          libxstream_opencl_config.device_id = device_id;
#  endif
          if ((2 <= libxstream_opencl_config.verbosity || 0 > libxstream_opencl_config.verbosity) &&
              (0 == libxstream_opencl_config.nrank))
          {
            char platform_name[LIBXSTREAM_BUFFERSIZE];
            for (i = 0; i < (cl_uint)libxstream_opencl_config.ndevices; ++i) {
              if (EXIT_SUCCESS == libxstream_opencl_device_name(libxstream_opencl_config.devices[i], buffer,
                                    LIBXSTREAM_BUFFERSIZE, platform_name, LIBXSTREAM_BUFFERSIZE, /*cleanup*/ 0))
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
#  if defined(__DBCSR_ACC)
    /* DBCSR shall call libxstream_init as well as libsmm_acc_init (since both interfaces are used).
     * Also, libsmm_acc_init may privately call libxstream_init (as it depends on the ACC interface).
     * The implementation of libxstream_init should hence be safe against "over initialization".
     * However, DBCSR only calls libxstream_init (and expects an implicit libsmm_acc_init).
     */
    if (EXIT_SUCCESS == result) result = libsmm_acc_init();
#  endif
  }
  CL_RETURN(result, "");
}


/* attempt to automatically initialize backend */
LIBXS_ATTRIBUTE_CTOR void libxstream_opencl_init(void) {
  if (NULL == libxstream_opencl_config.lock_main) { /* avoid to configure multiple times */
    libxstream_opencl_configure();
  }
#  if defined(LIBXSTREAM_PREINIT)
  LIBXS_EXPECT_DEBUG(EXIT_SUCCESS == libxstream_init());
#  endif
}


/* attempt to automatically finalize backend */
LIBXS_ATTRIBUTE_DTOR void libxstream_opencl_finalize(void) {
  assert(libxstream_opencl_config.ndevices < LIBXSTREAM_MAXNDEVS);
  if (0 != libxstream_opencl_config.ndevices) {
    const int precision[] = {0, 1};
    int i;
    LIBXS_STDIO_ACQUIRE();
    libxs_hist_print(stderr, libxstream_opencl_config.hist_h2d, "\nPROF ACC/OpenCL: H2D", precision);
    libxs_hist_print(stderr, libxstream_opencl_config.hist_d2h, "\nPROF ACC/OpenCL: D2H", precision);
    libxs_hist_print(stderr, libxstream_opencl_config.hist_d2d, "\nPROF ACC/OpenCL: D2D", precision);
    LIBXS_STDIO_RELEASE();
    for (i = 0; i < LIBXSTREAM_MAXNDEVS; ++i) {
      const cl_device_id device_id = libxstream_opencl_config.devices[i];
      if (NULL != device_id) {
#  if defined(CL_VERSION_1_2) && 0 /* avoid potential segfault */
        LIBXS_EXPECT_DEBUG(EXIT_SUCCESS == clReleaseDevice(device_id));
#  endif
      }
    }
    if (NULL != libxstream_opencl_config.device.stream.queue) { /* release private stream */
      clReleaseCommandQueue(libxstream_opencl_config.device.stream.queue); /* ignore return code */
    }
    if (NULL != libxstream_opencl_config.device.context) {
      const cl_context context = libxstream_opencl_config.device.context;
      libxstream_opencl_config.device.context = NULL;
      clReleaseContext(context); /* ignore return code */
    }
    for (i = 0; i < LIBXSTREAM_NLOCKS; ++i) { /* destroy locks */
      LIBXS_LOCK_DESTROY(LIBXS_LOCK, (libxs_lock_t*)(libxstream_opencl_locks + LIBXS_CACHELINE * i));
    }
    /* release/reset buffers */
    libxs_hist_destroy(libxstream_opencl_config.hist_h2d);
    libxs_hist_destroy(libxstream_opencl_config.hist_d2h);
    libxs_hist_destroy(libxstream_opencl_config.hist_d2d);
    libxs_free_pool(libxstream_opencl_config.pool_hst);
    /* NOTE: registered streams/events are not individually released here;
     * the OpenCL runtime reclaims resources at process exit (atexit context). */
    free(libxstream_opencl_config.memptrs);
    free(libxstream_opencl_config.memptr_data);
    free(libxstream_opencl_config.streams);
    free(libxstream_opencl_config.stream_data);
    free(libxstream_opencl_config.events);
    free(libxstream_opencl_config.event_data);
    /* clear entire configuration structure */
    memset(&libxstream_opencl_config, 0, sizeof(libxstream_opencl_config));
#  if defined(LIBXSTREAM_CACHE_DID)
    libxstream_opencl_active_id = 0; /* reset cached active device-ID */
#  endif
    libxs_finalize();
  }
}


int libxstream_finalize(void) {
#  if defined(_OPENMP)
  /* initialization/finalization is not meant to be thread-safe */
  int result = ((0 == omp_in_parallel() || /*main*/ 0 == omp_get_thread_num()) ? EXIT_SUCCESS : EXIT_FAILURE);
#  else
  int result = EXIT_SUCCESS;
#  endif
  static void (*cleanup)(void) = libxstream_opencl_finalize;
  assert(libxstream_opencl_config.ndevices < LIBXSTREAM_MAXNDEVS);
  if (0 != libxstream_opencl_config.ndevices && NULL != cleanup) {
#  if defined(__DBCSR_ACC)
    /* DBCSR may call libxstream_init as well as libsmm_acc_init() since both interface are used.
     * libsmm_acc_init may privately call libxstream_init (as it depends on the ACC interface).
     * The implementation of libxstream_init should be safe against "over initialization".
     * However, DBCSR only calls libxstream_init and expects an implicit libsmm_acc_init().
     */
    if (EXIT_SUCCESS == result) result = libsmm_acc_finalize();
#  endif
    if (EXIT_SUCCESS == result) result = atexit(cleanup);
    cleanup = NULL;
  }
  CL_RETURN(result, "");
}


int libxstream_opencl_use_cmem(const libxstream_opencl_device_t* devinfo) {
#  if defined(LIBXSTREAM_CMEM)
  return (0 != devinfo->size_maxalloc && devinfo->size_maxalloc <= devinfo->size_maxcmem) ? EXIT_SUCCESS : EXIT_FAILURE;
#  else
  return EXIT_FAILURE;
#  endif
}


int libxstream_device_count(int* ndevices) {
  int result;
#  if defined(__DBCSR_ACC) /* lazy initialization */
  /* DBCSR calls libxstream_device_count before calling libxstream_init. */
  result = libxstream_init();
  if (EXIT_SUCCESS == result)
#  endif
  {
    if (NULL != ndevices) {
      *ndevices = (0 < libxstream_opencl_config.ndevices ? libxstream_opencl_config.ndevices : 0);
      result = EXIT_SUCCESS;
    }
    else result = EXIT_FAILURE;
  }
  CL_RETURN(result, "");
}


int libxstream_opencl_device_id(cl_device_id device, int* device_id, int* global_id) {
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


int libxstream_opencl_device_vendor(cl_device_id device, const char vendor[], int use_platform_name) {
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


int libxstream_opencl_device_uid(cl_device_id device, const char devname[], unsigned int* uid) {
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


int libxstream_opencl_device_name(
  cl_device_id device, char name[], size_t name_maxlen, char platform[], size_t platform_maxlen, int cleanup) {
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


int libxstream_opencl_device_level(
  cl_device_id device, int std_clevel[2], int std_level[2], char std_flag[16], cl_device_type* type) {
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
      const int nchar = LIBXS_SNPRINTF(std_flag, 16, "-cl-std=CL%u.0", std_level_uint[0]);
      if (0 >= nchar || 16 <= nchar) result = EXIT_FAILURE;
    }
    else if (1 <= std_level_uint[0]) {
      if (1 <= std_level_uint[1]) {
        const int nchar = LIBXS_SNPRINTF(std_flag, 16, "-cl-std=CL%u.%u", std_level_uint[0], std_level_uint[1]);
        if (0 >= nchar || 16 <= nchar) result = EXIT_FAILURE;
      }
      else if (1 <= std_clevel_uint[0]) { /* fallback */
        const int nchar = LIBXS_SNPRINTF(std_flag, 16, "-cl-std=CL%u.%u", std_clevel_uint[0], std_clevel_uint[1]);
        if (0 >= nchar || 16 <= nchar) result = EXIT_FAILURE;
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


int libxstream_opencl_device_ext(cl_device_id device, const char* const extnames[], int num_exts) {
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
        ext = strtok(buffer, LIBXSTREAM_DELIMS " \t");
        for (; NULL != ext; ext = ((ext + 1) < end ? strtok((ext + 1) + strlen(ext), LIBXSTREAM_DELIMS " \t") : NULL)) {
          if (NULL == strstr(extensions, ext)) {
            return EXIT_FAILURE;
          }
        }
      }
    } while (0 < num_exts);
  }
  return result;
}


int libxstream_opencl_create_context(cl_device_id active_id, cl_context* context) {
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
            libxstream_opencl_config.ndevices, global_id, buffer, (void*)ctx, libxs_pid(),
            libxstream_opencl_config.nthreads);
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


int libxstream_opencl_set_active_device(libxs_lock_t* lock, int device_id) {
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
#  if defined(CL_VERSION_1_2)
          LIBXS_EXPECT_DEBUG(EXIT_SUCCESS == clReleaseDevice(context_id));
#  endif
          result = clReleaseContext(context);
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
          const char* const sgexts[] = {"cl_intel_required_subgroup_size", "cl_intel_subgroups", "cl_khr_subgroups"};
          size_t sgsizes[16], nbytes = 0, i;
          LIBXSTREAM_STREAM_PROPERTIES_TYPE properties[4] = {
            CL_QUEUE_PROPERTIES, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, 0 /* terminator */
          };
          devinfo->intel = (EXIT_SUCCESS == libxstream_opencl_device_vendor(active_id, "intel", 0 /*use_platform_name*/));
          devinfo->nv = (EXIT_SUCCESS == libxstream_opencl_device_vendor(active_id, "nvidia", 0 /*use_platform_name*/));
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
          if (EXIT_SUCCESS != clGetDeviceInfo(active_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), devinfo->wgsize, NULL)) {
            devinfo->wgsize[0] = 1;
          }
          if (EXIT_SUCCESS != clGetDeviceInfo(active_id, 4199 /*CL_DEVICE_PREFERRED_WORK_GROUP_SIZE_MULTIPLE*/, sizeof(size_t),
                                devinfo->wgsize + 1, NULL)) /* CL_VERSION_3_0 */
          {
            devinfo->wgsize[1] = 1;
          }
          assert(0 == devinfo->wgsize[2]);
          if (EXIT_SUCCESS == libxstream_opencl_device_ext(active_id, sgexts, 2) && 0 != devinfo->wgsize[1] &&
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
          else devinfo->wgsize[2] = 0;
          if (0 != devinfo->intel) {
            const char* const env_biggrf = getenv("LIBXSTREAM_BIGGRF");
            devinfo->biggrf = (NULL != env_biggrf && 0 != atoi(env_biggrf));
          }
#  if defined(LIBXSTREAM_XHINTS) && (1 >= LIBXSTREAM_USM)
          { /* cl_intel_unified_shared_memory extension */
            cl_platform_id platform = NULL;
            cl_bitfield bitfield = 0;
            if (0 != (1 & libxstream_opencl_config.xhints) && 2 <= *devinfo->std_level && 0 != devinfo->intel &&
                /*0 == libxstream_opencl_config.profile &&*/ (0 == devinfo->unified || NULL != (LIBXSTREAM_XHINTS)) &&
                EXIT_SUCCESS == clGetDeviceInfo(active_id, CL_DEVICE_PLATFORM, sizeof(cl_platform_id), &platform, NULL) &&
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
#  endif
#  if (0 != LIBXSTREAM_USM)
          { /* OpenCL 2.0 based SVM capabilities */
            const char* const env_usm = getenv("LIBXSTREAM_USM");
            cl_device_svm_capabilities svmcaps = 0;
            if (NULL == env_usm) {
              if (0 == devinfo->nv) { /* vendor workaround (force with LIBXSTREAM_USM=1) */
                result = clGetDeviceInfo(active_id, CL_DEVICE_SVM_CAPABILITIES, sizeof(cl_device_svm_capabilities), &svmcaps, NULL);
                assert(EXIT_SUCCESS == result || 0 == svmcaps);
              }
            }
            else svmcaps = (cl_device_svm_capabilities)atoi(env_usm);
            devinfo->usm = (cl_int)svmcaps;
          }
#  endif
#  if defined(LIBXSTREAM_CMDAGR)
          if (0 != devinfo->intel) { /* device vendor (above) can now be used */
            int result_cmdagr = EXIT_SUCCESS;
            const cl_command_queue q = LIBXSTREAM_CREATE_COMMAND_QUEUE(context, active_id, properties, &result_cmdagr);
            if (EXIT_SUCCESS == result_cmdagr) {
              assert(NULL != q);
              clReleaseCommandQueue(q);
            }
          }
#  endif
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


int libxstream_device_set_active(int device_id) {
  int result = EXIT_SUCCESS;
  if (0 <= device_id) {
#  if defined(__DBCSR_ACC) && defined(__OFFLOAD_OPENCL)
    if (0 == libxstream_opencl_config.ndevices) { /* not initialized */
      result = libxstream_init();
    }
#  endif
  }
  else result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result) {
    if (device_id < libxstream_opencl_config.ndevices) {
#  if defined(LIBXSTREAM_CACHE_DID)
      if (libxstream_opencl_active_id != (device_id + 1))
#  endif
      {
        result = libxstream_opencl_set_active_device(libxstream_opencl_config.lock_main, device_id);
#  if defined(LIBXSTREAM_CACHE_DID)
        if (EXIT_SUCCESS == result) libxstream_opencl_active_id = device_id + 1;
#  endif
      }
    }
    else result = EXIT_FAILURE;
  }
  CL_RETURN(result, "");
}


int libxstream_opencl_flags_atomics(const libxstream_opencl_device_t* devinfo, libxstream_opencl_atomic_fp_t kind,
  const char* exts[], size_t* exts_maxlen, char flags[], size_t flags_maxlen) {
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
          if (EXIT_SUCCESS == clGetDeviceInfo(device_id,
                                (cl_device_info)(libxstream_opencl_atomic_fp_64 == kind ? 0x4232 : 0x4231), sizeof(cl_bitfield),
                                &fp_atomics, NULL) &&
              0 != (/*add*/ (1 << 1) & fp_atomics))
          {
            exts[ext2] = "cl_ext_float_atomics";
#  if 1 /* enabling this permitted extension in source code causes compiler warning */
            *exts_maxlen = ext2; /* quietly report extension by reducing exts_maxlen */
#  endif
            atomic_exp = (libxstream_opencl_atomic_fp_64 == kind
                            ? "atomic_fetch_add_explicit((GLOBAL_VOLATILE(atomic_double)*)A,B,"
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


int libxstream_opencl_defines(const char defines[], char buffer[], size_t buffer_size, int cleanup) {
  const libxstream_opencl_device_t* const devinfo = &libxstream_opencl_config.device;
  int result = 0;
  if (NULL != buffer && NULL != devinfo->context) {
    const int std_clevel = 100 * devinfo->std_clevel[0] + 10 * devinfo->std_clevel[1];
    const int std_level = 100 * devinfo->std_level[0] + 10 * devinfo->std_level[1];
    result = LIBXS_SNPRINTF(buffer, buffer_size, " -DLIBXSTREAM_VERSION=%u -DLIBXSTREAM_C_VERSION=%u%s", std_level, std_clevel,
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


int libxstream_opencl_kernel_flags(const char build_params[], const char build_options[], const char try_options[],
  cl_program program, char buffer[], size_t buffer_size) {
  const libxstream_opencl_device_t* const devinfo = &libxstream_opencl_config.device;
  int result = EXIT_SUCCESS, nchar = 0;
  assert(NULL != program && (NULL != buffer || 0 == buffer_size));
  nchar = libxstream_opencl_defines(build_params, buffer, buffer_size, 1 /*cleanup*/);
  if (0 <= nchar && LIBXS_CAST_INT(buffer_size) > nchar) {
    const int debug = (0 != libxstream_opencl_config.debug && 0 != devinfo->intel && CL_DEVICE_TYPE_CPU != devinfo->type);
    int n = LIBXS_SNPRINTF(buffer + nchar, buffer_size - nchar, " %s%s %s%s", 0 == debug ? "" : "-gline-tables-only ",
      devinfo->std_flag, NULL != build_options ? build_options : "",
      0 != devinfo->biggrf ? " -cl-intel-256-GRF-per-thread" : "");
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


int libxstream_opencl_program(size_t source_kind, const char source[], const char name[], const char build_params[],
  const char build_options[], const char try_options[], int* try_ok, const char* const extnames[], size_t num_exts,
  cl_program* program) {
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
                      ? libxs_malloc(libxstream_opencl_config.pool_hst, size_src + source_is_cl /*terminator?*/, 0 /*auto-align*/)
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
          char* ext = strtok(strncpy(buffer, extnames[n - 1], LIBXSTREAM_BUFFERSIZE - 1), LIBXSTREAM_DELIMS " \t");
          for (; NULL != ext; ext = ((ext + 1) < end ? strtok((ext + 1) + strlen(ext), LIBXSTREAM_DELIMS " \t") : NULL), ++nflat) {
            size_ext += strlen(ext);
          }
        }
      }
      if (0 < size_ext && 0 < nflat) {
        const char* const enable_ext = "#pragma OPENCL EXTENSION %s : enable\n";
        const size_t size_src_ext = size_src + size_ext + nflat * (strlen(enable_ext) - 2 /*%s*/);
        char* const ext_source_buffer = (char*)libxs_malloc(libxstream_opencl_config.pool_hst, size_src_ext + 1 /*terminator*/, 0 /*auto-align*/);
        if (NULL != ext_source_buffer) {
          for (n = 0; 0 < num_exts; --num_exts) {
            if (NULL != extnames[num_exts - 1]) {
              const char* const end = buffer_name + strlen(extnames[num_exts - 1]); /* before strtok */
              char* ext;
              strncpy(buffer_name, extnames[num_exts - 1], LIBXSTREAM_MAXSTRLEN * 2 - 1);
              buffer_name[LIBXSTREAM_MAXSTRLEN * 2 - 1] = '\0';
              ext = strtok(buffer_name, LIBXSTREAM_DELIMS " \t");
              for (; NULL != ext; ext = ((ext + 1) < end ? strtok((ext + 1) + strlen(ext), LIBXSTREAM_DELIMS " \t") : NULL)) {
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
#  if !defined(NDEBUG)
                if (EXIT_SUCCESS == libxstream_opencl_device_ext(device_id, (const char* const*)&ext, 1))
#  endif
                { /* NDEBUG: assume given extension is supported (confirmed upfront) */
                  if (NULL == line) { /* extension is not already part of source */
                    n += LIBXS_SNPRINTF(
                      ext_source_buffer + n, size_src_ext + 1 /*terminator*/ - n, "#pragma OPENCL EXTENSION %s : enable\n", ext);
                  }
                }
#  if !defined(NDEBUG)
                else if (0 != strcmp("cl_intel_global_float_atomics", ext)) {
                  fprintf(stderr, "WARN ACC/OpenCL: extension \"%s\" is not supported.\n", ext);
                }
#  endif
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
        const int cpp = (NULL == env_cpp ? 1 /*default*/ : atoi(env_cpp));
#  if defined(LIBXSTREAM_CPPBIN)
        FILE* const file_cpp = (0 != cpp ? fopen(LIBXSTREAM_CPPBIN, "rb") : NULL);
#  else
        FILE* const file_cpp = NULL;
#  endif
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
#  if defined(LIBXSTREAM_CPPBIN)
        if (NULL != file_cpp && 0 <= file_dmp) { /* preprocess source-code */
          const char* sed_pattern = "";
#    if defined(LIBXSTREAM_SEDBIN)
          FILE* const file_sed = fopen(LIBXSTREAM_SEDBIN, "rb");
          if (NULL != file_sed) {
            sed_pattern = "| " LIBXSTREAM_SEDBIN " '/^[[:space:]]*\\(\\/\\/.*\\)*$/d'";
            fclose(file_sed); /* existence-check */
          }
#    endif
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
                                          ? libxs_malloc(libxstream_opencl_config.pool_hst, size_file + 1 /*terminator*/, 0 /*auto-align*/)
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
#  endif
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
        binary = (unsigned char*)(EXIT_SUCCESS ==
                                      clGetProgramInfo(*program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &size, NULL)
                                    ? libxs_malloc(libxstream_opencl_config.pool_hst, size, 0 /*auto-align*/)
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
#  if defined(CL_VERSION_2_1)
    if (0 != libxstream_opencl_config.dump) *program = clCreateProgramWithIL(devinfo->context, source, size_src, &result);
    else
#  endif
    {
      *program = clCreateProgramWithBinary(
        devinfo->context, 1, &device_id, &size_src, (const unsigned char**)&source, NULL /*binary_status*/, &result);
    }
    assert(EXIT_SUCCESS != result || NULL != *program);
    if (EXIT_SUCCESS == result) {
      ok = libxstream_opencl_kernel_flags(build_params, build_options, try_options, *program, buffer, LIBXSTREAM_BUFFERSIZE);
      if (EXIT_SUCCESS == ok) result = ok;
      else {
#  if defined(CL_VERSION_2_1)
        if (0 != libxstream_opencl_config.dump) *program = clCreateProgramWithIL(devinfo->context, source, size_src, &result);
        else
#  endif
        {
          *program = clCreateProgramWithBinary(
            devinfo->context, 1, &device_id, &size_src, (const unsigned char**)&source, NULL /*binary_status*/, &result);
        }
        assert(EXIT_SUCCESS != result || NULL != *program);
        if (EXIT_SUCCESS == result) {
          result = clBuildProgram(*program, 1 /*num_devices*/, &device_id, buffer, NULL /*callback*/, NULL /*user_data*/);
          ok = EXIT_FAILURE;
        }
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


int libxstream_opencl_kernel_query(cl_program program, const char kernel_name[], cl_kernel* kernel) {
  int result;
  assert(NULL != kernel);
  *kernel = NULL;
  if (NULL != program && NULL != kernel_name && '\0' != *kernel_name) {
    *kernel = clCreateKernel(program, kernel_name, &result);
#  if defined(CL_VERSION_1_2)
    if (EXIT_SUCCESS != result) { /* discover available kernels in program, and adopt the last kernel listed */
      char kbuf[LIBXSTREAM_BUFFERSIZE];
      if (EXIT_SUCCESS == clGetProgramInfo(program, CL_PROGRAM_KERNEL_NAMES, sizeof(kbuf), kbuf, NULL) && '\0' != *kbuf) {
        const char *const semicolon = strrchr(kbuf, ';'), *const kname = (NULL == semicolon ? kbuf : (semicolon + 1));
        *kernel = clCreateKernel(program, kname, &result);
      }
    }
#  endif
    assert(EXIT_SUCCESS != result || NULL != *kernel);
  }
  else result = EXIT_FAILURE;
  return result;
}


int libxstream_opencl_kernel(size_t source_kind, const char source[], const char kernel_name[], const char build_params[],
  const char build_options[], const char try_options[], int* try_ok, const char* const extnames[], size_t num_exts,
  cl_kernel* kernel) {
  cl_program program = NULL;
  int result;
  assert(NULL != kernel);
  *kernel = NULL;
  result = libxstream_opencl_program(source_kind, source, kernel_name, build_params,
    build_options, try_options, try_ok, extnames, num_exts, &program);
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


int libxstream_opencl_set_kernel_ptr(cl_kernel kernel, cl_uint arg_index, const void* arg_value) {
  libxstream_opencl_device_t* const devinfo = &libxstream_opencl_config.device;
  int result = EXIT_FAILURE;
  assert(NULL != devinfo->context);
#  if (1 >= LIBXSTREAM_USM)
  if (NULL != devinfo->clSetKernelArgMemPointerINTEL) {
    result = devinfo->clSetKernelArgMemPointerINTEL(kernel, arg_index, arg_value);
  }
  else
#  endif
#  if (0 != LIBXSTREAM_USM)
    if (0 != devinfo->usm)
  {
    result = clSetKernelArgSVMPointer(kernel, arg_index, arg_value);
  }
  else
#  elif defined(NDEBUG)
  LIBXS_UNUSED(devinfo);
#  endif
  {
    result = clSetKernelArg(kernel, arg_index, sizeof(cl_mem), &arg_value);
  }
  CL_RETURN(result, "");
}


double libxstream_opencl_duration(cl_event event, int* result_code) {
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


int libxstream_opencl_error_consume(void) {
  const int code = libxstream_opencl_config.device.error.code;
  libxstream_opencl_config.device.error.name = NULL;
  libxstream_opencl_config.device.error.code = EXIT_SUCCESS;
  return code;
}


const char* libxstream_opencl_strerror(cl_int err) {
  switch (err) {
    case   0: return "CL_SUCCESS";
    case  -1: return "CL_DEVICE_NOT_FOUND";
    case  -2: return "CL_DEVICE_NOT_AVAILABLE";
    case  -3: return "CL_COMPILER_NOT_AVAILABLE";
    case  -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case  -5: return "CL_OUT_OF_RESOURCES";
    case  -6: return "CL_OUT_OF_HOST_MEMORY";
    case  -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case  -8: return "CL_MEM_COPY_OVERLAP";
    case  -9: return "CL_IMAGE_FORMAT_MISMATCH";
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

#  if defined(__cplusplus)
}
#  endif

#endif /*__OPENCL*/
