/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxstream/libxstream_opencl.h>


#if defined(LIBXSTREAM_MPS_ICD)
/* The two names the OpenCL loader looks up in an ICD: libxstream_icd, a library of its own. */
LIBXS_EXTERN_C LIBXS_VISIBILITY_PUBLIC void* CL_API_CALL clGetExtensionFunctionAddress(const char* name)
{
  return libxstream_mps_icd_function(name);
}


LIBXS_EXTERN_C LIBXS_VISIBILITY_PUBLIC cl_int CL_API_CALL clIcdGetPlatformIDsKHR(
  cl_uint num_entries, cl_platform_id* platforms, cl_uint* num_platforms)
{
  return libxstream_mps_icd_platforms(num_entries, platforms, num_platforms);
}
#else
/* The server process (libxstream_mpsd), started by the first process of a node that needs it. */
int main(void)
{
  return libxstream_mps_server();
}
#endif
