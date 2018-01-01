/*****************************************************************************
 *  CP2K: A general program to perform molecular dynamics simulations        *
 *  Copyright (C) 2000 - 2018  CP2K developers group                         *
 *****************************************************************************/

#ifndef ACC_OPENCL_DEV_H
#define ACC_OPENCL_DEV_H

#if defined (__ACC) && defined (__OPENCL)

// maximum information line lenght (including null terminator)
// e.g. 'GPU\0'
#define MAX_DEV_TYPE_LEN 4

// struct definitions
typedef struct {
   cl_platform_id   platform_id;
   cl_device_id     device_id;
   cl_context       ctx;
} acc_opencl_dev_type;

// global (per MPI) device information
extern cl_uint acc_opencl_ndevices;
extern acc_opencl_dev_type *acc_opencl_devices;
extern acc_opencl_dev_type *acc_opencl_my_device;

// global configuration information
static cl_uint acc_opencl_ndevices_configured = 0;
static cl_uint acc_opencl_set_device_configured = 0;

#endif
#endif
//EOF
