/*****************************************************************************
 *  CP2K: A general program to perform molecular dynamics simulations        *
 *  Copyright (C) 2000 - 2014 the CP2K developers group                      *
 *****************************************************************************/

#if defined (__ACC) && defined (__OPENCL)

#include <CL/cl.h>
#include <string.h>
#include <stdio.h>

// defines error check functions and 'cl_error'
#include "acc_opencl_error.h"

// defines 'acc_opencl_my_device' and some default lenghts
#include "acc_opencl_dev.h"

// defines 'acc_opencl_host_buffer_node_type'
#include "acc_opencl_mem.h"
acc_opencl_host_buffer_node_type *host_buffer_list_head = NULL;
acc_opencl_host_buffer_node_type *host_buffer_list_tail = NULL;

// defines 'acc_opencl_stream_type'
#include "acc_opencl_stream.h"

// defines the ACC interface
#include "../include/acc.h"

static const int verbose_print = 0;


/****************************************************************************/
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Create a device buffer object of 'cl_mem' type
 *
 * Note: The data can't be accessed directly.
 */
int acc_dev_mem_allocate (void **dev_mem, size_t n){
  // debug info
  if (verbose_print){
    fprintf(stdout, "\n --- DEVICE MEMORY ALLOCATION --- \n");
    fprintf(stdout, " ---> Entering: acc_dev_mem_allocate.\n");
  }

  // create cl_mem buffer pointer
  *dev_mem = (void *) malloc(sizeof(cl_mem));
  cl_mem *dev_buffer = (cl_mem *) *dev_mem;

  // get a device buffer object
  *dev_buffer = clCreateBuffer(                // cl_mem
                  (*acc_opencl_my_device).ctx, // cl_context   context
                  (CL_MEM_READ_WRITE),         // cl_mem_flags flags
                  (size_t) n,                  // size_t       size [bytes]
                  NULL,                        // void         *host_ptr
                  &cl_error);                  // cl_int       *errcode_ret
  if (acc_opencl_error_check(cl_error, __LINE__)) return -1;

  // debug info
  if (verbose_print){
    fprintf(stdout, "      DEVICE buffer address: HEX=%p INT=%ld\n", dev_buffer, (uintptr_t) dev_buffer);
    fprintf(stdout, "      SIZE [bytes]:          INT=%ld\n", n);
    fprintf(stdout, " <--- Leaving: acc_dev_mem_allocate.\n");
  }

  // assign return value
  return 0;
}
#ifdef __cplusplus
}
#endif


/****************************************************************************/
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Destroy a device buffer object of 'cl_mem' type.
 */
int acc_dev_mem_deallocate (void *dev_mem){
  // debug info
  if (verbose_print){
    fprintf(stdout, "\n --- DEVICE MEMORY DEALLOCATION --- \n");
    fprintf(stdout, " ---> Entering: acc_dev_mem_deallocate.\n");
  }

  // local buffer object pointer 
  cl_mem *dev_buffer = (cl_mem *) dev_mem;

  // debug info
  if (verbose_print){
    fprintf(stdout, "      DEVICE buffer address:  HEX=%p INT=%ld\n", dev_buffer, (uintptr_t) dev_buffer);
  }

  // release device buffer object
  cl_error = clReleaseMemObject(*dev_buffer);
  if (acc_opencl_error_check(cl_error, __LINE__)) return -1;
  free(dev_buffer);
  dev_buffer = NULL;

  // debug info
  if (verbose_print){
    fprintf(stdout, " <--- Leaving: acc_dev_mem_deallocate.\n");
  }

  // assign return value
  return 0;
}
#ifdef __cplusplus
}
#endif


/****************************************************************************/
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Create a host memory pointer to memory of size 'n' bytes and an associated
 * host buffer object of 'cl_mem' type.
 *
 * Note: Only the pointer to the host_mem is given back.
 */
int acc_host_mem_allocate (void **host_mem, size_t n, void *stream){

  // debug info
  if (verbose_print){
    fprintf(stdout, "\n --- HOST MEMORY ALLOCATION --- \n");
    fprintf(stdout, " ---> Entering: acc_host_mem_allocate.\n");
  }

  // local stream object and memory object pointers
  acc_opencl_stream_type *opencl_stream = (acc_opencl_stream_type *) stream;
  acc_opencl_dev_type    opencl_device  = (*opencl_stream).device;
  cl_context             opencl_ctx     = opencl_device.ctx;
  cl_command_queue       opencl_queue   = (*opencl_stream).queue;

  // create a host pointer and an associated host buffer object
  cl_mem host_buffer = clCreateBuffer(                               // cl_mem
                        opencl_ctx,                                  // cl_context   context
                        (CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR), // cl_mem_flags flags
                        (size_t) n,                                  // size_t       size [bytes]
                        NULL,                                        // void         *host_ptr
                        &cl_error);                                  // cl_int       *errcode_ret
  if (acc_opencl_error_check(cl_error, __LINE__)) return -1;

  *host_mem = (void *) clEnqueueMapBuffer(             // cl_mem
                         opencl_queue,                 // cl_command_queue command_queue
                         host_buffer,                  // cl_mem           buffer
                         CL_TRUE,                      // cl_bool          blocking_map
                         (CL_MAP_READ | CL_MAP_WRITE), // cl_map_flags     map_flags
                         (size_t) 0,                   // size_t           offset
                         (size_t) n,                   // size_t           cb [bytes]
                         (cl_uint) 0,                  // cl_uint          num_events_in_wait_list
                         NULL,                         // const cl_event   *event_wait_list
                         NULL,                         // cl_event         *event
                         &cl_error);                   // cl_int           *errcode_ret
  if (acc_opencl_error_check(cl_error, __LINE__)) return -1;

  // keep 'buffer' and 'host_mem' information for deletion
  if (host_buffer_list_head == NULL){
    // create linked list and add 'buffer' as head node
    acc_opencl_host_buffer_node_type *buffer_node = (acc_opencl_host_buffer_node_type *) malloc(sizeof(acc_opencl_host_buffer_node_type));
    buffer_node->host_buffer = host_buffer;
    buffer_node->host_mem = (void *) *host_mem;
    buffer_node->next = NULL;
    host_buffer_list_head = host_buffer_list_tail = buffer_node;
  } else {
    // add to end of linked list of buffers
    acc_opencl_host_buffer_node_type *buffer_node = (acc_opencl_host_buffer_node_type *) malloc(sizeof(acc_opencl_host_buffer_node_type));
    buffer_node->host_buffer = host_buffer;
    buffer_node->host_mem = (void *) *host_mem;
    buffer_node->next = NULL;
    host_buffer_list_tail->next = buffer_node;
    host_buffer_list_tail = buffer_node;
  }

  // debug infos
  if (verbose_print){
    fprintf(stdout, "      HOST memory address:  HEX=%p INT=%ld\n", *host_mem, (uintptr_t) *host_mem);
    fprintf(stdout, "      SIZE [bytes]:         INT=%ld\n", n);
    fprintf(stdout, "      STREAM address:  HEX=%p INT=%ld\n", &opencl_queue, (uintptr_t) &opencl_queue);
    fprintf(stdout, "      STREAM value:  %u\n", opencl_queue);
    fprintf(stdout, " <--- Leaving: acc_host_mem_allocate.\n");
  }

  // assign return value
  return 0;
}
#ifdef __cplusplus
}
#endif


/****************************************************************************/
#ifdef __cplusplus
extern "C" {
#endif
int acc_host_mem_deallocate (void *host_mem, void *stream){

  // debug infos
  if (verbose_print){
    fprintf(stdout, "\n --- HOST MEMORY DEALLOCATION --- \n");
    fprintf(stdout, " ---> Entering: acc_host_mem_deallocate.\n");
    fprintf(stdout, "      HOST memory address:  HEX=%p INT=%ld\n", host_mem, (uintptr_t) host_mem);
  }

  // local stream object and memory object pointers
  acc_opencl_stream_type *opencl_stream = (acc_opencl_stream_type *) stream;
  cl_command_queue       opencl_queue   = (*opencl_stream).queue;

  // find corresponding 'buffer' object in host_buffer list
  acc_opencl_host_buffer_node_type *buffer_node_ptr = host_buffer_list_head;
  acc_opencl_host_buffer_node_type *buffer_node_prev = NULL;
  while (buffer_node_ptr != NULL){
    if (buffer_node_ptr->host_mem == host_mem){
      // extract node
      if (buffer_node_prev != NULL) buffer_node_prev->next = buffer_node_ptr->next;
      if (buffer_node_ptr == host_buffer_list_tail){
        host_buffer_list_tail = buffer_node_prev;
      } else if (buffer_node_ptr == host_buffer_list_head){
        host_buffer_list_head = buffer_node_ptr->next;
      }
      // unmap buffer
      cl_error = clEnqueueUnmapMemObject(        // cl_int
                   opencl_queue,                 // cl_command_queue command_queue
                   buffer_node_ptr->host_buffer, // cl_mem           memobj
                   host_mem,                     // void             *mapped_ptr
                   (cl_uint) 0,                  // cl_uint          num_evenets_in_wait_list
                   NULL,                         // cl_event         *event_wait_list
                   NULL);                        // cl_event         *event
      if (acc_opencl_error_check(cl_error, __LINE__)) return -1;
      // release buffer object
      cl_error = clReleaseMemObject(buffer_node_ptr->host_buffer);
      if (acc_opencl_error_check(cl_error, __LINE__)) return -1;
      // free buffer node
      free(buffer_node_ptr);
      buffer_node_ptr = NULL;
    } else {
      buffer_node_prev = buffer_node_ptr;
      buffer_node_ptr = buffer_node_ptr->next;
    }
  }

  // debug info
  if (verbose_print){
    fprintf(stdout, "      STREAM address:  HEX=%p INT=%ld\n", &opencl_queue, (uintptr_t) &opencl_queue);
    fprintf(stdout, "      STREAM value:  %u\n", opencl_queue);
    fprintf(stdout, " <--- Leaving: acc_host_mem_deallocate.\n");
  }

  // assign return value
  return 0;
}
#ifdef __cplusplus
}
#endif


/****************************************************************************/
#ifdef __cplusplus
extern "C" {
#endif
int acc_memcpy_h2d (const void *host_mem, void *dev_mem, size_t count, void *stream){
  // debug info
  if (verbose_print){
    fprintf(stdout, "\n === DATA TRANSFER (H2D) === \n");
    fprintf(stdout, " ---> Entering: acc_memcpy_h2d.\n");
  }

  // local buffer object pointer 
  cl_mem *dev_buffer  = (cl_mem *) dev_mem;

  // local stream object and memory object pointers
  acc_opencl_stream_type *opencl_stream = (acc_opencl_stream_type *) stream;
  cl_command_queue       opencl_queue   = (*opencl_stream).queue;

  // copy host memory to device buffer
  cl_error = clEnqueueWriteBuffer( // cl_int
               opencl_queue,       // cl_command_queue command_queue
               *dev_buffer,        // cl_mem           buffer
               CL_TRUE,            // cl_bool          blocking_write
               (size_t) 0,         // size_t           offset
               (size_t) count,     // size_t           cb
               host_mem,           // const void       *ptr
               (cl_uint) 0,        // cl_uint          num_evenets_in_wait_list
               NULL,               // cl_event         *event_wait_list
               NULL);              // cl_event         *event
  if (acc_opencl_error_check(cl_error, __LINE__)) return -1;

  // debug info
  if (verbose_print){
    fprintf(stdout, "      HOST memory address:   HEX=%p INT=%ld\n", host_mem, (uintptr_t) host_mem);
    fprintf(stdout, "      DEVICE buffer address: HEX=%p INT=%ld\n", dev_buffer, (uintptr_t) dev_buffer);
    fprintf(stdout, "      SIZE [bytes]:          INT=%ld\n", count);
    fprintf(stdout, "      STREAM address:  HEX=%p INT=%ld\n", &opencl_queue, (uintptr_t) &opencl_queue);
    fprintf(stdout, "      STREAM value:  %u\n", opencl_queue);
    fprintf(stdout, " <--- Leaving: acc_memcpy_h2d.\n");
  }

  // assign return value
  return 0;
}
#ifdef __cplusplus
}
#endif


/****************************************************************************/
#ifdef __cplusplus
extern "C" {
#endif
int acc_memcpy_d2h (const void *dev_mem, void *host_mem, size_t count, void *stream){
  // debug info
  if (verbose_print){
    fprintf(stdout, "\n === DATA TRANSFER (D2H) === \n");
    fprintf(stdout, " ---> Entering: acc_memcpy_d2h.\n");
  }

  // local buffer object pointer 
  const cl_mem *dev_buffer = (const cl_mem *) dev_mem;

  // local stream object and memory object pointers
  acc_opencl_stream_type *opencl_stream = (acc_opencl_stream_type *) stream;
  cl_command_queue       opencl_queue   = (*opencl_stream).queue;

  // copy host memory to device buffer
  cl_error = clEnqueueReadBuffer( // cl_int
               opencl_queue,      // cl_command_queue command_queue
               *dev_buffer,       // cl_mem           buffer
               CL_TRUE,           // cl_bool          blocking_read
               (size_t) 0,        // size_t           offset
               (size_t) count,    // size_t           cb
               host_mem,          // void             *ptr
               (cl_uint) 0,       // cl_uint          num_evenets_in_wait_list
               NULL,              // cl_event         *event_wait_list
               NULL);             // cl_event         *event
  if (acc_opencl_error_check(cl_error, __LINE__)) return -1;

  // debug info
  if (verbose_print){
    fprintf(stdout, "      DEVICE buffer address: HEX=%p INT=%ld\n", dev_buffer, (uintptr_t) dev_buffer);
    fprintf(stdout, "      HOST memory address:   HEX=%p INT=%ld\n", host_mem, (uintptr_t) host_mem);
    fprintf(stdout, "      SIZE [bytes]:          INT=%ld\n", count);
    fprintf(stdout, "      STREAM address:  HEX=%p INT=%ld\n", &opencl_queue, (uintptr_t) &opencl_queue);
    fprintf(stdout, "      STREAM value:  %u\n", opencl_queue);
    fprintf(stdout, " <--- Leaving: acc_memcpy_d2h.\n");
  }

  // assign return value
  return 0;
}
#ifdef __cplusplus
}
#endif


/****************************************************************************/
#ifdef __cplusplus
extern "C" {
#endif
int acc_memcpy_d2d (const void *devmem_src, void *devmem_dst, size_t count, void *stream){
  // debug info
  if (verbose_print){
    fprintf(stdout, "\n === DATA TRANSFER (D2D) === \n");
    fprintf(stdout, " ---> Entering: acc_memcpy_d2d.\n");
  }

  // local buffer object pointer 
  cl_mem *buffer_src = (cl_mem *) devmem_src;
  cl_mem *buffer_dst = (cl_mem *) devmem_dst;

  // local stream object and memory object pointers
  acc_opencl_stream_type *opencl_stream = (acc_opencl_stream_type *) stream;
  cl_command_queue       opencl_queue   = (*opencl_stream).queue;

  // copy device buffers from src to dst
  cl_error = clEnqueueCopyBuffer( // cl_int
               opencl_queue,      // cl_command_queue command_queue
               *buffer_src,       // cl_mem           src_buffer
               *buffer_dst,       // cl_mem           dst_buffer
               (size_t) 0,        // size_t           src_offset
               (size_t) 0,        // size_t           dst_offset
               (size_t) count,    // size_t           cb
               (cl_uint) 0,       // cl_uint          num_evenets_in_wait_list
               NULL,              // cl_event         *event_wait_list
               NULL);             // cl_event         *event
  if (acc_opencl_error_check(cl_error, __LINE__)) return -1;

  // debug info
  if (verbose_print){
    fprintf(stdout, "Coping %d bytes from device address %p to device address %p \n",
      count, buffer_src, buffer_dst);
    fprintf(stdout, "Leaving: acc_memcpy_d2d.\n");
  }
  if (verbose_print){
    fprintf(stdout, "      DEVICE buffer src address: HEX=%p INT=%ld\n", buffer_src, (uintptr_t) buffer_src);
    fprintf(stdout, "      DEVICE buffer dst address: HEX=%p INT=%ld\n", buffer_dst, (uintptr_t) buffer_dst);
    fprintf(stdout, "      SIZE [bytes]:          INT=%ld\n", count);
    fprintf(stdout, "      STREAM address:  HEX=%p INT=%ld\n", &opencl_queue, (uintptr_t) &opencl_queue);
    fprintf(stdout, "      STREAM value:  %u\n", opencl_queue);
    fprintf(stdout, " <--- Leaving: acc_memcpy_d2d.\n");
  }

  // assign return value
  return 0;
}
#ifdef __cplusplus
}
#endif


/****************************************************************************/
#ifdef __cplusplus
extern "C" {
#endif
int acc_memset_zero (void *dev_mem, size_t offset, size_t length, void *stream){
//       OpenCL 1.1 has no build in function for that!!!
//       We use cl_uchar because it's 8Bit = 1Byte long.
  // debug info
  if (verbose_print){
    fprintf(stdout, "\n --- ZERO DEVICE MEMORY --- \n");
    fprintf(stdout, " ---> Entering: acc_memset_zero.\n");
  }

  // local buffer object pointer 
  cl_mem *dev_buffer = (cl_mem *) dev_mem;

  // local stream object pointer
  acc_opencl_stream_type *opencl_stream = (acc_opencl_stream_type *) stream;
  cl_command_queue       opencl_queue   = (*opencl_stream).queue;

  // zero the values starting from offset in dev_mem
#ifdef CL_VERSION_1_2
  const cl_uchar zero = (cl_uchar) 0;

  cl_error = clEnqueueFillBuffer(         // cl_int
               opencl_queue,              // cl_command_queue command_queue
               *dev_buffer,               // cl_mem           buffer
               &zero,                     // const void       *pattern
               (size_t) sizeof(cl_uchar), // size_t           pattern_size
               (size_t) offset,           // size_t           offset
               (size_t) length,           // size_t           size [bytes]
               (cl_uint) 0,               // cl_uint          num_events_in_wait_list
               NULL,                      // const cl_event   *event_wait_list
               NULL);                     // cl_event         *event
  if (acc_opencl_error_check(cl_error, __LINE__)) return -1;
#else
  size_t i;
  // create a array of size 'lenght' and zero it
  cl_uchar *host_mem = (cl_uchar *) malloc(length * sizeof(cl_uchar));
  for (i = 0; i < length; i++){
    host_mem[i] = (cl_uchar) 0;
  }
  // transfer the 'zero_mem' to device buffer
  cl_error = clEnqueueWriteBuffer(      // cl_int
               opencl_queue,            // cl_command_queue command_queue
               *dev_buffer,             // cl_mem           buffer
               CL_TRUE,                 // cl_bool          blocking_write
               (size_t) offset,         // size_t           offset
               (size_t) length,         // size_t           cb
               (const void *) host_mem, // const void       *ptr
               (cl_uint) 0,             // cl_uint          num_event_in_wait_list
               NULL,                    // const cl_event   *event_wait_list
               NULL);                   // cl_event         *event
  if (acc_opencl_error_check(cl_error, __LINE__)) return -1;
  // free host array
  free(host_mem);
#endif

  // debug info
  if (verbose_print){
    fprintf(stdout, "     DEVICE buffer address:  HEX=%p INT=%ld\n", dev_buffer, (uintptr_t) dev_buffer);
    fprintf(stdout, "     STREAM address:  HEX=%p INT=%ld\n", &opencl_queue, (uintptr_t) &opencl_queue);
    fprintf(stdout, "     STREAM value:  %u\n", opencl_queue);
    fprintf(stdout, " <-- Leaving: acc_memset_zero.\n");
  }

  // assign return value
  return 0;
}
#ifdef __cplusplus
}
#endif


/****************************************************************************/
#ifdef __cplusplus
extern "C" {
#endif
int acc_dev_mem_info (size_t *free, size_t *avail){
// Note: OpenCL 1.x has no build in function for that!!!
  *free = 5500000000; // 5.5GByte
  *avail = *free;     // = same

  // assign return value
  return 0;

}
#ifdef __cplusplus
}
#endif

#endif
//EOF
