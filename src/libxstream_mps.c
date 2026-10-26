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
# if defined(__linux__) && defined(__has_include)
#   if __has_include(<CL/cl_icd.h>)
#     define LIBXSTREAM_MPS
#   endif
# endif
# if defined(LIBXSTREAM_MPS)
#   include <CL/cl_icd.h>
#   include <libxs/libxs_reg.h>
#   include <libxs/libxs_str.h>
#   include <libxs/libxs_timer.h>
#   include <libxs/libxs_utils.h>
#   include <sys/mman.h>
#   include <sys/socket.h>
#   include <sys/syscall.h>
#   include <sys/uio.h>
#   include <sys/un.h>
#   include <fcntl.h>
#   include <limits.h>
#   include <errno.h>
#   include <poll.h>
#   include <sys/wait.h>
#   include <dirent.h>
#   include <pthread.h>
#   include <semaphore.h>
#   include <signal.h>
#   include <spawn.h>
#   include <time.h>
#   include <unistd.h>
# endif

# if defined(LIBXSTREAM_MPS)
#   if !defined(LIBXSTREAM_MPS_MAXNCONN)
#     define LIBXSTREAM_MPS_MAXNCONN 1024
#   endif
#   if !defined(LIBXSTREAM_MPS_MAXNMAPS)
#     define LIBXSTREAM_MPS_MAXNMAPS 1024
#   endif
#   if !defined(LIBXSTREAM_MPS_CAPACITY)
#     define LIBXSTREAM_MPS_CAPACITY (1 << 20)
#   endif
#   if !defined(LIBXSTREAM_MPS_ARENA)
#     define LIBXSTREAM_MPS_ARENA 256
#   endif
/* seconds to wait for the server (client) or for the first client (server) */
#   if !defined(LIBXSTREAM_MPS_CONNECT)
#     define LIBXSTREAM_MPS_CONNECT 60
#   endif
#   if !defined(LIBXSTREAM_MPS_IDLE)
#     define LIBXSTREAM_MPS_IDLE 2
#   endif
/* installed server, relative to the library's directory (build systems pass the actual path) */
#   if !defined(LIBXSTREAM_MPS_BINDIR)
#     define LIBXSTREAM_MPS_BINDIR "../bin"
#   endif
#   define LIBXSTREAM_MPS_NARGS 11
#   define LIBXSTREAM_MPS_UNORDERED (~(uint64_t)0)
#   define LIBXSTREAM_MPS_FENCE (~(uint64_t)0)
/* a dependency's top bits: the object kind, and whether another thread used the object (LIBXSTREAM_VERBOSE=2) */
#   define LIBXSTREAM_MPS_TAG(KIND, CROSS) ((uint64_t)(((KIND) & 7) | (0 != (CROSS) ? 8 : 0)) << 60)
#   define LIBXSTREAM_MPS_UNTAG(DEP) ((DEP) & ~((uint64_t)15 << 60))
#   define LIBXSTREAM_MPS_TAGOF(DEP) ((unsigned int)((DEP) >> 60))
#   define LIBXSTREAM_MPS_NWAITS 17 /* tags, and the window or a fence */
#   if !defined(LIBXSTREAM_MPS_NDEPS)
#     define LIBXSTREAM_MPS_NDEPS 128
#   endif
#   if !defined(LIBXSTREAM_MPS_WINDOW)
#     define LIBXSTREAM_MPS_WINDOW (1 << 20)
#   endif
#   define LIBXSTREAM_MPS_NLOCAL 16
#   define LIBXSTREAM_MPS_NCLASSES 40
#   define LIBXSTREAM_MPS_PAGE 4096
#   define LIBXSTREAM_MPS_SUFFIX "MPS"
#   define LIBXSTREAM_MPS_NAME " (LIBXSTREAM-MPS)"
#   define LIBXSTREAM_MPS_INFO 9
#   define LIBXSTREAM_MPS_GEN 10

#   define LIBXSTREAM_MPS_HELLO 1
#   define LIBXSTREAM_MPS_PLATFORM_INFO 2
#   define LIBXSTREAM_MPS_DEVICE_IDS 3
#   define LIBXSTREAM_MPS_DEVICE_INFO 4
#   define LIBXSTREAM_MPS_CONTEXT 5
#   define LIBXSTREAM_MPS_QUEUE 6
#   define LIBXSTREAM_MPS_QUEUE_INFO 7
#   define LIBXSTREAM_MPS_BUFFER 8
#   define LIBXSTREAM_MPS_SUBBUFFER 9
#   define LIBXSTREAM_MPS_PROGRAM_SOURCE 10
#   define LIBXSTREAM_MPS_PROGRAM_BINARY 11
#   define LIBXSTREAM_MPS_BUILD 12
#   define LIBXSTREAM_MPS_PROGRAM_INFO 13
#   define LIBXSTREAM_MPS_BUILD_INFO 14
#   define LIBXSTREAM_MPS_KERNEL 15
#   define LIBXSTREAM_MPS_KERNEL_INFO 16
#   define LIBXSTREAM_MPS_KERNEL_WGINFO 17
#   define LIBXSTREAM_MPS_RELEASE 19
#   define LIBXSTREAM_MPS_WRITE 20
#   define LIBXSTREAM_MPS_READ 21
#   define LIBXSTREAM_MPS_COPY 22
#   define LIBXSTREAM_MPS_FILL 23
#   define LIBXSTREAM_MPS_NDRANGE 24
#   define LIBXSTREAM_MPS_MARKER 25
#   define LIBXSTREAM_MPS_MAP 26
#   define LIBXSTREAM_MPS_UNMAP 27
#   define LIBXSTREAM_MPS_FINISH 28
#   define LIBXSTREAM_MPS_FLUSH 29
#   define LIBXSTREAM_MPS_WAIT 30
#   define LIBXSTREAM_MPS_EVENT_STATUS 31
#   define LIBXSTREAM_MPS_NOPS 32

#   define LIBXSTREAM_MPS_KIND_DEVICE 1
#   define LIBXSTREAM_MPS_KIND_CONTEXT 2
#   define LIBXSTREAM_MPS_KIND_QUEUE 3
#   define LIBXSTREAM_MPS_KIND_MEM 4
#   define LIBXSTREAM_MPS_KIND_PROGRAM 5
#   define LIBXSTREAM_MPS_KIND_KERNEL 6
#   define LIBXSTREAM_MPS_KIND_EVENT 7
#   define LIBXSTREAM_MPS_KIND_PLATFORM 8
/* what else the reaper releases (libxstream_mpsd_reaper): a kernel's instances, a departed process */
#   define LIBXSTREAM_MPS_REAP_KERNEL 16
#   define LIBXSTREAM_MPS_REAP_CLIENT 17

#   define LIBXSTREAM_MPS_ARG_VALUE 0
#   define LIBXSTREAM_MPS_ARG_MEM 1
#   define LIBXSTREAM_MPS_ARG_LOCAL 2

#   define LIBXSTREAM_MPS_OBJ(HANDLE) ((libxstream_mps_obj_t*)(HANDLE))
#   define LIBXSTREAM_MPS_IS(HANDLE, KIND) (NULL != (HANDLE) && (KIND) == LIBXSTREAM_MPS_OBJ(HANDLE)->kind)
#   define LIBXSTREAM_MPS_ID(CLIENT, HANDLE) \
      (NULL != (HANDLE) ? (uint64_t)(LIBXSTREAM_MPS_OBJ(HANDLE) - (CLIENT)->objects) + 1 : 0)
#   define LIBXSTREAM_MPS_DEVID(CLIENT, HANDLE) ((uint64_t)(LIBXSTREAM_MPS_OBJ(HANDLE) - (CLIENT)->devices))
#   define LIBXSTREAM_MPS_COUNT(DST_PTR, VALUE) LIBXS_ATOMIC(LIBXS_ATOMIC_ADD_FETCH, 64)(DST_PTR, VALUE, LIBXS_ATOMIC_RELAXED)
# endif


# if defined(LIBXSTREAM_MPS)
/* A command: a fixed header followed by nbytes of payload. */
typedef struct libxstream_mps_msg_t { /* ndeps sequence numbers follow the payload */
  uint64_t op, nbytes, seq, ndeps;
  uint64_t arg[LIBXSTREAM_MPS_NARGS];
} libxstream_mps_msg_t;

/* The answer to a synchronous command, followed by nbytes of data. */
typedef struct libxstream_mps_reply_t {
  int64_t result;
  uint64_t nbytes, arg0;
} libxstream_mps_reply_t;

/* Client-side object: the loader dispatches through the first member. */
typedef struct libxstream_mps_obj_t {
  const struct _cl_icd_dispatch* dispatch;
  struct libxstream_mps_obj_t* context;
  struct libxstream_mps_obj_t* owner; /* program of a kernel, queue of an event, parent of a sub-buffer, free list */
  struct libxstream_mps_obj_t* device;
  char* host; /* what the application sees of a buffer: the shadow, or its own pointer (USE_HOST_PTR) */
  char* shadow; /* shared memory a buffer is created on */
  size_t size, offset; /* of a buffer */
  uint64_t flags; /* memory flags, queue properties, device mask of a context, or command type */
  uint64_t create, last; /* sequence numbers (plus one) of the object's creation and of its last use */
  uint64_t done; /* a queue's commands up to this sequence number (plus one) are known to be complete */
  struct libxstream_mps_obj_t** args; /* a kernel's buffer arguments: a launch touches them */
  uint64_t* pargs; /* a kernel's arguments, four words per index (libxstream_mps_kernel_arg) */
  void** bigs; /* a kernel's argument values wider than a word, per index */
  size_t npargs; /* indexes */
  unsigned int gen; /* tells the object from the slot's earlier objects */
  unsigned int ogen; /* generation of the owner when the object was created (an event's queue) */
  unsigned int tid; /* thread of the last command that touched the object (LIBXSTREAM_VERBOSE=2) */
  int refs, kind;
} libxstream_mps_obj_t;

typedef struct libxstream_mps_map_t {
  libxstream_mps_obj_t* buffer;
  char *ptr, *temp;
  size_t offset, size;
  uint64_t flags;
} libxstream_mps_map_t;

/* A lock of its own cache line: per-thread connection locks sit side by side. */
typedef union libxstream_mps_lock_t {
  libxs_lock_t lock;
  char pad[LIBXS_CACHELINE];
} libxstream_mps_lock_t;

/* Key of a cached platform or device query (libxstream_mps_cached). */
typedef struct libxstream_mps_key_t {
  int64_t device;
  uint64_t param;
} libxstream_mps_key_t;

typedef struct libxstream_mps_client_t {
  libxstream_mps_obj_t devices[LIBXSTREAM_MAXNDEVS];
  libxstream_mps_map_t maps[LIBXSTREAM_MPS_MAXNMAPS];
  libxstream_mps_lock_t conn_lock[LIBXSTREAM_MPS_MAXNCONN];
  int conn[LIBXSTREAM_MPS_MAXNCONN];
  char* arena_free[LIBXSTREAM_MPS_NCLASSES];
  char path[sizeof(((struct sockaddr_un*)NULL)->sun_path)];
  libxstream_mps_lock_t lock, order;
  libxstream_mps_obj_t *objects, *free_head, *free_tail; /* released slots, reused once the server released them */
  libxs_registry_t* cache; /* platform and device queries: answered once, never changing */
  uint32_t* released; /* per slot in shared memory: set by the server once the object is released */
  unsigned int* gens; /* per object slot, kept across reuse */
  char* arena;
  size_t arena_size, arena_top, nobjects, capacity;
  uint64_t seq, token;
  uint64_t cross[16]; /* dependencies on another thread's command, by object kind (LIBXSTREAM_VERBOSE=2) */
  int memfd, verbosity;
} libxstream_mps_client_t;

/* A mapping of a shared buffer not given back yet, undone when the client leaves (libxstream_mpsd_leave). */
typedef struct libxstream_mpsd_mapping_t {
  cl_command_queue queue; /* retained */
  cl_mem buffer;
  void* ptr;
} libxstream_mpsd_mapping_t;

/* An object the reaper releases (libxstream_mpsd_reap). */
typedef struct libxstream_mpsd_reap_t {
  void* ptr;
  int kind;
} libxstream_mpsd_reap_t;

typedef struct libxstream_mpsd_release_t {
  uint64_t after, id; /* released once every command before "after" ran */
  int kind;
} libxstream_mpsd_release_t;

/* A kernel as one connection launches it: launches from different threads share no arguments. */
typedef struct libxstream_mpsd_instance_t {
  struct libxstream_mpsd_instance_t* next;
  cl_kernel kernel;
  uint64_t* applied; /* per argument: the record last set (four words, the first one zero if none) */
  size_t napplied;
  int conn;
} libxstream_mpsd_instance_t;

/* What creates a kernel's instances (libxstream_mpsd_instance). */
typedef struct libxstream_mpsd_kernel_t {
  cl_program program;
  char* name;
  libxstream_mpsd_instance_t* instances;
} libxstream_mpsd_kernel_t;

typedef struct libxstream_mpsd_client_t {
  pthread_mutex_t lock;
  pthread_cond_t cond;
  struct libxstream_mpsd_client_t* next;
  void** handles;
  libxstream_mpsd_kernel_t** kernels; /* per kernel slot */
  cl_command_queue* lazy; /* per event slot without a driver event yet: its queue (libxstream_mpsd_event) */
  unsigned char* used; /* per event slot: 1 waited for, 2 polled (LIBXSTREAM_VERBOSE=2) */
  uint64_t nevents, nused[4]; /* events released, by how they were used (LIBXSTREAM_VERBOSE=2) */
  unsigned char* evop; /* per event slot: the command that made it (LIBXSTREAM_VERBOSE=2) */
  uint64_t op_events[LIBXSTREAM_MPS_NOPS], op_needed[LIBXSTREAM_MPS_NOPS]; /* released, waited for or polled (VERBOSE=2) */
  unsigned int* gens; /* of the events in handles, written before them */
  unsigned char* kinds;
  unsigned char* done; /* executed commands at or beyond low, by sequence number modulo the window */
  libxstream_mpsd_release_t* pending; /* deferred releases */
  size_t npending, maxpending;
  libxstream_mpsd_mapping_t* maps; /* mappings not given back yet */
  size_t nmaps, maxmaps;
  char* arena;
  size_t arena_size, capacity;
  uint64_t token, low; /* every command below low was executed */
  uint64_t ncmds, nstatus, wait_ns, exec_ns, post_ns, status_ns; /* LIBXSTREAM_VERBOSE=2 */
  uint64_t op_n[LIBXSTREAM_MPS_NOPS], op_ns[LIBXSTREAM_MPS_NOPS]; /* per command (LIBXSTREAM_VERBOSE=2) */
  uint64_t wait_n[LIBXSTREAM_MPS_NWAITS], wait_by[LIBXSTREAM_MPS_NWAITS]; /* by what blocked last (LIBXSTREAM_VERBOSE=2) */
  libxs_hist_t* hist; /* execution time per command: log2(ns) buckets with {us, wait us} (LIBXSTREAM_VERBOSE=2) */
  libxs_lock_t hist_lock;
  int nconn, connids, dead, pid, nerrors, ordinal;
} libxstream_mpsd_client_t;

/* What a command leaves to do once the next command may run: build, wait, answer. */
typedef struct libxstream_mpsd_post_t {
  cl_command_queue finish;
  cl_event event;
  cl_event* events;
  cl_uint nevents, nbuild;
  cl_program program;
  const cl_device_id* devices;
  const char* options;
  void* data;
  size_t nbytes;
  uint64_t arg0;
  cl_int result;
  int reply, conn; /* conn: the connection executing the command */
} libxstream_mpsd_post_t;

typedef struct libxstream_mpsd_t {
  cl_platform_id platform;
  cl_device_id devices[LIBXSTREAM_MAXNDEVS];
  cl_context contexts[LIBXSTREAM_MAXNDEVS];
  uint64_t masks[LIBXSTREAM_MAXNDEVS];
  int slots[LIBXSTREAM_MAXNDEVS]; /* which of the contexts per set of devices (LIBXSTREAM_MPS_CONTEXTS) */
  libxstream_mpsd_client_t* clients;
  sem_t gate; /* bounds the commands executing at once (LIBXSTREAM_MPS_ISSUERS) */
  cl_uint ndevices, ncontexts;
  int issuers; /* zero: unbounded */
  int nthreads, nserved, verbosity, ncontexts_per; /* contexts per set of devices */
} libxstream_mpsd_t;
# endif


# if defined(LIBXSTREAM_MPS)
extern char** environ;
static struct _cl_icd_dispatch libxstream_mps_dispatch;
static libxstream_mps_obj_t libxstream_mps_platform;
static pthread_once_t libxstream_mps_once = PTHREAD_ONCE_INIT;
static pthread_mutex_t libxstream_mps_lock = PTHREAD_MUTEX_INITIALIZER;
static libxstream_mps_client_t* libxstream_mps_client;
static int libxstream_mps_failed, libxstream_mps_unsupported_count;
static pthread_mutex_t libxstream_mpsd_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t libxstream_mpsd_reaplock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t libxstream_mpsd_reapcond = PTHREAD_COND_INITIALIZER;
static libxstream_mpsd_reap_t* libxstream_mpsd_reaps; /* queue of what the reaper releases, in order */
static size_t libxstream_mpsd_nreaps, libxstream_mpsd_reaphead, libxstream_mpsd_maxreaps;
static int libxstream_mpsd_reaping; /* 1: started, 2: releasing */
static libxstream_mpsd_t libxstream_mpsd;
# endif


# if defined(LIBXSTREAM_MPS)
static int libxstream_mps_recv(int fd, void* data, size_t size)
{
  int result = EXIT_SUCCESS;
  char* p = (char*)data;
  while (0 < size && EXIT_SUCCESS == result) {
    const ssize_t n = read(fd, p, size);
    if (0 < n) {
      p += n;
      size -= (size_t)n;
    }
    else if (0 > n && EINTR == errno) continue;
    else result = EXIT_FAILURE;
  }
  return result;
}


/* MSG_NOSIGNAL: a peer that went away must be an error, not a SIGPIPE. */
static int libxstream_mps_send(int fd, struct iovec* iov, int niov, int memfd)
{
  int result = EXIT_SUCCESS, first = 1;
  while (0 < niov && EXIT_SUCCESS == result) {
    union { struct cmsghdr header; char data[CMSG_SPACE(sizeof(int))]; } control;
    struct msghdr mh;
    ssize_t n;
    memset(&mh, 0, sizeof(mh));
    mh.msg_iov = iov;
    mh.msg_iovlen = (size_t)niov;
    if (0 != first && 0 <= memfd) { /* the descriptor travels with the first byte */
      struct cmsghdr* cmsg;
      memset(&control, 0, sizeof(control));
      mh.msg_control = control.data;
      mh.msg_controllen = sizeof(control.data);
      cmsg = CMSG_FIRSTHDR(&mh);
      cmsg->cmsg_level = SOL_SOCKET;
      cmsg->cmsg_type = SCM_RIGHTS;
      cmsg->cmsg_len = CMSG_LEN(sizeof(int));
      memcpy(CMSG_DATA(cmsg), &memfd, sizeof(int));
    }
    n = sendmsg(fd, &mh, MSG_NOSIGNAL);
    if (0 < n) {
      first = 0;
      while (0 < niov && (size_t)n >= iov->iov_len) {
        n -= (ssize_t)iov->iov_len;
        ++iov;
        --niov;
      }
      if (0 < niov) {
        iov->iov_base = (char*)iov->iov_base + n;
        iov->iov_len -= (size_t)n;
      }
    }
    else if (0 > n && EINTR == errno) continue;
    else result = EXIT_FAILURE;
  }
  return result;
}


/* Socket path of the server: beside the lock file of libxs_singleton_acquire, which it mirrors. */
static void libxstream_mps_paths(char name[], size_t name_size, char path[], size_t path_size)
{
  const char* key = getenv("LIBXSTREAM_MPS_KEY");
  if (NULL == key) key = getenv("SLURM_JOB_ID");
  if (NULL == key) key = getenv("PBS_JOBID");
  if (NULL == key) key = getenv("LSB_JOBID");
  if (NULL == key) key = "0";
  LIBXS_SNPRINTF(name, name_size, "mps.%s", key);
  LIBXS_SNPRINTF(path, path_size, "/tmp/.libxs.%u.%s.sock", (unsigned int)getuid(), name);
}


static intptr_t libxstream_mps_unsupported(void)
{
  if (1 == LIBXS_ATOMIC_ADD_FETCH(&libxstream_mps_unsupported_count, 1, LIBXS_ATOMIC_SEQ_CST)) {
    fprintf(stderr, "ERROR LIBXSTREAM-MPS: unsupported OpenCL function called\n");
  }
  return (intptr_t)CL_INVALID_OPERATION;
}


static cl_int libxstream_mps_answer(const void* data, size_t nbytes, size_t size, void* value, size_t* size_ret)
{
  cl_int result = CL_SUCCESS;
  if (NULL != value) {
    if (size >= nbytes) memcpy(value, data, nbytes);
    else result = CL_INVALID_VALUE;
  }
  if (NULL != size_ret) *size_ret = nbytes;
  return result;
}


static cl_int libxstream_mps_invalid(int kind)
{
  cl_int result;
  switch (kind) {
    case LIBXSTREAM_MPS_KIND_DEVICE: result = CL_INVALID_DEVICE; break;
    case LIBXSTREAM_MPS_KIND_CONTEXT: result = CL_INVALID_CONTEXT; break;
    case LIBXSTREAM_MPS_KIND_QUEUE: result = CL_INVALID_COMMAND_QUEUE; break;
    case LIBXSTREAM_MPS_KIND_MEM: result = CL_INVALID_MEM_OBJECT; break;
    case LIBXSTREAM_MPS_KIND_PROGRAM: result = CL_INVALID_PROGRAM; break;
    case LIBXSTREAM_MPS_KIND_KERNEL: result = CL_INVALID_KERNEL; break;
    case LIBXSTREAM_MPS_KIND_EVENT: result = CL_INVALID_EVENT; break;
    default: result = CL_INVALID_PLATFORM;
  }
  return result;
}


static size_t libxstream_mps_class(size_t size)
{
  size_t result = 0;
  while (((size_t)LIBXSTREAM_MPS_PAGE << result) < size) ++result;
  return result;
}


/* Shared memory in power-of-two classes: what a class held once is reused for any size of it. */
static char* libxstream_mps_arena_alloc(libxstream_mps_client_t* c, size_t size)
{
  const size_t k = libxstream_mps_class(size);
  char* result = NULL;
  if (LIBXSTREAM_MPS_NCLASSES > k) {
    const size_t nbytes = (size_t)LIBXSTREAM_MPS_PAGE << k;
    LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, &c->lock.lock);
    result = c->arena_free[k];
    if (NULL != result) memcpy(&c->arena_free[k], result, sizeof(char*));
    else if (c->arena_top + nbytes <= c->arena_size) {
      result = c->arena + c->arena_top;
      c->arena_top += nbytes;
    }
    LIBXS_LOCK_RELEASE(LIBXS_LOCK, &c->lock.lock);
  }
  return result;
}


static void libxstream_mps_arena_free(libxstream_mps_client_t* c, char* p, size_t size)
{
  if (NULL != p) {
    const size_t k = libxstream_mps_class(size);
    LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, &c->lock.lock);
    memcpy(p, &c->arena_free[k], sizeof(char*));
    c->arena_free[k] = p;
    LIBXS_LOCK_RELEASE(LIBXS_LOCK, &c->lock.lock);
  }
}


/* Offset (plus one) of a range in shared memory, or zero if it lies elsewhere. */
static uint64_t libxstream_mps_arena_offset(const libxstream_mps_client_t* c, const void* p, size_t size)
{
  const char* const q = (const char*)p;
  return (c->arena <= q && q + size <= c->arena + c->arena_size) ? (uint64_t)(q - c->arena) + 1 : 0;
}


static libxstream_mps_obj_t* libxstream_mps_new(libxstream_mps_client_t* c, int kind, libxstream_mps_obj_t* context)
{
  libxstream_mps_obj_t* result;
  LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, &c->lock.lock);
  result = c->free_head;
  if (NULL != result && 0 != __atomic_load_n(c->released + (result - c->objects), __ATOMIC_ACQUIRE)) {
    c->free_head = result->owner;
    if (NULL == c->free_head) c->free_tail = NULL;
    __atomic_store_n(c->released + (result - c->objects), 0, __ATOMIC_RELAXED);
  }
  else if (c->nobjects < c->capacity) result = c->objects + c->nobjects++;
  else result = NULL;
  LIBXS_LOCK_RELEASE(LIBXS_LOCK, &c->lock.lock);
  if (NULL != result) { /* the slot's next generation tells the object from the slot's earlier ones */
    memset(result, 0, sizeof(*result));
    result->gen = ++c->gens[result - c->objects];
    result->dispatch = &libxstream_mps_dispatch;
    result->context = context;
    result->kind = kind;
    result->refs = 1;
  }
  return result;
}


/* A slot is reused once released by the server (confirmed: there is nothing to release). */
static void libxstream_mps_delete(libxstream_mps_client_t* c, libxstream_mps_obj_t* obj, int confirmed)
{
  LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, &c->lock.lock);
  obj->kind = 0;
  obj->owner = NULL;
  if (0 != confirmed) __atomic_store_n(c->released + (obj - c->objects), 1, __ATOMIC_RELAXED);
  if (NULL != c->free_tail) c->free_tail->owner = obj;
  else c->free_head = obj;
  c->free_tail = obj;
  LIBXS_LOCK_RELEASE(LIBXS_LOCK, &c->lock.lock);
}


/* A kernel argument is a buffer if its bytes are the address of one: objects come from one pool. */
static libxstream_mps_obj_t* libxstream_mps_lookup(const libxstream_mps_client_t* c, const void* value)
{
  libxstream_mps_obj_t* result = NULL;
  char* p;
  memcpy(&p, value, sizeof(p));
  if ((const char*)c->objects <= p && p < (const char*)(c->objects + c->nobjects)
    && 0 == (size_t)(p - (const char*)c->objects) % sizeof(libxstream_mps_obj_t)
    && LIBXSTREAM_MPS_KIND_MEM == ((const libxstream_mps_obj_t*)p)->kind)
  {
    result = (libxstream_mps_obj_t*)(void*)p;
  }
  return result;
}


/* Path of a file relative to the library's real directory, symbolic links resolved. */
static int libxstream_mps_beside(const char file[], char path[], size_t size)
{
  int result = EXIT_FAILURE;
#   if defined(LIBXS_INTERCEPT_DYNAMIC)
  union { int (*fn)(const char[], char[], size_t); const void* ptr; } self;
  char real[PATH_MAX];
  Dl_info info;
  self.fn = libxstream_mps_beside;
  if (0 != dladdr(self.ptr, &info) && NULL != info.dli_fname && NULL != realpath(info.dli_fname, real)) {
    char* const slash = strrchr(real, '/');
    if (NULL != slash) {
      *slash = '\0';
      if (0 < LIBXS_SNPRINTF(path, size, "%s/%s", real, file) && 0 == access(path, R_OK)) result = EXIT_SUCCESS;
    }
  }
#   else
  LIBXS_UNUSED(file);
  LIBXS_UNUSED(path);
  LIBXS_UNUSED(size);
#   endif
  return result;
}


/**
 * The server leaves the launcher's session and output: a launcher otherwise waits
 * for the pipes it holds open, and signals it along with the job's processes.
 */
static void libxstream_mps_spawn(int verbosity)
{
  const char* const env_server = getenv("LIBXSTREAM_MPS_SERVER");
  char path[PATH_MAX], *server = NULL;
  int search = 0;
  if (NULL != env_server) {
    if (0 < LIBXS_SNPRINTF(path, sizeof(path), "%s", env_server)) server = path;
  }
  else if (EXIT_SUCCESS == libxstream_mps_beside(LIBXSTREAM_MPS_BINDIR "/libxstream_mpsd", path, sizeof(path))
    || EXIT_SUCCESS == libxstream_mps_beside("libxstream_mpsd", path, sizeof(path)))
  {
    server = path; /* installed, or beside the library in a build tree */
  }
  else if (0 < LIBXS_SNPRINTF(path, sizeof(path), "libxstream_mpsd")) {
    server = path; /* a command found by PATH */
    search = 1;
  }
  if (NULL != server) {
    posix_spawn_file_actions_t actions;
    posix_spawnattr_t attr;
    char detach[] = "--detach", *argv[3];
    DIR* const fds = opendir("/proc/self/fd");
    pid_t pid;
    argv[0] = server;
    argv[1] = detach;
    argv[2] = NULL;
    posix_spawn_file_actions_init(&actions);
    posix_spawnattr_init(&attr);
    posix_spawn_file_actions_addopen(&actions, STDIN_FILENO, "/dev/null", O_RDONLY, 0);
    posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, "/dev/null", O_WRONLY, 0);
    if (0 == verbosity) posix_spawn_file_actions_addopen(&actions, STDERR_FILENO, "/dev/null", O_WRONLY, 0);
    if (NULL != fds) { /* the launcher waits for its pipes to close, hence the server inherits none */
      const struct dirent* entry;
      while (NULL != (entry = readdir(fds))) {
        const int fd = atoi(entry->d_name);
        if (STDERR_FILENO < fd && fd != dirfd(fds)) posix_spawn_file_actions_addclose(&actions, fd);
      }
      closedir(fds);
    }
#   if defined(POSIX_SPAWN_SETSID)
    posix_spawnattr_setflags(&attr, POSIX_SPAWN_SETSID);
#   endif
    if (0 != (0 == search ? posix_spawn(&pid, server, &actions, &attr, argv, environ)
                          : posix_spawnp(&pid, server, &actions, &attr, argv, environ)))
    {
      fprintf(stderr, "ERROR LIBXSTREAM-MPS: cannot start %s\n", server);
    }
    else { /* collect the parent of the detached server: no process is left behind (bounded, if it serves itself) */
      int attempt = 0;
      while (0 == waitpid(pid, NULL, WNOHANG) && 100 > attempt++) usleep(10000);
    }
    posix_spawnattr_destroy(&attr);
    posix_spawn_file_actions_destroy(&actions);
  }
  else fprintf(stderr, "ERROR LIBXSTREAM-MPS: server not found (LIBXSTREAM_MPS_SERVER)\n");
}


static int libxstream_mps_hello(libxstream_mps_client_t* c, int fd, int main)
{
  libxstream_mps_msg_t msg;
  libxstream_mps_reply_t reply;
  struct iovec iov;
  int result;
  memset(&msg, 0, sizeof(msg));
  msg.op = LIBXSTREAM_MPS_HELLO;
  msg.arg[0] = c->token;
  msg.arg[1] = (uint64_t)getpid();
  msg.arg[2] = c->arena_size;
  msg.arg[3] = c->capacity;
  msg.arg[4] = (uint64_t)main;
  iov.iov_base = &msg;
  iov.iov_len = sizeof(msg);
  result = libxstream_mps_send(fd, &iov, 1, 0 != main ? c->memfd : -1);
  if (EXIT_SUCCESS == result) result = libxstream_mps_recv(fd, &reply, sizeof(reply));
  if (EXIT_SUCCESS == result && CL_SUCCESS != reply.result) result = EXIT_FAILURE;
  return result;
}


/* The first connection starts the server if there is none, and waits for it to come up. */
static int libxstream_mps_open(libxstream_mps_client_t* c, int main)
{
  const int nattempts = (0 != main ? LIBXSTREAM_MPS_CONNECT * 100 : 1);
  int result = -1, attempt = 0;
  for (; attempt < nattempts && 0 > result; ++attempt) {
    const int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (0 <= fd) {
      struct sockaddr_un addr;
      memset(&addr, 0, sizeof(addr));
      addr.sun_family = AF_UNIX;
      memcpy(addr.sun_path, c->path, sizeof(addr.sun_path)); /* same size, terminated */
      if (0 == connect(fd, (struct sockaddr*)&addr, sizeof(addr)) && EXIT_SUCCESS == libxstream_mps_hello(c, fd, main)) {
        result = fd;
      }
      else close(fd);
    }
    if (0 > result && attempt + 1 < nattempts) {
      /* again and again: a server losing the election exits, and one may be shutting down */
      if (0 == (attempt % 50)) libxstream_mps_spawn(c->verbosity);
      usleep(10000);
    }
  }
  return result;
}


static libxstream_mps_client_t* libxstream_mps_start(void)
{
  libxstream_mps_client_t* c = (libxstream_mps_client_t*)calloc(1, sizeof(libxstream_mps_client_t));
  if (NULL != c) {
    const char *const env_arena = getenv("LIBXSTREAM_MPS_ARENA"), *const env_verbose = getenv("LIBXSTREAM_VERBOSE");
    const int arena_gb = (NULL != env_arena ? atoi(env_arena) : LIBXSTREAM_MPS_ARENA);
    char name[64];
    int i, fd = -1;
    LIBXS_LOCK_ATTR_TYPE(LIBXS_LOCK) attr;
    LIBXS_LOCK_ATTR_INIT(LIBXS_LOCK, &attr);
    LIBXS_LOCK_INIT(LIBXS_LOCK, &c->lock.lock, &attr);
    LIBXS_LOCK_INIT(LIBXS_LOCK, &c->order.lock, &attr);
    for (i = 0; i < LIBXSTREAM_MPS_MAXNCONN; ++i) {
      LIBXS_LOCK_INIT(LIBXS_LOCK, &c->conn_lock[i].lock, &attr);
      c->conn[i] = -1;
    }
    LIBXS_LOCK_ATTR_DESTROY(LIBXS_LOCK, &attr);
    for (i = 0; i < LIBXSTREAM_MAXNDEVS; ++i) {
      c->devices[i].dispatch = &libxstream_mps_dispatch;
      c->devices[i].kind = LIBXSTREAM_MPS_KIND_DEVICE;
      c->devices[i].refs = 1;
    }
    c->verbosity = (NULL != env_verbose ? atoi(env_verbose) : 0);
    c->token = ((uint64_t)getpid() << 32) ^ (uint64_t)time(NULL) ^ (uint64_t)(uintptr_t)c;
    c->arena_size = (size_t)LIBXS_MAX(arena_gb, 1) << 30;
    c->capacity = LIBXSTREAM_MPS_CAPACITY;
    libxstream_mps_paths(name, sizeof(name), c->path, sizeof(c->path));
    c->memfd = (int)syscall(SYS_memfd_create, "libxstream-mps", 1u /*MFD_CLOEXEC*/);
    if (0 <= c->memfd && 0 == ftruncate(c->memfd, (off_t)c->arena_size)) {
      c->arena = (char*)mmap(NULL, c->arena_size, PROT_READ | PROT_WRITE, MAP_SHARED, c->memfd, 0);
      if (MAP_FAILED == c->arena) c->arena = NULL;
    }
    c->objects = (libxstream_mps_obj_t*)mmap(NULL, c->capacity * sizeof(libxstream_mps_obj_t),
      PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (MAP_FAILED == c->objects) c->objects = NULL;
    c->gens = (unsigned int*)calloc(c->capacity, sizeof(unsigned int));
    c->cache = libxs_registry_create();
    if (NULL != c->arena) { /* release confirmations first, the rest is allocated */
      c->released = (uint32_t*)(void*)c->arena;
      c->arena_top = LIBXS_UP2(sizeof(uint32_t) * c->capacity, LIBXSTREAM_MPS_PAGE);
    }
    if (NULL != c->arena && NULL != c->objects && NULL != c->gens && NULL != c->cache) fd = libxstream_mps_open(c, 1);
    if (0 <= fd) {
      c->conn[libxs_tid() % LIBXSTREAM_MPS_MAXNCONN] = fd;
      if (0 != c->verbosity) fprintf(stderr, "INFO LIBXSTREAM-MPS: connected to %s\n", c->path);
    }
    else {
      fprintf(stderr, "ERROR LIBXSTREAM-MPS: cannot connect to %s\n", c->path);
      if (NULL != c->objects) munmap(c->objects, c->capacity * sizeof(libxstream_mps_obj_t));
      if (NULL != c->arena) munmap(c->arena, c->arena_size);
      if (0 <= c->memfd) close(c->memfd);
      libxs_registry_destroy(c->cache);
      free(c->gens);
      free(c);
      c = NULL;
    }
  }
  return c;
}


static void libxstream_mps_report(void)
{
  const libxstream_mps_client_t* const c = libxstream_mps_client;
  if (NULL != c && 2 <= c->verbosity) {
    fprintf(stderr, "INFO LIBXSTREAM-MPS: process %i cross-thread dependencies context=%lu queue=%lu mem=%lu program=%lu"
      " kernel=%lu event=%lu\n", (int)getpid(), (unsigned long)c->cross[LIBXSTREAM_MPS_KIND_CONTEXT],
      (unsigned long)c->cross[LIBXSTREAM_MPS_KIND_QUEUE], (unsigned long)c->cross[LIBXSTREAM_MPS_KIND_MEM],
      (unsigned long)c->cross[LIBXSTREAM_MPS_KIND_PROGRAM], (unsigned long)c->cross[LIBXSTREAM_MPS_KIND_KERNEL],
      (unsigned long)c->cross[LIBXSTREAM_MPS_KIND_EVENT]);
  }
}


static libxstream_mps_client_t* libxstream_mps_get(void)
{
  libxstream_mps_client_t* result = __atomic_load_n(&libxstream_mps_client, __ATOMIC_ACQUIRE);
  if (NULL == result) {
    pthread_mutex_lock(&libxstream_mps_lock);
    result = libxstream_mps_client;
    if (NULL == result && 0 == libxstream_mps_failed) {
      result = libxstream_mps_start();
      if (NULL != result) {
        __atomic_store_n(&libxstream_mps_client, result, __ATOMIC_RELEASE);
        if (2 <= result->verbosity) atexit(libxstream_mps_report);
      }
      else libxstream_mps_failed = 1;
    }
    pthread_mutex_unlock(&libxstream_mps_lock);
  }
  return result;
}


static void libxstream_mps_dep(uint64_t value, uint64_t tag, uint64_t seq, uint64_t deps[], uint64_t* ndeps)
{
  if (0 != value && seq + 1 != value && LIBXSTREAM_MPS_FENCE != *ndeps) {
    if (LIBXSTREAM_MPS_NDEPS > *ndeps) deps[(*ndeps)++] = value | tag;
    else *ndeps = LIBXSTREAM_MPS_FENCE; /* too many: after everything issued before */
  }
}


/**
 * A queue orders every command in issue order. Any other object only needs to exist:
 * uses do not wait for one another, order between queues is the application's, by
 * events. A kernel's launches carry its arguments, hence no launch waits for another.
 */
static void libxstream_mps_use(libxstream_mps_obj_t* obj, uint64_t seq, uint64_t deps[], uint64_t* ndeps)
{
  if (NULL != obj) {
    const unsigned int tid = libxs_tid();
    const int chain = (LIBXSTREAM_MPS_KIND_QUEUE == obj->kind);
    const int cross = (0 != obj->last && tid != obj->tid), verbose = (2 <= libxstream_mps_client->verbosity);
    if (0 != cross && 0 != chain && 0 != verbose) ++libxstream_mps_client->cross[obj->kind & 15];
    obj->tid = tid;
    libxstream_mps_dep(0 != chain ? obj->last : obj->create, 0 != verbose ? LIBXSTREAM_MPS_TAG(obj->kind, cross) : 0,
      seq, deps, ndeps);
    obj->last = seq + 1;
  }
}


static void libxstream_mps_created(libxstream_mps_obj_t* obj, uint64_t seq)
{
  if (NULL != obj) {
    __atomic_store_n(&obj->create, seq + 1, __ATOMIC_RELEASE);
    obj->last = seq + 1;
  }
}


/**
 * Dependencies of a command, read from the command itself. A launch uses the buffers
 * set as the kernel's arguments, so that releasing one waits for the launch: a release
 * carries the object's last use, which the server waits for before releasing. Called
 * with the order lock held, which keeps every dependency on a smaller sequence number.
 */
static uint64_t libxstream_mps_order(libxstream_mps_client_t* c, libxstream_mps_msg_t* msg, const void* p1,
  uint64_t deps[])
{
  const uint64_t* const a = msg->arg;
  const uint64_t seq = msg->seq;
  uint64_t ndeps = 0, i;
#   define LIBXSTREAM_MPS_OBJID(ID) (0 != (ID) && (ID) <= c->capacity ? (c->objects + (ID) - 1) : NULL)
#   define LIBXSTREAM_MPS_USE(ID) libxstream_mps_use(LIBXSTREAM_MPS_OBJID(ID), seq, deps, &ndeps)
  switch (msg->op) {
    case LIBXSTREAM_MPS_CONTEXT: libxstream_mps_created(LIBXSTREAM_MPS_OBJID(a[0]), seq); break;
    case LIBXSTREAM_MPS_QUEUE:
    case LIBXSTREAM_MPS_BUFFER:
    case LIBXSTREAM_MPS_SUBBUFFER:
    case LIBXSTREAM_MPS_PROGRAM_SOURCE:
    case LIBXSTREAM_MPS_PROGRAM_BINARY:
    case LIBXSTREAM_MPS_KERNEL: {
      LIBXSTREAM_MPS_USE(a[1]);
      libxstream_mps_created(LIBXSTREAM_MPS_OBJID(a[0]), seq);
    } break;
    case LIBXSTREAM_MPS_QUEUE_INFO:
    case LIBXSTREAM_MPS_BUILD:
    case LIBXSTREAM_MPS_PROGRAM_INFO:
    case LIBXSTREAM_MPS_BUILD_INFO:
    case LIBXSTREAM_MPS_KERNEL_INFO:
    case LIBXSTREAM_MPS_KERNEL_WGINFO:
    case LIBXSTREAM_MPS_FINISH:
    case LIBXSTREAM_MPS_FLUSH: LIBXSTREAM_MPS_USE(a[0]); break;
    case LIBXSTREAM_MPS_RELEASE: { /* no dependency: deferred by the server until the object's last use ran */
      const libxstream_mps_obj_t* const obj = LIBXSTREAM_MPS_OBJID(a[1]);
      msg->arg[2] = (NULL != obj ? obj->last : 0);
    } break;
    case LIBXSTREAM_MPS_WRITE:
    case LIBXSTREAM_MPS_READ:
    case LIBXSTREAM_MPS_FILL:
    case LIBXSTREAM_MPS_MAP:
    case LIBXSTREAM_MPS_UNMAP:
    case LIBXSTREAM_MPS_COPY:
    case LIBXSTREAM_MPS_NDRANGE:
    case LIBXSTREAM_MPS_MARKER: {
      LIBXSTREAM_MPS_USE(a[0]);
      if (LIBXSTREAM_MPS_MARKER != msg->op) LIBXSTREAM_MPS_USE(a[1]);
      if (LIBXSTREAM_MPS_COPY == msg->op) LIBXSTREAM_MPS_USE(a[2]);
      if (LIBXSTREAM_MPS_NDRANGE == msg->op && 0 != a[1] && a[1] <= c->capacity) {
        const libxstream_mps_obj_t* const kernel = c->objects + a[1] - 1;
        for (i = 0; i < kernel->size; ++i) {
          if (NULL != kernel->args[i] && LIBXSTREAM_MPS_KIND_MEM == kernel->args[i]->kind) {
            libxstream_mps_use(kernel->args[i], seq, deps, &ndeps);
          }
        }
      }
      for (i = 0; i < a[6]; ++i) LIBXSTREAM_MPS_USE(((const uint64_t*)p1)[i]);
      libxstream_mps_created(LIBXSTREAM_MPS_OBJID(a[7]), seq);
    } break;
    case LIBXSTREAM_MPS_WAIT: {
      for (i = 0; i < a[6]; ++i) LIBXSTREAM_MPS_USE(((const uint64_t*)p1)[i]);
    } break;
    default: break; /* platform and device information */
  }
#   undef LIBXSTREAM_MPS_USE
#   undef LIBXSTREAM_MPS_OBJID
  return ndeps;
}


/**
 * Sends a command and its payload (two pieces), and receives the answer if one is
 * asked for. The sequence number is taken under the connection's lock, hence the
 * order on a connection is the order of numbers, which the server executes in.
 */
static cl_int libxstream_mps_call(libxstream_mps_client_t* c, libxstream_mps_msg_t* msg,
  const void* p1, size_t n1, const void* p2, size_t n2, libxstream_mps_reply_t* reply, void* data, size_t maxdata)
{
  const unsigned int tid = libxs_tid() % LIBXSTREAM_MPS_MAXNCONN;
  cl_int result = CL_SUCCESS;
  int fd;
  LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, &c->conn_lock[tid].lock);
  fd = c->conn[tid];
  if (0 > fd) {
    fd = libxstream_mps_open(c, 0);
    c->conn[tid] = fd;
  }
  if (0 <= fd) {
    union { const void* in; void* out; } piece;
    uint64_t deps[LIBXSTREAM_MPS_NDEPS];
    struct iovec iov[4];
    int niov = 1;
    msg->nbytes = (uint64_t)(n1 + n2);
    msg->ndeps = 0;
    if (LIBXSTREAM_MPS_EVENT_STATUS != msg->op) { /* number and dependencies together, see libxstream_mps_order */
      LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, &c->order.lock);
      msg->seq = c->seq++;
      msg->ndeps = libxstream_mps_order(c, msg, p1, deps);
      LIBXS_LOCK_RELEASE(LIBXS_LOCK, &c->order.lock);
    }
    else msg->seq = LIBXSTREAM_MPS_UNORDERED;
    iov[0].iov_base = msg;
    iov[0].iov_len = sizeof(*msg);
    if (0 != n1) {
      piece.in = p1;
      iov[niov].iov_base = piece.out;
      iov[niov++].iov_len = n1;
    }
    if (0 != n2) {
      piece.in = p2;
      iov[niov].iov_base = piece.out;
      iov[niov++].iov_len = n2;
    }
    if (0 != msg->ndeps && LIBXSTREAM_MPS_FENCE != msg->ndeps) {
      iov[niov].iov_base = deps;
      iov[niov++].iov_len = sizeof(uint64_t) * (size_t)msg->ndeps;
    }
    if (EXIT_SUCCESS != libxstream_mps_send(fd, iov, niov, -1)) result = CL_OUT_OF_RESOURCES;
    else if (NULL != reply) {
      if (EXIT_SUCCESS == libxstream_mps_recv(fd, reply, sizeof(*reply))) {
        const size_t n = LIBXS_MIN((size_t)reply->nbytes, maxdata);
        size_t rest = (size_t)reply->nbytes - n;
        char scratch[256];
        result = (cl_int)reply->result;
        if (0 != n && EXIT_SUCCESS != libxstream_mps_recv(fd, data, n)) result = CL_OUT_OF_RESOURCES;
        while (0 != rest && CL_OUT_OF_RESOURCES != result) { /* more than asked for */
          const size_t m = LIBXS_MIN(rest, sizeof(scratch));
          if (EXIT_SUCCESS != libxstream_mps_recv(fd, scratch, m)) result = CL_OUT_OF_RESOURCES;
          rest -= m;
        }
      }
      else result = CL_OUT_OF_RESOURCES;
    }
  }
  else result = CL_OUT_OF_RESOURCES;
  LIBXS_LOCK_RELEASE(LIBXS_LOCK, &c->conn_lock[tid].lock);
  if (CL_OUT_OF_RESOURCES == result && 0 > fd) {
    fprintf(stderr, "ERROR LIBXSTREAM-MPS: lost connection to %s\n", c->path);
  }
  return result;
}


static void libxstream_mps_init(libxstream_mps_msg_t* msg, uint64_t op)
{
  memset(msg, 0, sizeof(*msg));
  msg->op = op;
}


static cl_int libxstream_mps_query(libxstream_mps_client_t* c, libxstream_mps_msg_t* msg,
  size_t size, void* value, size_t* size_ret)
{
  libxstream_mps_reply_t reply;
  cl_int result;
  msg->arg[LIBXSTREAM_MPS_INFO] = (NULL != value ? size : 0);
  result = libxstream_mps_call(c, msg, NULL, 0, NULL, 0, &reply, value, NULL != value ? size : 0);
  if (CL_SUCCESS == result && NULL != size_ret) *size_ret = (size_t)reply.arg0;
  if (CL_SUCCESS == result && NULL != value && size < (size_t)reply.arg0) result = CL_INVALID_VALUE;
  return result;
}


/* Platform (device < 0) and device information does not change: answered once by the server. */
static cl_int libxstream_mps_cached(libxstream_mps_client_t* c, int device, uint64_t op, uint64_t param,
  size_t size, void* value, size_t* size_ret)
{
  libxs_lock_t* const lock = libxs_registry_lock(c->cache);
  libxstream_mps_key_t key;
  cl_int result = CL_SUCCESS;
  size_t nbytes;
  key.device = device;
  key.param = param;
  nbytes = libxs_registry_value_size(c->cache, &key, sizeof(key), lock);
  if (0 != nbytes) { /* entries are neither removed nor resized */
    if (NULL != value) {
      if (nbytes <= size) libxs_registry_get_copy(c->cache, &key, sizeof(key), value, nbytes, lock);
      else result = CL_INVALID_VALUE;
    }
    if (NULL != size_ret) *size_ret = nbytes;
  }
  else {
    libxstream_mps_msg_t msg;
    libxstream_mps_init(&msg, op);
    msg.arg[0] = (uint64_t)device;
    msg.arg[1] = param;
    result = libxstream_mps_query(c, &msg, size, value, &nbytes);
    if (NULL != size_ret) *size_ret = nbytes;
    if (CL_SUCCESS == result && NULL != value && 0 != nbytes && nbytes <= size) {
      libxs_registry_set(c->cache, &key, sizeof(key), value, nbytes, lock);
    }
  }
  return result;
}


static cl_int libxstream_mps_retain(void* handle, int kind)
{
  cl_int result = CL_SUCCESS;
  if (LIBXSTREAM_MPS_IS(handle, kind)) LIBXS_ATOMIC_ADD_FETCH(&LIBXSTREAM_MPS_OBJ(handle)->refs, 1, LIBXS_ATOMIC_SEQ_CST);
  else result = libxstream_mps_invalid(kind);
  return result;
}


/**
 * The server holds one reference per object, given up with the application's last one.
 * A kernel holds its program and a sub-buffer its buffer, as the OpenCL objects do: code may
 * release a program once it has the kernel, and later create further kernels from the program
 * the kernel answers (CL_KERNEL_PROGRAM). The slot of a released object is reused otherwise.
 * An event holds its queue, on which the server may still enqueue its marker (lazy events).
 */
static cl_int libxstream_mps_release(void* handle, int kind)
{
  libxstream_mps_obj_t* const obj = LIBXSTREAM_MPS_OBJ(handle);
  cl_int result = CL_SUCCESS;
  if (LIBXSTREAM_MPS_IS(handle, kind)) {
    if (0 == LIBXS_ATOMIC_SUB_FETCH(&obj->refs, 1, LIBXS_ATOMIC_SEQ_CST)) {
      libxstream_mps_client_t* const c = libxstream_mps_client;
      libxstream_mps_obj_t* const owner = obj->owner;
      libxstream_mps_msg_t msg;
      libxstream_mps_init(&msg, LIBXSTREAM_MPS_RELEASE);
      msg.arg[0] = (uint64_t)kind;
      msg.arg[1] = LIBXSTREAM_MPS_ID(c, obj);
      result = libxstream_mps_call(c, &msg, NULL, 0, NULL, 0, NULL, NULL, 0);
      if (LIBXSTREAM_MPS_KIND_MEM == kind && NULL == obj->owner) {
        libxstream_mps_arena_free(c, obj->shadow, obj->size);
      }
      if (LIBXSTREAM_MPS_KIND_KERNEL == kind) {
        size_t i;
        for (i = 0; NULL != obj->bigs && i < obj->npargs; ++i) free(obj->bigs[i]);
        free(obj->args);
        free(obj->bigs);
        free(obj->pargs);
      }
      libxstream_mps_delete(c, obj, 0);
      if (LIBXSTREAM_MPS_KIND_KERNEL == kind && NULL != owner) libxstream_mps_release(owner, LIBXSTREAM_MPS_KIND_PROGRAM);
      else if (LIBXSTREAM_MPS_KIND_MEM == kind && NULL != owner) libxstream_mps_release(owner, LIBXSTREAM_MPS_KIND_MEM);
      else if (LIBXSTREAM_MPS_KIND_EVENT == kind && NULL != owner) libxstream_mps_release(owner, LIBXSTREAM_MPS_KIND_QUEUE);
    }
  }
  else result = libxstream_mps_invalid(kind);
  return result;
}


/**
 * On an in-order queue, a command known to be complete completes every command issued
 * on the queue before it. Sequence numbers are issue order (libxstream_mps_order).
 */
static void libxstream_mps_complete(libxstream_mps_obj_t* queue, uint64_t upto)
{
  if (NULL != queue && 0 == (queue->flags & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)) {
    uint64_t done = __atomic_load_n(&queue->done, __ATOMIC_ACQUIRE);
    while (done < upto && !__atomic_compare_exchange_n(&queue->done, &done, upto, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE));
  }
}


/* An event is complete without asking the server if a later command on its queue completed. */
static int libxstream_mps_known(const libxstream_mps_obj_t* event)
{
  const libxstream_mps_obj_t* const queue = event->owner;
  const uint64_t create = __atomic_load_n(&event->create, __ATOMIC_ACQUIRE);
  return (NULL != queue && LIBXSTREAM_MPS_KIND_QUEUE == queue->kind && event->ogen == queue->gen && 0 != create
          && create <= __atomic_load_n(&queue->done, __ATOMIC_ACQUIRE));
}


static void libxstream_mps_completed(libxstream_mps_obj_t* event)
{
  if (NULL != event->owner && event->ogen == event->owner->gen) {
    libxstream_mps_complete(event->owner, __atomic_load_n(&event->create, __ATOMIC_ACQUIRE));
  }
}


/**
 * Common part of an enqueue: the queue, the wait list as identifiers (ids is the
 * local buffer or one allocated here), and the event object if one is asked for.
 */
static cl_int libxstream_mps_prepare(libxstream_mps_msg_t* msg, uint64_t op, cl_command_queue queue,
  cl_uint nwait, const cl_event* wait, cl_event* event, cl_command_type type, uint64_t local[], uint64_t** ids)
{
  libxstream_mps_client_t* const c = libxstream_mps_client;
  cl_int result = CL_SUCCESS;
  cl_uint i;
  libxstream_mps_init(msg, op);
  *ids = local;
  if (!LIBXSTREAM_MPS_IS(queue, LIBXSTREAM_MPS_KIND_QUEUE)) result = CL_INVALID_COMMAND_QUEUE;
  else if ((0 != nwait && NULL == wait) || (0 == nwait && NULL != wait)) result = CL_INVALID_EVENT_WAIT_LIST;
  else if (LIBXSTREAM_MPS_NLOCAL < nwait) {
    *ids = (uint64_t*)malloc(sizeof(uint64_t) * nwait);
    if (NULL == *ids) result = CL_OUT_OF_HOST_MEMORY;
  }
  for (i = 0; i < nwait && CL_SUCCESS == result; ++i) {
    if (LIBXSTREAM_MPS_IS(wait[i], LIBXSTREAM_MPS_KIND_EVENT)) (*ids)[i] = LIBXSTREAM_MPS_ID(c, wait[i]);
    else result = CL_INVALID_EVENT_WAIT_LIST;
  }
  if (CL_SUCCESS == result) {
    msg->arg[0] = LIBXSTREAM_MPS_ID(c, queue);
    msg->arg[6] = nwait;
    if (NULL != event) {
      libxstream_mps_obj_t* const obj = libxstream_mps_new(c, LIBXSTREAM_MPS_KIND_EVENT, LIBXSTREAM_MPS_OBJ(queue)->context);
      if (NULL != obj) {
        obj->owner = LIBXSTREAM_MPS_OBJ(queue);
        obj->ogen = LIBXSTREAM_MPS_OBJ(queue)->gen;
        LIBXS_ATOMIC_ADD_FETCH(&obj->owner->refs, 1, LIBXS_ATOMIC_SEQ_CST); /* see libxstream_mps_release */
        obj->flags = type;
        msg->arg[7] = LIBXSTREAM_MPS_ID(c, obj);
        msg->arg[LIBXSTREAM_MPS_GEN] = obj->gen;
        *event = (cl_event)obj;
      }
      else result = CL_OUT_OF_HOST_MEMORY;
    }
  }
  return result;
}


static void libxstream_mps_done(uint64_t local[], uint64_t* ids)
{
  if (local != ids) free(ids);
}


static cl_int CL_API_CALL libxstream_mps_platform_ids(cl_uint num_entries, cl_platform_id* platforms, cl_uint* num_platforms)
{
  cl_int result = CL_SUCCESS;
  if ((0 == num_entries && NULL != platforms) || (NULL == platforms && NULL == num_platforms)) result = CL_INVALID_VALUE;
  else {
    if (NULL != platforms) platforms[0] = (cl_platform_id)&libxstream_mps_platform;
    if (NULL != num_platforms) *num_platforms = 1;
  }
  return result;
}


static cl_int CL_API_CALL libxstream_mps_platform_info(cl_platform_id platform, cl_platform_info param,
  size_t size, void* value, size_t* size_ret)
{
  cl_int result = CL_SUCCESS;
  if (CL_PLATFORM_ICD_SUFFIX_KHR == param) { /* asked by the loader before anything else */
    result = libxstream_mps_answer(LIBXSTREAM_MPS_SUFFIX, sizeof(LIBXSTREAM_MPS_SUFFIX), size, value, size_ret);
  }
  else if (NULL != platform && (cl_platform_id)&libxstream_mps_platform != platform) result = CL_INVALID_PLATFORM;
  else {
    libxstream_mps_client_t* const c = libxstream_mps_get();
    if (NULL == c) result = CL_PLATFORM_NOT_FOUND_KHR;
    else if (CL_PLATFORM_NAME == param) {
      char name[256 + sizeof(LIBXSTREAM_MPS_NAME)];
      size_t nbytes = 0;
      result = libxstream_mps_cached(c, -1, LIBXSTREAM_MPS_PLATFORM_INFO, param, 256, name, &nbytes);
      if (CL_SUCCESS == result) {
        nbytes = strlen(name);
        memcpy(name + nbytes, LIBXSTREAM_MPS_NAME, sizeof(LIBXSTREAM_MPS_NAME));
        result = libxstream_mps_answer(name, nbytes + sizeof(LIBXSTREAM_MPS_NAME), size, value, size_ret);
      }
    }
    else result = libxstream_mps_cached(c, -1, LIBXSTREAM_MPS_PLATFORM_INFO, param, size, value, size_ret);
  }
  return result;
}


static cl_int CL_API_CALL libxstream_mps_device_ids(cl_platform_id platform, cl_device_type type,
  cl_uint num_entries, cl_device_id* devices, cl_uint* num_devices)
{
  libxstream_mps_client_t* const c = libxstream_mps_get();
  cl_int result = CL_SUCCESS;
  if (NULL != platform && (cl_platform_id)&libxstream_mps_platform != platform) result = CL_INVALID_PLATFORM;
  else if ((0 == num_entries && NULL != devices) || (NULL == devices && NULL == num_devices)) result = CL_INVALID_VALUE;
  else if (NULL == c) result = CL_DEVICE_NOT_FOUND;
  else {
    uint32_t index[LIBXSTREAM_MAXNDEVS];
    libxstream_mps_reply_t reply;
    libxstream_mps_msg_t msg;
    libxstream_mps_init(&msg, LIBXSTREAM_MPS_DEVICE_IDS);
    msg.arg[0] = type;
    result = libxstream_mps_call(c, &msg, NULL, 0, NULL, 0, &reply, index, sizeof(index));
    if (CL_SUCCESS == result && 0 == reply.arg0) result = CL_DEVICE_NOT_FOUND;
    if (CL_SUCCESS == result) {
      const cl_uint n = (cl_uint)LIBXS_MIN(reply.arg0, (uint64_t)LIBXSTREAM_MAXNDEVS);
      cl_uint i;
      for (i = 0; NULL != devices && i < n && i < num_entries; ++i) devices[i] = (cl_device_id)(c->devices + index[i]);
      if (NULL != num_devices) *num_devices = n;
    }
  }
  return result;
}


static cl_int CL_API_CALL libxstream_mps_device_info(cl_device_id device, cl_device_info param,
  size_t size, void* value, size_t* size_ret)
{
  libxstream_mps_client_t* const c = libxstream_mps_client;
  cl_int result = CL_SUCCESS;
  if (!LIBXSTREAM_MPS_IS(device, LIBXSTREAM_MPS_KIND_DEVICE) || NULL == c) result = CL_INVALID_DEVICE;
  else if (CL_DEVICE_PLATFORM == param) {
    const cl_platform_id platform = (cl_platform_id)&libxstream_mps_platform;
    result = libxstream_mps_answer(&platform, sizeof(platform), size, value, size_ret);
  }
  else if (CL_DEVICE_PARENT_DEVICE == param) {
    const cl_device_id parent = NULL;
    result = libxstream_mps_answer(&parent, sizeof(parent), size, value, size_ret);
  }
  else if (CL_DEVICE_SVM_CAPABILITIES == param) {
    const cl_device_svm_capabilities svm = 0;
    result = libxstream_mps_answer(&svm, sizeof(svm), size, value, size_ret);
  }
  else {
    result = libxstream_mps_cached(c, (int)LIBXSTREAM_MPS_DEVID(c, device), LIBXSTREAM_MPS_DEVICE_INFO, param,
      size, value, size_ret);
  }
  return result;
}


static cl_int CL_API_CALL libxstream_mps_sub_devices(cl_device_id device, const cl_device_partition_property* properties,
  cl_uint num_devices, cl_device_id* devices, cl_uint* num_devices_ret)
{
  LIBXS_UNUSED(device);
  LIBXS_UNUSED(properties);
  LIBXS_UNUSED(num_devices);
  LIBXS_UNUSED(devices);
  LIBXS_UNUSED(num_devices_ret);
  return CL_INVALID_VALUE;
}


static cl_int CL_API_CALL libxstream_mps_device_keep(cl_device_id device)
{
  return LIBXSTREAM_MPS_IS(device, LIBXSTREAM_MPS_KIND_DEVICE) ? CL_SUCCESS : CL_INVALID_DEVICE;
}


static cl_context CL_API_CALL libxstream_mps_context(const cl_context_properties* properties, cl_uint num_devices,
  const cl_device_id* devices, void (CL_CALLBACK* notify)(const char*, const void*, size_t, void*), void* user_data,
  cl_int* errcode_ret)
{
  libxstream_mps_client_t* const c = libxstream_mps_get();
  libxstream_mps_obj_t* obj = NULL;
  cl_int result = CL_SUCCESS;
  uint64_t mask = 0;
  cl_uint i;
  LIBXS_UNUSED(properties);
  LIBXS_UNUSED(notify);
  LIBXS_UNUSED(user_data);
  if (NULL == c) result = CL_INVALID_PLATFORM;
  else if (0 == num_devices || NULL == devices) result = CL_INVALID_VALUE;
  for (i = 0; i < num_devices && CL_SUCCESS == result; ++i) {
    if (LIBXSTREAM_MPS_IS(devices[i], LIBXSTREAM_MPS_KIND_DEVICE)) mask |= (uint64_t)1 << LIBXSTREAM_MPS_DEVID(c, devices[i]);
    else result = CL_INVALID_DEVICE;
  }
  if (CL_SUCCESS == result) {
    obj = libxstream_mps_new(c, LIBXSTREAM_MPS_KIND_CONTEXT, NULL);
    if (NULL != obj) {
      libxstream_mps_reply_t reply;
      libxstream_mps_msg_t msg;
      obj->flags = mask;
      obj->device = LIBXSTREAM_MPS_OBJ(devices[0]);
      obj->size = num_devices;
      libxstream_mps_init(&msg, LIBXSTREAM_MPS_CONTEXT);
      msg.arg[0] = LIBXSTREAM_MPS_ID(c, obj);
      msg.arg[1] = mask;
      result = libxstream_mps_call(c, &msg, NULL, 0, NULL, 0, &reply, NULL, 0);
      if (CL_SUCCESS != result) {
        libxstream_mps_delete(c, obj, 1);
        obj = NULL;
      }
    }
    else result = CL_OUT_OF_HOST_MEMORY;
  }
  if (NULL != errcode_ret) *errcode_ret = result;
  return (cl_context)obj;
}


static cl_context CL_API_CALL libxstream_mps_context_type(const cl_context_properties* properties, cl_device_type type,
  void (CL_CALLBACK* notify)(const char*, const void*, size_t, void*), void* user_data, cl_int* errcode_ret)
{
  cl_device_id devices[LIBXSTREAM_MAXNDEVS];
  cl_uint ndevices = 0;
  cl_context result = NULL;
  const cl_int status = libxstream_mps_device_ids(NULL, type, LIBXSTREAM_MAXNDEVS, devices, &ndevices);
  if (CL_SUCCESS == status) result = libxstream_mps_context(properties, ndevices, devices, notify, user_data, errcode_ret);
  else if (NULL != errcode_ret) *errcode_ret = status;
  return result;
}


static cl_int CL_API_CALL libxstream_mps_context_retain(cl_context context)
{
  return libxstream_mps_retain(context, LIBXSTREAM_MPS_KIND_CONTEXT);
}


static cl_int CL_API_CALL libxstream_mps_context_release(cl_context context)
{
  return libxstream_mps_release(context, LIBXSTREAM_MPS_KIND_CONTEXT);
}


static cl_int CL_API_CALL libxstream_mps_context_info(cl_context context, cl_context_info param,
  size_t size, void* value, size_t* size_ret)
{
  libxstream_mps_client_t* const c = libxstream_mps_client;
  const libxstream_mps_obj_t* const obj = LIBXSTREAM_MPS_OBJ(context);
  cl_int result = CL_SUCCESS;
  if (!LIBXSTREAM_MPS_IS(context, LIBXSTREAM_MPS_KIND_CONTEXT)) result = CL_INVALID_CONTEXT;
  else switch (param) {
    case CL_CONTEXT_REFERENCE_COUNT: {
      const cl_uint refs = (cl_uint)obj->refs;
      result = libxstream_mps_answer(&refs, sizeof(refs), size, value, size_ret);
    } break;
    case CL_CONTEXT_NUM_DEVICES: {
      const cl_uint n = (cl_uint)obj->size;
      result = libxstream_mps_answer(&n, sizeof(n), size, value, size_ret);
    } break;
    case CL_CONTEXT_DEVICES: {
      cl_device_id devices[LIBXSTREAM_MAXNDEVS];
      size_t n = 0, i;
      for (i = 0; i < LIBXSTREAM_MAXNDEVS && i < 64; ++i) {
        if (0 != (obj->flags & ((uint64_t)1 << i))) devices[n++] = (cl_device_id)(c->devices + i);
      }
      result = libxstream_mps_answer(devices, n * sizeof(cl_device_id), size, value, size_ret);
    } break;
    case CL_CONTEXT_PROPERTIES: {
      cl_context_properties properties[3];
      properties[0] = CL_CONTEXT_PLATFORM;
      properties[1] = (cl_context_properties)&libxstream_mps_platform;
      properties[2] = 0;
      result = libxstream_mps_answer(properties, sizeof(properties), size, value, size_ret);
    } break;
    default: result = CL_INVALID_VALUE;
  }
  return result;
}


static cl_command_queue CL_API_CALL libxstream_mps_queue(cl_context context, cl_device_id device,
  const cl_queue_properties* properties, cl_int* errcode_ret)
{
  libxstream_mps_client_t* const c = libxstream_mps_client;
  libxstream_mps_obj_t* obj = NULL;
  cl_int result = CL_SUCCESS;
  if (!LIBXSTREAM_MPS_IS(context, LIBXSTREAM_MPS_KIND_CONTEXT)) result = CL_INVALID_CONTEXT;
  else if (!LIBXSTREAM_MPS_IS(device, LIBXSTREAM_MPS_KIND_DEVICE)) result = CL_INVALID_DEVICE;
  else {
    obj = libxstream_mps_new(c, LIBXSTREAM_MPS_KIND_QUEUE, LIBXSTREAM_MPS_OBJ(context));
    if (NULL != obj) {
      libxstream_mps_reply_t reply;
      libxstream_mps_msg_t msg;
      size_t n = 0;
      while (NULL != properties && 0 != properties[n]) { /* key-value pairs */
        if (CL_QUEUE_PROPERTIES == properties[n]) obj->flags = (uint64_t)properties[n + 1];
        n += 2;
      }
      obj->device = LIBXSTREAM_MPS_OBJ(device);
      libxstream_mps_init(&msg, LIBXSTREAM_MPS_QUEUE);
      msg.arg[0] = LIBXSTREAM_MPS_ID(c, obj);
      msg.arg[1] = LIBXSTREAM_MPS_ID(c, context);
      msg.arg[2] = LIBXSTREAM_MPS_DEVID(c, device);
      result = libxstream_mps_call(c, &msg, properties, NULL != properties ? (n + 1) * sizeof(cl_queue_properties) : 0,
        NULL, 0, &reply, NULL, 0);
      if (CL_SUCCESS != result) {
        libxstream_mps_delete(c, obj, 1);
        obj = NULL;
      }
    }
    else result = CL_OUT_OF_HOST_MEMORY;
  }
  if (NULL != errcode_ret) *errcode_ret = result;
  return (cl_command_queue)obj;
}


static cl_command_queue CL_API_CALL libxstream_mps_queue_legacy(cl_context context, cl_device_id device,
  cl_command_queue_properties properties, cl_int* errcode_ret)
{
  cl_queue_properties list[3];
  list[0] = CL_QUEUE_PROPERTIES;
  list[1] = (cl_queue_properties)properties;
  list[2] = 0;
  return libxstream_mps_queue(context, device, list, errcode_ret);
}


static cl_int CL_API_CALL libxstream_mps_queue_retain(cl_command_queue queue)
{
  return libxstream_mps_retain(queue, LIBXSTREAM_MPS_KIND_QUEUE);
}


static cl_int CL_API_CALL libxstream_mps_queue_release(cl_command_queue queue)
{
  return libxstream_mps_release(queue, LIBXSTREAM_MPS_KIND_QUEUE);
}


static cl_int CL_API_CALL libxstream_mps_queue_info(cl_command_queue queue, cl_command_queue_info param,
  size_t size, void* value, size_t* size_ret)
{
  const libxstream_mps_obj_t* const obj = LIBXSTREAM_MPS_OBJ(queue);
  cl_int result = CL_SUCCESS;
  if (!LIBXSTREAM_MPS_IS(queue, LIBXSTREAM_MPS_KIND_QUEUE)) result = CL_INVALID_COMMAND_QUEUE;
  else if (CL_QUEUE_CONTEXT == param) result = libxstream_mps_answer(&obj->context, sizeof(cl_context), size, value, size_ret);
  else if (CL_QUEUE_DEVICE == param) result = libxstream_mps_answer(&obj->device, sizeof(cl_device_id), size, value, size_ret);
  else if (CL_QUEUE_PROPERTIES == param) {
    const cl_command_queue_properties properties = (cl_command_queue_properties)obj->flags;
    result = libxstream_mps_answer(&properties, sizeof(properties), size, value, size_ret);
  }
  else if (CL_QUEUE_REFERENCE_COUNT == param) {
    const cl_uint refs = (cl_uint)obj->refs;
    result = libxstream_mps_answer(&refs, sizeof(refs), size, value, size_ret);
  }
  else {
    libxstream_mps_client_t* const c = libxstream_mps_client;
    libxstream_mps_msg_t msg;
    libxstream_mps_init(&msg, LIBXSTREAM_MPS_QUEUE_INFO);
    msg.arg[0] = LIBXSTREAM_MPS_ID(c, queue);
    msg.arg[1] = param;
    result = libxstream_mps_query(c, &msg, size, value, size_ret);
  }
  return result;
}


/**
 * Host memory the application obtains through a buffer (ALLOC_HOST_PTR, USE_HOST_PTR)
 * is created on shared memory: the server wraps it with USE_HOST_PTR, so the data a
 * transfer reads or writes never passes through the socket.
 */
static cl_mem CL_API_CALL libxstream_mps_buffer(cl_context context, cl_mem_flags flags, size_t size, void* host_ptr,
  cl_int* errcode_ret)
{
  libxstream_mps_client_t* const c = libxstream_mps_client;
  const cl_mem_flags host_flags = CL_MEM_USE_HOST_PTR | CL_MEM_COPY_HOST_PTR;
  libxstream_mps_obj_t* obj = NULL;
  cl_int result = CL_SUCCESS;
  if (!LIBXSTREAM_MPS_IS(context, LIBXSTREAM_MPS_KIND_CONTEXT)) result = CL_INVALID_CONTEXT;
  else if (0 == size) result = CL_INVALID_BUFFER_SIZE;
  else if ((0 != (flags & host_flags)) != (NULL != host_ptr)) result = CL_INVALID_HOST_PTR;
  else {
    obj = libxstream_mps_new(c, LIBXSTREAM_MPS_KIND_MEM, LIBXSTREAM_MPS_OBJ(context));
    if (NULL != obj) {
      libxstream_mps_reply_t reply;
      libxstream_mps_msg_t msg;
      const void* payload = NULL;
      size_t nbytes = 0;
      libxstream_mps_init(&msg, LIBXSTREAM_MPS_BUFFER);
      obj->size = size;
      obj->flags = flags;
      msg.arg[2] = flags;
      if (0 != (flags & (CL_MEM_ALLOC_HOST_PTR | CL_MEM_USE_HOST_PTR))) {
        obj->shadow = libxstream_mps_arena_alloc(c, size);
        if (NULL != obj->shadow) {
          if (NULL != host_ptr) memcpy(obj->shadow, host_ptr, size);
          obj->host = (0 != (flags & CL_MEM_USE_HOST_PTR) ? (char*)host_ptr : obj->shadow);
          msg.arg[2] = (flags & ~(cl_mem_flags)(CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR)) | CL_MEM_USE_HOST_PTR;
          msg.arg[4] = libxstream_mps_arena_offset(c, obj->shadow, size);
        }
        else result = CL_OUT_OF_HOST_MEMORY;
      }
      else if (0 != (flags & CL_MEM_COPY_HOST_PTR)) {
        payload = host_ptr;
        nbytes = size;
      }
      if (CL_SUCCESS == result) {
        msg.arg[0] = LIBXSTREAM_MPS_ID(c, obj);
        msg.arg[1] = LIBXSTREAM_MPS_ID(c, context);
        msg.arg[3] = size;
        result = libxstream_mps_call(c, &msg, payload, nbytes, NULL, 0, &reply, NULL, 0);
      }
      if (CL_SUCCESS != result) {
        libxstream_mps_arena_free(c, obj->shadow, size);
        libxstream_mps_delete(c, obj, 1);
        obj = NULL;
      }
    }
    else result = CL_OUT_OF_HOST_MEMORY;
  }
  if (NULL != errcode_ret) *errcode_ret = result;
  return (cl_mem)obj;
}


static cl_mem CL_API_CALL libxstream_mps_subbuffer(cl_mem buffer, cl_mem_flags flags, cl_buffer_create_type type,
  const void* info, cl_int* errcode_ret)
{
  libxstream_mps_client_t* const c = libxstream_mps_client;
  libxstream_mps_obj_t* const parent = LIBXSTREAM_MPS_OBJ(buffer);
  const cl_buffer_region* const region = (const cl_buffer_region*)info;
  libxstream_mps_obj_t* obj = NULL;
  cl_int result = CL_SUCCESS;
  if (!LIBXSTREAM_MPS_IS(buffer, LIBXSTREAM_MPS_KIND_MEM) || NULL != parent->owner) result = CL_INVALID_MEM_OBJECT;
  else if (CL_BUFFER_CREATE_TYPE_REGION != type || NULL == region) result = CL_INVALID_VALUE;
  else if (0 == region->size || parent->size < region->origin + region->size) result = CL_INVALID_VALUE;
  else {
    obj = libxstream_mps_new(c, LIBXSTREAM_MPS_KIND_MEM, parent->context);
    if (NULL != obj) {
      libxstream_mps_reply_t reply;
      libxstream_mps_msg_t msg;
      obj->owner = parent;
      obj->size = region->size;
      obj->offset = region->origin;
      obj->flags = (0 != flags ? flags : parent->flags);
      if (NULL != parent->shadow) obj->shadow = parent->shadow + region->origin;
      if (NULL != parent->host) obj->host = parent->host + region->origin;
      libxstream_mps_init(&msg, LIBXSTREAM_MPS_SUBBUFFER);
      msg.arg[0] = LIBXSTREAM_MPS_ID(c, obj);
      msg.arg[1] = LIBXSTREAM_MPS_ID(c, buffer);
      msg.arg[2] = flags;
      msg.arg[3] = region->origin;
      msg.arg[4] = region->size;
      result = libxstream_mps_call(c, &msg, NULL, 0, NULL, 0, &reply, NULL, 0);
      if (CL_SUCCESS == result) LIBXS_ATOMIC_ADD_FETCH(&parent->refs, 1, LIBXS_ATOMIC_SEQ_CST); /* see libxstream_mps_release */
      else {
        libxstream_mps_delete(c, obj, 1);
        obj = NULL;
      }
    }
    else result = CL_OUT_OF_HOST_MEMORY;
  }
  if (NULL != errcode_ret) *errcode_ret = result;
  return (cl_mem)obj;
}


static cl_int CL_API_CALL libxstream_mps_mem_retain(cl_mem memobj)
{
  return libxstream_mps_retain(memobj, LIBXSTREAM_MPS_KIND_MEM);
}


static cl_int CL_API_CALL libxstream_mps_mem_release(cl_mem memobj)
{
  return libxstream_mps_release(memobj, LIBXSTREAM_MPS_KIND_MEM);
}


static cl_int CL_API_CALL libxstream_mps_mem_info(cl_mem memobj, cl_mem_info param, size_t size, void* value,
  size_t* size_ret)
{
  const libxstream_mps_obj_t* const obj = LIBXSTREAM_MPS_OBJ(memobj);
  cl_int result = CL_SUCCESS;
  if (!LIBXSTREAM_MPS_IS(memobj, LIBXSTREAM_MPS_KIND_MEM)) result = CL_INVALID_MEM_OBJECT;
  else switch (param) {
    case CL_MEM_TYPE: {
      const cl_mem_object_type type = CL_MEM_OBJECT_BUFFER;
      result = libxstream_mps_answer(&type, sizeof(type), size, value, size_ret);
    } break;
    case CL_MEM_FLAGS: {
      const cl_mem_flags flags = (cl_mem_flags)obj->flags;
      result = libxstream_mps_answer(&flags, sizeof(flags), size, value, size_ret);
    } break;
    case CL_MEM_SIZE: result = libxstream_mps_answer(&obj->size, sizeof(size_t), size, value, size_ret); break;
    case CL_MEM_OFFSET: result = libxstream_mps_answer(&obj->offset, sizeof(size_t), size, value, size_ret); break;
    case CL_MEM_HOST_PTR: {
      const void* const host = (0 != (obj->flags & CL_MEM_USE_HOST_PTR) ? obj->host : NULL);
      result = libxstream_mps_answer(&host, sizeof(host), size, value, size_ret);
    } break;
    case CL_MEM_MAP_COUNT:
    case CL_MEM_REFERENCE_COUNT: {
      const cl_uint count = (CL_MEM_REFERENCE_COUNT == param ? (cl_uint)obj->refs : 0);
      result = libxstream_mps_answer(&count, sizeof(count), size, value, size_ret);
    } break;
    case CL_MEM_CONTEXT: result = libxstream_mps_answer(&obj->context, sizeof(cl_context), size, value, size_ret); break;
    case CL_MEM_ASSOCIATED_MEMOBJECT: result = libxstream_mps_answer(&obj->owner, sizeof(cl_mem), size, value, size_ret); break;
    case CL_MEM_USES_SVM_POINTER: {
      const cl_bool svm = CL_FALSE;
      result = libxstream_mps_answer(&svm, sizeof(svm), size, value, size_ret);
    } break;
    default: result = CL_INVALID_VALUE;
  }
  return result;
}


static cl_program libxstream_mps_program_new(libxstream_mps_client_t* c, cl_context context, libxstream_mps_msg_t* msg,
  const void* p1, size_t n1, const void* p2, size_t n2, void* data, size_t maxdata, cl_int* errcode_ret)
{
  libxstream_mps_obj_t* obj = libxstream_mps_new(c, LIBXSTREAM_MPS_KIND_PROGRAM, LIBXSTREAM_MPS_OBJ(context));
  cl_int result = CL_SUCCESS;
  if (NULL != obj) {
    libxstream_mps_reply_t reply;
    msg->arg[0] = LIBXSTREAM_MPS_ID(c, obj);
    msg->arg[1] = LIBXSTREAM_MPS_ID(c, context);
    result = libxstream_mps_call(c, msg, p1, n1, p2, n2, &reply, data, maxdata);
    if (CL_SUCCESS != result) {
      libxstream_mps_delete(c, obj, 1);
      obj = NULL;
    }
  }
  else result = CL_OUT_OF_HOST_MEMORY;
  if (NULL != errcode_ret) *errcode_ret = result;
  return (cl_program)obj;
}


static cl_program CL_API_CALL libxstream_mps_program_source(cl_context context, cl_uint count, const char** strings,
  const size_t* lengths, cl_int* errcode_ret)
{
  libxstream_mps_client_t* const c = libxstream_mps_client;
  cl_program result = NULL;
  size_t nbytes = 0, i;
  char* source = NULL;
  cl_int status = CL_SUCCESS;
  if (!LIBXSTREAM_MPS_IS(context, LIBXSTREAM_MPS_KIND_CONTEXT)) status = CL_INVALID_CONTEXT;
  else if (0 == count || NULL == strings) status = CL_INVALID_VALUE;
  for (i = 0; i < count && CL_SUCCESS == status; ++i) {
    if (NULL != strings[i]) nbytes += ((NULL != lengths && 0 != lengths[i]) ? lengths[i] : strlen(strings[i]));
    else status = CL_INVALID_VALUE;
  }
  if (CL_SUCCESS == status) {
    source = (char*)malloc(nbytes + 1);
    if (NULL != source) {
      size_t offset = 0;
      libxstream_mps_msg_t msg;
      for (i = 0; i < count; ++i) {
        const size_t n = ((NULL != lengths && 0 != lengths[i]) ? lengths[i] : strlen(strings[i]));
        memcpy(source + offset, strings[i], n);
        offset += n;
      }
      libxstream_mps_init(&msg, LIBXSTREAM_MPS_PROGRAM_SOURCE);
      result = libxstream_mps_program_new(c, context, &msg, source, nbytes, NULL, 0, NULL, 0, &status);
      free(source);
    }
    else status = CL_OUT_OF_HOST_MEMORY;
  }
  if (NULL != errcode_ret) *errcode_ret = status;
  return result;
}


static cl_program CL_API_CALL libxstream_mps_program_binary(cl_context context, cl_uint num_devices,
  const cl_device_id* devices, const size_t* lengths, const unsigned char** binaries, cl_int* binary_status,
  cl_int* errcode_ret)
{
  libxstream_mps_client_t* const c = libxstream_mps_client;
  cl_program result = NULL;
  cl_int status = CL_SUCCESS;
  size_t nbytes = 0;
  cl_uint i;
  if (!LIBXSTREAM_MPS_IS(context, LIBXSTREAM_MPS_KIND_CONTEXT)) status = CL_INVALID_CONTEXT;
  else if (0 == num_devices || LIBXSTREAM_MAXNDEVS < num_devices || NULL == devices || NULL == lengths || NULL == binaries) {
    status = CL_INVALID_VALUE;
  }
  for (i = 0; i < num_devices && CL_SUCCESS == status; ++i) {
    if (!LIBXSTREAM_MPS_IS(devices[i], LIBXSTREAM_MPS_KIND_DEVICE)) status = CL_INVALID_DEVICE;
    else if (0 == lengths[i] || NULL == binaries[i]) status = CL_INVALID_VALUE;
    else nbytes += lengths[i];
  }
  if (CL_SUCCESS == status) {
    const size_t nhead = 2 * sizeof(uint64_t) * num_devices;
    char* const payload = (char*)malloc(nhead + nbytes);
    if (NULL != payload) {
      int32_t statuses[LIBXSTREAM_MAXNDEVS];
      uint64_t* const head = (uint64_t*)payload;
      size_t offset = nhead;
      libxstream_mps_msg_t msg;
      for (i = 0; i < num_devices; ++i) {
        head[i] = LIBXSTREAM_MPS_DEVID(c, devices[i]);
        head[num_devices + i] = lengths[i];
        memcpy(payload + offset, binaries[i], lengths[i]);
        offset += lengths[i];
      }
      libxstream_mps_init(&msg, LIBXSTREAM_MPS_PROGRAM_BINARY);
      msg.arg[2] = num_devices;
      result = libxstream_mps_program_new(c, context, &msg, payload, offset, NULL, 0, statuses,
        sizeof(int32_t) * num_devices, &status);
      for (i = 0; NULL != binary_status && i < num_devices; ++i) {
        binary_status[i] = (NULL != result ? statuses[i] : status);
      }
      free(payload);
    }
    else status = CL_OUT_OF_HOST_MEMORY;
  }
  if (NULL != errcode_ret) *errcode_ret = status;
  return result;
}


static cl_program CL_API_CALL libxstream_mps_program_il(cl_context context, const void* il, size_t length, cl_int* errcode_ret)
{
  LIBXS_UNUSED(context);
  LIBXS_UNUSED(il);
  LIBXS_UNUSED(length);
  if (NULL != errcode_ret) *errcode_ret = CL_INVALID_OPERATION;
  return NULL;
}


static cl_int CL_API_CALL libxstream_mps_program_retain(cl_program program)
{
  return libxstream_mps_retain(program, LIBXSTREAM_MPS_KIND_PROGRAM);
}


static cl_int CL_API_CALL libxstream_mps_program_release(cl_program program)
{
  return libxstream_mps_release(program, LIBXSTREAM_MPS_KIND_PROGRAM);
}


static cl_int CL_API_CALL libxstream_mps_build(cl_program program, cl_uint num_devices, const cl_device_id* devices,
  const char* options, void (CL_CALLBACK* notify)(cl_program, void*), void* user_data)
{
  libxstream_mps_client_t* const c = libxstream_mps_client;
  uint64_t ids[LIBXSTREAM_MAXNDEVS];
  cl_int result = CL_SUCCESS;
  cl_uint i;
  if (!LIBXSTREAM_MPS_IS(program, LIBXSTREAM_MPS_KIND_PROGRAM)) result = CL_INVALID_PROGRAM;
  else if (LIBXSTREAM_MAXNDEVS < num_devices || (0 != num_devices) != (NULL != devices)) result = CL_INVALID_VALUE;
  for (i = 0; i < num_devices && CL_SUCCESS == result; ++i) {
    if (LIBXSTREAM_MPS_IS(devices[i], LIBXSTREAM_MPS_KIND_DEVICE)) ids[i] = LIBXSTREAM_MPS_DEVID(c, devices[i]);
    else result = CL_INVALID_DEVICE;
  }
  if (CL_SUCCESS == result) {
    libxstream_mps_reply_t reply;
    libxstream_mps_msg_t msg;
    libxstream_mps_init(&msg, LIBXSTREAM_MPS_BUILD);
    msg.arg[0] = LIBXSTREAM_MPS_ID(c, program);
    msg.arg[1] = num_devices;
    result = libxstream_mps_call(c, &msg, ids, sizeof(uint64_t) * num_devices,
      NULL != options ? options : "", NULL != options ? strlen(options) : 0, &reply, NULL, 0);
    if (NULL != notify) notify(program, user_data);
  }
  return result;
}


static cl_int CL_API_CALL libxstream_mps_program_info(cl_program program, cl_program_info param, size_t size,
  void* value, size_t* size_ret)
{
  libxstream_mps_client_t* const c = libxstream_mps_client;
  const libxstream_mps_obj_t* const obj = LIBXSTREAM_MPS_OBJ(program);
  cl_int result = CL_SUCCESS;
  if (!LIBXSTREAM_MPS_IS(program, LIBXSTREAM_MPS_KIND_PROGRAM)) result = CL_INVALID_PROGRAM;
  else if (CL_PROGRAM_CONTEXT == param) result = libxstream_mps_answer(&obj->context, sizeof(cl_context), size, value, size_ret);
  else if (CL_PROGRAM_NUM_DEVICES == param || CL_PROGRAM_DEVICES == param) {
    result = libxstream_mps_context_info((cl_context)obj->context,
      CL_PROGRAM_DEVICES == param ? CL_CONTEXT_DEVICES : CL_CONTEXT_NUM_DEVICES, size, value, size_ret);
  }
  else if (CL_PROGRAM_REFERENCE_COUNT == param) {
    const cl_uint refs = (cl_uint)obj->refs;
    result = libxstream_mps_answer(&refs, sizeof(refs), size, value, size_ret);
  }
  else if (CL_PROGRAM_BINARIES == param) { /* the server cannot write into the application's buffers */
    const size_t ndevices = obj->context->size;
    size_t sizes[LIBXSTREAM_MAXNDEVS], total = 0, i;
    if (NULL != value) {
      libxstream_mps_msg_t msg;
      libxstream_mps_init(&msg, LIBXSTREAM_MPS_PROGRAM_INFO);
      msg.arg[0] = LIBXSTREAM_MPS_ID(c, program);
      msg.arg[1] = CL_PROGRAM_BINARY_SIZES;
      result = libxstream_mps_query(c, &msg, sizeof(size_t) * ndevices, sizes, NULL);
      if (size < sizeof(unsigned char*) * ndevices) result = CL_INVALID_VALUE;
      for (i = 0; i < ndevices; ++i) total += sizes[i];
      if (CL_SUCCESS == result && 0 != total) {
        char* const data = (char*)malloc(total);
        if (NULL != data) {
          unsigned char** const binaries = (unsigned char**)value;
          size_t offset = 0;
          msg.arg[1] = CL_PROGRAM_BINARIES;
          result = libxstream_mps_query(c, &msg, total, data, NULL);
          for (i = 0; i < ndevices && CL_SUCCESS == result; ++i) {
            if (NULL != binaries[i]) memcpy(binaries[i], data + offset, sizes[i]);
            offset += sizes[i];
          }
          free(data);
        }
        else result = CL_OUT_OF_HOST_MEMORY;
      }
    }
    if (NULL != size_ret) *size_ret = sizeof(unsigned char*) * ndevices;
  }
  else {
    libxstream_mps_msg_t msg;
    libxstream_mps_init(&msg, LIBXSTREAM_MPS_PROGRAM_INFO);
    msg.arg[0] = LIBXSTREAM_MPS_ID(c, program);
    msg.arg[1] = param;
    result = libxstream_mps_query(c, &msg, size, value, size_ret);
  }
  return result;
}


static cl_int CL_API_CALL libxstream_mps_build_info(cl_program program, cl_device_id device, cl_program_build_info param,
  size_t size, void* value, size_t* size_ret)
{
  libxstream_mps_client_t* const c = libxstream_mps_client;
  cl_int result = CL_SUCCESS;
  if (!LIBXSTREAM_MPS_IS(program, LIBXSTREAM_MPS_KIND_PROGRAM)) result = CL_INVALID_PROGRAM;
  else if (!LIBXSTREAM_MPS_IS(device, LIBXSTREAM_MPS_KIND_DEVICE)) result = CL_INVALID_DEVICE;
  else {
    libxstream_mps_msg_t msg;
    libxstream_mps_init(&msg, LIBXSTREAM_MPS_BUILD_INFO);
    msg.arg[0] = LIBXSTREAM_MPS_ID(c, program);
    msg.arg[1] = LIBXSTREAM_MPS_DEVID(c, device);
    msg.arg[2] = param;
    result = libxstream_mps_query(c, &msg, size, value, size_ret);
  }
  return result;
}


static cl_kernel CL_API_CALL libxstream_mps_kernel(cl_program program, const char* name, cl_int* errcode_ret)
{
  libxstream_mps_client_t* const c = libxstream_mps_client;
  libxstream_mps_obj_t* obj = NULL;
  cl_int result = CL_SUCCESS;
  if (!LIBXSTREAM_MPS_IS(program, LIBXSTREAM_MPS_KIND_PROGRAM)) result = CL_INVALID_PROGRAM;
  else if (NULL == name) result = CL_INVALID_VALUE;
  else {
    obj = libxstream_mps_new(c, LIBXSTREAM_MPS_KIND_KERNEL, LIBXSTREAM_MPS_OBJ(program)->context);
    if (NULL != obj) {
      libxstream_mps_reply_t reply;
      libxstream_mps_msg_t msg;
      obj->owner = LIBXSTREAM_MPS_OBJ(program);
      libxstream_mps_init(&msg, LIBXSTREAM_MPS_KERNEL);
      msg.arg[0] = LIBXSTREAM_MPS_ID(c, obj);
      msg.arg[1] = LIBXSTREAM_MPS_ID(c, program);
      result = libxstream_mps_call(c, &msg, name, strlen(name), NULL, 0, &reply, NULL, 0);
      if (CL_SUCCESS == result) LIBXS_ATOMIC_ADD_FETCH(&obj->owner->refs, 1, LIBXS_ATOMIC_SEQ_CST); /* see libxstream_mps_release */
      else {
        libxstream_mps_delete(c, obj, 1);
        obj = NULL;
      }
    }
    else result = CL_OUT_OF_HOST_MEMORY;
  }
  if (NULL != errcode_ret) *errcode_ret = result;
  return (cl_kernel)obj;
}


static cl_int CL_API_CALL libxstream_mps_kernel_retain(cl_kernel kernel)
{
  return libxstream_mps_retain(kernel, LIBXSTREAM_MPS_KIND_KERNEL);
}


static cl_int CL_API_CALL libxstream_mps_kernel_release(cl_kernel kernel)
{
  return libxstream_mps_release(kernel, LIBXSTREAM_MPS_KIND_KERNEL);
}


/**
 * Kept with the kernel, and every launch carries all of them: the server launches an instance
 * of the kernel per connection, hence one thread's arguments are not another one's. A record
 * is the index and kind, the size and a buffer's generation, the buffer, and the value.
 * The buffers set as arguments are what a launch depends on (libxstream_mps_order).
 */
static cl_int CL_API_CALL libxstream_mps_kernel_arg(cl_kernel kernel, cl_uint index, size_t size, const void* value)
{
  libxstream_mps_client_t* const c = libxstream_mps_client;
  cl_int result = CL_SUCCESS;
  if (!LIBXSTREAM_MPS_IS(kernel, LIBXSTREAM_MPS_KIND_KERNEL)) result = CL_INVALID_KERNEL;
  else if (1024 <= index) result = CL_INVALID_ARG_INDEX;
  else if (NULL != value && 0 == size) result = CL_INVALID_ARG_SIZE;
  else {
    libxstream_mps_obj_t* const obj = LIBXSTREAM_MPS_OBJ(kernel);
    libxstream_mps_obj_t* const mem = (NULL != value && sizeof(cl_mem) == size) ? libxstream_mps_lookup(c, value) : NULL;
    const uint64_t kind = (NULL == value ? LIBXSTREAM_MPS_ARG_LOCAL : (NULL != mem ? LIBXSTREAM_MPS_ARG_MEM : LIBXSTREAM_MPS_ARG_VALUE));
    void* big = NULL;
    if (LIBXSTREAM_MPS_ARG_VALUE == kind && sizeof(uint64_t) < size) {
      big = malloc(size);
      if (NULL != big) memcpy(big, value, size);
      else result = CL_OUT_OF_HOST_MEMORY;
    }
    LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, &c->order.lock); /* the launch reads the buffer arguments under the order lock */
    if (obj->size <= index && LIBXSTREAM_MPS_ARG_MEM == kind && CL_SUCCESS == result) {
      const size_t n = LIBXS_MAX((size_t)index + 1, 2 * obj->size);
      libxstream_mps_obj_t** const args = (libxstream_mps_obj_t**)realloc(obj->args, sizeof(void*) * n);
      if (NULL != args) {
        memset(args + obj->size, 0, sizeof(void*) * (n - obj->size));
        obj->args = args;
        obj->size = n;
      }
      else result = CL_OUT_OF_HOST_MEMORY;
    }
    if (index < obj->size && CL_SUCCESS == result) obj->args[index] = mem;
    LIBXS_LOCK_RELEASE(LIBXS_LOCK, &c->order.lock);
    if (CL_SUCCESS == result) {
      LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, &c->lock.lock);
      if (obj->npargs <= index) {
        const size_t n = LIBXS_MAX((size_t)index + 1, 2 * obj->npargs);
        uint64_t* const pargs = (uint64_t*)realloc(obj->pargs, sizeof(uint64_t) * 4 * n);
        void** const bigs = (NULL != pargs ? (void**)realloc(obj->bigs, sizeof(void*) * n) : NULL);
        if (NULL != pargs) obj->pargs = pargs;
        if (NULL != bigs) {
          memset(obj->pargs + 4 * obj->npargs, 0, sizeof(uint64_t) * 4 * (n - obj->npargs));
          memset(bigs + obj->npargs, 0, sizeof(void*) * (n - obj->npargs));
          obj->bigs = bigs;
          obj->npargs = n;
        }
      }
      if (index < obj->npargs) {
        uint64_t* const record = obj->pargs + 4 * index;
        record[0] = (uint64_t)index | (kind << 32) | ((uint64_t)1 << 63); /* set */
        record[1] = (uint64_t)size | ((uint64_t)(NULL != mem ? mem->gen : 0) << 32);
        record[2] = (NULL != mem ? LIBXSTREAM_MPS_ID(c, mem) : 0);
        record[3] = 0;
        if (LIBXSTREAM_MPS_ARG_VALUE == kind && NULL == big) memcpy(record + 3, value, size);
        free(obj->bigs[index]);
        obj->bigs[index] = big;
        big = NULL;
      }
      else result = CL_OUT_OF_HOST_MEMORY;
      LIBXS_LOCK_RELEASE(LIBXS_LOCK, &c->lock.lock);
    }
    free(big);
  }
  return result;
}


static cl_int CL_API_CALL libxstream_mps_kernel_info(cl_kernel kernel, cl_kernel_info param, size_t size, void* value,
  size_t* size_ret)
{
  const libxstream_mps_obj_t* const obj = LIBXSTREAM_MPS_OBJ(kernel);
  cl_int result = CL_SUCCESS;
  if (!LIBXSTREAM_MPS_IS(kernel, LIBXSTREAM_MPS_KIND_KERNEL)) result = CL_INVALID_KERNEL;
  else if (CL_KERNEL_PROGRAM == param) result = libxstream_mps_answer(&obj->owner, sizeof(cl_program), size, value, size_ret);
  else if (CL_KERNEL_CONTEXT == param) result = libxstream_mps_answer(&obj->context, sizeof(cl_context), size, value, size_ret);
  else if (CL_KERNEL_REFERENCE_COUNT == param) {
    const cl_uint refs = (cl_uint)obj->refs;
    result = libxstream_mps_answer(&refs, sizeof(refs), size, value, size_ret);
  }
  else {
    libxstream_mps_client_t* const c = libxstream_mps_client;
    libxstream_mps_msg_t msg;
    libxstream_mps_init(&msg, LIBXSTREAM_MPS_KERNEL_INFO);
    msg.arg[0] = LIBXSTREAM_MPS_ID(c, kernel);
    msg.arg[1] = param;
    result = libxstream_mps_query(c, &msg, size, value, size_ret);
  }
  return result;
}


static cl_int CL_API_CALL libxstream_mps_kernel_wginfo(cl_kernel kernel, cl_device_id device,
  cl_kernel_work_group_info param, size_t size, void* value, size_t* size_ret)
{
  libxstream_mps_client_t* const c = libxstream_mps_client;
  cl_int result = CL_SUCCESS;
  if (!LIBXSTREAM_MPS_IS(kernel, LIBXSTREAM_MPS_KIND_KERNEL)) result = CL_INVALID_KERNEL;
  else if (NULL != device && !LIBXSTREAM_MPS_IS(device, LIBXSTREAM_MPS_KIND_DEVICE)) result = CL_INVALID_DEVICE;
  else {
    libxstream_mps_msg_t msg;
    libxstream_mps_init(&msg, LIBXSTREAM_MPS_KERNEL_WGINFO);
    msg.arg[0] = LIBXSTREAM_MPS_ID(c, kernel);
    msg.arg[1] = LIBXSTREAM_MPS_DEVID(c, NULL != device ? device : (cl_device_id)LIBXSTREAM_MPS_OBJ(kernel)->context->device);
    msg.arg[2] = param;
    result = libxstream_mps_query(c, &msg, size, value, size_ret);
  }
  return result;
}


static cl_int CL_API_CALL libxstream_mps_wait(cl_uint num_events, const cl_event* events)
{
  libxstream_mps_client_t* const c = libxstream_mps_client;
  uint64_t local[LIBXSTREAM_MPS_NLOCAL], *ids = local;
  cl_int result = CL_SUCCESS;
  cl_uint i, n = 0;
  if (0 == num_events || NULL == events) result = CL_INVALID_VALUE;
  else if (LIBXSTREAM_MPS_NLOCAL < num_events) {
    ids = (uint64_t*)malloc(sizeof(uint64_t) * num_events);
    if (NULL == ids) result = CL_OUT_OF_HOST_MEMORY;
  }
  for (i = 0; i < num_events && CL_SUCCESS == result; ++i) { /* events known to be complete are not waited for */
    if (!LIBXSTREAM_MPS_IS(events[i], LIBXSTREAM_MPS_KIND_EVENT)) result = CL_INVALID_EVENT;
    else if (0 == libxstream_mps_known(LIBXSTREAM_MPS_OBJ(events[i]))) ids[n++] = LIBXSTREAM_MPS_ID(c, events[i]);
  }
  if (CL_SUCCESS == result && 0 != n) {
    libxstream_mps_reply_t reply;
    libxstream_mps_msg_t msg;
    libxstream_mps_init(&msg, LIBXSTREAM_MPS_WAIT);
    msg.arg[6] = n;
    result = libxstream_mps_call(c, &msg, ids, sizeof(uint64_t) * n, NULL, 0, &reply, NULL, 0);
    for (i = 0; i < num_events && CL_SUCCESS == result; ++i) libxstream_mps_completed(LIBXSTREAM_MPS_OBJ(events[i]));
  }
  libxstream_mps_done(local, ids);
  return result;
}


static cl_int CL_API_CALL libxstream_mps_event_info(cl_event event, cl_event_info param, size_t size, void* value,
  size_t* size_ret)
{
  const libxstream_mps_obj_t* const obj = LIBXSTREAM_MPS_OBJ(event);
  cl_int result = CL_SUCCESS;
  if (!LIBXSTREAM_MPS_IS(event, LIBXSTREAM_MPS_KIND_EVENT)) result = CL_INVALID_EVENT;
  else switch (param) {
    case CL_EVENT_COMMAND_QUEUE: result = libxstream_mps_answer(&obj->owner, sizeof(cl_command_queue), size, value, size_ret); break;
    case CL_EVENT_CONTEXT: result = libxstream_mps_answer(&obj->context, sizeof(cl_context), size, value, size_ret); break;
    case CL_EVENT_COMMAND_TYPE: {
      const cl_command_type type = (cl_command_type)obj->flags;
      result = libxstream_mps_answer(&type, sizeof(type), size, value, size_ret);
    } break;
    case CL_EVENT_REFERENCE_COUNT: {
      const cl_uint refs = (cl_uint)obj->refs;
      result = libxstream_mps_answer(&refs, sizeof(refs), size, value, size_ret);
    } break;
    case CL_EVENT_COMMAND_EXECUTION_STATUS: {
      cl_int status = CL_COMPLETE;
      if (0 == libxstream_mps_known(obj)) {
        libxstream_mps_client_t* const c = libxstream_mps_client;
        libxstream_mps_reply_t reply;
        libxstream_mps_msg_t msg;
        libxstream_mps_init(&msg, LIBXSTREAM_MPS_EVENT_STATUS);
        msg.arg[0] = LIBXSTREAM_MPS_ID(c, event);
        msg.arg[1] = obj->gen;
        result = libxstream_mps_call(c, &msg, NULL, 0, NULL, 0, &reply, NULL, 0);
        if (CL_SUCCESS == result) {
          status = (cl_int)reply.arg0;
          if (CL_COMPLETE == status) libxstream_mps_completed(LIBXSTREAM_MPS_OBJ(event));
        }
      }
      if (CL_SUCCESS == result) result = libxstream_mps_answer(&status, sizeof(status), size, value, size_ret);
    } break;
    default: result = CL_INVALID_VALUE;
  }
  return result;
}


static cl_int CL_API_CALL libxstream_mps_event_retain(cl_event event)
{
  return libxstream_mps_retain(event, LIBXSTREAM_MPS_KIND_EVENT);
}


static cl_int CL_API_CALL libxstream_mps_event_release(cl_event event)
{
  return libxstream_mps_release(event, LIBXSTREAM_MPS_KIND_EVENT);
}


static cl_int CL_API_CALL libxstream_mps_profiling(cl_event event, cl_profiling_info param, size_t size, void* value,
  size_t* size_ret)
{
  LIBXS_UNUSED(event);
  LIBXS_UNUSED(param);
  LIBXS_UNUSED(size);
  LIBXS_UNUSED(value);
  LIBXS_UNUSED(size_ret);
  return CL_PROFILING_INFO_NOT_AVAILABLE;
}


static cl_int CL_API_CALL libxstream_mps_callback(cl_event event, cl_int type,
  void (CL_CALLBACK* notify)(cl_event, cl_int, void*), void* user_data)
{
  LIBXS_UNUSED(event);
  LIBXS_UNUSED(type);
  LIBXS_UNUSED(notify);
  LIBXS_UNUSED(user_data);
  return CL_INVALID_OPERATION;
}


/* Finishing a queue that received nothing since it was last finished needs no round trip. */
static cl_int libxstream_mps_queue_op(cl_command_queue queue, uint64_t op)
{
  libxstream_mps_client_t* const c = libxstream_mps_client;
  libxstream_mps_obj_t* const obj = LIBXSTREAM_MPS_OBJ(queue);
  cl_int result = CL_SUCCESS;
  if (LIBXSTREAM_MPS_IS(queue, LIBXSTREAM_MPS_KIND_QUEUE)) {
    /* last is the queue's latest command, taken under the order lock: issued, not merely announced */
    if (__atomic_load_n(&obj->done, __ATOMIC_ACQUIRE) < __atomic_load_n(&obj->last, __ATOMIC_ACQUIRE)) {
      libxstream_mps_reply_t reply;
      libxstream_mps_msg_t msg;
      libxstream_mps_init(&msg, op);
      msg.arg[0] = LIBXSTREAM_MPS_ID(c, queue);
      result = libxstream_mps_call(c, &msg, NULL, 0, NULL, 0, LIBXSTREAM_MPS_FINISH == op ? &reply : NULL, NULL, 0);
      if (CL_SUCCESS == result && LIBXSTREAM_MPS_FINISH == op) libxstream_mps_complete(obj, msg.seq + 1);
    }
  }
  else result = CL_INVALID_COMMAND_QUEUE;
  return result;
}


static cl_int CL_API_CALL libxstream_mps_flush(cl_command_queue queue)
{
  return libxstream_mps_queue_op(queue, LIBXSTREAM_MPS_FLUSH);
}


static cl_int CL_API_CALL libxstream_mps_finish(cl_command_queue queue)
{
  return libxstream_mps_queue_op(queue, LIBXSTREAM_MPS_FINISH);
}


/* Data in shared memory is named by its offset; any other is sent along (write) or returned (read). */
static cl_int libxstream_mps_transfer(uint64_t op, cl_command_queue queue, cl_mem buffer, cl_bool blocking,
  size_t offset, size_t size, const void* src, void* dst, cl_uint nwait, const cl_event* wait, cl_event* event)
{
  libxstream_mps_client_t* const c = libxstream_mps_client;
  uint64_t local[LIBXSTREAM_MPS_NLOCAL], *ids;
  libxstream_mps_msg_t msg;
  cl_int result = libxstream_mps_prepare(&msg, op, queue, nwait, wait, event,
    LIBXSTREAM_MPS_WRITE == op ? CL_COMMAND_WRITE_BUFFER : CL_COMMAND_READ_BUFFER, local, &ids);
  const void* const ptr = (LIBXSTREAM_MPS_WRITE == op ? src : dst);
  if (CL_SUCCESS == result && !LIBXSTREAM_MPS_IS(buffer, LIBXSTREAM_MPS_KIND_MEM)) result = CL_INVALID_MEM_OBJECT;
  else if (CL_SUCCESS == result && (NULL == ptr || LIBXSTREAM_MPS_OBJ(buffer)->size < offset + size)) result = CL_INVALID_VALUE;
  if (CL_SUCCESS == result) {
    const uint64_t arena = libxstream_mps_arena_offset(c, ptr, size);
    const int sync = (CL_FALSE != blocking || (0 == arena && LIBXSTREAM_MPS_READ == op));
    libxstream_mps_reply_t reply;
    msg.arg[1] = LIBXSTREAM_MPS_ID(c, buffer);
    msg.arg[2] = (uint64_t)sync;
    msg.arg[3] = offset;
    msg.arg[4] = size;
    msg.arg[5] = arena;
    result = libxstream_mps_call(c, &msg, ids, sizeof(uint64_t) * nwait,
      (0 == arena && LIBXSTREAM_MPS_WRITE == op) ? src : NULL, (0 == arena && LIBXSTREAM_MPS_WRITE == op) ? size : 0,
      0 != sync ? &reply : NULL, (0 == arena && LIBXSTREAM_MPS_READ == op) ? dst : NULL,
      (0 == arena && LIBXSTREAM_MPS_READ == op) ? size : 0);
    if (CL_SUCCESS == result && 0 != sync) libxstream_mps_complete(LIBXSTREAM_MPS_OBJ(queue), msg.seq + 1);
  }
  libxstream_mps_done(local, ids);
  return result;
}


static cl_int CL_API_CALL libxstream_mps_read(cl_command_queue queue, cl_mem buffer, cl_bool blocking, size_t offset,
  size_t size, void* ptr, cl_uint nwait, const cl_event* wait, cl_event* event)
{
  return libxstream_mps_transfer(LIBXSTREAM_MPS_READ, queue, buffer, blocking, offset, size, NULL, ptr, nwait, wait, event);
}


static cl_int CL_API_CALL libxstream_mps_write(cl_command_queue queue, cl_mem buffer, cl_bool blocking, size_t offset,
  size_t size, const void* ptr, cl_uint nwait, const cl_event* wait, cl_event* event)
{
  return libxstream_mps_transfer(LIBXSTREAM_MPS_WRITE, queue, buffer, blocking, offset, size, ptr, NULL, nwait, wait, event);
}


static cl_int CL_API_CALL libxstream_mps_copy(cl_command_queue queue, cl_mem src, cl_mem dst, size_t src_offset,
  size_t dst_offset, size_t size, cl_uint nwait, const cl_event* wait, cl_event* event)
{
  libxstream_mps_client_t* const c = libxstream_mps_client;
  uint64_t local[LIBXSTREAM_MPS_NLOCAL], *ids;
  libxstream_mps_msg_t msg;
  cl_int result = libxstream_mps_prepare(&msg, LIBXSTREAM_MPS_COPY, queue, nwait, wait, event, CL_COMMAND_COPY_BUFFER,
    local, &ids);
  if (CL_SUCCESS == result && (!LIBXSTREAM_MPS_IS(src, LIBXSTREAM_MPS_KIND_MEM) || !LIBXSTREAM_MPS_IS(dst, LIBXSTREAM_MPS_KIND_MEM))) {
    result = CL_INVALID_MEM_OBJECT;
  }
  if (CL_SUCCESS == result) {
    msg.arg[1] = LIBXSTREAM_MPS_ID(c, src);
    msg.arg[2] = LIBXSTREAM_MPS_ID(c, dst);
    msg.arg[3] = src_offset;
    msg.arg[4] = dst_offset;
    msg.arg[8] = size;
    result = libxstream_mps_call(c, &msg, ids, sizeof(uint64_t) * nwait, NULL, 0, NULL, NULL, 0);
  }
  libxstream_mps_done(local, ids);
  return result;
}


static cl_int CL_API_CALL libxstream_mps_fill(cl_command_queue queue, cl_mem buffer, const void* pattern,
  size_t pattern_size, size_t offset, size_t size, cl_uint nwait, const cl_event* wait, cl_event* event)
{
  libxstream_mps_client_t* const c = libxstream_mps_client;
  uint64_t local[LIBXSTREAM_MPS_NLOCAL], *ids;
  libxstream_mps_msg_t msg;
  cl_int result = libxstream_mps_prepare(&msg, LIBXSTREAM_MPS_FILL, queue, nwait, wait, event, CL_COMMAND_FILL_BUFFER,
    local, &ids);
  if (CL_SUCCESS == result && !LIBXSTREAM_MPS_IS(buffer, LIBXSTREAM_MPS_KIND_MEM)) result = CL_INVALID_MEM_OBJECT;
  else if (CL_SUCCESS == result && (NULL == pattern || 0 == pattern_size)) result = CL_INVALID_VALUE;
  if (CL_SUCCESS == result) {
    msg.arg[1] = LIBXSTREAM_MPS_ID(c, buffer);
    msg.arg[2] = pattern_size;
    msg.arg[3] = offset;
    msg.arg[4] = size;
    result = libxstream_mps_call(c, &msg, ids, sizeof(uint64_t) * nwait, pattern, pattern_size, NULL, NULL, 0);
  }
  libxstream_mps_done(local, ids);
  return result;
}


static cl_int CL_API_CALL libxstream_mps_ndrange(cl_command_queue queue, cl_kernel kernel, cl_uint work_dim,
  const size_t* global_offset, const size_t* global_size, const size_t* local_size, cl_uint nwait,
  const cl_event* wait, cl_event* event)
{
  libxstream_mps_client_t* const c = libxstream_mps_client;
  uint64_t local[LIBXSTREAM_MPS_NLOCAL], *ids;
  libxstream_mps_msg_t msg;
  cl_int result = libxstream_mps_prepare(&msg, LIBXSTREAM_MPS_NDRANGE, queue, nwait, wait, event,
    CL_COMMAND_NDRANGE_KERNEL, local, &ids);
  if (CL_SUCCESS == result && !LIBXSTREAM_MPS_IS(kernel, LIBXSTREAM_MPS_KIND_KERNEL)) result = CL_INVALID_KERNEL;
  else if (CL_SUCCESS == result && (0 == work_dim || 3 < work_dim)) result = CL_INVALID_WORK_DIMENSION;
  else if (CL_SUCCESS == result && NULL == global_size) result = CL_INVALID_GLOBAL_WORK_SIZE;
  if (CL_SUCCESS == result) { /* sizes, followed by the kernel's arguments */
    const libxstream_mps_obj_t* const obj = LIBXSTREAM_MPS_OBJ(kernel);
    uint64_t buffer[9 + 128], *sizes = buffer;
    size_t nargs = 0, j;
    cl_uint i;
    LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, &c->lock.lock);
    for (j = 0; j < obj->npargs; ++j) {
      const uint64_t* const record = obj->pargs + 4 * j;
      if (0 != (record[0] >> 63)) {
        nargs += 3 + (LIBXSTREAM_MPS_ARG_VALUE == ((record[0] >> 32) & 0xFF)
          ? LIBXS_UPDIV(record[1] & 0xFFFFFFFF, sizeof(uint64_t)) : 0);
      }
    }
    if (sizeof(buffer) / sizeof(*buffer) < 9 + nargs) sizes = (uint64_t*)malloc(sizeof(uint64_t) * (9 + nargs));
    if (NULL != sizes) {
      uint64_t* out = sizes + 3 * work_dim;
      for (j = 0; j < obj->npargs; ++j) {
        const uint64_t* const record = obj->pargs + 4 * j;
        if (0 != (record[0] >> 63)) {
          const size_t size = (size_t)(record[1] & 0xFFFFFFFF);
          const size_t nvalue = (LIBXSTREAM_MPS_ARG_VALUE == ((record[0] >> 32) & 0xFF) ? LIBXS_UPDIV(size, sizeof(uint64_t)) : 0);
          out[0] = record[0] & ~((uint64_t)1 << 63);
          out[1] = record[1];
          out[2] = record[2];
          if (1 == nvalue) out[3] = record[3];
          else if (1 < nvalue) {
            out[3 + nvalue - 1] = 0;
            memcpy(out + 3, obj->bigs[j], size);
          }
          out += 3 + nvalue;
        }
      }
    }
    LIBXS_LOCK_RELEASE(LIBXS_LOCK, &c->lock.lock);
    if (NULL != sizes) {
      for (i = 0; i < work_dim; ++i) {
        sizes[i] = (NULL != global_offset ? global_offset[i] : 0);
        sizes[work_dim + i] = global_size[i];
        sizes[2 * work_dim + i] = (NULL != local_size ? local_size[i] : 0);
      }
      msg.arg[1] = LIBXSTREAM_MPS_ID(c, kernel);
      msg.arg[2] = work_dim;
      msg.arg[3] = (NULL != global_offset ? 1 : 0) | (NULL != local_size ? 2 : 0);
      msg.arg[4] = nargs;
      result = libxstream_mps_call(c, &msg, ids, sizeof(uint64_t) * nwait, sizes,
        sizeof(uint64_t) * (3 * work_dim + nargs), NULL, NULL, 0);
      if (buffer != sizes) free(sizes);
    }
    else result = CL_OUT_OF_HOST_MEMORY;
  }
  libxstream_mps_done(local, ids);
  return result;
}


static cl_int libxstream_mps_mark(cl_command_queue queue, cl_uint nwait, const cl_event* wait, cl_event* event, int barrier)
{
  libxstream_mps_client_t* const c = libxstream_mps_client;
  uint64_t local[LIBXSTREAM_MPS_NLOCAL], *ids;
  libxstream_mps_msg_t msg;
  cl_int result = libxstream_mps_prepare(&msg, LIBXSTREAM_MPS_MARKER, queue, nwait, wait, event,
    0 != barrier ? CL_COMMAND_BARRIER : CL_COMMAND_MARKER, local, &ids);
  if (CL_SUCCESS == result) {
    msg.arg[1] = (uint64_t)barrier;
    result = libxstream_mps_call(c, &msg, ids, sizeof(uint64_t) * nwait, NULL, 0, NULL, NULL, 0);
  }
  libxstream_mps_done(local, ids);
  return result;
}


static cl_int CL_API_CALL libxstream_mps_marker(cl_command_queue queue, cl_uint nwait, const cl_event* wait,
  cl_event* event)
{
  return libxstream_mps_mark(queue, nwait, wait, event, 0);
}


/* On an in-order queue, a barrier on the queue's own events orders nothing: not even sent. */
static cl_int CL_API_CALL libxstream_mps_barrier(cl_command_queue queue, cl_uint nwait, const cl_event* wait,
  cl_event* event)
{
  cl_int result = CL_SUCCESS;
  int redundant = (NULL == event && 0 != nwait && NULL != wait && LIBXSTREAM_MPS_IS(queue, LIBXSTREAM_MPS_KIND_QUEUE)
    && 0 == (LIBXSTREAM_MPS_OBJ(queue)->flags & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE));
  cl_uint i;
  for (i = 0; i < nwait && 0 != redundant; ++i) {
    redundant = (LIBXSTREAM_MPS_IS(wait[i], LIBXSTREAM_MPS_KIND_EVENT) && LIBXSTREAM_MPS_OBJ(queue) == LIBXSTREAM_MPS_OBJ(wait[i])->owner);
  }
  if (0 == redundant) result = libxstream_mps_mark(queue, nwait, wait, event, 1);
  return result;
}


/**
 * A buffer created on shared memory is mapped by the server, and the application
 * sees the shadow (or its own pointer, which is then synchronized here). Any other
 * buffer is read into, and written back from, temporary shared memory.
 */
static void* CL_API_CALL libxstream_mps_map(cl_command_queue queue, cl_mem buffer, cl_bool blocking, cl_map_flags flags,
  size_t offset, size_t size, cl_uint nwait, const cl_event* wait, cl_event* event, cl_int* errcode_ret)
{
  libxstream_mps_client_t* const c = libxstream_mps_client;
  libxstream_mps_obj_t* const obj = LIBXSTREAM_MPS_OBJ(buffer);
  uint64_t local[LIBXSTREAM_MPS_NLOCAL], *ids;
  libxstream_mps_msg_t msg;
  void* result_ptr = NULL;
  cl_int result = libxstream_mps_prepare(&msg, LIBXSTREAM_MPS_MAP, queue, nwait, wait, event, CL_COMMAND_MAP_BUFFER,
    local, &ids);
  LIBXS_UNUSED(blocking);
  if (CL_SUCCESS == result && !LIBXSTREAM_MPS_IS(buffer, LIBXSTREAM_MPS_KIND_MEM)) result = CL_INVALID_MEM_OBJECT;
  else if (CL_SUCCESS == result && (0 == size || obj->size < offset + size)) result = CL_INVALID_VALUE;
  if (CL_SUCCESS == result) {
    char* const temp = (NULL == obj->shadow ? libxstream_mps_arena_alloc(c, size) : NULL);
    char* const view = (NULL != temp ? temp : (obj->shadow + offset));
    if (NULL != obj->shadow || NULL != temp) {
      libxstream_mps_reply_t reply;
      msg.arg[1] = LIBXSTREAM_MPS_ID(c, buffer);
      msg.arg[2] = flags;
      msg.arg[3] = offset;
      msg.arg[4] = size;
      msg.arg[5] = libxstream_mps_arena_offset(c, view, size);
      msg.arg[8] = (NULL != obj->shadow ? 1 : 0);
      result = libxstream_mps_call(c, &msg, ids, sizeof(uint64_t) * nwait, NULL, 0, &reply, NULL, 0);
      if (CL_SUCCESS == result) {
        int i;
        result_ptr = (NULL != obj->host ? (obj->host + offset) : view);
        if (result_ptr != view && 0 == (flags & CL_MAP_WRITE_INVALIDATE_REGION)) memcpy(result_ptr, view, size);
        LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, &c->lock.lock);
        for (i = 0; i < LIBXSTREAM_MPS_MAXNMAPS && NULL != c->maps[i].buffer; ++i);
        if (LIBXSTREAM_MPS_MAXNMAPS > i) {
          c->maps[i].buffer = obj;
          c->maps[i].ptr = (char*)result_ptr;
          c->maps[i].temp = temp;
          c->maps[i].offset = offset;
          c->maps[i].size = size;
          c->maps[i].flags = flags;
        }
        else result = CL_OUT_OF_RESOURCES;
        LIBXS_LOCK_RELEASE(LIBXS_LOCK, &c->lock.lock);
      }
      if (CL_SUCCESS != result) {
        libxstream_mps_arena_free(c, temp, size);
        result_ptr = NULL;
      }
    }
    else result = CL_OUT_OF_HOST_MEMORY;
  }
  libxstream_mps_done(local, ids);
  if (NULL != errcode_ret) *errcode_ret = result;
  return result_ptr;
}


static cl_int CL_API_CALL libxstream_mps_unmap(cl_command_queue queue, cl_mem memobj, void* ptr, cl_uint nwait,
  const cl_event* wait, cl_event* event)
{
  libxstream_mps_client_t* const c = libxstream_mps_client;
  uint64_t local[LIBXSTREAM_MPS_NLOCAL], *ids;
  libxstream_mps_msg_t msg;
  libxstream_mps_map_t map;
  cl_int result = libxstream_mps_prepare(&msg, LIBXSTREAM_MPS_UNMAP, queue, nwait, wait, event,
    CL_COMMAND_UNMAP_MEM_OBJECT, local, &ids);
  memset(&map, 0, sizeof(map));
  if (CL_SUCCESS == result && !LIBXSTREAM_MPS_IS(memobj, LIBXSTREAM_MPS_KIND_MEM)) result = CL_INVALID_MEM_OBJECT;
  if (CL_SUCCESS == result) {
    int i;
    LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, &c->lock.lock);
    for (i = 0; i < LIBXSTREAM_MPS_MAXNMAPS; ++i) {
      if (LIBXSTREAM_MPS_OBJ(memobj) == c->maps[i].buffer && (char*)ptr == c->maps[i].ptr) {
        map = c->maps[i];
        c->maps[i].buffer = NULL;
        break;
      }
    }
    LIBXS_LOCK_RELEASE(LIBXS_LOCK, &c->lock.lock);
    if (NULL == map.buffer) result = CL_INVALID_VALUE;
  }
  if (CL_SUCCESS == result) {
    char* const view = (NULL != map.temp ? map.temp : (map.buffer->shadow + map.offset));
    libxstream_mps_reply_t reply;
    if (view != map.ptr && 0 != (map.flags & (CL_MAP_WRITE | CL_MAP_WRITE_INVALIDATE_REGION))) {
      memcpy(view, map.ptr, map.size);
    }
    msg.arg[1] = LIBXSTREAM_MPS_ID(c, memobj);
    msg.arg[2] = map.flags;
    msg.arg[3] = map.offset;
    msg.arg[4] = map.size;
    msg.arg[5] = libxstream_mps_arena_offset(c, view, map.size);
    msg.arg[8] = (NULL == map.temp ? 1 : 0);
    result = libxstream_mps_call(c, &msg, ids, sizeof(uint64_t) * nwait, NULL, 0,
      NULL != map.temp ? &reply : NULL, NULL, 0);
    libxstream_mps_arena_free(c, map.temp, map.size);
  }
  libxstream_mps_done(local, ids);
  return result;
}


static void* CL_API_CALL libxstream_mps_svm_alloc(cl_context context, cl_svm_mem_flags flags, size_t size, cl_uint alignment)
{
  LIBXS_UNUSED(context);
  LIBXS_UNUSED(flags);
  LIBXS_UNUSED(size);
  LIBXS_UNUSED(alignment);
  return NULL;
}


static void CL_API_CALL libxstream_mps_svm_free(cl_context context, void* pointer)
{
  LIBXS_UNUSED(context);
  LIBXS_UNUSED(pointer);
}


static void* CL_API_CALL libxstream_mps_extension(const char* name)
{
  LIBXS_UNUSED(name);
  return NULL;
}


static void* CL_API_CALL libxstream_mps_extension_platform(cl_platform_id platform, const char* name)
{
  LIBXS_UNUSED(platform);
  LIBXS_UNUSED(name);
  return NULL;
}


static cl_int CL_API_CALL libxstream_mps_unload(cl_platform_id platform)
{
  LIBXS_UNUSED(platform);
  return CL_SUCCESS;
}


static void libxstream_mps_setup(void)
{
  void (**slot)(void) = (void (**)(void))(void*)&libxstream_mps_dispatch;
  size_t i;
  for (i = 0; i < sizeof(libxstream_mps_dispatch) / sizeof(*slot); ++i) slot[i] = (void (*)(void))libxstream_mps_unsupported;
  libxstream_mps_dispatch.clGetPlatformIDs = libxstream_mps_platform_ids;
  libxstream_mps_dispatch.clGetPlatformInfo = libxstream_mps_platform_info;
  libxstream_mps_dispatch.clGetDeviceIDs = libxstream_mps_device_ids;
  libxstream_mps_dispatch.clGetDeviceInfo = libxstream_mps_device_info;
  libxstream_mps_dispatch.clCreateSubDevices = libxstream_mps_sub_devices;
  libxstream_mps_dispatch.clRetainDevice = libxstream_mps_device_keep;
  libxstream_mps_dispatch.clReleaseDevice = libxstream_mps_device_keep;
  libxstream_mps_dispatch.clCreateContext = libxstream_mps_context;
  libxstream_mps_dispatch.clCreateContextFromType = libxstream_mps_context_type;
  libxstream_mps_dispatch.clRetainContext = libxstream_mps_context_retain;
  libxstream_mps_dispatch.clReleaseContext = libxstream_mps_context_release;
  libxstream_mps_dispatch.clGetContextInfo = libxstream_mps_context_info;
  libxstream_mps_dispatch.clCreateCommandQueue = libxstream_mps_queue_legacy;
  libxstream_mps_dispatch.clCreateCommandQueueWithProperties = libxstream_mps_queue;
  libxstream_mps_dispatch.clRetainCommandQueue = libxstream_mps_queue_retain;
  libxstream_mps_dispatch.clReleaseCommandQueue = libxstream_mps_queue_release;
  libxstream_mps_dispatch.clGetCommandQueueInfo = libxstream_mps_queue_info;
  libxstream_mps_dispatch.clCreateBuffer = libxstream_mps_buffer;
  libxstream_mps_dispatch.clCreateSubBuffer = libxstream_mps_subbuffer;
  libxstream_mps_dispatch.clRetainMemObject = libxstream_mps_mem_retain;
  libxstream_mps_dispatch.clReleaseMemObject = libxstream_mps_mem_release;
  libxstream_mps_dispatch.clGetMemObjectInfo = libxstream_mps_mem_info;
  libxstream_mps_dispatch.clCreateProgramWithSource = libxstream_mps_program_source;
  libxstream_mps_dispatch.clCreateProgramWithBinary = libxstream_mps_program_binary;
  libxstream_mps_dispatch.clCreateProgramWithIL = libxstream_mps_program_il;
  libxstream_mps_dispatch.clRetainProgram = libxstream_mps_program_retain;
  libxstream_mps_dispatch.clReleaseProgram = libxstream_mps_program_release;
  libxstream_mps_dispatch.clBuildProgram = libxstream_mps_build;
  libxstream_mps_dispatch.clUnloadPlatformCompiler = libxstream_mps_unload;
  libxstream_mps_dispatch.clGetProgramInfo = libxstream_mps_program_info;
  libxstream_mps_dispatch.clGetProgramBuildInfo = libxstream_mps_build_info;
  libxstream_mps_dispatch.clCreateKernel = libxstream_mps_kernel;
  libxstream_mps_dispatch.clRetainKernel = libxstream_mps_kernel_retain;
  libxstream_mps_dispatch.clReleaseKernel = libxstream_mps_kernel_release;
  libxstream_mps_dispatch.clSetKernelArg = libxstream_mps_kernel_arg;
  libxstream_mps_dispatch.clGetKernelInfo = libxstream_mps_kernel_info;
  libxstream_mps_dispatch.clGetKernelWorkGroupInfo = libxstream_mps_kernel_wginfo;
  libxstream_mps_dispatch.clWaitForEvents = libxstream_mps_wait;
  libxstream_mps_dispatch.clGetEventInfo = libxstream_mps_event_info;
  libxstream_mps_dispatch.clRetainEvent = libxstream_mps_event_retain;
  libxstream_mps_dispatch.clReleaseEvent = libxstream_mps_event_release;
  libxstream_mps_dispatch.clGetEventProfilingInfo = libxstream_mps_profiling;
  libxstream_mps_dispatch.clSetEventCallback = libxstream_mps_callback;
  libxstream_mps_dispatch.clFlush = libxstream_mps_flush;
  libxstream_mps_dispatch.clFinish = libxstream_mps_finish;
  libxstream_mps_dispatch.clEnqueueReadBuffer = libxstream_mps_read;
  libxstream_mps_dispatch.clEnqueueWriteBuffer = libxstream_mps_write;
  libxstream_mps_dispatch.clEnqueueCopyBuffer = libxstream_mps_copy;
  libxstream_mps_dispatch.clEnqueueFillBuffer = libxstream_mps_fill;
  libxstream_mps_dispatch.clEnqueueMapBuffer = libxstream_mps_map;
  libxstream_mps_dispatch.clEnqueueUnmapMemObject = libxstream_mps_unmap;
  libxstream_mps_dispatch.clEnqueueNDRangeKernel = libxstream_mps_ndrange;
  libxstream_mps_dispatch.clEnqueueMarkerWithWaitList = libxstream_mps_marker;
  libxstream_mps_dispatch.clEnqueueBarrierWithWaitList = libxstream_mps_barrier;
  libxstream_mps_dispatch.clSVMAlloc = libxstream_mps_svm_alloc;
  libxstream_mps_dispatch.clSVMFree = libxstream_mps_svm_free;
  libxstream_mps_dispatch.clGetExtensionFunctionAddress = libxstream_mps_extension;
  libxstream_mps_dispatch.clGetExtensionFunctionAddressForPlatform = libxstream_mps_extension_platform;
  libxstream_mps_platform.dispatch = &libxstream_mps_dispatch;
  libxstream_mps_platform.kind = LIBXSTREAM_MPS_KIND_PLATFORM;
  libxstream_mps_platform.refs = 1;
}


/* Whether the client waits for an answer: decided by the command, never by its outcome. */
static int libxstream_mpsd_answered(const libxstream_mps_msg_t* msg)
{
  int result;
  switch (msg->op) {
    case LIBXSTREAM_MPS_RELEASE:
    case LIBXSTREAM_MPS_COPY:
    case LIBXSTREAM_MPS_FILL:
    case LIBXSTREAM_MPS_NDRANGE:
    case LIBXSTREAM_MPS_MARKER:
    case LIBXSTREAM_MPS_FLUSH: result = 0; break;
    case LIBXSTREAM_MPS_WRITE:
    case LIBXSTREAM_MPS_READ: result = (0 != msg->arg[2]); break;
    case LIBXSTREAM_MPS_UNMAP: result = (0 == msg->arg[8]); break;
    default: result = 1;
  }
  return result;
}


static void* libxstream_mpsd_handle(const libxstream_mpsd_client_t* cl, uint64_t id)
{
  return (0 != id && id <= cl->capacity) ? cl->handles[id - 1] : NULL;
}


/* Release order: an unordered status query that sees the handle also sees its generation. */
static void libxstream_mpsd_set(libxstream_mpsd_client_t* cl, uint64_t id, void* handle, int kind)
{
  if (0 != id && id <= cl->capacity) {
    cl->kinds[id - 1] = (unsigned char)(NULL != handle ? kind : 0);
    __atomic_store_n(cl->handles + id - 1, handle, __ATOMIC_RELEASE);
  }
}


/**
 * A command's event costs a driver event and its release, while most events are never
 * waited for or polled. Hence the command is enqueued without one, and an event that is
 * needed later becomes a marker on its queue: it completes no earlier than the command
 * (in-order, or after everything enqueued before), and nothing enqueued before the marker
 * depends on what needs the event, since dependencies follow the order of issue.
 */
static void libxstream_mpsd_lazy(libxstream_mpsd_client_t* cl, const libxstream_mps_msg_t* msg, cl_command_queue queue)
{
  const uint64_t id = msg->arg[7];
  if (0 != id && id <= cl->capacity) {
    if (NULL != cl->evop) cl->evop[id - 1] = (unsigned char)msg->op;
    __atomic_store_n(cl->lazy + id - 1, queue, __ATOMIC_RELEASE);
    __atomic_store_n(cl->gens + id - 1, (unsigned int)msg->arg[LIBXSTREAM_MPS_GEN], __ATOMIC_RELEASE);
    cl->kinds[id - 1] = LIBXSTREAM_MPS_KIND_EVENT;
    __atomic_store_n(cl->handles + id - 1, NULL, __ATOMIC_RELEASE);
  }
}


/* The driver event of an event, created as a marker if the command was enqueued without one. */
static cl_event libxstream_mpsd_event(libxstream_mpsd_client_t* cl, uint64_t id)
{
  cl_event result = NULL;
  if (0 != id && id <= cl->capacity) {
    result = (cl_event)__atomic_load_n(cl->handles + id - 1, __ATOMIC_ACQUIRE);
    if (NULL == result) {
      const cl_command_queue queue = __atomic_load_n(cl->lazy + id - 1, __ATOMIC_ACQUIRE);
      cl_event marker = NULL;
      if (NULL != queue && CL_SUCCESS == clEnqueueMarkerWithWaitList(queue, 0, NULL, &marker)) {
        void* expected = NULL;
        /* the process does not know the marker, hence never flushes for it: waiting on an event from
         * another queue requires its queue to be flushed, or a driver may never submit the command */
        clFlush(queue);
        if (__atomic_compare_exchange_n(cl->handles + id - 1, &expected, marker, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
          result = marker;
        }
        else { /* another thread was first */
          clReleaseEvent(marker);
          result = (cl_event)expected;
        }
      }
    }
  }
  return result;
}


/**
 * Answered without waiting for the commands issued before: an event the server has
 * not created yet, or a slot holding an earlier event, cannot be complete. Reading
 * the generation around the handle rules out the slot changing in between.
 */
static cl_int libxstream_mpsd_status(libxstream_mpsd_client_t* cl, uint64_t id, uint64_t gen)
{
  cl_int status = CL_QUEUED;
  if (0 != id && id <= cl->capacity) {
    const unsigned int g0 = __atomic_load_n(cl->gens + id - 1, __ATOMIC_ACQUIRE);
    cl_event event = (cl_event)__atomic_load_n(cl->handles + id - 1, __ATOMIC_ACQUIRE);
    const unsigned int g1 = __atomic_load_n(cl->gens + id - 1, __ATOMIC_ACQUIRE);
    if (NULL != cl->used) cl->used[id - 1] |= 2;
    if (NULL == event && g0 == gen && g1 == gen) event = libxstream_mpsd_event(cl, id);
    if (NULL != event && g0 == gen && g1 == gen
      && CL_SUCCESS != clGetEventInfo(event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(status), &status, NULL))
    {
      status = CL_QUEUED;
    }
  }
  return status;
}


static cl_int libxstream_mpsd_release(void* handle, int kind)
{
  cl_int result = CL_SUCCESS;
  if (NULL != handle) {
    switch (kind) {
      case LIBXSTREAM_MPS_KIND_CONTEXT: result = clReleaseContext((cl_context)handle); break;
      case LIBXSTREAM_MPS_KIND_QUEUE: result = clReleaseCommandQueue((cl_command_queue)handle); break;
      case LIBXSTREAM_MPS_KIND_MEM: result = clReleaseMemObject((cl_mem)handle); break;
      case LIBXSTREAM_MPS_KIND_PROGRAM: result = clReleaseProgram((cl_program)handle); break;
      case LIBXSTREAM_MPS_KIND_KERNEL: result = clReleaseKernel((cl_kernel)handle); break;
      case LIBXSTREAM_MPS_KIND_EVENT: result = clReleaseEvent((cl_event)handle); break;
      default: result = CL_INVALID_VALUE;
    }
  }
  return result;
}


/**
 * Keeps the mappings of shared buffers (mapped is non-zero) until given back (mapped is
 * zero). A process may end with buffers mapped, and a driver may wait forever for their
 * unmapping when the buffer is released, which holds the server and hence the job.
 */
static void libxstream_mpsd_mapped(libxstream_mpsd_client_t* cl, cl_command_queue queue, cl_mem buffer, void* ptr,
  int mapped)
{
  size_t i;
  pthread_mutex_lock(&cl->lock);
  if (0 != mapped) {
    if (cl->nmaps == cl->maxmaps) {
      const size_t n = LIBXS_MAX(2 * cl->maxmaps, 64);
      libxstream_mpsd_mapping_t* const maps = (libxstream_mpsd_mapping_t*)realloc(cl->maps, sizeof(*maps) * n);
      if (NULL != maps) {
        cl->maps = maps;
        cl->maxmaps = n;
      }
    }
    if (cl->nmaps < cl->maxmaps && CL_SUCCESS == clRetainCommandQueue(queue)) {
      cl->maps[cl->nmaps].queue = queue;
      cl->maps[cl->nmaps].buffer = buffer;
      cl->maps[cl->nmaps].ptr = ptr;
      ++cl->nmaps;
    }
  }
  else {
    for (i = 0; i < cl->nmaps && (buffer != cl->maps[i].buffer || ptr != cl->maps[i].ptr); ++i);
    if (i < cl->nmaps) {
      clReleaseCommandQueue(cl->maps[i].queue);
      cl->maps[i] = cl->maps[--cl->nmaps];
    }
  }
  pthread_mutex_unlock(&cl->lock);
}


static void libxstream_mpsd_kernel_free(libxstream_mpsd_kernel_t* entry)
{
  if (NULL != entry) {
    libxstream_mpsd_instance_t* instance = entry->instances;
    while (NULL != instance) {
      libxstream_mpsd_instance_t* const next = instance->next;
      if (NULL != instance->kernel) clReleaseKernel(instance->kernel);
      free(instance->applied);
      free(instance);
      instance = next;
    }
    free(entry->name);
    free(entry);
  }
}


/* What a departed process leaves to the server: all its objects, its shared memory, its state. */
static void libxstream_mpsd_teardown(libxstream_mpsd_client_t* cl)
{
  const libxs_timer_tick_t start = libxs_timer_tick();
  size_t i;
  int kind;
  for (i = 0; i < cl->nmaps; ++i) { /* what the process left mapped, before its buffers are released */
    clEnqueueUnmapMemObject(cl->maps[i].queue, cl->maps[i].buffer, cl->maps[i].ptr, 0, NULL, NULL);
    clFinish(cl->maps[i].queue);
    clReleaseCommandQueue(cl->maps[i].queue);
  }
  if (0 != cl->nmaps && 0 != libxstream_mpsd.verbosity) {
    fprintf(stderr, "INFO LIBXSTREAM-MPS: process %i left %lu buffers mapped\n", cl->pid, (unsigned long)cl->nmaps);
  }
  for (i = 0; i < cl->capacity; ++i) { /* nothing may still access the shared memory */
    if (LIBXSTREAM_MPS_KIND_QUEUE == cl->kinds[i]) clFinish((cl_command_queue)cl->handles[i]);
  }
  for (i = 0; i < cl->capacity; ++i) libxstream_mpsd_kernel_free(cl->kernels[i]);
  for (kind = LIBXSTREAM_MPS_KIND_EVENT; LIBXSTREAM_MPS_KIND_CONTEXT <= kind; --kind) {
    for (i = 0; i < cl->capacity; ++i) {
      if (kind == cl->kinds[i]) libxstream_mpsd_release(cl->handles[i], kind);
    }
  }
  if (0 != libxstream_mpsd.verbosity) {
    fprintf(stderr, "INFO LIBXSTREAM-MPS: process %i released in %.2fs\n", cl->pid, libxs_timer_duration(start, libxs_timer_tick()));
  }
  if (0 != libxstream_mpsd.verbosity) fprintf(stderr, "INFO LIBXSTREAM-MPS: process %i detached\n", cl->pid);
  if (2 <= libxstream_mpsd.verbosity || 0 > libxstream_mpsd.verbosity) {
    fprintf(stderr, "INFO LIBXSTREAM-MPS: process %i commands=%lu wait=%.2fs exec=%.2fs post=%.2fs status=%lu/%.2fs\n",
      cl->pid, (unsigned long)cl->ncmds, 1E-9 * cl->wait_ns, 1E-9 * cl->exec_ns, 1E-9 * cl->post_ns,
      (unsigned long)cl->nstatus, 1E-9 * cl->status_ns);
    if (NULL != cl->hist) {
      const char* const kinds[] = {"?", "device", "context", "queue", "mem", "program", "kernel", "event"};
      const int prec[] = {1, 1, 1};
      int op;
      fprintf(stderr, "INFO LIBXSTREAM-MPS: process %i waits by last blocker (* other thread):", cl->pid);
      for (op = 0; op < LIBXSTREAM_MPS_NWAITS; ++op) {
        if (0 != cl->wait_n[op]) {
          fprintf(stderr, " %s%s=%lu/%.2fs", LIBXSTREAM_MPS_NWAITS - 1 == op ? "window" : kinds[op & 7],
            (LIBXSTREAM_MPS_NWAITS - 1 != op && 0 != (op & 8)) ? "*" : "", (unsigned long)cl->wait_n[op],
            1E-9 * cl->wait_by[op]);
        }
      }
      fprintf(stderr, "\n");
      fprintf(stderr, "INFO LIBXSTREAM-MPS: process %i events=%lu unused=%lu waited=%lu polled=%lu both=%lu\n", cl->pid,
        (unsigned long)cl->nevents, (unsigned long)cl->nused[0], (unsigned long)cl->nused[1], (unsigned long)cl->nused[2],
        (unsigned long)cl->nused[3]);
      fprintf(stderr, "INFO LIBXSTREAM-MPS: process %i events per op (needed):", cl->pid);
      for (op = 0; op < LIBXSTREAM_MPS_NOPS; ++op) {
        if (0 != cl->op_events[op]) {
          fprintf(stderr, " %i=%lu(%lu)", op, (unsigned long)cl->op_events[op], (unsigned long)cl->op_needed[op]);
        }
      }
      fprintf(stderr, "\n");
      fprintf(stderr, "INFO LIBXSTREAM-MPS: process %i exec per op:", cl->pid);
      for (op = 0; op < LIBXSTREAM_MPS_NOPS; ++op) {
        if (0 != cl->op_n[op]) {
          fprintf(stderr, " %i=%lu/%.2fs", op, (unsigned long)cl->op_n[op], 1E-9 * cl->op_ns[op]);
        }
      }
      libxs_hist_print(stderr, cl->hist, prec, "\nINFO LIBXSTREAM-MPS: process %i exec log2(ns): count -> log2 us wait-us\n",
        cl->pid);
    }
  }
  libxs_hist_destroy(cl->hist);
  if (NULL != cl->hist) LIBXS_LOCK_DESTROY(LIBXS_LOCK, &cl->hist_lock);
  munmap(cl->arena, cl->arena_size);
  pthread_cond_destroy(&cl->cond);
  pthread_mutex_destroy(&cl->lock);
  free(cl->handles);
  free(cl->kernels);
  free(cl->lazy);
  free(cl->used);
  free(cl->evop);
  free(cl->gens);
  free(cl->kinds);
  free(cl->done);
  free(cl->pending);
  free(cl->maps);
  free(cl);
}


/**
 * Releases what the server let go of, one object at a time on a thread of its own. A driver
 * may not return from a release, which then holds this thread only: never the commands of
 * the processes still attached, nor the server's exit (libxstream_mps_server).
 */
static void* libxstream_mpsd_reaper(void* arg)
{
  LIBXS_UNUSED(arg);
  pthread_mutex_lock(&libxstream_mpsd_reaplock);
  for (;;) {
    libxstream_mpsd_reap_t reap;
    while (libxstream_mpsd_reaphead == libxstream_mpsd_nreaps) {
      pthread_cond_wait(&libxstream_mpsd_reapcond, &libxstream_mpsd_reaplock);
    }
    reap = libxstream_mpsd_reaps[libxstream_mpsd_reaphead++];
    if (libxstream_mpsd_reaphead == libxstream_mpsd_nreaps) libxstream_mpsd_reaphead = libxstream_mpsd_nreaps = 0;
    libxstream_mpsd_reaping = 2;
    pthread_mutex_unlock(&libxstream_mpsd_reaplock);
    if (LIBXSTREAM_MPS_REAP_CLIENT == reap.kind) libxstream_mpsd_teardown((libxstream_mpsd_client_t*)reap.ptr);
    else if (LIBXSTREAM_MPS_REAP_KERNEL == reap.kind) libxstream_mpsd_kernel_free((libxstream_mpsd_kernel_t*)reap.ptr);
    else libxstream_mpsd_release(reap.ptr, reap.kind);
    pthread_mutex_lock(&libxstream_mpsd_reaplock);
    libxstream_mpsd_reaping = 1;
  }
  return NULL;
}


/* Hands an object to the reaper; released at once only if the reaper cannot be started. */
static void libxstream_mpsd_reap(void* ptr, int kind)
{
  if (NULL != ptr) {
    int queued = 0;
    pthread_mutex_lock(&libxstream_mpsd_reaplock);
    if (0 == libxstream_mpsd_reaping) {
      pthread_attr_t attr;
      pthread_t thread;
      pthread_attr_init(&attr);
      pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
      if (0 == pthread_create(&thread, &attr, libxstream_mpsd_reaper, NULL)) libxstream_mpsd_reaping = 1;
      pthread_attr_destroy(&attr);
    }
    if (0 != libxstream_mpsd_reaping && libxstream_mpsd_nreaps == libxstream_mpsd_maxreaps) {
      const size_t n = LIBXS_MAX(2 * libxstream_mpsd_maxreaps, 1024);
      libxstream_mpsd_reap_t* const reaps = (libxstream_mpsd_reap_t*)realloc(libxstream_mpsd_reaps, sizeof(*reaps) * n);
      if (NULL != reaps) {
        libxstream_mpsd_reaps = reaps;
        libxstream_mpsd_maxreaps = n;
      }
    }
    if (0 != libxstream_mpsd_reaping && libxstream_mpsd_nreaps < libxstream_mpsd_maxreaps) {
      libxstream_mpsd_reaps[libxstream_mpsd_nreaps].ptr = ptr;
      libxstream_mpsd_reaps[libxstream_mpsd_nreaps].kind = kind;
      ++libxstream_mpsd_nreaps;
      pthread_cond_signal(&libxstream_mpsd_reapcond);
      queued = 1;
    }
    pthread_mutex_unlock(&libxstream_mpsd_reaplock);
    if (0 == queued) {
      if (LIBXSTREAM_MPS_REAP_CLIENT == kind) libxstream_mpsd_teardown((libxstream_mpsd_client_t*)ptr);
      else if (LIBXSTREAM_MPS_REAP_KERNEL == kind) libxstream_mpsd_kernel_free((libxstream_mpsd_kernel_t*)ptr);
      else libxstream_mpsd_release(ptr, kind);
    }
  }
}


/* Whether the reaper has nothing left to release. */
static int libxstream_mpsd_reaped(void)
{
  int result;
  pthread_mutex_lock(&libxstream_mpsd_reaplock);
  result = (libxstream_mpsd_reaphead == libxstream_mpsd_nreaps && 2 != libxstream_mpsd_reaping);
  pthread_mutex_unlock(&libxstream_mpsd_reaplock);
  return result;
}


/**
 * The instance of a kernel for the given connection, created on first use. Only that
 * connection adds its instance, hence it is created without holding the lock.
 */
static libxstream_mpsd_instance_t* libxstream_mpsd_instance(libxstream_mpsd_client_t* cl, uint64_t id, int conn)
{
  libxstream_mpsd_instance_t* result = NULL;
  libxstream_mpsd_kernel_t* entry = NULL;
  if (0 != id && id <= cl->capacity) {
    pthread_mutex_lock(&cl->lock);
    entry = cl->kernels[id - 1];
    if (NULL != entry) for (result = entry->instances; NULL != result && conn != result->conn; result = result->next);
    pthread_mutex_unlock(&cl->lock);
  }
  if (NULL == result && NULL != entry) {
    cl_int status = CL_SUCCESS;
    const cl_kernel kernel = clCreateKernel(entry->program, entry->name, &status);
    result = (CL_SUCCESS == status ? (libxstream_mpsd_instance_t*)calloc(1, sizeof(libxstream_mpsd_instance_t)) : NULL);
    if (NULL != result) {
      result->kernel = kernel;
      result->conn = conn;
      pthread_mutex_lock(&cl->lock);
      result->next = entry->instances;
      entry->instances = result;
      pthread_mutex_unlock(&cl->lock);
    }
    else if (CL_SUCCESS == status) clReleaseKernel(kernel);
  }
  return result;
}


/* Sets an argument unless the instance has it already: the same record names the same object (generation). */
static cl_int libxstream_mpsd_kernel_arg(libxstream_mpsd_client_t* cl, libxstream_mpsd_instance_t* instance,
  const uint64_t record[], const uint64_t* end, size_t* nwords)
{
  const cl_uint index = (cl_uint)(record[0] & 0xFFFFFFFF);
  const uint64_t kind = (record[0] >> 32) & 0xFF;
  const size_t size = (size_t)(record[1] & 0xFFFFFFFF);
  const size_t nvalue = (LIBXSTREAM_MPS_ARG_VALUE == kind ? LIBXS_UPDIV(size, sizeof(uint64_t)) : 0);
  const int cache = (LIBXSTREAM_MPS_ARG_VALUE != kind || 1 >= nvalue);
  cl_int result = CL_SUCCESS;
  *nwords = 3 + nvalue;
  if (1024 <= index) result = CL_INVALID_ARG_INDEX;
  else if (end < record + 3 + nvalue) result = CL_INVALID_ARG_SIZE;
  else {
    const uint64_t value = (0 != nvalue ? record[3] : 0);
    uint64_t* applied = NULL;
    if (0 != cache) {
      if (instance->napplied <= index) {
        const size_t n = LIBXS_MAX((size_t)index + 1, 2 * instance->napplied);
        uint64_t* const grown = (uint64_t*)realloc(instance->applied, sizeof(uint64_t) * 4 * n);
        if (NULL != grown) {
          memset(grown + 4 * instance->napplied, 0, sizeof(uint64_t) * 4 * (n - instance->napplied));
          instance->applied = grown;
          instance->napplied = n;
        }
      }
      if (index < instance->napplied) applied = instance->applied + 4 * index;
    }
    if (NULL == applied || (record[0] | ((uint64_t)1 << 63)) != applied[0] || record[1] != applied[1]
      || record[2] != applied[2] || value != applied[3])
    {
      if (LIBXSTREAM_MPS_ARG_MEM == kind) {
        const cl_mem memory = (cl_mem)libxstream_mpsd_handle(cl, record[2]);
        result = clSetKernelArg(instance->kernel, index, sizeof(cl_mem), &memory);
      }
      else if (LIBXSTREAM_MPS_ARG_LOCAL == kind) result = clSetKernelArg(instance->kernel, index, size, NULL);
      else result = clSetKernelArg(instance->kernel, index, size, record + 3);
      if (NULL != applied) {
        applied[0] = (CL_SUCCESS == result ? (record[0] | ((uint64_t)1 << 63)) : 0);
        applied[1] = record[1];
        applied[2] = record[2];
        applied[3] = value;
      }
    }
  }
  return result;
}


/* Releases an object and confirms it to the client, which may reuse the slot from then on. */
static cl_int libxstream_mpsd_dispose(libxstream_mpsd_client_t* cl, uint64_t id, int kind)
{
  const cl_int result = CL_SUCCESS; /* the driver releases later, see libxstream_mpsd_reap */
  libxstream_mpsd_reap(libxstream_mpsd_handle(cl, id), kind);
  if (LIBXSTREAM_MPS_KIND_EVENT == kind && 0 != id && id <= cl->capacity) {
    __atomic_store_n(cl->lazy + id - 1, NULL, __ATOMIC_RELEASE);
  }
  if (LIBXSTREAM_MPS_KIND_EVENT == kind && NULL != cl->used && 0 != id && id <= cl->capacity) {
    pthread_mutex_lock(&cl->lock);
    ++cl->nevents;
    ++cl->nused[cl->used[id - 1] & 3];
    if (NULL != cl->evop && cl->evop[id - 1] < LIBXSTREAM_MPS_NOPS) {
      ++cl->op_events[cl->evop[id - 1]];
      if (0 != cl->used[id - 1]) ++cl->op_needed[cl->evop[id - 1]];
    }
    cl->used[id - 1] = 0;
    pthread_mutex_unlock(&cl->lock);
  }
  if (LIBXSTREAM_MPS_KIND_KERNEL == kind && 0 != id && id <= cl->capacity) {
    libxstream_mpsd_kernel_t* entry;
    pthread_mutex_lock(&cl->lock);
    entry = cl->kernels[id - 1];
    cl->kernels[id - 1] = NULL;
    pthread_mutex_unlock(&cl->lock);
    libxstream_mpsd_reap(entry, LIBXSTREAM_MPS_REAP_KERNEL);
  }
  libxstream_mpsd_set(cl, id, NULL, 0);
  if (0 != id && id <= cl->capacity) __atomic_store_n((uint32_t*)(void*)cl->arena + id - 1, 1, __ATOMIC_RELEASE);
  return result;
}


static void libxstream_mpsd_error(libxstream_mpsd_client_t* cl, uint64_t op, cl_int code)
{
  if (CL_SUCCESS != code && 16 > cl->nerrors++) {
    fprintf(stderr, "ERROR LIBXSTREAM-MPS: command %u of process %i failed with %s (code=%i)\n",
      (unsigned int)op, cl->pid, libxstream_opencl_strerror(code), code);
  }
}


/**
 * One context per set of devices, shared by every client: contexts are what the device
 * time-slices. LIBXSTREAM_MPS_CONTEXTS spreads the clients over several contexts per set,
 * since the driver serializes the calls into one context.
 */
static cl_context libxstream_mpsd_context(uint64_t mask, int slot, cl_int* result)
{
  cl_context context = NULL;
  cl_uint i;
  pthread_mutex_lock(&libxstream_mpsd_lock);
  for (i = 0; i < libxstream_mpsd.ncontexts; ++i) {
    if (mask == libxstream_mpsd.masks[i] && slot == libxstream_mpsd.slots[i]) break;
  }
  if (i < libxstream_mpsd.ncontexts) {
    context = libxstream_mpsd.contexts[i];
    *result = CL_SUCCESS;
  }
  else if (LIBXSTREAM_MAXNDEVS > libxstream_mpsd.ncontexts) {
    cl_device_id devices[LIBXSTREAM_MAXNDEVS];
    cl_context_properties properties[3];
    cl_uint n = 0;
    for (i = 0; i < libxstream_mpsd.ndevices; ++i) {
      if (0 != (mask & ((uint64_t)1 << i))) devices[n++] = libxstream_mpsd.devices[i];
    }
    properties[0] = CL_CONTEXT_PLATFORM;
    properties[1] = (cl_context_properties)libxstream_mpsd.platform;
    properties[2] = 0;
    context = (0 != n ? clCreateContext(properties, n, devices, NULL, NULL, result) : NULL);
    if (NULL != context) {
      libxstream_mpsd.masks[libxstream_mpsd.ncontexts] = mask;
      libxstream_mpsd.slots[libxstream_mpsd.ncontexts] = slot;
      libxstream_mpsd.contexts[libxstream_mpsd.ncontexts++] = context;
    }
    else if (0 == n) *result = CL_INVALID_DEVICE;
  }
  else *result = CL_OUT_OF_RESOURCES;
  pthread_mutex_unlock(&libxstream_mpsd_lock);
  return context;
}


static void CL_CALLBACK libxstream_mpsd_free(cl_event event, cl_int status, void* data)
{
  LIBXS_UNUSED(status);
  free(data);
  clReleaseEvent(event);
}


/* Who holds the event an enqueue returned: the client, the waiting answer, the payload to free. */
static void libxstream_mpsd_track(libxstream_mpsd_client_t* cl, const libxstream_mps_msg_t* msg, cl_event event,
  int wait, char** payload, libxstream_mpsd_post_t* post)
{
  if (NULL != event) {
    const int nuse = (0 != msg->arg[7] ? 1 : 0) + (0 != wait ? 1 : 0) + (NULL != payload ? 1 : 0);
    int i;
    for (i = 1; i < nuse; ++i) clRetainEvent(event);
    if (0 == nuse) clReleaseEvent(event);
    if (0 != msg->arg[7] && msg->arg[7] <= cl->capacity) {
      if (NULL != cl->evop) cl->evop[msg->arg[7] - 1] = (unsigned char)msg->op;
      __atomic_store_n(cl->gens + msg->arg[7] - 1, (unsigned int)msg->arg[LIBXSTREAM_MPS_GEN], __ATOMIC_RELEASE);
      libxstream_mpsd_set(cl, msg->arg[7], event, LIBXSTREAM_MPS_KIND_EVENT);
    }
    if (0 != wait) {
      post->event = event;
      post->events = &post->event;
      post->nevents = 1;
    }
    if (NULL != payload) {
      if (CL_SUCCESS == clSetEventCallback(event, CL_COMPLETE, libxstream_mpsd_free, *payload)) *payload = NULL;
      else { /* the payload is freed by the caller, hence nothing may still read it */
        clWaitForEvents(1, &event);
        clReleaseEvent(event);
      }
    }
  }
}


/* The part of a command that must run in order; what may follow concurrently is left in post. */
static void libxstream_mpsd_execute(libxstream_mpsd_client_t* cl, const libxstream_mps_msg_t* msg, char** payload,
  libxstream_mpsd_post_t* post)
{
  const uint64_t* const a = msg->arg;
  const size_t info_size = (size_t)a[LIBXSTREAM_MPS_INFO];
  const cl_uint nwait = (cl_uint)a[6];
  cl_event local[LIBXSTREAM_MPS_NLOCAL], *waits = local;
  cl_event event = NULL, *const pevent = &event;
  cl_int result = CL_SUCCESS;
  cl_uint i;
  post->reply = libxstream_mpsd_answered(msg);
  if (LIBXSTREAM_MPS_NLOCAL < nwait) {
    waits = (cl_event*)malloc(sizeof(cl_event) * nwait);
    if (NULL == waits) result = CL_OUT_OF_HOST_MEMORY;
  }
  for (i = 0; i < nwait && CL_SUCCESS == result; ++i) {
    const uint64_t id = ((const uint64_t*)*payload)[i];
    waits[i] = libxstream_mpsd_event(cl, id);
    if (NULL != cl->used && 0 != id && id <= cl->capacity) cl->used[id - 1] |= 1;
  }
  if (0 != info_size && CL_SUCCESS == result) {
    post->data = malloc(info_size);
    if (NULL == post->data) result = CL_OUT_OF_HOST_MEMORY;
  }
  if (CL_SUCCESS == result) {
    const char* const data = *payload + sizeof(uint64_t) * nwait;
    char* const arena = (0 != a[5] ? cl->arena + a[5] - 1 : NULL);
    size_t size_ret = 0;
    switch (msg->op) {
      case LIBXSTREAM_MPS_PLATFORM_INFO: {
        result = clGetPlatformInfo(libxstream_mpsd.platform, (cl_platform_info)a[1], info_size, post->data, &size_ret);
      } break;
      case LIBXSTREAM_MPS_DEVICE_IDS: {
        uint32_t* const index = (uint32_t*)malloc(sizeof(uint32_t) * LIBXSTREAM_MAXNDEVS);
        cl_uint n = 0;
        for (i = 0; NULL != index && i < libxstream_mpsd.ndevices; ++i) {
          cl_device_type type = 0;
          if (CL_SUCCESS == clGetDeviceInfo(libxstream_mpsd.devices[i], CL_DEVICE_TYPE, sizeof(type), &type, NULL)
            && (0 != (type & (cl_device_type)a[0]) || (CL_DEVICE_TYPE_DEFAULT == a[0] && 0 == n)))
          {
            index[n++] = i;
          }
        }
        free(post->data);
        post->data = index;
        post->nbytes = sizeof(uint32_t) * n;
        post->arg0 = n;
        if (NULL == index) result = CL_OUT_OF_HOST_MEMORY;
      } break;
      case LIBXSTREAM_MPS_DEVICE_INFO: {
        result = (a[0] < libxstream_mpsd.ndevices ? clGetDeviceInfo(libxstream_mpsd.devices[a[0]], (cl_device_info)a[1],
          info_size, post->data, &size_ret) : CL_INVALID_DEVICE);
      } break;
      case LIBXSTREAM_MPS_CONTEXT: {
        const int slot = cl->ordinal % libxstream_mpsd.ncontexts_per;
        const cl_context context = libxstream_mpsd_context(a[1], slot, &result);
        if (NULL != context && CL_SUCCESS == clRetainContext(context)) {
          libxstream_mpsd_set(cl, a[0], context, LIBXSTREAM_MPS_KIND_CONTEXT);
        }
      } break;
      case LIBXSTREAM_MPS_QUEUE: {
        cl_command_queue queue = NULL;
        if (a[2] < libxstream_mpsd.ndevices) {
          queue = clCreateCommandQueueWithProperties((cl_context)libxstream_mpsd_handle(cl, a[1]),
            libxstream_mpsd.devices[a[2]], 0 != msg->nbytes ? (const cl_queue_properties*)*payload : NULL, &result);
        }
        else result = CL_INVALID_DEVICE;
        libxstream_mpsd_set(cl, a[0], queue, LIBXSTREAM_MPS_KIND_QUEUE);
      } break;
      case LIBXSTREAM_MPS_QUEUE_INFO: {
        result = clGetCommandQueueInfo((cl_command_queue)libxstream_mpsd_handle(cl, a[0]), (cl_command_queue_info)a[1],
          info_size, post->data, &size_ret);
      } break;
      case LIBXSTREAM_MPS_BUFFER: {
        const cl_mem buffer = clCreateBuffer((cl_context)libxstream_mpsd_handle(cl, a[1]), (cl_mem_flags)a[2], (size_t)a[3],
          0 != a[4] ? (cl->arena + a[4] - 1) : (0 != msg->nbytes ? *payload : NULL), &result);
        libxstream_mpsd_set(cl, a[0], buffer, LIBXSTREAM_MPS_KIND_MEM);
      } break;
      case LIBXSTREAM_MPS_SUBBUFFER: {
        cl_buffer_region region;
        cl_mem buffer;
        region.origin = (size_t)a[3];
        region.size = (size_t)a[4];
        buffer = clCreateSubBuffer((cl_mem)libxstream_mpsd_handle(cl, a[1]), (cl_mem_flags)a[2],
          CL_BUFFER_CREATE_TYPE_REGION, &region, &result);
        libxstream_mpsd_set(cl, a[0], buffer, LIBXSTREAM_MPS_KIND_MEM);
      } break;
      case LIBXSTREAM_MPS_PROGRAM_SOURCE: {
        const char* source = *payload;
        const cl_program program = clCreateProgramWithSource((cl_context)libxstream_mpsd_handle(cl, a[1]), 1, &source,
          NULL, &result);
        libxstream_mpsd_set(cl, a[0], program, LIBXSTREAM_MPS_KIND_PROGRAM);
      } break;
      case LIBXSTREAM_MPS_PROGRAM_BINARY: {
        const cl_uint n = (cl_uint)LIBXS_MIN(a[2], (uint64_t)LIBXSTREAM_MAXNDEVS);
        const uint64_t* const head = (const uint64_t*)*payload;
        const unsigned char* binaries[LIBXSTREAM_MAXNDEVS];
        cl_device_id devices[LIBXSTREAM_MAXNDEVS];
        size_t lengths[LIBXSTREAM_MAXNDEVS], offset = 2 * sizeof(uint64_t) * n;
        cl_int* const status = (cl_int*)malloc(sizeof(cl_int) * LIBXSTREAM_MAXNDEVS);
        cl_program program = NULL;
        for (i = 0; i < n; ++i) {
          devices[i] = libxstream_mpsd.devices[head[i] < libxstream_mpsd.ndevices ? head[i] : 0];
          lengths[i] = (size_t)head[n + i];
          binaries[i] = (const unsigned char*)*payload + offset;
          offset += lengths[i];
        }
        if (NULL != status) {
          program = clCreateProgramWithBinary((cl_context)libxstream_mpsd_handle(cl, a[1]), n, devices, lengths,
            binaries, status, &result);
        }
        else result = CL_OUT_OF_HOST_MEMORY;
        libxstream_mpsd_set(cl, a[0], program, LIBXSTREAM_MPS_KIND_PROGRAM);
        post->data = status;
        post->nbytes = sizeof(cl_int) * n;
      } break;
      case LIBXSTREAM_MPS_BUILD: { /* built once the next command may run, compiling takes long */
        const cl_uint n = (cl_uint)LIBXS_MIN(a[1], (uint64_t)LIBXSTREAM_MAXNDEVS);
        cl_device_id* const devices = (cl_device_id*)malloc(sizeof(cl_device_id) * (n + 1));
        if (NULL != devices) {
          for (i = 0; i < n; ++i) {
            const uint64_t id = ((const uint64_t*)*payload)[i];
            devices[i] = libxstream_mpsd.devices[id < libxstream_mpsd.ndevices ? id : 0];
          }
          post->program = (cl_program)libxstream_mpsd_handle(cl, a[0]);
          post->devices = devices;
          post->nbuild = n;
          post->options = *payload + sizeof(uint64_t) * n;
          post->data = devices;
        }
        else result = CL_OUT_OF_HOST_MEMORY;
      } break;
      case LIBXSTREAM_MPS_PROGRAM_INFO: {
        const cl_program program = (cl_program)libxstream_mpsd_handle(cl, a[0]);
        if (CL_PROGRAM_BINARIES == a[1]) {
          size_t sizes[LIBXSTREAM_MAXNDEVS], offset = 0;
          unsigned char* binaries[LIBXSTREAM_MAXNDEVS];
          cl_uint n = 0;
          result = clGetProgramInfo(program, CL_PROGRAM_NUM_DEVICES, sizeof(n), &n, NULL);
          if (CL_SUCCESS == result) result = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t) * n, sizes, NULL);
          for (i = 0; i < n && CL_SUCCESS == result; ++i) {
            binaries[i] = (unsigned char*)post->data + offset;
            offset += sizes[i];
          }
          if (CL_SUCCESS == result && offset <= info_size) {
            result = clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(unsigned char*) * n, binaries, NULL);
          }
          size_ret = offset;
        }
        else result = clGetProgramInfo(program, (cl_program_info)a[1], info_size, post->data, &size_ret);
      } break;
      case LIBXSTREAM_MPS_BUILD_INFO: {
        result = (a[1] < libxstream_mpsd.ndevices ? clGetProgramBuildInfo((cl_program)libxstream_mpsd_handle(cl, a[0]),
          libxstream_mpsd.devices[a[1]], (cl_program_build_info)a[2], info_size, post->data, &size_ret) : CL_INVALID_DEVICE);
      } break;
      case LIBXSTREAM_MPS_KERNEL: { /* the kernel answers queries, its instances are launched */
        const cl_program program = (cl_program)libxstream_mpsd_handle(cl, a[1]);
        cl_kernel kernel = clCreateKernel(program, *payload, &result);
        if (CL_SUCCESS == result && 0 != a[0] && a[0] <= cl->capacity) {
          libxstream_mpsd_kernel_t* const entry = (libxstream_mpsd_kernel_t*)calloc(1, sizeof(libxstream_mpsd_kernel_t));
          const size_t len = strlen(*payload) + 1;
          if (NULL != entry) entry->name = (char*)malloc(len);
          if (NULL != entry && NULL != entry->name) {
            memcpy(entry->name, *payload, len);
            entry->program = program;
            pthread_mutex_lock(&cl->lock);
            libxstream_mpsd_kernel_free(cl->kernels[a[0] - 1]);
            cl->kernels[a[0] - 1] = entry;
            pthread_mutex_unlock(&cl->lock);
          }
          else {
            free(entry);
            clReleaseKernel(kernel);
            kernel = NULL;
            result = CL_OUT_OF_HOST_MEMORY;
          }
        }
        libxstream_mpsd_set(cl, a[0], kernel, LIBXSTREAM_MPS_KIND_KERNEL);
      } break;
      case LIBXSTREAM_MPS_KERNEL_INFO: {
        result = clGetKernelInfo((cl_kernel)libxstream_mpsd_handle(cl, a[0]), (cl_kernel_info)a[1], info_size, post->data,
          &size_ret);
      } break;
      case LIBXSTREAM_MPS_KERNEL_WGINFO: {
        result = (a[1] < libxstream_mpsd.ndevices ? clGetKernelWorkGroupInfo((cl_kernel)libxstream_mpsd_handle(cl, a[0]),
          libxstream_mpsd.devices[a[1]], (cl_kernel_work_group_info)a[2], info_size, post->data, &size_ret)
          : CL_INVALID_DEVICE);
      } break;
      case LIBXSTREAM_MPS_RELEASE: { /* deferred until the object's last use ran */
        int now = 1;
        pthread_mutex_lock(&cl->lock);
        if (0 != a[2] && cl->low <= a[2] - 1) {
          if (cl->npending == cl->maxpending) {
            const size_t n = LIBXS_MAX(2 * cl->maxpending, 1024);
            libxstream_mpsd_release_t* const pending = (libxstream_mpsd_release_t*)realloc(cl->pending,
              sizeof(libxstream_mpsd_release_t) * n);
            if (NULL != pending) {
              cl->pending = pending;
              cl->maxpending = n;
            }
          }
          if (cl->npending < cl->maxpending) {
            cl->pending[cl->npending].after = a[2];
            cl->pending[cl->npending].id = a[1];
            cl->pending[cl->npending].kind = (int)a[0];
            ++cl->npending;
            now = 0;
          }
        }
        pthread_mutex_unlock(&cl->lock);
        if (0 != now) result = libxstream_mpsd_dispose(cl, a[1], (int)a[0]);
      } break;
      case LIBXSTREAM_MPS_WRITE:
      case LIBXSTREAM_MPS_READ: { /* read into temporary memory if returned with the answer */
        const cl_command_queue queue = (cl_command_queue)libxstream_mpsd_handle(cl, a[0]);
        const cl_mem buffer = (cl_mem)libxstream_mpsd_handle(cl, a[1]);
        const int inline_write = (LIBXSTREAM_MPS_WRITE == msg->op && NULL == arena);
        cl_event* const need = ((0 != a[2] || 0 != inline_write) ? pevent : NULL); /* waited for, or frees the payload */
        char* const host = (NULL != arena ? arena
          : (LIBXSTREAM_MPS_WRITE == msg->op ? (*payload + sizeof(uint64_t) * nwait) : (char*)malloc((size_t)a[4])));
        if (NULL != host) {
          if (LIBXSTREAM_MPS_WRITE == msg->op) {
            result = clEnqueueWriteBuffer(queue, buffer, CL_FALSE, (size_t)a[3], (size_t)a[4], host, nwait,
              0 != nwait ? waits : NULL, need);
          }
          else {
            result = clEnqueueReadBuffer(queue, buffer, CL_FALSE, (size_t)a[3], (size_t)a[4], host, nwait,
              0 != nwait ? waits : NULL, need);
          }
        }
        else result = CL_OUT_OF_HOST_MEMORY;
        if (CL_SUCCESS == result) {
          if (NULL != need) libxstream_mpsd_track(cl, msg, event, 0 != a[2], 0 != inline_write ? payload : NULL, post);
          else libxstream_mpsd_lazy(cl, msg, queue);
          if (NULL == arena && LIBXSTREAM_MPS_READ == msg->op) {
            post->data = host;
            post->nbytes = (size_t)a[4];
          }
        }
        else if (NULL == arena && LIBXSTREAM_MPS_READ == msg->op) free(host);
      } break;
      case LIBXSTREAM_MPS_COPY: {
        const cl_command_queue queue = (cl_command_queue)libxstream_mpsd_handle(cl, a[0]);
        result = clEnqueueCopyBuffer(queue, (cl_mem)libxstream_mpsd_handle(cl, a[1]), (cl_mem)libxstream_mpsd_handle(cl, a[2]),
          (size_t)a[3], (size_t)a[4], (size_t)a[8], nwait, 0 != nwait ? waits : NULL, NULL);
        if (CL_SUCCESS == result) libxstream_mpsd_lazy(cl, msg, queue);
      } break;
      case LIBXSTREAM_MPS_FILL: {
        const cl_command_queue queue = (cl_command_queue)libxstream_mpsd_handle(cl, a[0]);
        result = clEnqueueFillBuffer(queue, (cl_mem)libxstream_mpsd_handle(cl, a[1]), data, (size_t)a[2], (size_t)a[3],
          (size_t)a[4], nwait, 0 != nwait ? waits : NULL, NULL);
        if (CL_SUCCESS == result) libxstream_mpsd_lazy(cl, msg, queue);
      } break;
      case LIBXSTREAM_MPS_NDRANGE: {
        const cl_uint dims = (cl_uint)LIBXS_MIN(a[2], 3);
        const uint64_t* const sizes = (const uint64_t*)data;
        const uint64_t* const end = sizes + (msg->nbytes - sizeof(uint64_t) * nwait) / sizeof(uint64_t);
        const uint64_t* record = sizes + 3 * dims;
        libxstream_mpsd_instance_t* const instance = libxstream_mpsd_instance(cl, a[1], post->conn);
        size_t offsets[3], global[3], local_size[3], nwords = 0;
        for (i = 0; i < dims; ++i) {
          offsets[i] = (size_t)sizes[i];
          global[i] = (size_t)sizes[dims + i];
          local_size[i] = (size_t)sizes[2 * dims + i];
        }
        if (NULL == instance) result = CL_INVALID_KERNEL;
        for (; record + 3 <= end && record < sizes + 3 * dims + a[4] && CL_SUCCESS == result; record += nwords) {
          result = libxstream_mpsd_kernel_arg(cl, instance, record, end, &nwords); /* the instance skips what it has */
        }
        if (CL_SUCCESS == result) {
          result = clEnqueueNDRangeKernel((cl_command_queue)libxstream_mpsd_handle(cl, a[0]), instance->kernel, dims,
            0 != (a[3] & 1) ? offsets : NULL, global, 0 != (a[3] & 2) ? local_size : NULL, nwait,
            0 != nwait ? waits : NULL, NULL);
        }
        if (CL_SUCCESS == result) libxstream_mpsd_lazy(cl, msg, (cl_command_queue)libxstream_mpsd_handle(cl, a[0]));
      } break;
      case LIBXSTREAM_MPS_MARKER: {
        const cl_command_queue queue = (cl_command_queue)libxstream_mpsd_handle(cl, a[0]);
        if (0 != a[1]) { /* a barrier orders the queue, its event is made when needed */
          result = clEnqueueBarrierWithWaitList(queue, nwait, 0 != nwait ? waits : NULL, NULL);
          if (CL_SUCCESS == result) libxstream_mpsd_lazy(cl, msg, queue);
        }
        else if (0 != nwait) { /* waits for the list rather than for the queue */
          result = clEnqueueMarkerWithWaitList(queue, nwait, waits, 0 != a[7] ? pevent : NULL);
          if (CL_SUCCESS == result) libxstream_mpsd_track(cl, msg, event, 0, NULL, post);
        }
        else libxstream_mpsd_lazy(cl, msg, queue); /* a marker made when needed is the same marker */
      } break;
      case LIBXSTREAM_MPS_MAP: { /* a shadow is mapped where it lives, a temporary buffer is read into */
        const cl_command_queue queue = (cl_command_queue)libxstream_mpsd_handle(cl, a[0]);
        const cl_mem buffer = (cl_mem)libxstream_mpsd_handle(cl, a[1]);
        if (NULL == arena) result = CL_INVALID_VALUE;
        else if (0 != a[8]) {
          const void* const p = clEnqueueMapBuffer(queue, buffer, CL_FALSE, (cl_map_flags)a[2], (size_t)a[3], (size_t)a[4],
            nwait, 0 != nwait ? waits : NULL, pevent, &result);
          if (CL_SUCCESS == result && p != (const void*)arena) {
            fprintf(stderr, "ERROR LIBXSTREAM-MPS: mapping did not return the host pointer\n");
          }
          if (CL_SUCCESS == result) libxstream_mpsd_mapped(cl, queue, buffer, arena, 1);
        }
        else if (0 == (a[2] & CL_MAP_WRITE_INVALIDATE_REGION)) {
          result = clEnqueueReadBuffer(queue, buffer, CL_FALSE, (size_t)a[3], (size_t)a[4], arena, nwait,
            0 != nwait ? waits : NULL, pevent);
        }
        else result = clEnqueueMarkerWithWaitList(queue, nwait, 0 != nwait ? waits : NULL, pevent);
        if (CL_SUCCESS == result) libxstream_mpsd_track(cl, msg, event, 1, NULL, post);
      } break;
      case LIBXSTREAM_MPS_UNMAP: {
        const cl_command_queue queue = (cl_command_queue)libxstream_mpsd_handle(cl, a[0]);
        const cl_mem buffer = (cl_mem)libxstream_mpsd_handle(cl, a[1]);
        const int temp = (0 == a[8]);
        if (NULL == arena) result = CL_INVALID_VALUE;
        else if (0 == temp) {
          result = clEnqueueUnmapMemObject(queue, buffer, arena, nwait, 0 != nwait ? waits : NULL, 0 != a[7] ? pevent : NULL);
          if (CL_SUCCESS == result) libxstream_mpsd_mapped(cl, queue, buffer, arena, 0);
        }
        else if (0 != (a[2] & (CL_MAP_WRITE | CL_MAP_WRITE_INVALIDATE_REGION))) {
          result = clEnqueueWriteBuffer(queue, buffer, CL_FALSE, (size_t)a[3], (size_t)a[4], arena, nwait,
            0 != nwait ? waits : NULL, pevent);
        }
        else result = clEnqueueMarkerWithWaitList(queue, nwait, 0 != nwait ? waits : NULL, pevent);
        if (CL_SUCCESS == result) libxstream_mpsd_track(cl, msg, event, temp, NULL, post);
      } break;
      case LIBXSTREAM_MPS_FINISH: { /* finished by the connection's thread, once in order (may also wait for later commands) */
        post->finish = (cl_command_queue)libxstream_mpsd_handle(cl, a[0]);
      } break;
      case LIBXSTREAM_MPS_FLUSH: {
        result = clFlush((cl_command_queue)libxstream_mpsd_handle(cl, a[0]));
      } break;
      case LIBXSTREAM_MPS_WAIT: { /* the client may release these while waiting */
        post->events = (cl_event*)malloc(sizeof(cl_event) * LIBXS_MAX(nwait, 1));
        if (NULL != post->events) {
          for (i = 0; i < nwait && CL_SUCCESS == result; ++i) {
            result = clRetainEvent(waits[i]);
            if (CL_SUCCESS == result) post->events[post->nevents++] = waits[i];
          }
        }
        else result = CL_OUT_OF_HOST_MEMORY;
      } break;
      case LIBXSTREAM_MPS_EVENT_STATUS: {
        cl_int status = CL_COMPLETE;
        result = clGetEventInfo(libxstream_mpsd_event(cl, a[0]), CL_EVENT_COMMAND_EXECUTION_STATUS,
          sizeof(status), &status, NULL);
        post->arg0 = (uint64_t)(int64_t)status;
      } break;
      default: result = CL_INVALID_OPERATION;
    }
    if (0 != info_size || LIBXSTREAM_MPS_PLATFORM_INFO == msg->op || LIBXSTREAM_MPS_DEVICE_INFO == msg->op
      || LIBXSTREAM_MPS_QUEUE_INFO == msg->op || LIBXSTREAM_MPS_PROGRAM_INFO == msg->op || LIBXSTREAM_MPS_BUILD_INFO == msg->op
      || LIBXSTREAM_MPS_KERNEL_INFO == msg->op || LIBXSTREAM_MPS_KERNEL_WGINFO == msg->op)
    {
      post->arg0 = size_ret;
      post->nbytes = (CL_SUCCESS == result ? LIBXS_MIN(size_ret, info_size) : 0);
    }
  }
  if (local != waits) free(waits);
  post->result = result;
  if (0 == post->reply) libxstream_mpsd_error(cl, msg->op, result);
}


static int libxstream_mpsd_complete(libxstream_mpsd_client_t* cl, int fd, libxstream_mpsd_post_t* post)
{
  int result = EXIT_SUCCESS;
  cl_uint i;
  if (NULL != post->finish && CL_SUCCESS == post->result) post->result = clFinish(post->finish);
  if (NULL != post->program && CL_SUCCESS == post->result) {
    post->result = clBuildProgram(post->program, post->nbuild, 0 != post->nbuild ? post->devices : NULL, post->options,
      NULL, NULL);
  }
  if (0 != post->nevents) {
    const cl_int status = clWaitForEvents(post->nevents, post->events);
    if (CL_SUCCESS == post->result) post->result = status;
    for (i = 0; i < post->nevents; ++i) clReleaseEvent(post->events[i]);
    if (&post->event != post->events) free(post->events);
  }
  else if (NULL != post->events) free(post->events);
  if (0 != post->reply) {
    libxstream_mps_reply_t reply;
    struct iovec iov[2];
    reply.result = post->result;
    reply.nbytes = (CL_SUCCESS == post->result ? post->nbytes : 0);
    reply.arg0 = post->arg0;
    iov[0].iov_base = &reply;
    iov[0].iov_len = sizeof(reply);
    iov[1].iov_base = post->data;
    iov[1].iov_len = (size_t)reply.nbytes;
    result = libxstream_mps_send(fd, iov, 0 != reply.nbytes ? 2 : 1, -1);
  }
  LIBXS_UNUSED(cl);
  free(post->data);
  return result;
}


/* A command runs once the commands it depends on ran (a fence: once every earlier one ran). */
static uint64_t libxstream_mpsd_ns(libxs_timer_tick_t t0, libxs_timer_tick_t t1)
{
  return (uint64_t)(1E9 * libxs_timer_duration(t0, t1));
}


static int libxstream_mpsd_ready(libxstream_mpsd_client_t* cl, uint64_t seq, const uint64_t deps[], uint64_t ndeps)
{
  int result = EXIT_SUCCESS, ready = 0, waited = 0;
  unsigned int blocker = LIBXSTREAM_MPS_NWAITS - 1;
  libxs_timer_tick_t start = 0;
  pthread_mutex_lock(&cl->lock);
  while (0 == ready && 0 == cl->dead) {
    uint64_t i;
    ready = (LIBXSTREAM_MPS_FENCE != ndeps ? (seq - cl->low < LIBXSTREAM_MPS_WINDOW) : (seq == cl->low));
    if (0 == ready) blocker = LIBXSTREAM_MPS_NWAITS - 1;
    for (i = 0; i < ndeps && 0 != ready && LIBXSTREAM_MPS_FENCE != ndeps; ++i) {
      const uint64_t dep = LIBXSTREAM_MPS_UNTAG(deps[i]) - 1;
      ready = (dep < cl->low || 0 != cl->done[dep % LIBXSTREAM_MPS_WINDOW]);
      if (0 == ready) blocker = LIBXSTREAM_MPS_TAGOF(deps[i]);
    }
    if (0 == ready) {
      if (0 == waited && 2 <= libxstream_mpsd.verbosity) start = libxs_timer_tick();
      waited = 1;
      pthread_cond_wait(&cl->cond, &cl->lock);
    }
  }
  if (0 != waited && 2 <= libxstream_mpsd.verbosity) {
    ++cl->wait_n[blocker];
    cl->wait_by[blocker] += libxstream_mpsd_ns(start, libxs_timer_tick());
  }
  if (0 != cl->dead) result = EXIT_FAILURE;
  pthread_mutex_unlock(&cl->lock);
  return result;
}


/* Marks a command executed; releases deferred until then are carried out (outside the lock). */
static void libxstream_mpsd_executed(libxstream_mpsd_client_t* cl, uint64_t seq)
{
  libxstream_mpsd_release_t ready[64];
  size_t nready = 0, i;
  pthread_mutex_lock(&cl->lock);
  cl->done[seq % LIBXSTREAM_MPS_WINDOW] = 1;
  if (0 != cl->done[cl->low % LIBXSTREAM_MPS_WINDOW]) {
    do {
      cl->done[cl->low % LIBXSTREAM_MPS_WINDOW] = 0;
      ++cl->low;
    } while (0 != cl->done[cl->low % LIBXSTREAM_MPS_WINDOW]);
    for (i = 0; i < cl->npending && nready < sizeof(ready) / sizeof(*ready);) {
      if (cl->pending[i].after - 1 < cl->low) {
        ready[nready++] = cl->pending[i];
        cl->pending[i] = cl->pending[--cl->npending];
      }
      else ++i;
    }
  }
  pthread_cond_broadcast(&cl->cond);
  pthread_mutex_unlock(&cl->lock);
  for (i = 0; i < nready; ++i) libxstream_mpsd_dispose(cl, ready[i].id, ready[i].kind);
}


static libxstream_mpsd_client_t* libxstream_mpsd_hello(int fd)
{
  union { struct cmsghdr header; char data[CMSG_SPACE(sizeof(int))]; } control;
  libxstream_mpsd_client_t* result = NULL;
  libxstream_mps_reply_t reply;
  libxstream_mps_msg_t msg;
  struct msghdr mh;
  struct iovec iov;
  int memfd = -1;
  ssize_t n;
  memset(&mh, 0, sizeof(mh));
  memset(&control, 0, sizeof(control));
  iov.iov_base = &msg;
  iov.iov_len = sizeof(msg);
  mh.msg_iov = &iov;
  mh.msg_iovlen = 1;
  mh.msg_control = control.data;
  mh.msg_controllen = sizeof(control.data);
  n = recvmsg(fd, &mh, MSG_CMSG_CLOEXEC);
  if (0 < n && ((size_t)n == sizeof(msg) || EXIT_SUCCESS == libxstream_mps_recv(fd, (char*)&msg + n, sizeof(msg) - (size_t)n))
    && LIBXSTREAM_MPS_HELLO == msg.op)
  {
    const struct cmsghdr* const cmsg = CMSG_FIRSTHDR(&mh);
    if (NULL != cmsg && SOL_SOCKET == cmsg->cmsg_level && SCM_RIGHTS == cmsg->cmsg_type) {
      memcpy(&memfd, CMSG_DATA(cmsg), sizeof(int));
    }
    if (0 != msg.arg[4] && 0 <= memfd) { /* first connection of a process */
      result = (libxstream_mpsd_client_t*)calloc(1, sizeof(libxstream_mpsd_client_t));
      if (NULL != result) {
        result->arena = (char*)mmap(NULL, (size_t)msg.arg[2], PROT_READ | PROT_WRITE, MAP_SHARED, memfd, 0);
        result->handles = (void**)calloc((size_t)msg.arg[3], sizeof(void*));
        result->kernels = (libxstream_mpsd_kernel_t**)calloc((size_t)msg.arg[3], sizeof(libxstream_mpsd_kernel_t*));
        result->lazy = (cl_command_queue*)calloc((size_t)msg.arg[3], sizeof(cl_command_queue));
        result->gens = (unsigned int*)calloc((size_t)msg.arg[3], sizeof(unsigned int));
        result->kinds = (unsigned char*)calloc((size_t)msg.arg[3], 1);
        result->done = (unsigned char*)calloc(LIBXSTREAM_MPS_WINDOW, 1);
        if (MAP_FAILED != result->arena && NULL != result->handles && NULL != result->gens && NULL != result->kinds
          && NULL != result->done && NULL != result->kernels && NULL != result->lazy)
        {
          pthread_mutex_init(&result->lock, NULL);
          pthread_cond_init(&result->cond, NULL);
          if (2 <= libxstream_mpsd.verbosity || 0 > libxstream_mpsd.verbosity) {
            result->used = (unsigned char*)calloc((size_t)msg.arg[3], 1);
            result->evop = (unsigned char*)calloc((size_t)msg.arg[3], 1);
            result->hist = libxs_hist_create(24 /*nbuckets*/, 3 /*nvals*/, NULL, NULL, NULL);
            if (NULL != result->hist) {
              LIBXS_LOCK_ATTR_TYPE(LIBXS_LOCK) attr;
              LIBXS_LOCK_ATTR_INIT(LIBXS_LOCK, &attr);
              LIBXS_LOCK_INIT(LIBXS_LOCK, &result->hist_lock, &attr);
              LIBXS_LOCK_ATTR_DESTROY(LIBXS_LOCK, &attr);
            }
          }
          result->arena_size = (size_t)msg.arg[2];
          result->capacity = (size_t)msg.arg[3];
          result->token = msg.arg[0];
          result->pid = (int)msg.arg[1];
          pthread_mutex_lock(&libxstream_mpsd_lock);
          result->next = libxstream_mpsd.clients;
          libxstream_mpsd.clients = result;
          result->ordinal = libxstream_mpsd.nserved++;
          pthread_mutex_unlock(&libxstream_mpsd_lock);
          if (0 != libxstream_mpsd.verbosity) fprintf(stderr, "INFO LIBXSTREAM-MPS: process %i attached\n", result->pid);
        }
        else {
          if (MAP_FAILED != result->arena) munmap(result->arena, (size_t)msg.arg[2]);
          free(result->handles);
          free(result->kernels);
          free(result->lazy);
          free(result->used);
          free(result->evop);
          free(result->gens);
          free(result->kinds);
          free(result->done);
          free(result);
          result = NULL;
        }
      }
    }
    else if (0 == msg.arg[4]) {
      pthread_mutex_lock(&libxstream_mpsd_lock);
      for (result = libxstream_mpsd.clients; NULL != result && msg.arg[0] != result->token; result = result->next);
      pthread_mutex_unlock(&libxstream_mpsd_lock);
    }
    if (NULL != result) {
      pthread_mutex_lock(&result->lock);
      ++result->nconn;
      pthread_mutex_unlock(&result->lock);
    }
    reply.result = (NULL != result ? CL_SUCCESS : CL_INVALID_VALUE);
    reply.nbytes = reply.arg0 = 0;
    iov.iov_base = &reply;
    iov.iov_len = sizeof(reply);
    libxstream_mps_send(fd, &iov, 1, -1);
  }
  if (0 <= memfd) close(memfd);
  return result;
}


/* Any connection closing means the process is gone; the last one cleans up after it. */
static void libxstream_mpsd_leave(libxstream_mpsd_client_t* cl)
{
  int last;
  pthread_mutex_lock(&cl->lock);
  cl->dead = 1;
  last = (0 == --cl->nconn);
  pthread_cond_broadcast(&cl->cond);
  pthread_mutex_unlock(&cl->lock);
  if (0 != last) { /* the process is gone: what it leaves is the reaper's */
    libxstream_mpsd_client_t** link;
    pthread_mutex_lock(&libxstream_mpsd_lock);
    for (link = &libxstream_mpsd.clients; NULL != *link && cl != *link; link = &(*link)->next);
    if (NULL != *link) *link = cl->next;
    pthread_mutex_unlock(&libxstream_mpsd_lock);
    libxstream_mpsd_reap(cl, LIBXSTREAM_MPS_REAP_CLIENT);
  }
}


static void* libxstream_mpsd_connection(void* arg)
{
  const int fd = (int)(intptr_t)arg;
  libxstream_mpsd_client_t* const cl = libxstream_mpsd_hello(fd);
  if (NULL != cl) {
    const int conn = LIBXS_ATOMIC_ADD_FETCH(&cl->connids, 1, LIBXS_ATOMIC_SEQ_CST);
    for (;;) {
      uint64_t deps[LIBXSTREAM_MPS_NDEPS];
      libxs_timer_tick_t t0;
      libxstream_mpsd_post_t post;
      int status;
      libxstream_mps_msg_t msg;
      char* payload = NULL;
      if (EXIT_SUCCESS != libxstream_mps_recv(fd, &msg, sizeof(msg))) break;
      payload = (char*)malloc((size_t)msg.nbytes + 1);
      if (NULL == payload) break;
      payload[msg.nbytes] = '\0'; /* a string payload is terminated */
      if ((0 != msg.nbytes && EXIT_SUCCESS != libxstream_mps_recv(fd, payload, (size_t)msg.nbytes))
        || (LIBXSTREAM_MPS_FENCE != msg.ndeps && LIBXSTREAM_MPS_NDEPS < msg.ndeps)
        || (LIBXSTREAM_MPS_FENCE != msg.ndeps && 0 != msg.ndeps
          && EXIT_SUCCESS != libxstream_mps_recv(fd, deps, sizeof(uint64_t) * (size_t)msg.ndeps)))
      {
        free(payload);
        break;
      }
      memset(&post, 0, sizeof(post));
      post.conn = conn;
      t0 = libxs_timer_tick();
      if (LIBXSTREAM_MPS_UNORDERED == msg.seq) { /* takes no turn, hence waits for no other command */
        post.arg0 = (uint64_t)(int64_t)libxstream_mpsd_status(cl, msg.arg[0], msg.arg[1]);
        post.reply = 1;
        LIBXSTREAM_MPS_COUNT(&cl->nstatus, 1);
        LIBXSTREAM_MPS_COUNT(&cl->status_ns, libxstream_mpsd_ns(t0, libxs_timer_tick()));
      }
      else if (EXIT_SUCCESS == libxstream_mpsd_ready(cl, msg.seq, deps, msg.ndeps)) {
        libxs_timer_tick_t t1;
        uint64_t wait_ns, exec_ns;
        if (0 != libxstream_mpsd.issuers) while (0 != sem_wait(&libxstream_mpsd.gate) && EINTR == errno);
        t1 = libxs_timer_tick();
        libxstream_mpsd_execute(cl, &msg, &payload, &post);
        libxstream_mpsd_executed(cl, msg.seq);
        if (0 != libxstream_mpsd.issuers) sem_post(&libxstream_mpsd.gate);
        wait_ns = libxstream_mpsd_ns(t0, t1);
        exec_ns = libxstream_mpsd_ns(t1, libxs_timer_tick());
        LIBXSTREAM_MPS_COUNT(&cl->ncmds, 1);
        LIBXSTREAM_MPS_COUNT(&cl->wait_ns, wait_ns);
        LIBXSTREAM_MPS_COUNT(&cl->exec_ns, exec_ns);
        if (NULL != cl->hist && msg.op < LIBXSTREAM_MPS_NOPS) {
          double vals[3];
          vals[0] = (double)LIBXS_ILOG2(exec_ns);
          vals[1] = 1E-3 * (double)exec_ns;
          vals[2] = 1E-3 * (double)wait_ns;
          libxs_hist_push(&cl->hist_lock, cl->hist, vals);
          LIBXSTREAM_MPS_COUNT(cl->op_n + msg.op, 1);
          LIBXSTREAM_MPS_COUNT(cl->op_ns + msg.op, exec_ns);
        }
      }
      else {
        free(payload);
        break;
      }
      t0 = libxs_timer_tick();
      status = libxstream_mpsd_complete(cl, fd, &post);
      LIBXSTREAM_MPS_COUNT(&cl->post_ns, libxstream_mpsd_ns(t0, libxs_timer_tick()));
      if (EXIT_SUCCESS != status) {
        free(payload);
        break;
      }
      free(payload);
    }
    libxstream_mpsd_leave(cl);
  }
  close(fd);
  pthread_mutex_lock(&libxstream_mpsd_lock);
  --libxstream_mpsd.nthreads;
  pthread_mutex_unlock(&libxstream_mpsd_lock);
  return NULL;
}


/**
 * One platform is served, chosen as LIBXSTREAM chooses devices: the type LIBXSTREAM_DEVTYPE
 * asks for (inherited from the process starting the server), otherwise GPUs first. The
 * forwarding platform is skipped should the loader still list it.
 */
static int libxstream_mpsd_discover(void)
{
  const char* const env_devtype = getenv("LIBXSTREAM_DEVTYPE");
  cl_device_type types[3] = {CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_ACCELERATOR, CL_DEVICE_TYPE_ALL};
  cl_platform_id platforms[LIBXSTREAM_MAXNDEVS];
  cl_uint nplatforms = 0, ntypes = 3, i, t;
  int result = EXIT_FAILURE;
  if (NULL != env_devtype && '\0' != *env_devtype) {
    ntypes = 1;
    if (NULL != libxs_stristr(env_devtype, "gpu")) types[0] = CL_DEVICE_TYPE_GPU;
    else if (NULL != libxs_stristr(env_devtype, "cpu")) types[0] = CL_DEVICE_TYPE_CPU;
    else if (NULL != libxs_stristr(env_devtype, "acc") || NULL != libxs_stristr(env_devtype, "other")) {
      types[0] = CL_DEVICE_TYPE_ACCELERATOR;
    }
    else types[0] = CL_DEVICE_TYPE_ALL;
  }
  if (CL_SUCCESS == clGetPlatformIDs(LIBXSTREAM_MAXNDEVS, platforms, &nplatforms)) {
    for (t = 0; t < ntypes && EXIT_SUCCESS != result; ++t) {
      for (i = 0; i < nplatforms && EXIT_SUCCESS != result; ++i) {
        char suffix[16] = "";
        cl_uint ndevices = 0;
        clGetPlatformInfo(platforms[i], CL_PLATFORM_ICD_SUFFIX_KHR, sizeof(suffix), suffix, NULL);
        if (0 != strcmp(LIBXSTREAM_MPS_SUFFIX, suffix)
          && CL_SUCCESS == clGetDeviceIDs(platforms[i], types[t], LIBXSTREAM_MAXNDEVS, libxstream_mpsd.devices, &ndevices)
          && 0 != ndevices)
        {
          libxstream_mpsd.platform = platforms[i];
          libxstream_mpsd.ndevices = LIBXS_MIN(ndevices, LIBXSTREAM_MAXNDEVS);
          result = EXIT_SUCCESS;
        }
      }
    }
  }
  return result;
}
# endif


# if defined(LIBXSTREAM_MPS)
LIBXSTREAM_API_INTERN void libxstream_mps_presetup(int verbosity)
{
  const char* const env_mps = getenv("LIBXSTREAM_MPS");
  if (NULL == getenv("OCL_ICD_FILENAMES") && NULL == getenv("OCL_ICD_VENDORS") && (NULL == env_mps || 0 != atoi(env_mps))) {
    int active = (NULL != env_mps);
    if (0 == active) { /* a server already running is the opt-in */
      char name[64];
      struct sockaddr_un addr;
      const int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
      memset(&addr, 0, sizeof(addr));
      addr.sun_family = AF_UNIX;
      libxstream_mps_paths(name, sizeof(name), addr.sun_path, sizeof(addr.sun_path));
      if (0 <= fd) {
        active = (0 == connect(fd, (struct sockaddr*)&addr, sizeof(addr)));
        close(fd);
      }
    }
    if (0 != active) {
      char path[PATH_MAX];
      /**
       * Both loaders: ocl-icd reads only OCL_ICD_VENDORS (an ICD named there replaces the
       * installed ones), the Khronos loader adds OCL_ICD_FILENAMES and takes OCL_ICD_VENDORS
       * as a directory, hence the installed ICDs are hidden either way.
       */
      if (EXIT_SUCCESS == libxstream_mps_beside("libxstream_icd.so", path, sizeof(path))
        && 0 == setenv("OCL_ICD_FILENAMES", path, 1) && 0 == setenv("OCL_ICD_VENDORS", path, 1))
      {
        /* the driver serializes calls per context, while contexts time-slice: a few contexts for many ranks */
        if (NULL == getenv("LIBXSTREAM_MPS_CONTEXTS")) setenv("LIBXSTREAM_MPS_CONTEXTS", 4 <= libxs_nranks() ? "2" : "1", 0);
        if (0 != verbosity) fprintf(stderr, "INFO LIBXSTREAM-MPS: forwarding through %s\n", path);
      }
      else fprintf(stderr, "ERROR LIBXSTREAM-MPS: libxstream_icd.so not found beside the library\n");
    }
  }
}


LIBXSTREAM_API void* libxstream_mps_icd_function(const char name[])
{
  void* result = NULL;
  if (NULL != name && 0 == strcmp("clIcdGetPlatformIDsKHR", name)) {
    union { cl_int (*fn)(cl_uint, cl_platform_id[], cl_uint*); void* ptr; } addr;
    addr.fn = libxstream_mps_icd_platforms;
    result = addr.ptr;
  }
  return result;
}


LIBXSTREAM_API cl_int libxstream_mps_icd_platforms(cl_uint num_entries, cl_platform_id platforms[], cl_uint* num_platforms)
{
  pthread_once(&libxstream_mps_once, libxstream_mps_setup);
  return libxstream_mps_platform_ids(num_entries, platforms, num_platforms);
}


LIBXSTREAM_API int libxstream_mps_server(void)
{
  const char *const env_icd = getenv("LIBXSTREAM_MPS_ICD"), *const env_verbose = getenv("LIBXSTREAM_VERBOSE");
  const char *const env_idle = getenv("LIBXSTREAM_MPS_TIMEOUT"), *const env_issuers = getenv("LIBXSTREAM_MPS_ISSUERS");
  const char* const env_contexts = getenv("LIBXSTREAM_MPS_CONTEXTS");
  const int idle_limit = (NULL != env_idle ? atoi(env_idle) : LIBXSTREAM_MPS_IDLE);
  char name[64], path[sizeof(((struct sockaddr_un*)NULL)->sun_path)];
  unsigned long cpus[64];
  int result = EXIT_SUCCESS, handle;
  /* every CPU of the job (the kernel intersects the mask with the cpuset), not those of the rank that started it */
  memset(cpus, 0xFF, sizeof(cpus));
  syscall(SYS_sched_setaffinity, 0, sizeof(cpus), cpus);
  /* before the first OpenCL call: the server must not load the forwarding ICD it serves */
  if (NULL != env_icd) {
    setenv("OCL_ICD_FILENAMES", env_icd, 1);
    setenv("OCL_ICD_VENDORS", env_icd, 1);
  }
  else {
    unsetenv("OCL_ICD_FILENAMES");
    unsetenv("OCL_ICD_VENDORS");
  }
  signal(SIGPIPE, SIG_IGN);
  libxstream_mpsd.verbosity = (NULL != env_verbose ? atoi(env_verbose) : 0);
  libxstream_mpsd.issuers = LIBXS_MAX(NULL != env_issuers ? atoi(env_issuers) : 0, 0);
  libxstream_mpsd.ncontexts_per = LIBXS_MAX(NULL != env_contexts ? atoi(env_contexts) : 1, 1);
  if (0 != libxstream_mpsd.issuers && 0 != sem_init(&libxstream_mpsd.gate, 0, (unsigned int)libxstream_mpsd.issuers)) {
    libxstream_mpsd.issuers = 0;
  }
  libxstream_mps_paths(name, sizeof(name), path, sizeof(path));
  handle = libxs_singleton_acquire(name);
  if (0 <= handle) { /* otherwise another server runs */
    int fd = -1;
    result = libxstream_mpsd_discover();
    if (EXIT_SUCCESS == result) {
      struct sockaddr_un addr;
      memset(&addr, 0, sizeof(addr));
      addr.sun_family = AF_UNIX;
      memcpy(addr.sun_path, path, sizeof(addr.sun_path)); /* same size, terminated */
      unlink(path); /* left behind by a server that did not exit */
      fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
      if (0 > fd || 0 != bind(fd, (struct sockaddr*)&addr, sizeof(addr)) || 0 != listen(fd, 256)) result = EXIT_FAILURE;
    }
    else fprintf(stderr, "ERROR LIBXSTREAM-MPS: no OpenCL device found\n");
    if (EXIT_SUCCESS == result) {
      int idle = 0, limit = 1000 * LIBXSTREAM_MPS_CONNECT;
      if (0 != libxstream_mpsd.verbosity) fprintf(stderr, "INFO LIBXSTREAM-MPS: serving %s\n", path);
      while (0 > idle_limit || idle < limit) { /* until no client was seen for a while */
        struct pollfd pfd;
        int busy;
        pfd.fd = fd;
        pfd.events = POLLIN;
        pfd.revents = 0;
        if (0 < poll(&pfd, 1, 100) && 0 != (pfd.revents & POLLIN)) {
          const int conn = accept(fd, NULL, NULL);
          if (0 <= conn) {
            pthread_attr_t attr;
            pthread_t thread;
            pthread_attr_init(&attr);
            pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
            pthread_mutex_lock(&libxstream_mpsd_lock);
            ++libxstream_mpsd.nthreads;
            pthread_mutex_unlock(&libxstream_mpsd_lock);
            if (0 != pthread_create(&thread, &attr, libxstream_mpsd_connection, (void*)(intptr_t)conn)) {
              pthread_mutex_lock(&libxstream_mpsd_lock);
              --libxstream_mpsd.nthreads;
              pthread_mutex_unlock(&libxstream_mpsd_lock);
              close(conn);
            }
            pthread_attr_destroy(&attr);
          }
        }
        pthread_mutex_lock(&libxstream_mpsd_lock);
        busy = (0 != libxstream_mpsd.nthreads);
        if (0 != libxstream_mpsd.nserved) limit = 1000 * idle_limit;
        pthread_mutex_unlock(&libxstream_mpsd_lock);
        idle = (0 != busy ? 0 : (idle + 100));
      }
      unlink(path);
    }
    if (0 <= fd) close(fd);
    if (0 != libxstream_mpsd_reaped()) { /* a release that did not return would hold the context too */
      while (0 != libxstream_mpsd.ncontexts) clReleaseContext(libxstream_mpsd.contexts[--libxstream_mpsd.ncontexts]);
    }
    else if (0 != libxstream_mpsd.verbosity) fprintf(stderr, "INFO LIBXSTREAM-MPS: exits with releases pending\n");
    libxs_singleton_release(handle);
  }
  return result;
}
# else
LIBXSTREAM_API_INTERN void libxstream_mps_presetup(int verbosity)
{
  LIBXS_UNUSED(verbosity);
}


LIBXSTREAM_API void* libxstream_mps_icd_function(const char name[])
{
  LIBXS_UNUSED(name);
  return NULL;
}


LIBXSTREAM_API cl_int libxstream_mps_icd_platforms(cl_uint num_entries, cl_platform_id platforms[], cl_uint* num_platforms)
{
  LIBXS_UNUSED(num_entries);
  LIBXS_UNUSED(platforms);
  if (NULL != num_platforms) *num_platforms = 0;
#   if defined(CL_PLATFORM_NOT_FOUND_KHR)
  return CL_PLATFORM_NOT_FOUND_KHR;
#   else /* headers without cl_khr_icd (Apple) */
  return CL_INVALID_PLATFORM;
#   endif
}


LIBXSTREAM_API int libxstream_mps_server(void)
{
  fprintf(stderr, "ERROR LIBXSTREAM-MPS: not available in this build\n");
  return EXIT_FAILURE;
}
# endif

#endif /*__OPENCL*/
