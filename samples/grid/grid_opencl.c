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
#include <libxs/libxs_reg.h>
#include <offload/offload_library.h>
#include <offload/offload_runtime.h>
#include <grid/gpu/grid_gpu_task_list.h>
#include <grid/cpu/grid_cpu_task_list.h>
#include <grid/common/grid_sphere_cache.h>
#include <grid_kernels.h>
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "grid_opencl.h"
#if defined(__GNUC__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include <grid/common/grid_prepare_pab.h>
#include <grid/common/grid_process_vab.h>
#if defined(__GNUC__)
#  pragma GCC diagnostic pop
#endif

#if !defined(OPENCL_KERNELS_SOURCE_COLLOCATE) || !defined(OPENCL_KERNELS_SOURCE_INTEGRATE)
#  error "OpenCL kernel source code not found!"
#endif


/* Key of a code version (memset-initialized: compared bytewise) */
typedef struct {
  int kind; /* function of collocate, or -1 minus the flags of integrate */
  int ortho, lmax, tile[3];
  int engine, rows; /* integrate only */
} grid_opencl_key_t;

typedef struct {
  cl_kernel first, second; /* collocate: coefficients, tiles; integrate: coefficients, blocks */
  int engine; /* integrate: lanes of an engine, narrower than GRID_ENGINE if local memory is short */
} grid_opencl_kernels_t;

/* Tunables resolved once, and the code versions built so far */
typedef struct {
  libxs_registry_t* registry;
  int tile[3]; /* GRID_TILE: points of a tile owned by a work-group, TY x TZ rows of TX points */
  int engine; /* GRID_ENGINE: lanes of an engine taking tasks from the queue (power of two) */
  int rows; /* GRID_ROWS: rows per lane before a task gets more lanes */
  int slots; /* GRID_SLOTS: engines per compute unit */
  int nunits; /* compute units */
  size_t localmem; /* local memory per work-group */
} grid_opencl_config_t;

typedef struct {
  int npts[3], ntiles[3], tile_base;
  cl_double16 dh;
} grid_opencl_level_t;

typedef struct {
  grid_cpu_task_list* cpu; /* integrate, distributed grids, no device */
  int tile[3];
  offloadStream_t stream;
  grid_opencl_level_t* levels;
  int ntasks, nlevels, natoms, ortho, lmax, delegate, nsegs;
  int *lab, *cinfo; /* host: la_max and lb_max per task, coefficient offset and lp per task */
  int *level_start, *seg_atoms; /* host: tasks per level, atoms (and whether the same) per segment */
  double* fv; /* host: forces and virial per segment */
  int *tint_dev, *box_dev, *cinfo_dev, *tile_start_dev, *tile_task_dev;
  int *level_task_dev, *seg_start_dev, *seg_task_dev, *queue_dev;
  double *tdbl_dev, *geo_dev, *sphi_dev, *coef_dev, *fv_dev;
  size_t ncoef;
} grid_opencl_task_list_t;


static grid_opencl_config_t grid_opencl_config;


static inline void cab_add(cab_store* cab, const orbital a, const orbital b, const double value)
{
  cab->data[idx(b) * cab->n1 + idx(a)] += value;
}


static inline double cab_get(const cab_store* cab, const orbital a, const orbital b)
{
  return cab->data[idx(b) * cab->n1 + idx(a)];
}


/* segment order: block, iset, jset, then task */
static int grid_opencl_cmp(const void* a, const void* b)
{
  const int *const x = (const int*)a, *const y = (const int*)b;
  int i = 0;
  while (i < 3 && x[i] == y[i]) ++i;
  return (x[i] < y[i] ? -1 : (x[i] > y[i] ? 1 : 0));
}


static int grid_opencl_pmod(const int a, const int n)
{
  const int r = a % n;
  return (0 > r ? (r + n) : r);
}


/* Squared distance from s to the nearest covered point of each tile (-1: not covered), h = 0 if unknown. */
static void grid_opencl_reach(
  const int lo, const int hi, const double s, const double h, const int n, const int size, const int ntiles, double* d2)
{
  int u, t;
  for (t = 0; t < ntiles; ++t) d2[t] = -1.0;
  for (u = lo; u <= hi; ++u) {
    const double d = (u - s) * h, dd = d * d;
    t = grid_opencl_pmod(u, n) / size;
    if (0 > d2[t] || dd < d2[t]) d2[t] = dd;
  }
}


static void* grid_opencl_upload(const void* host, size_t size)
{
  void* result = NULL;
  offloadMalloc(&result, 0 < size ? size : 1);
  if (0 < size) offloadMemcpyHtoD(result, host, size);
  return result;
}


static int grid_opencl_arg(cl_kernel kernel, cl_uint index, const void* ptr, int* offset)
{
  libxstream_opencl_info_memptr_t info;
  size_t ioffset = 0;
  int result = libxstream_opencl_info_devptr_lock(&info, NULL /*lock*/, ptr, sizeof(double), NULL /*amount*/, &ioffset);
  if (EXIT_SUCCESS == result) result = libxstream_opencl_set_kernel_ptr(kernel, index, info.memory);
  if (NULL != offset) *offset = (int)ioffset;
  else if (0 != ioffset) result = EXIT_FAILURE;
  return result;
}


static void grid_opencl_init(void)
{
  LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, libxstream_opencl_config.lock_main);
  if (NULL == grid_opencl_config.registry) {
    const char* const env = getenv("GRID_TILE");
    int* const tile = grid_opencl_config.tile;
    if (NULL == env || 3 != sscanf(env, "%i,%i,%i", tile, tile + 1, tile + 2) || 0 >= tile[0] || 0 >= tile[1] ||
        0 >= tile[2])
    {
      tile[0] = 4;
      tile[1] = 8;
      tile[2] = 4;
    }
    { /* the preferred sub-group size fills one hardware thread */
      const char* const env_engine = getenv("GRID_ENGINE");
      const int sg = (int)libxstream_opencl_config.device.wgsize[2];
      int engine = (NULL != env_engine ? atoi(env_engine) : sg);
      if (0 >= engine || 64 < engine || 0 != (engine & (engine - 1))) engine = 16;
      grid_opencl_config.engine = engine;
    }
    {
      const char* const env_rows = getenv("GRID_ROWS");
      const char* const env_slots = getenv("GRID_SLOTS");
      const int rows = (NULL != env_rows ? atoi(env_rows) : 0), slots = (NULL != env_slots ? atoi(env_slots) : 0);
      const cl_device_id device = libxstream_opencl_config.devices[libxstream_opencl_config.device_id];
      cl_uint nunits = 0;
      cl_ulong localmem = 0;
      if (EXIT_SUCCESS != clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &localmem, NULL)) {
        localmem = 0;
      }
      grid_opencl_config.localmem = (size_t)localmem;
      grid_opencl_config.rows = (0 < rows ? rows : 8);
      grid_opencl_config.slots = (0 < slots ? slots : 8);
      if (EXIT_SUCCESS != clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &nunits, NULL)) {
        nunits = 0;
      }
      grid_opencl_config.nunits = (0 < (int)nunits ? (int)nunits : 1);
    }
    grid_opencl_config.registry = libxs_registry_create();
  }
  LIBXS_LOCK_RELEASE(LIBXS_LOCK, libxstream_opencl_config.lock_main);
}


/* Code versions are built once per key and kept: CP2K recreates task lists every step. */
static const grid_opencl_kernels_t* grid_opencl_kernels(const int kind, const int ortho, const int lmax, const int tile[3])
{
  libxs_registry_t* const registry = grid_opencl_config.registry;
  static int warned = 0;
  const grid_opencl_kernels_t* result;
  grid_opencl_key_t key;
  memset(&key, 0, sizeof(key));
  key.kind = kind;
  key.ortho = ortho;
  key.lmax = lmax;
  memcpy(key.tile, tile, sizeof(key.tile));
  if (0 > kind) {
    key.engine = grid_opencl_config.engine;
    key.rows = grid_opencl_config.rows;
  }
  result = (const grid_opencl_kernels_t*)libxs_registry_get(registry, &key, sizeof(key), libxs_registry_lock(registry));
  if (NULL == result) {
    const char* const source = (0 <= kind ? OPENCL_KERNELS_SOURCE_COLLOCATE : OPENCL_KERNELS_SOURCE_INTEGRATE);
    const char* const name1 = (0 <= kind ? "grid_collocate_coef" : "grid_integrate_coef");
    const char* const name2 = (0 <= kind ? "grid_collocate_tile" : "grid_integrate_hab");
    grid_opencl_kernels_t kernels = {NULL, NULL, 0};
    char params[512];
    int ok;
    if (0 <= kind) {
      const prepare_ldiffs ld = prepare_get_ldiffs((enum grid_func)kind);
      const int l1 = lmax + imax(ld.la_max_diff, ld.lb_max_diff);
      snprintf(params, sizeof(params),
        "-DGRID_FUNC=%i -DGRID_ORTHO=%i -DNPAB=%i -DNCAB=%i -DLMAX1=%i -DLP1=%i -DTX=%i -DTY=%i -DTZ=%i", kind,
        ortho, ncoset(lmax), ncoset(l1), l1 + 1, 2 * l1 + 1, tile[0], tile[1], tile[2]);
    }
    else {
      const int flags = -1 - kind;
      const process_ldiffs ld = process_get_ldiffs(0 != (1 & flags), 0 != (2 & flags), 0 != (4 & flags));
      const int la = lmax + ld.la_max_diff, lb = lmax + ld.lb_max_diff, lp1 = la + lb + 1;
      const size_t ncoef = (size_t)lp1 * (lp1 + 1) * (lp1 + 2) / 6;
      int engine = key.engine;
      while (1 < engine && grid_opencl_config.localmem < sizeof(double) * ncoef * engine + 1024) engine >>= 1;
      kernels.engine = engine;
      snprintf(params, sizeof(params),
        "-DGRID_ORTHO=%i -DFORCES=%i -DVIRIAL=%i -DTAU=%i -DNPAB=%i -DNCA=%i -DNCB=%i -DLMAX1A=%i -DLMAX1B=%i "
        "-DLP1=%i -DENGINE=%i -DROWS=%i",
        ortho, 1 & flags, (2 & flags) >> 1, (4 & flags) >> 2, ncoset(lmax), ncoset(la), ncoset(lb), la + 1, lb + 1,
        lp1, engine, key.rows);
    }
    ok = libxstream_opencl_kernel(0 /*source*/, source, name1, params, NULL /*options*/, NULL /*try*/,
      NULL /*try_ok*/, NULL /*exts*/, 0, &kernels.first);
    if (EXIT_SUCCESS == ok) {
      ok = libxstream_opencl_kernel(0 /*source*/, source, name2, params, NULL /*options*/, NULL /*try*/,
        NULL /*try_ok*/, NULL /*exts*/, 0, &kernels.second);
    }
    if (EXIT_SUCCESS != ok) {
      if (NULL != kernels.first) clReleaseKernel(kernels.first);
      kernels.first = kernels.second = NULL; /* registered as well: no second attempt */
      if (0 == warned) {
        fprintf(stderr, "WARNING GRID/OpenCL: kernels failed to build, the host takes over\n");
        warned = 1;
      }
    }
    result = (const grid_opencl_kernels_t*)libxs_registry_set(
      registry, &key, sizeof(key), &kernels, sizeof(kernels), libxs_registry_lock(registry));
  }
  return (NULL != result && NULL != result->first) ? result : NULL;
}


void grid_gpu_create_task_list(const bool ortho, const int ntasks, const int nlevels, const int natoms, const int nkinds,
  const int nblocks, const int* block_offsets, const double* atom_positions, const int* atom_kinds,
  const grid_basis_set** basis_sets, const int* level_list, const int* iatom_list, const int* jatom_list,
  const int* iset_list, const int* jset_list, const int* ipgf_list, const int* jpgf_list, const int* border_mask_list,
  const int* block_num_list, const double* radius_list, const double* rab_list, const int* npts_global,
  const int* npts_local, const int* shift_local, const int* border_width, const double* dh, const double* dh_inv,
  grid_gpu_task_list** ptr)
{
  grid_opencl_task_list_t* const tl = (grid_opencl_task_list_t*)calloc(1, sizeof(grid_opencl_task_list_t));
  int *tint = NULL, *box = NULL, *tile_start = NULL, *tile_task = NULL, *sphi_offset = NULL;
  int *level_task = NULL, *seg_key = NULL, *seg_start = NULL;
  double *tdbl = NULL, *geo = NULL, *sphi = NULL, *d2 = NULL;
  int i, k, level, ntiles = 0, maxtiles = 0, nsphi = 0;
  assert(NULL != ptr && NULL != tl);
  if (NULL != *ptr) grid_gpu_free_task_list(*ptr);
  offload_init();
  offload_activate_chosen_device();
  grid_opencl_init();
  memcpy(tl->tile, grid_opencl_config.tile, sizeof(tl->tile));
  grid_cpu_create_task_list(ortho, ntasks, nlevels, natoms, nkinds, nblocks, block_offsets,
    (const double(*)[3])atom_positions, atom_kinds, basis_sets, level_list, iatom_list, jatom_list, iset_list,
    jset_list, ipgf_list, jpgf_list, border_mask_list, block_num_list, radius_list, (const double(*)[3])rab_list,
    (const int(*)[3])npts_global, (const int(*)[3])npts_local, (const int(*)[3])shift_local,
    (const int(*)[3])border_width, (const double(*)[3][3])dh, (const double(*)[3][3])dh_inv, &tl->cpu);
  tl->ntasks = ntasks;
  tl->nlevels = nlevels;
  tl->natoms = natoms;
  tl->ortho = (ortho ? 1 : 0);
  tl->levels = (grid_opencl_level_t*)calloc(0 < nlevels ? nlevels : 1, sizeof(grid_opencl_level_t));
  assert(NULL != tl->levels);
  for (level = 0; level < nlevels; ++level) {
    grid_opencl_level_t* const lv = &tl->levels[level];
    for (k = 0; k < 3; ++k) {
      if (npts_local[3 * level + k] != npts_global[3 * level + k] || 0 != shift_local[3 * level + k] ||
          0 != border_width[3 * level + k])
      {
        tl->delegate = 1; /* distributed grid */
      }
      lv->npts[k] = npts_local[3 * level + k];
      lv->ntiles[k] = (lv->npts[k] + tl->tile[k] - 1) / tl->tile[k];
      maxtiles = imax(maxtiles, lv->ntiles[k]);
    }
    for (k = 0; k < 9; ++k) lv->dh.s[k] = dh[9 * level + k];
    lv->tile_base = ntiles;
    ntiles += lv->ntiles[0] * lv->ntiles[1] * lv->ntiles[2];
  }
  for (i = 0; i < ntasks; ++i) {
    if (0 != border_mask_list[i]) tl->delegate = 1;
  }
  tl->stream = NULL;
  if (0 == tl->delegate) {
    tl->lab = (int*)malloc(sizeof(int) * 2 * (0 < ntasks ? ntasks : 1));
    tl->cinfo = (int*)malloc(sizeof(int) * 2 * (0 < ntasks ? ntasks : 1));
    tint = (int*)malloc(sizeof(int) * GRID_OPENCL_NI * (0 < ntasks ? ntasks : 1));
    box = (int*)malloc(sizeof(int) * GRID_OPENCL_NB * (0 < ntasks ? ntasks : 1));
    tdbl = (double*)malloc(sizeof(double) * GRID_OPENCL_ND * (0 < ntasks ? ntasks : 1));
    geo = (double*)malloc(sizeof(double) * GRID_OPENCL_NG * (0 < ntasks ? ntasks : 1));
    tile_start = (int*)calloc(ntiles + 1, sizeof(int));
    sphi_offset = (int*)malloc(sizeof(int) * (0 < nkinds ? nkinds : 1));
    d2 = (double*)malloc(sizeof(double) * 3 * (0 < maxtiles ? maxtiles : 1));
    assert(NULL != tl->lab && NULL != tl->cinfo && NULL != tint && NULL != box && NULL != tdbl && NULL != geo);
    assert(NULL != tile_start && NULL != sphi_offset && NULL != d2);
    for (k = 0; k < nkinds; ++k) {
      sphi_offset[k] = nsphi;
      nsphi += basis_sets[k]->nsgf * basis_sets[k]->maxco;
    }
    sphi = (double*)malloc(sizeof(double) * (0 < nsphi ? nsphi : 1));
    assert(NULL != sphi);
    for (k = 0; k < nkinds; ++k) {
      memcpy(sphi + sphi_offset[k], basis_sets[k]->sphi, sizeof(double) * basis_sets[k]->nsgf * basis_sets[k]->maxco);
    }
    for (i = 0; i < ntasks; ++i) {
      const int iatom = iatom_list[i] - 1, jatom = jatom_list[i] - 1;
      const int iset = iset_list[i] - 1, jset = jset_list[i] - 1;
      const int ipgf = ipgf_list[i] - 1, jpgf = jpgf_list[i] - 1;
      const int ikind = atom_kinds[iatom] - 1, jkind = atom_kinds[jatom] - 1;
      const grid_basis_set *const ib = basis_sets[ikind], *const jb = basis_sets[jkind];
      const int la_max = ib->lmax[iset], lb_max = jb->lmax[jset];
      const int sgfa = ib->first_sgf[iset] - 1, sgfb = jb->first_sgf[jset] - 1;
      const int block_offset = block_offsets[block_num_list[i] - 1];
      const double zeta = ib->zet[iset * ib->maxpgf + ipgf], zetb = jb->zet[jset * jb->maxpgf + jpgf];
      const double zetp = zeta + zetb, f = zetb / zetp;
      const double *const ra = atom_positions + 3 * iatom, *const rab = rab_list + 3 * i;
      const double rab2 = rab[0] * rab[0] + rab[1] * rab[1] + rab[2] * rab[2];
      const double* const di = dh_inv + 9 * (level_list[i] - 1);
      const double cutoff = radius_list[i] + grid_sphere_spacing(dh + 9 * (level_list[i] - 1));
      int* const ti = tint + GRID_OPENCL_NI * i;
      double* const td = tdbl + GRID_OPENCL_ND * i;
      double* const gt = geo + GRID_OPENCL_NG * i;
      double dh_max = 0.0;
      tl->lab[2 * i + 0] = la_max;
      tl->lab[2 * i + 1] = lb_max;
      tl->lmax = imax(tl->lmax, imax(la_max, lb_max));
      ti[0] = la_max;
      ti[1] = ib->lmin[iset];
      ti[2] = lb_max;
      ti[3] = jb->lmin[jset];
      if (iatom <= jatom) { /* transposed block */
        ti[4] = block_offset + sgfb * ib->nsgf + sgfa;
        ti[5] = 1;
        ti[6] = ib->nsgf;
      }
      else {
        ti[4] = block_offset + sgfa * jb->nsgf + sgfb;
        ti[5] = jb->nsgf;
        ti[6] = 1;
      }
      ti[7] = ib->nsgf_set[iset];
      ti[8] = jb->nsgf_set[jset];
      ti[9] = sphi_offset[ikind] + sgfa * ib->maxco + ipgf * ncoset(la_max);
      ti[10] = ib->maxco;
      ti[11] = sphi_offset[jkind] + sgfb * jb->maxco + jpgf * ncoset(lb_max);
      ti[12] = jb->maxco;
      ti[13] = (iatom == jatom ? 1 : 0);
      for (k = 0; k < 3; ++k) {
        td[k] = ra[k];
        td[3 + k] = ra[k] + rab[k];
        td[6 + k] = ra[k] + f * rab[k];
        td[12 + k] = rab[k];
      }
      td[9] = zeta;
      td[10] = zetb;
      td[11] = (iatom == jatom ? 1.0 : 2.0) * exp(-zeta * f * rab2);
      for (k = 0; k < 3; ++k) {
        const double s = td[6] * di[k] + td[7] * di[3 + k] + td[8] * di[6 + k];
        const double e = cutoff * sqrt(di[k] * di[k] + di[3 + k] * di[3 + k] + di[6 + k] * di[6 + k]);
        gt[k] = s;
        box[GRID_OPENCL_NB * i + k] = (int)ceil(s - e);
        box[GRID_OPENCL_NB * i + 3 + k] = (int)floor(s + e);
      }
      gt[3] = zetp;
      gt[4] = cutoff * cutoff;
      for (k = 0; k < 9; ++k) dh_max = fmax(dh_max, fabs(dh[9 * (level_list[i] - 1) + k]));
      if (2.0 * radius_list[i] < dh_max) gt[4] = 0.0; /* as CPU: skip a task narrower than the grid */
    }
    /* tile lists in task order (deterministic sum), first counted, then filled */
    tile_task = NULL;
    for (k = 0; k < 2; ++k) {
      int* const pos = (0 == k ? tile_start : (int*)malloc(sizeof(int) * (ntiles + 1)));
      assert(NULL != pos);
      if (0 != k) {
        int n = 0, t;
        for (t = 0; t < ntiles; ++t) {
          const int c = tile_start[t];
          tile_start[t] = n;
          pos[t] = n;
          n += c;
        }
        tile_start[ntiles] = n;
        tile_task = (int*)malloc(sizeof(int) * (0 < n ? n : 1));
        assert(NULL != tile_task);
      }
      for (i = 0; i < ntasks; ++i) {
        const grid_opencl_level_t* const lv = &tl->levels[level_list[i] - 1];
        const double* const dl = dh + 9 * (level_list[i] - 1);
        const double* const gt = geo + GRID_OPENCL_NG * i;
        const int* const bt = box + GRID_OPENCL_NB * i;
        int j;
        if (0 < gt[4]) {
          int t0, t1, t2;
          for (j = 0; j < 3; ++j) {
            grid_opencl_reach(bt[j], bt[3 + j], gt[j], 0 != tl->ortho ? fabs(dl[4 * j]) : 0.0, lv->npts[j], tl->tile[j],
              lv->ntiles[j], d2 + j * maxtiles);
          }
          for (t2 = 0; t2 < lv->ntiles[2]; ++t2) {
            const double z2 = d2[2 * maxtiles + t2];
            for (t1 = 0; t1 < lv->ntiles[1] && 0 <= z2 && z2 < gt[4]; ++t1) {
              const double y2 = d2[maxtiles + t1];
              for (t0 = 0; t0 < lv->ntiles[0] && 0 <= y2 && z2 + y2 < gt[4]; ++t0) {
                const double x2 = d2[t0];
                if (0 <= x2 && z2 + y2 + x2 < gt[4]) {
                  const int t = lv->tile_base + (t2 * lv->ntiles[1] + t1) * lv->ntiles[0] + t0;
                  if (0 == k) ++pos[t];
                  else tile_task[pos[t]++] = i;
                }
              }
            }
          }
        }
      }
      if (0 != k) free(pos);
    }
    level_task = (int*)malloc(sizeof(int) * (0 < ntasks ? ntasks : 1));
    seg_key = (int*)malloc(sizeof(int) * 4 * (0 < ntasks ? ntasks : 1));
    seg_start = (int*)malloc(sizeof(int) * (ntasks + 1));
    tl->level_start = (int*)calloc(nlevels + 1, sizeof(int));
    assert(NULL != level_task && NULL != seg_key && NULL != seg_start && NULL != tl->level_start);
    for (i = 0; i < ntasks; ++i) ++tl->level_start[level_list[i]];
    for (level = 0; level < nlevels; ++level) tl->level_start[level + 1] += tl->level_start[level];
    for (level = 0, k = 0; level < nlevels; ++level) {
      for (i = 0; i < ntasks; ++i) {
        if (level_list[i] - 1 == level) level_task[k++] = i;
      }
    }
    for (i = 0; i < ntasks; ++i) {
      seg_key[4 * i + 0] = block_num_list[i];
      seg_key[4 * i + 1] = iset_list[i];
      seg_key[4 * i + 2] = jset_list[i];
      seg_key[4 * i + 3] = i;
    }
    qsort(seg_key, ntasks, 4 * sizeof(int), grid_opencl_cmp);
    for (i = 0; i < ntasks; ++i) {
      if (0 == i || 0 != memcmp(seg_key + 4 * i, seg_key + 4 * (i - 1), 3 * sizeof(int))) seg_start[tl->nsegs++] = i;
    }
    for (i = 0; i < ntasks; ++i) seg_key[i] = seg_key[4 * i + 3]; /* tasks in segment order */
    seg_start[tl->nsegs] = ntasks;
    tl->seg_atoms = (int*)malloc(sizeof(int) * 3 * (0 < tl->nsegs ? tl->nsegs : 1));
    tl->fv = (double*)malloc(sizeof(double) * 24 * (0 < tl->nsegs ? tl->nsegs : 1));
    assert(NULL != tl->seg_atoms && NULL != tl->fv);
    for (i = 0; i < tl->nsegs; ++i) {
      const int t = seg_key[seg_start[i]];
      tl->seg_atoms[3 * i + 0] = iatom_list[t] - 1;
      tl->seg_atoms[3 * i + 1] = jatom_list[t] - 1;
      tl->seg_atoms[3 * i + 2] = (iatom_list[t] == jatom_list[t] ? 1 : 0);
    }
    offloadStreamCreate(&tl->stream);
    tl->level_task_dev = (int*)grid_opencl_upload(level_task, sizeof(int) * ntasks);
    tl->seg_start_dev = (int*)grid_opencl_upload(seg_start, sizeof(int) * (tl->nsegs + 1));
    tl->seg_task_dev = (int*)grid_opencl_upload(seg_key, sizeof(int) * ntasks);
    offloadMalloc((void**)&tl->fv_dev, sizeof(double) * 24 * (0 < tl->nsegs ? tl->nsegs : 1));
    offloadMalloc((void**)&tl->queue_dev, sizeof(int));
    tl->tint_dev = (int*)grid_opencl_upload(tint, sizeof(int) * GRID_OPENCL_NI * ntasks);
    tl->tdbl_dev = (double*)grid_opencl_upload(tdbl, sizeof(double) * GRID_OPENCL_ND * ntasks);
    tl->geo_dev = (double*)grid_opencl_upload(geo, sizeof(double) * GRID_OPENCL_NG * ntasks);
    tl->box_dev = (int*)grid_opencl_upload(box, sizeof(int) * GRID_OPENCL_NB * ntasks);
    tl->sphi_dev = (double*)grid_opencl_upload(sphi, sizeof(double) * nsphi);
    tl->tile_start_dev = (int*)grid_opencl_upload(tile_start, sizeof(int) * (ntiles + 1));
    tl->tile_task_dev = (int*)grid_opencl_upload(tile_task, sizeof(int) * tile_start[ntiles]);
    offloadMalloc((void**)&tl->cinfo_dev, sizeof(int) * 2 * (0 < ntasks ? ntasks : 1));
  }
  free(tint);
  free(box);
  free(tdbl);
  free(geo);
  free(sphi);
  free(sphi_offset);
  free(tile_start);
  free(tile_task);
  free(d2);
  free(level_task);
  free(seg_key);
  free(seg_start);
  *ptr = (grid_gpu_task_list*)tl;
}


void grid_gpu_free_task_list(grid_gpu_task_list* ptr)
{
  grid_opencl_task_list_t* const tl = (grid_opencl_task_list_t*)ptr;
  if (NULL != tl) {
    grid_cpu_free_task_list(tl->cpu);
    if (NULL != tl->stream) {
      offloadStreamSynchronize(tl->stream);
      offloadFree(tl->tint_dev);
      offloadFree(tl->tdbl_dev);
      offloadFree(tl->geo_dev);
      offloadFree(tl->box_dev);
      offloadFree(tl->sphi_dev);
      offloadFree(tl->tile_start_dev);
      offloadFree(tl->tile_task_dev);
      offloadFree(tl->cinfo_dev);
      offloadFree(tl->level_task_dev);
      offloadFree(tl->seg_start_dev);
      offloadFree(tl->seg_task_dev);
      offloadFree(tl->fv_dev);
      offloadFree(tl->queue_dev);
      if (NULL != tl->coef_dev) offloadFree(tl->coef_dev);
      offloadStreamDestroy(tl->stream);
    }
    free(tl->lab);
    free(tl->cinfo);
    free(tl->level_start);
    free(tl->seg_atoms);
    free(tl->fv);
    free(tl->levels);
    free(tl);
  }
}


void grid_gpu_collocate_task_list(const grid_gpu_task_list* ptr, const enum grid_func func, const int nlevels,
  const offload_buffer* pab_blocks, offload_buffer** grids)
{
  grid_opencl_task_list_t* const tl = (grid_opencl_task_list_t*)ptr;
  const grid_opencl_kernels_t* const kernels = (0 == tl->delegate ? grid_opencl_kernels(func, tl->ortho, tl->lmax, tl->tile)
                                                                   : NULL);
  assert(NULL != tl && tl->nlevels == nlevels);
  if (NULL != kernels && NULL != pab_blocks->device_buffer) {
    const prepare_ldiffs ld = prepare_get_ldiffs(func);
    libxstream_stream_t* const stream = (libxstream_stream_t*)tl->stream;
    const size_t local[3] = {1, (size_t)tl->tile[1], (size_t)tl->tile[2]};
    const cl_int ntasks = tl->ntasks;
    size_t ncoef = 0, global[3];
    cl_int pab_offset = 0, grid_offset = 0, n[3];
    int result = EXIT_SUCCESS, i, level;
    offload_activate_chosen_device();
    for (i = 0; i < ntasks; ++i) {
      const int lp = tl->lab[2 * i] + ld.la_max_diff + tl->lab[2 * i + 1] + ld.lb_max_diff;
      tl->cinfo[2 * i + 0] = (int)ncoef;
      tl->cinfo[2 * i + 1] = lp;
      ncoef += ncoset(lp);
    }
    if (tl->ncoef < ncoef || NULL == tl->coef_dev) {
      if (NULL != tl->coef_dev) {
        offloadStreamSynchronize(tl->stream);
        offloadFree(tl->coef_dev);
      }
      offloadMalloc((void**)&tl->coef_dev, sizeof(double) * (0 < ncoef ? ncoef : 1));
      tl->ncoef = ncoef;
    }
    if (0 < ntasks) offloadMemcpyAsyncHtoD(tl->cinfo_dev, tl->cinfo, sizeof(int) * 2 * ntasks, tl->stream);
    if (pab_blocks->host_buffer != pab_blocks->device_buffer) {
      offloadMemcpyAsyncHtoD(pab_blocks->device_buffer, pab_blocks->host_buffer, pab_blocks->size, tl->stream);
    }
    if (0 < ntasks) {
      result |= clSetKernelArg(kernels->first, 0, sizeof(cl_int), &ntasks);
      result |= grid_opencl_arg(kernels->first, 1, pab_blocks->device_buffer, &pab_offset);
      result |= clSetKernelArg(kernels->first, 2, sizeof(cl_int), &pab_offset);
      result |= grid_opencl_arg(kernels->first, 3, tl->sphi_dev, NULL);
      result |= grid_opencl_arg(kernels->first, 4, tl->tint_dev, NULL);
      result |= grid_opencl_arg(kernels->first, 5, tl->tdbl_dev, NULL);
      result |= grid_opencl_arg(kernels->first, 6, tl->cinfo_dev, NULL);
      result |= grid_opencl_arg(kernels->first, 7, tl->coef_dev, NULL);
      global[0] = (size_t)ntasks;
      result |= libxstream_opencl_launch_work(stream, kernels->first, 1, NULL, global, NULL, 0, NULL, NULL, 0, 0);
    }
    for (level = 0; level < nlevels; ++level) {
      const grid_opencl_level_t* const lv = &tl->levels[level];
      for (i = 0; i < 3; ++i) {
        n[i] = lv->npts[i];
        global[i] = (size_t)lv->ntiles[i] * local[i]; /* x: one work-item per row of the tile */
      }
      result |= clSetKernelArg(kernels->second, 0, sizeof(cl_int), &n[0]);
      result |= clSetKernelArg(kernels->second, 1, sizeof(cl_int), &n[1]);
      result |= clSetKernelArg(kernels->second, 2, sizeof(cl_int), &n[2]);
      result |= clSetKernelArg(kernels->second, 3, sizeof(cl_double16), &lv->dh);
      result |= clSetKernelArg(kernels->second, 4, sizeof(cl_int), &lv->tile_base);
      result |= grid_opencl_arg(kernels->second, 5, tl->tile_start_dev, NULL);
      result |= grid_opencl_arg(kernels->second, 6, tl->tile_task_dev, NULL);
      result |= grid_opencl_arg(kernels->second, 7, tl->geo_dev, NULL);
      result |= grid_opencl_arg(kernels->second, 8, tl->box_dev, NULL);
      result |= grid_opencl_arg(kernels->second, 9, tl->cinfo_dev, NULL);
      result |= grid_opencl_arg(kernels->second, 10, tl->coef_dev, NULL);
      result |= grid_opencl_arg(kernels->second, 11, grids[level]->device_buffer, &grid_offset);
      result |= clSetKernelArg(kernels->second, 12, sizeof(cl_int), &grid_offset);
      if (0 < global[0] && 0 < global[1] && 0 < global[2]) {
        result |= libxstream_opencl_launch_work(stream, kernels->second, 3, NULL, global, local, 0, NULL, NULL, 0, 0);
      }
      if (grids[level]->host_buffer != grids[level]->device_buffer) {
        offloadMemcpyAsyncDtoH(grids[level]->host_buffer, grids[level]->device_buffer,
          sizeof(double) * (size_t)n[0] * n[1] * n[2], tl->stream);
      }
    }
    offloadStreamSynchronize(tl->stream);
    if (EXIT_SUCCESS != result) {
      fprintf(stderr, "ERROR GRID/OpenCL: collocation failed\n");
      abort();
    }
  }
  else {
    grid_cpu_collocate_task_list(tl->cpu, func, nlevels, pab_blocks, grids);
  }
}


void grid_gpu_integrate_task_list(const grid_gpu_task_list* ptr, const bool compute_tau, const int nlevels,
  const offload_buffer* pab_blocks, const offload_buffer** grids, offload_buffer* hab_blocks, double* forces,
  double* virial)
{
  grid_opencl_task_list_t* const tl = (grid_opencl_task_list_t*)ptr;
  const int flags = (NULL != forces ? 1 : 0) | (NULL != virial ? 2 : 0) | (compute_tau ? 4 : 0);
  const grid_opencl_kernels_t* const kernels = (0 == tl->delegate ? grid_opencl_kernels(-1 - flags, tl->ortho, tl->lmax, tl->tile)
                                                                   : NULL);
  assert(NULL != tl && tl->nlevels == nlevels && NULL != hab_blocks);
  if (NULL != kernels && NULL != hab_blocks->device_buffer && (NULL == pab_blocks || NULL != pab_blocks->device_buffer)) {
    const process_ldiffs ld = process_get_ldiffs(NULL != forces, NULL != virial, compute_tau);
    libxstream_stream_t* const stream = (libxstream_stream_t*)tl->stream;
    const size_t local = (size_t)kernels->engine;
    const cl_int ntasks = tl->ntasks, nsegs = tl->nsegs;
    const void* const pab = (NULL != pab_blocks ? (const void*)pab_blocks->device_buffer : (const void*)tl->sphi_dev);
    size_t ncoef = 0, global;
    cl_int pab_offset = 0, hab_offset = 0, grid_offset = 0, n[3];
    int result = EXIT_SUCCESS, i, k, level;
    offload_activate_chosen_device();
    for (i = 0; i < ntasks; ++i) {
      const int lp = tl->lab[2 * i] + ld.la_max_diff + tl->lab[2 * i + 1] + ld.lb_max_diff;
      tl->cinfo[2 * i + 0] = (int)ncoef;
      tl->cinfo[2 * i + 1] = lp;
      ncoef += ncoset(lp);
    }
    if (tl->ncoef < ncoef || NULL == tl->coef_dev) {
      if (NULL != tl->coef_dev) {
        offloadStreamSynchronize(tl->stream);
        offloadFree(tl->coef_dev);
      }
      offloadMalloc((void**)&tl->coef_dev, sizeof(double) * (0 < ncoef ? ncoef : 1));
      tl->ncoef = ncoef;
    }
    if (0 < ntasks) offloadMemcpyAsyncHtoD(tl->cinfo_dev, tl->cinfo, sizeof(int) * 2 * ntasks, tl->stream);
    if (NULL != pab_blocks && 0 != flags % 4 && pab_blocks->host_buffer != pab_blocks->device_buffer) {
      offloadMemcpyAsyncHtoD(pab_blocks->device_buffer, pab_blocks->host_buffer, pab_blocks->size, tl->stream);
    }
    for (level = 0; level < nlevels; ++level) {
      const grid_opencl_level_t* const lv = &tl->levels[level];
      const cl_int task_base = tl->level_start[level], nlevel = tl->level_start[level + 1] - task_base;
      const int nengines = imin(grid_opencl_config.slots * grid_opencl_config.nunits, nlevel);
      for (i = 0; i < 3; ++i) n[i] = lv->npts[i];
      if (grids[level]->host_buffer != grids[level]->device_buffer) {
        offloadMemcpyAsyncHtoD(grids[level]->device_buffer, grids[level]->host_buffer,
          sizeof(double) * (size_t)n[0] * n[1] * n[2], tl->stream);
      }
      result |= clSetKernelArg(kernels->first, 0, sizeof(cl_int), &n[0]);
      result |= clSetKernelArg(kernels->first, 1, sizeof(cl_int), &n[1]);
      result |= clSetKernelArg(kernels->first, 2, sizeof(cl_int), &n[2]);
      result |= clSetKernelArg(kernels->first, 3, sizeof(cl_double16), &lv->dh);
      result |= clSetKernelArg(kernels->first, 4, sizeof(cl_int), &task_base);
      result |= clSetKernelArg(kernels->first, 5, sizeof(cl_int), &nlevel);
      result |= grid_opencl_arg(kernels->first, 6, tl->level_task_dev, NULL);
      result |= grid_opencl_arg(kernels->first, 7, tl->geo_dev, NULL);
      result |= grid_opencl_arg(kernels->first, 8, tl->box_dev, NULL);
      result |= grid_opencl_arg(kernels->first, 9, tl->cinfo_dev, NULL);
      result |= grid_opencl_arg(kernels->first, 10, grids[level]->device_buffer, &grid_offset);
      result |= clSetKernelArg(kernels->first, 11, sizeof(cl_int), &grid_offset);
      result |= grid_opencl_arg(kernels->first, 12, tl->coef_dev, NULL);
      result |= grid_opencl_arg(kernels->first, 13, tl->queue_dev, NULL);
      global = (size_t)nengines * local; /* persistent engines: the queue distributes the tasks */
      if (0 < global) {
        offloadMemsetAsync(tl->queue_dev, 0, sizeof(int), tl->stream);
        result |= libxstream_opencl_launch_work(stream, kernels->first, 1, NULL, &global, &local, 0, NULL, NULL, 0, 0);
      }
    }
    if (0 < hab_blocks->size) offloadMemsetAsync(hab_blocks->device_buffer, 0, hab_blocks->size, tl->stream);
    if (0 < nsegs) {
      result |= clSetKernelArg(kernels->second, 0, sizeof(cl_int), &nsegs);
      result |= grid_opencl_arg(kernels->second, 1, tl->seg_start_dev, NULL);
      result |= grid_opencl_arg(kernels->second, 2, tl->seg_task_dev, NULL);
      result |= grid_opencl_arg(kernels->second, 3, pab, &pab_offset);
      result |= clSetKernelArg(kernels->second, 4, sizeof(cl_int), &pab_offset);
      result |= grid_opencl_arg(kernels->second, 5, tl->sphi_dev, NULL);
      result |= grid_opencl_arg(kernels->second, 6, tl->tint_dev, NULL);
      result |= grid_opencl_arg(kernels->second, 7, tl->tdbl_dev, NULL);
      result |= grid_opencl_arg(kernels->second, 8, tl->cinfo_dev, NULL);
      result |= grid_opencl_arg(kernels->second, 9, tl->coef_dev, NULL);
      result |= grid_opencl_arg(kernels->second, 10, hab_blocks->device_buffer, &hab_offset);
      result |= clSetKernelArg(kernels->second, 11, sizeof(cl_int), &hab_offset);
      result |= grid_opencl_arg(kernels->second, 12, tl->fv_dev, NULL);
      global = (size_t)nsegs;
      result |= libxstream_opencl_launch_work(stream, kernels->second, 1, NULL, &global, NULL, 0, NULL, NULL, 0, 0);
      if (0 != flags % 4) offloadMemcpyAsyncDtoH(tl->fv, tl->fv_dev, sizeof(double) * 24 * nsegs, tl->stream);
    }
    if (0 < hab_blocks->size && hab_blocks->host_buffer != hab_blocks->device_buffer) {
      offloadMemcpyAsyncDtoH(hab_blocks->host_buffer, hab_blocks->device_buffer, hab_blocks->size, tl->stream);
    }
    offloadStreamSynchronize(tl->stream);
    if (EXIT_SUCCESS != result) {
      fprintf(stderr, "ERROR GRID/OpenCL: integration failed\n");
      abort();
    }
    if (NULL != forces) memset(forces, 0, sizeof(double) * 3 * tl->natoms);
    if (NULL != virial) memset(virial, 0, sizeof(double) * 9);
    for (i = 0; i < nsegs && 0 != flags % 4; ++i) { /* segment order: reproducible */
      const int* const atoms = tl->seg_atoms + 3 * i;
      const double* const fv = tl->fv + 24 * i;
      const double scalef = (0 != atoms[2] ? 1.0 : 2.0);
      if (NULL != forces) {
        for (k = 0; k < 3; ++k) {
          forces[3 * atoms[0] + k] += scalef * fv[k];
          forces[3 * atoms[1] + k] += scalef * fv[3 + k];
        }
      }
      if (NULL != virial) {
        for (k = 0; k < 9; ++k) virial[k] += scalef * fv[6 + k] + scalef * fv[15 + k];
      }
    }
  }
  else {
    grid_cpu_integrate_task_list(tl->cpu, compute_tau, tl->natoms, nlevels, pab_blocks, grids, hab_blocks,
      (double(*)[3])forces, (double(*)[3])virial);
  }
}
