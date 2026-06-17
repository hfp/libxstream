/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "stencil_opencl.h"
#include <libxs/libxs_timer.h>
#include <libxs/libxs_mem.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


/* Velocity model types */
typedef enum {
  VEL_CONSTANT,
  VEL_GRADIENT,
  VEL_LAYERED,
  VEL_FILE
} vel_model_t;

static void fd_weights_2nd(double* w, int radius, double h);
static int load_velocity_file(const char* path, float* vel, int nx, int ny, int nz);
static void generate_velocity(float* vel, int nx, int ny, int nz,
                              vel_model_t model, float v_top, float v_bot);
static void inject_source(float* p, int nx, int ny, int nz,
                          float dt, int tstep, float freq);
static void usage(const char* prog);


int main(int argc, char* argv[])
{
  stencil_context_t ctx;
  int result = EXIT_SUCCESS;
  const int blk = STENCIL_BLK;
  const int radius = STENCIL_RADIUS;
  int nx = 256, ny = 256, nz = 256;
  int nterms = 3;
  int ntsteps = 100;
  int warmup = 5;
  double h = 10.0;
  float v_min = 1500.0f, v_max = 4500.0f;
  float freq = 25.0f;
  vel_model_t vel_model = VEL_GRADIENT;
  const char* vel_file = NULL;
  double fd_w[2 * STENCIL_RADIUS + 1];
  int method_override = -1;
  int ndevices = 0;
  int initialized = 0;
  int trace = 0;
  int argi;

  LIBXS_MEMZERO(&ctx);
  trace = (NULL != getenv("STENCIL_TRACE"));

  for (argi = 1; argi < argc; ++argi) {
    if (0 == strcmp(argv[argi], "-n") && argi + 1 < argc) {
      nx = ny = nz = atoi(argv[++argi]);
    }
    else if (0 == strcmp(argv[argi], "-nx") && argi + 1 < argc) {
      nx = atoi(argv[++argi]);
    }
    else if (0 == strcmp(argv[argi], "-ny") && argi + 1 < argc) {
      ny = atoi(argv[++argi]);
    }
    else if (0 == strcmp(argv[argi], "-nz") && argi + 1 < argc) {
      nz = atoi(argv[++argi]);
    }
    else if (0 == strcmp(argv[argi], "-t") && argi + 1 < argc) {
      ntsteps = atoi(argv[++argi]);
    }
    else if (0 == strcmp(argv[argi], "-d") && argi + 1 < argc) {
      nterms = atoi(argv[++argi]);
    }
    else if (0 == strcmp(argv[argi], "-h") && argi + 1 < argc) {
      h = atof(argv[++argi]);
    }
    else if (0 == strcmp(argv[argi], "-v") && argi + 1 < argc) {
      const char* arg = argv[++argi];
      if (0 == strcmp(arg, "const")) vel_model = VEL_CONSTANT;
      else if (0 == strcmp(arg, "grad")) vel_model = VEL_GRADIENT;
      else if (0 == strcmp(arg, "layered")) vel_model = VEL_LAYERED;
      else { vel_model = VEL_FILE; vel_file = arg; }
    }
    else if (0 == strcmp(argv[argi], "-vmin") && argi + 1 < argc) {
      v_min = (float)atof(argv[++argi]);
    }
    else if (0 == strcmp(argv[argi], "-vmax") && argi + 1 < argc) {
      v_max = (float)atof(argv[++argi]);
    }
    else if (0 == strcmp(argv[argi], "-f") && argi + 1 < argc) {
      freq = (float)atof(argv[++argi]);
    }
    else if (0 == strcmp(argv[argi], "-w") && argi + 1 < argc) {
      warmup = atoi(argv[++argi]);
    }
    else if (0 == strcmp(argv[argi], "-seg-salt")) {
      nx = 676; ny = 676; nz = 210; h = 20.0;
      v_min = 1500.0f; v_max = 4500.0f;
      vel_model = VEL_LAYERED;
    }
    else if (0 == strcmp(argv[argi], "-overthrust")) {
      nx = 801; ny = 801; nz = 187; h = 25.0;
      v_min = 2000.0f; v_max = 6000.0f;
      vel_model = VEL_GRADIENT;
    }
    else if (0 == strcmp(argv[argi], "-m") && argi + 1 < argc) {
      method_override = atoi(argv[++argi]);
    }
    else if (0 == strcmp(argv[argi], "--help")) {
      usage(argv[0]); return EXIT_SUCCESS;
    }
  }

  {
    const double gpoints = (double)nx * ny * nz * 1.0e-9;
    const float dt = 0.6f * (float)h
                   / (v_max * (float)sqrt(3.0) * (2.0f * radius + 1.0f));
    printf("Stencil BF16-DPAS benchmark\n");
    printf("  Grid:       %d x %d x %d (%.3f GPoints)\n",
           nx, ny, nz, gpoints);
    printf("  Block:      %d^3, radius=%d (order %d)\n",
           blk, radius, 2 * radius);
    printf("  Digits:     A=%d, X=%d (products/dim=%d, total=%d)\n",
           STENCIL_NDIGITS_A, STENCIL_NDIGITS_X,
           STENCIL_NDIGITS_A * STENCIL_NDIGITS_X,
           nterms * STENCIL_NDIGITS_A * STENCIL_NDIGITS_X);
    printf("  Terms:      %d (%s)\n", nterms,
           nterms <= 3 ? "isotropic" : "TTI");
    printf("  Steps:      %d (+ %d warmup)\n", ntsteps, warmup);
    printf("  Spacing:    %.1f m, dt=%.3e s\n", h, (double)dt);
    printf("  Velocity:   %.0f - %.0f m/s", (double)v_min, (double)v_max);
    if (NULL != vel_file) printf(" (file: %s)", vel_file);
    printf("\n");
  }

  fd_weights_2nd(fd_w, radius, h);

  if (EXIT_SUCCESS == result) {
    result = libxstream_init();
    if (EXIT_SUCCESS == result) initialized = 1;
  }
  if (EXIT_SUCCESS == result) {
    result = libxstream_device_count(&ndevices);
    if (EXIT_SUCCESS == result && 0 < ndevices) {
      result = libxstream_device_set_active(0);
    }
    else if (EXIT_SUCCESS == result) {
      fprintf(stderr, "ERROR: no ACC device found\n");
      result = EXIT_FAILURE;
    }
  }
  if (EXIT_SUCCESS == result) {
    char devname[256] = "";
    libxstream_opencl_device_name(
      libxstream_opencl_config.devices[libxstream_opencl_config.device_id],
      devname, sizeof(devname), NULL, 0, 1);
    printf("  Device:     %s\n\n", devname);
  }
  if (EXIT_SUCCESS == result) {
    result = stencil_init(&ctx, 1, method_override);
  }
  if (EXIT_SUCCESS == result) {
    const char* mnames[] = {"direct", "staged-r1", "staged-r2", "staged-fit"};
    ctx.nterms = nterms;
    printf("  Method:     %s (K=%d, r=%d, strips/WG=%d)\n",
           mnames[(int)ctx.method], ctx.k_steps, ctx.r_per_step,
           ctx.strips_per_wg);
  }
  if (EXIT_SUCCESS == result) {
    if (0 != trace) fprintf(stderr, "TRACE: configure\n");
    result = stencil_configure(&ctx, nx, ny, nz);
    if (0 != trace) fprintf(stderr, "TRACE: configure done result=%d\n", result);
  }
  if (EXIT_SUCCESS == result) {
    if (0 != trace) fprintf(stderr, "TRACE: precompute operators\n");
    result = stencil_precompute_operators(&ctx, fd_w, radius);
    if (0 != trace) fprintf(stderr, "TRACE: precompute operators done result=%d\n", result);
  }
  if (EXIT_SUCCESS == result) {
    const size_t grid_bytes = (size_t)nx * ny * nz * sizeof(float);
    const double gpoints = (double)nx * ny * nz * 1.0e-9;
    const float dt_local = 0.6f * (float)h
                         / (v_max * (float)sqrt(3.0) * (2.0f * radius + 1.0f));
    const float dt2 = dt_local * dt_local;
    void* p_buf[3] = { NULL, NULL, NULL };
    void* vel_dev = NULL;
    float* p_host = NULL;
    float* vel_host = NULL;
    libxs_timer_tick_t t0, t1;
    double t_elapsed, gpts_per_s;
    int t, cur, old, new_idx;

    if (EXIT_SUCCESS == result) {
      if (0 != trace) fprintf(stderr, "TRACE: allocate p_host\n");
      result = libxstream_mem_host_allocate((void**)&p_host, grid_bytes, ctx.stream);
      if (0 != trace) fprintf(stderr, "TRACE: allocate p_host done result=%d\n", result);
    }
    if (EXIT_SUCCESS == result) {
      if (0 != trace) fprintf(stderr, "TRACE: allocate vel_host\n");
      result = libxstream_mem_host_allocate((void**)&vel_host, grid_bytes, ctx.stream);
      if (0 != trace) fprintf(stderr, "TRACE: allocate vel_host done result=%d\n", result);
    }

    if (EXIT_SUCCESS == result) {
      if (VEL_FILE == vel_model && NULL != vel_file) {
        result = load_velocity_file(vel_file, vel_host, nx, ny, nz);
      }
      else {
        generate_velocity(vel_host, nx, ny, nz, vel_model, v_min, v_max);
      }
    }

    if (EXIT_SUCCESS == result) {
      const int n = nx * ny * nz;
      int i;
      for (i = 0; i < n; ++i) p_host[i] = 0.0f;
    }

    if (EXIT_SUCCESS == result) result = libxstream_mem_allocate(&p_buf[0], grid_bytes);
    if (EXIT_SUCCESS == result) result = libxstream_mem_allocate(&p_buf[1], grid_bytes);
    if (EXIT_SUCCESS == result) result = libxstream_mem_allocate(&p_buf[2], grid_bytes);
    if (EXIT_SUCCESS == result) result = libxstream_mem_allocate(&vel_dev, grid_bytes);

    if (EXIT_SUCCESS == result) {
      result = libxstream_mem_copy_h2d(p_host, p_buf[0], grid_bytes, ctx.stream);
    }
    if (EXIT_SUCCESS == result) {
      result = libxstream_mem_zero(p_buf[1], 0, grid_bytes, ctx.stream);
    }
    if (EXIT_SUCCESS == result) {
      result = libxstream_mem_copy_h2d(vel_host, vel_dev, grid_bytes, ctx.stream);
    }
    if (EXIT_SUCCESS == result) {
      result = libxstream_stream_sync(ctx.stream);
    }

    cur = 0; old = 1; new_idx = 2;

    for (t = 0; t < warmup && EXIT_SUCCESS == result; ++t) {
      int tmp;
      inject_source(p_host, nx, ny, nz, dt_local, t, freq);
      result = libxstream_mem_copy_h2d(p_host, p_buf[cur], grid_bytes, ctx.stream);
      if (EXIT_SUCCESS == result) {
        result = stencil_apply_laplacian(&ctx,
          p_buf[cur], p_buf[old], p_buf[new_idx], vel_dev, dt2, nterms);
      }
      tmp = old; old = cur; cur = new_idx; new_idx = tmp;
    }
    if (EXIT_SUCCESS == result) {
      result = libxstream_stream_sync(ctx.stream);
    }

    t0 = libxs_timer_tick();

    for (t = 0; t < ntsteps && EXIT_SUCCESS == result; ++t) {
      int tmp;
      result = stencil_apply_laplacian(&ctx,
        p_buf[cur], p_buf[old], p_buf[new_idx], vel_dev, dt2, nterms);
      tmp = old; old = cur; cur = new_idx; new_idx = tmp;
    }
    if (EXIT_SUCCESS == result) {
      result = libxstream_stream_sync(ctx.stream);
    }

    t1 = libxs_timer_tick();
    t_elapsed = libxs_timer_duration(t0, t1);

    if (EXIT_SUCCESS == result) {
      gpts_per_s = gpoints * ntsteps / t_elapsed;
      printf("Results:\n");
      printf("  Time:       %.3f s (%d steps)\n", t_elapsed, ntsteps);
      printf("  Throughput: %.3f GPoints/s\n", gpts_per_s);
      printf("  Per step:   %.3f ms\n", 1000.0 * t_elapsed / ntsteps);
      printf("  Bandwidth:  %.1f GB/s (effective, read+write)\n",
             gpoints * ntsteps * 2.0 * sizeof(float) / t_elapsed);
    }

    if (NULL != vel_dev) libxstream_mem_deallocate(vel_dev);
    if (NULL != p_buf[2]) libxstream_mem_deallocate(p_buf[2]);
    if (NULL != p_buf[1]) libxstream_mem_deallocate(p_buf[1]);
    if (NULL != p_buf[0]) libxstream_mem_deallocate(p_buf[0]);
    if (NULL != vel_host) libxstream_mem_host_deallocate(vel_host, ctx.stream);
    if (NULL != p_host) libxstream_mem_host_deallocate(p_host, ctx.stream);
  }

  stencil_finalize(&ctx);
  if (0 != initialized) libxstream_finalize();
  return result;
}


static void fd_weights_2nd(double* w, int radius, double h)
{
  const double h2 = h * h;
  int r;
  (void)r;
  w[radius] = -2.0 / h2;
  if (1 <= radius) { w[radius - 1] = 1.0 / h2; w[radius + 1] = 1.0 / h2; }
  if (2 <= radius) {
    switch (radius) {
      case 2:
        w[radius] = -5.0 / (2.0 * h2);
        w[radius - 1] = 4.0 / (3.0 * h2);
        w[radius + 1] = 4.0 / (3.0 * h2);
        w[radius - 2] = -1.0 / (12.0 * h2);
        w[radius + 2] = -1.0 / (12.0 * h2);
        break;
      case 3:
        w[radius] = -49.0 / (18.0 * h2);
        w[radius - 1] = 3.0 / (2.0 * h2);
        w[radius + 1] = 3.0 / (2.0 * h2);
        w[radius - 2] = -3.0 / (20.0 * h2);
        w[radius + 2] = -3.0 / (20.0 * h2);
        w[radius - 3] = 1.0 / (90.0 * h2);
        w[radius + 3] = 1.0 / (90.0 * h2);
        break;
      case 4:
        w[radius] = -205.0 / (72.0 * h2);
        w[radius - 1] = 8.0 / (5.0 * h2);
        w[radius + 1] = 8.0 / (5.0 * h2);
        w[radius - 2] = -1.0 / (5.0 * h2);
        w[radius + 2] = -1.0 / (5.0 * h2);
        w[radius - 3] = 8.0 / (315.0 * h2);
        w[radius + 3] = 8.0 / (315.0 * h2);
        w[radius - 4] = -1.0 / (560.0 * h2);
        w[radius + 4] = -1.0 / (560.0 * h2);
        break;
      default:
        break;
    }
  }
  else {
    for (r = 2; r <= radius; ++r) {
      w[radius - r] = 0.0;
      w[radius + r] = 0.0;
    }
  }
}


static int load_velocity_file(const char* path, float* vel,
                              int nx, int ny, int nz)
{
  int result = EXIT_SUCCESS;
  FILE* f = fopen(path, "rb");
  const size_t n = (size_t)nx * ny * nz;
  size_t nread;

  if (NULL == f) {
    fprintf(stderr, "Cannot open velocity file: %s\n", path);
    result = EXIT_FAILURE;
  }

  if (EXIT_SUCCESS == result) {
    nread = fread(vel, sizeof(float), n, f);
    if (nread != n) {
      fprintf(stderr, "Velocity file too short: read %lu of %lu elements\n",
              (unsigned long)nread, (unsigned long)n);
      result = EXIT_FAILURE;
    }
  }

  if (NULL != f) fclose(f);

  if (EXIT_SUCCESS == result) {
    size_t i;
    for (i = 0; i < n; ++i) vel[i] = vel[i] * vel[i];
  }
  return result;
}


static void generate_velocity(float* vel, int nx, int ny, int nz,
                              vel_model_t model, float v_top, float v_bot)
{
  const int n = nx * ny * nz;
  int i;

  switch (model) {
    case VEL_GRADIENT: {
      int ix, iy, iz;
      const float dv = (v_bot - v_top) / (float)(nz > 1 ? nz - 1 : 1);
      for (iz = 0; iz < nz; ++iz) {
        const float v = v_top + dv * (float)iz;
        const float v2 = v * v;
        for (iy = 0; iy < ny; ++iy) {
          for (ix = 0; ix < nx; ++ix) {
            vel[iz * ny * nx + iy * nx + ix] = v2;
          }
        }
      }
    } break;
    case VEL_LAYERED: {
      const int nlayers = 5;
      const float layer_v[] = {1500.0f, 2000.0f, 2500.0f, 3500.0f, 4500.0f};
      int ix, iy, iz;
      for (iz = 0; iz < nz; ++iz) {
        const int li = (iz * nlayers) / nz;
        const float v2 = layer_v[li] * layer_v[li];
        for (iy = 0; iy < ny; ++iy) {
          for (ix = 0; ix < nx; ++ix) {
            vel[iz * ny * nx + iy * nx + ix] = v2;
          }
        }
      }
    } break;
    default:
      for (i = 0; i < n; ++i) vel[i] = v_top * v_top;
      break;
  }
}


static void inject_source(float* p, int nx, int ny, int nz,
                          float dt, int tstep, float freq)
{
  const float t = (float)tstep * dt;
  const float t0 = 1.2f / freq;
  const float arg = 3.14159265f * freq * (t - t0);
  const float ricker = (1.0f - 2.0f * arg * arg)
                     * (float)exp((double)(-arg * arg));
  const int sx = nx / 2, sy = ny / 2, sz = nz / 2;
  p[sz * ny * nx + sy * nx + sx] += ricker;
}


static void usage(const char* prog)
{
  printf("Usage: %s [options]\n"
         "  -n <N>        grid dimension (NxNxN, default 256)\n"
         "  -nx/ny/nz <N> individual grid dimensions\n"
         "  -t <steps>    number of time steps (default 100)\n"
         "  -d <dims>     operator terms: 3=isotropic, 9=TTI (default 3)\n"
         "  -m <method>   operator method: 0=direct 1=staged-r1 2=staged-r2 3=staged-fit\n"
         "  -h <spacing>  grid spacing in meters (default 10.0)\n"
         "  -v <model>    velocity model: const|grad|layered|<file.bin>\n"
         "  -vmin <vel>   min velocity m/s (default 1500)\n"
         "  -vmax <vel>   max velocity m/s (default 4500)\n"
         "  -f <freq>     source frequency Hz (default 25)\n"
         "  -w <steps>    warmup steps (default 5)\n"
         "\n"
         "Benchmark models (shortcuts):\n"
         "  -seg-salt      SEG/EAGE Salt (676x676x210, h=20m)\n"
         "  -overthrust    SEG/EAGE Overthrust (801x801x187, h=25m)\n"
         "\n"
         "Environment: STENCIL_METHOD, STENCIL_STRIPS_PER_WG, STENCIL_SG, STENCIL_GRF256, STENCIL_TRIM\n"
         "\n"
         "Performance is reported in GPoints/s.\n", prog);
}
