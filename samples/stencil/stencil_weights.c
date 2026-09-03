/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "stencil_weights.h"
#include <libxs/libxs_math.h>

#include <math.h>
#include <stdlib.h>


typedef struct {
  double tmax;
  int nq;
} stencil_fit_data_t;


int stencil_method_params(stencil_method_t method, int* k_steps, int* r_per_step)
{
  int result = EXIT_SUCCESS;
  switch (method) {
    case STENCIL_DIRECT:
      *k_steps = 1; *r_per_step = STENCIL_RADIUS;
      break;
    case STENCIL_COMPACT_R1:
      *k_steps = STENCIL_RADIUS; *r_per_step = 1;
      break;
    case STENCIL_COMPACT_R2:
      *k_steps = 2; *r_per_step = (STENCIL_RADIUS + 1) / 2;
      break;
    case STENCIL_COMPACT_FIT: {
      const char* rfit_env = getenv("STENCIL_RADIUS_FIT");
      const int rfit = (NULL != rfit_env) ? atoi(rfit_env) : 3;
      *r_per_step = (rfit >= 1 && rfit <= STENCIL_RADIUS) ? rfit : 3;
      *k_steps = (STENCIL_RADIUS + *r_per_step - 1) / *r_per_step;
    } break;
    default:
      result = EXIT_FAILURE;
      break;
  }
  return result;
}


double stencil_fd_weight(const double* fd_weights, int radius, int dist)
{
  double result = 0.0;
  if (dist >= -radius && dist <= radius) {
    result = fd_weights[dist + radius];
  }
  return result;
}


double stencil_compact_weight(int radius, int dist, double inv_h2)
{
  double result = 0.0;
  if (dist >= -radius && dist <= radius) {
    if (1 == radius) {
      result = (0 == dist) ? -2.0 : 1.0;
    }
    else if (2 == radius) {
      if (0 == dist) result = -5.0 / 2.0;
      else if (1 == dist || -1 == dist) result = 4.0 / 3.0;
      else result = -1.0 / 12.0;
    }
  }
  result *= inv_h2;
  return result;
}


static double stencil_ricker_weight(double t, double tpeak)
{
  const double u = t / tpeak;
  return u * u * exp(1.0 - u * u);
}


static double stencil_fit_error_r2(double alpha, double t)
{
  const double S = -(2.0 - 6.0 * alpha)
    + 2.0 * (1.0 - 4.0 * alpha) * cos(t) + 2.0 * alpha * cos(2.0 * t);
  return S + t * t;
}


static double stencil_fit_maxerr_r2(double alpha, double tmax, int nq)
{
  double mx = 0.0;
  int q;
  for (q = 0; q < nq; ++q) {
    const double t = (q + 0.5) * tmax / nq;
    const double e = fabs(stencil_fit_error_r2(alpha, t));
    if (e > mx) mx = e;
  }
  return mx;
}


static double stencil_fit_error_r3(double alpha, double beta, double t)
{
  const double a0 = -(2.0 - 6.0 * beta - 16.0 * alpha);
  const double a1 = 1.0 - 4.0 * beta - 9.0 * alpha;
  const double S = a0 + 2.0 * a1 * cos(t)
    + 2.0 * beta * cos(2.0 * t) + 2.0 * alpha * cos(3.0 * t);
  return S + t * t;
}


static double stencil_fit_maxerr_r3(double alpha, double beta,
                                    double tmax, int nq)
{
  double mx = 0.0;
  int q;
  for (q = 0; q < nq; ++q) {
    const double t = (q + 0.5) * tmax / nq;
    const double e = fabs(stencil_fit_error_r3(alpha, beta, t));
    if (e > mx) mx = e;
  }
  return mx;
}


static double stencil_gss_maxerr_r2(double alpha, const void* data)
{
  const stencil_fit_data_t* d = (const stencil_fit_data_t*)data;
  return stencil_fit_maxerr_r2(alpha, d->tmax, d->nq);
}


static double stencil_fit_optimal_beta(double alpha, double tmax, int nq)
{
  const double dt = tmax / nq;
  double num = 0.0, den = 0.0;
  int q;
  for (q = 0; q < nq; ++q) {
    const double t = (q + 0.5) * dt;
    const double a0 = -(2.0 - 16.0 * alpha);
    const double a1 = 1.0 - 9.0 * alpha;
    const double S0 = a0 + 2.0 * a1 * cos(t) + 2.0 * alpha * cos(3.0 * t);
    const double Bb = 6.0 - 8.0 * cos(t) + 2.0 * cos(2.0 * t);
    const double rhs = S0 + t * t;
    num += rhs * Bb;
    den += Bb * Bb;
  }
  return (den > 1e-30) ? -num / den : -3.0 / 20.0;
}


static double stencil_gss_maxerr_r3(double alpha, const void* data)
{
  const stencil_fit_data_t* d = (const stencil_fit_data_t*)data;
  const double beta = stencil_fit_optimal_beta(alpha, d->tmax, d->nq);
  return stencil_fit_maxerr_r3(alpha, beta, d->tmax, d->nq);
}


void stencil_fit_coeffs(int radius, double ppw, int fit_method,
                               double* coeffs)
{
  const double tmax = 2.0 * M_PI / ppw;
  const double tpeak = 2.0 * M_PI / (ppw * 0.6);
  const int nq = 1024;
  const double dt = tmax / nq;
  int q;

  if (2 == radius) {
    double alpha;
    if (2 == fit_method) {
      stencil_fit_data_t gss_data;
      double xmin;
      gss_data.tmax = tmax;
      gss_data.nq = nq;
      libxs_gss_min(stencil_gss_maxerr_r2, &gss_data,
        -0.5, 0.5, &xmin, 100, LIBXS_GSS_EVAL_ENDPOINTS, 1e-12, NULL);
      alpha = xmin;
    }
    else {
      double num = 0.0, den = 0.0;
      for (q = 0; q < nq; ++q) {
        const double t = (q + 0.5) * dt;
        const double w = (1 == fit_method)
          ? stencil_ricker_weight(t, tpeak) : 1.0;
        const double A = -2.0 + 2.0 * cos(t);
        const double B = 6.0 - 8.0 * cos(t) + 2.0 * cos(2.0 * t);
        num += w * (A + t * t) * B;
        den += w * B * B;
      }
      alpha = (den > 1e-30) ? -num / den : -1.0 / 12.0;
    }
    coeffs[2] = -(2.0 - 6.0 * alpha);
    coeffs[1] = 1.0 - 4.0 * alpha;
    coeffs[0] = alpha;
  }
  else if (3 == radius) {
    double alpha, beta;
    if (2 == fit_method) {
      stencil_fit_data_t gss_data;
      double xmin;
      gss_data.tmax = tmax;
      gss_data.nq = nq;
      libxs_gss_min(stencil_gss_maxerr_r3, &gss_data,
        -0.2, 0.2, &xmin, 100, LIBXS_GSS_EVAL_ENDPOINTS, 1e-12, NULL);
      alpha = xmin;
      beta = stencil_fit_optimal_beta(alpha, tmax, nq);
    }
    else {
      double m00 = 0.0, m01 = 0.0, m11 = 0.0;
      double v0 = 0.0, v1 = 0.0, det;
      for (q = 0; q < nq; ++q) {
        const double t = (q + 0.5) * dt;
        const double w = (1 == fit_method)
          ? stencil_ricker_weight(t, tpeak) : 1.0;
        const double A = -2.0 + 2.0 * cos(t);
        const double Ba = 16.0 - 18.0 * cos(t) + 2.0 * cos(3.0 * t);
        const double Bb = 6.0 - 8.0 * cos(t) + 2.0 * cos(2.0 * t);
        const double rhs = A + t * t;
        m00 += w * Ba * Ba;
        m01 += w * Ba * Bb;
        m11 += w * Bb * Bb;
        v0 += w * rhs * Ba;
        v1 += w * rhs * Bb;
      }
      det = m00 * m11 - m01 * m01;
      if (det * det < 1e-30) {
        alpha = 1.0 / 90.0;
        beta = -3.0 / 20.0;
      }
      else {
        alpha = -(m11 * v0 - m01 * v1) / det;
        beta = -(m00 * v1 - m01 * v0) / det;
      }
    }
    coeffs[3] = -(2.0 - 6.0 * beta - 16.0 * alpha);
    coeffs[2] = 1.0 - 4.0 * beta - 9.0 * alpha;
    coeffs[1] = beta;
    coeffs[0] = alpha;
  }
}


double stencil_fit_weight(int radius, int dist, double inv_h2,
                                 double ppw, int fit_method)
{
  double result = 0.0;
  if (dist >= -radius && dist <= radius) {
    if (3 == radius) {
      double c[4];
      stencil_fit_coeffs(3, ppw, fit_method, c);
      if (0 == dist) result = c[3];
      else if (1 == dist || -1 == dist) result = c[2];
      else if (2 == dist || -2 == dist) result = c[1];
      else result = c[0];
    }
    else if (2 == radius) {
      double c[3];
      stencil_fit_coeffs(2, ppw, fit_method, c);
      if (0 == dist) result = c[2];
      else if (1 == dist || -1 == dist) result = c[1];
      else result = c[0];
    }
    else if (1 == radius) {
      result = (0 == dist) ? -2.0 : 1.0;
    }
  }
  result *= inv_h2;
  return result;
}
