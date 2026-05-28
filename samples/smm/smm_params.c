/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#if defined(__OPENCL)
#  include "smm_acc_opencl.h"
#  include <libxs_math.h>
#  include <ctype.h>

#  define OPENCL_LIBSMM_AI(M, N, K, TYPESIZE) ((2.0 * (M) * (N) * (K)) / ((TYPESIZE) * (K) * ((M) + (N))))
#  define OPENCL_LIBSMM_TYPESIZE(TYPEID) \
    (dbcsr_type_real_8 == (TYPEID) ? ((int)sizeof(double)) : (dbcsr_type_real_4 == (TYPEID) ? ((int)sizeof(float)) : 0 /*unknown*/))

#  if defined(__cplusplus)
extern "C" {
#  endif


int opencl_libsmm_write_trans_params(FILE* stream, int only_key, const opencl_libsmm_transkey_t* key,
  const opencl_libsmm_trans_t* config, const char* delim, const char* begin, const char* close) {
  int result = 0;
  if (NULL != stream) {
    const char d = (NULL == delim ? *LIBXS_DELIMS : *delim);
    if (NULL != key || 0 == only_key) result += fprintf(stream, "%c", NULL == begin ? '{' : *begin);
    if (NULL != config) {
      if (NULL != key) {
        result += fprintf(stream, "%i%c%i%c%i", LIBXS_CAST_INT(key->type), d, key->m, d, key->n);
        /*if (0 == only_key) result += fprintf(stream, "%c", d);*/
      }
    }
    else {
      if (NULL != key) {
        result += fprintf(stream, "t%cm%cn", d, d);
        /*if (0 == only_key) result += fprintf(stream, "%c", d);*/
      }
    }
    if (NULL != key || 0 == only_key) result += fprintf(stream, "%c", NULL == close ? '}' : *close);
  }
  else {
    result = -1;
  }
  LIBXS_ASSERT(0 < result);
  return result;
}


int opencl_libsmm_write_smm_params(FILE* stream, int only_key, const opencl_libsmm_smmkey_t* key, const opencl_libsmm_smm_t* config,
  const char* delim, const char* begin, const char* close) {
  int result = 0;
  if (NULL != stream) {
    const char d = (NULL == delim ? *LIBXS_DELIMS : *delim);
    if (NULL != key || 0 == only_key) result += fprintf(stream, "%c", NULL == begin ? '{' : *begin);
    if (NULL != config) {
      if (NULL != key) {
        result += fprintf(stream, "%i%c%i%c%i%c%i", LIBXS_CAST_INT(key->type), d, key->m, d, key->n, d, key->k);
        if (0 == only_key) result += fprintf(stream, "%c ", d);
      }
      if (0 == only_key) {
        result += fprintf(stream, "%i%c%i%c%i%c%i%c %i%c%i%c %i%c%i%c%i%c %i%c%i%c %i%c%i%c%i%c%i", config->bs, d, config->bm, d,
          config->bn, d, config->bk, d, config->ws, d, config->wg, d, config->lu, d, config->nz, d, config->al, d, config->tb, d,
          config->tc, d, config->ap, d, config->aa, d, config->ab, d, config->ac);
        if (0 != config->flags) result += fprintf(stream, "%c %i", d, config->flags);
      }
    }
    else {
      if (NULL != key) {
        result += fprintf(stream, "t%cm%cn%ck", d, d, d);
        if (0 == only_key) result += fprintf(stream, "%c ", d);
      }
      if (0 == only_key) {
        result += fprintf(
          stream, "bs%cbm%cbn%cbk%c ws%cwg%c lu%cnz%cal%c tb%ctc%c ap%caa%cab%cac", d, d, d, d, d, d, d, d, d, d, d, d, d, d);
      }
    }
    if (NULL != key || 0 == only_key) result += fprintf(stream, "%c", NULL == close ? '}' : *close);
  }
  else {
    result = -1;
  }
  LIBXS_ASSERT(0 < result);
  return result;
}


int opencl_libsmm_read_smm_params(char* parambuf, opencl_libsmm_smmkey_t* key, opencl_libsmm_smm_t* value,
  opencl_libsmm_perfest_t* perfest, char* device, int* key_ok) {
  const char* const end = parambuf + strlen(parambuf); /* before strtok */
  char* s = strtok(parambuf, LIBXS_DELIMS);
  const int opt_consumed = (NULL != perfest ? 2 : 0) + (NULL != device ? 1 : 0);
  int result = EXIT_SUCCESS, i = 0, ivalue, consumed = 0, c = 0, max_consumed = opt_consumed + 19;
  double gflops;
  LIBXS_ASSERT(NULL != key && NULL != value);
  LIBXS_MEMZERO(key); /* potentially heterogeneous key-data (alignment gaps) */
  LIBXS_MEMZERO(value);
  for (; NULL != s;
    ++i, s = (c != consumed ? ((s + 1) < end ? strtok((s + 1) + strlen(s), LIBXS_DELIMS) : NULL) : s), c = consumed)
  {
    switch (i) {
      case 0:
        if (NULL != device && 1 == sscanf(s, "%[^" LIBXS_DELIMS "]", device)) {
          ++consumed; /* optional device name */
        }
        break;
      case 1:
        if (1 == sscanf(s, "%i", &ivalue)) {
          key->type = (libsmm_acc_data_t)ivalue;
          ++consumed;
        }
        break;
      case 2:
        if (1 == sscanf(s, "%i", &ivalue) && 0 < ivalue) {
          key->m = ivalue;
          ++consumed;
        }
        break;
      case 3:
        if (1 == sscanf(s, "%i", &ivalue) && 0 < ivalue) {
          key->n = ivalue;
          ++consumed;
        }
        break;
      case 4:
        if (1 == sscanf(s, "%i", &ivalue) && 0 < ivalue) {
          key->k = ivalue;
          ++consumed;
        }
        break;
      case 5:
        if (NULL != perfest && 1 == sscanf(s, "%i", &ivalue)) {
          value->s = ivalue;
          ++consumed; /* optional "S" param */
        }
        break;
      case 6:
        if (NULL != perfest && 1 == sscanf(s, "%lf", &gflops) && 0 <= gflops) {
          value->gflops = gflops;
          ++consumed; /* optional "GFLOPS" param */
        }
        break;
      case 7:
        if (1 == sscanf(s, "%i", &ivalue)) {
          value->bs = ivalue;
          ++consumed;
        }
        break;
      case 8:
        if (1 == sscanf(s, "%i", &ivalue)) {
          value->bm = ivalue;
          ++consumed;
        }
        break;
      case 9:
        if (1 == sscanf(s, "%i", &ivalue)) {
          value->bn = ivalue;
          ++consumed;
        }
        break;
      case 10:
        if (1 == sscanf(s, "%i", &ivalue)) {
          value->bk = ivalue;
          ++consumed;
        }
        break;
      case 11:
        if (1 == sscanf(s, "%i", &ivalue)) {
          value->ws = ivalue;
          ++consumed;
        }
        break;
      case 12:
        if (1 == sscanf(s, "%i", &ivalue)) {
          value->wg = ivalue;
          ++consumed;
        }
        break;
      case 13:
        if (1 == sscanf(s, "%i", &ivalue)) {
          value->lu = ivalue;
          ++consumed;
        }
        break;
      case 14:
        if (1 == sscanf(s, "%i", &ivalue)) {
          value->nz = ivalue;
          ++consumed;
        }
        break;
      case 15:
        if (1 == sscanf(s, "%i", &ivalue)) {
          value->al = ivalue;
          ++consumed;
        }
        break;
      case 16:
        if (1 == sscanf(s, "%i", &ivalue)) {
          value->tb = ivalue;
          ++consumed;
        }
        break;
      case 17:
        if (1 == sscanf(s, "%i", &ivalue)) {
          value->tc = ivalue;
          ++consumed;
        }
        break;
      case 18:
        if (1 == sscanf(s, "%i", &ivalue)) {
          value->ap = ivalue;
          ++consumed;
        }
        break;
      case 19:
        if (1 == sscanf(s, "%i", &ivalue)) {
          value->aa = ivalue;
          ++consumed;
        }
        break;
      case 20:
        if (1 == sscanf(s, "%i", &ivalue)) {
          value->ab = ivalue;
          ++consumed;
        }
        break;
      case 21:
        if (1 == sscanf(s, "%i", &ivalue)) {
          value->ac = ivalue;
          ++consumed;
        }
        break;
      case 22:
        if (1 == sscanf(s, "%i", &ivalue)) {
          value->flags = ivalue;
          ++max_consumed;
          ++consumed;
        }
        break;
      default: s = NULL; /* break */
    }
  }
  if (max_consumed == consumed) {
    switch (key->type) {
      case dbcsr_type_real_8:
        if (NULL != perfest && 0 < gflops) {
          const double ratio = gflops / OPENCL_LIBSMM_AI(key->m, key->n, key->k, sizeof(double));
          libxs_kahan_sum(log(ratio), &perfest->gf_ai_dratio_sumlog, &perfest->gf_ai_dratio_kahan);
          if (perfest->gf_ai_dratio_max < ratio) perfest->gf_ai_dratio_max = ratio;
          ++perfest->dcount;
        }
        break;
      case dbcsr_type_real_4:
        if (NULL != perfest && 0 < gflops) {
          const double ratio = gflops / OPENCL_LIBSMM_AI(key->m, key->n, key->k, sizeof(float));
          libxs_kahan_sum(log(ratio), &perfest->gf_ai_sratio_sumlog, &perfest->gf_ai_sratio_kahan);
          if (perfest->gf_ai_sratio_max < ratio) perfest->gf_ai_sratio_max = ratio;
          ++perfest->scount;
        }
        break;
      default: result = EXIT_FAILURE;
    }
  }
  else {
    result = EXIT_FAILURE;
  }
  if (NULL != key_ok && 4 <= consumed) *key_ok = 1;
  return result;
}

#  if defined(__cplusplus)
}
#  endif

#endif /*defined(__OPENCL)*/
