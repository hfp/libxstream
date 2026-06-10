/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs/libxs_predict.h>
#include <libxs/libxs_timer.h>

#if defined(_DEBUG)
# define FPRINTF(STREAM, ...) do { fprintf(STREAM, __VA_ARGS__); } while(0)
#else
# define FPRINTF(STREAM, ...) do {} while(0)
#endif

enum { NINPUTS = 3, NOUTPUTS = 16 };


int main(void)
{
  int result = EXIT_SUCCESS;
  libxs_predict_t* model = libxs_predict_create(NINPUTS, NOUTPUTS);
  if (NULL == model) {
    result = EXIT_FAILURE;
  }
  else {
    const int nentries = 64;
    int i, j;
    for (i = 0; i < nentries && EXIT_SUCCESS == result; ++i) {
      double inputs[NINPUTS], outputs[NOUTPUTS];
      inputs[0] = 4.0 + (i % 8) * 4;
      inputs[1] = 4.0 + ((i / 8) % 8) * 4;
      inputs[2] = 4.0 + (i % 5) * 8;
      for (j = 0; j < NOUTPUTS; ++j) {
        outputs[j] = (double)((i + j * 3) % 7);
      }
      if (EXIT_SUCCESS != libxs_predict_push(NULL, model, inputs, outputs)) {
        result = EXIT_FAILURE;
      }
    }
    if (EXIT_SUCCESS == result) {
      result = libxs_predict_build(model, 0, 2, 0);
    }
    if (EXIT_SUCCESS == result) {
      libxs_predict_query_t qinfo;
      libxs_predict_query(model, &qinfo);
      FPRINTF(stderr, "Built: %d entries, %d clusters, order=%d\n",
        qinfo.nentries, qinfo.nclusters, qinfo.order);
      if (qinfo.nentries != nentries || qinfo.nclusters <= 0 || qinfo.order <= 0) {
        FPRINTF(stderr, "ERROR: unexpected query results\n");
        result = EXIT_FAILURE;
      }
    }
    if (EXIT_SUCCESS == result) {
      size_t size = 0;
      void* buffer = NULL;
      libxs_predict_t* loaded = NULL;
      if (EXIT_SUCCESS != libxs_predict_save(model, NULL, &size) || 0 == size) {
        FPRINTF(stderr, "ERROR: save size query failed\n");
        result = EXIT_FAILURE;
      }
      if (EXIT_SUCCESS == result) {
        buffer = malloc(size);
        if (NULL == buffer) {
          result = EXIT_FAILURE;
        }
      }
      if (EXIT_SUCCESS == result) {
        if (EXIT_SUCCESS != libxs_predict_save(model, buffer, &size)) {
          FPRINTF(stderr, "ERROR: save failed\n");
          result = EXIT_FAILURE;
        }
      }
      if (EXIT_SUCCESS == result) {
        loaded = libxs_predict_load(buffer, size);
        if (NULL == loaded) {
          FPRINTF(stderr, "ERROR: load failed\n");
          result = EXIT_FAILURE;
        }
      }
      if (EXIT_SUCCESS == result) {
        libxs_predict_query_t qi_orig, qi_load;
        libxs_predict_query(model, &qi_orig);
        libxs_predict_query(loaded, &qi_load);
        if (qi_orig.nentries != qi_load.nentries ||
            qi_orig.nclusters != qi_load.nclusters ||
            qi_orig.order != qi_load.order)
        {
          FPRINTF(stderr, "ERROR: query mismatch after load"
            " (entries %d/%d, clusters %d/%d, order %d/%d)\n",
            qi_orig.nentries, qi_load.nentries,
            qi_orig.nclusters, qi_load.nclusters,
            qi_orig.order, qi_load.order);
          result = EXIT_FAILURE;
        }
      }
      if (EXIT_SUCCESS == result) {
        size_t size2 = 0;
        void* buffer2 = NULL;
        libxs_predict_t* loaded2 = NULL;
        libxs_predict_save(loaded, NULL, &size2);
        buffer2 = malloc(size2);
        if (NULL != buffer2) {
          if (EXIT_SUCCESS == libxs_predict_save(loaded, buffer2, &size2)) {
            loaded2 = libxs_predict_load(buffer2, size2);
          }
        }
        if (NULL != loaded2) {
          for (i = 0; i < nentries && EXIT_SUCCESS == result; ++i) {
            double inputs[NINPUTS], out1[NOUTPUTS], out2[NOUTPUTS];
            inputs[0] = 4.0 + (i % 8) * 4;
            inputs[1] = 4.0 + ((i / 8) % 8) * 4;
            inputs[2] = 4.0 + (i % 5) * 8;
            libxs_predict_eval(NULL, loaded, inputs, out1, NULL, 1);
            libxs_predict_eval(NULL, loaded2, inputs, out2, NULL, 1);
            for (j = 0; j < NOUTPUTS; ++j) {
              const double delta = out1[j] - out2[j];
              if (delta > 1e-10 || delta < -1e-10) {
                FPRINTF(stderr, "ERROR: roundtrip mismatch at entry %d"
                  " output %d (%.6f vs %.6f)\n", i, j, out1[j], out2[j]);
                result = EXIT_FAILURE;
              }
            }
          }
          if (EXIT_SUCCESS == result) {
            double novel[NINPUTS], out1[NOUTPUTS], out2[NOUTPUTS];
            novel[0] = 14.0;
            novel[1] = 22.0;
            novel[2] = 18.0;
            libxs_predict_eval(NULL, loaded, novel, out1, NULL, 0);
            libxs_predict_eval(NULL, loaded2, novel, out2, NULL, 0);
            for (j = 0; j < NOUTPUTS; ++j) {
              const double delta = out1[j] - out2[j];
              if (delta > 1e-10 || delta < -1e-10) {
                FPRINTF(stderr, "ERROR: novel roundtrip mismatch at"
                  " output %d (%.6f vs %.6f)\n", j, out1[j], out2[j]);
                result = EXIT_FAILURE;
              }
            }
          }
          libxs_predict_destroy(loaded2);
        }
        else {
          FPRINTF(stderr, "ERROR: double roundtrip load failed\n");
          result = EXIT_FAILURE;
        }
        free(buffer2);
      }
      libxs_predict_destroy(loaded);
      free(buffer);
    }
    libxs_predict_destroy(model);
  }
  return result;
}
