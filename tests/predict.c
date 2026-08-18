/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
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

enum { NINPUTS = 3, NOUTPUTS = 16, NQUERY = 64 };

/**
 * Models saved by LIBXS 1.0.0 (format version 1), which the current loader must
 * keep reading (see predict_v1_README.md).  A round-trip test cannot cover this
 * because it only ever exercises the current writer.  All carry 3 inputs and 16
 * outputs, hence share NINPUTS/NOUTPUTS with the model built below.
 */
static const char *const internal_predict_fixtures[] = {
  "predict_v1_flat.bin",
  "predict_v1_flat_c09.bin",
  "predict_v1_flat_interp.bin",
  "predict_v1_flat_rf.bin",
  "predict_v1_hknn.bin",
  "predict_v1_hknn_c09.bin"
};

/**
 * Candidates relative to the working directory, tried after the directory of the
 * test binary: the runner launches by absolute path without setting a working
 * directory, and an out-of-tree build separates the binary from the data.
 */
static const char *const internal_predict_dirs[] = { "", "tests/", "../tests/" };


static void* internal_predict_read(const char* dir, const char* name,
  size_t* size)
{
  void* result = NULL;
  const size_t nd = strlen(dir), nn = strlen(name);
  char path[1024];
  *size = 0;
  if (sizeof(path) > (nd + nn)) {
    FILE* file;
    memcpy(path, dir, nd);
    memcpy(path + nd, name, nn + 1);
    file = fopen(path, "rb");
    if (NULL != file) {
      long len = 0;
      if (0 == fseek(file, 0, SEEK_END)) len = ftell(file);
      if (0 < len && 0 == fseek(file, 0, SEEK_SET)) {
        void* buffer = malloc((size_t)len);
        if (NULL != buffer) {
          if (1 == fread(buffer, (size_t)len, 1, file)) {
            result = buffer;
            *size = (size_t)len;
          }
          else free(buffer);
        }
      }
      fclose(file);
    }
  }
  return result;
}


/**
 * A fixture must load, must predict the same as the model written when it is
 * re-saved in the current format, and its re-saved form must be rejected when
 * damaged (the checksum introduced with format version 2).
 */
static int internal_predict_check(const void* buffer, size_t size,
  const char* name)
{
  int result = EXIT_SUCCESS;
  libxs_predict_t* model = libxs_predict_load(buffer, size);
  if (NULL == model) {
    FPRINTF(stderr, "ERROR: %s failed to load\n", name);
    result = EXIT_FAILURE;
  }
  else {
    libxs_predict_query_t qinfo;
    unsigned short version = 0;
    size_t rsize = 0;
    /**
     * These fixtures exist to cover an older format, so the declared version is
     * asserted: regenerating one with a current LIBXS would still load, and the
     * coverage would disappear without any test failing.  Both containers begin
     * with a 4-byte magic followed by the version, written host-endian.
     */
    if (sizeof(version) + 4 <= size) {
      memcpy(&version, (const unsigned char*)buffer + 4, sizeof(version));
    }
    if (1 != version) {
      FPRINTF(stderr, "ERROR: %s declares format version %u, expected 1\n",
        name, (unsigned int)version);
      result = EXIT_FAILURE;
    }
    libxs_predict_query(model, &qinfo);
    if (EXIT_SUCCESS == result && (0 >= qinfo.nentries || 0 >= qinfo.nclusters)) {
      FPRINTF(stderr, "ERROR: %s reports %d entries in %d clusters\n",
        name, qinfo.nentries, qinfo.nclusters);
      result = EXIT_FAILURE;
    }
    if (EXIT_SUCCESS == result
      && (EXIT_SUCCESS != libxs_predict_save(model, NULL, &rsize) || 0 == rsize))
    {
      FPRINTF(stderr, "ERROR: %s save size query failed\n", name);
      result = EXIT_FAILURE;
    }
    if (EXIT_SUCCESS == result) {
      void* rbuf = malloc(rsize);
      if (NULL != rbuf) {
        libxs_predict_t* again = NULL;
        if (EXIT_SUCCESS == libxs_predict_save(model, rbuf, &rsize)) {
          again = libxs_predict_load(rbuf, rsize);
        }
        if (NULL != again) {
          int i, j;
          for (i = 1; i <= NQUERY && EXIT_SUCCESS == result; ++i) {
            double inputs[NINPUTS], out1[NOUTPUTS], out2[NOUTPUTS];
            inputs[0] = inputs[1] = inputs[2] = (double)i;
            libxs_predict_eval(NULL, model, inputs, out1, NULL, 1);
            libxs_predict_eval(NULL, again, inputs, out2, NULL, 1);
            for (j = 0; j < NOUTPUTS; ++j) {
              const double delta = out1[j] - out2[j];
              if (delta > 1e-10 || delta < -1e-10) {
                FPRINTF(stderr, "ERROR: %s migration mismatch at query %d"
                  " output %d (%.6f vs %.6f)\n", name, i, j, out1[j], out2[j]);
                result = EXIT_FAILURE;
              }
            }
          }
          libxs_predict_destroy(again);
        }
        else {
          FPRINTF(stderr, "ERROR: %s failed to reload after save\n", name);
          result = EXIT_FAILURE;
        }
        if (EXIT_SUCCESS == result) {
          libxs_predict_t* damaged;
          ((unsigned char*)rbuf)[rsize / 2] ^= 0xFF;
          damaged = libxs_predict_load(rbuf, rsize);
          if (NULL != damaged) {
            FPRINTF(stderr, "ERROR: %s loaded despite a damaged payload\n", name);
            libxs_predict_destroy(damaged);
            result = EXIT_FAILURE;
          }
        }
        free(rbuf);
      }
      else result = EXIT_FAILURE;
    }
    libxs_predict_destroy(model);
  }
  return result;
}


/**
 * Directory of the test binary, including the trailing separator, which is where
 * the fixtures sit for an in-tree build.
 */
static void internal_predict_bindir(const char* argv0, char* dir, size_t size)
{
  const char* sep = NULL;
  dir[0] = '\0';
  if (NULL != argv0) {
    const char* c = argv0;
    for (; '\0' != *c; ++c) {
      if ('/' == *c || '\\' == *c) sep = c;
    }
  }
  if (NULL != sep) {
    const size_t n = (size_t)(sep - argv0) + 1;
    if (size > n) {
      memcpy(dir, argv0, n);
      dir[n] = '\0';
    }
  }
}


int main(int argc, char* argv[])
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
  if (EXIT_SUCCESS == result) {
    const int nfixtures = (int)(sizeof(internal_predict_fixtures)
      / sizeof(*internal_predict_fixtures));
    const int ndirs = (int)(sizeof(internal_predict_dirs)
      / sizeof(*internal_predict_dirs));
    char bindir[1024];
    int nchecked = 0, i, d;
    internal_predict_bindir(1 <= argc ? argv[0] : NULL, bindir, sizeof(bindir));
    for (i = 0; i < nfixtures && EXIT_SUCCESS == result; ++i) {
      size_t size = 0;
      void* buffer = ('\0' != bindir[0])
        ? internal_predict_read(bindir, internal_predict_fixtures[i], &size)
        : NULL;
      for (d = 0; d < ndirs && NULL == buffer; ++d) {
        buffer = internal_predict_read(internal_predict_dirs[d],
          internal_predict_fixtures[i], &size);
      }
      /**
       * A missing fixture is not a failure: the files are large enough that a
       * checkout may omit them, and an absent reference cannot be violated.
       */
      if (NULL != buffer) {
        result = internal_predict_check(buffer, size,
          internal_predict_fixtures[i]);
        ++nchecked;
        free(buffer);
      }
    }
    FPRINTF(stderr, "Fixtures: %d of %d checked\n", nchecked, nfixtures);
  }
  return result;
}
