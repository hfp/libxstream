/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "ozaki_opencl.h"
#include <libxs/libxs_mem.h>


char* ozaki_wgmma_splice(const char* ptx, size_t size, const char* entry_name, int maxnreg)
{
  static const char marker[] = "// WGMMA_SLOT n";
  static const char marker_wait[] = "// WGMMA_WAIT";
  static const char marker_fence[] = "// WGMMA_FENCE";
  static const char marker_commit[] = "// WGMMA_COMMIT";
  /* An instruction as the kernel wrote it, its operands named by the compiler: uncommented. */
  static const char marker_splice[] = "// SPLICE ";
  char entry[128];
  char* retargeted = NULL;
  size_t size_rt = 0, cap = 0;
  char* out = NULL;
  LIBXS_SNPRINTF(entry, sizeof(entry), ".entry %s", entry_name);
  if (EXIT_SUCCESS == libxstream_opencl_retarget_ptx(ptx, size,
    maxnreg, entry_name, &retargeted, &size_rt))
  {
    const char* scan = retargeted;
    size_t nmarker = 0;
    while (NULL != (scan = strstr(scan, marker))) {
      ++nmarker;
      ++scan;
    }
    scan = retargeted;
    while (NULL != (scan = strstr(scan, marker_wait))) {
      ++nmarker;
      ++scan;
    }
    scan = retargeted;
    while (NULL != (scan = strstr(scan, marker_fence))) {
      ++nmarker;
      ++scan;
    }
    scan = retargeted;
    while (NULL != (scan = strstr(scan, marker_commit))) {
      ++nmarker;
      ++scan;
    }
    scan = retargeted;
    while (NULL != (scan = strstr(scan, marker_splice))) {
      ++nmarker;
      ++scan;
    }
    if (0 != nmarker) {
      cap = size_rt + nmarker * 1024 + 256;
      out = (char*)libxs_malloc(NULL, cap, 0 /*auto-align*/);
    }
    if (NULL != out) {
      const char* src = retargeted;
      const char* found = strstr(retargeted, entry);
      const char* brace = (NULL != found) ? strchr(found, '{') : NULL;
      size_t off = 0;
      int ok = (NULL != brace) ? EXIT_SUCCESS : EXIT_FAILURE;
      if (EXIT_SUCCESS == ok) { /* the B descriptor temporary, declared once */
        const size_t prefix = (size_t)(brace - retargeted) + 1;
        memcpy(out, retargeted, prefix);
        off = prefix;
        off += (size_t)LIBXS_SNPRINTF(out + off, cap - off, "\n\t.reg .b64 %%wgdb;\n");
        src = brace + 1;
      }
      while (EXIT_SUCCESS == ok) {
        const char* const w = strstr(src, marker_wait);
        const char* const f = strstr(src, marker_fence);
        const char* const c = strstr(src, marker_commit);
        const char* const m = strstr(src, marker);
        const char* const x = strstr(src, marker_splice);
        const char* first = w;
        const char* dlist;
        const char* dend;
        const char* line;
        const char* eol;
        size_t head;
        char pb[32];
        int width = 0;
        if (NULL == first || (NULL != f && f < first)) first = f;
        if (NULL == first || (NULL != c && c < first)) first = c;
        if (NULL == first || (NULL != m && m < first)) first = m;
        if (NULL == first || (NULL != x && x < first)) first = x;
        if (NULL == first) break;
        line = first;
        while (line > retargeted && '\n' != line[-1]) --line;
        eol = strchr(first, '\n');
        head = (size_t)(line - src);
        if (NULL == eol || cap <= (off + head + 1024)) {
          ok = EXIT_FAILURE;
        }
        else if (first == w) { /* the group wait; an optional count keeps that many groups in flight */
          int nwait = 0;
          if (1 != sscanf(w + sizeof(marker_wait) - 1, "%i", &nwait) || 0 > nwait) nwait = 0;
          memcpy(out + off, src, head);
          off += head;
          off += (size_t)LIBXS_SNPRINTF(out + off, cap - off, "\twgmma.wait_group.sync.aligned %i;\n", nwait);
          src = eol + 1;
        }
        else if (first == c) { /* closes a group, which is what a wait can then leave in flight */
          memcpy(out + off, src, head);
          off += head;
          off += (size_t)LIBXS_SNPRINTF(out + off, cap - off, "\twgmma.commit_group.sync.aligned;\n");
          src = eol + 1;
        }
        else if (first == x) { /* the rest of the line is the instruction */
          const char* const text = x + sizeof(marker_splice) - 1;
          memcpy(out + off, src, head);
          off += head;
          out[off++] = '\t';
          memcpy(out + off, text, (size_t)(eol - text) + 1);
          off += (size_t)(eol - text) + 1;
          src = eol + 1;
        }
        else if (first == f) { /* the fence pair, once per round rather than per issue */
          memcpy(out + off, src, head);
          off += head;
          off += (size_t)LIBXS_SNPRINTF(out + off, cap - off,
            "\tfence.proxy.async.shared::cta;\n"
            "\twgmma.fence.sync.aligned;\n");
          src = eol + 1;
        }
        else {
          const char* alist = NULL;
          const char* aend = NULL;
          char etype[8];
          int sbo = 0, lbo = 0, bf16 = 0;
          dlist = strstr(m, " d={");
          dend = (NULL != dlist) ? strchr(dlist, '}') : NULL;
          if (NULL != dlist) dlist += 4; /* past " d={" */
          if (NULL != dend && 0 == strncmp(dend, "} a={", 5)) {
            alist = dend + 5;
            aend = strchr(alist, '}');
          }
          if (4 == sscanf(m + sizeof(marker) - 1, "%i %7s sbo=%i lbo=%i", &width, etype, &sbo, &lbo)) {
            bf16 = (0 == strcmp("bf16", etype)) ? 1 : 0;
            if (0 == bf16 && 0 != strcmp("u8", etype) && 0 != strcmp("s8", etype)) width = 0;
          }
          else width = 0;
          if (NULL == dend || NULL == alist || NULL == aend || eol < aend ||
              (64 != width && 128 != width) || 0 >= sbo || 0 >= lbo || 16383 < sbo || 16383 < lbo ||
              1 != sscanf(aend, "} pb=%31s", pb))
          {
            ok = EXIT_FAILURE;
          }
          else {
            /**
             * The floating-point form of wgmma is not the integer form with a different
             * type suffix: it takes four more trailing immediates (imm-scale-a,
             * imm-scale-b, imm-trans-a, imm-trans-b) after scale-d, and ptxas rejects the
             * integer argument list with "Arguments mismatch for instruction
             * 'wgmma.mma_async'" rather than anything that points at the cause. Verified
             * against the hardware with a standalone probe before it went in here.
             */
            const char* const atype = (0 != bf16) ? "f32" : "s32";
            /* RS form (A from registers) takes no imm-trans-a, unlike SS: seven args, not eight. */
            const char* const dtail = (0 != bf16) ? ", 1, 1, 1, 0" : ", 1";
            const int kdim = (0 != bf16) ? 16 : 32;
            const unsigned int desc_hi = (unsigned int)sbo; /* column-octet stride / 16 */
            const unsigned int desc_lo = (unsigned int)lbo << 16; /* k-block stride / 16, at bit 16 */
            const size_t ndig = (size_t)(dend - dlist);
            memcpy(out + off, src, head);
            off += head;
            off += (size_t)LIBXS_SNPRINTF(out + off, cap - off,
              "\tshr.u64 %%wgdb, %s, 4;\n\tand.b64 %%wgdb, %%wgdb, 16383;\n"
              "\tor.b64 %%wgdb, %%wgdb, 0x%x%08x;\n"
              "\twgmma.mma_async.sync.aligned.m64n%ik%i.%s.%s.%s {",
              pb, desc_hi, desc_lo, width, kdim, atype, etype, etype);
            memcpy(out + off, dlist, ndig);
            off += ndig;
            off += (size_t)LIBXS_SNPRINTF(out + off, cap - off, "}, {");
            memcpy(out + off, alist, (size_t)(aend - alist));
            off += (size_t)(aend - alist);
            off += (size_t)LIBXS_SNPRINTF(out + off, cap - off, "}, %%wgdb%s;\n", dtail);
            src = eol + 1;
          }
        }
      }
      if (EXIT_SUCCESS == ok) {
        const size_t tail = strlen(src);
        if (cap > (off + tail)) {
          memcpy(out + off, src, tail);
          out[off + tail] = '\0';
        }
        else ok = EXIT_FAILURE;
      }
      /**
       * The family target where nothing needs the architecture-specific one: a program
       * without kind::i8 and without wgmma runs on every member of the family, including
       * one newer than the compiler that emitted the PTX knows (sm_103f on 10.7). A family
       * target needs PTX 8.8, and a lower version's text is valid at 8.8.
       */
      if (EXIT_SUCCESS == ok && NULL == strstr(out, "kind::i8") && NULL == strstr(out, "wgmma.")) {
        char* const target = strstr(out, ".target sm_");
        char* const version = strstr(out, ".version ");
        char* sfx = (NULL != target) ? (target + 11) : NULL;
        while (NULL != sfx && '0' <= *sfx && '9' >= *sfx) ++sfx;
        if (NULL != sfx && 'a' == *sfx) *sfx = 'f';
        if (NULL != version && '0' <= version[9] && '9' >= version[9] && '.' == version[10]
          && '0' <= version[11] && '9' >= version[11] && 0 == ('0' <= version[12] && '9' >= version[12])
          && 88 > (version[9] - '0') * 10 + (version[11] - '0'))
        {
          version[9] = '8';
          version[11] = '8';
        }
      }
      if (EXIT_SUCCESS != ok) {
        libxs_free(out);
        out = NULL;
      }
    }
    libxs_free(retargeted);
  }
  return out;
}
