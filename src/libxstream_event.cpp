/******************************************************************************
** Copyright (c) 2014-2015, Intel Corporation                                **
** All rights reserved.                                                      **
**                                                                           **
** Redistribution and use in source and binary forms, with or without        **
** modification, are permitted provided that the following conditions        **
** are met:                                                                  **
** 1. Redistributions of source code must retain the above copyright         **
**    notice, this list of conditions and the following disclaimer.          **
** 2. Redistributions in binary form must reproduce the above copyright      **
**    notice, this list of conditions and the following disclaimer in the    **
**    documentation and/or other materials provided with the distribution.   **
** 3. Neither the name of the copyright holder nor the names of its          **
**    contributors may be used to endorse or promote products derived        **
**    from this software without specific prior written permission.          **
**                                                                           **
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              **
******************************************************************************/
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#include "libxstream_event.hpp"
#include "libxstream_capture.hpp"

#include <algorithm>

#if defined(LIBXSTREAM_OFFLOAD)
# include <offload.h>
#endif

// allows to wait for an event issued prior to the pending signal
//#define LIBXSTREAM_EVENT_WAIT_PAST
// alternative wait routine; waits until the effect occurred
//#define LIBXSTREAM_EVENT_WAIT_OCCURRED


/*static*/void libxstream_event::enqueue(int thread, libxstream_stream& stream, libxstream_event::slot_type slots[], size_t& expected, bool reset)
{
#if defined(LIBXSTREAM_DEBUG)
  LIBXSTREAM_ASSERT((LIBXSTREAM_MAX_NDEVICES * LIBXSTREAM_MAX_NSTREAMS) > ((reset && 0 < expected) ? (expected - 1) : expected));
#endif

  if (reset) {
#if defined(LIBXSTREAM_DEBUG)
    std::fill_n(slots, LIBXSTREAM_MAX_NDEVICES * LIBXSTREAM_MAX_NSTREAMS, slot_type());
#endif
    expected = 0;
  }

  slot_type& slot = slots[expected];
  slot = slot_type(thread, stream);
  ++expected;
}


/*static*/void libxstream_event::update(int thread, libxstream_event::slot_type& slot)
{
  const libxstream_signal pending_slot = slot.pending();

  if (0 != pending_slot) {
    const libxstream_signal pending_stream = slot.stream().pending(thread);

    if (0 != pending_stream) {
#if defined(LIBXSTREAM_EVENT_WAIT_PAST)
      const libxstream_signal signal = pending_slot;
#else
      const libxstream_signal signal = pending_stream;
#endif
#if defined(LIBXSTREAM_OFFLOAD) && !defined(__MIC__)
      if (0 != _Offload_signaled(slot.stream().device(), reinterpret_cast<void*>(signal)))
#endif
      {
        if (signal == pending_stream) {
          slot.stream().pending(thread, 0);
        }
        slot.pending(0);
      }
    }
    else {
      slot.pending(0);
    }
  }
}


libxstream_event::slot_type::slot_type(int thread, libxstream_stream& stream)
  : m_stream(&stream) // no need to lock the stream
  , m_pending(stream.pending(thread))
{}


libxstream_event::libxstream_event()
  : m_expected(0)
{}


size_t libxstream_event::expected() const
{
  LIBXSTREAM_ASSERT((LIBXSTREAM_MAX_NDEVICES * LIBXSTREAM_MAX_NSTREAMS) >= m_expected);
  return m_expected;
}


void libxstream_event::enqueue(libxstream_stream& stream, bool reset)
{ 
  LIBXSTREAM_ASYNC_BEGIN(stream, m_slots, &m_expected, static_cast<libxstream_bool>(reset ? LIBXSTREAM_TRUE : LIBXSTREAM_FALSE))
  {
    libxstream_event::enqueue(thread(), *LIBXSTREAM_ASYNC_STREAM, ptr<slot_type,0>(), *ptr<size_t,1>(), val<bool,2>());
  }
  LIBXSTREAM_ASYNC_END(LIBXSTREAM_CALL_UNLOCK);
}


int libxstream_event::query(bool& occurred, libxstream_stream* stream) const
{
  int result = LIBXSTREAM_ERROR_NONE;

  LIBXSTREAM_ASYNC_BEGIN(stream, &result, &m_expected, m_slots, &occurred)
  {
    const size_t expected = *ptr<const size_t,1>();
    slot_type *const slots = ptr<slot_type,2>();
    bool occurred = true; // everythig "occurred" if nothing is expected

    if (LIBXSTREAM_ASYNC_STREAM) {
      *ptr<int,0>() = LIBXSTREAM_ASYNC_STREAM->reset();
    }

    for (size_t i = 0; i < expected; ++i) {
      slot_type& slot = slots[i];

      if (slot.match(LIBXSTREAM_ASYNC_STREAM) && 0 != slot.pending()) {
        libxstream_event::update(thread(), slot);
        occurred = occurred && 0 == slot.pending();
      }
    }

    *ptr<bool,3>() = occurred;
  }
  LIBXSTREAM_ASYNC_END(LIBXSTREAM_CALL_WAIT);

  return result;
}


int libxstream_event::wait(libxstream_stream* stream)
{
  int result = LIBXSTREAM_ERROR_NONE;

  LIBXSTREAM_ASYNC_BEGIN(stream, &result, &m_expected, m_slots)
  {
    size_t& expected = *ptr<size_t,1>();
    slot_type *const slots = ptr<slot_type,2>();
    size_t completed = 0;

    if (LIBXSTREAM_ASYNC_STREAM) {
      *ptr<int,0>() = LIBXSTREAM_ASYNC_STREAM->reset();
    }

    for (size_t i = 0; i < expected; ++i) {
      slot_type& slot = slots[i];
      const libxstream_signal pending_stream = slot.stream().pending(thread());
      const libxstream_signal pending_slot = slot.pending();

      if (slot.match(LIBXSTREAM_ASYNC_STREAM) && 0 != pending_stream && 0 != pending_slot) {
#if defined(LIBXSTREAM_EVENT_WAIT_OCCURRED)
        do { // spin/yield
          libxstream_event::update(thread(), slot);
          this_thread_yield();
        }
        while(0 != slot.pending());
#else
# if defined(LIBXSTREAM_EVENT_WAIT_PAST)
        const libxstream_signal signal = pending_slot;
# else
        const libxstream_signal signal = pending_stream;
# endif
# if defined(LIBXSTREAM_OFFLOAD) && defined(LIBXSTREAM_ASYNC) && (0 != (2*LIBXSTREAM_ASYNC+1)/2)
        if (0 <= slot.stream().device()) {
          LIBXSTREAM_ASYNC_DEVICE_UPDATE(slot.stream().device());
#         pragma offload_wait LIBXSTREAM_ASYNC_TARGET wait(signal)
        }
# endif
        if (signal == pending_stream) {
          slot.stream().pending(thread(), 0);
        }
#endif
        ++completed;
      }
    }

    LIBXSTREAM_ASSERT(completed <= expected);
    expected -= completed;
  }
  LIBXSTREAM_ASYNC_END(LIBXSTREAM_CALL_WAIT | LIBXSTREAM_CALL_UNLOCK);

  return result;
}
