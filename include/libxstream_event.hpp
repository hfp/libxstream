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
#ifndef LIBXSTREAM_EVENT_HPP
#define LIBXSTREAM_EVENT_HPP


struct libxstream_stream;


struct libxstream_event {
private:
  class slot_type {
    libxstream_stream* m_stream;
    mutable libxstream_signal m_pending;
  public:
    slot_type(): m_stream(0), m_pending(0) {}
    explicit slot_type(libxstream_stream& stream);
    libxstream_stream& stream() { return *m_stream; }
    libxstream_signal pending() const { return m_pending; }
    void pending(libxstream_signal signal) { m_pending = signal; }
    bool match(const libxstream_stream* stream) const {
      return !stream || stream == m_stream;
    }
  };

  static void enqueue(libxstream_stream& stream, libxstream_event::slot_type slots[], size_t& expected, bool reset);
  static void update(libxstream_event::slot_type& slot);

public:
  libxstream_event();

public:
  size_t expected() const;
  void query(bool& occurred, libxstream_stream* stream) const;
  void enqueue(libxstream_stream& stream, bool reset);
  void wait(libxstream_stream* stream);

private:
  size_t m_expected;
  mutable slot_type m_slots[LIBXSTREAM_MAX_DEVICES*LIBXSTREAM_MAX_STREAMS];
};

#endif // LIBXSTREAM_EVENT_HPP
