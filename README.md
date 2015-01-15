LIBXSTREAM
==========
Library to work with streams, events, and code regions that are able to run asynchronous while preserving the usual stream conditions. The library is targeting Intel Architecture (x86) and helps to offload work to an Intel Xeon Phi coprocessor (an instance of the Intel Many Integrated Core Architecture "MIC"). For example, using two streams may be an alternative to the usual double-buffering approach used to hide buffer transfer time behind compute using these buffers.

The library's C API completely seals the implementation and only forward declares some types. Beside of some minor syntactical sugar, the C++ API allows make use of the stream and event types directly. The C++ API is currently required when defining own code which can be enqueued into a stream. However, a future release will allow to only rely on a function pointer, and also provide a native FORTRAN interface. The library's implementation allows enqueuing work from multiple host threads in a thread-safe manner.

The actual implementation vehicle can be configured using a configuration header. Currently Intel's Language Extensions for Offload (LEO) are used to perform asynchronous execution and data transfers using signal/wait clauses. Other mechanism could used e.g., hStreams or COI (both are part of the Intel Manycore Platform Software Stack), or OpenMP 4.0 offload directives.
