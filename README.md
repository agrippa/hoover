# HOOVER
Low Latency, Distributed, Flexible, Streaming Graph Analytics

HOOVER is a software framework for modeling systems that can be expressed as a
streaming, dynamic graph problem. HOOVER's backend coordinates the
communication, execution, and memory management needed to scalably execute the
specified problem. The application developer simply provides callbacks that
implement application-specific functionality (e.g. updating the edges of a node
in a graph).

HOOVER is a distributed graph processing framework, enabling it to support and
scale to larger graph problems than other shared memory or GPU-based frameworks.
In the backend, HOOVER uses any OpenSHMEM 1.4 compliant (or later) runtime for
SPMD process creation and inter-process communication. HOOVER has been tested on
the following OpenSHMEM implementations:

1. OSSS (https://bitbucket.org/sbuopenshmem/osss-ucx)
2. SoS (https://github.com/Sandia-OpenSHMEM/SOS)
3. Cray SHMEM 7.7.0

For more information on building and using HOOVER, please see the top-level
doc/ folder in this repo.
