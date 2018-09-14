/* For license: see LICENSE.txt file at top-level */

#ifndef _HOOVER_COMMON_H
#define _HOOVER_COMMON_H

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

#define BITS_PER_BYTE 8
#define BITS_PER_WORD (BITS_PER_BYTE * sizeof(unsigned))

/*
 * Type definition for vertex IDs and indices
 *
 * A vertex ID is a user-assigned globally unique identifier for a vertex in the
 * distributed graph.
 *
 * A vertex index is a runtime-assigned globally unique identifier for a vertex
 * in the distributed graph that coalesces vertices into a contiguous,
 * one-dimensional address space. It is the runtime's responsibility to map from
 * vertex IDs to vertex indices.
 *
 * It is important to not mix these up.
 */
typedef uint64_t hvr_vertex_id_t;
typedef hvr_vertex_id_t hvr_vertex_index_t;

/*
 * Type definition for a graph identifier.
 *
 * Under the hood, this is used as a bitvector. As a result, we are currently
 * limited to at most 32 graphs.
 */
typedef uint32_t hvr_graph_id_t;

#define HVR_INVALID_GRAPH 0
#define HVR_ALL_GRAPHS 0xFFFFFFFF

/*
 * Vertex ID for any local vertices. For remotely accessible vertices, the top
 * 32 bits stores the PE that the vertex sits on, the bottom 32 bits store its
 * offset in the pool of that PE.
 */
#define HVR_INVALID_VERTEX_ID UINT64_MAX

#define VERTEX_ID_PE(my_id) ((my_id) >> 32)
#define VERTEX_ID_OFFSET(my_id) ((my_id) & UINT32_MAX)

static inline hvr_vertex_id_t construct_vertex_id(int pe, uint64_t offset) {
    uint64_t id = pe;
    id = (id << 32);

    // Double check that the offset is encodable in 32 bits.
    assert(offset <= UINT32_MAX);
    id = (id | offset);

    return id;
}

typedef int32_t hvr_time_t;

typedef struct _hvr_internal_ctx_t hvr_internal_ctx_t;
typedef hvr_internal_ctx_t *hvr_ctx_t;

typedef uint32_t hvr_partition_t;
#define HVR_INVALID_PARTITION UINT32_MAX

static inline size_t get_symm_pool_nelements() {
    static size_t symm_pool_nelements = 0;

    if (symm_pool_nelements == 0) {
        if (getenv("HVR_SYMM_POOL_SIZE")) {
            symm_pool_nelements = atoi(getenv("HVR_SYMM_POOL_SIZE"));
            assert(symm_pool_nelements > 0);
        } else {
            symm_pool_nelements = 1024UL * 1024UL; // Default
        }
    }
    return symm_pool_nelements;
}

#endif
