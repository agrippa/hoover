/* For license: see LICENSE.txt file at top-level */

#ifndef _HOOVER_COMMON_H
#define _HOOVER_COMMON_H

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

#define BITS_PER_BYTE 8
#define BITS_PER_WORD (BITS_PER_BYTE * sizeof(unsigned))

typedef enum _hvr_edge_type_t {
    NO_EDGE = 0,
    DIRECTED_IN = 1,
    DIRECTED_OUT = 2,
    BIDIRECTIONAL = 3
} hvr_edge_type_t;

typedef enum _hvr_edge_create_type_t {
    IMPLICIT_EDGE = 0,
    EXPLICIT_EDGE = 1
} hvr_edge_create_type_t;

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
    id = (id | offset);
    return id;
}

typedef int32_t hvr_time_t;

typedef struct _hvr_internal_ctx_t hvr_internal_ctx_t;
typedef hvr_internal_ctx_t *hvr_ctx_t;

typedef uint32_t hvr_partition_t;
#define HVR_INVALID_PARTITION UINT32_MAX

typedef struct _hvr_vertex_update_t hvr_vertex_update_t;

typedef hvr_vertex_id_t hvr_edge_info_t;

static inline hvr_edge_type_t flip_edge_direction(hvr_edge_type_t dir) {
    switch (dir) {
        case (DIRECTED_IN):
            return DIRECTED_OUT;
        case (DIRECTED_OUT):
            return DIRECTED_IN;
        case (BIDIRECTIONAL):
            return BIDIRECTIONAL;
        case (NO_EDGE):
            return NO_EDGE;
        default:
            abort();
    }
}

// Select out the bottom 61 bits for the vertex ID
#define EDGE_INFO_VERTEX(my_edge_info) (0x1fffffffffffffff & (my_edge_info))
// Select out the 62nd bit for the creation type (implicit or explicit)
#define EDGE_INFO_CREATION(my_edge_info) ((0x2fffffffffffffff & (my_edge_info)) >> (uint64_t)61)
// Select out the top 2 bits (63 and 64) for edge type
#define EDGE_INFO_EDGE(my_edge_info) ((hvr_edge_type_t)((my_edge_info) >> 62))

static inline hvr_edge_info_t construct_edge_info(hvr_vertex_id_t vert,
        hvr_edge_type_t edge, hvr_edge_create_type_t creation_type) {
    uint64_t vertex_info = vert;
    uint64_t edge_mask = ((uint64_t)edge) << ((uint64_t)62);
    uint64_t creation_mask = ((uint64_t)creation_type) << ((uint64_t)61);

    /*
     * Assert that top 3 bits are unused in vertex ID, and so that space can be
     * used by edge and creation type.
     */
    assert(EDGE_INFO_VERTEX(vertex_info) == vertex_info);

    vertex_info = (vertex_info | creation_mask | edge_mask);
    return vertex_info;
}

typedef struct _process_perf_info_t {
    unsigned n_received_updates;
    unsigned long long time_handling_deletes;
    unsigned long long time_handling_news;
    unsigned long long time_updating;
    unsigned long long time_updating_edges;
    unsigned long long time_creating_edges;
    unsigned count_new_should_have_edges;
    unsigned long long time_creating;
} process_perf_info_t;

extern void *shmem_malloc_wrapper(size_t nbytes);
extern void *malloc_helper(size_t nbytes);

#endif
