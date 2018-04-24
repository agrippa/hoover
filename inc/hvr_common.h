/* For license: see LICENSE.txt file at top-level */

#ifndef _HOOVER_COMMON_H
#define _HOOVER_COMMON_H

#include <stdint.h>
#include <stdlib.h>

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
typedef size_t vertex_id_t;
typedef vertex_id_t vertex_index_t;

typedef int32_t hvr_time_t;

typedef struct _hvr_internal_ctx_t hvr_internal_ctx_t;
typedef hvr_internal_ctx_t *hvr_ctx_t;

#endif
