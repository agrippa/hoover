#ifndef _HVR_PARTITION_LIST_H
#define _HVR_PARTITION_LIST_H

#include "hoover.h"
#include "hvr_vertex.h"

void hvr_partition_list_init(hvr_partition_t n_partitions,
        hvr_partition_list_t *l);

void prepend_to_partition_list(hvr_vertex_t *curr,
        hvr_partition_list_t *l, hvr_internal_ctx_t *ctx);

void remove_from_partition_list(hvr_vertex_t *curr,
        hvr_partition_list_t *l, hvr_internal_ctx_t *ctx);

void remove_from_partition_list_helper(hvr_vertex_t *curr,
        hvr_partition_t partition, hvr_partition_list_t *l,
        hvr_internal_ctx_t *ctx);

void update_partition_list_membership(hvr_vertex_t *curr,
        hvr_partition_t old_partition, hvr_partition_list_t *l,
        hvr_internal_ctx_t *ctx);

void hvr_partition_list_destroy(hvr_partition_list_t *l);

#endif
