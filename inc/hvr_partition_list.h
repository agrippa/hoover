#ifndef _HVR_PARTITION_LIST_H
#define _HVR_PARTITION_LIST_H

#include "hoover.h"

void hvr_partition_list_init(hvr_partition_t n_partitions,
        hvr_partition_list_t *l);

hvr_partition_t prepend_to_partition_list(hvr_vertex_t *curr,
        hvr_partition_list_t *l, hvr_internal_ctx_t *ctx);

void remove_from_partition_list(hvr_vertex_t *curr,
        hvr_partition_list_t *l, hvr_internal_ctx_t *ctx);

void remove_from_partition_list_helper(const hvr_vertex_t *curr,
        hvr_partition_t partition, hvr_partition_list_t *l,
        hvr_internal_ctx_t *ctx);

void update_partition_list_membership(hvr_vertex_t *curr,
        hvr_partition_t old_partition, hvr_partition_t optional_new_partition,
        hvr_partition_list_t *l, hvr_internal_ctx_t *ctx);

hvr_vertex_t *hvr_partition_list_head(hvr_partition_t part,
        hvr_partition_list_t *l);

void hvr_partition_list_destroy(hvr_partition_list_t *l);

size_t hvr_partition_list_mem_used(hvr_partition_list_t *l);

#endif
