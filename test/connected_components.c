/* For license: see LICENSE.txt file at top-level */

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <shmem.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <limits.h>

#include <hoover.h>

// Timing variables
static unsigned long long start_time = 0;
static unsigned long long time_limit_s = 0;
static long long elapsed_time = 0;
static long long max_elapsed, total_time;

// SHMEM variables
static int pe, npes;
long long p_wrk[SHMEM_REDUCE_MIN_WRKDATA_SIZE];
long p_sync[SHMEM_REDUCE_SYNC_SIZE];

static unsigned n_edges_to_add = 100;
static long long n_edges_added = 0;
static long long total_n_edges_added = 0;
static unsigned int g_seed;
static uint64_t nvertices = 0;
static uint64_t nvertices_per_pe = 0;

// Used to seed the generator.
static inline void fast_srand(int seed) {
    g_seed = seed;
}

// Compute a pseudorandom integer.
// Output value in range [0, 32767]
static inline uint64_t fast_rand(void) {
    g_seed = (214013*g_seed+2531011);
    int lower = (g_seed>>16)&0x7FFF;

    g_seed = (214013*g_seed+2531011);
    int upper = (g_seed>>16)&0x7FFF;

    return (((uint64_t)upper) << 32) + lower;
}

hvr_partition_t actor_to_partition(const hvr_vertex_t *actor, hvr_ctx_t ctx) {
    return HVR_INVALID_PARTITION;
}

hvr_edge_type_t should_have_edge(const hvr_vertex_t *a, const hvr_vertex_t *b,
        hvr_ctx_t ctx) {
    abort();
}

void start_time_step(hvr_vertex_iter_t *iter, hvr_set_t *couple_with,
        hvr_ctx_t ctx) {
    for (int e = 0; e < n_edges_to_add; e++) {
        uint64_t dst_vertex_pe = fast_rand() % npes;
        // uint64_t dst_vertex_pe = ctx->pe;

        uint64_t src_vertex_offset = fast_rand() % nvertices_per_pe;
        uint64_t dst_vertex_offset = fast_rand() % nvertices_per_pe;

        hvr_vertex_id_t src_vert_id = construct_vertex_id(pe,
                src_vertex_offset);
        hvr_vertex_id_t dst_vert_id = construct_vertex_id(dst_vertex_pe,
                dst_vertex_offset);

        hvr_vertex_t *local = hvr_get_vertex(src_vert_id, ctx);
        assert(local);

        hvr_create_edge_with_vertex_id(local, dst_vert_id, BIDIRECTIONAL, ctx);
    }

    n_edges_added += n_edges_to_add;
}

void update_vertex(hvr_vertex_t *vertex, hvr_set_t *couple_with,
        hvr_ctx_t ctx) {
    // Find connected components in graph via label propagation

    hvr_neighbors_t neighbors;
    hvr_get_neighbors(vertex, &neighbors, ctx);

    hvr_vertex_t *neighbor;
    hvr_edge_type_t neighbor_dir;
    hvr_neighbors_next(&neighbors, &neighbor, &neighbor_dir);

    uint64_t min_supernode_lbl = hvr_vertex_get_uint64(0, vertex, ctx);
    while (neighbor) {
        uint64_t neighbor_lbl = hvr_vertex_get_uint64(0, neighbor, ctx);
        if (neighbor_lbl < min_supernode_lbl) {
            min_supernode_lbl = neighbor_lbl;
        }
        hvr_neighbors_next(&neighbors, &neighbor, &neighbor_dir);
    }

    hvr_vertex_set_uint64(0, min_supernode_lbl, vertex, ctx);

    // hvr_release_neighbors(neighbors, neighbor_dirs, n_neighbors, ctx);
}

void might_interact(const hvr_partition_t partition,
        hvr_partition_t *interacting_partitions,
        unsigned *out_n_interacting_partitions,
        unsigned interacting_partitions_capacity,
        hvr_ctx_t ctx) {
    abort();
}

void update_coupled_val(hvr_vertex_iter_t *iter, hvr_ctx_t ctx,
        hvr_vertex_t *out_coupled_metric) {
    hvr_vertex_set(0, 0.0, out_coupled_metric, ctx);
}

int main(int argc, char **argv) {
    hvr_ctx_t hvr_ctx;

    if (argc != 4) {
        fprintf(stderr, "usage: %s <time-limit-in-seconds> <nvertices> "
                "<edges-to-add-per-step>\n", argv[0]);
        return 1;
    }

    time_limit_s = atoi(argv[1]);
    nvertices = atol(argv[2]);
    n_edges_to_add = atoi(argv[3]);

    for (int i = 0; i < SHMEM_REDUCE_SYNC_SIZE; i++) {
        p_sync[i] = SHMEM_SYNC_VALUE;
    }
    shmem_init();

    pe = shmem_my_pe();
    npes = shmem_n_pes();
    assert(nvertices >= npes);

    if ((nvertices % npes) != 0) {
        uint64_t new_nvertices = nvertices + (npes - (nvertices % npes));
        if (pe == 0) {
            fprintf(stderr, "WARNING: rounding # vertices up from %llu to %llu\n",
                    nvertices, new_nvertices);
        }
        nvertices = new_nvertices;
    }
    assert(nvertices % npes == 0);
    nvertices_per_pe = nvertices / npes;

    if (pe == 0) {
        printf("Time limit = %llu s, # vertices = %llu, %d PEs\n", time_limit_s,
                nvertices, npes);
    }

    hvr_ctx_create(&hvr_ctx);

    fast_srand(123 + pe);

    // Initialize the desired # of vertices
    for (int v = 0; v < nvertices_per_pe; v++) {
        hvr_vertex_t *vert = hvr_vertex_create(hvr_ctx);

        // Initially each vertex in its own component
        hvr_vertex_set_uint64(0, vert->id, vert, hvr_ctx);
    }

    hvr_init(1, // # partitions
            update_vertex,
            might_interact,
            update_coupled_val,
            actor_to_partition,
            start_time_step,
            should_have_edge,
            NULL, // should_terminate
            time_limit_s,
            1,
            hvr_ctx);

    shmem_barrier_all();

    start_time = hvr_current_time_us();
    hvr_exec_info info = hvr_body(hvr_ctx);
    elapsed_time = hvr_current_time_us() - start_time;

    // Get a total wallclock time across all PEs
    shmem_longlong_sum_to_all(&total_time, &elapsed_time, 1, 0, 0, npes, p_wrk,
            p_sync);
    shmem_barrier_all();

    // Get a max wallclock time across all PEs
    shmem_longlong_max_to_all(&max_elapsed, &elapsed_time, 1, 0, 0, npes, p_wrk,
            p_sync);
    shmem_barrier_all();

    shmem_longlong_sum_to_all(&total_n_edges_added, &n_edges_added, 1, 0, 0,
            npes, p_wrk, p_sync);
    shmem_barrier_all();

    if (pe == 0) {
        printf("%d PEs, total CPU time = %f ms, max elapsed = %f ms, "
                "%d iterations completed on PE 0\n", npes,
                (double)total_time / 1000.0, (double)max_elapsed / 1000.0,
                info.executed_iters);
        printf("%lld edges inserted across all PEs\n", total_n_edges_added);

        for (size_t i = 0; i < hvr_ctx->my_vert_subs.nsegs; i++) {
            hvr_sparse_arr_seg_t *seg = (hvr_ctx->my_vert_subs.segs)[i];
            if (seg) {
                for (int j = 0; j < HVR_SPARSE_ARR_SEGMENT_SIZE; j++) {
                    unsigned len = seg->seg_size[j];
                    if (len > 0) {
                        printf("%lu %d %d\n", i, j, len);
                    }
                }
            }
        }

#if 0
        hvr_vertex_iter_t iter;
        hvr_vertex_iter_init(&iter, hvr_ctx);
        for (hvr_vertex_t *vert = hvr_vertex_iter_next(&iter); vert;
                vert = hvr_vertex_iter_next(&iter)) {
            
            hvr_vertex_t **neighbors;
            hvr_edge_type_t *neighbor_dirs;
            int n_neighbors = hvr_get_neighbors(vert, &neighbors,
                    &neighbor_dirs, hvr_ctx);

            if (n_neighbors > 0) {
                printf("PE %d vert %llu # neighbors %d\n",
                        hvr_ctx->pe, vert->id, n_neighbors);
            }

            hvr_release_neighbors(neighbors, neighbor_dirs, n_neighbors, hvr_ctx);
        }
#endif
    }

    hvr_finalize(hvr_ctx);

    shmem_finalize();

    return 0;
}
