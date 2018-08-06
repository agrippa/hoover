#include <iostream>
#include <shmem.h>
#include <hoover.h>


// PE N will move a vertex to PE 0 when: ctx->timestep - delay = N. (delay must be at least 1)
// Results:
//   delay == 1: wrong
//   delay == 2: wrong, but a different kind of wrong
//   delay >= 3: correct
const int delay = 1;


hvr_partition_t actor_to_partition(hvr_sparse_vec_t *actor, hvr_ctx_t ctx) {
    // One partition per PE, and vertices are placed at the center of the partitions.
    const double x = hvr_sparse_vec_get(0, actor, ctx);
    for (hvr_partition_t p = 0; p < (unsigned)ctx->npes; p++)
        if (double(p + 1) > x)
            return p;
    assert(0);
}


void update_metadata(hvr_sparse_vec_t *vec,
                     hvr_sparse_vec_t *neighbors,
                     const size_t n_neighbors,
                     hvr_set_t *couple_with,
                     hvr_ctx_t ctx) {
    if ((ctx->timestep - delay == ctx->pe) && (VERTEX_ID_OFFSET(vec->id) == 1)) {
        hvr_sparse_vec_set(0, 0.5, vec, ctx);
        hvr_sparse_vec_set(1, 0.5, vec, ctx);
        std::cout << "At timestep " << ctx->timestep << ", PE " << ctx->pe
                  << " moved its 2nd vertex to (0.5, 0.5)\n";
    }

    if ((ctx->pe == 0) && (VERTEX_ID_OFFSET(vec->id) == 0))
        std::cout << "At timestep " << ctx->timestep << ", vertex 0 of PE 0 has "
                  << n_neighbors << " neighbors, while it should have "
                  << std::max(0, ctx->timestep - delay) << " neighbors.\n";
}


int might_interact(const hvr_partition_t partition,
                   hvr_set_t *partitions,
                   hvr_partition_t *interacting_partitions,
                   unsigned *n_interacting_partitions,
                   const unsigned interacting_partitions_capacity,
                   hvr_ctx_t ctx) {
    // Too lazy to write something smart
    *n_interacting_partitions = ctx->npes;
    for (int i = 0; i < ctx->npes; i++)
        interacting_partitions[i] = i;
    return 1;
}


int check_abort(hvr_vertex_iter_t *iter,
                hvr_ctx_t ctx,
                hvr_set_t *to_couple_with,
                hvr_sparse_vec_t *out_coupled_metric) {
    hvr_sparse_vec_set(0, 0.0, out_coupled_metric, ctx);
    return 0;
}


int main() {
    shmem_init();

    const int mype = shmem_my_pe();
    const int npes = shmem_n_pes();

    // Very small threshold so only vertices at the same point will be considered neighbors
    const double connectivity_threshold = 0.0001;

    // Execute one more step so the vertices can arrive
    const hvr_time_t max_timestep = npes + delay + 1;

    const uint64_t vecs_per_pe = 2;

    hvr_ctx_t hvr_ctx;
    hvr_ctx_create(&hvr_ctx);

    hvr_graph_id_t main_graph = hvr_graph_create(hvr_ctx);

    hvr_sparse_vec_t* vecs = hvr_sparse_vec_create_n(vecs_per_pe, main_graph, hvr_ctx);

    // Place the first vertex at x = PE + 0.5, y = 0.5
    hvr_sparse_vec_set(0, double(mype) + 0.5, vecs, hvr_ctx);
    hvr_sparse_vec_set(1, 0.5, vecs, hvr_ctx);

    // Place the second vertex at x = PE + 0.5, y = 0.1
    hvr_sparse_vec_set(0, double(mype) + 0.5, vecs + 1, hvr_ctx);
    hvr_sparse_vec_set(1, 0.1, vecs + 1, hvr_ctx);

    hvr_init(npes, update_metadata, might_interact, check_abort, actor_to_partition,
             NULL, &main_graph, 1, connectivity_threshold, 0, 1, max_timestep, hvr_ctx);

    hvr_body(hvr_ctx);

    shmem_barrier_all();

    hvr_finalize(hvr_ctx);

    shmem_finalize();
}
