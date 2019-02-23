#include <stdlib.h>
#include <string.h>

#include "hvr_2d_edge_set.h"

void hvr_2d_edge_set_init(hvr_2d_edge_set_t *s, size_t dim, size_t max_n_tiles) {
    s->dim = dim;
    s->ntiles_per_dim = (dim + TILE_DIM - 1) / TILE_DIM;

    s->tiles = (hvr_2d_edge_set_tile_t **)malloc(s->ntiles_per_dim *
            s->ntiles_per_dim * sizeof(s->tiles[0]));
    assert(s->tiles);
    memset(s->tiles, 0x00, s->ntiles_per_dim * s->ntiles_per_dim *
            sizeof(s->tiles[0]));

    s->preallocated = (hvr_2d_edge_set_tile_t *)malloc(
            max_n_tiles * sizeof(s->preallocated[0]));
    assert(s->preallocated);
    memset(s->preallocated, 0x00, max_n_tiles * sizeof(s->preallocated[0]));
    for (size_t i = 0; i < max_n_tiles - 1; i++) {
        s->preallocated[i].next = s->preallocated + (i + 1);
    }
    s->preallocated[max_n_tiles - 1].next = NULL;
}

void hvr_2d_set(size_t i, size_t j, hvr_edge_type_t e,
        hvr_2d_edge_set_t *s) {
    assert(i < s->dim);
    assert(j < s->dim);

    size_t i_tile = i / TILE_DIM;
    size_t j_tile = j / TILE_DIM;
    size_t tile_index = i_tile * s->ntiles_per_dim + j_tile;

    size_t i_within_tile = i % TILE_DIM;
    size_t j_within_tile = j % TILE_DIM;

    if (s->tiles[tile_index] == NULL) {
        hvr_2d_edge_set_tile_t *allocated = s->preallocated;
        assert(allocated);
        s->preallocated = allocated->next;
        allocated->next = NULL;

        s->tiles[tile_index] = allocated;
    }

    hvr_tile_ele_t *tile = &(s->tiles[tile_index]->tile[0]);

    size_t tile_bit_offset = i_within_tile * TILE_DIM + j_within_tile;
    tile_bit_offset *= BITS_PER_EDGE;

    size_t tile_word = tile_bit_offset / (sizeof(*tile) * BITS_PER_BYTE);
    size_t tile_bit = tile_bit_offset % (sizeof(*tile) * BITS_PER_BYTE);

    hvr_tile_ele_t clear_mask = 0x3;
    clear_mask = (clear_mask << tile_bit);
    clear_mask = (~clear_mask);

    hvr_tile_ele_t set_mask = e;
    set_mask = (set_mask << tile_bit);

    hvr_tile_ele_t curr_value = tile[tile_word];
    hvr_tile_ele_t new_value = (curr_value & clear_mask);
    new_value = (new_value | set_mask);

    tile[tile_word] = new_value;
}

hvr_edge_type_t hvr_2d_get(size_t i, size_t j, hvr_2d_edge_set_t *s) {
    assert(i < s->dim);
    assert(j < s->dim);

    size_t i_tile = i / TILE_DIM;
    size_t j_tile = j / TILE_DIM;
    size_t tile_index = i_tile * s->ntiles_per_dim + j_tile;

    size_t i_within_tile = i % TILE_DIM;
    size_t j_within_tile = j % TILE_DIM;

    if (s->tiles[tile_index] == NULL) {
        return NO_EDGE;
    }

    hvr_tile_ele_t *tile = &(s->tiles[tile_index]->tile[0]);

    size_t tile_bit_offset = i_within_tile * TILE_DIM + j_within_tile;
    tile_bit_offset *= BITS_PER_EDGE;

    size_t tile_word = tile_bit_offset / (sizeof(*tile) * BITS_PER_BYTE);
    size_t tile_bit = tile_bit_offset % (sizeof(*tile) * BITS_PER_BYTE);

    hvr_tile_ele_t get_mask = 0x3;
    get_mask = (get_mask << tile_bit);

    hvr_tile_ele_t curr_value = tile[tile_word];
    hvr_tile_ele_t masked_value = (curr_value & get_mask);
    masked_value = (masked_value >> tile_bit);
    assert(masked_value <= 3 && masked_value >= 0);
    return masked_value;
}
