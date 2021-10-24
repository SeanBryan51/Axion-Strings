
#include "amr/amr_internal.hpp"

static void setup_grid(std::vector<level_data *> &hierarchy) {

    level_data *root_level = (level_data *) calloc(1, sizeof(level_data));
    root_level->length = 4 * 4;
    root_level->tau_int = 0;
    root_level->b_data = { (block_data) {
        .index_global = 0,
        .index_sv = 0,
        .size = 4,
        .has_buffer = 0,
        .origin_global = {0, 0}
    } };

    hierarchy = { root_level };

    root_level->phi1      = (data_t *) calloc(root_level->length, sizeof(data_t));
    root_level->phi2      = (data_t *) calloc(root_level->length, sizeof(data_t));
    root_level->phidot1   = (data_t *) calloc(root_level->length, sizeof(data_t));
    root_level->phidot2   = (data_t *) calloc(root_level->length, sizeof(data_t));
    root_level->ker1_curr = (data_t *) calloc(root_level->length, sizeof(data_t));
    root_level->ker2_curr = (data_t *) calloc(root_level->length, sizeof(data_t));
    root_level->ker1_next = (data_t *) calloc(root_level->length, sizeof(data_t));
    root_level->ker2_next = (data_t *) calloc(root_level->length, sizeof(data_t));
    root_level->flagged = (int *) calloc(root_level->length, sizeof(int));
    root_level->phi1_prev = (data_t *) calloc(root_level->length, sizeof(data_t));
    root_level->phi2_prev = (data_t *) calloc(root_level->length, sizeof(data_t));
    assert(root_level->phi1 != NULL && root_level->phi2 != NULL && root_level->phidot1 != NULL && root_level->phidot2 != NULL);
    assert(root_level->ker1_curr != NULL && root_level->ker2_curr != NULL && root_level->ker1_next != NULL && root_level->ker2_next != NULL);
    assert(root_level->flagged != NULL);
    assert(root_level->phi1_prev != NULL && root_level->phi2_prev != NULL);

    for (int i = 0; i < root_level->b_data[0].size; i++) {
        for (int j = 0; j < root_level->b_data[0].size; j++) {
            root_level->phi1[offset2(i, j, root_level->b_data[0].size, 0)] = i + j;
        }
    }
}

static void cleanup_hierarchy(std::vector<level_data *> &hierarchy) {
    // Clean up memory:
    for (level_data *data : hierarchy) {
        free(data->phi1);
        free(data->phi2);
        free(data->phidot1);
        free(data->phidot2);
        free(data->ker1_curr);
        free(data->ker2_curr);
        free(data->ker1_next);
        free(data->ker2_next);
        free(data->axion);
        free(data->saxion);
        free(data->flagged);
        free(data);
    }
}

void test_buffer_interpolation_top_left() {

    std::vector<level_data *> hierarchy;
    setup_grid(hierarchy);

    level_data *root_level = hierarchy[0];

    level_data *new_level = (level_data *) calloc(1, sizeof(level_data));
    hierarchy.push_back(new_level);

    new_level->length = (4 + 2) * (4 + 2);
    new_level->tau_int = 0;
    new_level->b_data = { (block_data) {
        .index_global = 0,
        .index_sv = 4 + 2 + 1,
        .size = 4 + 2,
        .has_buffer = 1,
        .origin_global = {0, 0}
    } };

    new_level->phi1      = (data_t *) calloc(new_level->length, sizeof(data_t));
    new_level->phi2      = (data_t *) calloc(new_level->length, sizeof(data_t));
    new_level->phidot1   = (data_t *) calloc(new_level->length, sizeof(data_t));
    new_level->phidot2   = (data_t *) calloc(new_level->length, sizeof(data_t));
    new_level->ker1_curr = (data_t *) calloc(new_level->length, sizeof(data_t));
    new_level->ker2_curr = (data_t *) calloc(new_level->length, sizeof(data_t));
    new_level->ker1_next = (data_t *) calloc(new_level->length, sizeof(data_t));
    new_level->ker2_next = (data_t *) calloc(new_level->length, sizeof(data_t));
    new_level->flagged = (int *) calloc(new_level->length, sizeof(int));
    new_level->phi1_prev = (data_t *) calloc(new_level->length, sizeof(data_t));
    new_level->phi2_prev = (data_t *) calloc(new_level->length, sizeof(data_t));
    assert(new_level->phi1 != NULL && new_level->phi2 != NULL && new_level->phidot1 != NULL && new_level->phidot2 != NULL);
    assert(new_level->ker1_curr != NULL && new_level->ker2_curr != NULL && new_level->ker1_next != NULL && new_level->ker2_next != NULL);
    assert(new_level->flagged != NULL);
    assert(new_level->phi1_prev != NULL && new_level->phi2_prev != NULL);

    int sv_size = new_level->b_data[0].size - BUFFER_STENCIL;
    for (int i = 0; i < sv_size; i++) {
        for (int j = 0; j < sv_size; j++) {
            vec2i global_coordinate = coordinate_global(i, j, new_level->b_data[0]);

            global_to_local_return_t lc = coordinate_global_to_local(1, global_coordinate, 0, root_level->b_data[0]);
            int i_root_level = lc.local_coordinate.x;
            int j_root_level = lc.local_coordinate.y;
            int x_offset = lc.offsets.x;
            int y_offset = lc.offsets.y;

            // compute bilinear interpolation between four points:

            data_t phi1_00 = root_level->phi1[offset2(i_root_level, j_root_level, root_level->b_data[0].size, root_level->b_data[0].index_global + root_level->b_data[0].index_sv)];
            data_t phi1_10 = root_level->phi1[offset2(i_root_level+1, j_root_level, root_level->b_data[0].size, root_level->b_data[0].index_global + root_level->b_data[0].index_sv)];
            data_t phi1_01 = root_level->phi1[offset2(i_root_level, j_root_level+1, root_level->b_data[0].size, root_level->b_data[0].index_global + root_level->b_data[0].index_sv)];
            data_t phi1_11 = root_level->phi1[offset2(i_root_level+1, j_root_level+1, root_level->b_data[0].size, root_level->b_data[0].index_global + root_level->b_data[0].index_sv)];

            data_t v = (1.0f - x_offset / 2.0f) * (1.0f - y_offset / 2.0f) * phi1_00 + x_offset / 2.0f * (1.0f - y_offset / 2.0f) * phi1_10 + (1.0f - x_offset / 2.0f) * y_offset / 2.0f * phi1_01 + x_offset * y_offset / 4.0f * phi1_11;
            new_level->phi1[offset2(i, j, new_level->b_data[0].size, new_level->b_data[0].index_sv)] = v;
        }
    }

    data_t sol[6 * 6] = {
        3.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f,
        1.5f, 0.0f, 0.5f, 1.0f, 1.5f, 2.0f,
        2.0f, 0.5f, 1.0f, 1.5f, 2.0f, 2.5f,
        2.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f,
        3.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f,
        3.5f, 2.0f, 2.5f, 3.0f, 3.5f, 4.0f
    };

    fill_block_buffer(hierarchy, 1, new_level->b_data[0]);
    data_t *phi1 = &hierarchy[1]->phi1[new_level->b_data[0].index_global];
    for (int j = 0; j < new_level->b_data[0].size; j++) {
        for (int i = 0; i < new_level->b_data[0].size; i++) {
            int offset = offset2(i, j, new_level->b_data[0].size, 0);
            assert(sol[offset] == phi1[offset]);
        }
    }

    cleanup_hierarchy(hierarchy);
}


void test_buffer_interpolation_simple() {

    std::vector<level_data *> hierarchy;
    setup_grid(hierarchy);

    level_data *root_level = hierarchy[0];

    level_data *new_level = (level_data *) calloc(1, sizeof(level_data));
    new_level->length = (3 + 2) * (3 + 2);
    new_level->tau_int = 0;
    new_level->b_data = { (block_data) {
        .index_global = 0,
        .index_sv = 3 + 2 + 1,
        .size = 3 + 2,
        .has_buffer = 1,
        .origin_global = {2, 2} // in level 1 units
    } };

    hierarchy.push_back(new_level);

    new_level->phi1      = (data_t *) calloc(new_level->length, sizeof(data_t));
    new_level->phi2      = (data_t *) calloc(new_level->length, sizeof(data_t));
    new_level->phidot1   = (data_t *) calloc(new_level->length, sizeof(data_t));
    new_level->phidot2   = (data_t *) calloc(new_level->length, sizeof(data_t));
    new_level->ker1_curr = (data_t *) calloc(new_level->length, sizeof(data_t));
    new_level->ker2_curr = (data_t *) calloc(new_level->length, sizeof(data_t));
    new_level->ker1_next = (data_t *) calloc(new_level->length, sizeof(data_t));
    new_level->ker2_next = (data_t *) calloc(new_level->length, sizeof(data_t));
    new_level->flagged = (int *) calloc(new_level->length, sizeof(int));
    new_level->phi1_prev = (data_t *) calloc(new_level->length, sizeof(data_t));
    new_level->phi2_prev = (data_t *) calloc(new_level->length, sizeof(data_t));
    assert(new_level->phi1 != NULL && new_level->phi2 != NULL && new_level->phidot1 != NULL && new_level->phidot2 != NULL);
    assert(new_level->ker1_curr != NULL && new_level->ker2_curr != NULL && new_level->ker1_next != NULL && new_level->ker2_next != NULL);
    assert(new_level->flagged != NULL);
    assert(new_level->phi1_prev != NULL && new_level->phi2_prev != NULL);

    int sv_size = new_level->b_data[0].size - BUFFER_STENCIL;
    for (int i = 0; i < sv_size; i++) {
        for (int j = 0; j < sv_size; j++) {
            vec2i global_coordinate = coordinate_global(i, j, new_level->b_data[0]);

            global_to_local_return_t lc = coordinate_global_to_local(1, global_coordinate, 0, root_level->b_data[0]);
            int i_root_level = lc.local_coordinate.x;
            int j_root_level = lc.local_coordinate.y;
            int x_offset = lc.offsets.x;
            int y_offset = lc.offsets.y;

            // compute bilinear interpolation between four points:

            data_t phi1_00 = root_level->phi1[offset2(i_root_level, j_root_level, root_level->b_data[0].size, root_level->b_data[0].index_global + root_level->b_data[0].index_sv)];
            data_t phi1_10 = root_level->phi1[offset2(i_root_level+1, j_root_level, root_level->b_data[0].size, root_level->b_data[0].index_global + root_level->b_data[0].index_sv)];
            data_t phi1_01 = root_level->phi1[offset2(i_root_level, j_root_level+1, root_level->b_data[0].size, root_level->b_data[0].index_global + root_level->b_data[0].index_sv)];
            data_t phi1_11 = root_level->phi1[offset2(i_root_level+1, j_root_level+1, root_level->b_data[0].size, root_level->b_data[0].index_global + root_level->b_data[0].index_sv)];

            data_t v = (1.0f - x_offset / 2.0f) * (1.0f - y_offset / 2.0f) * phi1_00 + x_offset / 2.0f * (1.0f - y_offset / 2.0f) * phi1_10 + (1.0f - x_offset / 2.0f) * y_offset / 2.0f * phi1_01 + x_offset * y_offset / 4.0f * phi1_11;
            new_level->phi1[offset2(i, j, new_level->b_data[0].size, new_level->b_data[0].index_sv)] = v;
        }
    }

    data_t sol_2[5 * 5] = {
        1.0f, 1.5f, 2.0f, 2.5f, 3.0f,
        1.5f, 2.0f, 2.5f, 3.0f, 3.5f,
        2.0f, 2.5f, 3.0f, 3.5f, 4.0f,
        2.5f, 3.0f, 3.5f, 4.0f, 4.5f,
        3.0f, 3.5f, 4.0f, 4.5f, 5.0f,
    };

    fill_block_buffer(hierarchy, 1, new_level->b_data[0]);
    data_t *phi1 = &hierarchy[1]->phi1[new_level->b_data[0].index_global];
    for (int j = 0; j < new_level->b_data[0].size; j++) {
        for (int i = 0; i < new_level->b_data[0].size; i++) {
            int offset = offset2(i, j, new_level->b_data[0].size, 0);
            data_t s = sol_2[offset];
            data_t r = phi1[offset];
            assert(sol_2[offset] == phi1[offset]);
        }
    }

    cleanup_hierarchy(hierarchy);
}

int main(void) {
    assert(BUFFER_STENCIL == 2);

    test_buffer_interpolation_top_left();
    test_buffer_interpolation_simple();

    // TODO: test multiple nested grids
    // TODO: test multiple grids on the same level

    return EXIT_SUCCESS;
}