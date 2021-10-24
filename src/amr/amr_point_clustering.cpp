#include "amr_internal.hpp"

#include <fftw3.h>

typedef struct cluster_t {
    vec2i centroid;
    std::vector<vec2i> points;
} cluster_t;

/**
 * See question 3.11. from http://www.fftw.org/faq/
 */
static void shift2D(fftw_complex *arr, int N) {
    #pragma omp parallel for schedule(static) collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int offset = offset2(i, j, N, 0);
            arr[offset][0] *= pow(-1.0f, i + j);
            arr[offset][1] *= pow(-1.0f, i + j);
        }
    }
}

/**
 * Given a binary solution vector where '1' represents a point flagged for refinement, this function generates an appropriate set
 * of initial seed points around which we will build clusters. This is done by convolving the binary 'image' with a gaussian
 * filter and locating the peaks in the convolved 'image'.
 * 
 * TODO: applying a gaussian filter in real space might be faster than the fft? O(N) vs O(NlogN) (this is easy, similar to laplacian computation)
 * TODO: generate seeds by convolving signature arrays in x and y direction with a gaussian.
 */
static void gen_initial_cluster_seeds(std::vector<cluster_t> &clusters, int *flagged, int N) {

    int length = N * N; // buffer length

    fftw_plan_with_nthreads(omp_get_max_threads());

    fftw_complex *fft_out, *fft_in;
    fftw_plan p1, p2;

    // Pad with 0's on the boundary to prevent periodic boundary conditions:
    int pad = 10;
    int length_padded = pow_2(N + pad);
    fft_in  = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * length_padded);
    fft_out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * length_padded);

    p1 = fftw_plan_dft_2d(N + pad, N + pad, fft_in, fft_out, FFTW_FORWARD, FFTW_ESTIMATE);
    p2 = fftw_plan_dft_2d(N + pad, N + pad, fft_out, fft_in, FFTW_BACKWARD, FFTW_ESTIMATE);

    // Fill fft input buffer:
    #pragma omp parallel for schedule(static) collapse(2)
    for (int i = 0; i < N + pad; i++) {
        for (int j = 0; j < N + pad; j++) {
            int offset = offset2(i, j, N + pad, 0);
            if (i >= pad / 2 && i < N + pad / 2 && j >= pad / 2 && j < N + pad / 2) {
                fft_in[offset][0] = flagged[offset2(i - pad / 2, j - pad / 2, N, 0)];
                fft_in[offset][1] = 0.0f;
            } else {
                fft_in[offset][0] = 0.0f;
                fft_in[offset][1] = 0.0f;
            }
        }
    }

    // TODO: remove shift operation for performance.
    shift2D(fft_in, N + pad);
    fftw_execute(p1);

    #pragma omp parallel for schedule(static)
    for (int m = 0; m < length_padded; m++) {
        // convolve image with gaussian:
        int i, j;
        coordinate2(&i, &j, m, N + pad, 0);
        double kx = (- (N + pad) / 2.0f + i);
        double ky = (- (N + pad) / 2.0f + j);

        double variance = 0.005f; // controls width of gaussian
        double amp = 1.0f;
        double gauss = amp * exp(-variance * (pow_2(kx) + pow_2(ky)));
        fft_out[m][0] *= gauss;
        fft_out[m][1] *= gauss;
    }

    fftw_execute(p2);

    double *flagged_preprocessed = (double *) calloc(length, sizeof(double));

    // Fill preprocessed image of flagged grid:
    #pragma omp parallel for schedule(static) collapse(2)
    for (int i = 0; i < N + pad; i++) {
        for (int j = 0; j < N + pad; j++) {

            int offset = offset2(i, j, N + pad, 0);
            // result[offset] = fft_in[offset][0];
            if (i >= pad / 2 && i < N + pad / 2 && j >= pad / 2 && j < N + pad / 2) {
                // Remember to include normalisation of the dft.
                double re = fft_in[offset][0] / pow_2(N + pad);
                double im = fft_in[offset][1] / pow_2(N + pad);
                flagged_preprocessed[offset2(i - pad / 2, j - pad / 2, N, 0)] = sqrt(pow_2(re) + pow_2(im));
            }

        }
    }

    fftw_destroy_plan(p1);
    fftw_destroy_plan(p2);
    fftw_free(fft_in);
    fftw_free(fft_out);

    // find local maxima in preprocessed image:
    #pragma omp declare reduction (merge : std::vector<cluster_t> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
    #pragma omp parallel for reduction(merge: clusters)
    for (int m = 0; m < length; m++) {
        int i, j;
        coordinate2(&i, &j, m, N, 0);

        double v = flagged_preprocessed[m];
        // No periodic boundary conditions:
        double v_east  = (i < N - 1) ? flagged_preprocessed[offset2(i+1,j,N,0)] : 0.0f;
        double v_west  = (i > 0) ? flagged_preprocessed[offset2(i-1,j,N,0)] : 0.0f;
        double v_north = (j < N - 1) ? flagged_preprocessed[offset2(i,j+1,N,0)] : 0.0f;
        double v_south = (j > 0) ? flagged_preprocessed[offset2(i,j-1,N,0)] : 0.0f;

        double threshold = 0.001f;
        if (v > threshold && v > v_east && v > v_west && v > v_north && v > v_south)
            clusters.push_back((cluster_t) { .centroid = {i, j}, .points = {}});
    }

#if 0
    // debug
    FILE *fp = fopen("/Users/seanbryan/Documents/UNI/2021T1-2/Project/Axion-Strings/output_files/snapshot-flagged-preprocessed", "w");
    assert(fp != NULL);
    fwrite(flagged_preprocessed, sizeof(double), length, fp);
    fclose(fp);

    for (cluster_t c : clusters) {
        printf("(%d,%d),\n", c.centroid.x, c.centroid.y);
    }
#endif

    free(flagged_preprocessed);
}


void gen_refinement_blocks(std::vector<block_spec_t> &to_refine, int *flagged, block_data b) {

    int b_sv_size = (b.has_buffer) ? b.size - BUFFER_STENCIL : b.size; // size of solution vector

    // Set clusters to initial seed values:
    std::vector<cluster_t> clusters;
    gen_initial_cluster_seeds(clusters, flagged, b.size);

    // Store flagged points:
    std::vector<vec2i> flagged_coords;
    #pragma omp declare reduction (merge : std::vector<vec2i> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
    #pragma omp parallel for collapse(2) reduction(merge: flagged_coords)
    for (int i = 0; i < b_sv_size; i++) {
        for (int j = 0; j < b_sv_size; j++) {
            int offset = offset2(i, j, b.size, b.index_sv);
            if (flagged[offset]) flagged_coords.push_back((vec2i) {i, j});
        }
    }

    // Go next if no points have been flagged:
    if (flagged_coords.size() == 0) return;

    // This should never occur:
    assert(!(clusters.size() == 0 && flagged_coords.size() != 0));

    // Naive k-nearest-neighbour (KNN) clustering solution:
    // TODO: implement proper cluster merging algorithm so we have no overlapping grids
    for (vec2i coord : flagged_coords) {
        int i = coord.x, j = coord.y;

        // Push point into nearest cluster:
        float min_dist = pow_2(i - clusters[0].centroid.x) + pow_2(j - clusters[0].centroid.y);
        int nearest = 0; // track id of nearest cluster.
        for (int c_id = 0; c_id < clusters.size(); c_id++) {
            float dist = pow_2(i - clusters[c_id].centroid.x) + pow_2(j - clusters[c_id].centroid.y);
            if (dist < min_dist) {
                min_dist = dist;
                nearest = c_id;
            }
        }
        clusters[nearest].points.push_back((vec2i) {i, j});
    }

    // For each cluster found, calculate the block starting corner and size:
    for (cluster_t c : clusters) {

        if (c.points.size() == 0) {
            // TODO: this should be handled once proper cluster merging is implemented
            printf("Error: redundant clusters\n");
            continue;
        }

        int min_x, min_y, max_x, max_y;
        min_x = max_x = c.points[0].x;
        min_y = max_y = c.points[0].y;
        for (vec2i pt : c.points) {
            if (pt.x < min_x) min_x = pt.x;
            if (pt.x > max_x) max_x = pt.x;
            if (pt.y < min_y) min_y = pt.y;
            if (pt.y > max_y) max_y = pt.y;
        }

        int size = std::max(max_x - min_x + 1, max_y - min_y + 1); // assuming blocks are square for now
        // Check if block is up against the boundary:
        if (max_x == b.size - 1) min_x = max_x - (size - 1);
        if (max_y == b.size - 1) min_y = max_y - (size - 1);
        vec2i starting_coord = {min_x, min_y};

        to_refine.push_back((block_spec_t) {.coord = starting_coord, .size = size});
    }

}
