#include "amr_internal.hpp"

#include <fftw3.h>

typedef struct cluster_t {
    vec2i centroid;
    std::vector<vec2i> points;
} cluster_t;

static inline double gaussian_kernel(int *flagged, int i, int j, int N, int so = 0) {
    return 0.0f;
}

/**
 * See question 3.11. from http://www.fftw.org/faq/
 */
static void shift2D(fftw_complex *arr, int N) {
    int length = get_length();
    #pragma omp parallel for schedule(static)
    for (int m = 0; m < length; m++) {
        int i, j;
        coordinate2(&i, &j, m, N);
        arr[m][0] *= pow(-1.0f, i + j);
        arr[m][1] *= pow(-1.0f, i + j);
    }
}

/**
 * Given a binary solution vector where '1' represents a point flagged for refinement, this function generates an appropriate set
 * of initial seed points around which we will build clusters. This is done by convolving the binary 'image' with a gaussian
 * filter and locating the peaks in the convolved 'image'.
 * 
 * TODO: applying a gaussian filter in real space might be faster than the fft? O(N) vs O(NlogN)
 * 
 * @param ld : "leading dimension" of image data, i.e. the dimension of the buffer.
 * 
 */
static void gen_initial_cluster_seeds(std::vector<cluster_t> &clusters, level_data data, int b_id) {

    // buffer size
    int b_size = data.b_data[b_id].size;
    // size of solution vector
    int b_sv_size = (data.b_data[b_id].has_buffer) ? b_size - 2 : b_size; // TODO: hard coded stencil = 2
    // starting index for solution vector
    int b_sv_idx = data.b_data[b_id].index_sv;
    // max offset at which i = j = b_size - stencil - 1 = b_sv_size - 1
    int b_max_offset = b_sv_idx + (b_sv_size - 1) + b_size * (b_sv_size - 1); // assume 2D for now.

    int *flagged = &data.flagged[data.b_data[b_id].index_global];

    // Apply gaussian blur:
    double *flagged_convolved = (double *) calloc(b_size*b_size, sizeof(double));

    #pragma omp parallel for schedule(static) collapse(2)
    for (int i = 0; i < b_sv_size; i++) {
        for (int j = 0; j < b_sv_size; j++) {
            int offset = offset2(i, j, b_size, b_sv_idx);
            // 3x3 gaussian kernel (assuming 2D of course):
            flagged_convolved[offset] = gaussian_kernel(flagged, i, j, b_size, b_sv_idx);
        }
    }

    // Apply maximum filter and find peaks:

    free(flagged_convolved);

#if 0
    int length = pow_2(b_size);

    fftw_plan_with_nthreads(omp_get_max_threads());

    fftw_complex *f_fourier, *f_real;
    fftw_plan p1, p2;

    f_fourier = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * length);
    f_real    = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * length);

    #pragma omp parallel for schedule(static)
    for (int m = 0; m < length; m++) {
        f_real[m][0] = (double) flagged[m];
        f_real[m][1] = 0.0f;
    }

    p1 = fftw_plan_dft_2d(b_size, b_size, f_real, f_fourier, FFTW_FORWARD, FFTW_ESTIMATE);
    p2 = fftw_plan_dft_2d(b_size, b_size, f_fourier, f_real, FFTW_BACKWARD, FFTW_ESTIMATE);

    shift2D(f_real, b_size);
    fftw_execute(p1);

    #pragma omp parallel for schedule(static)
    for (int m = 0; m < length; m++) {
        // convolve image with gaussian:
        int i, j;
        coordinate2(&i, &j, m, b_size);
        double x = (- b_size / 2.0f + i);
        double y = (- b_size / 2.0f + j);

        double variance = 0.01f; // controls width of gaussian
        double gauss = exp(-variance * (pow_2(x) + pow_2(y)));
        f_fourier[m][0] *= gauss;
        f_fourier[m][1] *= gauss;
    }

    fftw_execute(p2);

    double *flagged_preprocessed = (double *) calloc(length, sizeof(double));

    #pragma omp parallel for schedule(static)
    for (int m = 0; m < length; m++) {
        // Remember to include normalisation of the dft.
        double re = f_real[m][0] / pow_2(b_size);
        double im = f_real[m][1] / pow_2(b_size);
        flagged_preprocessed[m] = sqrt(pow_2(re) + pow_2(im));
    }

    #pragma omp declare reduction (merge : std::vector<cluster_t> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))

    // find local maxima in preprocessed image:
    #pragma omp parallel for reduction(merge: clusters)
    for (int m = 0; m < length; m++) {
        int i, j;
        coordinate2(&i, &j, m, b_size);

        double v = flagged_preprocessed[m];
        double v_east  = flagged_preprocessed[offset2(i+1,j,b_size)];
        double v_west  = flagged_preprocessed[offset2(i-1,j,b_size)];
        double v_north = flagged_preprocessed[offset2(i,j+1,b_size)];
        double v_south = flagged_preprocessed[offset2(i,j-1,b_size)];

        double threshold = 0.001f;
        if (v > threshold && v > v_east && v > v_west && v > v_north && v > v_south)
            clusters.push_back((cluster_t) { .centroid = {i, j}, .points = {}});
    }

    // TODO: need to resolve case when two peaks are close to each other?

#if 0
    // debugging
    fp = fopen("/Users/seanbryan/Documents/UNI/2021T1-2/Project/Axion-Strings/output_files/snapshot-flagged-preprocessed", "w");
    assert(fp != NULL);
    fwrite(flagged_preprocessed, sizeof(double), length, fp);
    fclose(fp);
#endif

    fftw_destroy_plan(p1);
    fftw_destroy_plan(p2);
    fftw_free(f_real);
    fftw_free(f_fourier);
    free(flagged_preprocessed);
#endif
}


void gen_refinement_blocks(std::vector<vec2i> &block_coords, std::vector<int> &block_size, std::vector<level_data> hierarchy, int level) {

    level_data data = hierarchy[level];

    int n_blocks = data.b_data.size();
    for (int b_id = 0; b_id < n_blocks; b_id++) {
        // buffer size
        int b_size = data.b_data[b_id].size;
        // size of solution vector
        int b_sv_size = (data.b_data[b_id].has_buffer) ? b_size - 2 : b_size; // TODO: hard coded stencil = 2
        // starting index for solution vector
        int b_sv_idx = data.b_data[b_id].index_sv;
        // max offset at which i = j = b_size - stencil - 1 = b_sv_size - 1
        int b_max_offset = b_sv_idx + (b_sv_size - 1) + b_size * (b_sv_size - 1); // assume 2D for now.

        int *flagged = &data.flagged[data.b_data[b_id].index_global];

        std::vector<cluster_t> clusters;

        gen_initial_cluster_seeds(clusters, data, b_id);

        // Naive k-nearest-neighbour (KNN) solution:

        std::vector<vec2i> flagged_coords;

        #pragma omp declare reduction (merge : std::vector<vec2i> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))

        #pragma omp parallel for collapse(2) reduction(merge: flagged_coords)
        for (int i = 0; i < b_sv_size; i++) {
            for (int j = 0; j < b_sv_size; j++) {
                int offset = offset2(i, j, b_size, b_sv_idx);
                if (flagged[offset])
                    flagged_coords.push_back((vec2i) {i, j});
            }
        }

        if (clusters.size() == 0 && flagged_coords.size() != 0) {
            // TODO: handle this
            printf("Error: clusters seeds unable to detect flagged points.\n");
            return;
        }

        for (vec2i coord : flagged_coords) {
            int i = coord.x, j = coord.y;

            // Push point into nearest cluster:
            // TODO: don't forget periodic boundary conditions LOL.
            // Soln: fuck this, impose that clusters cannot be on top of the periodic boundary (can't even get my head around it)
            //       need to modify point clustering function so seeds are generated as if there are no periodic boundary conds.
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

        // for each cluster found, calculate the block starting corner and size:
        for (cluster_t c : clusters) {

            if (c.points.size() == 0) continue; // TODO: handle redundant clusters

            int min_x, min_y, max_x, max_y;
            min_x = max_x = c.points[0].x;
            min_y = max_y = c.points[0].y;
            for (vec2i pt : c.points) {
                if (pt.x < min_x) min_x = pt.x;
                if (pt.x > max_x) max_x = pt.x;
                if (pt.y < min_y) min_y = pt.y;
                if (pt.y > max_y) max_y = pt.y;
            }

            vec2i starting_coord = {min_x, min_y};
            int size = std::max(max_x - min_x + 1, max_y - min_y + 1); // assuming blocks are square for now
            block_coords.push_back(starting_coord);
            block_size.push_back(size);
        }

    }
}
