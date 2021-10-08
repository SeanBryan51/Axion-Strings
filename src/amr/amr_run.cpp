
/*
def initialise_coarse_grid(phi1, phi2, phidot1, phidot2, dx_coarsest):
    '''
    Input is a high resolution snapshot of the standard non-amr simulation. The function
    should return a grid hierarchy with the coarsest level having a resolution defined by
    dx_coarsest and the finest level having a resolution equal to the given snapshot.
    We basically coarsen areas of the high resolution which are not needed in the given
    snapshot whilst keeping the high resolution around areas of interest (strings) which
    can be futher refined. This should hopefully decrease the number of total gridpoints
    and speed up the simulation.

    @phi1, @phi2, @phidot1, @phidot2: field values on a regular mesh.
    @dx_coarsest: must be a factor of 2 multiple of dx (of the high resolution snapshot) i.e.
    dx_coarsest = 2^k * dx for some integer k.

      (coarsest) level 0: grid separation = dx_coarsest / 2^0
                 level 1: grid separation = dx_coarsest / 2^1
                 level 2: grid separation = dx_coarsest / 2^2
                  ...
      (finest)   level k: grid separation = dx_coarsest / 2^k = dx
                    where k = log2(dx_coarsest / dx)
    '''

    grid_hierarchy = {
        'levels': [],
    }

    coarse_fine_ratio = dx_coarsest // dx
    max_levels = int(np.log2(coarse_fine_ratio) + 1)

    # First define the coarsest level:
    grid_hierarchy['levels'].append({
        'level_id': 0,
        'blocks': [{
            'block_id': 0,
            'x_range': (0, L),
            'y_range': (0, L),
            'phi1':    np.copy(phi1[::coarse_fine_ratio,::coarse_fine_ratio]),
            'phi2':    np.copy(phi2[::coarse_fine_ratio,::coarse_fine_ratio]),
            'phidot1': np.copy(phidot1[::coarse_fine_ratio,::coarse_fine_ratio]),
            'phidot2': np.copy(phidot2[::coarse_fine_ratio,::coarse_fine_ratio]),
        }]
    })

    add_stencil_buffer(grid_hierarchy, 0, 0)

    # Note for any level k phi1[::coarse_fine_ratio // 2**k, ::coarse_fine_ratio // 2**k]

    # Traverse fields over coarsest spatial step and see if 
    # we need to refine.
    for level in range(max_levels):

        # Traverse over fields every (coarse_fine_ratio / 2**k) grid point calculating
        # the gradients:
        for block in grid_hierarchy['levels'][level]:
            width, height = block['phi1'].shape
            gradient = np.zeros(shape=(width, height))
            # if StencilOrder == 2:
            #     for i in range(width):
            #         for j in range(height):
            #             dphi[i,j] = ((-phi[np.mod(i+2,N),j]+8*phi[np.mod(i+1,N),j]-8*phi[i-1,j] + phi[i-2,j])\
            #                 + (-phi[i,np.mod(j+2,N)] + 8*phi[i,np.mod(j+1,N)] -8*phi[i,j-1] + phi[i,j-2]))/(12*dx) 

        # If we require refinement, move onto the next level 'level 1'
        # and repeat.

        # Continue until we no longer need to refine, or if we reach 
        # the finest level.


def add_stencil_buffer(grid_hierarchy, level_id, block_id):
    # Stencil order is hard coded to 2.
    pass
*/

#include "amr_internal.hpp"
#include "amr_interface.hpp"

void regrid(std::vector<level_data> hierarchy);
void integrate_level(std::vector<level_data> hierarchy, int level);
void reflux();
void average_down();

/*
 * Implementation of the Berger-Collela time-stepping algorithm:
 * Function takes in a grid hierarchy and evolves the level
 * specified by level by one time step.
 */
void evolve_level(std::vector<level_data> hierarchy, int level) {

    integrate_level(hierarchy, level); // integrate current level without knowing the fine level data

    if (1) {
        // Should refine grid:

        // Called at a specific timestep (not everytime as it is computationally expensive?).
        // Solution: integrate_level() should check the refinement condition after integrating fields.
        
        // Buffers allow features to move within a patch between refinement periods. Can also gamble by using
        // a very accurate refinement criterion so we can be sure we don't lose resolution of features in between 
        // refinement periods?
        // Solution: for now just check after every integration.
        regrid(hierarchy);
    }

    if (level + 1 < hierarchy.size()) {
        // There exists a finer level:

        // Perform subcycling:
        // For a factor of 2 refinement between levels, we need to evolve the level twice:
        evolve_level(hierarchy, level + 1);
        evolve_level(hierarchy, level + 1);

        reflux();
        average_down(); // set covered coarse cells to be the average of fine
    }

}

void run_amr() {

    std::vector<level_data> hierarchy;
    level_data root_level;
    root_level.size = get_length();
    root_level.b_index = {0}; // the root grid starts at index 0
    root_level.b_size = {parameters.N};

    set_physics_variables();

    // Sanity check on input parameters:
    assert(parameters.N % 2 == 0); // Number of grid points should always be some power of 2.
    assert(parameters.NDIMS == 2 || parameters.NDIMS == 3);

    root_level.phi1      = (data_t *) calloc(root_level.size, sizeof(data_t));
    root_level.phi2      = (data_t *) calloc(root_level.size, sizeof(data_t));
    root_level.phidot1   = (data_t *) calloc(root_level.size, sizeof(data_t));
    root_level.phidot2   = (data_t *) calloc(root_level.size, sizeof(data_t));
    root_level.ker1_curr = (data_t *) calloc(root_level.size, sizeof(data_t));
    root_level.ker2_curr = (data_t *) calloc(root_level.size, sizeof(data_t));
    root_level.ker1_next = (data_t *) calloc(root_level.size, sizeof(data_t));
    root_level.ker2_next = (data_t *) calloc(root_level.size, sizeof(data_t));
    root_level.axion = NULL;
    root_level.saxion = NULL;

    assert(root_level.phi1 != NULL && root_level.phi2 != NULL && root_level.phidot1 != NULL && root_level.phidot2 != NULL);
    assert(root_level.ker1_curr != NULL && root_level.ker2_curr != NULL && root_level.ker1_next != NULL && root_level.ker2_next != NULL);

    // Set initial field values:
    gaussian_thermal(root_level.phi1, root_level.phi2, root_level.phidot1, root_level.phidot2);

    hierarchy = {root_level};

    int final_step = round(light_crossing_time / parameters.time_step) - round(parameters.space_step / parameters.time_step) + 1;
    for (int tstep = 0; tstep < final_step; tstep++) {
        
        evolve_level(hierarchy, 0);
    }

    // Clean up memory:
    for (level_data level : hierarchy) {
        free(level.phi1);
        free(level.phi2);
        free(level.phidot1);
        free(level.phidot2);
        free(level.ker1_curr);
        free(level.ker2_curr);
        free(level.ker1_next);
        free(level.ker2_next);
        free(level.axion);
        free(level.saxion);
    }
}
