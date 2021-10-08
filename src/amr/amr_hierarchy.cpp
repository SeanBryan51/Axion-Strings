#include <stdio.h>

#include "amr.h"


Hierarchy construct_grid_hierarchy(float *phi1, float *phidot1, float *phi2, float *phidot2) {
    // TODO
    Hierarchy ret = {0, NULL};
    return ret;
}

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