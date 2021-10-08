#include "amr.h"
#include "interface.h"

#define TOTAL_TIME_STEPS 100

void evolve_mesh_level(Hierarchy grid_hierarchy, int level);
void regrid(Hierarchy grid_hierarchy);
void integrate_level(Hierarchy grid_hierarchy, int level);
void reflux(Hierarchy grid_hierarchy, int level);
void average_down(Hierarchy grid_hierarchy, int level);

void run_amr(float * phi1, float * phidot1, float * phi2, float * phidot2) {

    // Construct initial grid hierarchy:
    Hierarchy grid_hierarchy = construct_grid_hierarchy(phi1, phidot1, phi2, phidot2);

    float dtau = grid_hierarchy.levels[0].dtau;
    
    // TODO: ask Giovanni what he does here:
    float t_evol = 1.0f; // dtau/DeltaRatio - dtau
    for (int count = 0; count < TOTAL_TIME_STEPS; count++) {

        evolve_mesh_level(grid_hierarchy, 0);

        t_evol = t_evol + dtau;
    }

    // save_snapshot(grid_hierarchy);
}

/*
 * Implementation of the Berger-Collela time-stepping algorithm:
 * Function takes in a grid hierarchy and evolves the level
 * specified by @level by one time step.
 */
void evolve_mesh_level(Hierarchy grid_hierarchy, int level) {

    // TODO: when to refine grid hierarchy
    bool should_refine_grid = false;
    if (should_refine_grid) {
        // Called at a specific timestep (not everytime as it is computationally expensive?).
        // Buffers allow features to move within a patch between refinement periods.
        regrid(grid_hierarchy);
    }

    integrate_level(grid_hierarchy, level); // integrate current level without knowing the fine level data

    if (level < grid_hierarchy.num_levels - 1) {
        // There exists a finer level:

        // Perform subcycling:
        // For a factor of 2 refinement between levels, we need to evolve the level twice:
        evolve_mesh_level(grid_hierarchy, level + 1);
        evolve_mesh_level(grid_hierarchy, level + 1);

        reflux(grid_hierarchy, level);
        average_down(grid_hierarchy, level); // set covered coarse cells to be the average of fine
    }

}

void regrid(Hierarchy grid_hierarchy) {

}

void integrate_level(Hierarchy grid_hierarchy, int level) {

}

void reflux(Hierarchy grid_hierarchy, int level) {

}
void average_down(Hierarchy grid_hierarchy, int level) {

}
