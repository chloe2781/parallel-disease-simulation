// this file will manage the simulation, handling CPU GPU communication, running the phases, and managing memory
#include "config.h"
#include <stdio.h>
#include "simulation.cuh"
#include <curand_kernel.h>
#include <cstdint>
#include <cuda_runtime.h>
#include <string>
#include <atomic>
#include <iostream>

__host__ void simulation() {

        //Sim Variables
        //cell_grid[value] = ID of person in cell
        int* cell_grid_first = new int[GRID_SIZE * GRID_SIZE];
        int* cell_grid_last = new int[GRID_SIZE * GRID_SIZE];
        //next[value] = ID of next person in cell, 0 if none
        int* next = new int[(POPULATION)];
        //array of variants
        int variant_count = 0;
        int variant_cap = 64;
        Variant* variants = new Variant[variant_cap]; 

        //Person Variables, SoA style
        //x position of each person, may not be needed
        int* positions = new int[(POPULATION)];
        //variant[value] = ID of variant, negative value if not infected
        int* variant = new int[(POPULATION)];
        //time remaining until immunity expires, 0 or negative if no longer immune
        int* immunity = new int[(POPULATION)];
        // whether the person is dead, 0 or positive is alive and negative is dead
        // also used to keep track of infection time where value of alive keeps track of infection period
        int* dead = new int[(POPULATION)]; 

        //current memory footprint
        //per person: 
        //overall sim:

        //initialize on host
        printf("Initializing data\n");

        //zero out SoA
        for (int i = 0; i < POPULATION; i++) {
            positions[i] = 0;
            variant[i] = -1;
            immunity[i] = 0;
            dead[i] = 0;
        }

        //place 2 people in the sim
        positions[0] = 128 * 256 + 128;
        positions[1] = 128 * 256 + 128;

        //set up a variant
        variant_count = 1;
        Variant v {};
        v.id = 0,
        v.recovery_time = 10;
        v.mortality_rate = 0;
        v.infection_rate = 1;
        v.mutation_rate = 0;
        v.immunity_time = 10;
        
        variants[0] = v;

        //infect the first person with the variant
        variant[0] = 0;


        //show initial values
        printf("Initial Values:\n");
        for (int i = 0; i < POPULATION; i++) {
            printf("Person %d is at position (%d, %d). Infected: %d, Immunity: %d, Dead: %d\n", i, positions[i] % GRID_SIZE, positions[i] / GRID_SIZE, variant[i], immunity[i], dead[i]);
        }
        printf("==============================================================================\n");

        //set up GPU memory
        int *d_cell_grid_first = NULL;
        int *d_cell_grid_last = NULL;
        int *d_next = NULL;
        int *d_variant_count = NULL;
        int *d_variant_cap = NULL;
        Variant *d_variants = NULL;

        int *d_position = NULL;
        int *d_variant = NULL;
        int *d_immunity = NULL;
        int* d_dead = NULL;

        printf("Allocating GPU memory\n");
        cudaMalloc((void**)&d_cell_grid_first, sizeof(int) * GRID_SIZE * GRID_SIZE);
        cudaMalloc((void**)&d_cell_grid_last, sizeof(int) * GRID_SIZE * GRID_SIZE);
        cudaMalloc((void**)&d_next, sizeof(int) * GRID_SIZE * GRID_SIZE);
        cudaMalloc((void**)&d_variant_count, sizeof(int));
        cudaMalloc((void**)&d_variant_cap, sizeof(int));

        cudaMalloc((void**)&d_variants, sizeof(Variant) * variant_cap);
        cudaMalloc((void**)&d_position, sizeof(int) * (POPULATION));
        cudaMalloc((void**)&d_variant, sizeof(int) * (POPULATION));
        cudaMalloc((void**)&d_immunity, sizeof(int) * (POPULATION));
        cudaMalloc((void**)&d_dead, sizeof(int) * (POPULATION));
        cudaCheck("Error allocating GPU memory");

        printf("Copying data to GPU\n");
        cudaMemcpy(d_cell_grid_first,   cell_grid_first, sizeof(int) * GRID_SIZE * GRID_SIZE,   cudaMemcpyHostToDevice);
        cudaMemcpy(d_cell_grid_last,    cell_grid_last, sizeof(int) * GRID_SIZE * GRID_SIZE,    cudaMemcpyHostToDevice);
        cudaMemcpy(d_next, next,        sizeof(int) * GRID_SIZE * GRID_SIZE,                    cudaMemcpyHostToDevice);
        cudaMemcpy(d_variant_count,     &variant_count, sizeof(int),                            cudaMemcpyHostToDevice);
        cudaMemcpy(d_variant_cap,       &variant_cap, sizeof(int),                              cudaMemcpyHostToDevice);
        cudaMemcpy(d_variants,          variants, sizeof(Variant) * variant_cap,                cudaMemcpyHostToDevice);

        cudaMemcpy(d_position,          positions, sizeof(int) * (POPULATION),                  cudaMemcpyHostToDevice);
        cudaMemcpy(d_variant,           variant, sizeof(int) * (POPULATION),                    cudaMemcpyHostToDevice);
        cudaMemcpy(d_immunity,          immunity, sizeof(int) * (POPULATION),                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_dead, dead,        sizeof(int) * (POPULATION),                             cudaMemcpyHostToDevice);
        cudaCheck("Error copying data to GPU");

        // run the gpu code once
        for(int i = 0; i < EPOCHS; i++){
            printf("Epoch %d =====================================================================\n", i);
            printf("Launching movePeople\n");
            movePeople<<<MOVE_BLOCKS, MOVE_THREADS>>>(d_position);
            cudaDeviceSynchronize();
            printf("Launching infectPeople\n");
            infectPeople<<<INFECTION_THREADS, INFECTION_THREADS>>>(d_variants, d_position, d_variant_count, d_variant_cap, d_variant, d_immunity, d_dead);
            cudaDeviceSynchronize();
            printf("Launching killPeople\n");
            killPeople<<<KILL_BLOCKS,KILL_THREADS>>>(d_variants, d_variant, d_dead);
            cudaDeviceSynchronize();
            printf("Launching tick\n");
            tick<<<TICK_BLOCKS,TICK_THREADS>>>(d_variants, d_immunity, d_variant, d_dead);
            cudaDeviceSynchronize();
            cudaCheck("Error running kernels");
            for (int i = 0; i < POPULATION; i++) {
            printf("Person %d is at position (%d, %d). Infected: %d, Immunity: %d, Dead: %d\n", i, positions[i] % GRID_SIZE, positions[i] / GRID_SIZE, variant[i], immunity[i], dead[i]);
            }
        }
        printf("Epochs complete\n");

        printf("Copying data back to CPU\n");
        cudaMemcpy(cell_grid_first,     d_cell_grid_first, sizeof(int) * GRID_SIZE * GRID_SIZE, cudaMemcpyDeviceToHost);
        cudaMemcpy(cell_grid_last,      d_cell_grid_last, sizeof(int) * GRID_SIZE * GRID_SIZE,  cudaMemcpyDeviceToHost);
        cudaMemcpy(next, d_next,        sizeof(int) * GRID_SIZE * GRID_SIZE,                    cudaMemcpyDeviceToHost); 
        cudaMemcpy(&variant_cap,        d_variant_cap, sizeof(int),                             cudaMemcpyDeviceToHost);
        cudaMemcpy(variants,            d_variants, sizeof(Variant) * variant_cap,              cudaMemcpyDeviceToHost);

        cudaMemcpy(positions,           d_position, sizeof(int) * (POPULATION),                 cudaMemcpyDeviceToHost);
        cudaMemcpy(variant,             d_variant, sizeof(int) * (POPULATION),                  cudaMemcpyDeviceToHost);
        cudaMemcpy(immunity,            d_immunity, sizeof(int) * (POPULATION),                 cudaMemcpyDeviceToHost);
        cudaMemcpy(dead,                d_dead, sizeof(int) * (POPULATION),                     cudaMemcpyDeviceToHost);
        cudaCheck("Error copying data back to CPU");

        //print out the positions of all the people
        printf("====================================\n");
        printf("Final Values:\n");
        for (int i = 0; i < POPULATION; i++) {
            printf("Person %d is at position (%d, %d). Infected: %d, Immunity: %d, Dead: %d\n", i, positions[i] % GRID_SIZE, positions[i] / GRID_SIZE, variant[i], immunity[i], dead[i]);
        }

        printf("Freeing GPU memory\n");
        cudaFree(d_cell_grid_first);
        cudaFree(d_cell_grid_last);
        cudaFree(d_next);
        cudaFree(d_variant_count);
        cudaFree(d_variant_cap);
        cudaFree(d_variants);

        cudaFree(d_position);
        cudaFree(d_variant);
        cudaFree(d_immunity);
        cudaFree(d_dead);
        cudaCheck("Error freeing GPU memory");
}

//shortens the cuda error checking code to one line whereever it is called
void cudaCheck(const std::string &message){
    if (cudaGetLastError() != cudaSuccess){
        std::cout << message << std::endl;
        return;
    }
}
//TODO: replace two kernel calls with one, and just barrier sync
// move people randomly around the grid
__global__ void movePeople(int *positions) {
    //move every person in the grid a random amount in each direction
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    if(tid > POPULATION){
        return;
    }
    //iterate over people, move them
    for(int i = tid; i < POPULATION; i += stride){
        int position = positions[i];

        //get movements
        int rand_x = randomMovement();
        int rand_y = randomMovement();

        //add the movements back into position
        int x = position % GRID_SIZE;
        int y = position / GRID_SIZE;

        //move, wrap, and stay positive
        x = (x + rand_x + GRID_SIZE) % GRID_SIZE;
        y = (y + rand_y + GRID_SIZE) % GRID_SIZE;

        position = y * GRID_SIZE + x;
        positions[i] = position;
    }
}

//this function will update the cell_grids and next arrays
__global__ void updateCellOccupancy(int *cell_grid_first, int *cell_grid_last, int *positions, int *next){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    if (tid > POPULATION){
        return;
    }
    //iterate over people, updating the cell grid for each person
    for(int i = tid; i < POPULATION; i += stride){
        //get the cell they should be in
        int cell = positions[i];

        //check if that cell is occupied, if not, occupy it (use atomics)
        if(atomicCAS(&cell_grid_first[cell], -1, i) == -1){
            //was not occupied, cell_grid_first[cell] is now i
            cell_grid_last[cell] = i;
        } else {
            while(1){
                //try to CAS the next[last] to our index
                if(atomicCAS(&next[cell_grid_last[cell]], -1, i) == -1){
                    //it worked, which means next[last] is now i
                    //now update cell_grid_last to be i
                    //and set next[i] to -1
                    //other threads will fail to CAS, so no other thread will touch these values
                    cell_grid_last[cell] = i;
                    next[i] = -1;
                    break;
                }
            }
        }
    }
}

//this function will update the infection status based on people sharing cells
__global__ void infectPeople(Variant* variants, int* positions, int *variant_count, int *variant_cap, int* variant, int* immunity, int* dead) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid > POPULATION){
        //nothing to do
        return;
    }

    if (dead[tid] < 0) {
        // dead, cannot infect anyone
        return;
    }

    if(variant[tid] < 0){
        //uninfected, cannot infect anyone
        return;
    }
    //get the variant of the person
    Variant our_variant = variants[variant[tid]];
    //get the position of the person
    int our_position = positions[tid];
    //iterate over everyone, comparing them to the tid'th person to see if they are infected
    int stride = blockDim.x * gridDim.x;
    for(int i = 0; i < POPULATION; i += stride)
    {

        //check if the person is in the same cell as us
        if(positions[i] == our_position){
            //check if infection occurs
            if(immunity[i] < 1 && randomFloat() < our_variant.infection_rate){
                //infection occurs, check for mutation
                if(randomFloat() < our_variant.mutation_rate){
                    //mutation occurs, create a new variant
                    int new_variant = createVariant(variants, variant_count, variant_cap, variant[tid]);
                    variant[i] = new_variant;

                    // Set recovery time for person
                    dead[i] = variants[new_variant].recovery_time;
                } else {
                    //congrats you caught the same variant
                    variant[i] = variant[tid];

                    // Set recovery time for person
                    dead[i] = variants[variant[tid]].recovery_time;
                }
            } //lucky them
        }
    }
}

//device function to create a variant
__device__ int createVariant(Variant *variants, int *variant_count, int *variant_cap, int source_variant) {
    //check if we need to reallocate the variants array    
    if(*variant_count == *variant_cap){
        *variant_cap *= 2;
        //realloc not allowed, but memcpy and malloc are
        Variant *new_variants = (Variant*)malloc(sizeof(Variant) * (*variant_cap));
        //copy the old data over
        memcpy(new_variants, variants, sizeof(Variant) * (*variant_cap));
        //free the old data
        free(variants);
        //set the variants pointer to the new data
        variants = new_variants;
    }

    float mutation_range = variants[source_variant].mutation_rate;
    //create the new variant, based on the index of the source variant
    Variant new_variant = variants[source_variant];
    //change the parameters of the new variant by up to MUTATION_RANGE
    new_variant.mortality_rate *= (1 + randomFloat() * mutation_range * 2 - mutation_range);
    new_variant.infection_rate *= (1 + randomFloat() * mutation_range * 2 - mutation_range);
    new_variant.mutation_rate *= (1 + randomFloat() * mutation_range * 2 - mutation_range);

    float recovery_change = randomFloat() * mutation_range * 2 - mutation_range;
    float immunity_change = randomFloat() * mutation_range * 2 - mutation_range;
    //floor a negative recovery change, ceil a positive recovery change
    new_variant.recovery_time += (recovery_change < 0 ? floor(recovery_change) : ceil(recovery_change));
    new_variant.immunity_time += (immunity_change < 0 ? floor(immunity_change) : ceil(immunity_change));
    //prevent negative recovery time or immunity time
    new_variant.recovery_time = max(new_variant.recovery_time, 1);
    new_variant.immunity_time = max(new_variant.immunity_time, 1);
    //put this variant in the variants array, increment the variant count, and return the index of the new variant
    //do this atomically, multiple threads may be trying to create variants at the same time
    int new_variant_index = atomicAdd(variant_count, 1);
    variants[new_variant_index] = new_variant;
    return new_variant_index;
}

// Kills people based on variant mortality rate
__global__ void killPeople(Variant* variants, int* variant, int* dead) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid > POPULATION){
        return;
    }

    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < POPULATION; i += stride) {
        if (variant[i] < 0){
            // Uninfected, cannot kill
            return;
        }

        // Get the variant of the person
        Variant our_variant = variants[variant[i]];

        // Roll die to determine if killed off
        if (randomFloat() < our_variant.mortality_rate) { 
            dead[i] = -1; // Mark as dead
        }
    }
}

// Ticks immunity and infection times for individuals
__global__ void tick(Variant* variants, int* immunity, int* variant, int* dead) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid > POPULATION){
        return;
    }

    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < POPULATION; i += stride) {
        if (dead[i] < 0) {
            // dead, cannot tick
            return;
        }

        // Tick immunity time
        immunity[i] = max(immunity[i]--, -1);

        //currently infected but survived
        if (dead[i] > 0) {
            // Tick infection time
            dead[i] = max(dead[i]--, 0);
            return;
        }

        //either recovering, or were never infected
        //check if variant is > 0 to see if infected
        if (variant[i] > 0) {
            // Gain immunity
            immunity[i] = variants[variant[i]].immunity_time;
            // Mark as uninfected
            variant[i] = 0;
        }
    }
}



// device function to make a random number
__device__ int randomMovement() {
    curandState_t state;
    curand_init(RANDOM_SEED, clock(), 0, &state);
    //make it between -RANGE and RANGE
    return curand_uniform(&state) * (MOVE_RANGE * 2) - MOVE_RANGE;
}

__device__ float randomFloat() {
    curandState_t state;
    curand_init(RANDOM_SEED, clock(), 0, &state);
    return curand_uniform(&state);
}

//TODO: test inline
__host__ __device__ int coordToIndex(int x, int y) {
    return x * GRID_SIZE + y;
}