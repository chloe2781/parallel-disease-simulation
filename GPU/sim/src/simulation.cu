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

//THIS CODE SPONSORED BY SHADOW WIZARD MONEY GANG
//"We love casting spells"

__host__ void simulation() {

        //Sim Variables
        int variant_count = 0;
        int variant_cap = 256;
        Variant* variants = new Variant[variant_cap]; 

        //Person Variables, SoA
        //position is stored as a single int, with the x and y coordinates packed into it
        int* positions = new int[(POPULATION)];
        // variant[value] = ID of variant, negative value if not infected
        int* variant = new int[(POPULATION)];
        //time remaining until immunity expires, negative if no longer immune
        int* immunity = new int[(POPULATION)];
        // -1 is dead, 0 is uninfected, positive is time until recovery
        int* dead = new int[(POPULATION)]; 
        // fresh is true if the person was infected this epoch (keeps them from infecting others in the same epoch)
        bool* fresh = new bool[(POPULATION)];

        int sim_bytes_used = 0;
        int people_bytes_used = 0;
        sim_bytes_used += sizeof(Variant) * variant_cap + 8; //variants + variant_count + variant_cap
        people_bytes_used += sizeof(int) * (POPULATION) * 4; //positions + variant + immunity + dead
        people_bytes_used += sizeof(bool) * (POPULATION); //fresh

        double sim_bytes_used_GB = sim_bytes_used / pow(1024.0, 3);
        double people_bytes_used_GB = people_bytes_used / pow(1024.0, 3);
        printf("Sim memory footprint: %f bytes\n", sim_bytes_used_GB);
        printf("People memory footprint: %f bytes\n", people_bytes_used_GB);

        //initialize on host
        printf("Initializing data\n");

        //zero out SoA
        for (int i = 0; i < POPULATION; i++) {
            positions[i] = 0;
            variant[i] = -1;
            immunity[i] = -1;
            dead[i] = 0;
            fresh[i] = false;
        }

        //place 2 people in the sim together
        positions[0] = 128 * 256 + 128;
        positions[1] = 128 * 256 + 128;

        //set up the first variant
        variant_count = 1;
        Variant v {};
        v.id = 0,
        v.recovery_time = 14;
        v.mortality_rate = 0.015;
        v.infection_rate = 0.3;
        v.mutation_rate = 0.001;
        v.immunity_time = 90;
        
        variants[0] = v;

        //infect the first person with the variant
        variant[0] = 0;
        dead[0] = v.recovery_time;

        //infect someone else with the variant
        variant[2] = 0;
        dead[2] = v.recovery_time;

        //show initial values
        printf("Initial Values:\n");
        for (int i = 0; i < POPULATION; i++) {
            printf("Person %d is at position (%d, %d). Fresh: %d Variant: %d, Immunity: %d, Dead: %d\n", i, positions[i] % GRID_SIZE, positions[i] / GRID_SIZE, fresh[i], variant[i], immunity[i], dead[i]);
        }

        //set up GPU memory
        int *d_variant_count = NULL;
        int *d_variant_cap = NULL;
        Variant *d_variants = NULL;

        int *d_position = NULL;
        int *d_variant = NULL;
        int *d_immunity = NULL;
        int* d_dead = NULL;
        bool* d_fresh = NULL;

        printf("Allocating GPU memory\n");
        cudaMalloc((void**)&d_variant_count, sizeof(int));
        cudaMalloc((void**)&d_variant_cap, sizeof(int));

        cudaMalloc((void**)&d_variants, sizeof(Variant) * variant_cap);
        cudaMalloc((void**)&d_position, sizeof(int) * (POPULATION));
        cudaMalloc((void**)&d_variant, sizeof(int) * (POPULATION));
        cudaMalloc((void**)&d_immunity, sizeof(int) * (POPULATION));
        cudaMalloc((void**)&d_dead, sizeof(int) * (POPULATION));
        cudaMalloc((void**)&d_fresh, sizeof(bool) * (POPULATION));
        cudaCheck("Error allocating GPU memory");

        printf("Copying data to GPU\n");
        cudaMemcpy(d_variant_count,     &variant_count, sizeof(int),                            cudaMemcpyHostToDevice);
        cudaMemcpy(d_variant_cap,       &variant_cap, sizeof(int),                              cudaMemcpyHostToDevice);
        cudaMemcpy(d_variants,          variants, sizeof(Variant) * variant_cap,                cudaMemcpyHostToDevice);

        cudaMemcpy(d_position,          positions, sizeof(int) * (POPULATION),                  cudaMemcpyHostToDevice);
        cudaMemcpy(d_variant,           variant, sizeof(int) * (POPULATION),                    cudaMemcpyHostToDevice);
        cudaMemcpy(d_immunity,          immunity, sizeof(int) * (POPULATION),                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_dead, dead,        sizeof(int) * (POPULATION),                             cudaMemcpyHostToDevice);
        cudaMemcpy(d_fresh, fresh,      sizeof(bool) * (POPULATION),                            cudaMemcpyHostToDevice);
        cudaCheck("Error copying data to GPU");

        // Run the sim
        printf("==============================================================================\n");
        printf("Running simulationfor %d epoch(s)\n", EPOCHS);
        for(int i = 0; i < EPOCHS; i++){
            printf("Epoch %d =====================================================================\n", i + 1);
            printf("running movePeople()\n");
            //movePeople<<<MOVE_BLOCKS, MOVE_THREADS>>>(d_position, i);
            cudaDeviceSynchronize();
            cudaCheck("movePeople error");
            printf("running infectPeople()\n");
            infectPeople<<<INFECTION_BLOCKS, INFECTION_THREADS>>>(d_variants, d_position, d_variant_count, d_variant_cap, d_variant, d_immunity, d_dead, d_fresh);
            cudaDeviceSynchronize();
            cudaCheck("infectPeople error");
            printf("running killPeople()\n");
            killPeople<<<KILL_BLOCKS,KILL_THREADS>>>(d_variants, d_variant, d_dead, d_fresh);
            cudaDeviceSynchronize();
            cudaCheck("killPeople error");
            printf("running tick()\n");
            tick<<<TICK_BLOCKS,TICK_THREADS>>>(d_variants, d_immunity, d_variant, d_dead, d_fresh);
            cudaDeviceSynchronize();
            cudaCheck("tick error");
            printf("running gpuPeek()\n");
            gpuPeek<<<1, 1>>>(d_position, d_variant, d_immunity, d_dead, d_fresh);
            cudaDeviceSynchronize();
            cudaCheck("gpuPeek error");
            printf("running showVariants()\n");
            showVariants<<<1, 1>>>(d_variants, d_variant_count);
            cudaDeviceSynchronize();
            cudaCheck("showVariants error");
            //zero the fresh array for next epoch
            cudaMemset(d_fresh, 0, POPULATION*sizeof(bool));
            cudaCheck("gpuPeek error");

        }
        printf("Epochs complete\n");

        printf("Copying data back to CPU\n");
        cudaMemcpy(&variant_cap,        d_variant_cap, sizeof(int),                             cudaMemcpyDeviceToHost);
        cudaMemcpy(variants,            d_variants, sizeof(Variant) * variant_cap,              cudaMemcpyDeviceToHost);

        cudaMemcpy(positions,           d_position, sizeof(int) * (POPULATION),                 cudaMemcpyDeviceToHost);
        cudaMemcpy(variant,             d_variant, sizeof(int) * (POPULATION),                  cudaMemcpyDeviceToHost);
        cudaMemcpy(immunity,            d_immunity, sizeof(int) * (POPULATION),                 cudaMemcpyDeviceToHost);
        cudaMemcpy(dead,                d_dead, sizeof(int) * (POPULATION),                     cudaMemcpyDeviceToHost);
        cudaMemcpy(fresh,               d_fresh, sizeof(bool) * (POPULATION),                   cudaMemcpyDeviceToHost);
        cudaCheck("Error copying data back to CPU");

        //print out the positions of all the people
        printf("====================================\n");
        printf("Final Values:\n");
        for (int i = 0; i < POPULATION; i++) {
            printf("Person %d is at position (%d, %d). Fresh: %d Variant: %d, Immunity: %d, Dead: %d\n", i, positions[i] % GRID_SIZE, positions[i] / GRID_SIZE, fresh[i], variant[i], immunity[i], dead[i]);
        }

        printf("Freeing GPU memory\n");
        cudaFree(d_variant_count);
        cudaFree(d_variant_cap);
        cudaFree(d_variants);

        cudaFree(d_position);
        cudaFree(d_variant);
        cudaFree(d_immunity);
        cudaFree(d_dead);
        cudaFree(d_fresh);
        cudaCheck("Error freeing GPU memory");

        printf("Simulation complete\n");
}

//print out the stats of all the variants, expected to be 1 thread, 1 block
__global__ void showVariants(Variant* variants, int * variant_count){
    printf("Current Variants =====================================================================\n");
    for (int i = 0; i < *variant_count; i++) {
        //use the Variant.toString() method
        printf("Variant %d - mort: %f, inf: %f, mut: %f, rec: %d, imm: %d\n", i, variants[i].mortality_rate, variants[i].mutation_rate, variants[i].immunity_time, variants[i].recovery_time, variants[i].immunity_time);
    }
}

//print out the stats of all the people, expected to be 1 thread, 1 block
__global__ void gpuPeek(int* positions, int* variant, int* immunity, int* dead, bool* fresh){
    //expected to be 1 thread, 1 block
    for (int i = 0; i < POPULATION; i++) {
        printf("Person %d is at position (%d, %d). Fresh: %d Variant: %d, Immunity: %d, Dead: %d\n", i, positions[i] % GRID_SIZE, positions[i] / GRID_SIZE, fresh[i], variant[i], immunity[i], dead[i]);
    }
}

//shortens the cuda error checking code to one line whereever it is called
void cudaCheck(const std::string &message){
    if (cudaGetLastError() != cudaSuccess){
        std::cout << message << std::endl;
        return;
    }
}

// move people randomly around the grid, wrapping on edges
__global__ void movePeople(int *positions, int epoch) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    if(tid > POPULATION){
        return;
    }

    for(int i = tid; i < POPULATION; i += stride){
        int position = positions[i];

        //seed with epoch and thread id
        int rand_x = randomMovement(tid, (epoch + 1));
        int rand_y = randomMovement(tid, (epoch + 1) * 2);

        printf("Person %d is moving by (%d, %d)\n", i, rand_x, rand_y);

        //retrieve x and y from position
        int x = position % GRID_SIZE;
        int y = position / GRID_SIZE;

        //move, wrap, and stay positive
        x = (x + rand_x + GRID_SIZE) % GRID_SIZE;
        y = (y + rand_y + GRID_SIZE) % GRID_SIZE;

        //repack and assign
        position = y * GRID_SIZE + x;
        positions[i] = position;
    }
}

//this function will update the infection status based on people sharing cells
__global__ void infectPeople(Variant* variants, int* positions, int *variant_count, int *variant_cap, int* variant, int* immunity, int* dead, bool* fresh) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    //handle threads > population
    if (tid >= POPULATION){
        printf("Unused thread\n");
        return;
    }

    //outer loop iterates over infection sources
    for (int src = tid; src < POPULATION; src += stride) {
        Variant src_variant = variants[variant[tid]];
        int src_pos = positions[src];
        //dead do not infect
        if(dead[src] < 0){
            printf("T%d - Ignore %d, dead\n", tid, src);
            continue;
        }
        //uninfected do not infect
        if(variant[src] < 0){
            printf("T%d - Ignore %d, uninfected\n", tid, src);
            continue;
        }
        //fresh infections do not infect
        if(fresh[src]){
            printf("T%d - Ignore %d, fresh\n", tid, src);
            continue;
        }
        //inner loop iterates over potential victims
        for (int dst_offset = 0; dst_offset < POPULATION; dst_offset++) {
            int dst = (src + tid + dst_offset) % POPULATION;
            //ignore immune
            if(immunity[dst] > 0){
                printf("T%d - %d Ignore %d, immune\n", tid, src, dst);
                continue;
            }
            //ignore already infected
            if(variant[dst] >= 0){
                printf("T%d - %d Ignore %d, already infected\n", tid, src, dst);
                continue;
            }
            //ignore dead
            if(dead[dst] < 0){
                printf("T%d - %d Ignore %d, dead\n", tid, src, dst);
                continue;
            }
            //ignore self
            if(src == dst){
                printf("T%d - %d Ignore %d, self\n", tid, src, dst);
                continue;
            }
            int dst_pos = positions[dst];
            printf("T%d - Checking: %d and %d\n", tid, src, dst);
            //check if cell shared
            if(src_pos == dst_pos){
                printf("T%d - Pos Match: %d to %d\n", tid, src, dst);
                //check for infection
                if(randomFloat(dst) < src_variant.infection_rate){
                    //check for mutation
                    if(randomFloat(dst) < src_variant.mutation_rate){
                        //atomicCAS the fresh infection to true
                        if(atomicCAS(&fresh[dst], 0, 1) == 0){
                            //give new variant to dst person
                            printf("T%d - Mutation: P%d to P%d\n", tid, src, dst);
                            int dst_variant = createVariant(variants, variant_count, variant_cap, variant[src]);
                            variant[dst] = dst_variant;
                            dead[dst] = variants[dst_variant].recovery_time;
                        } else {
                            printf("T%d - %d Ignore %d, contended\n", tid, src, dst);
                        } //else, someone else got there first
                    } else {
                        //atomicCAS the fresh infection to true
                        if(atomicCAS(&fresh[dst], 0, 1) == 0){
                            //give same variant to dst person
                            printf("T%d - Infection: P%d to P%d\n", tid, src, dst);
                            variant[dst] = variant[src];
                            dead[dst] = variants[variant[src]].recovery_time;
                        } else {
                            printf("T%d - %d Ignore %d, contended\n", tid, src, dst);
                        } //else, someone else got there first
                    }
                } //lucky them
            }
        }
    }
}

//takes a pointer to a float and overwrites it with a new float +- MUTATION_RANGE% of the original
__device__ void mutate_helper(float *original, int seed) {
    float rand_percent = 2 * randomFloat(seed) - 1;

    *original *= 1 + rand_percent * MUTATION_RANGE;
    *original = max(*original, 0.0f);
    *original = min(*original, 1.0f);
}

//takes a pointer to an int and overwrites it with a new int +- MUTATION_RANGE% of the original
__device__ void int_mutate_helper(int *original, int seed){
    //due to int rounding, small values will get "stuck" at 1 or 0 so minimum mutation is is +- 1 for ints
    float rand_percent = 2 * randomFloat(seed) - 1;
    float mutation = rand_percent * MUTATION_RANGE * (*original);

    if(abs(mutation) < 1){
        mutation = mutation > 0 ? 1 : -1;
    }
    *original += mutation;
    *original = min(*original, 100);
    *original = max(*original, 1);
    
}

//device function to create a variant
__device__ int createVariant(Variant *variants, int *variant_count, int *variant_cap, int source_variant) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    //resizing logic   
    if(*variant_count == *variant_cap){
        printf("Resizing variants\n");
        *variant_cap *= 2;
        //allocate new, copy memory, free old, point to new
        Variant *new_variants = (Variant*)malloc(sizeof(Variant) * (*variant_cap));
        memcpy(new_variants, variants, sizeof(Variant) * (*variant_cap));
        free(variants);
        variants = new_variants;
    }

    //create the new variant, copy the old variant
    Variant new_variant = variants[source_variant];
    new_variant.id = *variant_count;
    //mutate the floats
    mutate_helper(&new_variant.mortality_rate, tid + 1);
    mutate_helper(&new_variant.infection_rate, tid + 2);
    mutate_helper(&new_variant.mutation_rate, tid + 3);
    //cap the floats at 0 and 1
    int_mutate_helper(&new_variant.recovery_time, tid + 4);
    int_mutate_helper(&new_variant.immunity_time, tid + 5);
    //put this variant in the variants array, increment the variant count, and return the index of the new variant
    int new_variant_index = atomicAdd(variant_count, 1);
    variants[new_variant_index] = new_variant;
    return new_variant_index;
}

// Kills people based on variant mortality rate
__global__ void killPeople(Variant* variants, int* variant, int* dead, bool* fresh) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= POPULATION){
        return;
    }

    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < POPULATION; i += stride) {
        if (variant[i] < 0 || fresh[i]){
            // Uninfected, cannot kill
            return;
        }

        // Get the variant of the person
        Variant our_variant = variants[variant[i]];
        

        // Roll die to determine if killed off
        if (randomFloat(tid) < our_variant.mortality_rate) { 
            dead[i] = -1; // Mark as dead
        }
    }
}

// Ticks immunity and infection times for individuals
__global__ void tick(Variant* variants, int* immunity, int* variant, int* dead, bool* fresh) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= POPULATION){
        return;
    }

    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < POPULATION; i += stride) {

        if (dead[i] < 0) {
            // dead, cannot tick
            continue;
        }

        // Tick immunity time
        immunity[i] = ::max(--immunity[i], -1);

        //currently infected but survived, don't tick fresh
        if (dead[i] > 0) {
            // Tick infection time
            if(fresh[i] == false){
                dead[i] = ::max(--dead[i], 0);
            }
            continue;
        }

        //either recovering, or were never infected
        //check if variant is > 0 to see if infected
        if (variant[i] >= 0) {
            printf("T%d - Recovered: P%d\n", tid, i);
            // Gain immunity
            immunity[i] = variants[variant[i]].immunity_time;
            // Mark as uninfected
            variant[i] = -1;
        }
    }
}

// device function to make a random movement
__device__ int randomMovement(int thread_id, int offset) {
    curandState_t state;
    curand_init(RANDOM_SEED, thread_id, offset, &state);
    //make it between -RANGE and RANGE
    return curand_uniform(&state) * (MOVE_RANGE * 2) - MOVE_RANGE;
}

// device function to make a random float between 0 and 1
__device__ float randomFloat(int thread_id) {
    curandState_t state;
    curand_init(RANDOM_SEED, thread_id, 0, &state);
    float result = curand_uniform(&state);
    //printf("Random float: %f\n", result);
    return result;
}