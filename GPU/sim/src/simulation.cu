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
#include <fstream>
#include <cstdlib> // for rand() and srand()
#include <ctime>   // for time()

//THIS CODE SPONSORED BY SHADOW WIZARD MONEY GANG
//"We love casting spells"


__host__ void simulation() {

        //seed host RNG
        srand(static_cast<unsigned>(time(0)));

        //Sim Variables
        int variant_count = 0;
        int variant_cap = 256;
        Variant* variants = new Variant[variant_cap]; 
        //container for snapshots of the sim
        Snapshot *snapshots = new Snapshot[EPOCHS];

        //Person Variables (SoA)
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
        sim_bytes_used += (sizeof(Variant) * variant_cap) + 8 + (sizeof(Snapshot) * EPOCHS);
        people_bytes_used += (sizeof(int) * (POPULATION) * 4) + (sizeof(bool) * (POPULATION)); //positions + variant + immunity + dead

        double sim_bytes_used_GB = sim_bytes_used / 1000000000.0;
        double people_bytes_used_GB = people_bytes_used / 1000000000.0;
        printf("Sim memory footprint: %f bytes\n", sim_bytes_used_GB);
        printf("People memory footprint: %f bytes\n", people_bytes_used_GB);

        printf("Initializing data\n");
        //zero out SoA
        for (int i = 0; i < POPULATION; i++) {
            positions[i] = 0;
            variant[i] = -1;
            immunity[i] = -1;
            dead[i] = 0;
            fresh[i] = false;
        }

        //set up the first variant
        variant_count = 1;
        Variant v {};
        v.recovery_time = 14;
        v.mortality_rate = 0.015f;
        v.infection_rate = 0.3f;
        v.mutation_rate = 0.001f;
        v.immunity_time = 90;
        variants[0] = v;

        //place people randomly
        for (int i = 0; i < POPULATION; i++) {
            positions[i] = rand() % (GRID_SIZE * GRID_SIZE);
        }

        //infect the first people
        int remaining_infected = min(STARTING_INFECTED, POPULATION);
        while (remaining_infected > 0) {
            int person = rand() % (POPULATION);
            if (variant[person] == -1) {
                variant[person] = 0;
                dead[person] = v.recovery_time;
                remaining_infected--;
            }
        }

        //set up GPU memory
        int *d_variant_count = NULL;
        int *d_variant_cap = NULL;
        Variant *d_variants = NULL;
        Snapshot *d_snapshots = NULL;

        int *d_position = NULL;
        int *d_variant = NULL;
        int *d_immunity = NULL;
        int* d_dead = NULL;
        bool* d_fresh = NULL;

{        printf("Allocating GPU memory\n");
        cudaMalloc((void**)&d_variant_count, sizeof(int));
        cudaMalloc((void**)&d_variant_cap, sizeof(int));
        cudaMalloc((void**)&d_snapshots, sizeof(Snapshot) * EPOCHS);

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
        cudaMemcpy(d_snapshots,         snapshots, sizeof(Snapshot) * EPOCHS,                   cudaMemcpyHostToDevice);

        cudaMemcpy(d_position,          positions, sizeof(int) * (POPULATION),                  cudaMemcpyHostToDevice);
        cudaMemcpy(d_variant,           variant, sizeof(int) * (POPULATION),                    cudaMemcpyHostToDevice);
        cudaMemcpy(d_immunity,          immunity, sizeof(int) * (POPULATION),                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_dead, dead,        sizeof(int) * (POPULATION),                             cudaMemcpyHostToDevice);
        cudaMemcpy(d_fresh, fresh,      sizeof(bool) * (POPULATION),                            cudaMemcpyHostToDevice);
        cudaCheck("Error copying data to GPU");
}
        
        // Run the sim
        //you can use gpuPeek and showVariants to see the state of the sim at any point
        printf("==============================================================================\n");
        printf("Running simulation for %d epoch(s)\n", EPOCHS);
        for(int i = 0; i < EPOCHS; i++){
            printf("Epoch %d =====================================================================\n", i + 1);
            printf("running movePeople()\n");
            movePeople<<<MOVE_BLOCKS, MOVE_THREADS>>>(d_position, i);
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
            // gpuPeek<<<1, 1>>>(d_position, d_variant, d_immunity, d_dead, d_fresh);
            // cudaDeviceSynchronize();
            // cudaCheck("gpuPeek error");
            printf("running takeSnapshot()\n");
            takeSnapshot<<<1, SNAPSHOT_THREADS>>>(d_snapshots, i, d_immunity, d_dead, d_fresh, d_variant_count);
            cudaDeviceSynchronize();
            cudaCheck("takeSnapshot error");
            //zero the fresh array for next epoch
            printf("zeroing fresh array\n");
            cudaMemset(d_fresh, 0, POPULATION*sizeof(bool));
            cudaCheck("zeroing fresh array error");
            cudaDeviceSynchronize();
        }

        printf("==============================================================================\n");
        printf("Epochs complete\n");

{       printf("Copying data back to CPU\n");
        cudaMemcpy(&variant_cap,        d_variant_cap, sizeof(int),                             cudaMemcpyDeviceToHost);
        cudaMemcpy(variants,            d_variants, sizeof(Variant) * variant_cap,              cudaMemcpyDeviceToHost);
        cudaMemcpy(snapshots,           d_snapshots, sizeof(Snapshot) * EPOCHS,                 cudaMemcpyDeviceToHost);

        cudaMemcpy(positions,           d_position, sizeof(int) * (POPULATION),                 cudaMemcpyDeviceToHost);
        cudaMemcpy(variant,             d_variant, sizeof(int) * (POPULATION),                  cudaMemcpyDeviceToHost);
        cudaMemcpy(immunity,            d_immunity, sizeof(int) * (POPULATION),                 cudaMemcpyDeviceToHost);
        cudaMemcpy(dead,                d_dead, sizeof(int) * (POPULATION),                     cudaMemcpyDeviceToHost);
        cudaMemcpy(fresh,               d_fresh, sizeof(bool) * (POPULATION),                   cudaMemcpyDeviceToHost);
        cudaCheck("Error copying data back to CPU");

        printf("Freeing GPU memory\n");
        cudaFree(d_variant_count);
        cudaFree(d_variant_cap);
        cudaFree(d_variants);
        cudaFree(d_snapshots);

        cudaFree(d_position);
        cudaFree(d_variant);
        cudaFree(d_immunity);
        cudaFree(d_dead);
        cudaFree(d_fresh);
        cudaCheck("Error freeing GPU memory");}

        printf("Simulation complete\n");

        printf("==============================================================================\n");
        printf("Preliminary results\n");
        //print the details of the last snapshot
        //structure of the snapshot is:
        //     int epoch;
        //     int variant_count;
        //     int alive;
        //     int dead;
        //     int infected;
        //     int uninfected;
        //     int immune;
        //     int fresh;
        printf("Snapshot of epoch %d\n", snapshots[EPOCHS - 1].epoch);
        printf("Variant count: %d\n", snapshots[EPOCHS - 1].variant_count);
        printf("Alive: %d\n", snapshots[EPOCHS - 1].alive);
        printf("Dead: %d\n", snapshots[EPOCHS - 1].dead);
        printf("Infected: %d\n", snapshots[EPOCHS - 1].infected);
        printf("Uninfected: %d\n", snapshots[EPOCHS - 1].uninfected);
        printf("Immune: %d\n", snapshots[EPOCHS - 1].immune);
        printf("Infections this epoch: %d\n", snapshots[EPOCHS - 1].fresh);

        outputSnapshots(snapshots);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// UTILITY CODE ////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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

//take a snapshot of the current state of the simulation and store it in the snapshots array
__global__ void takeSnapshot(Snapshot *snapshots, int epoch, int *immunity, int *dead, bool *fresh, int * variant_count){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

        // Allocate shared memory for the block
    __shared__ int num_alive_shared[SNAPSHOT_THREADS];
    __shared__ int num_dead_shared[SNAPSHOT_THREADS];
    __shared__ int num_infected_shared[SNAPSHOT_THREADS];
    __shared__ int num_uninfected_shared[SNAPSHOT_THREADS];
    __shared__ int num_immune_shared[SNAPSHOT_THREADS];
    __shared__ int num_fresh_infected_shared[SNAPSHOT_THREADS];

    // Initialize shared memory
    num_alive_shared[threadIdx.x] = 0;
    num_dead_shared[threadIdx.x] = 0;
    num_infected_shared[threadIdx.x] = 0;
    num_uninfected_shared[threadIdx.x] = 0;
    num_immune_shared[threadIdx.x] = 0;
    num_fresh_infected_shared[threadIdx.x] = 0;

    __syncthreads();

    //calculate the sums
    for (int i = tid; i < POPULATION; i += stride){
        int person_status = dead[i];
        int immune_status = immunity[i];

        num_alive_shared[threadIdx.x] += (person_status >= 0);
        num_dead_shared[threadIdx.x] += (person_status == -1);
        num_infected_shared[threadIdx.x] += (person_status > 0);
        num_uninfected_shared[threadIdx.x] += (person_status == 0);
        num_immune_shared[threadIdx.x] += (immune_status > 0);
        num_fresh_infected_shared[threadIdx.x] += (fresh[i] == true);
    }

    //parallel reduction
    for (int i = blockDim.x / 2; i > 0; i >>= 1){
        if (threadIdx.x < i){
            num_alive_shared[threadIdx.x] += num_alive_shared[threadIdx.x + i];
            num_dead_shared[threadIdx.x] += num_dead_shared[threadIdx.x + i];
            num_infected_shared[threadIdx.x] += num_infected_shared[threadIdx.x + i];
            num_uninfected_shared[threadIdx.x] += num_uninfected_shared[threadIdx.x + i];
            num_immune_shared[threadIdx.x] += num_immune_shared[threadIdx.x + i];
            num_fresh_infected_shared[threadIdx.x] += num_fresh_infected_shared[threadIdx.x + i];
        }
        __syncthreads();
    }

    //write to global memory
    if (threadIdx.x == 0){
        snapshots[epoch].alive = num_alive_shared[0];
        snapshots[epoch].dead = num_dead_shared[0];
        snapshots[epoch].infected = num_infected_shared[0];
        snapshots[epoch].uninfected = num_uninfected_shared[0];
        snapshots[epoch].immune = num_immune_shared[0];
        snapshots[epoch].fresh = num_fresh_infected_shared[0];
        //update variant count and epoch
        snapshots[epoch].variant_count = *variant_count;
        snapshots[epoch].epoch = epoch;
    }
}

//put the snapshots into a file
void outputSnapshots(Snapshot *snapshots){
    //each snapshot is 1 line
    std::ofstream output_file;
    output_file.open("snapshots.txt");
    //print out the names of the columns
    output_file << "ep al de in un im va fr" << std::endl;
    for (int i = 0; i < EPOCHS; i++){
        output_file << snapshots[i].epoch << "  " << snapshots[i].alive << "  " << snapshots[i].dead << "  " << snapshots[i].infected << "  " << snapshots[i].uninfected << "  " << snapshots[i].immune << "  " << snapshots[i].variant_count << "  " << snapshots[i].fresh << std::endl;
    }
    output_file.close();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// SIM CODE ////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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

        //printf("Person %d is moving by (%d, %d)\n", i, rand_x, rand_y);

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
        //printf("Unused thread\n");
        return;
    }

    //outer loop iterates over infection sources
    for (int src = tid; src < POPULATION; src += stride) {
        Variant src_variant = variants[variant[tid]];
        int src_pos = positions[src];
        //dead do not infect
        if(dead[src] < 0){
            //printf("T%d - Ignore %d, dead\n", tid, src);
            continue;
        }
        //uninfected do not infect
        if(variant[src] < 0){
            //printf("T%d - Ignore %d, uninfected\n", tid, src);
            continue;
        }
        //fresh infections do not infect
        if(fresh[src]){
            //printf("T%d - Ignore %d, fresh\n", tid, src);
            continue;
        }
        //inner loop iterates over potential victims
        for (int dst_offset = 0; dst_offset < POPULATION; dst_offset++) {
            int dst = (src + tid + dst_offset) % POPULATION;
            //ignore immune
            if(immunity[dst] > 0){
                //printf("T%d - %d Ignore %d, immune\n", tid, src, dst);
                continue;
            }
            //ignore already infected
            if(variant[dst] >= 0){
                //printf("T%d - %d Ignore %d, already infected\n", tid, src, dst);
                continue;
            }
            //ignore dead
            if(dead[dst] < 0){
                //printf("T%d - %d Ignore %d, dead\n", tid, src, dst);
                continue;
            }
            //ignore self
            if(src == dst){
                //printf("T%d - %d Ignore %d, self\n", tid, src, dst);
                continue;
            }
            int dst_pos = positions[dst];
            //printf("T%d - Checking: %d and %d\n", tid, src, dst);
            //check if cell shared
            if(src_pos == dst_pos){
                //printf("T%d - Pos Match: %d to %d\n", tid, src, dst);
                //check for infection
                if(randomFloat(dst) < src_variant.infection_rate){
                    //check for mutation
                    if(randomFloat(dst) < src_variant.mutation_rate){
                        //atomicCAS the fresh infection to true
                        if(atomicCAS(&fresh[dst], 0, 1) == 0){
                            //give new variant to dst person
                            //printf("T%d - Mutation: P%d to P%d\n", tid, src, dst);
                            int dst_variant = createVariant(variants, variant_count, variant_cap, variant[src]);
                            variant[dst] = dst_variant;
                            dead[dst] = variants[dst_variant].recovery_time;
                        } else {
                            //printf("T%d - %d Ignore %d, contended\n", tid, src, dst);
                        } //else, someone else got there first
                    } else {
                        //atomicCAS the fresh infection to true
                        if(atomicCAS(&fresh[dst], 0, 1) == 0){
                            //give same variant to dst person
                            //printf("T%d - Infection: P%d to P%d\n", tid, src, dst);
                            variant[dst] = variant[src];
                            dead[dst] = variants[variant[src]].recovery_time;
                        } else {
                            //printf("T%d - %d Ignore %d, contended\n", tid, src, dst);
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
        //printf("Resizing variants\n");
        *variant_cap *= 2;
        //allocate new, copy memory, free old, point to new
        Variant *new_variants = (Variant*)malloc(sizeof(Variant) * (*variant_cap));
        memcpy(new_variants, variants, sizeof(Variant) * (*variant_cap));
        free(variants);
        variants = new_variants;
    }

    //create the new variant, copy the old variant
    Variant new_variant = variants[source_variant];
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
            //printf("T%d - Recovered: P%d\n", tid, i);
            // Gain immunity
            immunity[i] = variants[variant[i]].immunity_time;
            // Mark as uninfected
            variant[i] = -1;
        }
    }
}


//TODO: support infection range > 1

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// RNG CODE ////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// get a random cell, invariant under grid size, uses config value
int getRandomCell() {
    int maxValue = GRID_SIZE * GRID_SIZE - 1;
    int randomNumber = rand() % (maxValue + 1);
    return randomNumber;
}

// get a random person's index, invariant under population size, uses config value
int getRandomPerson(){
    int maxValue = POPULATION - 1;
    int randomNumber = rand() % (maxValue + 1);
    return randomNumber;
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

