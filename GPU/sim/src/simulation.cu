// this file will manage the simulation, handling CPU GPU communication, running the phases, and managing memory
#include "config.h"
#include <stdio.h>
#include "simulation.cuh"
#include <curand_kernel.h>
#include <cstdint>
#include <cuda_runtime.h>
#include <string>
#include <atomic>
//allow use of uint8_t

__host__ void simulation() {

        //template struct for variants
        typedef struct {
        int id;                // variant number of the disease
        int recovery_time;      // days it takes to no longer be contagious
        float mortality_rate;   // percent chance on each day that a person dies
        float infection_rate;   // percent chance that a person within the infected range is infected
        int infection_range;    // distance at which a person can be infected
        float mutation_rate;    // percent chance that an infection mutates upon infection of another person
        int immunity;           // number of days until the person is no longer immune
        } Variant;

        //Grid Variables
        //cell_grid[value] = ID of person in cell
        int* cell_grid = new int[GRID_SIZE * GRID_SIZE];
        //next[value] = ID of next person in cell, 0 if none
        int* next = new int[(POPULATION)];
        //lock for each cell, prevents multiple threads from moving people into the same cell at the same time
        int * cell_locks = new int[GRID_SIZE * GRID_SIZE];
        //TODO: compress positions into a single variable
        
        //Person Variables, SoA style
        //x position of each person, may not be needed
        int* pos_x = new int[(POPULATION)];
        //y position of each person, may not be needed
        int* pos_y = new int[(POPULATION)];
        //timestep of infection, -1 if not infected
        bool* infected = new bool[(POPULATION)];
        //variant[value] = ID of variant, -1 if not infected
        int* variant = new int[(POPULATION)];
        //time remaining until immunity expires, -1 if not infected
        int* immunity = new int[(POPULATION)];

        //current memory footprint per person
        //pos_x + pos_y + infected + variant + immunity + next = 6 * 4 bytes = 24 bytes
        // 300 million people * 24 bytes = 7.2 GB
        // plus the grid, 256 * 256 * 8 bytes = 1.6 MB
        // trying to fit this into 8 GB of VRAM
       
        
        //initialize on host
        printf("Initializing data\n");
        //zero all
        for (int i = 0; i < POPULATION; i++) {
            next[i] = -1;
            pos_x[i] = 0;
            pos_y[i] = 0;
        }
        for (int i = 0; i < GRID_SIZE * GRID_SIZE; i++) {
            if(i == coordToIndex(128, 128)){
                cell_grid[i] = 0;
            } else {
                cell_grid[i] = -1;
            }
        }

        for (int i = 0; i < GRID_SIZE * GRID_SIZE; i++) {
            if(cell_grid[i] != -1){
                printf("This should print once, and only once: %d\n", cell_grid[i]);
            }
        }

        //put everyone in the middle of the grid
        for (int i = 0; i < POPULATION; i++) {
            pos_x[i] = GRID_SIZE / 2;
            pos_y[i] = GRID_SIZE / 2;
            infected[i] = 0;
            //put them in the linked list, index from 1 because 0 indicates no one is there
            next[i] = i + 1 == POPULATION ? -1 : i + 1;
        }

        printf("Initial data:\n");
        for (int i = 0; i < POPULATION; i++) {
            printf("Person %d is at (%d, %d)\n", i, pos_x[i], pos_y[i]);
        }
        for (int i=0; i < POPULATION; i++) {
            printf("next[%d] = %d\n", i, next[i]);
        }
        printf("====================================\n");

        //set up GPU memory
        int *d_cell_grid = NULL;
        int *d_pos_x = NULL;
        int *d_pos_y = NULL;
        int *d_infected = NULL;
        int *d_next = NULL;

        printf("Allocating GPU memory\n");
        cudaMalloc((void**)&d_cell_grid, sizeof(int) * GRID_SIZE * GRID_SIZE);
        cudaMalloc((void**)&d_pos_x, sizeof(int) * (POPULATION));
        cudaMalloc((void**)&d_pos_y, sizeof(int) * (POPULATION));
        cudaMalloc((void**)&d_infected, sizeof(int) * (POPULATION));
        cudaMalloc((void**)&d_next, sizeof(int) * GRID_SIZE * GRID_SIZE);
        if (cudaGetLastError() != cudaSuccess){
            printf("Error allocating GPU memory\n");
            return;
        }

        printf("Copying data to GPU\n");
        cudaMemcpy(d_cell_grid, cell_grid, sizeof(int) * GRID_SIZE * GRID_SIZE, cudaMemcpyHostToDevice);
        cudaMemcpy(d_pos_x, pos_x, sizeof(int) * (POPULATION), cudaMemcpyHostToDevice);
        cudaMemcpy(d_pos_y, pos_y, sizeof(int) * (POPULATION), cudaMemcpyHostToDevice);
        cudaMemcpy(d_infected, infected, sizeof(int) * (POPULATION), cudaMemcpyHostToDevice);
        cudaMemcpy(d_next, next, sizeof(int) * GRID_SIZE * GRID_SIZE, cudaMemcpyHostToDevice);
        if (cudaGetLastError() != cudaSuccess){
            printf("Error copying data to GPU\n");
            return;
        }

        // run the gpu code once
        printf("Launching movePeople\n");
        movePeople<<<MOVE_BLOCKS, MOVE_THREADS>>>(d_cell_grid, d_pos_x, d_pos_y, d_next, cell_locks);
        cudaDeviceSynchronize();
        printf("Launching infectPeople\n");
        infectPeople<<<INFECTION_THREADS, INFECTION_THREADS>>>(d_cell_grid, d_pos_x, d_pos_y, d_infected, d_next);
        cudaDeviceSynchronize();
        if (cudaGetLastError() != cudaSuccess){
            printf("Error running kernels\n");
            return;
        }
        printf("Kernels Complete\n");

        printf("Copying data back to CPU\n");
        cudaMemcpy(cell_grid, d_cell_grid, sizeof(int) * GRID_SIZE * GRID_SIZE, cudaMemcpyDeviceToHost);
        cudaMemcpy(pos_x, d_pos_x, sizeof(int) * (POPULATION), cudaMemcpyDeviceToHost);
        cudaMemcpy(pos_y, d_pos_y, sizeof(int) * (POPULATION), cudaMemcpyDeviceToHost);
        cudaMemcpy(infected, d_infected, sizeof(int) * (POPULATION), cudaMemcpyDeviceToHost);
        cudaMemcpy(next, d_next, sizeof(int) * GRID_SIZE * GRID_SIZE, cudaMemcpyDeviceToHost); 
        if (cudaGetLastError() != cudaSuccess){
            printf("Error copying data back to CPU\n");
            return;
        }

        //print out the positions of all the people
        printf("====================================\n");
        printf("Printing results\n");
        for (int i = 0; i < POPULATION; i++) {
            printf("Person %d is at (%d, %d)\n", i, pos_x[i], pos_y[i]);
        }

        printf("Freeing GPU memory\n");
        cudaFree(d_cell_grid);
        cudaFree(d_pos_x);
        cudaFree(d_pos_y);
        cudaFree(d_infected);
        cudaFree(d_next);
}

//TODO: replace two kernel calls with one, and just barrier sync
// move people randomly around the grid
__global__ void movePeople(int* cell_grid, int* pos_x, int* pos_y, int* next, int *lock) {
    //each thread gets some cells, figure out which cell you are
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid >= POPULATION){
        printf("Extra");
        return;
    }
    printf("I am thread %d\n", tid);

    __syncthreads();

    //use the threads to check all the cells without overlap
    for(int i = tid; i < GRID_SIZE * GRID_SIZE; i += MOVE_BLOCKS * MOVE_THREADS){

        //get the first person in the cell
        int person = cell_grid[i];
        while(person != -1){
            int next_person = next[person];
            //move all the people in this cell, wrap around the grid
            int new_x = abs((pos_x[person] + generateCuRand()) % GRID_SIZE);
            int new_y = abs((pos_y[person] + generateCuRand()) % GRID_SIZE);
            printf("Thread<%d>: Person %d was at (%d, %d), now at (%d, %d)\n", i, person, pos_x[person], pos_y[person], new_x, new_y);
            pos_x[person] = new_x;
            pos_y[person] = new_y;
            int new_cell = coordToIndex(new_x, new_y);
            //this only ever happens to people at the start of the list, so update the cell_grid
            
            //if you leave a slot empty, point the cell grid to -1
            //if not, point it to the next person in the list
            if(next[person] != -1){
                cell_grid[i] = next[person];
            } else {
                cell_grid[i] = -1;
            }

            //if your destination is empty, point it to you and point yourself to -1
            //if not, point the cell at yourself and point yourself to the old list head
            if(cell_grid[new_cell] == -1){
                cell_grid[new_cell] = person;
                next[person] = -1;
            } else {
                next[person] = cell_grid[new_cell];
                cell_grid[new_cell] = person;
            }

            //get the next person in the cell
            printf("Moving onto person %d\n", next_person);
            if(next_person == -1){
                break;
            }
            person = next_person; 
        }
    }
}

//these are called on the device, use CUDA atomic functions
__device__ void acquireCell(int *cell){
    int expected = 0;
    while(atomicCAS(cell, expected, 1) != expected){
        expected = 0;
    }
}

__device__ void releaseCell(int *cell){
    atomicExch(cell, 0);
}

// people sharing a cell have a chance to infect each other
__global__ void infectPeople(int* cell_grid, int* pos_x, int* pos_y, int* infected, int* next) {

}

// device function to make a random number
__device__ int generateCuRand() {
    curandState_t state;
    curand_init(RANDOM_SEED, clock(), 0, &state);
    //make it between -RANGE and RANGE
    return curand_uniform(&state) * (MOVE_RANGE * 2) - MOVE_RANGE;
}

//TODO: test inline
__host__ __device__ uint32_t coordToIndex(int x, int y) {
    return x * GRID_SIZE + y;
}