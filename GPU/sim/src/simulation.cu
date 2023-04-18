// this file will manage the simulation, handling CPU GPU communication, running the phases, and managing memory
#include "config.h"
#include <stdio.h>
#include "simulation.cuh"
#include <curand_kernel.h>
#include <cstdint>
#include <cuda_runtime.h>
//allow use of uint8_t

__host__ void simulation() {

        //cell_grid[value] = ID of person in cell
        uint32_t* cell_grid = new uint32_t[GRID_SIZE * GRID_SIZE];
        //x position of each person, may not be needed
        uint8_t* pos_x = new uint8_t[POPULATION];
        //y position of each person, may not be needed
        uint8_t* pos_y = new uint8_t[POPULATION];
        //1 if infected, or if variants are introduced, the ID of the variant, 0 if not infected
        int* infected = new int[POPULATION];
        //next[value] = ID of next person in cell, 0 if none
        uint32_t* next = new uint32_t[GRID_SIZE * GRID_SIZE];

        //this gives each person a current memory footprint of 
        //4 + 1 + 1 + 4 + 4 = 14 bytes
        
        //initialize on host
        printf("Initializing data\n");
        for (int i = 0; i < GRID_SIZE * GRID_SIZE; i++) {
            cell_grid[i] = 0;
            next[i] = 0;
        }

        //set up GPU memory
        uint32_t *d_cell_grid = NULL;
        uint8_t *d_pos_x = NULL;
        uint8_t *d_pos_y = NULL;
        int *d_infected = NULL;
        uint32_t *d_next = NULL;

        printf("Allocating GPU memory\n");
        cudaMalloc((void**)&d_cell_grid, sizeof(uint32_t) * GRID_SIZE * GRID_SIZE);
        cudaMalloc((void**)&d_pos_x, sizeof(uint8_t) * POPULATION);
        cudaMalloc((void**)&d_pos_y, sizeof(uint8_t) * POPULATION);
        cudaMalloc((void**)&d_infected, sizeof(int) * POPULATION);
        cudaMalloc((void**)&d_next, sizeof(uint32_t) * GRID_SIZE * GRID_SIZE);
        if (cudaGetLastError() != cudaSuccess){
            printf("Error allocating GPU memory\n");
            return;
        }

        printf("Copying data to GPU\n");
        cudaMemcpy(d_cell_grid, cell_grid, sizeof(uint32_t) * GRID_SIZE * GRID_SIZE, cudaMemcpyHostToDevice);
        cudaMemcpy(d_pos_x, pos_x, sizeof(uint8_t) * POPULATION, cudaMemcpyHostToDevice);
        cudaMemcpy(d_pos_y, pos_y, sizeof(uint8_t) * POPULATION, cudaMemcpyHostToDevice);
        cudaMemcpy(d_infected, infected, sizeof(int) * POPULATION, cudaMemcpyHostToDevice);
        cudaMemcpy(d_next, next, sizeof(uint32_t) * GRID_SIZE * GRID_SIZE, cudaMemcpyHostToDevice);
        if (cudaGetLastError() != cudaSuccess){
            printf("Error copying data to GPU\n");
            return;
        }

        // run the gpu code once
        printf("Launching movePeople\n");
        movePeople<<<MOVE_BLOCKS, MOVE_THREADS>>>(d_cell_grid, d_pos_x, d_pos_y, next);
        cudaDeviceSynchronize();
        printf("Launching infectPeople\n");
        infectPeople<<<INFECTION_THREADS, INFECTION_THREADS>>>(d_cell_grid, d_pos_x, d_pos_y, d_infected, d_next);
        cudaDeviceSynchronize();
        printf("Kernels Complete\n");
        

        printf("Copying data back to CPU\n");
        cudaMemcpy(cell_grid, d_cell_grid, sizeof(uint32_t) * GRID_SIZE * GRID_SIZE, cudaMemcpyDeviceToHost);
        cudaMemcpy(pos_x, d_pos_x, sizeof(uint8_t) * POPULATION, cudaMemcpyDeviceToHost);
        cudaMemcpy(pos_y, d_pos_y, sizeof(uint8_t) * POPULATION, cudaMemcpyDeviceToHost);
        cudaMemcpy(infected, d_infected, sizeof(int) * POPULATION, cudaMemcpyDeviceToHost);
        cudaMemcpy(next, d_next, sizeof(uint32_t) * GRID_SIZE * GRID_SIZE, cudaMemcpyDeviceToHost);
        if (cudaGetLastError() != cudaSuccess){
            printf("Error copying data from GPU\n");
            return;
        }

        printf("Freeing GPU memory\n");
        cudaFree(d_cell_grid);
        cudaFree(d_pos_x);
        cudaFree(d_pos_y);
        cudaFree(d_infected);
        cudaFree(d_next);
}

// move people randomly around the grid
__global__ void movePeople(uint32_t* cell_grid, uint8_t* pos_x, uint8_t* pos_y, uint32_t* next) {

}


// people sharing a cell have a chance to infect each other
__global__ void infectPeople(uint32_t* cell_grid, uint8_t* pos_x, uint8_t* pos_y, int* infected, uint32_t* next) {

}

// device function to make a random number
__device__ int generateCuRand() {
    curandState_t state;
    curand_init(RANDOM_SEED, threadIdx.x, 0, &state);
    //make it between -RANGE and RANGE
    return curand_uniform(&state) * (MOVE_RANGE * 2) - MOVE_RANGE;
}

//TODO: test inline
__device__ int coordToIndex(int x, int y) {
    return x * GRID_SIZE + y;
}