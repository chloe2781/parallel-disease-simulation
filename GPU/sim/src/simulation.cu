// this file will manage the simulation, handling CPU GPU communication, running the phases, and managing memory
#include "config.h"
#include <stdio.h>
#include "simulation.cuh"
#include <curand_kernel.h>
#include <cstdint>
#include <cuda_runtime.h>
//allow use of uint8_t

__host__ void simulation(uint8_t* pos_x, uint8_t* pos_y) {

        // set up GPU memory
        uint8_t* d_pos_x;
        uint8_t* d_pos_y;

        int size = sizeof(uint8_t) * POPULATION;

        cudaMalloc((void**)&d_pos_x, size);
        cudaMalloc((void**)&d_pos_y, size);
        
        //ensure that malloc worked
        if (d_pos_x == NULL){
            printf("Error allocating GPU memory 1\n");
            return;
        }
        if (d_pos_y == NULL){
            printf("Error allocating GPU memory 2\n");
            return;
        }
        printf("Done\n");
        //printf("Copying data to GPU...");
        //copy data into GPU memory
        cudaMemcpy(d_pos_x, pos_x, sizeof(uint8_t) * POPULATION, cudaMemcpyHostToDevice);
        cudaMemcpy(d_pos_y, pos_y, sizeof(uint8_t) * POPULATION, cudaMemcpyHostToDevice);
        if (cudaGetLastError() != cudaSuccess){
            printf("Error copying data to GPU\n");
            return;
        }
        //printf("Done\n");

        // run the gpu code once
        printf("Before movePeople\n");
        movePeople<<<BLOCKS, THREADS>>>(d_pos_x, d_pos_y);
        cudaDeviceSynchronize();
        printf("After movePeople\n");
        
        //copy data back into CPU memory
        cudaMemcpy(pos_x, d_pos_x, sizeof(uint8_t) * POPULATION, cudaMemcpyDeviceToHost);
        cudaMemcpy(pos_y, d_pos_y, sizeof(uint8_t) * POPULATION, cudaMemcpyDeviceToHost);
        if (cudaGetLastError() != cudaSuccess){
            printf("Error copying data from GPU\n");
            return;
        }


        //free GPU memory
        cudaFree(d_pos_x);
        cudaFree(d_pos_y);

}

// copy the relevant data into shared GPU memory
__global__ void movePeople(uint8_t *pos_x, uint8_t *pos_y) {
    
    pos_x[blockIdx.x * blockDim.x + threadIdx.x] = (pos_x[blockIdx.x * blockDim.x + threadIdx.x] + generateCuRand() + GRID_SIZE) % GRID_SIZE;
    pos_y[blockIdx.x * blockDim.x + threadIdx.x] = (pos_y[blockIdx.x * blockDim.x + threadIdx.x] + generateCuRand() + GRID_SIZE) % GRID_SIZE;
}

// device function to make a random number
__device__ int generateCuRand() {
    curandState_t state;
    curand_init(RANDOM_SEED, threadIdx.x, 0, &state);
    //make it between -RANGE and RANGE
    return curand_uniform(&state) * (MOVE_RANGE * 2) - MOVE_RANGE;
}