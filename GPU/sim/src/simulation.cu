// this file will manage the simulation, handling CPU GPU communication, running the phases, and managing memory
#include "config.h"
#include <stdio.h>
#include "simulation.cuh"
#include <curand_kernel.h>

__host__ void simulation(int *pos_x, int *pos_y) {

        // set up GPU memory
        int* d_pos_x;
        int* d_pos_y;

        cudaMalloc((void**)&d_pos_x, sizeof(int) * POPULATION);
        cudaMalloc((void**)&d_pos_y, sizeof(int) * POPULATION);

        //ensure that malloc worked
        if (d_pos_x == NULL || d_pos_y == NULL) {
            printf("Error: malloc failed");
            return;
        }

        //copy data into GPU memory
        cudaMemcpy(d_pos_x, pos_x, sizeof(int) * POPULATION, cudaMemcpyHostToDevice);
        cudaMemcpy(d_pos_y, pos_y, sizeof(int) * POPULATION, cudaMemcpyHostToDevice);
        
        // run the gpu code once
        printf("Before movePeople\n");
        movePeople<<<BLOCKS, THREADS>>>(d_pos_x, d_pos_y);
        cudaDeviceSynchronize();
        printf("After movePeople");
        
        //copy data back into CPU memory
        cudaMemcpy(pos_x, d_pos_x, sizeof(int) * POPULATION, cudaMemcpyDeviceToHost);
        cudaMemcpy(pos_y, d_pos_y, sizeof(int) * POPULATION, cudaMemcpyDeviceToHost);


        //free GPU memory
        cudaFree(d_pos_x);
        cudaFree(d_pos_y);

}

// copy the relevant data into shared GPU memory
__global__ void movePeople(int *pos_x, int *pos_y) {
    
    //each thread will be responsible for one vector operation
    //so shared memory should contain threads * vector_size elements

    const int num_elements = THREADS;
    __shared__ int s_pos_x[num_elements];
    __shared__ int s_pos_y[num_elements];

    //block 0 handles 0 to num_elements
    //block 1 handles num_elements to THREADS * 8, etc
    int offset = num_elements * blockIdx.x;

    // need to coalesce memory accesses
    // each thread will handle 4 elements
    for(int i = 0; i < num_elements; i += THREADS) {
        s_pos_x[i + threadIdx.x] = pos_x[offset + i + threadIdx.x];
        s_pos_y[i + threadIdx.x] = pos_y[offset + i + threadIdx.x];
    }

    __syncthreads();


    for(int i = 0; i < num_elements; i += THREADS) {
        //add in grid size to ensure that the number is positive
        s_pos_x[i + threadIdx.x] = (s_pos_x[i + threadIdx.x] + generateCuRand() + GRID_SIZE) % GRID_SIZE;
        s_pos_y[i + threadIdx.x] = (s_pos_y[i + threadIdx.x] + generateCuRand() + GRID_SIZE) % GRID_SIZE;
        // if(s_pos_x[i + threadIdx.x] == 42069){
        //     printf("error: thread overlap");
        // }

        // s_pos_x[i + threadIdx.x] = 42069;
        // s_pos_y[i + threadIdx.x] = threadIdx.x;

    }

    __syncthreads();

    //now we need to copy the data back into global memory
    for(int i = 0; i < num_elements; i += THREADS) {
        pos_x[offset + i + threadIdx.x] = s_pos_x[i + threadIdx.x];
        pos_y[offset + i + threadIdx.x] = s_pos_y[i + threadIdx.x];
    }
}

// device function to make a random number
__device__ int generateCuRand() {
    curandState_t state;
    curand_init(RANDOM_SEED, threadIdx.x, 0, &state);
    //make it between -RANGE and RANGE
    return curand_uniform(&state) * (MOVE_RANGE * 2) - MOVE_RANGE;
}