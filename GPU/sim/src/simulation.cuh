#include <cstdint>
#include <string>
#include <atomic>
__host__ void simulation();
__global__ void movePeople(int* cell_grid, int* pos_x, int* pos_y, int* next, int *lock);
__device__ void acquireCell(int *cell);
__device__ void releaseCell(int *cell);
__global__ void infectPeople(int* cell_grid, int* pos_x, int* pos_y, int* infected, int* next);
__device__ int generateCuRand();
__host__ __device__ uint32_t coordToIndex(int x, int y);
