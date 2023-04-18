#include <cstdint>

__host__ void simulation();
__global__ void movePeople(uint32_t* cell_grid, uint8_t* pos_x, uint8_t* pos_y, uint32_t* next);
__global__ void infectPeople(uint32_t* cell_grid, uint8_t* pos_x, uint8_t* pos_y, int* infected, uint32_t* next);
__device__ int generateCuRand();