#include <cstdint>

__host__ void simulation(uint8_t* pos_x, uint8_t* pos_y);
__global__ void movePeople(uint8_t* pos_x, uint8_t* pos_y);
__device__ int generateCuRand();