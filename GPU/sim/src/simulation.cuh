__host__ void simulation(int *pos_x, int *pos_y);
__global__ void movePeople(int *pos_x, int *pos_y);
__global__ void generateRandomInt4(int4 *out);
__device__ int generateCuRand();