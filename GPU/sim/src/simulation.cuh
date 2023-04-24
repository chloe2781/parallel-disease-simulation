#include <cstdint>
#include <string>
#include <atomic>

//template struct for variants
struct Variant {
int id;                // variant number of the disease
int recovery_time;      // days it takes to no longer be contagious
float mortality_rate;   // percent chance on each day that a person dies
float infection_rate;   // percent chance that a person within the infected range is infected
float mutation_rate;    // percent chance that an infection mutates upon infection of another person
int immunity_time;           // number of days until the person is no longer immune
};


__host__ void simulation();

void cudaCheck(const std::string &message);
__global__ void movePeople(int *positions);
__global__ void updateCellOccupancy(int *cell_grid_first, int *cell_grid_last, int *positions, int *next);
__global__ void infectPeople(Variant* variants, int* positions, int *variant_count, int *variant_cap, int* variant, int* immunity, int* dead);
__device__ int createVariant(Variant *variants, int *variant_count, int *variant_cap, int source_variant);
__global__ void killPeople(Variant* variants, int* variant, int* dead);
__global__ void tick(Variant* variants, int* immunity, int* variant, int* dead);
__device__ int randomMovement();
__device__ float randomFloat();
__host__ __device__ int coordToIndex(int x, int y);