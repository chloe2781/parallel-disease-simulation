#include <cstdint>
#include <string>
#include <atomic>

//template struct for variants

//i know this is noob but it helps readability
using namespace std;

struct Variant {
int id;                // variant number of the disease
int recovery_time;      // days it takes to no longer be contagious
float mortality_rate;   // percent chance on each day that a person dies
float infection_rate;   // percent chance that a person within the infected range is infected
float mutation_rate;    // percent chance that an infection mutates upon infection of another person
int immunity_time;           // number of days until the person is no longer immune

    string toString(){
        return "Variant " + to_string(id) + " - " + " mort: " + to_string(mortality_rate) + " inf: " + to_string(infection_rate) + " mut: " + to_string(mutation_rate) + " rec: " + to_string(recovery_time) + " imm: " + to_string(immunity_time);
    }
};


__host__ void simulation();

void cudaCheck(const std::string &message);
__global__ void showVariants(Variant* variants, int * variant_count);
__global__ void gpuPeek(int* positions, int* variant, int* immunity, int* dead, bool* fresh);
__global__ void movePeople(int *positions, int epoch);
__global__ void updateCellOccupancy(int *cell_grid_first, int *cell_grid_last, int *positions, int *next);
__global__ void infectPeople(Variant* variants, int* positions, int *variant_count, int *variant_cap, int* variant, int* immunity, int* dead, bool* fresh);
__device__ int createVariant(Variant *variants, int *variant_count, int *variant_cap, int source_variant);
__global__ void killPeople(Variant* variants, int* variant, int* dead, bool* fresh);
__global__ void tick(Variant* variants, int* immunity, int* variant, int* dead, bool* fresh);
__device__ int randomMovement(int tid);
__device__ float randomFloat(int tid);
__host__ __device__ int coordToIndex(int x, int y);

static __inline__ __device__ bool atomicCAS(bool *address, bool compare, bool val)
{
    unsigned long long addr = (unsigned long long)address;
    unsigned pos = addr & 3;  // byte position within the int
    int *int_addr = (int *)(addr - pos);  // int-aligned address
    int old = *int_addr, assumed, ival;

    bool current_value;

    do
    {
        current_value = (bool)(old & (1U << (8 * pos)));

        if(current_value != compare) // If we expected that bool to be different, then
            break; // stop trying to update it and just return its current value

        assumed = old;
        if(val)
            ival = old | (1 << (8 * pos));
        else
            ival = old & (~(1U << (8 * pos)));
        old = atomicCAS(int_addr, assumed, ival);
    } while(assumed != old);

    return current_value;
}