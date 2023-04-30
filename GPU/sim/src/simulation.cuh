#include <cstdint>
#include <string>
#include <atomic>

//template struct for variants

//i know this is noob but it helps readability
using namespace std;

struct Variant {
int id;                 // variant number of the disease
int recovery_time;      // days it takes to no longer be contagious
float mortality_rate;   // percent chance on each day that a person dies
float infection_rate;   // percent chance that a person within the infected range is infected
float mutation_rate;    // percent chance that an infection mutates upon infection of another person
int immunity_time;      // number of days until the person is no longer immune
};


//stores aggregate data from an epoch
struct Snapshot {
    int epoch;
    int variant_count;
    int alive;
    int dead;
    int infected;
    int uninfected;
    int immune;
    int fresh;
};

__host__ void simulation();

void cudaCheck(const std::string &message);
__global__ void showVariants(Variant* variants, int * variant_count);
__global__ void gpuPeek(int* positions, int* variant, int* immunity, int* dead, bool* fresh);
__global__ void movePeople(int *positions, int epoch);
__global__ void infectPeople(Variant* variants, int* positions, int *variant_count, int *variant_cap, int* variant, int* immunity, int* dead, bool* fresh);
__device__ int createVariant(Variant *variants, int *variant_count, int *variant_cap, int source_variant);
__global__ void killPeople(Variant* variants, int* variant, int* dead, bool* fresh);
__global__ void tick(Variant* variants, int* immunity, int* variant, int* dead, bool* fresh);
__device__ int randomMovement(int thread_id, int offset);
__device__ float randomFloat(int tid);
__global__ void takeSnapshot(Snapshot *snapshots, int epoch, int *immunity, int *dead, bool *fresh, int * variant_count);
void outputSnapshots(Snapshot *snapshots);


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