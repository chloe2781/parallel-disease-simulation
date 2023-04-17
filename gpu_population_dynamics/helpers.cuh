#include <iostream>
#include <thread>
#include <vector>
#include <mutex>
#include <atomic>
#include <cstdlib>
#include <cmath>

// Some helpful constants
#define BOARD_LENGTH 256
#define BOARD_WIDTH 256
#define MAX_STARTING_POPULATION 1000000
#define NUM_TIME_PERIODS 25

// data structure to store information about a person
// unlike in the CPU case we can't explicitly put atomics here
// or mutexs etc.
typedef struct {
    int x;                 // x location in grid
    int y;                 // y location in grid
    int id;                // identifier for the person
    bool diseased;         // whether the person currently is infected (<14 days since day_infected)
    int day_infected;      // latest timestep at which individual was infected
} Person;

// data structure to store information about a variation
typedef struct {
    int variant_num;                 // variant number of the disease
    int recovery_time;               // days it takes to no longer be contagious
    float mortality_rate;            // percent chance on each day that a person dies
    float infection_rate;            // percent chance that a person within the infected range is infected
    int infection_range;             // distance at which a person can be infected
    float mutation_rate;             // percent chance that an infection mutates upon infection of another person
} Variant;

// random range integer in range [0,max_range)
__host__
int rand_range(int max_range){return rand() % max_range;}
// random float in range [0,1]
__host__
float rand01(){return (float)rand() / (float)RAND_MAX;}