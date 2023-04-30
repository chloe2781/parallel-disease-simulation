// define constants for the simulation
#ifndef CONFIG_H
#define CONFIG_H


//overall parameters
const int GRID_SIZE = 100; // size of the grid
const int POPULATION = 5; // population of the simulation, 2^20, targeting up to 300 million atm
const int STARTING_INFECTED = 2; // number of infected agents at the start of the simulation
const int RANDOM_SEED = 1337; // seed for the random number generator
const int EPOCHS = 4; // number of epochs to run the simulation for
const int SHMEM_KB = 48; //change per GPU

//movement parameters, 
const int MOVE_RANGE = 5; // range of movement of the agents
const int MOVE_THREADS  = 4; // number of threads per block
const int MOVE_BLOCKS = POPULATION / MOVE_THREADS + (POPULATION % MOVE_THREADS != 0); // number of blocks

//infection parameters, infection is parallelized over each grid cell
const int INFECTION_THREADS = 4; // number of threads per block
const int INFECTION_BLOCKS = POPULATION / INFECTION_THREADS + (POPULATION % INFECTION_THREADS != 0); // number of blocks

//mutation parameters
__constant__ const float MUTATION_RANGE = 0.05f; // how much the stats of a new agent can deviate from the parents (0 to 1)

//kill parameters
const int KILL_THREADS = 4; // number of threads per block
const int KILL_BLOCKS = POPULATION / KILL_THREADS + (POPULATION % KILL_THREADS != 0); // number of blocks

//tick parameters
const int TICK_THREADS = 4; // number of threads per block
const int TICK_BLOCKS = POPULATION / TICK_THREADS + (POPULATION % TICK_THREADS != 0); // number of blocks

//snapshot parameters
const int SNAPSHOT_THREADS = 4; // number of threads per block
#endif // CONFIG_H