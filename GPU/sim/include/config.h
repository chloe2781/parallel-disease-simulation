// define constants for the simulation
#ifndef CONFIG_H
#define CONFIG_H


const int GRID_SIZE = 256; // size of the grid, if you turn this up past 256, you will need to use bigger data types for the grid coordinates
const int POPULATION = 16; // population of the simulation, 2^20, targeting up to 300 million atm
const int RANDOM_SEED = 1337; // seed for the random number generator

//movement parameters, 
const int MOVE_RANGE = 5; // range of movement of the agents
const int MOVE_THREADS  = 4; // number of threads per block
const int MOVE_BLOCKS = POPULATION / MOVE_THREADS + (POPULATION % MOVE_THREADS != 0); // number of blocks

//infection parameters, infection is parallelized over each grid cell
const int INFECTION_THREADS = 128; // number of threads per block
const int INFECTION_BLOCKS = POPULATION / INFECTION_THREADS + (POPULATION % INFECTION_THREADS != 0); // number of blocks

//mutation parameters
//const float MUTATION_RANGE = 0.1; // how much the stats of a new agent can deviate from the parents

//kill parameters
const int KILL_THREADS = 128; // number of threads per block
const int KILL_BLOCKS = POPULATION / KILL_THREADS + (POPULATION % KILL_THREADS != 0); // number of blocks

//tick parameters
const int TICK_THREADS = 128; // number of threads per block
const int TICK_BLOCKS = POPULATION / TICK_THREADS + (POPULATION % TICK_THREADS != 0); // number of blocks


#endif // CONFIG_H