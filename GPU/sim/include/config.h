// define constants for the simulation
#ifndef CONFIG_H
#define CONFIG_H


const int GRID_SIZE = 256; // size of the grid, if you turn this up past 256, you will need to use bigger data types for the grid coordinates
const int POPULATION = 1048576; // population of the simulation, 2^20, targeting up to 300 million atm
const int RANDOM_SEED = 1337; // seed for the random number generator

//movement parameters
const int MOVE_RANGE = 5; // range of movement of the agents
const int MOVE_THREADS  = 32; // number of threads per block
const int MOVE_BLOCKS = POPULATION / MOVE_THREADS; // number of blocks

//infection parameters
const int INFECTION_THREADS = 32; // number of threads per block
const int INFECTION_BLOCKS = POPULATION / INFECTION_THREADS; // number of blocks
#endif // CONFIG_H