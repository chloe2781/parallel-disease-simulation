// define constants for the simulation
#ifndef CONFIG_H
#define CONFIG_H


const int GRID_SIZE = 256; // size of the grid
const int POPULATION = 1000000; // population of the simulation, 2^20
const int RANDOM_SEED = 1337; // seed for the random number generator

//movement parameters
const int MOVE_RANGE = 5; // range of movement of the agents

const int THREADS = 256; // number of threads per block
const int BLOCKS = max(1, POPULATION / THREADS); // number of blocks
#endif // CONFIG_H