#include <iostream>
#include <thread>
#include <vector>
#include <mutex>
#include <atomic>
#include <cstdlib>
#include <cmath>

// Some helpful constants
#define NUM_REGIONS 2
#define NUM_SPECIES 2
#define COMMUNITIES_PER_NUM_SPECIES 2
#define COMMUNITIES_PER_REGION (NUM_SPECIES*COMMUNITIES_PER_NUM_SPECIES)
#define TOTAL_COMMUNITIES (NUM_REGIONS*COMMUNITIES_PER_REGION)
#define MAX_STARTING_POPULATION 10
#define NUM_TIME_PERIODS 5

/************* 
 * With Presets of 2,2,2,10,5 you should get
 * 
 * ID[0]: of type [1]: in region [0]: had final population [11]
 * ID[1]: of type [1]: in region [0]: had final population [14]
 * ID[2]: of type [1]: in region [1]: had final population [237]
 * ID[3]: of type [0]: in region [1]: had final population [9]
 * ID[4]: of type [0]: in region [0]: had final population [97]
 * ID[5]: of type [0]: in region [0]: had final population [24]
 * ID[6]: of type [0]: in region [0]: had final population [5]
 * ID[7]: of type [1]: in region [1]: had final population [218]
 *
 * OR, running on Mac you may get:
 * ID[0]: of type [1]: in region [0]: had final population [14]
 * ID[1]: of type [0]: in region [0]: had final population [81]
 * ID[2]: of type [0]: in region [0]: had final population [43]
 * ID[3]: of type [0]: in region [0]: had final population [23]
 * ID[4]: of type [0]: in region [1]: had final population [14]
 * ID[5]: of type [0]: in region [1]: had final population [170]
 * ID[6]: of type [1]: in region [1]: had final population [8]
 * ID[7]: of type [0]: in region [0]: had final population [5]
 * ************/

// data structure to store information about each species
typedef struct {
    std::atomic<int> population;        // the population of a speciies
    std::atomic<int> food_collected;    // the food collected in the current time period
    int region_of_world;                // region of this species community
    int species_type;                   // type of species for this species community
    float growth_rate;                  // growth_rate for this species community
    bool flag;                          // flag in case helpful to have one (you may not need this)
    std::atomic<int> atomic_helper;     // atomic in case helpful to have one (you may not need this)
    std::atomic<float> atomic_helper2;  // atomic in case helpful to have one (you may not need this)
    std::mutex mtx;                     // mutex in case helpful to have one (you may not need this)
} Species_Community;

// food oracle function call
// call this with a community id to get a "random" amount of food back
// this represents one community member going out to get food
// we hardcode to 1 for determinism in testing but in theory should be random
int food_oracle(int community_id){return 1;};

// random range integer in range [0,max_range)
int randRange(int max_range){return rand() % max_range;}
// random float in range [0,1]
float rand01(){return (float)rand() / (float)RAND_MAX;}