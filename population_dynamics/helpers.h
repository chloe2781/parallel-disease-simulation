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
#define MAX_MOVEMENT 8
#define MAX_STARTING_INFECTED 3

// data structure to store configuration about the game
// can be altered to test different factors
typedef struct {
  int end_day;             //day at which simulation ends
  int start_population;    //starting population for simulation
  int starting_infected;   //start number of people infected
  int length;              //board length
  int width;               //board width
  bool masking;            //whether there is masking enabled
  bool vaccination;        //whether there is vaccination enabled
  bool social_distancing;  //whether there is social distancing enabled
} GameConfig;

// data structure to store information about a person
typedef struct {
    std::atomic<int> x;    // x location in grid
    std::atomic<int> y;    // y location in grid
    int id;                // identifier for the person
    bool diseased;         // whether the person currently is infected (<14 days since day_infected)
    int day_infected;      // latest timestep at which individual was infected
    bool dead;             // whether the person is dead
    int variant;           // variant of the disease the person is infected with
    int immunity;          // number of days until the person is no longer immune
} Person;


// data structure to store information about a variation
typedef struct {
    std::atomic<int> variant_num;    // variant number of the disease
    int recovery_time;               // days it takes to no longer be contagious
    float mortality_rate;            // percent chance on each day that a person dies
    float infection_rate;            // percent chance that a person within the infected range is infected
    int infection_range;             // distance at which a person can be infected
    float mutation_rate;             // percent chance that an infection mutates upon infection of another person
    int immunity;                    // number of days until the person is no longer immune
} Variant;

// random range integer in range [0,max_range)
int randRange(int max_range){
    // Generate a random integer between 0 and (2 * max_range)
    int num = std::rand() % (2 * max_range+1);
    // Subtract max_range to get a random integer between -max_range and +max_range
    num -= max_range;

    return num;
}

int randRangePos(int max_range){return rand() % max_range;}
// random float in range [0,1]
float rand01(){return (float)rand() / (float)RAND_MAX;}

// Function to calculate distance between two people
double calculateDistance(const Person& person1, const Person& person2) {
    int dx = person1.x - person2.x;
    int dy = person1.y - person2.y;
    return std::sqrt(dx * dx + dy * dy);
}

float addPossibleVariation(float num) {
    float variation = rand01();
    int multiplier = rand01() >= .5 ? 1 : -1;
    return num + (num * variation * multiplier);
}

float addPossibleVariationInt(int num) {
    float variation = rand01();
    int multiplier = rand01() >= .5 ? 1 : -1;
    return int(num + (num * variation * multiplier));
}