#include <iostream>
#include <thread>
#include <vector>
#include <mutex>
#include <atomic>
#include <cstdlib>
#include <cmath>

// Some helpful constants
#define BOARD_LENGTH 100
#define BOARD_WIDTH 100
#define MAX_STARTING_POPULATION 10000
#define NUM_TIME_PERIODS 365
#define MAX_MOVEMENT 8
#define MAX_STARTING_INFECTED 1
#define MAX_VARIANTS 10000
#define INFECTION_RANGE 2

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
  float percent_vaxed;     //percent of the population vaccinated
} GameConfig;

// data structure to store information about a person

/*
  CONSIDER:
  - status:
     - -1: dead (bool dead)
      - 0: alive but no infection (bool dead = true, bool infect = false, int day_infected = 0)
      - >0: alive and infected (bool dead = true, bool infect = true, int day_infected = >1)
  => from 8 members to 5 members

  ** Could make immunity to be the date that the person died. Just a thought
*/

typedef struct {
    std::atomic<int> x;    // x location in grid
    std::atomic<int> y;    // y location in grid
//    int id;                // identifier for the person
//    bool infected;         // whether the person currently is infected (<14 days since day_infected)
//    int day_infected;      // latest timestep at which individual was infected
//    bool dead;             // whether the person is dead
    int status;            // status of the person
    int variant;           // variant of the disease the person is infected with
    int immunity;          // number of days until the person is no longer immune
} Person;


// data structure to store information about a variation
typedef struct {
    std::atomic<int> variant_num;    // variant number of the disease
    int recovery_time;               // days it takes to no longer be contagious
    float mortality_rate;            // percent chance on each day that a person dies
    float infection_rate;            // percent chance that a person within the infected range is infected
//    int infection_range;             // distance at which a person can be infected
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

// Potential random number generator for mutating variants
double generateRandomNumber() {
    // Generate a random integer between 0 and 100
    int randomInt = std::rand() % 101;
    // Convert the random integer to a double between 0.0 and 0.01
    double randomNumber = randomInt / 10000.0;
    return randomNumber;
}

// Function to calculate distance between two people
// Switched to manhattan distance for performance
double calculateDistance(const Person& person1, const Person& person2) {
//    int dx = person1.x - person2.x;
//    int dy = person1.y - person2.y;
//    return std::sqrt(dx * dx + dy * dy);
    return abs(person1.x - person2.x) + abs(person1.y - person2.y);
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