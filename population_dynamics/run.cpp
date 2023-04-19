#include "util.h"

int main() {

// BRIAN'S CODE
//
//  // initialize random
//  srand(1337);
//  Species_Community communities[TOTAL_COMMUNITIES];
//
//  for (int community_id = 0; community_id < TOTAL_COMMUNITIES; community_id++){
//    communities[community_id].population = randRange(MAX_STARTING_POPULATION) + 5;
//    communities[community_id].region_of_world = randRange(NUM_REGIONS);
//    communities[community_id].species_type = randRange(NUM_SPECIES);
//    communities[community_id].growth_rate = rand01();
//  }
//
//  for (int community_id = 0; community_id < TOTAL_COMMUNITIES; community_id++){
//    std::cout << "ID[" << community_id << "]: of type [" << communities[community_id].species_type <<
//                 "]: in region [" << communities[community_id].region_of_world << "]: had initial population [" <<
//                 communities[community_id].population << "]" << std::endl;
//  }
//
//  // run the simulation
//  population_dynamics(communities);
//
//  // print the final populations
//  std::cout << "\n---------\n---------\n";
//  for (int community_id = 0; community_id < TOTAL_COMMUNITIES; community_id++){
//    std::cout << "ID[" << community_id << "]: of type [" << communities[community_id].species_type <<
//                 "]: in region [" << communities[community_id].region_of_world << "]: had final population [" <<
//                 communities[community_id].population << "]" << std::endl;
//  }

    // Initialize the random number generator
    std::srand(std::time(0));
    // Seed the random number generator with current time

    // Setting up game config which can account for different factors of simulation run
    GameConfig config;
    config.end_day = NUM_TIME_PERIODS;
    config.start_population = MAX_STARTING_POPULATION; //assuming MAX_STARTING_POPULATION is defined in helper.h
    config.starting_infected = MAX_STARTING_INFECTED;
    config.length = BOARD_LENGTH;
    config.width = BOARD_WIDTH;
    //these can change depending on which simulation we run
    config.masking = false;
    config.vaccination = false;
    config.social_distancing = false;

    Person people[config.start_population];

    // ... code to initialize people ...

    // Initialize the people array
    for (int i = 0; i < config.start_population; i++) {
        people[i].x = randRangePos(config.length); // assuming BOARD_LENGTH is defined in helper.h
        people[i].y = randRangePos(config.width);  // assuming BOARD_WIDTH is defined in helper.h
        people[i].id = i;                      // set identifier for the person as the index
        people[i].diseased = false;            // set initial disease status as not diseased
        people[i].day_infected = -1;           // set initial day infected as -1 (not infected)
        people[i].dead = false;                // set initial dead status as not dead
        people[i].variant = -1;                 // set initial variant as -1 (not infected)
        people[i].immunity = 0;                // set initial immunity as 0 (not immune)
    }


    // Generate random indexes for people to be infected
    // moved from util.h because we only run this once
    for (int i = 0; i < config.starting_infected; i++) {
        int id = randRangePos(config.starting_infected);
        people[id].diseased = true;
        people[id].day_infected = 0;
        people[id].variant = 0;
    }


    for (int i = 0; i < config.start_population; i++) {
        std::cout << "Person " << i << " - ID: " << people[i].id << ", X: " << people[i].x
                << ", Y: " << people[i].y << ", Diseased: " << (people[i].diseased ? "Yes" : "No")
                << ", Day Infected: " << people[i].day_infected << std::endl;
    }

    // Move the people within a fixed distance
    move(people, 0, 17);

    // ... code to do something with the updated people ...
    std::cout << " ----------------------------------------- " << std::endl;

    for (int i = 0; i < config.start_population; i++) {
        std::cout << "Person " << i << " - ID: " << people[i].id << ", X: " << people[i].x
                << ", Y: " << people[i].y << ", Diseased: " << (people[i].diseased ? "Yes" : "No")
                << ", Day Infected: " << people[i].day_infected << std::endl;
    }

    return 0;

}
