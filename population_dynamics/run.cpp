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
  Person people[MAX_STARTING_POPULATION]; // assuming MAX_STARTING_POPULATION is defined in helper.h

  // ... code to initialize people ...

  // Initialize the people array
  for (int i = 0; i < MAX_STARTING_POPULATION; i++) {
      people[i].x = randRangePos(BOARD_LENGTH); // assuming BOARD_LENGTH is defined in helper.h
      people[i].y = randRangePos(BOARD_WIDTH);  // assuming BOARD_WIDTH is defined in helper.h
      people[i].id = i;                      // set identifier for the person as the index
      people[i].diseased = false;            // set initial disease status as not diseased
      people[i].day_infected = -1;           // set initial day infected as -1 (not infected)
  }

  for (int i = 0; i < MAX_STARTING_POPULATION; i++) {
      std::cout << "Person " << i << " - ID: " << people[i].id << ", X: " << people[i].x
                << ", Y: " << people[i].y << ", Diseased: " << (people[i].diseased ? "Yes" : "No")
                << ", Day Infected: " << people[i].day_infected << std::endl;
  }

  // Move the people within a fixed distance
  move(people);

  // ... code to do something with the updated people ...
  std::cout << " ----------------------------------------- " << std::endl;

  for (int i = 0; i < MAX_STARTING_POPULATION; i++) {
      std::cout << "Person " << i << " - ID: " << people[i].id << ", X: " << people[i].x
                << ", Y: " << people[i].y << ", Diseased: " << (people[i].diseased ? "Yes" : "No")
                << ", Day Infected: " << people[i].day_infected << std::endl;
  }

  return 0;


}
