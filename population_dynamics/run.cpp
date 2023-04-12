#include "util.h"

int main() {
  // initialize random
  srand(1337);
  Species_Community communities[TOTAL_COMMUNITIES];
  
  for (int community_id = 0; community_id < TOTAL_COMMUNITIES; community_id++){
    communities[community_id].population = randRange(MAX_STARTING_POPULATION) + 5;
    communities[community_id].region_of_world = randRange(NUM_REGIONS);
    communities[community_id].species_type = randRange(NUM_SPECIES);
    communities[community_id].growth_rate = rand01();
  }

  for (int community_id = 0; community_id < TOTAL_COMMUNITIES; community_id++){
    std::cout << "ID[" << community_id << "]: of type [" << communities[community_id].species_type <<
                 "]: in region [" << communities[community_id].region_of_world << "]: had initial population [" << 
                 communities[community_id].population << "]" << std::endl;
  }
  
  // run the simulation
  population_dynamics(communities);
  
  // print the final populations
  std::cout << "\n---------\n---------\n";
  for (int community_id = 0; community_id < TOTAL_COMMUNITIES; community_id++){
    std::cout << "ID[" << community_id << "]: of type [" << communities[community_id].species_type <<
                 "]: in region [" << communities[community_id].region_of_world << "]: had final population [" << 
                 communities[community_id].population << "]" << std::endl;
  }
  return 0;
}
