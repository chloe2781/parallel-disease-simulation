#include "util.cuh"

__host__
int main() {
  // initialize random and data
  srand(1337);
  Species_Community h_communities[TOTAL_COMMUNITIES];
  for (int community_id = 0; community_id < TOTAL_COMMUNITIES; community_id++){
    h_communities[community_id].population = rand_range(MAX_STARTING_POPULATION) + 5;
    h_communities[community_id].region_of_world = rand_range(NUM_REGIONS);
    h_communities[community_id].species_type = rand_range(NUM_SPECIES);
    h_communities[community_id].growth_rate = rand01();
  }

  for (int community_id = 0; community_id < TOTAL_COMMUNITIES; community_id++){
    std::cout << "ID[" << community_id << "]: of type [" << h_communities[community_id].species_type <<
                 "]: in region [" << h_communities[community_id].region_of_world << "]: had initial population [" << 
                 h_communities[community_id].population << "]" << std::endl;
  }
  
  // the main function
  population_dynamics(h_communities);

  // print the final populations
  std::cout << "\n---------\n---------\n";
  for (int community_id = 0; community_id < TOTAL_COMMUNITIES; community_id++){
    std::cout << "ID[" << community_id << "]: of type [" << h_communities[community_id].species_type <<
                 "]: in region [" << h_communities[community_id].region_of_world << "]: had final population [" << 
                 h_communities[community_id].population << "]" << std::endl;
  }
  return 0;
}
