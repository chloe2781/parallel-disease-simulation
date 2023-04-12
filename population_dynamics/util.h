#include "helpers.h"
// I suggest testing with this set to 1 first!
#define MAX_THREADS 8

// function to simulate population change for one community of one species
//
// Note: 1) The change in population for a community is proportional to 
//          its growth_rate and local_population_share
//       2) If it has collected enough food to feed the population it grows, else it shrinks
//       3) If the population drops below 5 it goes extinct
void update_community_population(Species_Community *communities, int community_id, float local_population_share) {
  //
  // # TODO: add implementation
  //

  // changes in population
  float change = communities[community_id].growth_rate * local_population_share * communities[community_id].population;
  if (communities[community_id].food_collected >= communities[community_id].population) {
    communities[community_id].population += change; // grow
  }
  else {
    communities[community_id].population -= change; //shrink
  }
  if (communities[community_id].population < 5) {
    communities[community_id].population = 0; //extinct
  }
}

// function to find the local population share for one community of one species
//
// Note: 1) Population share is defined as the percentage of population in a region
//          that is a given species across all communities of all species
float compute_local_population_share(Species_Community *communities, int community_id){
  //
  // # TODO: add implementation
  //
  float total_pop = 0;
  float species_pop = 0;
  for (int i = 0; i < TOTAL_COMMUNITIES; i++) {
    // if same region, add to total_pop
    if (communities[i].region_of_world == communities[community_id].region_of_world) {
      total_pop += communities[i].population;
      // if same species in same region, add to species_pop
      if (communities[i].species_type == communities[community_id].species_type) {
        species_pop += communities[i].population;
      }
    }
  }
  return species_pop/total_pop;
}

// Helper function for compute_local_population_shares
void compute_helper (Species_Community *communities, int start, int end) {
  for (int id = start; id < end; id++) {
    communities[id].atomic_helper2 = compute_local_population_share(communities, id);
  }
}

// Helper function for update_community_population
void compute_helper2 (Species_Community *communities, int start, int end) {
  for (int id = start; id < end; id++) {
    update_community_population(communities, id, communities[id].atomic_helper2);
  }
}

// Updates the population for all communities of all species
//
// Note: 1) You will want to launch MAX_THREADS to compute this
//       2) You will need to use compute_local,_population_share and update_community_population
//       3) Make sure your logic is thread safe! Warning there likely is a data dependency!
//       4) Feel free to use helper functions if that makes your life easier!
void update_all_populations(Species_Community *communities){
  //
  // # TODO: add implementation
  //
  std::thread t[MAX_THREADS];
  // compute local population shares
  for (int i = 0; i < MAX_THREADS; i++) {
    int start = i * (TOTAL_COMMUNITIES/MAX_THREADS);
    int end = (i+1) * (TOTAL_COMMUNITIES/MAX_THREADS);
    t[i] = std::thread(compute_helper, communities, start, end);
  }
  for (int i = 0; i < MAX_THREADS; i++) {
    t[i].join();
  }
  // update population
  for (int i = 0; i < MAX_THREADS; i++) {
    int start = i * (TOTAL_COMMUNITIES/MAX_THREADS);
    int end = (i+1) * (TOTAL_COMMUNITIES/MAX_THREADS);
    t[i] = std::thread(compute_helper2, communities, start, end);
  }
  for (int i = 0; i < MAX_THREADS; i++) {
    t[i].join();
  }
}

// Helper function for food_oracle
void compute_helper3 (Species_Community *communities, int start, int end, int id) {
  for (int i = start; i < end; i++) {
    communities[id].food_collected += food_oracle(i);
  }
}

// function to simulate food gathering
// 
// Note: 1) Each round food starts at 0 and each member of the population tries to collect food
//       2) Please use food_oracle() to get a new amount of food for each member of the population
//       3) Please use MAX_THREADS threads per Species_Community! (Not spread across them but for each one)
//       4) All other implementation details are up to you!
void gather_all_food(Species_Community *communities) {
  //
  // # TODO: add implementation
  //
  std::thread t[MAX_THREADS];
  for (int i = 0; i < TOTAL_COMMUNITIES; i++) {
    communities[i].food_collected = 0;
    // use MAX_THREADS threads per Species_Community
    for (int j = 0; j < MAX_THREADS; j++) {
      int start = j * (communities[i].population/MAX_THREADS);
      int end = (j+1) * (communities[i].population/MAX_THREADS);
      // if last thread, add remainder
      if (j == MAX_THREADS-1) {end += communities[i].population % MAX_THREADS; }
      t[j] = std::thread(compute_helper3, communities, start, end, i);
    }
    for (int j = 0; j < MAX_THREADS; j++) {
      t[j].join();
    }
  }
}

// the main function
//
// Hints: gathers food and updates the population for everyone
//        for NUM_TIME_PERIODS
void population_dynamics(Species_Community *communities){
  //
  // # TODO
  //
  for (int i = 0; i < NUM_TIME_PERIODS; i++) {
    gather_all_food(communities);
    update_all_populations(communities);
  }
}