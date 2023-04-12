#include "helpers.h"
// I suggest testing with these set to 1 first!
#define NUM_BLOCKS 1
#define NUM_THREADS TOTAL_COMMUNITIES


// function to simulate population change for one community of one species
//
// Note: 1) The change in population for a community is proportional to
//          its growth_rate and local_population_share
//       2) If it has collected enough food to feed the population it grows, else it shrinks
//       3) If the population drops below 5 it goes extinct
//
// Hint: this should remain basically unchanged from the CPU version (depending on the rest of your implementation)
__device__
void update_community_population(Species_Community *s_communities, int community_id, float local_population_share) {
  //
  // # TODO
  //

  // changes in population
  float change = s_communities[community_id].growth_rate * local_population_share * s_communities[community_id].population;
  if (s_communities[community_id].food_collected >= s_communities[community_id].population) {
    s_communities[community_id].population += change; // grow
  }
  else {
    s_communities[community_id].population -= change; //shrink
  }
  if (s_communities[community_id].population < 5) {
    s_communities[community_id].population = 0; //extinct
  }
}

// function to find the local population share for one community of one species
//
// Note: 1) Population share is defined as the percentage of population in a region
//          that is a given species across all communities of all species
//
// Hint: this should remain basically unchanged from the CPU version (depending on the rest of your implementation)
__device__
float compute_local_population_share(Species_Community *s_communities, int community_id){
  //
  // # TODO
  //
  float total_pop = 0;
  float species_pop = 0;
  for (int i = 0; i < TOTAL_COMMUNITIES; i++) {
    // if same region, add to total_pop
    if (s_communities[i].region_of_world == s_communities[community_id].region_of_world) {
      total_pop += s_communities[i].population;
      // if same species in same region, add to species_pop
      if (s_communities[i].species_type == s_communities[community_id].species_type) {
        species_pop += s_communities[i].population;
      }
    }
  }
  return species_pop/total_pop;
}


// Updates the population for all communities of all species
//
// Note: 1) You will need to use compute_local_population_share and update_community_population
//       3) Make sure your logic is thread safe! Again this is likely to have a race condition!
//       4) Feel free to use helper functions if that makes your life easier!
__device__
void update_all_populations(Species_Community *communities){
  //
  // # TODO
  //

  // get local_population_share
  for (int i = threadIdx.x; i < TOTAL_COMMUNITIES; i += blockDim.x) {
    communities[i].helperf = compute_local_population_share(communities, i);
  }
  __syncthreads();

  // update population
  for (int i = threadIdx.x; i < TOTAL_COMMUNITIES; i += blockDim.x) {
    update_community_population(communities, i, communities[i].helperf);
  }
  __syncthreads();
}

// function to simulate food gathering
//
// Note: 1) Each round food starts at 0 and each member of the population tries to collect food
//       2) Please use food_oracle() to get a new amount of food for each member of the population
//       3) You can allocate threads to communites however you want!
//       3) All other implementation details are up to you! (Don't worry if your design isn't perfect!)
__device__
void gather_all_food(Species_Community *s_communities) {
  //
  // # TODO
  //

  // go through each community
  for (int i = threadIdx.x; i < TOTAL_COMMUNITIES; i += blockDim.x) {
    s_communities[threadIdx.x].food_collected = 0;
    // each member collect food
    for (int j = 0; j < s_communities[threadIdx.x].population; j += blockDim.y) {
      s_communities[threadIdx.x].food_collected += food_oracle(threadIdx.y);
    }
    __syncthreads();
  }
}

// the main kernel that computes the population dynamics over time
//
// Hints: 1) using shared memory may speed things up
//           but then make sure to save things back to RAM
//        2) make sure you do all NUM_TIME_PERIODS of gather_all_food
//           and update_all_populations
__global__
void population_dynamics_kernel(Species_Community *d_communities){
  //
  // # TODO
  //

  __shared__ Species_Community s_communities[TOTAL_COMMUNITIES];

  // load data from d to s
  for (int i = 0; i < TOTAL_COMMUNITIES; i++) {
    s_communities[i] = d_communities[i];
  }

  for (int i = 0; i < NUM_TIME_PERIODS; i++) {
    gather_all_food(s_communities);
    update_all_populations(s_communities);
  }
  __syncthreads();
  // load data from s to d
  for (int i = 0; i < TOTAL_COMMUNITIES; i++) {
    d_communities[i] = s_communities[i];
  }
}

// the main function
//
// Hints: set up GPU memory, run the kernel, clean up GPU memory
__host__
void population_dynamics(Species_Community *h_communities){
  //
  // #TODO
  //

  // set up GPU mem
  Species_Community *d_com;
  cudaMalloc(&d_com, sizeof(Species_Community)*TOTAL_COMMUNITIES);
  // copy data to GPU
  cudaMemcpy(d_com, h_communities, sizeof(Species_Community)*TOTAL_COMMUNITIES, cudaMemcpyHostToDevice);
  // run kernal
  population_dynamics_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(d_com);
  // sync
  cudaDeviceSynchronize();
  // copy data back to CPU
  cudaMemcpy(h_communities, d_com, sizeof(Species_Community)*TOTAL_COMMUNITIES, cudaMemcpyDeviceToHost);
  // clean up GPU mem
  cudaFree(d_com);
}