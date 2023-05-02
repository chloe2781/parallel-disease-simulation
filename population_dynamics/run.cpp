#include "util.h"
#include <iostream>
#include <chrono>
int main() {

    // Initialize the random number generator
//    std::cout << " Checkpoint 0 " << std::endl;
    std::srand(std::time(0));

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
    config.percent_vaxed = .9

//    std::cout << " Checkpoint 0.1 " << std::endl;

    Person people[config.start_population];

    // ... code to initialize people ...
//    std::cout << " Checkpoint 1 " << std::endl;

    // Initialize the people array
    for (int i = 0; i < config.start_population; i++) {
        people[i].x = randRangePos(config.length); // assuming BOARD_LENGTH is defined in helper.h
        people[i].y = randRangePos(config.width);  // assuming BOARD_WIDTH is defined in helper.h
//        people[i].id = i;                      // set identifier for the person as the index
//        people[i].infected = false;            // set initial disease status as not infected
//        people[i].day_infected = -1;           // set initial day infected as -1 (not infected)
//        people[i].dead = false;                // set initial dead status as not dead
        people[i].status = 0;                   // set initial status as 0 (alive and not infected)
        people[i].variant = -1;                 // set initial variant as -1 (not infected and never caught the disease)
        people[i].immunity = (config.vaccination && rand01() < config.percent_vaxed) ? 90 : -1;                // set initial immunity as -1 (not immune and never caught the disease)
    }

//    std::cout << " Checkpoint 2 " << std::endl;

    // Initialize the variants array
    Variant variants[MAX_VARIANTS];

    variants[0].variant_num = 0;
    variants[0].recovery_time = 14;
    variants[0].mortality_rate = 0.015;
    variants[0].infection_rate = 0.5;
//    variants[0].infection_range = 3;
    variants[0].mutation_rate = 0.001;
    variants[0].immunity = 90;

    // Generate random indexes for people to be infected
    for (int i = 0; i < config.starting_infected; i++) {
        int id = randRangePos(config.start_population);
        std::cout << "Person " << id << " is infected" << std::endl;
//        people[id].status = true;
//        people[id].day_infected = 0;
        people[id].status = variants[0].recovery_time;
        people[id].variant = 0;
    }

//    std::cout << " Checkpoint 3" << std::endl;

    std::cout << " ----------------------------------------- " << std::endl;
    std::cout << " Starting Population" << std::endl;

    for (int i = 0; i < config.start_population; i++) {
        std::cout << "Person " << i
//                  << " - ID: " << people[i].id
                  << ", X: " << people[i].x
                  << ", Y: " << people[i].y
                  << ", Dead: " << (people[i].status < 0 ? "Yes" : "No")
                  << ", Immunity: " << people[i].immunity
                  << ", infected: " << (people[i].status > 0 ? "Yes" : "No")
                  << ", Day Infected To-go: " << people[i].status << std::endl;
    }

//    std::cout << " Checkpoint 4 " << std::endl;


    //time the simulation
    auto start = std::chrono::high_resolution_clock::now(); // Start time

    // simulate
    int max_var = disease_simulation(people, variants, config.end_day);

    //stop timer
    auto stop = std::chrono::high_resolution_clock::now(); // Stop time
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);


//    std::cout << " Checkpoint 5 " << std::endl;

    // ... code to do something with the updated people ...

    std::cout << " ----------------------------------------- " << std::endl;

    std::cout << " Ending Population" << std::endl;

    int end_population_size = 0;
    int end_immune_size = 0;
    // print the people after the simulation
    for (int i = 0; i < config.start_population; i++) {
        end_population_size += people[i].status < 0 ? 0 : 1;
        end_immune_size += people[i].immunity ? 1 : 0;
        std::cout << "Person " << i
//                  << " - ID: " << people[i].id
                  << ", X: " << people[i].x
                  << ", Y: " << people[i].y
                  << ", Dead: " << (people[i].status < 0 ? "Yes" : "No")
                  << ", Immunity: " << people[i].immunity
                  << ", infected: " << (people[i].status > 0 ? "Yes" : "No")
                  << ", Day Infected To-go: " << people[i].status << std::endl;
    }

    // print the variants
    std::cout << " ----------------------------------------- " << std::endl;
    for (int i = 0; i < max_var+1; i++) {
        std::cout << "Variant " << i
                  << " - Variant Number: " << variants[i].variant_num
                  << ", Recovery Time: " << variants[i].recovery_time
                  << ", Mortality Rate: " << variants[i].mortality_rate
                  << ", Infection Rate: " << variants[i].infection_rate
//                  << ", Infection Range: " << variants[i].infection_range
                  << ", Mutation Rate: " << variants[i].mutation_rate
                  << ", Immunity: " << variants[i].immunity
                  << std::endl;
    }

    std::cout << " ----------------------------------------- \n" << std::endl;
    std::cout << "Statistics" << std::endl;
    std::cout << "Time taken by disease_simulation: " << duration.count() << " microseconds" << std::endl;
    std::cout << "Start population size: " << config.start_population << std::endl;
    std::cout << "Ending population size: " << end_population_size << std::endl;
    std::printf("%s %.2f", "Mortality Rate:", float(100) * (1 - float(end_population_size)/float(config.start_population)));
    std::cout << '\n' << std::endl;

    //std::printf("%s %.2f", "Percent Immune:", float(100) * (1 - float(end_immune_size)/float(config.start_population)));

    return 0;
}
