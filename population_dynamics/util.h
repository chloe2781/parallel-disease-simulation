#include "helpers.h"
#include <thread>
#include <atomic>
#define MAX_THREADS 16 // subject to change

std::atomic<int> max_variant(0);

//create a flat vector to store the board and all the people in each cell
// std::vector<std::vector<Person>> board(BOARD_WIDTH * BOARD_LENGTH);

/* function to move ONE person within a fixed distance
   randomly move on both axes
   edges are wrapped based on world size
*/
void move(Person *people, int start, int end) {
    //for (int i = 0; i < MAX_STARTING_POPULATION; i++) { //removed to thread

    for (int i = start; i < end; i++) {
        // Generate random offsets for x and y coordinates within the movement range

        int offsetX = randRange(MAX_MOVEMENT);
        int offsetY = randRange(MAX_MOVEMENT);

        // Update the x and y coordinates of the person, wrapping around the world size
        people[i].x = (people[i].x + offsetX);
        people[i].y = (people[i].y + offsetY);

        // Wrap around the world size
        if (people[i].x >= BOARD_LENGTH) {
          people[i].x = BOARD_LENGTH -1;
        }
        else if (people[i].x <= 0) people[i].x = 1;

        if (people[i].y >= BOARD_WIDTH) {
          people[i].y = BOARD_WIDTH -1;
        }
        else if (people[i].y <= 0) people[i].y = 1;

//         doesn't work yet
//        add people to cell in board
//        board[people[i].x * BOARD_WIDTH + people[i].y].push_back(&person[i]);

    }
}

/* Function to simulates ONE infected person dying based on variant mortality rate
     Generate a random val btw 0 and 1 and compare to mortality rate
     Those who survive a disease long enough gain immunity.
     Immunity is temporary dictated by virus stats
*/
void die(Person *people, Variant *variants, int start, int end, int curr_day) {
    //for (int i = 0; i < MAX_STARTING_POPULATION; i++) { //removed to thread

    for (int i = start; i < end; i++) {

      // if person is infected, decrement day infected and update if they die
      if (people[i].status > 0) {
        people[i].status--; // increment time to account for current day we're on
        float prob = rand01();
        if (prob < variants[people[i].variant].mortality_rate) {
          people[i].status = -1;
          continue;
        }

      // if person is alive and not infected, update immunity
      } else if (people[i].status == 0 && people[i].immunity < 1) {
//        people[i].infected = false;
        people[i].immunity = variants[people[i].variant].immunity;
//        std::cout << "Person " << i << " is now immune" << std::endl;
//        std::cout << "immunity: " << people[i].immunity << std::endl;
//        people[i].day_infected = 0;
      } else if (people[i].immunity > 0){
        people[i].immunity--;
      }
    }
}

/* Mutations are kept within the *variants list
    If a new mutation is made, it is modified from its parent variant and added into the list of mutations
*/
int mutate(Variant *variants, int i){
    max_variant += 1;
    Variant &new_variant = variants[max_variant];
    new_variant.variant_num.store(max_variant);
    new_variant.recovery_time = addPossibleVariationInt(variants[i].recovery_time);
    new_variant.mortality_rate = addPossibleVariation(variants[i].mutation_rate);
    new_variant.infection_rate = addPossibleVariation(variants[i].infection_rate);
    new_variant.mutation_rate = addPossibleVariation(variants[i].mutation_rate);
    new_variant.immunity = addPossibleVariationInt(variants[i].immunity);
    return max_variant;
}

/* function for ONE person to infect others within infection radius
    for all people, check if they are within infection radius of infected person
    if they are, generate a random val btw 0 and 1 and compare to infection rate
    if random val is less than infection rate, infect person
    if person is already infected, do nothing
    if person is already dead, do nothing
*/
void infect(Person *people, Variant *variants, int start, int end, int curr_day){

    for (int i = start; i < end; i++) {
        //ensure people are not dead AND we cannot infect someone unless they have been infected for at least 1 day
        if (people[i].status > 0 && (people[i].status < variants[people[i].variant].recovery_time)) {
            for (int j = 0; j < MAX_STARTING_POPULATION; j++) {
                if (i != j && people[j].status == 0 && people[j].immunity == 0) {

                    Variant& v = variants[people[i].variant];

                    if (calculateDistance(people[i], people[j]) <= INFECTION_RANGE) {

                        float infectionProb = rand01();
                        if (infectionProb < v.infection_rate) {
//                            people[j].infected = true;
//                            people[j].day_infected = curr_day;
//                            people[j].variant = v.variant_num; //either variant from person infected, or small variation
                            float mutationProb = rand01();
                            if (mutationProb < v.mutation_rate && max_variant < MAX_VARIANTS) { //small chance of variation
                                people[j].variant = mutate(variants, v.variant_num);
                            }else{
                                people[j].variant = v.variant_num;
                                people[j].status = v.recovery_time;
                            }
                        }
                    }
                }
            }
        }
    }
}


/* Updates the population for all people
    people will move, die, and infect others if still alive
*/
void update_all_people(Person *people, Variant *variants, int curr_day) {

    int num_threads = MAX_THREADS;
    std::thread t[MAX_THREADS];

    if (MAX_THREADS > MAX_STARTING_POPULATION){
        std::thread t[MAX_STARTING_POPULATION];
        num_threads = MAX_STARTING_POPULATION;
    }

    // update people if people died
    for (int i = 0; i < num_threads; i++) {
        int start = i * (MAX_STARTING_POPULATION/num_threads);
        int end = (i+1) * (MAX_STARTING_POPULATION/num_threads);
        t[i] = std::thread(die, people, variants, start, end, curr_day);
    }

    for (int i = 0; i < num_threads; i++) { t[i].join(); }

//    //clear board
//    for (int i = 0; i < num_threads; i++) {
//        int start = i * (BOARD_LENGTH * BOARD_WIDTH / num_threads);
//        int end = (i+1) * (BOARD_LENGTH * BOARD_WIDTH / num_threads);
//        if (i > MAX_STARTING_POPULATION){ break; }
//        t[i] = std::thread(clear_board, std::ref(board), start, end);
//    }

    // move all people
    for (int i = 0; i < num_threads; i++) {
        int start = i * (MAX_STARTING_POPULATION/num_threads);
        int end = (i+1) * (MAX_STARTING_POPULATION/num_threads);
        t[i] = std::thread(move, people, start, end);
    }

    for (int i = 0; i < num_threads; i++) { t[i].join(); }

    // infect people
    for (int i = 0; i < num_threads; i++) {
        int start = i * (MAX_STARTING_POPULATION/num_threads);
        int end = (i+1) * (MAX_STARTING_POPULATION/num_threads);
        t[i] = std::thread(infect, people, variants, start, end, curr_day);
    }
    for (int i = 0; i < num_threads; i++) { t[i].join(); }

}


// the main function
int disease_simulation(Person *people, Variant *variants, int end_day){ //end day is passed from config.end_day
  for (int i = 0; i < end_day; i++) {
    update_all_people(people, variants, i);
  }
  int max_var = int(max_variant);
  return max_var;
}

// ------------------------------------------------------------------------------------------

//COMMENTED OUT BELOW bc wasn't used and we can just check in each function that needs it
// we would have to call this on every thread anyway, so i don't think it would help like i thought

//function to get all the currently infected people
// for all people, check if they are currently infected
// store a list tuples of their positions (x,y)
// want to calculate only once per time period so doesn't need to be parallel?
// get after people move, so we have their current coordinates
//std::list<std::tuple<int, int>> get_infected(Person* people) {
//    //list<tuple<int, int>> infected;
//    //for (int i = 0; i < TOTAL_POPULATION; i++){
//    //    if (people[i].infected == true){
//    //        infected.push_back(make_tuple(people[i].x, people[i].y));
//    //}
//    //return infected;
//
//    std::list<std::tuple<int, int>> infected;
//
//    for (int i = 0; i < MAX_STARTING_POPULATION; i++) {
//        if (people[i].infected) { // Check if person is currently infected
//            infected.push_back(std::make_tuple(people[i].x.load(), people[i].y.load())); // Add position (x, y) as tuple
//        }
//    }
//
//    return infected;
//}

////clear before every move (won't happen do anything on first run, but that's ok)
//void clear_board(std::vector<std::vector<std::pair<int, int>>>& board, int start, int end){
//    for (int i = start; i < end; i++){
//        board[i].clear();
//    }
//}

// ------------------------------------------------------------------------------------------
