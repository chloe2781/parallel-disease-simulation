# PS1: Population Dynamics Simulation (10 Points)

## Problem Statement
In this assignment you are going to build a simulation for population dynamics using C++ in a multithreaded environment. We have provided some starter code for you, and your job is to fill in the missing code as specified by the `#TODO#` blocks in the code. You can either just work in the ipynb OR you can work locally with the various files in this folder.

## Simulation Description
The world is split into `NUM_REGIONS` different regions. Each region is filled with `COMMUNITIES_PER_NUM_SPECIES` different communities of each of the `NUM_SPECIES` different species. Each of the communities is intialized with this information and a `population` and a `growth_rate` which is all packed into a `Species_Community` struct. Note that the struct also contains additional variables which you may or may not need to use depending upon your implementation.

```
typedef struct {
    std::atomic<int> population;        // the population of a speciies
    std::atomic<int> food_collected;    // the food collected in the current time period
    int region_of_world;                // region of this species community
    int species_type;                   // type of species for this species community
    float growth_rate;                  // growth_rate for this species community
    bool flag;                          // flag in case helpful to have one (you may not need this)
    std::atomic<int> atomic_helper;     // atomic in case helpful to have one (you may not need this)
    std::atomic<float> atomic_helper2;  // atomic in case helpful to have one (you may not need this)
    std::mutex mtx;                     // mutex in case helpful to have one (you may not need this)
} Species_Community;
```

For `NUM_TIMEPERIODS` the simulation runs. At each time period all of the members of each species calls the `food_oracle` inorder for everyone to `gather_all_food`. After all food is collected we can `update_all_populations` based on the amount of food gathered. In order to do so we need to first `compute_local_population_share` which is the percentage of all species WITHIN A SINGLE REGION that belong to a given species. We can then use that to `update_community_population` for each community of each species based on 3 rules as listed in later sections of this document.

When the simulation is done it prints out the populations of the various communities of species.

## Functions You'll Need To Implement
All functions you need to implement are in `util.h` and that is the only file you need to submit to gradescope!

First we'll implement helper functions:

`update_community_population` (1.5 points)

For a given community, update its population based on the input `local_population_share` and the following three rules:
1. The change in population for a community is directly proportional to its growth_rate and local_population_share
2. If it has collected enough food to feed the population it grows, else it shrinks
3. If the population drops below 5 it goes extinct

`compute_local_population_share` (1.5 points)

Returns the population share for a given community. Population share is defined as the percentage of population in a region that is a given species across all communities of all species.

Then we'll implement the overall population update step:

`update_all_populations` (3 points)
Updates the population for all communities of all species. Some quick hints/notes:
1. You will want to launch MAX_THREADS to compute this
2. You will need to use compute_local_population_share and update_community_population
3. Make sure your logic is thread safe!
4. Feel free to use helper functions if that makes your life easier!

Next we'll implement the food gathering step
`gather_all_food` (3 points)

Each simualtion step we reset the food counts to 0 and then each member of the population tries to collect food using the food_oracle(). **Here we are adding a twist of difficulty. You must use MAX_THREADS threads of parallelism to compute the updated food count for each community of each species -- not spread across the various communities.** While this isn't the naturally most efficient way to code it up, it will make you have to use some amount of synchronization!

Finally we'll implement the main function
`population_dynamics` (1 point)

Here you'll just need to make sure to gather food and update the population for all NUM_TIME_PERIODS

## Submission
Once you are done, download and submit (or just submit if you are working locally) your `util.h` file to **Courseworks**! As we do not have an autograder we can't use Gradescope.

## Notes and Hints
+ **DO NOT CHANGE FUNCTION DEFINITIONS** or you will break our grading scripts
+ See the syllabus for our course collaboration policy (long story short you are welcome to collaborate at a high level but please do not copy each others code).
+ If you are working in Colab, you can change the formatting of the code to different color schemes: just change the `%%cpp -n <filename>.h -s xcode` to a different `-s` flag. The list can be [found at this link](https://gist.github.com/akshaykhadse/7acc91dd41f52944c6150754e5530c4b).
+ If you are running the code locally, note that you can compile and execute with the run.sh script. If you get errors of the form `error: no matching constructor for initialization of 'std::thread'` when you attempt to do this (likely to happen if you are on Mac), you can add the extra compile flag `-std=c++11` and it should resolve the issue.
+ Please reach out on Slack with any and all questions!