#include <iostream>
#include "config.h"
#include "simulation.cuh"
#include <cstdint>

int main()
{
    std::cout << "Hello from CPU!" << std::endl;

    uint8_t* h_pos_x = new uint8_t[POPULATION];
    uint8_t* h_pos_y = new uint8_t[POPULATION];

    for(int i = 0; i < POPULATION; i++)
    {
        h_pos_x[i] = 0;
        h_pos_y[i] = 0;
    }

    //run simulation() with h_pos_x and h_pos_y
    simulation(h_pos_x, h_pos_y);

    // printf("Positions: \n");
    // for(int i = 0; i < POPULATION; i++)
    // {
    //     printf("Index: %d -- x: %d, y: %d\n", i, h_pos_x[i], h_pos_y[i]);
    // }
    
}