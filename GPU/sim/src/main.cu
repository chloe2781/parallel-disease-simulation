#include <iostream>
#include "config.h"
#include "simulation.cuh"

int main()
{
    std::cout << "Hello from CPU!" << std::endl;
    
    int* h_pos_x = new int[POPULATION];
    int* h_pos_y = new int[POPULATION];

    printf("Value: %d\n", -3 % 255);

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