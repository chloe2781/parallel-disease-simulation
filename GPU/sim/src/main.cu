#include <iostream>

__global__ void hello_from_gpu()
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
        printf("Hello from GPU!\n");
}

int main()
{
    std::cout << "Hello from CPU!" << std::endl;

    hello_from_gpu<<<1, 1>>>();

    cudaDeviceSynchronize();
    return 0;
}