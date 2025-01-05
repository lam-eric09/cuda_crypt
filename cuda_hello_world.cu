#include <stdio.h>

__global__ void kernel_hello_world(){
    printf("Hello World !\n");
}

int main() {
    kernel_hello_world<<<1,1>>>();
    cudaDeviceSynchronize();
    return 0;
}