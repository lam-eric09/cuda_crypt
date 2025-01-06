#include <iostream>
#include <cuda_runtime.h>

int main(){
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0){
        std::cout << "No CUDA devices found" << std::endl;
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout << "Device name: " << prop.name << std::endl;
    std::cout << "Number of SM (multiprocessors): " << prop.multiProcessorCount << std::endl;
    std::cout << "Max blocks per SM: " << prop.maxBlocksPerMultiProcessor << std::endl;
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Max threads per SM: " << prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "Warp: " << prop.warpSize << std::endl;
    std::cout << "VRAM: " << prop.totalGlobalMem << std::endl;
    return 0;
}