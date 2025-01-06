#include <cstdlib>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#define N 200000000

__global__ void vector_add(float *out, float *a, float *b){
    int i = threadIdx.x;
    out[i] = a[i] + b[i];
}

int run_one_iter(int current_iter){
    float *a, *b, *out;
    // allocate PINNED memory on host
    if(current_iter == 0){
        cudaError_t err1 = cudaMallocHost((void**)&a, sizeof(float)*N);
        cudaError_t err2 = cudaMallocHost((void**)&b, sizeof(float)*N);
        cudaError_t err3 = cudaMallocHost((void**)&out, sizeof(float)*N);
    }
   
    // initialize arrays
    for(int i=0; i<N; i++){
        a[i] = i * 1.0f;
        b[i] = i * 2.0f;
    }

    float *d_a, *d_b, *d_out;
    // allocate memory on device
    cudaMalloc((void**)&d_a, sizeof(float)*N);
    cudaMalloc((void**)&d_b, sizeof(float)*N);
    cudaMalloc((void**)&d_out, sizeof(float)*N);

    // transfert data to it
    cudaEvent_t start_cpy, end_cpy, start_kernel, end_kernel;
    cudaEventCreate(&start_cpy);
    cudaEventCreate(&end_cpy);
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&end_kernel);

    cudaEventRecord(start_cpy);
    cudaMemcpy(d_a, a, sizeof(float)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float)*N, cudaMemcpyHostToDevice);
    cudaEventRecord(end_cpy);
    cudaEventSynchronize(end_cpy);
    float millisec = 0;
    cudaEventElapsedTime(&millisec, start_cpy, end_cpy);
    printf("Copy execution time: %f ms\n", millisec);
    cudaEventDestroy(start_cpy); 
    cudaEventDestroy(end_cpy); 
    // Nvidia 960M has 5 SM, 32 blocks per SM, 2064 max threads per SM
    // so we use all blocks with 64 threads per block
    // warp size is 32, hence the number of threads

    cudaEventRecord(start_kernel);
    vector_add<<<160,64>>>(d_out, d_a, d_b);
    cudaEventRecord(end_kernel);
    cudaEventSynchronize(end_kernel);
    float millisec_k = 0;
    cudaEventElapsedTime(&millisec_k, start_kernel, end_kernel);
    printf("Kernel execution time: %f ms\n", millisec_k);
    cudaEventDestroy(start_kernel); 
    cudaEventDestroy(end_kernel); 

    cudaMemcpy(out, d_out, sizeof(float)*N, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    if(current_iter == 9){
        cudaFreeHost(a);
        cudaFreeHost(b);
        cudaFreeHost(out);
    }
   return 0;    
}

int main(){
    int n = 10;
    for(int i=0; i<n; i++){
        run_one_iter(i);
    }
    return 0;
}