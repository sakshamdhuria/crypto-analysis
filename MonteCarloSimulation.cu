#include "MonteCarloSimulation.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <vector>

// CUDA error checking macro
#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
              << cudaGetErrorString(x) << std::endl; \
    exit(EXIT_FAILURE); }} while(0)

// CUDA kernel: Initialize random number generator states
__global__ void setupRNGKernel(curandState *states, unsigned long seed, int numPaths) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPaths) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

// CUDA kernel: Monte Carlo simulation
__global__ void monteCarloKernel(
    double* d_paths,
    const double* d_volatility,
    curandState* states,
    double initialPrice,
    int numPaths,
    int steps) {
    int pathIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (pathIdx < numPaths) {
        curandState localState = states[pathIdx];
        double price = initialPrice;

        for (int t = 0; t < steps; ++t) {
            double randShock = curand_normal_double(&localState); // Generate random shock
            double volatility = d_volatility[t]; // Get volatility for the step
            double returnRate = randShock * sqrt(volatility);
            price *= exp(returnRate); // Update price
            d_paths[pathIdx * steps + t] = price; // Store the result
        }
    }
}

// Monte Carlo simulation function
std::vector<std::vector<double>> monteCarloSimulation(
    double initialPrice, const std::vector<double>& volatility, int numPaths) {
    int steps = volatility.size();

    // Allocate host memory
    std::vector<std::vector<double>> paths(numPaths, std::vector<double>(steps, initialPrice));
    std::vector<double> flattenedPaths(numPaths * steps);
    std::vector<double> h_volatility = volatility;

    // Allocate device memory
    double *d_paths, *d_volatility;
    curandState *d_states;
    CUDA_CALL(cudaMalloc(&d_paths, numPaths * steps * sizeof(double)));
    CUDA_CALL(cudaMalloc(&d_volatility, steps * sizeof(double)));
    CUDA_CALL(cudaMalloc(&d_states, numPaths * sizeof(curandState)));

    // Copy data to the device
    CUDA_CALL(cudaMemcpy(d_volatility, h_volatility.data(), steps * sizeof(double), cudaMemcpyHostToDevice));

    // Configure CUDA kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numPaths + threadsPerBlock - 1) / threadsPerBlock;

    // Setup RNG states
    setupRNGKernel<<<blocksPerGrid, threadsPerBlock>>>(d_states, 12345, numPaths);
    CUDA_CALL(cudaDeviceSynchronize());

    // Launch the Monte Carlo kernel
    monteCarloKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_paths, d_volatility, d_states, initialPrice, numPaths, steps);
    CUDA_CALL(cudaDeviceSynchronize());

    // Copy results back to the host
    CUDA_CALL(cudaMemcpy(flattenedPaths.data(), d_paths, numPaths * steps * sizeof(double), cudaMemcpyDeviceToHost));

    // Unflatten the paths for easy access
    for (int p = 0; p < numPaths; ++p) {
        for (int t = 0; t < steps; ++t) {
            paths[p][t] = flattenedPaths[p * steps + t];
        }
    }

    // Free device memory
    CUDA_CALL(cudaFree(d_paths));
    CUDA_CALL(cudaFree(d_volatility));
    CUDA_CALL(cudaFree(d_states));

    return paths;
}
