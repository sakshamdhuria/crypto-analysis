#ifndef MONTE_CARLO_SIMULATION_H
#define MONTE_CARLO_SIMULATION_H

#include <vector>

// Function prototypes
std::vector<std::vector<double>> monteCarloSimulation(
    double initialPrice, const std::vector<double>& volatility, int numPaths);

#endif // MONTE_CARLO_SIMULATION_H
