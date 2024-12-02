#ifndef VISUALIZATION_H
#define VISUALIZATION_H

#include <vector>
#include <string>
#include <cmath>

// Function prototypes
void plotLineGraph(const std::vector<double>& data, const std::string& xAxisLabel, const std::string& yAxisLabel, const std::string& outputGraphPath);
void plotACF(const std::vector<double>& acfValues, int dataSize, const std::string& outputGraphPath);

#endif // VISUALIZATION_H
