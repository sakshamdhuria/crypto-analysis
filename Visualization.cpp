#include "Visualization.h"
#include <fstream>
#include <iostream>
#include <string>

void plotLineGraph(const std::vector<double>& data, const std::string& xAxisLabel, const std::string& yAxisLabel, const std::string& outputGraphPath) {
    std::ofstream plotData("line_plot_data.csv");
    if (!plotData.is_open()) {
        std::cerr << "Failed to open file for plotting data." << std::endl;
        return;
    }

    for (size_t i = 0; i < data.size(); ++i) {
        plotData << i << "," << data[i] << "\n";
    }
    plotData.close();

    std::string command =
        "gnuplot -e \"set terminal png size 800,600; "
        "set output '" + outputGraphPath + "'; "
        "set title 'Line Plot'; "
        "set xlabel '" + xAxisLabel + "'; "
        "set ylabel '" + yAxisLabel + "'; "
        "set grid; "
        "plot 'line_plot_data.csv' using 1:2 with lines title 'Data'\"";
    system(command.c_str());
}

void plotACF(const std::vector<double>& acfValues, int dataSize, const std::string& outputGraphPath) {
    std::ofstream plotData("acf_plot_data.csv");
    if (!plotData.is_open()) {
        std::cerr << "Failed to open file for plotting data." << std::endl;
        return;
    }

    double threshold = 1.96 / std::sqrt(dataSize);

    plotData << "Lag,ACF,SignificancePos,SignificanceNeg\n";
    for (size_t lag = 0; lag < acfValues.size(); ++lag) {
        plotData << lag << "," << acfValues[lag] << "," << threshold << "," << -threshold << "\n";
    }
    plotData.close();

    std::string command =
        "gnuplot -e \"set terminal png size 800,600; "
        "set output '" + outputGraphPath + "'; "
        "set title 'Autocorrelation Function (ACF)'; "
        "set xlabel 'Lag'; "
        "set ylabel 'ACF'; "
        "set grid; "
        "plot 'acf_plot_data.csv' using 1:2 with lines title 'ACF', "
        "'acf_plot_data.csv' using 1:3 with lines title 'Significance Threshold', "
        "'acf_plot_data.csv' using 1:4 with lines title 'Significance Threshold (Negative)'\"";
    system(command.c_str());
}
