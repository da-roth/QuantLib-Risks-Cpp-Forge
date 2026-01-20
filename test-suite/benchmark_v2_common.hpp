/*******************************************************************************
 *
 *  QuantLib-Risks Swaption Benchmark v2 - Common Header
 *
 *  Shared utilities, configuration, and output formatting for the benchmark.
 *
 *  Copyright (C) 2025 Xcelerit Computing Limited
 *  SPDX-License-Identifier: AGPL-3.0-or-later
 *
 ******************************************************************************/

#ifndef BENCHMARK_V2_COMMON_HPP
#define BENCHMARK_V2_COMMON_HPP

#include "PlatformInfo.hpp"

// QuantLib includes
#include <ql/quantlib.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace benchmark_v2 {

using namespace QuantLib;

// ============================================================================
// Statistics Helpers
// ============================================================================

inline double computeMean(const std::vector<double>& v)
{
    if (v.empty()) return 0.0;
    return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

inline double computeStddev(const std::vector<double>& v)
{
    if (v.size() <= 1) return 0.0;
    double m = computeMean(v);
    double sq_sum = 0.0;
    for (double x : v)
    {
        double diff = x - m;
        sq_sum += diff * diff;
    }
    return std::sqrt(sq_sum / static_cast<double>(v.size() - 1));
}

// ============================================================================
// Benchmark Configuration
// ============================================================================

// Maximum paths for finite differences (FD is O(n) per path for n parameters)
constexpr int FD_MAX_PATHS = 1000;

struct BenchmarkConfig
{
    // Market data
    Size numDeposits = 4;
    Size numSwaps = 5;
    std::vector<Period> depoTenors;
    std::vector<Period> swapTenors;
    std::vector<double> depoRates;
    std::vector<double> swapRates;

    // LMM parameters
    Size size = 10;       // Number of forward rates
    Size i_opt = 2;       // Option exercise index
    Size j_opt = 2;       // Swap length
    Size steps = 8;       // Time steps

    // Curve end date
    int curveEndYears = 6;

    // Test configuration
    std::vector<int> pathCounts;
    size_t warmupIterations = 2;
    size_t benchmarkIterations = 5;

    // Instrument description
    std::string instrumentDesc = "European swaption (1Y into 1Y)";
    std::string benchmarkName = "Small Swaption (1Y into 1Y)";

    BenchmarkConfig()
    {
        depoTenors = {1 * Days, 1 * Months, 3 * Months, 6 * Months};
        swapTenors = {1 * Years, 2 * Years, 3 * Years, 4 * Years, 5 * Years};
        depoRates = {0.0350, 0.0365, 0.0380, 0.0400};
        swapRates = {0.0420, 0.0480, 0.0520, 0.0550, 0.0575};
        pathCounts = {10, 100, 1000, 10000, 100000};
    }

    // Configure for larger swaption (5Y into 5Y)
    void setLargeConfig()
    {
        numDeposits = 4;
        numSwaps = 10;
        depoTenors = {1 * Days, 1 * Months, 3 * Months, 6 * Months};
        swapTenors = {1 * Years, 2 * Years, 3 * Years, 4 * Years, 5 * Years,
                      6 * Years, 7 * Years, 8 * Years, 9 * Years, 10 * Years};
        depoRates = {0.0320, 0.0335, 0.0355, 0.0375};
        swapRates = {0.0400, 0.0435, 0.0460, 0.0480, 0.0495,
                     0.0505, 0.0515, 0.0522, 0.0528, 0.0532};

        size = 20;
        i_opt = 10;
        j_opt = 10;
        steps = 20;

        curveEndYears = 12;

        instrumentDesc = "European swaption (5Y into 5Y)";
        benchmarkName = "Large Swaption (5Y into 5Y)";
    }

    Size numMarketQuotes() const { return numDeposits + numSwaps; }
};

// ============================================================================
// Timing Results Structure
// ============================================================================

struct TimingResult
{
    int pathCount = 0;
    double fd_mean = 0, fd_std = 0;           // Finite differences (bump-and-revalue)
    double xad_mean = 0, xad_std = 0;         // XAD tape-based AAD
    double jit_mean = 0, jit_std = 0;         // Forge JIT scalar
    double jit_avx_mean = 0, jit_avx_std = 0; // Forge JIT AVX2
    bool fd_enabled = false;                  // Whether FD was run for this path count
    bool xad_enabled = false;
    bool jit_enabled = false;
    bool jit_avx_enabled = false;
};

// ============================================================================
// Output Formatting
// ============================================================================

inline std::string formatPathCount(int paths)
{
    if (paths >= 1000)
        return std::to_string(paths / 1000) + "K";
    return std::to_string(paths);
}

inline void printHeader()
{
    std::cout << "================================================================================\n";
    std::cout << "  QuantLib-Risks Swaption Benchmark\n";
    std::cout << "================================================================================\n";
    std::cout << "\n";
}

inline void printEnvironment()
{
    std::cout << "  ENVIRONMENT\n";
    std::cout << "--------------------------------------------------------------------------------\n";
    std::cout << "  Platform:     " << platform_info::getPlatformInfo() << "\n";
    std::cout << "  CPU:          " << platform_info::getCpuInfo() << "\n";
    std::cout << "  RAM:          " << platform_info::getMemoryInfo() << "\n";
    std::cout << "  SIMD:         " << platform_info::getSimdInfo() << "\n";
    std::cout << "  Compiler:     " << platform_info::getCompilerInfo() << "\n";
    std::cout << "\n";
}

inline void printBenchmarkHeader(const BenchmarkConfig& config, int benchmarkNum)
{
    std::cout << "================================================================================\n";
    std::cout << "  BENCHMARK " << benchmarkNum << ": " << config.benchmarkName << "\n";
    std::cout << "================================================================================\n";
    std::cout << "\n";

    std::cout << "  INSTRUMENT\n";
    std::cout << "--------------------------------------------------------------------------------\n";
    std::cout << "  Instrument:   " << config.instrumentDesc << "\n";
    std::cout << "  Model:        LIBOR Market Model (LMM)\n";
    std::cout << "  Forward rates:" << config.size << " (semi-annual)\n";
    std::cout << "  Time steps:   " << config.steps << "\n";
    std::cout << "  Inputs:       " << config.numMarketQuotes() << " market quotes\n";
    std::cout << "\n";

    std::cout << "  BENCHMARK CONFIGURATION\n";
    std::cout << "--------------------------------------------------------------------------------\n";
    std::cout << "  Path counts:  ";
    for (size_t i = 0; i < config.pathCounts.size(); ++i)
    {
        if (i > 0) std::cout << ", ";
        std::cout << formatPathCount(config.pathCounts[i]);
    }
    std::cout << "\n";
    std::cout << "  Warmup:       " << config.warmupIterations << " iterations\n";
    std::cout << "  Measured:     " << config.benchmarkIterations << " iterations\n";
    std::cout << "\n";

    std::cout << "  METHODS\n";
    std::cout << "--------------------------------------------------------------------------------\n";
    std::cout << "  FD       Finite Differences (bump-and-revalue, paths <= " << FD_MAX_PATHS << " only)\n";
    std::cout << "  XAD      XAD tape-based reverse-mode AAD\n";
    std::cout << "  JIT      Forge JIT-compiled native code\n";
    std::cout << "  JIT-AVX  Forge JIT + AVX2 SIMD (4 paths/instruction)\n";
    std::cout << "\n";
}

inline void printResultsTable(const std::vector<TimingResult>& results)
{
    std::cout << "================================================================================\n";
    std::cout << "  RESULTS (mean +/- stddev, in ms)\n";
    std::cout << "================================================================================\n";
    std::cout << "\n";

    // Table header
    std::cout << "| Paths  |    Method |     Mean |   StdDev | Speedup |\n";
    std::cout << "|-------:|----------:|---------:|---------:|--------:|\n";

    for (size_t i = 0; i < results.size(); ++i)
    {
        const auto& r = results[i];
        std::string pathStr = formatPathCount(r.pathCount);

        // Determine baseline for speedup calculation
        double baseline = r.fd_enabled ? r.fd_mean : r.xad_mean;
        bool first = true;

        auto printRow = [&](const std::string& method, double mean, double stddev, bool enabled)
        {
            if (!enabled) return;

            std::cout << "| " << std::setw(6);
            if (first)
            {
                std::cout << pathStr;
                first = false;
            }
            else
            {
                std::cout << "";
            }

            std::cout << " | " << std::setw(9) << method;
            std::cout << " | " << std::setw(8) << std::fixed << std::setprecision(1) << mean;
            std::cout << " | " << std::setw(8) << std::fixed << std::setprecision(1) << stddev;

            if (mean == baseline)
            {
                std::cout << " |     --- |";
            }
            else
            {
                double speedup = baseline / mean;
                std::cout << " | " << std::setw(6) << std::fixed << std::setprecision(2) << speedup << "x |";
            }
            std::cout << "\n";
        };

        printRow("FD", r.fd_mean, r.fd_std, r.fd_enabled);
        printRow("XAD", r.xad_mean, r.xad_std, r.xad_enabled);
        printRow("JIT", r.jit_mean, r.jit_std, r.jit_enabled);
        printRow("JIT-AVX", r.jit_avx_mean, r.jit_avx_std, r.jit_avx_enabled);

        // Separator between path count groups
        if (i < results.size() - 1)
        {
            std::cout << "|--------+-----------+----------+----------+---------|\n";
        }
    }

    std::cout << "\n";
    std::cout << "  Speedup = FD / Method (or XAD / Method when FD not available). All times in ms.\n";
    std::cout << "  FD only runs for paths <= " << FD_MAX_PATHS << " due to O(n) cost per path.\n";
    std::cout << "\n";
}

inline void printFooter()
{
    std::cout << "================================================================================\n";
    std::cout << "  All benchmarks complete.\n";
}

} // namespace benchmark_v2

#endif // BENCHMARK_V2_COMMON_HPP
