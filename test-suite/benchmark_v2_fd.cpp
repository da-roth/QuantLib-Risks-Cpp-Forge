/*******************************************************************************
 *
 *  QuantLib-Risks Swaption Benchmark v2 - FD Runner
 *
 *  Finite Differences benchmark using plain double QuantLib.
 *  This executable is compiled WITHOUT XAD to ensure fair FD comparison.
 *
 *  Usage:
 *    ./benchmark_fd_v2 [--small|--large|--both] [--quick]
 *
 *  Output format is designed to be parsed and combined with AAD results.
 *
 *  Copyright (C) 2025 Xcelerit Computing Limited
 *  SPDX-License-Identifier: AGPL-3.0-or-later
 *
 ******************************************************************************/

#include "benchmark_v2_common.hpp"
#include "benchmark_v2_pricing.hpp"

#include <chrono>
#include <cstring>
#include <fstream>

using namespace benchmark_v2;
using Clock = std::chrono::high_resolution_clock;
using DurationMs = std::chrono::duration<double, std::milli>;

// ============================================================================
// FD Benchmark Runner
// ============================================================================

std::vector<TimingResult> runFDBenchmark(const BenchmarkConfig& config, bool quickMode)
{
    std::vector<TimingResult> results;

    // Setup LMM (pre-compute grid, randoms, etc.)
    LMMSetup setup(config);

    std::cout << "================================================================================\n";
    std::cout << "  RUNNING FD BENCHMARKS\n";
    std::cout << "================================================================================\n";
    std::cout << "\n";

    for (size_t tc = 0; tc < config.pathCounts.size(); ++tc)
    {
        int paths = config.pathCounts[tc];
        Size nrTrails = static_cast<Size>(paths);

        TimingResult result;
        result.pathCount = paths;

        // Only run FD for small path counts
        if (paths > FD_MAX_PATHS)
        {
            std::cout << "  [" << (tc + 1) << "/" << config.pathCounts.size() << "] "
                      << formatPathCount(paths) << " paths - SKIPPED (paths > " << FD_MAX_PATHS << ")\n";
            results.push_back(result);
            continue;
        }

        std::cout << "  [" << (tc + 1) << "/" << config.pathCounts.size() << "] "
                  << formatPathCount(paths) << " paths " << std::flush;

        std::vector<double> fd_times;
        double eps = 1e-5;

        size_t warmup = quickMode ? 1 : config.warmupIterations;
        size_t bench = quickMode ? 2 : config.benchmarkIterations;

        for (size_t iter = 0; iter < warmup + bench; ++iter)
        {
            auto t_start = Clock::now();

            // Base rates as plain double
            std::vector<double> baseDepo(config.depoRates.begin(), config.depoRates.end());
            std::vector<double> baseSwap(config.swapRates.begin(), config.swapRates.end());

            // Compute base price
            double basePrice = priceSwaption<double>(config, setup, baseDepo, baseSwap, nrTrails);

            // Compute FD sensitivities (bump each rate)
            std::vector<double> derivatives(config.numMarketQuotes());
            for (Size q = 0; q < config.numMarketQuotes(); ++q)
            {
                std::vector<double> bumpedDepo = baseDepo;
                std::vector<double> bumpedSwap = baseSwap;

                if (q < config.numDeposits)
                    bumpedDepo[q] += eps;
                else
                    bumpedSwap[q - config.numDeposits] += eps;

                double bumpedPrice = priceSwaption<double>(config, setup, bumpedDepo, bumpedSwap, nrTrails);
                derivatives[q] = (bumpedPrice - basePrice) / eps;
            }

            auto t_end = Clock::now();

            if (iter >= warmup)
            {
                fd_times.push_back(DurationMs(t_end - t_start).count());
            }

            // Suppress unused warnings
            (void)basePrice;
            (void)derivatives;
        }

        result.fd_mean = computeMean(fd_times);
        result.fd_std = computeStddev(fd_times);
        result.fd_enabled = true;

        std::cout << "done (" << std::fixed << std::setprecision(1)
                  << result.fd_mean << " ms)\n";

        results.push_back(result);
    }

    std::cout << "\n";
    return results;
}

// ============================================================================
// Output Results in Machine-Parseable Format
// ============================================================================

void outputResultsForParsing(const std::vector<TimingResult>& results,
                              const std::string& benchmarkName)
{
    // Output format: FD_BENCHMARK_NAME:paths=mean,std;paths=mean,std;...
    std::cout << "FD_" << benchmarkName << ":";
    for (size_t i = 0; i < results.size(); ++i)
    {
        const auto& r = results[i];
        if (i > 0) std::cout << ";";
        std::cout << r.pathCount << "=" << r.fd_mean << "," << r.fd_std
                  << "," << (r.fd_enabled ? "1" : "0");
    }
    std::cout << std::endl;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[])
{
    bool runSmall = true;
    bool runLarge = true;
    bool quickMode = false;

    // Parse arguments
    for (int i = 1; i < argc; ++i)
    {
        if (strcmp(argv[i], "--small") == 0)
        {
            runSmall = true;
            runLarge = false;
        }
        else if (strcmp(argv[i], "--large") == 0)
        {
            runSmall = false;
            runLarge = true;
        }
        else if (strcmp(argv[i], "--both") == 0)
        {
            runSmall = true;
            runLarge = true;
        }
        else if (strcmp(argv[i], "--quick") == 0)
        {
            quickMode = true;
        }
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0)
        {
            std::cout << "Usage: " << argv[0] << " [options]\n";
            std::cout << "Options:\n";
            std::cout << "  --small   Run only small swaption (1Y into 1Y)\n";
            std::cout << "  --large   Run only large swaption (5Y into 5Y)\n";
            std::cout << "  --both    Run both benchmarks (default)\n";
            std::cout << "  --quick   Quick mode (fewer iterations)\n";
            std::cout << "  --help    Show this message\n";
            return 0;
        }
    }

    printHeader();
    printEnvironment();

    if (runSmall)
    {
        BenchmarkConfig smallConfig;
        printBenchmarkHeader(smallConfig, 1);

        auto results = runFDBenchmark(smallConfig, quickMode);
        outputResultsForParsing(results, "SMALL");
    }

    if (runLarge)
    {
        BenchmarkConfig largeConfig;
        largeConfig.setLargeConfig();
        printBenchmarkHeader(largeConfig, 2);

        auto results = runFDBenchmark(largeConfig, quickMode);
        outputResultsForParsing(results, "LARGE");
    }

    std::cout << "================================================================================\n";
    std::cout << "  FD benchmarks complete.\n";

    return 0;
}
