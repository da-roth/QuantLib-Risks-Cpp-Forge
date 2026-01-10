/*******************************************************************************

   QuantLib-Risks Swaption Benchmark - Standalone Executable

   This benchmark compares different approaches for computing sensitivities
   in Monte Carlo pricing of swaptions using the LIBOR Market Model.

   APPROACHES TESTED:
     XAD     - XAD tape-based reverse-mode AAD
     JIT     - Forge JIT-compiled native code (scalar)
     JIT-AVX - Forge JIT + AVX2 SIMD (4 paths/instruction)

   Usage:
     ./quantlib-risks-benchmark-standalone [options]

   Options:
     --help, -h        Show this help message
     --quick           Run quick benchmark (fewer iterations, fewer path counts)
     --small-only      Run only the small swaption benchmark (1Y into 1Y)
     --large-only      Run only the large swaption benchmark (5Y into 5Y)
     --decomposition   Run performance decomposition analysis

   By default, runs both small (1Y into 1Y) and large (5Y into 5Y) benchmarks.

   Copyright (C) 2025 Xcelerit Computing Limited

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU Affero General Public License as published
   by the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

******************************************************************************/

#include "PlatformInfo.hpp"

// QuantLib includes
#include <ql/indexes/ibor/euribor.hpp>
#include <ql/instruments/vanillaswap.hpp>
#include <ql/pricingengines/swap/discountingswapengine.hpp>
#include <ql/termstructures/yield/piecewiseyieldcurve.hpp>
#include <ql/termstructures/yield/ratehelpers.hpp>
#include <ql/termstructures/yield/zerocurve.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/daycounters/actual360.hpp>
#include <ql/time/daycounters/thirty360.hpp>

// LMM Monte Carlo includes
#include <ql/legacy/libormarketmodels/lfmcovarproxy.hpp>
#include <ql/legacy/libormarketmodels/liborforwardmodel.hpp>
#include <ql/legacy/libormarketmodels/lmexpcorrmodel.hpp>
#include <ql/legacy/libormarketmodels/lmlinexpvolmodel.hpp>
#include <ql/math/randomnumbers/rngtraits.hpp>
#include <ql/methods/montecarlo/multipathgenerator.hpp>

// XAD includes
#include <XAD/XAD.hpp>

// Forge JIT backends (conditionally included)
#if defined(QLRISKS_HAS_FORGE)
#include <xad-forge/ForgeBackend.hpp>
#include <xad-forge/ForgeBackendAVX.hpp>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using namespace QuantLib;
using platform_info::getCompilerInfo;
using platform_info::getCpuInfo;
using platform_info::getMemoryInfo;
using platform_info::getPlatformInfo;
using platform_info::getSimdInfo;

// Overload value() for plain double (for consistency with utilities_xad.hpp)
inline double value(double x) { return x; }

// ============================================================================
// XAD Type Definitions
// ============================================================================

// Note: QuantLib's Real is already defined as xad::AReal<double> via qlrisks.hpp
// (via #define QL_REAL xad::AReal<double> in qlrisks.hpp)
// We use QuantLib::Real directly through "using namespace QuantLib"
// The tape_type is accessed as Real::tape_type where needed

// ============================================================================
// Statistics Helpers
// ============================================================================

double computeMean(const std::vector<double>& v)
{
    if (v.empty()) return 0.0;
    return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

double computeStddev(const std::vector<double>& v)
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
// Helper: Create IborIndex with ZeroCurve
// ============================================================================

ext::shared_ptr<IborIndex> makeIndex(std::vector<Date> dates,
                                     const std::vector<Rate>& rates)
{
    DayCounter dayCounter = Actual360();
    RelinkableHandle<YieldTermStructure> termStructure;
    ext::shared_ptr<IborIndex> index(new Euribor6M(termStructure));

    Date todaysDate = index->fixingCalendar().adjust(Date(4, September, 2005));
    Settings::instance().evaluationDate() = todaysDate;

    dates[0] = index->fixingCalendar().advance(todaysDate,
                                                index->fixingDays(), Days);

    termStructure.linkTo(ext::shared_ptr<YieldTermStructure>(
        new ZeroCurve(dates, rates, dayCounter)));

    return index;
}

// ============================================================================
// Chain Rule Helper
// ============================================================================

inline void applyChainRule(const double* __restrict jacobian,
                           const double* __restrict derivatives,
                           double* __restrict result,
                           std::size_t numIntermediates,
                           std::size_t numInputs)
{
    for (std::size_t j = 0; j < numInputs; ++j)
        result[j] = 0.0;

    for (std::size_t i = 0; i < numIntermediates; ++i)
    {
        const double deriv_i = derivatives[i];
        const double* jac_row = jacobian + i * numInputs;
        for (std::size_t j = 0; j < numInputs; ++j)
        {
            result[j] += deriv_i * jac_row[j];
        }
    }
}

// ============================================================================
// Benchmark Configuration
// ============================================================================

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

    BenchmarkConfig()
    {
        depoTenors = {1 * Days, 1 * Months, 3 * Months, 6 * Months};
        swapTenors = {1 * Years, 2 * Years, 3 * Years, 4 * Years, 5 * Years};
        depoRates = {0.0350, 0.0365, 0.0380, 0.0400};
        swapRates = {0.0420, 0.0480, 0.0520, 0.0550, 0.0575};
        pathCounts = {10, 100, 1000, 10000};
    }

    // Configure for larger swaption (5Y into 5Y)
    void setLargeConfig()
    {
        // More market quotes for longer curve
        numDeposits = 4;
        numSwaps = 10;
        depoTenors = {1 * Days, 1 * Months, 3 * Months, 6 * Months};
        swapTenors = {1 * Years, 2 * Years, 3 * Years, 4 * Years, 5 * Years,
                      6 * Years, 7 * Years, 8 * Years, 9 * Years, 10 * Years};
        depoRates = {0.0320, 0.0335, 0.0355, 0.0375};
        swapRates = {0.0400, 0.0435, 0.0460, 0.0480, 0.0495,
                     0.0505, 0.0515, 0.0522, 0.0528, 0.0532};

        // Larger LMM: 20 forward rates (semi-annual to 10Y)
        size = 20;
        i_opt = 10;    // Option starts at 5Y (10 × 6 months)
        j_opt = 10;    // 5Y swap (10 × 6 months)
        steps = 20;    // More time steps for 5Y simulation

        curveEndYears = 12;  // Extra buffer for accrual periods

        instrumentDesc = "European swaption (5Y into 5Y)";
    }
};

// ============================================================================
// Timing Results Structure
// ============================================================================

struct TimingResult
{
    double xad_mean = 0, xad_std = 0;
    double jit_mean = 0, jit_std = 0;
    double jit_avx_mean = 0, jit_avx_std = 0;
};

// ============================================================================
// Main Benchmark Function
// ============================================================================

void runBenchmark(const BenchmarkConfig& config, bool quickMode)
{
    using Clock = std::chrono::high_resolution_clock;
    using Duration = std::chrono::duration<double, std::milli>;

    Calendar calendar = TARGET();
    Date todaysDate(4, September, 2005);
    Settings::instance().evaluationDate() = todaysDate;
    Integer fixingDays = 2;
    Date settlementDate = calendar.adjust(calendar.advance(todaysDate, fixingDays, Days));
    DayCounter dayCounter = Actual360();

    Size numMarketQuotes = config.numDeposits + config.numSwaps;

    // Build base curve (using first deposit rate and last swap rate as endpoints)
    std::vector<Rate> baseZeroRates = {config.depoRates[0], config.swapRates.back()};
    std::vector<Date> baseDates = {settlementDate, settlementDate + config.curveEndYears * Years};
    auto baseIndex = makeIndex(baseDates, baseZeroRates);

    ext::shared_ptr<LiborForwardModelProcess> baseProcess(
        new LiborForwardModelProcess(config.size, baseIndex));
    ext::shared_ptr<LmCorrelationModel> baseCorrModel(
        new LmExponentialCorrelationModel(config.size, 0.5));
    ext::shared_ptr<LmVolatilityModel> baseVolaModel(
        new LmLinearExponentialVolatilityModel(baseProcess->fixingTimes(),
                                               0.291, 1.483, 0.116, 0.00001));
    baseProcess->setCovarParam(ext::shared_ptr<LfmCovarianceParameterization>(
        new LfmCovarianceProxy(baseVolaModel, baseCorrModel)));

    // Grid setup
    std::vector<Time> fixingTimes = baseProcess->fixingTimes();
    TimeGrid grid(fixingTimes.begin(), fixingTimes.end(), config.steps);

    std::vector<Size> location;
    for (Size idx = 0; idx < fixingTimes.size(); ++idx)
    {
        location.push_back(
            std::find(grid.begin(), grid.end(), fixingTimes[idx]) - grid.begin());
    }

    Size numFactors = baseProcess->factors();
    Size exerciseStep = location[config.i_opt];
    Size fullGridSteps = grid.size() - 1;
    Size fullGridRandoms = fullGridSteps * numFactors;

    // Swap schedule
    BusinessDayConvention convention = baseIndex->businessDayConvention();
    Date fwdStart = settlementDate + Period(6 * config.i_opt, Months);
    Date fwdMaturity = fwdStart + Period(6 * config.j_opt, Months);
    Schedule schedule(fwdStart, fwdMaturity, baseIndex->tenor(), calendar,
                      convention, convention, DateGeneration::Forward, false);

    // Accrual periods
    std::vector<double> accrualStart(config.size), accrualEnd(config.size);
    for (Size k = 0; k < config.size; ++k)
    {
        accrualStart[k] = value(baseProcess->accrualStartTimes()[k]);
        accrualEnd[k] = value(baseProcess->accrualEndTimes()[k]);
    }

    Size numIntermediates = config.size + 1;

    // Pre-generate random numbers
    Size maxPaths = static_cast<Size>(*std::max_element(config.pathCounts.begin(), config.pathCounts.end()));
    std::cout << "  Generating " << maxPaths << " x " << fullGridRandoms << " random numbers..." << std::flush;

    typedef PseudoRandom::rsg_type rsg_type;
    rsg_type rsg_base = PseudoRandom::make_sequence_generator(fullGridRandoms, BigNatural(42));

    std::vector<std::vector<double>> allRandoms(maxPaths);
    for (Size n = 0; n < maxPaths; ++n)
    {
        allRandoms[n].resize(fullGridRandoms);
        const auto& seq = rsg_base.nextSequence();
        for (Size m = 0; m < fullGridRandoms; ++m)
        {
            allRandoms[n][m] = value(seq.value[m]);
        }
    }
    std::cout << " Done." << std::endl;

    // Results storage
    std::vector<TimingResult> results(config.pathCounts.size());

    // Run benchmarks for each path count
    for (size_t tc = 0; tc < config.pathCounts.size(); ++tc)
    {
        Size nrTrails = static_cast<Size>(config.pathCounts[tc]);

        std::cout << "  [" << (tc + 1) << "/" << config.pathCounts.size() << "] ";
        if (config.pathCounts[tc] >= 1000)
            std::cout << (config.pathCounts[tc] / 1000) << "K";
        else
            std::cout << config.pathCounts[tc];
        std::cout << " paths " << std::flush;

        std::vector<double> xad_times, jit_times, jit_avx_times;

        for (size_t iter = 0; iter < config.warmupIterations + config.benchmarkIterations; ++iter)
        {
            bool recordTiming = (iter >= config.warmupIterations);

            // =================================================================
            // XAD Tape-based approach (direct evolve)
            // =================================================================
            {
                auto t_start = Clock::now();

                using tape_type = Real::tape_type;
                tape_type tape;

                std::vector<Real> depositRates(config.numDeposits);
                std::vector<Real> swapRatesAD(config.numSwaps);
                for (Size idx = 0; idx < config.numDeposits; ++idx)
                    depositRates[idx] = config.depoRates[idx];
                for (Size idx = 0; idx < config.numSwaps; ++idx)
                    swapRatesAD[idx] = config.swapRates[idx];

                tape.registerInputs(depositRates);
                tape.registerInputs(swapRatesAD);
                tape.newRecording();

                // Build curve
                RelinkableHandle<YieldTermStructure> euriborTS;
                auto euribor6m = ext::make_shared<Euribor6M>(euriborTS);
                euribor6m->addFixing(Date(2, September, 2005), 0.04);

                std::vector<ext::shared_ptr<RateHelper>> instruments;
                for (Size idx = 0; idx < config.numDeposits; ++idx)
                {
                    auto depoQuote = ext::make_shared<SimpleQuote>(depositRates[idx]);
                    instruments.push_back(ext::make_shared<DepositRateHelper>(
                        Handle<Quote>(depoQuote), config.depoTenors[idx], fixingDays,
                        calendar, ModifiedFollowing, true, dayCounter));
                }
                for (Size idx = 0; idx < config.numSwaps; ++idx)
                {
                    auto swapQuote = ext::make_shared<SimpleQuote>(swapRatesAD[idx]);
                    instruments.push_back(ext::make_shared<SwapRateHelper>(
                        Handle<Quote>(swapQuote), config.swapTenors[idx],
                        calendar, Annual, Unadjusted, Thirty360(Thirty360::BondBasis),
                        euribor6m));
                }

                auto yieldCurve = ext::make_shared<PiecewiseYieldCurve<ZeroYield, Linear>>(
                    settlementDate, instruments, dayCounter);
                yieldCurve->enableExtrapolation();

                std::vector<Date> curveDates;
                std::vector<Real> zeroRates;
                curveDates.push_back(settlementDate);
                zeroRates.push_back(yieldCurve->zeroRate(settlementDate, dayCounter, Continuous).rate());
                Date endDate = settlementDate + config.curveEndYears * Years;
                curveDates.push_back(endDate);
                zeroRates.push_back(yieldCurve->zeroRate(endDate, dayCounter, Continuous).rate());

                std::vector<Rate> zeroRates_ql;
                for (const auto& r : zeroRates) zeroRates_ql.push_back(r);

                RelinkableHandle<YieldTermStructure> termStructure;
                ext::shared_ptr<IborIndex> index(new Euribor6M(termStructure));
                index->addFixing(Date(2, September, 2005), 0.04);
                termStructure.linkTo(ext::make_shared<ZeroCurve>(curveDates, zeroRates_ql, dayCounter));

                ext::shared_ptr<LiborForwardModelProcess> process(
                    new LiborForwardModelProcess(config.size, index));
                process->setCovarParam(ext::shared_ptr<LfmCovarianceParameterization>(
                    new LfmCovarianceProxy(
                        ext::make_shared<LmLinearExponentialVolatilityModel>(process->fixingTimes(), 0.291, 1.483, 0.116, 0.00001),
                        ext::make_shared<LmExponentialCorrelationModel>(config.size, 0.5))));

                ext::shared_ptr<VanillaSwap> fwdSwap(
                    new VanillaSwap(Swap::Receiver, 1.0,
                                    schedule, 0.05, dayCounter,
                                    schedule, index, 0.0, index->dayCounter()));
                fwdSwap->setPricingEngine(ext::make_shared<DiscountingSwapEngine>(
                    index->forwardingTermStructure()));
                Real swapRate = fwdSwap->fairRate();

                Array initRates = process->initialValues();

                // MC simulation
                Real price = Real(0.0);
                for (Size n = 0; n < nrTrails; ++n)
                {
                    Array asset(config.size);
                    for (Size k = 0; k < config.size; ++k) asset[k] = initRates[k];

                    Array assetAtExercise(config.size);
                    for (Size step = 1; step <= fullGridSteps; ++step)
                    {
                        Size offset = (step - 1) * numFactors;
                        Time t = grid[step - 1];
                        Time dt = grid.dt(step - 1);

                        Array dw(numFactors);
                        for (Size f = 0; f < numFactors; ++f)
                            dw[f] = allRandoms[n][offset + f];

                        asset = process->evolve(t, asset, dt, dw);

                        if (step == exerciseStep)
                        {
                            for (Size k = 0; k < config.size; ++k)
                                assetAtExercise[k] = asset[k];
                        }
                    }

                    // Discount factors
                    Array dis(config.size);
                    Real df = Real(1.0);
                    for (Size k = 0; k < config.size; ++k)
                    {
                        Real accrual = accrualEnd[k] - accrualStart[k];
                        df = df / (Real(1.0) + assetAtExercise[k] * accrual);
                        dis[k] = df;
                    }

                    // NPV
                    Real npv = Real(0.0);
                    for (Size m = config.i_opt; m < config.i_opt + config.j_opt; ++m)
                    {
                        Real accrual = accrualEnd[m] - accrualStart[m];
                        npv += (swapRate - assetAtExercise[m]) * accrual * dis[m];
                    }
                    price += max(npv, Real(0.0));
                }
                price /= Real(static_cast<double>(nrTrails));

                tape.registerOutput(price);
                derivative(price) = 1.0;
                tape.computeAdjoints();

                auto t_end = Clock::now();
                if (recordTiming)
                {
                    xad_times.push_back(Duration(t_end - t_start).count());
                }
                tape.deactivate();
            }

#if defined(QLRISKS_HAS_FORGE)
            // =================================================================
            // JIT Scalar approach (ForgeBackend)
            // =================================================================
            {
                auto t_start = Clock::now();

                using tape_type = Real::tape_type;
                tape_type tape;

                std::vector<Real> depositRates(config.numDeposits);
                std::vector<Real> swapRatesAD(config.numSwaps);
                for (Size idx = 0; idx < config.numDeposits; ++idx)
                    depositRates[idx] = config.depoRates[idx];
                for (Size idx = 0; idx < config.numSwaps; ++idx)
                    swapRatesAD[idx] = config.swapRates[idx];

                tape.registerInputs(depositRates);
                tape.registerInputs(swapRatesAD);
                tape.newRecording();

                // Build curve
                RelinkableHandle<YieldTermStructure> euriborTS;
                auto euribor6m = ext::make_shared<Euribor6M>(euriborTS);
                euribor6m->addFixing(Date(2, September, 2005), 0.04);

                std::vector<ext::shared_ptr<RateHelper>> instruments;
                for (Size idx = 0; idx < config.numDeposits; ++idx)
                {
                    auto depoQuote = ext::make_shared<SimpleQuote>(depositRates[idx]);
                    instruments.push_back(ext::make_shared<DepositRateHelper>(
                        Handle<Quote>(depoQuote), config.depoTenors[idx], fixingDays,
                        calendar, ModifiedFollowing, true, dayCounter));
                }
                for (Size idx = 0; idx < config.numSwaps; ++idx)
                {
                    auto swapQuote = ext::make_shared<SimpleQuote>(swapRatesAD[idx]);
                    instruments.push_back(ext::make_shared<SwapRateHelper>(
                        Handle<Quote>(swapQuote), config.swapTenors[idx],
                        calendar, Annual, Unadjusted, Thirty360(Thirty360::BondBasis),
                        euribor6m));
                }

                auto yieldCurve = ext::make_shared<PiecewiseYieldCurve<ZeroYield, Linear>>(
                    settlementDate, instruments, dayCounter);
                yieldCurve->enableExtrapolation();

                std::vector<Date> curveDates;
                std::vector<Real> zeroRates;
                curveDates.push_back(settlementDate);
                zeroRates.push_back(yieldCurve->zeroRate(settlementDate, dayCounter, Continuous).rate());
                Date endDate = settlementDate + config.curveEndYears * Years;
                curveDates.push_back(endDate);
                zeroRates.push_back(yieldCurve->zeroRate(endDate, dayCounter, Continuous).rate());

                std::vector<Rate> zeroRates_ql;
                for (const auto& r : zeroRates) zeroRates_ql.push_back(r);

                RelinkableHandle<YieldTermStructure> termStructure;
                ext::shared_ptr<IborIndex> index(new Euribor6M(termStructure));
                index->addFixing(Date(2, September, 2005), 0.04);
                termStructure.linkTo(ext::make_shared<ZeroCurve>(curveDates, zeroRates_ql, dayCounter));

                ext::shared_ptr<LiborForwardModelProcess> process(
                    new LiborForwardModelProcess(config.size, index));
                process->setCovarParam(ext::shared_ptr<LfmCovarianceParameterization>(
                    new LfmCovarianceProxy(
                        ext::make_shared<LmLinearExponentialVolatilityModel>(process->fixingTimes(), 0.291, 1.483, 0.116, 0.00001),
                        ext::make_shared<LmExponentialCorrelationModel>(config.size, 0.5))));

                ext::shared_ptr<VanillaSwap> fwdSwap(
                    new VanillaSwap(Swap::Receiver, 1.0,
                                    schedule, 0.05, dayCounter,
                                    schedule, index, 0.0, index->dayCounter()));
                fwdSwap->setPricingEngine(ext::make_shared<DiscountingSwapEngine>(
                    index->forwardingTermStructure()));
                Real swapRate_tape = fwdSwap->fairRate();

                Array initRates = process->initialValues();

                // Jacobian computation
                std::vector<double> jacobian(numIntermediates * numMarketQuotes, 0.0);
                for (Size k = 0; k < config.size; ++k)
                {
                    if (initRates[k].shouldRecord())
                    {
                        tape.clearDerivatives();
                        tape.registerOutput(initRates[k]);
                        derivative(initRates[k]) = 1.0;
                        tape.computeAdjoints();

                        double* jac_row = jacobian.data() + k * numMarketQuotes;
                        for (Size m = 0; m < config.numDeposits; ++m)
                            jac_row[m] = derivative(depositRates[m]);
                        for (Size m = 0; m < config.numSwaps; ++m)
                            jac_row[config.numDeposits + m] = derivative(swapRatesAD[m]);
                    }
                }
                if (swapRate_tape.shouldRecord())
                {
                    tape.clearDerivatives();
                    tape.registerOutput(swapRate_tape);
                    derivative(swapRate_tape) = 1.0;
                    tape.computeAdjoints();

                    double* jac_row = jacobian.data() + config.size * numMarketQuotes;
                    for (Size m = 0; m < config.numDeposits; ++m)
                        jac_row[m] = derivative(depositRates[m]);
                    for (Size m = 0; m < config.numSwaps; ++m)
                        jac_row[config.numDeposits + m] = derivative(swapRatesAD[m]);
                }
                tape.deactivate();

                // JIT kernel creation with ForgeBackend
                auto forgeBackend = std::make_unique<xad::forge::ForgeBackend<double>>(false);
                xad::JITCompiler<double> jit(std::move(forgeBackend));

                std::vector<xad::AD> jit_initRates(config.size);
                xad::AD jit_swapRate;
                std::vector<xad::AD> jit_randoms(fullGridRandoms);

                for (Size k = 0; k < config.size; ++k)
                {
                    jit_initRates[k] = xad::AD(value(initRates[k]));
                    jit.registerInput(jit_initRates[k]);
                }
                jit_swapRate = xad::AD(value(swapRate_tape));
                jit.registerInput(jit_swapRate);
                for (Size m = 0; m < fullGridRandoms; ++m)
                {
                    jit_randoms[m] = xad::AD(0.0);
                    jit.registerInput(jit_randoms[m]);
                }

                jit.newRecording();

                std::vector<xad::AD> asset_jit(config.size);
                std::vector<xad::AD> assetAtExercise_jit(config.size);
                for (Size k = 0; k < config.size; ++k) asset_jit[k] = jit_initRates[k];

                for (Size step = 1; step <= fullGridSteps; ++step)
                {
                    Size offset = (step - 1) * numFactors;
                    Time t = grid[step - 1];
                    Time dt = grid.dt(step - 1);

                    Array dw(numFactors);
                    for (Size f = 0; f < numFactors; ++f)
                        dw[f] = jit_randoms[offset + f];

                    Array asset_arr(config.size);
                    for (Size k = 0; k < config.size; ++k) asset_arr[k] = asset_jit[k];

                    Array evolved = process->evolve(t, asset_arr, dt, dw);
                    for (Size k = 0; k < config.size; ++k) asset_jit[k] = evolved[k];

                    if (step == exerciseStep)
                    {
                        for (Size k = 0; k < config.size; ++k)
                            assetAtExercise_jit[k] = asset_jit[k];
                    }
                }

                std::vector<xad::AD> dis_jit(config.size);
                xad::AD df_jit = xad::AD(1.0);
                for (Size k = 0; k < config.size; ++k)
                {
                    double accrual = accrualEnd[k] - accrualStart[k];
                    df_jit = df_jit / (xad::AD(1.0) + assetAtExercise_jit[k] * accrual);
                    dis_jit[k] = df_jit;
                }

                xad::AD jit_npv = xad::AD(0.0);
                for (Size m = config.i_opt; m < config.i_opt + config.j_opt; ++m)
                {
                    double accrual = accrualEnd[m] - accrualStart[m];
                    jit_npv = jit_npv + (jit_swapRate - assetAtExercise_jit[m]) * accrual * dis_jit[m];
                }

                xad::AD jit_payoff = xad::less(jit_npv, xad::AD(0.0)).If(xad::AD(0.0), jit_npv);
                jit.registerOutput(jit_payoff);

                // Compile the JIT kernel
                jit.compile();

                // MC execution
                double mcPrice = 0.0;
                std::vector<double> dPrice_dInitRates(config.size, 0.0);
                double dPrice_dSwapRate = 0.0;

                const auto& graph = jit.getGraph();
                uint32_t outputSlot = graph.output_ids[0];

                for (Size n = 0; n < nrTrails; ++n)
                {
                    for (Size k = 0; k < config.size; ++k)
                        value(jit_initRates[k]) = value(initRates[k]);
                    value(jit_swapRate) = value(swapRate_tape);
                    for (Size m = 0; m < fullGridRandoms; ++m)
                        value(jit_randoms[m]) = allRandoms[n][m];

                    double payoff_value;
                    jit.forward(&payoff_value);
                    mcPrice += payoff_value;

                    jit.clearDerivatives();
                    jit.setDerivative(outputSlot, 1.0);
                    jit.computeAdjoints();

                    for (Size k = 0; k < config.size; ++k)
                        dPrice_dInitRates[k] += jit.derivative(graph.input_ids[k]);
                    dPrice_dSwapRate += jit.derivative(graph.input_ids[config.size]);
                }

                mcPrice /= static_cast<double>(nrTrails);
                for (Size k = 0; k < config.size; ++k)
                    dPrice_dInitRates[k] /= static_cast<double>(nrTrails);
                dPrice_dSwapRate /= static_cast<double>(nrTrails);

                // Chain rule
                std::vector<double> dPrice_dIntermediates(numIntermediates);
                for (Size k = 0; k < config.size; ++k)
                    dPrice_dIntermediates[k] = dPrice_dInitRates[k];
                dPrice_dIntermediates[config.size] = dPrice_dSwapRate;

                std::vector<double> dPrice_market(numMarketQuotes);
                applyChainRule(jacobian.data(), dPrice_dIntermediates.data(), dPrice_market.data(),
                               numIntermediates, numMarketQuotes);

                auto t_end = Clock::now();
                if (recordTiming)
                {
                    jit_times.push_back(Duration(t_end - t_start).count());
                }
            }

            // =================================================================
            // JIT-AVX approach (ForgeBackendAVX with 4-path batching)
            // =================================================================
            {
                auto t_start = Clock::now();

                using tape_type = Real::tape_type;
                tape_type tape;

                std::vector<Real> depositRates(config.numDeposits);
                std::vector<Real> swapRatesAD(config.numSwaps);
                for (Size idx = 0; idx < config.numDeposits; ++idx)
                    depositRates[idx] = config.depoRates[idx];
                for (Size idx = 0; idx < config.numSwaps; ++idx)
                    swapRatesAD[idx] = config.swapRates[idx];

                tape.registerInputs(depositRates);
                tape.registerInputs(swapRatesAD);
                tape.newRecording();

                // Build curve
                RelinkableHandle<YieldTermStructure> euriborTS;
                auto euribor6m = ext::make_shared<Euribor6M>(euriborTS);
                euribor6m->addFixing(Date(2, September, 2005), 0.04);

                std::vector<ext::shared_ptr<RateHelper>> instruments;
                for (Size idx = 0; idx < config.numDeposits; ++idx)
                {
                    auto depoQuote = ext::make_shared<SimpleQuote>(depositRates[idx]);
                    instruments.push_back(ext::make_shared<DepositRateHelper>(
                        Handle<Quote>(depoQuote), config.depoTenors[idx], fixingDays,
                        calendar, ModifiedFollowing, true, dayCounter));
                }
                for (Size idx = 0; idx < config.numSwaps; ++idx)
                {
                    auto swapQuote = ext::make_shared<SimpleQuote>(swapRatesAD[idx]);
                    instruments.push_back(ext::make_shared<SwapRateHelper>(
                        Handle<Quote>(swapQuote), config.swapTenors[idx],
                        calendar, Annual, Unadjusted, Thirty360(Thirty360::BondBasis),
                        euribor6m));
                }

                auto yieldCurve = ext::make_shared<PiecewiseYieldCurve<ZeroYield, Linear>>(
                    settlementDate, instruments, dayCounter);
                yieldCurve->enableExtrapolation();

                std::vector<Date> curveDates;
                std::vector<Real> zeroRates;
                curveDates.push_back(settlementDate);
                zeroRates.push_back(yieldCurve->zeroRate(settlementDate, dayCounter, Continuous).rate());
                Date endDate = settlementDate + config.curveEndYears * Years;
                curveDates.push_back(endDate);
                zeroRates.push_back(yieldCurve->zeroRate(endDate, dayCounter, Continuous).rate());

                std::vector<Rate> zeroRates_ql;
                for (const auto& r : zeroRates) zeroRates_ql.push_back(r);

                RelinkableHandle<YieldTermStructure> termStructure;
                ext::shared_ptr<IborIndex> index(new Euribor6M(termStructure));
                index->addFixing(Date(2, September, 2005), 0.04);
                termStructure.linkTo(ext::make_shared<ZeroCurve>(curveDates, zeroRates_ql, dayCounter));

                ext::shared_ptr<LiborForwardModelProcess> process(
                    new LiborForwardModelProcess(config.size, index));
                process->setCovarParam(ext::shared_ptr<LfmCovarianceParameterization>(
                    new LfmCovarianceProxy(
                        ext::make_shared<LmLinearExponentialVolatilityModel>(process->fixingTimes(), 0.291, 1.483, 0.116, 0.00001),
                        ext::make_shared<LmExponentialCorrelationModel>(config.size, 0.5))));

                ext::shared_ptr<VanillaSwap> fwdSwap(
                    new VanillaSwap(Swap::Receiver, 1.0,
                                    schedule, 0.05, dayCounter,
                                    schedule, index, 0.0, index->dayCounter()));
                fwdSwap->setPricingEngine(ext::make_shared<DiscountingSwapEngine>(
                    index->forwardingTermStructure()));
                Real swapRate_tape = fwdSwap->fairRate();

                Array initRates = process->initialValues();

                // Jacobian computation
                std::vector<double> jacobian(numIntermediates * numMarketQuotes, 0.0);
                for (Size k = 0; k < config.size; ++k)
                {
                    if (initRates[k].shouldRecord())
                    {
                        tape.clearDerivatives();
                        tape.registerOutput(initRates[k]);
                        derivative(initRates[k]) = 1.0;
                        tape.computeAdjoints();

                        double* jac_row = jacobian.data() + k * numMarketQuotes;
                        for (Size m = 0; m < config.numDeposits; ++m)
                            jac_row[m] = derivative(depositRates[m]);
                        for (Size m = 0; m < config.numSwaps; ++m)
                            jac_row[config.numDeposits + m] = derivative(swapRatesAD[m]);
                    }
                }
                if (swapRate_tape.shouldRecord())
                {
                    tape.clearDerivatives();
                    tape.registerOutput(swapRate_tape);
                    derivative(swapRate_tape) = 1.0;
                    tape.computeAdjoints();

                    double* jac_row = jacobian.data() + config.size * numMarketQuotes;
                    for (Size m = 0; m < config.numDeposits; ++m)
                        jac_row[m] = derivative(depositRates[m]);
                    for (Size m = 0; m < config.numSwaps; ++m)
                        jac_row[config.numDeposits + m] = derivative(swapRatesAD[m]);
                }
                tape.deactivate();

                // JIT kernel creation for AVX backend
                xad::JITCompiler<double> jit;

                std::vector<xad::AD> jit_initRates(config.size);
                xad::AD jit_swapRate;
                std::vector<xad::AD> jit_randoms(fullGridRandoms);

                for (Size k = 0; k < config.size; ++k)
                {
                    jit_initRates[k] = xad::AD(value(initRates[k]));
                    jit.registerInput(jit_initRates[k]);
                }
                jit_swapRate = xad::AD(value(swapRate_tape));
                jit.registerInput(jit_swapRate);
                for (Size m = 0; m < fullGridRandoms; ++m)
                {
                    jit_randoms[m] = xad::AD(0.0);
                    jit.registerInput(jit_randoms[m]);
                }

                jit.newRecording();

                std::vector<xad::AD> asset_jit(config.size);
                std::vector<xad::AD> assetAtExercise_jit(config.size);
                for (Size k = 0; k < config.size; ++k) asset_jit[k] = jit_initRates[k];

                for (Size step = 1; step <= fullGridSteps; ++step)
                {
                    Size offset = (step - 1) * numFactors;
                    Time t = grid[step - 1];
                    Time dt = grid.dt(step - 1);

                    Array dw(numFactors);
                    for (Size f = 0; f < numFactors; ++f)
                        dw[f] = jit_randoms[offset + f];

                    Array asset_arr(config.size);
                    for (Size k = 0; k < config.size; ++k) asset_arr[k] = asset_jit[k];

                    Array evolved = process->evolve(t, asset_arr, dt, dw);
                    for (Size k = 0; k < config.size; ++k) asset_jit[k] = evolved[k];

                    if (step == exerciseStep)
                    {
                        for (Size k = 0; k < config.size; ++k)
                            assetAtExercise_jit[k] = asset_jit[k];
                    }
                }

                std::vector<xad::AD> dis_jit(config.size);
                xad::AD df_jit = xad::AD(1.0);
                for (Size k = 0; k < config.size; ++k)
                {
                    double accrual = accrualEnd[k] - accrualStart[k];
                    df_jit = df_jit / (xad::AD(1.0) + assetAtExercise_jit[k] * accrual);
                    dis_jit[k] = df_jit;
                }

                xad::AD jit_npv = xad::AD(0.0);
                for (Size m = config.i_opt; m < config.i_opt + config.j_opt; ++m)
                {
                    double accrual = accrualEnd[m] - accrualStart[m];
                    jit_npv = jit_npv + (jit_swapRate - assetAtExercise_jit[m]) * accrual * dis_jit[m];
                }

                xad::AD jit_payoff = xad::less(jit_npv, xad::AD(0.0)).If(xad::AD(0.0), jit_npv);
                jit.registerOutput(jit_payoff);

                // Get the JIT graph and deactivate the JIT compiler
                const auto& jitGraph = jit.getGraph();
                jit.deactivate();

                // AVX backend with 4-path batching
                xad::forge::ForgeBackendAVX<double> avxBackend(false);
                avxBackend.compile(jitGraph);

                // MC execution with 4-path batching
                double mcPrice = 0.0;
                std::vector<double> dPrice_dInitRates(config.size, 0.0);
                double dPrice_dSwapRate = 0.0;

                constexpr int BATCH_SIZE = xad::forge::ForgeBackendAVX<double>::VECTOR_WIDTH;
                Size numBatches = (nrTrails + BATCH_SIZE - 1) / BATCH_SIZE;

                std::vector<double> inputBatch(BATCH_SIZE);
                std::vector<double> outputBatch(BATCH_SIZE);

                for (Size batch = 0; batch < numBatches; ++batch)
                {
                    Size batchStart = batch * BATCH_SIZE;
                    Size actualBatchSize = std::min(static_cast<Size>(BATCH_SIZE), nrTrails - batchStart);

                    // Set initRates (same for all paths in batch)
                    for (Size k = 0; k < config.size; ++k)
                    {
                        for (int lane = 0; lane < BATCH_SIZE; ++lane)
                            inputBatch[lane] = value(initRates[k]);
                        avxBackend.setInput(k, inputBatch.data());
                    }

                    // Set swapRate (same for all paths in batch)
                    for (int lane = 0; lane < BATCH_SIZE; ++lane)
                        inputBatch[lane] = value(swapRate_tape);
                    avxBackend.setInput(config.size, inputBatch.data());

                    // Set random numbers (different for each path in batch)
                    for (Size m = 0; m < fullGridRandoms; ++m)
                    {
                        for (int lane = 0; lane < BATCH_SIZE; ++lane)
                        {
                            Size pathIdx = batchStart + lane;
                            inputBatch[lane] = (pathIdx < nrTrails) ? allRandoms[pathIdx][m] : 0.0;
                        }
                        avxBackend.setInput(config.size + 1 + m, inputBatch.data());
                    }

                    // Execute forward + backward
                    std::size_t numInputs = avxBackend.numInputs();
                    std::vector<double> inputGradients(numInputs * BATCH_SIZE);
                    avxBackend.forwardAndBackward(outputBatch.data(), inputGradients.data());

                    // Accumulate MC price
                    for (Size lane = 0; lane < actualBatchSize; ++lane)
                    {
                        mcPrice += outputBatch[lane];
                    }

                    // Accumulate gradients for initRates
                    for (Size k = 0; k < config.size; ++k)
                    {
                        for (Size lane = 0; lane < actualBatchSize; ++lane)
                        {
                            dPrice_dInitRates[k] += inputGradients[k * BATCH_SIZE + lane];
                        }
                    }

                    // Accumulate gradient for swap rate
                    for (Size lane = 0; lane < actualBatchSize; ++lane)
                    {
                        dPrice_dSwapRate += inputGradients[config.size * BATCH_SIZE + lane];
                    }
                }

                mcPrice /= static_cast<double>(nrTrails);
                for (Size k = 0; k < config.size; ++k)
                    dPrice_dInitRates[k] /= static_cast<double>(nrTrails);
                dPrice_dSwapRate /= static_cast<double>(nrTrails);

                // Chain rule
                std::vector<double> dPrice_dIntermediates(numIntermediates);
                for (Size k = 0; k < config.size; ++k)
                    dPrice_dIntermediates[k] = dPrice_dInitRates[k];
                dPrice_dIntermediates[config.size] = dPrice_dSwapRate;

                std::vector<double> dPrice_market(numMarketQuotes);
                applyChainRule(jacobian.data(), dPrice_dIntermediates.data(), dPrice_market.data(),
                               numIntermediates, numMarketQuotes);

                auto t_end = Clock::now();
                if (recordTiming)
                {
                    jit_avx_times.push_back(Duration(t_end - t_start).count());
                }
            }
#endif // QLRISKS_HAS_FORGE
        }

        results[tc].xad_mean = computeMean(xad_times);
        results[tc].xad_std = computeStddev(xad_times);
#if defined(QLRISKS_HAS_FORGE)
        results[tc].jit_mean = computeMean(jit_times);
        results[tc].jit_std = computeStddev(jit_times);
        results[tc].jit_avx_mean = computeMean(jit_avx_times);
        results[tc].jit_avx_std = computeStddev(jit_avx_times);
#endif

        std::cout << "done\n";
    }

    // Print results table
    std::cout << "\n";
    std::cout << std::string(80, '=') << "\n";
    std::cout << "  RESULTS (mean +/- stddev, in ms)\n";
    std::cout << std::string(80, '=') << "\n";
    std::cout << "\n";

    std::cout << "| Paths  |    Method |     Mean |   StdDev | Speedup |\n";
    std::cout << "|-------:|----------:|---------:|---------:|--------:|\n";

    for (size_t tc = 0; tc < config.pathCounts.size(); ++tc)
    {
        std::string pathLabel;
        if (config.pathCounts[tc] >= 1000)
            pathLabel = std::to_string(config.pathCounts[tc] / 1000) + "K";
        else
            pathLabel = std::to_string(config.pathCounts[tc]);

        std::cout << std::fixed << std::setprecision(1);

        std::cout << "| " << std::setw(6) << pathLabel << " |       XAD |"
                  << std::setw(9) << results[tc].xad_mean << " |"
                  << std::setw(9) << results[tc].xad_std << " |     --- |\n";

#if defined(QLRISKS_HAS_FORGE)
        double jit_speedup = results[tc].xad_mean / results[tc].jit_mean;
        double avx_speedup = results[tc].xad_mean / results[tc].jit_avx_mean;

        std::cout << "|        |       JIT |"
                  << std::setw(9) << results[tc].jit_mean << " |"
                  << std::setw(9) << results[tc].jit_std << " |"
                  << std::setw(7) << std::setprecision(2) << jit_speedup << "x |\n";

        std::cout << "|        |   JIT-AVX |"
                  << std::setw(9) << std::setprecision(1) << results[tc].jit_avx_mean << " |"
                  << std::setw(9) << results[tc].jit_avx_std << " |"
                  << std::setw(7) << std::setprecision(2) << avx_speedup << "x |\n";
#endif

        if (tc < config.pathCounts.size() - 1)
            std::cout << "|--------+-----------+----------+----------+---------|\n";
    }

    std::cout << "\n";
    std::cout << "  Speedup = XAD / Method. All times in ms.\n";
    std::cout << "\n";
}

// ============================================================================
// Performance Decomposition
// ============================================================================

void runDecomposition(const BenchmarkConfig& config)
{
    using Clock = std::chrono::high_resolution_clock;
    using Duration = std::chrono::duration<double, std::milli>;

    std::cout << "\n";
    std::cout << std::string(80, '=') << "\n";
    std::cout << "  PERFORMANCE DECOMPOSITION (1K paths)\n";
    std::cout << std::string(80, '=') << "\n";

#if defined(QLRISKS_HAS_FORGE)
    // Setup (simplified version for decomposition)
    Calendar calendar = TARGET();
    Date todaysDate(4, September, 2005);
    Settings::instance().evaluationDate() = todaysDate;
    Integer fixingDays = 2;
    Date settlementDate = calendar.adjust(calendar.advance(todaysDate, fixingDays, Days));
    DayCounter dayCounter = Actual360();

    // Build base curve (using first deposit rate and last swap rate as endpoints)
    std::vector<Rate> baseZeroRates = {config.depoRates[0], config.swapRates.back()};
    std::vector<Date> baseDates = {settlementDate, settlementDate + config.curveEndYears * Years};
    auto baseIndex = makeIndex(baseDates, baseZeroRates);

    ext::shared_ptr<LiborForwardModelProcess> baseProcess(
        new LiborForwardModelProcess(config.size, baseIndex));
    baseProcess->setCovarParam(ext::shared_ptr<LfmCovarianceParameterization>(
        new LfmCovarianceProxy(
            ext::make_shared<LmLinearExponentialVolatilityModel>(baseProcess->fixingTimes(), 0.291, 1.483, 0.116, 0.00001),
            ext::make_shared<LmExponentialCorrelationModel>(config.size, 0.5))));

    std::vector<Time> fixingTimes = baseProcess->fixingTimes();
    TimeGrid grid(fixingTimes.begin(), fixingTimes.end(), config.steps);

    Size numFactors = baseProcess->factors();
    Size fullGridSteps = grid.size() - 1;
    Size fullGridRandoms = fullGridSteps * numFactors;

    // Accrual periods
    std::vector<double> accrualStart(config.size), accrualEnd(config.size);
    for (Size k = 0; k < config.size; ++k)
    {
        accrualStart[k] = value(baseProcess->accrualStartTimes()[k]);
        accrualEnd[k] = value(baseProcess->accrualEndTimes()[k]);
    }

    std::vector<Size> location;
    for (Size idx = 0; idx < fixingTimes.size(); ++idx)
        location.push_back(std::find(grid.begin(), grid.end(), fixingTimes[idx]) - grid.begin());
    Size exerciseStep = location[config.i_opt];

    // Swap schedule
    BusinessDayConvention convention = baseIndex->businessDayConvention();
    Date fwdStart = settlementDate + Period(6 * config.i_opt, Months);
    Date fwdMaturity = fwdStart + Period(6 * config.j_opt, Months);
    Schedule schedule(fwdStart, fwdMaturity, baseIndex->tenor(), calendar,
                      convention, convention, DateGeneration::Forward, false);

    // Build process for recording
    RelinkableHandle<YieldTermStructure> termStructure;
    ext::shared_ptr<IborIndex> index(new Euribor6M(termStructure));
    index->addFixing(Date(2, September, 2005), 0.04);

    std::vector<Date> curveDates = {settlementDate, settlementDate + config.curveEndYears * Years};
    std::vector<Rate> zeroRates_ql = {config.depoRates[0], config.swapRates.back()};
    termStructure.linkTo(ext::make_shared<ZeroCurve>(curveDates, zeroRates_ql, dayCounter));

    ext::shared_ptr<LiborForwardModelProcess> process(
        new LiborForwardModelProcess(config.size, index));
    process->setCovarParam(ext::shared_ptr<LfmCovarianceParameterization>(
        new LfmCovarianceProxy(
            ext::make_shared<LmLinearExponentialVolatilityModel>(process->fixingTimes(), 0.291, 1.483, 0.116, 0.00001),
            ext::make_shared<LmExponentialCorrelationModel>(config.size, 0.5))));

    Array initRates = process->initialValues();

    // Build JIT graph
    auto t_graph_start = Clock::now();

    xad::JITCompiler<double> jit;

    std::vector<xad::AD> jit_initRates(config.size);
    xad::AD jit_swapRate;
    std::vector<xad::AD> jit_randoms(fullGridRandoms);

    for (Size k = 0; k < config.size; ++k)
    {
        jit_initRates[k] = xad::AD(value(initRates[k]));
        jit.registerInput(jit_initRates[k]);
    }
    jit_swapRate = xad::AD(0.05);  // Placeholder
    jit.registerInput(jit_swapRate);
    for (Size m = 0; m < fullGridRandoms; ++m)
    {
        jit_randoms[m] = xad::AD(0.0);
        jit.registerInput(jit_randoms[m]);
    }

    jit.newRecording();

    std::vector<xad::AD> asset_jit(config.size);
    std::vector<xad::AD> assetAtExercise_jit(config.size);
    for (Size k = 0; k < config.size; ++k) asset_jit[k] = jit_initRates[k];

    for (Size step = 1; step <= fullGridSteps; ++step)
    {
        Size offset = (step - 1) * numFactors;
        Time t = grid[step - 1];
        Time dt = grid.dt(step - 1);

        Array dw(numFactors);
        for (Size f = 0; f < numFactors; ++f)
            dw[f] = jit_randoms[offset + f];

        Array asset_arr(config.size);
        for (Size k = 0; k < config.size; ++k) asset_arr[k] = asset_jit[k];

        Array evolved = process->evolve(t, asset_arr, dt, dw);
        for (Size k = 0; k < config.size; ++k) asset_jit[k] = evolved[k];

        if (step == exerciseStep)
        {
            for (Size k = 0; k < config.size; ++k)
                assetAtExercise_jit[k] = asset_jit[k];
        }
    }

    std::vector<xad::AD> dis_jit(config.size);
    xad::AD df_jit = xad::AD(1.0);
    for (Size k = 0; k < config.size; ++k)
    {
        double accrual = accrualEnd[k] - accrualStart[k];
        df_jit = df_jit / (xad::AD(1.0) + assetAtExercise_jit[k] * accrual);
        dis_jit[k] = df_jit;
    }

    xad::AD jit_npv = xad::AD(0.0);
    for (Size m = config.i_opt; m < config.i_opt + config.j_opt; ++m)
    {
        double accrual = accrualEnd[m] - accrualStart[m];
        jit_npv = jit_npv + (jit_swapRate - assetAtExercise_jit[m]) * accrual * dis_jit[m];
    }

    xad::AD jit_payoff = xad::less(jit_npv, xad::AD(0.0)).If(xad::AD(0.0), jit_npv);
    jit.registerOutput(jit_payoff);

    auto t_graph_end = Clock::now();
    double graphTimeMs = Duration(t_graph_end - t_graph_start).count();

    // Compile with ForgeBackend
    auto t_compile_start = Clock::now();
    const auto& jitGraph = jit.getGraph();
    xad::forge::ForgeBackendAVX<double> avxBackend(false);
    avxBackend.compile(jitGraph);
    auto t_compile_end = Clock::now();
    double compileTimeMs = Duration(t_compile_end - t_compile_start).count();

    // Run MC loop for timing
    Size nrTrails = 1000;
    constexpr int BATCH_SIZE = xad::forge::ForgeBackendAVX<double>::VECTOR_WIDTH;

    // Generate random numbers
    typedef PseudoRandom::rsg_type rsg_type;
    rsg_type rsg_base = PseudoRandom::make_sequence_generator(fullGridRandoms, BigNatural(42));

    std::vector<std::vector<double>> allRandoms(nrTrails);
    for (Size n = 0; n < nrTrails; ++n)
    {
        allRandoms[n].resize(fullGridRandoms);
        const auto& seq = rsg_base.nextSequence();
        for (Size m = 0; m < fullGridRandoms; ++m)
            allRandoms[n][m] = value(seq.value[m]);
    }

    // Time each phase
    double setInputTime = 0.0;
    double forwardTime = 0.0;
    double accumulateTime = 0.0;

    Size numBatches = (nrTrails + BATCH_SIZE - 1) / BATCH_SIZE;
    std::vector<double> inputBatch(BATCH_SIZE);
    std::vector<double> outputBatch(BATCH_SIZE);

    for (Size batch = 0; batch < numBatches; ++batch)
    {
        Size batchStart = batch * BATCH_SIZE;
        Size actualBatchSize = std::min(static_cast<Size>(BATCH_SIZE), nrTrails - batchStart);

        // Set inputs
        auto t1 = Clock::now();
        for (Size k = 0; k < config.size; ++k)
        {
            for (int lane = 0; lane < BATCH_SIZE; ++lane)
                inputBatch[lane] = value(initRates[k]);
            avxBackend.setInput(k, inputBatch.data());
        }
        for (int lane = 0; lane < BATCH_SIZE; ++lane)
            inputBatch[lane] = 0.05;
        avxBackend.setInput(config.size, inputBatch.data());
        for (Size m = 0; m < fullGridRandoms; ++m)
        {
            for (int lane = 0; lane < BATCH_SIZE; ++lane)
            {
                Size pathIdx = batchStart + lane;
                inputBatch[lane] = (pathIdx < nrTrails) ? allRandoms[pathIdx][m] : 0.0;
            }
            avxBackend.setInput(config.size + 1 + m, inputBatch.data());
        }
        auto t2 = Clock::now();
        setInputTime += Duration(t2 - t1).count();

        // Forward + backward (combined for AVX backend)
        auto t3 = Clock::now();
        std::size_t numInputs = avxBackend.numInputs();
        std::vector<double> inputGradients(numInputs * BATCH_SIZE);
        avxBackend.forwardAndBackward(outputBatch.data(), inputGradients.data());
        auto t4 = Clock::now();
        forwardTime += Duration(t4 - t3).count();

        // Accumulate results (simplified)
        auto t5 = Clock::now();
        double dummy = 0.0;
        for (Size lane = 0; lane < actualBatchSize; ++lane)
            dummy += outputBatch[lane];
        auto t6 = Clock::now();
        accumulateTime += Duration(t6 - t5).count();
        (void)dummy;
    }

    double totalTime = graphTimeMs + compileTimeMs + setInputTime + forwardTime + accumulateTime;
    double execTime = totalTime - graphTimeMs - compileTimeMs;

    // Print in xad-jit style
    std::cout << "\n  JIT-AVX (Forge + AVX2 SIMD)\n";
    std::cout << std::string(80, '-') << "\n";

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Total time:              " << std::setw(10) << totalTime << " ms (100.0%)\n";
    std::cout << "  -------------------------\n";
    std::cout << "  Graph build (one-time):  " << std::setw(10) << graphTimeMs << " ms ("
              << std::setw(5) << std::setprecision(1) << (graphTimeMs / totalTime * 100) << "%)\n";
    std::cout << "  Compile (one-time):      " << std::setw(10) << std::setprecision(2) << compileTimeMs << " ms ("
              << std::setw(5) << std::setprecision(1) << (compileTimeMs / totalTime * 100) << "%)\n";
    std::cout << "  Set inputs:              " << std::setw(10) << std::setprecision(2) << setInputTime << " ms ("
              << std::setw(5) << std::setprecision(1) << (setInputTime / totalTime * 100) << "%)\n";
    std::cout << "  Forward+Backward:        " << std::setw(10) << std::setprecision(2) << forwardTime << " ms ("
              << std::setw(5) << std::setprecision(1) << (forwardTime / totalTime * 100) << "%)\n";
    std::cout << "  Accumulate results:      " << std::setw(10) << std::setprecision(2) << accumulateTime << " ms ("
              << std::setw(5) << std::setprecision(1) << (accumulateTime / totalTime * 100) << "%)\n";
    std::cout << "  -------------------------\n";
    std::cout << "  Execution (excl compile):" << std::setw(10) << std::setprecision(2) << execTime << " ms\n";
    std::cout << "  Batches (4 paths each):  " << std::setw(10) << numBatches << "\n";
    std::cout << "  Time per batch:          " << std::setw(10) << std::setprecision(2) << (execTime / numBatches * 1000) << " us\n";
    std::cout << "  Time per path:           " << std::setw(10) << std::setprecision(2) << (execTime / nrTrails * 1000) << " us\n";

#else
    std::cout << "\n  [Decomposition requires Forge JIT support]\n";
#endif

    std::cout << "\n";
}

// ============================================================================
// Main
// ============================================================================

void printUsage(const char* progName)
{
    std::cout << "Usage: " << progName << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --help, -h        Show this help message\n";
    std::cout << "  --quick           Run quick benchmark (fewer iterations, fewer path counts)\n";
    std::cout << "  --small-only      Run only the small swaption benchmark (1Y into 1Y)\n";
    std::cout << "  --large-only      Run only the large swaption benchmark (5Y into 5Y)\n";
    std::cout << "  --decomposition   Run performance decomposition analysis\n";
    std::cout << "\nBenchmarks (both run by default):\n";
    std::cout << "  Small:  1Y into 1Y swaption (10 forward rates, 8 time steps)\n";
    std::cout << "  Large:  5Y into 5Y swaption (20 forward rates, 20 time steps)\n";
    std::cout << "\nThis benchmark compares AD approaches for swaption pricing.\n";
#if defined(QLRISKS_HAS_FORGE)
    std::cout << "Build: With Forge JIT support\n";
#else
    std::cout << "Build: Without Forge JIT (XAD tape only)\n";
#endif
}

void runSingleBenchmark(const BenchmarkConfig& config, bool quickMode)
{
    std::cout << "\n  INSTRUMENT\n";
    std::cout << std::string(80, '-') << "\n";
    std::cout << "  Instrument:   " << config.instrumentDesc << "\n";
    std::cout << "  Model:        LIBOR Market Model (LMM)\n";
    std::cout << "  Forward rates:" << config.size << " (semi-annual)\n";
    std::cout << "  Time steps:   " << config.steps << "\n";
    std::cout << "  Inputs:       " << (config.numDeposits + config.numSwaps) << " market quotes\n";

    std::cout << "\n  BENCHMARK CONFIGURATION\n";
    std::cout << std::string(80, '-') << "\n";
    std::cout << "  Path counts:  ";
    for (size_t i = 0; i < config.pathCounts.size(); ++i)
    {
        if (i > 0) std::cout << ", ";
        if (config.pathCounts[i] >= 1000)
            std::cout << (config.pathCounts[i] / 1000) << "K";
        else
            std::cout << config.pathCounts[i];
    }
    std::cout << "\n";
    std::cout << "  Warmup:       " << config.warmupIterations << " iterations\n";
    std::cout << "  Measured:     " << config.benchmarkIterations << " iterations\n";

    std::cout << "\n  METHODS\n";
    std::cout << std::string(80, '-') << "\n";
    std::cout << "  XAD      XAD tape-based reverse-mode AAD\n";
#if defined(QLRISKS_HAS_FORGE)
    std::cout << "  JIT      Forge JIT-compiled native code\n";
    std::cout << "  JIT-AVX  Forge JIT + AVX2 SIMD (4 paths/instruction)\n";
#endif

    std::cout << "\n";
    std::cout << std::string(80, '=') << "\n";
    std::cout << "  RUNNING BENCHMARKS\n";
    std::cout << std::string(80, '=') << "\n";
    std::cout << "\n";

    runBenchmark(config, quickMode);
}

int main(int argc, char** argv)
{
    bool quickMode = false;
    bool smallOnly = false;
    bool largeOnly = false;
    bool decompositionMode = false;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h")
        {
            printUsage(argv[0]);
            return 0;
        }
        else if (arg == "--quick")
            quickMode = true;
        else if (arg == "--small-only")
            smallOnly = true;
        else if (arg == "--large-only" || arg == "--large")
            largeOnly = true;
        else if (arg == "--decomposition")
            decompositionMode = true;
    }

    // Print header
    std::cout << "\n";
    std::cout << std::string(80, '=') << "\n";
    std::cout << "  QuantLib-Risks Swaption Benchmark\n";
    std::cout << std::string(80, '=') << "\n";

    std::cout << "\n  ENVIRONMENT\n";
    std::cout << std::string(80, '-') << "\n";
    std::cout << "  Platform:     " << getPlatformInfo() << "\n";
    std::cout << "  CPU:          " << getCpuInfo() << "\n";
    std::cout << "  RAM:          " << getMemoryInfo() << "\n";
    std::cout << "  SIMD:         " << getSimdInfo() << "\n";
    std::cout << "  Compiler:     " << getCompilerInfo() << "\n";

    if (decompositionMode)
    {
        // Run decomposition for both configs
        BenchmarkConfig smallConfig;
        if (quickMode)
        {
            smallConfig.pathCounts = {100, 1000};
            smallConfig.warmupIterations = 1;
            smallConfig.benchmarkIterations = 2;
        }

        if (!largeOnly)
        {
            std::cout << "\n";
            std::cout << std::string(80, '=') << "\n";
            std::cout << "  DECOMPOSITION: Small Swaption (1Y into 1Y)\n";
            std::cout << std::string(80, '=') << "\n";
            runDecomposition(smallConfig);
        }

        if (!smallOnly)
        {
            BenchmarkConfig largeConfig;
            largeConfig.setLargeConfig();
            if (quickMode)
            {
                largeConfig.pathCounts = {100, 1000};
                largeConfig.warmupIterations = 1;
                largeConfig.benchmarkIterations = 2;
            }

            std::cout << "\n";
            std::cout << std::string(80, '=') << "\n";
            std::cout << "  DECOMPOSITION: Large Swaption (5Y into 5Y)\n";
            std::cout << std::string(80, '=') << "\n";
            runDecomposition(largeConfig);
        }
    }
    else
    {
        // Run benchmarks
        bool runSmall = !largeOnly;
        bool runLarge = !smallOnly;

        if (runSmall)
        {
            BenchmarkConfig smallConfig;
            if (quickMode)
            {
                smallConfig.pathCounts = {100, 1000};
                smallConfig.warmupIterations = 1;
                smallConfig.benchmarkIterations = 2;
            }

            std::cout << "\n";
            std::cout << std::string(80, '=') << "\n";
            std::cout << "  BENCHMARK 1: Small Swaption (1Y into 1Y)\n";
            std::cout << std::string(80, '=') << "\n";

            runSingleBenchmark(smallConfig, quickMode);
        }

        if (runLarge)
        {
            BenchmarkConfig largeConfig;
            largeConfig.setLargeConfig();
            if (quickMode)
            {
                largeConfig.pathCounts = {100, 1000};
                largeConfig.warmupIterations = 1;
                largeConfig.benchmarkIterations = 2;
            }

            std::cout << "\n";
            std::cout << std::string(80, '=') << "\n";
            std::cout << "  BENCHMARK 2: Large Swaption (5Y into 5Y)\n";
            std::cout << std::string(80, '=') << "\n";

            runSingleBenchmark(largeConfig, quickMode);
        }
    }

    std::cout << "\n";
    std::cout << std::string(80, '=') << "\n";
    std::cout << "  All benchmarks complete.\n";
    std::cout << std::string(80, '=') << "\n";
    std::cout << "\n";

    return 0;
}
