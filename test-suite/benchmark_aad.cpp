/*******************************************************************************
 *
 *  QuantLib-Risks Swaption Benchmark - AAD Runner
 *
 *  AAD benchmarks using XAD tape, Forge JIT, and Forge JIT-AVX.
 *  This executable is compiled WITH XAD (and optionally Forge).
 *
 *  Usage:
 *    ./benchmark_aad [--lite|--lite-extended|--production|--all] [--quick] [--xad-only]
 *
 *  Output format is designed to be parsed and combined with FD results.
 *
 *  Copyright (C) 2025 Xcelerit Computing Limited
 *  SPDX-License-Identifier: AGPL-3.0-or-later
 *
 ******************************************************************************/

#include "benchmark_common.hpp"
#include "benchmark_pricing.hpp"

// XAD includes
#include <XAD/XAD.hpp>

// Forge JIT backends (conditionally included)
#if defined(QLRISKS_HAS_FORGE)
#include <xad-forge/ForgeBackend.hpp>
#include <xad-forge/ForgeBackendAVX.hpp>
#endif

#include <chrono>
#include <cstring>

using namespace benchmark;
using Clock = std::chrono::high_resolution_clock;
using DurationMs = std::chrono::duration<double, std::milli>;

// Use QuantLib's Real which is xad::AReal<double> via qlrisks.hpp
using RealAD = QuantLib::Real;
using tape_type = RealAD::tape_type;

// ============================================================================
// XAD Tape-based AAD Benchmark (Single-Curve)
// ============================================================================

void runXADBenchmark(const BenchmarkConfig& config, const LMMSetup& setup,
                     Size nrTrails, size_t warmup, size_t bench,
                     double& mean, double& stddev)
{
    std::vector<double> times;

    for (size_t iter = 0; iter < warmup + bench; ++iter)
    {
        auto t_start = Clock::now();

        tape_type tape;

        // Register inputs
        std::vector<RealAD> depositRates(config.numDeposits);
        std::vector<RealAD> swapRatesAD(config.numSwaps);
        for (Size idx = 0; idx < config.numDeposits; ++idx)
            depositRates[idx] = config.depoRates[idx];
        for (Size idx = 0; idx < config.numSwaps; ++idx)
            swapRatesAD[idx] = config.swapRates[idx];

        tape.registerInputs(depositRates);
        tape.registerInputs(swapRatesAD);
        tape.newRecording();

        // Price using templated function
        RealAD price = priceSwaption<RealAD>(config, setup, depositRates, swapRatesAD, nrTrails);

        // Compute adjoints
        tape.registerOutput(price);
        derivative(price) = 1.0;
        tape.computeAdjoints();

        auto t_end = Clock::now();

        if (iter >= warmup)
        {
            times.push_back(DurationMs(t_end - t_start).count());
        }

        tape.clearAll();
    }

    mean = computeMean(times);
    stddev = computeStddev(times);
}

// ============================================================================
// XAD Tape-based AAD Benchmark (Dual-Curve / Production)
// ============================================================================

void runXADBenchmarkDualCurve(const BenchmarkConfig& config, const LMMSetup& setup,
                               Size nrTrails, size_t warmup, size_t bench,
                               double& mean, double& stddev)
{
    std::vector<double> times;

    for (size_t iter = 0; iter < warmup + bench; ++iter)
    {
        auto t_start = Clock::now();

        tape_type tape;

        // Register forecasting curve inputs
        std::vector<RealAD> depositRates(config.numDeposits);
        std::vector<RealAD> swapRatesAD(config.numSwaps);
        for (Size idx = 0; idx < config.numDeposits; ++idx)
            depositRates[idx] = config.depoRates[idx];
        for (Size idx = 0; idx < config.numSwaps; ++idx)
            swapRatesAD[idx] = config.swapRates[idx];

        // Register discounting curve inputs (OIS)
        std::vector<RealAD> oisDepoRates(config.numOisDeposits);
        std::vector<RealAD> oisSwapRatesAD(config.numOisSwaps);
        for (Size idx = 0; idx < config.numOisDeposits; ++idx)
            oisDepoRates[idx] = config.oisDepoRates[idx];
        for (Size idx = 0; idx < config.numOisSwaps; ++idx)
            oisSwapRatesAD[idx] = config.oisSwapRates[idx];

        tape.registerInputs(depositRates);
        tape.registerInputs(swapRatesAD);
        tape.registerInputs(oisDepoRates);
        tape.registerInputs(oisSwapRatesAD);
        tape.newRecording();

        // Price using dual-curve function
        RealAD price = priceSwaptionDualCurve<RealAD>(
            config, setup, depositRates, swapRatesAD, oisDepoRates, oisSwapRatesAD, nrTrails);

        // Compute adjoints
        tape.registerOutput(price);
        derivative(price) = 1.0;
        tape.computeAdjoints();

        auto t_end = Clock::now();

        if (iter >= warmup)
        {
            times.push_back(DurationMs(t_end - t_start).count());
        }

        tape.clearAll();
    }

    mean = computeMean(times);
    stddev = computeStddev(times);
}

#if defined(QLRISKS_HAS_FORGE)

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
// Forge JIT Scalar Benchmark (Single-Curve)
// ============================================================================

template <typename BackendType>
void runJITBenchmarkImpl(const BenchmarkConfig& config, const LMMSetup& setup,
                         Size nrTrails, size_t warmup, size_t bench,
                         double& mean, double& stddev)
{
    std::vector<double> times;

    Size numMarketQuotes = config.numMarketQuotes();
    Size numIntermediates = config.size + 1;  // forward rates + swap rate

    for (size_t iter = 0; iter < warmup + bench; ++iter)
    {
        auto t_start = Clock::now();

        // =====================================================================
        // Phase 1: XAD tape - curve bootstrap and Jacobian computation
        // =====================================================================
        tape_type tape;

        // Register inputs
        std::vector<RealAD> depositRates(config.numDeposits);
        std::vector<RealAD> swapRatesAD(config.numSwaps);
        for (Size idx = 0; idx < config.numDeposits; ++idx)
            depositRates[idx] = config.depoRates[idx];
        for (Size idx = 0; idx < config.numSwaps; ++idx)
            swapRatesAD[idx] = config.swapRates[idx];

        tape.registerInputs(depositRates);
        tape.registerInputs(swapRatesAD);
        tape.newRecording();

        // Build curve and get intermediates
        RelinkableHandle<YieldTermStructure> euriborTS;
        auto euribor6m = ext::make_shared<Euribor6M>(euriborTS);
        euribor6m->addFixing(Date(2, September, 2005), 0.04);

        std::vector<ext::shared_ptr<RateHelper>> instruments;
        for (Size idx = 0; idx < config.numDeposits; ++idx)
        {
            auto depoQuote = ext::make_shared<SimpleQuote>(depositRates[idx]);
            instruments.push_back(ext::make_shared<DepositRateHelper>(
                Handle<Quote>(depoQuote), config.depoTenors[idx], setup.fixingDays,
                setup.calendar, ModifiedFollowing, true, setup.dayCounter));
        }
        for (Size idx = 0; idx < config.numSwaps; ++idx)
        {
            auto swapQuote = ext::make_shared<SimpleQuote>(swapRatesAD[idx]);
            instruments.push_back(ext::make_shared<SwapRateHelper>(
                Handle<Quote>(swapQuote), config.swapTenors[idx],
                setup.calendar, Annual, Unadjusted, Thirty360(Thirty360::BondBasis),
                euribor6m));
        }

        auto yieldCurve = ext::make_shared<PiecewiseYieldCurve<ZeroYield, Linear>>(
            setup.settlementDate, instruments, setup.dayCounter);
        yieldCurve->enableExtrapolation();

        // Extract zero rates
        std::vector<Date> curveDates;
        std::vector<RealAD> zeroRates;
        curveDates.push_back(setup.settlementDate);
        zeroRates.push_back(yieldCurve->zeroRate(setup.settlementDate, setup.dayCounter, Continuous).rate());
        Date endDate = setup.settlementDate + config.curveEndYears * Years;
        curveDates.push_back(endDate);
        zeroRates.push_back(yieldCurve->zeroRate(endDate, setup.dayCounter, Continuous).rate());

        std::vector<Rate> zeroRates_ql;
        for (const auto& r : zeroRates) zeroRates_ql.push_back(r);

        // Build LMM process
        RelinkableHandle<YieldTermStructure> termStructure;
        ext::shared_ptr<IborIndex> index(new Euribor6M(termStructure));
        index->addFixing(Date(2, September, 2005), 0.04);
        termStructure.linkTo(ext::make_shared<ZeroCurve>(curveDates, zeroRates_ql, setup.dayCounter));

        ext::shared_ptr<LiborForwardModelProcess> process(
            new LiborForwardModelProcess(config.size, index));
        process->setCovarParam(ext::shared_ptr<LfmCovarianceParameterization>(
            new LfmCovarianceProxy(
                ext::make_shared<LmLinearExponentialVolatilityModel>(
                    process->fixingTimes(), 0.291, 1.483, 0.116, 0.00001),
                ext::make_shared<LmExponentialCorrelationModel>(config.size, 0.5))));

        ext::shared_ptr<VanillaSwap> fwdSwap(
            new VanillaSwap(Swap::Receiver, 1.0,
                            setup.schedule, 0.05, setup.dayCounter,
                            setup.schedule, index, 0.0, index->dayCounter()));
        fwdSwap->setPricingEngine(ext::make_shared<DiscountingSwapEngine>(
            index->forwardingTermStructure()));
        RealAD swapRate = fwdSwap->fairRate();

        // Extract intermediates (forward rates + swap rate)
        Array initRates = process->initialValues();
        std::vector<RealAD> intermediates(numIntermediates);
        for (Size k = 0; k < config.size; ++k)
            intermediates[k] = initRates[k];
        intermediates[config.size] = swapRate;

        // Register intermediates as outputs
        tape.registerOutputs(intermediates);

        // Compute Jacobian: d(intermediates) / d(market inputs)
        std::vector<double> jacobian(numIntermediates * numMarketQuotes);
        for (Size i = 0; i < numIntermediates; ++i)
        {
            tape.clearDerivatives();
            derivative(intermediates[i]) = 1.0;
            tape.computeAdjoints();
            for (Size j = 0; j < config.numDeposits; ++j)
                jacobian[i * numMarketQuotes + j] = derivative(depositRates[j]);
            for (Size j = 0; j < config.numSwaps; ++j)
                jacobian[i * numMarketQuotes + config.numDeposits + j] = derivative(swapRatesAD[j]);
        }

        tape.clearAll();

        // Extract intermediate values as plain doubles
        std::vector<double> intermediateValues(numIntermediates);
        for (Size k = 0; k < numIntermediates; ++k)
            intermediateValues[k] = value(intermediates[k]);

        // =====================================================================
        // Phase 2: JIT graph recording and compilation
        // =====================================================================
        auto backend = std::make_unique<BackendType>();
        xad::JITCompiler<double> jit(std::move(backend));

        // Register JIT inputs: forward rates, swap rate, and randoms
        std::vector<xad::AD> jit_initRates(config.size);
        xad::AD jit_swapRate;
        std::vector<xad::AD> jit_randoms(setup.fullGridRandoms);

        for (Size k = 0; k < config.size; ++k)
        {
            jit_initRates[k] = xad::AD(intermediateValues[k]);
            jit.registerInput(jit_initRates[k]);
        }
        jit_swapRate = xad::AD(intermediateValues[config.size]);
        jit.registerInput(jit_swapRate);

        for (Size m = 0; m < setup.fullGridRandoms; ++m)
        {
            jit_randoms[m] = xad::AD(0.0);
            jit.registerInput(jit_randoms[m]);
        }

        jit.newRecording();

        // Record path evolution
        std::vector<xad::AD> asset(config.size);
        std::vector<xad::AD> assetAtExercise(config.size);
        for (Size k = 0; k < config.size; ++k)
            asset[k] = jit_initRates[k];

        for (Size step = 1; step <= setup.fullGridSteps; ++step)
        {
            Size offset = (step - 1) * setup.numFactors;
            Time t = setup.grid[step - 1];
            Time dt = setup.grid.dt(step - 1);

            Array dw(setup.numFactors);
            for (Size f = 0; f < setup.numFactors; ++f)
                dw[f] = jit_randoms[offset + f];

            Array asset_arr(config.size);
            for (Size k = 0; k < config.size; ++k)
                asset_arr[k] = asset[k];

            Array evolved = process->evolve(t, asset_arr, dt, dw);
            for (Size k = 0; k < config.size; ++k)
                asset[k] = evolved[k];

            if (step == setup.exerciseStep)
            {
                for (Size k = 0; k < config.size; ++k)
                    assetAtExercise[k] = asset[k];
            }
        }

        // Compute payoff: discount factors and NPV
        std::vector<xad::AD> dis(config.size);
        xad::AD df = xad::AD(1.0);
        for (Size k = 0; k < config.size; ++k)
        {
            double accrual = setup.accrualEnd[k] - setup.accrualStart[k];
            df = df / (xad::AD(1.0) + assetAtExercise[k] * accrual);
            dis[k] = df;
        }

        xad::AD npv = xad::AD(0.0);
        for (Size m = config.i_opt; m < config.i_opt + config.j_opt; ++m)
        {
            double accrual = setup.accrualEnd[m] - setup.accrualStart[m];
            npv = npv + (jit_swapRate - assetAtExercise[m]) * accrual * dis[m];
        }

        // Payoff = max(npv, 0)
        xad::AD payoff = xad::less(npv, xad::AD(0.0)).If(xad::AD(0.0), npv);
        jit.registerOutput(payoff);

        // Compile the JIT kernel
        jit.compile();

        // =====================================================================
        // Phase 3: Execute JIT kernel for all MC paths
        // =====================================================================
        const auto& graph = jit.getGraph();

        double mcPrice = 0.0;
        std::vector<double> dPrice_dIntermediates(numIntermediates, 0.0);

        for (Size n = 0; n < nrTrails; ++n)
        {
            // Set inputs
            for (Size k = 0; k < config.size; ++k)
                value(jit_initRates[k]) = intermediateValues[k];
            value(jit_swapRate) = intermediateValues[config.size];
            for (Size m = 0; m < setup.fullGridRandoms; ++m)
                value(jit_randoms[m]) = setup.allRandoms[n][m];

            // Execute forward + backward
            double payoff_value;
            jit.forward(&payoff_value);
            mcPrice += payoff_value;

            // Accumulate gradients w.r.t. intermediates
            jit.clearDerivatives();
            jit.setDerivative(graph.output_ids[0], 1.0);
            jit.computeAdjoints();

            for (Size k = 0; k < config.size; ++k)
                dPrice_dIntermediates[k] += jit.derivative(graph.input_ids[k]);
            dPrice_dIntermediates[config.size] += jit.derivative(graph.input_ids[config.size]);
        }

        // Average
        mcPrice /= static_cast<double>(nrTrails);
        for (Size k = 0; k < numIntermediates; ++k)
            dPrice_dIntermediates[k] /= static_cast<double>(nrTrails);

        // =====================================================================
        // Phase 4: Apply chain rule to get market sensitivities
        // =====================================================================
        std::vector<double> finalDerivatives(numMarketQuotes);
        applyChainRule(jacobian.data(), dPrice_dIntermediates.data(), finalDerivatives.data(),
                       numIntermediates, numMarketQuotes);

        auto t_end = Clock::now();

        if (iter >= warmup)
        {
            times.push_back(DurationMs(t_end - t_start).count());
        }

        (void)finalDerivatives;
        (void)mcPrice;
    }

    mean = computeMean(times);
    stddev = computeStddev(times);
}

void runJITBenchmark(const BenchmarkConfig& config, const LMMSetup& setup,
                     Size nrTrails, size_t warmup, size_t bench,
                     double& mean, double& stddev)
{
    runJITBenchmarkImpl<xad::forge::ForgeBackend<double>>(
        config, setup, nrTrails, warmup, bench, mean, stddev);
}

// ============================================================================
// Forge JIT-AVX Benchmark (Single-Curve)
// ============================================================================

void runJITAVXBenchmark(const BenchmarkConfig& config, const LMMSetup& setup,
                        Size nrTrails, size_t warmup, size_t bench,
                        double& mean, double& stddev)
{
    runJITBenchmarkImpl<xad::forge::ForgeBackendAVX<double>>(
        config, setup, nrTrails, warmup, bench, mean, stddev);
}

// ============================================================================
// Forge JIT Benchmark (Dual-Curve) - Template Implementation
// ============================================================================

template <typename BackendType>
void runJITBenchmarkDualCurveImpl(const BenchmarkConfig& config, const LMMSetup& setup,
                                   Size nrTrails, size_t warmup, size_t bench,
                                   double& mean, double& stddev,
                                   double& phase1_curve_mean, double& phase2_jacobian_mean,
                                   double& phase3_compile_mean)
{
    std::vector<double> times;
    std::vector<double> phase1_times;  // Curve bootstrap
    std::vector<double> phase2_times;  // Jacobian computation
    std::vector<double> phase3_times;  // JIT record + compile

    Size numMarketQuotes = config.numMarketQuotes();
    // Intermediates: forward rates + swap rate + OIS discount factors
    Size numIntermediates = config.size + 1 + config.size;  // 2*size + 1

    for (size_t iter = 0; iter < warmup + bench; ++iter)
    {
        auto t_start = Clock::now();

        // =====================================================================
        // Phase 1: XAD tape - dual-curve bootstrap
        // =====================================================================
        tape_type tape;

        // Register forecasting curve inputs
        std::vector<RealAD> depositRates(config.numDeposits);
        std::vector<RealAD> swapRatesAD(config.numSwaps);
        for (Size idx = 0; idx < config.numDeposits; ++idx)
            depositRates[idx] = config.depoRates[idx];
        for (Size idx = 0; idx < config.numSwaps; ++idx)
            swapRatesAD[idx] = config.swapRates[idx];

        // Register discounting curve inputs (OIS)
        std::vector<RealAD> oisDepoRates(config.numOisDeposits);
        std::vector<RealAD> oisSwapRatesAD(config.numOisSwaps);
        for (Size idx = 0; idx < config.numOisDeposits; ++idx)
            oisDepoRates[idx] = config.oisDepoRates[idx];
        for (Size idx = 0; idx < config.numOisSwaps; ++idx)
            oisSwapRatesAD[idx] = config.oisSwapRates[idx];

        tape.registerInputs(depositRates);
        tape.registerInputs(swapRatesAD);
        tape.registerInputs(oisDepoRates);
        tape.registerInputs(oisSwapRatesAD);
        tape.newRecording();

        // Build FORECASTING curve (Euribor deposits + swaps)
        RelinkableHandle<YieldTermStructure> euriborTS;
        auto euribor6m = ext::make_shared<Euribor6M>(euriborTS);
        euribor6m->addFixing(Date(2, September, 2005), 0.04);

        std::vector<ext::shared_ptr<RateHelper>> forecastingInstruments;
        for (Size idx = 0; idx < config.numDeposits; ++idx)
        {
            auto depoQuote = ext::make_shared<SimpleQuote>(depositRates[idx]);
            forecastingInstruments.push_back(ext::make_shared<DepositRateHelper>(
                Handle<Quote>(depoQuote), config.depoTenors[idx], setup.fixingDays,
                setup.calendar, ModifiedFollowing, true, setup.dayCounter));
        }
        for (Size idx = 0; idx < config.numSwaps; ++idx)
        {
            auto swapQuote = ext::make_shared<SimpleQuote>(swapRatesAD[idx]);
            forecastingInstruments.push_back(ext::make_shared<SwapRateHelper>(
                Handle<Quote>(swapQuote), config.swapTenors[idx],
                setup.calendar, Annual, Unadjusted, Thirty360(Thirty360::BondBasis),
                euribor6m));
        }

        auto forecastingCurve = ext::make_shared<PiecewiseYieldCurve<ZeroYield, Linear>>(
            setup.settlementDate, forecastingInstruments, setup.dayCounter);
        forecastingCurve->enableExtrapolation();
        euriborTS.linkTo(forecastingCurve);

        // Build DISCOUNTING curve (OIS deposits + swaps)
        RelinkableHandle<YieldTermStructure> oisTS;
        auto eonia = ext::make_shared<Eonia>(oisTS);

        std::vector<ext::shared_ptr<RateHelper>> discountingInstruments;
        for (Size idx = 0; idx < config.numOisDeposits; ++idx)
        {
            auto oisDepoQuote = ext::make_shared<SimpleQuote>(oisDepoRates[idx]);
            discountingInstruments.push_back(ext::make_shared<DepositRateHelper>(
                Handle<Quote>(oisDepoQuote), config.oisDepoTenors[idx], setup.fixingDays,
                setup.calendar, ModifiedFollowing, true, Actual360()));
        }
        for (Size idx = 0; idx < config.numOisSwaps; ++idx)
        {
            auto oisSwapQuote = ext::make_shared<SimpleQuote>(oisSwapRatesAD[idx]);
            discountingInstruments.push_back(ext::make_shared<OISRateHelper>(
                2, config.oisSwapTenors[idx], Handle<Quote>(oisSwapQuote), eonia));
        }

        auto discountingCurve = ext::make_shared<PiecewiseYieldCurve<ZeroYield, Linear>>(
            setup.settlementDate, discountingInstruments, setup.dayCounter);
        discountingCurve->enableExtrapolation();
        oisTS.linkTo(discountingCurve);

        // Extract zero rates for LMM from forecasting curve
        std::vector<Date> curveDates;
        std::vector<RealAD> zeroRates;
        curveDates.push_back(setup.settlementDate);
        zeroRates.push_back(forecastingCurve->zeroRate(setup.settlementDate, setup.dayCounter, Continuous).rate());
        Date endDate = setup.settlementDate + config.curveEndYears * Years;
        curveDates.push_back(endDate);
        zeroRates.push_back(forecastingCurve->zeroRate(endDate, setup.dayCounter, Continuous).rate());

        std::vector<Rate> zeroRates_ql;
        for (const auto& r : zeroRates) zeroRates_ql.push_back(r);

        // Build LMM process using forecasting curve
        RelinkableHandle<YieldTermStructure> termStructure;
        ext::shared_ptr<IborIndex> index(new Euribor6M(termStructure));
        index->addFixing(Date(2, September, 2005), 0.04);
        termStructure.linkTo(ext::make_shared<ZeroCurve>(curveDates, zeroRates_ql, setup.dayCounter));

        ext::shared_ptr<LiborForwardModelProcess> process(
            new LiborForwardModelProcess(config.size, index));
        process->setCovarParam(ext::shared_ptr<LfmCovarianceParameterization>(
            new LfmCovarianceProxy(
                ext::make_shared<LmLinearExponentialVolatilityModel>(
                    process->fixingTimes(), 0.291, 1.483, 0.116, 0.00001),
                ext::make_shared<LmExponentialCorrelationModel>(config.size, 0.5))));

        // Get swap rate (using OIS curve for discounting)
        ext::shared_ptr<VanillaSwap> fwdSwap(
            new VanillaSwap(Swap::Receiver, 1.0,
                            setup.schedule, 0.05, setup.dayCounter,
                            setup.schedule, index, 0.0, index->dayCounter()));
        fwdSwap->setPricingEngine(ext::make_shared<DiscountingSwapEngine>(
            Handle<YieldTermStructure>(discountingCurve)));
        RealAD swapRate = fwdSwap->fairRate();

        // Extract intermediates:
        // [0, size-1]: forward rates from LMM process
        // [size]: swap rate
        // [size+1, 2*size]: OIS discount factors at accrual end times
        Array initRates = process->initialValues();
        std::vector<RealAD> intermediates(numIntermediates);

        // Forward rates
        for (Size k = 0; k < config.size; ++k)
            intermediates[k] = initRates[k];

        // Swap rate
        intermediates[config.size] = swapRate;

        // OIS discount factors
        for (Size k = 0; k < config.size; ++k)
        {
            Time t = setup.accrualEnd[k];
            intermediates[config.size + 1 + k] = discountingCurve->discount(t);
        }

        // Register intermediates as outputs
        tape.registerOutputs(intermediates);

        auto t_curve_end = Clock::now();

        // =====================================================================
        // Phase 1b: Compute Jacobian via XAD adjoints
        // =====================================================================
        std::vector<double> jacobian(numIntermediates * numMarketQuotes);

        for (Size i = 0; i < numIntermediates; ++i)
        {
            tape.clearDerivatives();
            derivative(intermediates[i]) = 1.0;
            tape.computeAdjoints();

            Size col = 0;
            // Forecasting inputs
            for (Size j = 0; j < config.numDeposits; ++j)
                jacobian[i * numMarketQuotes + col++] = derivative(depositRates[j]);
            for (Size j = 0; j < config.numSwaps; ++j)
                jacobian[i * numMarketQuotes + col++] = derivative(swapRatesAD[j]);
            // Discounting inputs
            for (Size j = 0; j < config.numOisDeposits; ++j)
                jacobian[i * numMarketQuotes + col++] = derivative(oisDepoRates[j]);
            for (Size j = 0; j < config.numOisSwaps; ++j)
                jacobian[i * numMarketQuotes + col++] = derivative(oisSwapRatesAD[j]);
        }

        tape.clearAll();

        auto t_jacobian_end = Clock::now();

        // Extract intermediate values as plain doubles
        std::vector<double> intermediateValues(numIntermediates);
        for (Size k = 0; k < numIntermediates; ++k)
            intermediateValues[k] = value(intermediates[k]);

        // =====================================================================
        // Phase 2: JIT graph recording and compilation
        // =====================================================================
        auto backend = std::make_unique<BackendType>();
        xad::JITCompiler<double> jit(std::move(backend));

        // Register JIT inputs: forward rates, swap rate, OIS discount factors, and randoms
        std::vector<xad::AD> jit_initRates(config.size);
        xad::AD jit_swapRate;
        std::vector<xad::AD> jit_oisDiscounts(config.size);
        std::vector<xad::AD> jit_randoms(setup.fullGridRandoms);

        for (Size k = 0; k < config.size; ++k)
        {
            jit_initRates[k] = xad::AD(intermediateValues[k]);
            jit.registerInput(jit_initRates[k]);
        }
        jit_swapRate = xad::AD(intermediateValues[config.size]);
        jit.registerInput(jit_swapRate);

        for (Size k = 0; k < config.size; ++k)
        {
            jit_oisDiscounts[k] = xad::AD(intermediateValues[config.size + 1 + k]);
            jit.registerInput(jit_oisDiscounts[k]);
        }

        for (Size m = 0; m < setup.fullGridRandoms; ++m)
        {
            jit_randoms[m] = xad::AD(0.0);
            jit.registerInput(jit_randoms[m]);
        }

        jit.newRecording();

        // Record path evolution
        std::vector<xad::AD> asset(config.size);
        std::vector<xad::AD> assetAtExercise(config.size);
        for (Size k = 0; k < config.size; ++k)
            asset[k] = jit_initRates[k];

        for (Size step = 1; step <= setup.fullGridSteps; ++step)
        {
            Size offset = (step - 1) * setup.numFactors;
            Time t = setup.grid[step - 1];
            Time dt = setup.grid.dt(step - 1);

            Array dw(setup.numFactors);
            for (Size f = 0; f < setup.numFactors; ++f)
                dw[f] = jit_randoms[offset + f];

            Array asset_arr(config.size);
            for (Size k = 0; k < config.size; ++k)
                asset_arr[k] = asset[k];

            Array evolved = process->evolve(t, asset_arr, dt, dw);
            for (Size k = 0; k < config.size; ++k)
                asset[k] = evolved[k];

            if (step == setup.exerciseStep)
            {
                for (Size k = 0; k < config.size; ++k)
                    assetAtExercise[k] = asset[k];
            }
        }

        // Compute payoff using OIS discount factors for discounting
        std::vector<xad::AD> dis(config.size);
        xad::AD df = xad::AD(1.0);
        for (Size k = 0; k < config.size; ++k)
        {
            double accrual = setup.accrualEnd[k] - setup.accrualStart[k];
            df = df / (xad::AD(1.0) + assetAtExercise[k] * accrual);
            // Use OIS discount factors for dual-curve discounting
            dis[k] = jit_oisDiscounts[k];
        }

        xad::AD npv = xad::AD(0.0);
        for (Size m = config.i_opt; m < config.i_opt + config.j_opt; ++m)
        {
            double accrual = setup.accrualEnd[m] - setup.accrualStart[m];
            npv = npv + (jit_swapRate - assetAtExercise[m]) * accrual * dis[m];
        }

        // Payoff = max(npv, 0)
        xad::AD payoff = xad::less(npv, xad::AD(0.0)).If(xad::AD(0.0), npv);
        jit.registerOutput(payoff);

        // Compile the JIT kernel
        jit.compile();

        auto t_compile_end = Clock::now();

        // =====================================================================
        // Phase 4: Execute JIT kernel for all MC paths
        // =====================================================================
        const auto& graph = jit.getGraph();

        double mcPrice = 0.0;
        std::vector<double> dPrice_dIntermediates(numIntermediates, 0.0);

        for (Size n = 0; n < nrTrails; ++n)
        {
            // Set inputs
            for (Size k = 0; k < config.size; ++k)
                value(jit_initRates[k]) = intermediateValues[k];
            value(jit_swapRate) = intermediateValues[config.size];
            for (Size k = 0; k < config.size; ++k)
                value(jit_oisDiscounts[k]) = intermediateValues[config.size + 1 + k];
            for (Size m = 0; m < setup.fullGridRandoms; ++m)
                value(jit_randoms[m]) = setup.allRandoms[n][m];

            // Execute forward
            double payoff_value;
            jit.forward(&payoff_value);
            mcPrice += payoff_value;

            // Accumulate gradients w.r.t. intermediates
            jit.clearDerivatives();
            jit.setDerivative(graph.output_ids[0], 1.0);
            jit.computeAdjoints();

            // Forward rates
            for (Size k = 0; k < config.size; ++k)
                dPrice_dIntermediates[k] += jit.derivative(graph.input_ids[k]);
            // Swap rate
            dPrice_dIntermediates[config.size] += jit.derivative(graph.input_ids[config.size]);
            // OIS discount factors
            for (Size k = 0; k < config.size; ++k)
                dPrice_dIntermediates[config.size + 1 + k] += jit.derivative(graph.input_ids[config.size + 1 + k]);
        }

        // Average
        mcPrice /= static_cast<double>(nrTrails);
        for (Size k = 0; k < numIntermediates; ++k)
            dPrice_dIntermediates[k] /= static_cast<double>(nrTrails);

        // =====================================================================
        // Phase 5: Apply chain rule to get market sensitivities
        // =====================================================================
        std::vector<double> finalDerivatives(numMarketQuotes);
        applyChainRule(jacobian.data(), dPrice_dIntermediates.data(), finalDerivatives.data(),
                       numIntermediates, numMarketQuotes);

        auto t_end = Clock::now();

        if (iter >= warmup)
        {
            times.push_back(DurationMs(t_end - t_start).count());
            phase1_times.push_back(DurationMs(t_curve_end - t_start).count());
            phase2_times.push_back(DurationMs(t_jacobian_end - t_curve_end).count());
            phase3_times.push_back(DurationMs(t_compile_end - t_jacobian_end).count());
        }

        (void)finalDerivatives;
        (void)mcPrice;
    }

    mean = computeMean(times);
    stddev = computeStddev(times);
    phase1_curve_mean = computeMean(phase1_times);
    phase2_jacobian_mean = computeMean(phase2_times);
    phase3_compile_mean = computeMean(phase3_times);
}

void runJITBenchmarkDualCurve(const BenchmarkConfig& config, const LMMSetup& setup,
                               Size nrTrails, size_t warmup, size_t bench,
                               double& mean, double& stddev,
                               double& phase1_curve_mean, double& phase2_jacobian_mean,
                               double& phase3_compile_mean)
{
    runJITBenchmarkDualCurveImpl<xad::forge::ForgeBackend<double>>(
        config, setup, nrTrails, warmup, bench, mean, stddev,
        phase1_curve_mean, phase2_jacobian_mean, phase3_compile_mean);
}

// ============================================================================
// Forge JIT-AVX Benchmark (Dual-Curve)
// ============================================================================

void runJITAVXBenchmarkDualCurve(const BenchmarkConfig& config, const LMMSetup& setup,
                                  Size nrTrails, size_t warmup, size_t bench,
                                  double& mean, double& stddev,
                                  double& phase1_curve_mean, double& phase2_jacobian_mean,
                                  double& phase3_compile_mean)
{
    runJITBenchmarkDualCurveImpl<xad::forge::ForgeBackendAVX<double>>(
        config, setup, nrTrails, warmup, bench, mean, stddev,
        phase1_curve_mean, phase2_jacobian_mean, phase3_compile_mean);
}

#endif // QLRISKS_HAS_FORGE

// ============================================================================
// Main AAD Benchmark Runner (Single-Curve)
// ============================================================================

std::vector<TimingResult> runAADBenchmark(const BenchmarkConfig& config,
                                           bool quickMode, bool xadOnly)
{
    std::vector<TimingResult> results;

    // Setup LMM
    LMMSetup setup(config);

    std::cout << "================================================================================\n";
    std::cout << "  RUNNING AAD BENCHMARKS\n";
    std::cout << "================================================================================\n";
    std::cout << "\n";

    for (size_t tc = 0; tc < config.pathCounts.size(); ++tc)
    {
        int paths = config.pathCounts[tc];
        Size nrTrails = static_cast<Size>(paths);

        TimingResult result;
        result.pathCount = paths;

        std::cout << "  [" << (tc + 1) << "/" << config.pathCounts.size() << "] "
                  << formatPathCount(paths) << " paths " << std::flush;

        // For high path counts, skip warmup and use fewer iterations
        size_t warmup, bench;
        if (nrTrails >= 10000) {
            warmup = 0;
            bench = 2;
        } else {
            warmup = quickMode ? 1 : config.warmupIterations;
            bench = quickMode ? 2 : config.benchmarkIterations;
        }

        // XAD tape
        runXADBenchmark(config, setup, nrTrails, warmup, bench,
                        result.xad_mean, result.xad_std);
        result.xad_enabled = true;
        std::cout << "XAD=" << std::fixed << std::setprecision(1) << result.xad_mean << "ms ";

#if defined(QLRISKS_HAS_FORGE)
        if (!xadOnly)
        {
            // Forge JIT
            runJITBenchmark(config, setup, nrTrails, warmup, bench,
                            result.jit_mean, result.jit_std);
            result.jit_enabled = true;
            std::cout << "JIT=" << result.jit_mean << "ms ";

            // Forge JIT-AVX
            runJITAVXBenchmark(config, setup, nrTrails, warmup, bench,
                               result.jit_avx_mean, result.jit_avx_std);
            result.jit_avx_enabled = true;
            std::cout << "JIT-AVX=" << result.jit_avx_mean << "ms";
        }
#else
        (void)xadOnly;
#endif

        std::cout << "\n";
        results.push_back(result);
    }

    std::cout << "\n";
    return results;
}

// ============================================================================
// Main AAD Benchmark Runner (Dual-Curve / Production)
// ============================================================================

std::vector<TimingResult> runAADBenchmarkDualCurve(const BenchmarkConfig& config,
                                                    bool quickMode, bool xadOnly)
{
    std::vector<TimingResult> results;

    // Setup LMM
    LMMSetup setup(config);

    std::cout << "================================================================================\n";
    std::cout << "  RUNNING AAD BENCHMARKS (Dual-Curve)\n";
    std::cout << "================================================================================\n";
    std::cout << "\n";

    for (size_t tc = 0; tc < config.pathCounts.size(); ++tc)
    {
        int paths = config.pathCounts[tc];
        Size nrTrails = static_cast<Size>(paths);

        TimingResult result;
        result.pathCount = paths;

        std::cout << "  [" << (tc + 1) << "/" << config.pathCounts.size() << "] "
                  << formatPathCount(paths) << " paths (" << config.numMarketQuotes()
                  << " sensitivities) " << std::flush;

        // For high path counts, skip warmup and use fewer iterations
        size_t warmup, bench;
        if (nrTrails >= 10000) {
            warmup = 0;
            bench = 2;
        } else {
            warmup = quickMode ? 1 : config.warmupIterations;
            bench = quickMode ? 2 : config.benchmarkIterations;
        }

        // XAD tape (dual-curve)
        runXADBenchmarkDualCurve(config, setup, nrTrails, warmup, bench,
                                  result.xad_mean, result.xad_std);
        result.xad_enabled = true;
        std::cout << "XAD=" << std::fixed << std::setprecision(1) << result.xad_mean << "ms ";

#if defined(QLRISKS_HAS_FORGE)
        if (!xadOnly)
        {
            // Forge JIT (dual-curve)
            double jit_p1 = 0, jit_p2 = 0, jit_p3 = 0;
            runJITBenchmarkDualCurve(config, setup, nrTrails, warmup, bench,
                                      result.jit_mean, result.jit_std,
                                      jit_p1, jit_p2, jit_p3);
            result.jit_enabled = true;
            result.jit_phase1_curve_mean = jit_p1;
            result.jit_phase2_jacobian_mean = jit_p2;
            result.jit_phase3_compile_mean = jit_p3;
            result.jit_fixed_mean = jit_p1 + jit_p2 + jit_p3;
            std::cout << "JIT=" << result.jit_mean << "ms ";

            // Forge JIT-AVX (dual-curve)
            double avx_p1 = 0, avx_p2 = 0, avx_p3 = 0;
            runJITAVXBenchmarkDualCurve(config, setup, nrTrails, warmup, bench,
                                         result.jit_avx_mean, result.jit_avx_std,
                                         avx_p1, avx_p2, avx_p3);
            result.jit_avx_enabled = true;
            // Note: AVX uses same setup phases, so we just use JIT's phase timings
            std::cout << "JIT-AVX=" << result.jit_avx_mean << "ms";
        }
#else
        (void)xadOnly;
#endif

        std::cout << "\n";
        results.push_back(result);
    }

    std::cout << "\n";
    return results;
}

// ============================================================================
// Output Results in Machine-Parseable Format
// ============================================================================

void outputResultsForParsing(const std::vector<TimingResult>& results,
                              const std::string& configId)
{
    // XAD results
    std::cout << "XAD_" << configId << ":";
    for (size_t i = 0; i < results.size(); ++i)
    {
        const auto& r = results[i];
        if (i > 0) std::cout << ";";
        std::cout << r.pathCount << "=" << r.xad_mean << "," << r.xad_std
                  << "," << (r.xad_enabled ? "1" : "0");
    }
    std::cout << std::endl;

#if defined(QLRISKS_HAS_FORGE)
    // JIT results (now includes fixed cost: mean,std,enabled,fixed_cost)
    std::cout << "JIT_" << configId << ":";
    for (size_t i = 0; i < results.size(); ++i)
    {
        const auto& r = results[i];
        if (i > 0) std::cout << ";";
        std::cout << r.pathCount << "=" << r.jit_mean << "," << r.jit_std
                  << "," << (r.jit_enabled ? "1" : "0")
                  << "," << r.jit_fixed_mean;
    }
    std::cout << std::endl;

    // JIT-AVX results (now includes fixed cost: mean,std,enabled,fixed_cost)
    std::cout << "JITAVX_" << configId << ":";
    for (size_t i = 0; i < results.size(); ++i)
    {
        const auto& r = results[i];
        if (i > 0) std::cout << ";";
        std::cout << r.pathCount << "=" << r.jit_avx_mean << "," << r.jit_avx_std
                  << "," << (r.jit_avx_enabled ? "1" : "0")
                  << "," << r.jit_fixed_mean;  // Same fixed cost as JIT
    }
    std::cout << std::endl;

    // JIT phase breakdown (one-time costs: phase1_curve, phase2_jacobian, phase3_compile)
    // Output from first result that has JIT enabled (phases are same for all path counts)
    for (const auto& r : results)
    {
        if (r.jit_enabled)
        {
            std::cout << "JIT_PHASES_" << configId << ":"
                      << r.jit_phase1_curve_mean << ","
                      << r.jit_phase2_jacobian_mean << ","
                      << r.jit_phase3_compile_mean << std::endl;
            break;
        }
    }
#endif
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[])
{
    bool runLite = false;
    bool runLiteExtended = false;
    bool runProduction = false;
    bool runAll = true;  // Default: run lite and lite-extended (not production)
    bool quickMode = false;
    bool xadOnly = false;

    // Parse arguments
    for (int i = 1; i < argc; ++i)
    {
        if (strcmp(argv[i], "--lite") == 0)
        {
            runLite = true;
            runAll = false;
        }
        else if (strcmp(argv[i], "--lite-extended") == 0)
        {
            runLiteExtended = true;
            runAll = false;
        }
        else if (strcmp(argv[i], "--production") == 0)
        {
            runProduction = true;
            runAll = false;
        }
        else if (strcmp(argv[i], "--all") == 0)
        {
            runLite = true;
            runLiteExtended = true;
            runProduction = true;
            runAll = false;
        }
        else if (strcmp(argv[i], "--quick") == 0)
        {
            quickMode = true;
        }
        else if (strcmp(argv[i], "--xad-only") == 0)
        {
            xadOnly = true;
        }
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0)
        {
            std::cout << "Usage: " << argv[0] << " [options]\n";
            std::cout << "Options:\n";
            std::cout << "  --lite           Run lite benchmark (1Y into 1Y, 9 sensitivities)\n";
            std::cout << "  --lite-extended  Run lite-extended benchmark (5Y into 5Y, 14 sensitivities)\n";
            std::cout << "  --production     Run production benchmark (5Y into 5Y dual-curve, 47 sensitivities)\n";
            std::cout << "  --all            Run all benchmarks including production\n";
            std::cout << "  --quick          Quick mode (fewer iterations)\n";
            std::cout << "  --xad-only       Run only XAD tape (no JIT)\n";
            std::cout << "  --help           Show this message\n";
            std::cout << "\n";
            std::cout << "Default: runs lite and lite-extended (not production)\n";
            return 0;
        }
    }

    // Default behavior: run lite and lite-extended
    if (runAll)
    {
        runLite = true;
        runLiteExtended = true;
        runProduction = false;  // Production opt-in only
    }

    printHeader();
    printEnvironment();

    int benchmarkNum = 1;

    if (runLite)
    {
        BenchmarkConfig liteConfig;
        printBenchmarkHeader(liteConfig, benchmarkNum++);

        auto results = runAADBenchmark(liteConfig, quickMode, xadOnly);
        printResultsTable(results);
        printResultsFooter(liteConfig);
        outputResultsForParsing(results, liteConfig.configId);
    }

    if (runLiteExtended)
    {
        BenchmarkConfig liteExtConfig;
        liteExtConfig.setLiteExtendedConfig();
        printBenchmarkHeader(liteExtConfig, benchmarkNum++);

        auto results = runAADBenchmark(liteExtConfig, quickMode, xadOnly);
        printResultsTable(results);
        printResultsFooter(liteExtConfig);
        outputResultsForParsing(results, liteExtConfig.configId);
    }

    if (runProduction)
    {
        BenchmarkConfig prodConfig;
        prodConfig.setProductionConfig();
        printBenchmarkHeader(prodConfig, benchmarkNum++);

        auto results = runAADBenchmarkDualCurve(prodConfig, quickMode, xadOnly);
        printResultsTable(results);
        printResultsFooter(prodConfig);
        outputResultsForParsing(results, prodConfig.configId);
    }

    printFooter();

    return 0;
}
