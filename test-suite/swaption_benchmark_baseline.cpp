/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*
 Copyright (C) 2025 Xcelerit Computing Limited

 This file is part of QuantLib-Risks / XAD integration.

 Swaption Baseline Benchmark - XAD tape-based reverse-mode AD performance.
 This benchmark uses only standard XAD tape operations (no JIT/Forge).

 Benchmarks:
 1. Simple Swaption (1Y into 1Y) - basic scaling test
 2. Larger Swaption (5Y into 5Y) - attempts to approximate a realistic setup

 This serves as a baseline for comparison with JIT-accelerated versions.
*/

#include "toplevelfixture.hpp"
#include "utilities_xad.hpp"

// QuantLib includes
#include <ql/indexes/ibor/eonia.hpp>
#include <ql/indexes/ibor/euribor.hpp>
#include <ql/instruments/vanillaswap.hpp>
#include <ql/instruments/swaption.hpp>
#include <ql/math/interpolations/cubicinterpolation.hpp>
#include <ql/pricingengines/swap/discountingswapengine.hpp>
#include <ql/termstructures/yield/oisratehelper.hpp>
#include <ql/termstructures/yield/piecewiseyieldcurve.hpp>
#include <ql/termstructures/yield/ratehelpers.hpp>
#include <ql/termstructures/yield/zerocurve.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/daycounters/actual360.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/time/daycounters/thirty360.hpp>

// LMM Monte Carlo includes
#include <ql/legacy/libormarketmodels/lfmcovarproxy.hpp>
#include <ql/legacy/libormarketmodels/lfmswaptionengine.hpp>
#include <ql/legacy/libormarketmodels/liborforwardmodel.hpp>
#include <ql/legacy/libormarketmodels/lmexpcorrmodel.hpp>
#include <ql/legacy/libormarketmodels/lmlinexpvolmodel.hpp>
#include <ql/math/randomnumbers/rngtraits.hpp>
#include <ql/math/statistics/generalstatistics.hpp>
#include <ql/methods/montecarlo/multipathgenerator.hpp>

#include <boost/test/unit_test.hpp>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <map>


using namespace QuantLib;
using namespace boost::unit_test_framework;

BOOST_FIXTURE_TEST_SUITE(QuantLibRisksTests, TopLevelFixture)

BOOST_AUTO_TEST_SUITE(SwaptionBenchmarkTests)

//////////////////////////////////////////////////////////////////////////////
// Helper: Create IborIndex with ZeroCurve (from libormarketmodel.cpp)
//////////////////////////////////////////////////////////////////////////////

namespace {

ext::shared_ptr<IborIndex> makeIndex(std::vector<Date> dates,
                                     const std::vector<Rate>& rates) {
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

//////////////////////////////////////////////////////////////////////////////
// High-performance chain rule: result = jacobian^T * derivatives
// For hybrid AD workflows: one stage computes dOutput/dIntermediate,
// Tape computes dIntermediate/dInput, chain rule gives dOutput/dInput.
//////////////////////////////////////////////////////////////////////////////

/// Apply chain rule: result[j] = sum_i(derivatives[i] * jacobian[i * numInputs + j])
/// @param jacobian      Row-major flat array [numIntermediates x numInputs]
/// @param derivatives   Vector [numIntermediates] (dOutput/dIntermediate)
/// @param result        Output vector [numInputs] (dOutput/dInput) - zeroed first
/// @param numIntermediates  Number of intermediate variables
/// @param numInputs     Number of input variables
inline void applyChainRule(const double* __restrict jacobian,
                           const double* __restrict derivatives,
                           double* __restrict result,
                           std::size_t numIntermediates,
                           std::size_t numInputs)
{
    // Zero result
    for (std::size_t j = 0; j < numInputs; ++j)
        result[j] = 0.0;

    // Accumulate: result[j] += derivatives[i] * jacobian[i,j]
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

} // anonymous namespace

//////////////////////////////////////////////////////////////////////////////
// Benchmark 1: Simple Swaption Scaling (1Y into 1Y)
//////////////////////////////////////////////////////////////////////////////

BOOST_AUTO_TEST_CASE(testBenchmark_SimpleSwaptionScaling)
{
    BOOST_TEST_MESSAGE("Running Simple Swaption Scaling Benchmark...");

    std::cout << "\n";
    std::cout << "=============================================================================\n";
    std::cout << "  XAD BASELINE BENCHMARK: Simple Swaption (1Y into 1Y)\n";
    std::cout << "=============================================================================\n";
    std::cout << std::endl;
    std::cout << "  This benchmark measures XAD tape-based AD performance for\n";
    std::cout << "  computing sensitivities in Monte Carlo swaption pricing.\n";
    std::cout << std::endl;
    std::cout << "  TWO MC IMPLEMENTATIONS:\n";
    std::cout << "    QL = QuantLib's MultiPathGenerator (full path storage)\n";
    std::cout << "    RR = Direct process->evolve() calls (explicit inputs)\n";
    std::cout << std::endl;
    std::cout << "  APPROACHES TESTED:\n";
    std::cout << "    XAD(QL)   - XAD tape + QuantLib MultiPathGenerator\n";
    std::cout << "    XAD(RR)   - XAD tape + direct evolve\n";
    std::cout << std::endl;
    std::cout << "  INSTRUMENT:\n";
    std::cout << "    European payer swaption: 1Y option into 1Y swap\n";
    std::cout << "    Model: LIBOR Market Model (LMM) with lognormal forwards\n";
    std::cout << "    Sensitivities: dPrice/dMarketQuotes (9 inputs)\n";
    std::cout << std::endl;

    using Clock = std::chrono::high_resolution_clock;
    using Duration = std::chrono::duration<double, std::milli>;

    // Market data setup (same as Stage 4)
    Calendar calendar = TARGET();
    Date todaysDate(4, September, 2005);
    Settings::instance().evaluationDate() = todaysDate;
    Integer fixingDays = 2;
    Date settlementDate = calendar.adjust(calendar.advance(todaysDate, fixingDays, Days));
    DayCounter dayCounter = Actual360();

    Size numDeposits = 4;
    Size numSwaps = 5;
    std::vector<Period> depoTenors = {1 * Days, 1 * Months, 3 * Months, 6 * Months};
    std::vector<Period> swapTenors = {1 * Years, 2 * Years, 3 * Years, 4 * Years, 5 * Years};

    std::vector<double> depoRates_val = {0.0350, 0.0365, 0.0380, 0.0400};
    std::vector<double> swapRates_val = {0.0420, 0.0480, 0.0520, 0.0550, 0.0575};

    Size numMarketQuotes = numDeposits + numSwaps;

    // LMM parameters
    Size size = 10;
    Size i_opt = 2;
    Size j_opt = 2;
    Size steps = 8;

    // Build base curve and process setup
    std::vector<Rate> baseZeroRates = {0.0350, 0.0575};
    std::vector<Date> baseDates = {settlementDate, settlementDate + 6 * Years};
    auto baseIndex = makeIndex(baseDates, baseZeroRates);

    ext::shared_ptr<LiborForwardModelProcess> baseProcess(
        new LiborForwardModelProcess(size, baseIndex));
    ext::shared_ptr<LmCorrelationModel> baseCorrModel(
        new LmExponentialCorrelationModel(size, 0.5));
    ext::shared_ptr<LmVolatilityModel> baseVolaModel(
        new LmLinearExponentialVolatilityModel(baseProcess->fixingTimes(),
                                               0.291, 1.483, 0.116, 0.00001));
    baseProcess->setCovarParam(ext::shared_ptr<LfmCovarianceParameterization>(
        new LfmCovarianceProxy(baseVolaModel, baseCorrModel)));

    // Grid and timing setup
    std::vector<Time> fixingTimes = baseProcess->fixingTimes();
    TimeGrid grid(fixingTimes.begin(), fixingTimes.end(), steps);

    std::vector<Size> location;
    for (Size idx = 0; idx < fixingTimes.size(); ++idx) {
        location.push_back(
            std::find(grid.begin(), grid.end(), fixingTimes[idx]) - grid.begin());
    }

    Size numFactors = baseProcess->factors();
    Size exerciseStep = location[i_opt];
    Size fullGridSteps = grid.size() - 1;
    Size fullGridRandoms = fullGridSteps * numFactors;

    // Types for MC generators
    typedef PseudoRandom::rsg_type rsg_type;
    typedef MultiPathGenerator<rsg_type>::sample_type sample_type;

    // Get fair swap rate (same approach as Stage 4)
    BusinessDayConvention convention = baseIndex->businessDayConvention();
    Date settlement = baseIndex->forwardingTermStructure()->referenceDate();
    Date fwdStart = settlement + Period(6 * i_opt, Months);
    Date fwdMaturity = fwdStart + Period(6 * j_opt, Months);

    Schedule schedule(fwdStart, fwdMaturity, baseIndex->tenor(), calendar,
                      convention, convention, DateGeneration::Forward, false);

    // Accrual periods
    std::vector<double> accrualStart(size), accrualEnd(size);
    for (Size k = 0; k < size; ++k) {
        accrualStart[k] = value(baseProcess->accrualStartTimes()[k]);
        accrualEnd[k] = value(baseProcess->accrualEndTimes()[k]);
    }

    Size numIntermediates = size + 1;

    // Helper lambda for curve building
    auto buildCurveAndProcess = [&](const std::vector<double>& depoRates,
                                    const std::vector<double>& swapRates,
                                    std::vector<Date>& curveDates,
                                    std::vector<Rate>& zeroRates) {
        RelinkableHandle<YieldTermStructure> euriborTS;
        auto euribor6m = ext::make_shared<Euribor6M>(euriborTS);
        euribor6m->addFixing(Date(2, September, 2005), 0.04);

        std::vector<ext::shared_ptr<RateHelper>> instruments;
        for (Size idx = 0; idx < numDeposits; ++idx) {
            auto depoQuote = ext::make_shared<SimpleQuote>(depoRates[idx]);
            instruments.push_back(ext::make_shared<DepositRateHelper>(
                Handle<Quote>(depoQuote), depoTenors[idx], fixingDays,
                calendar, ModifiedFollowing, true, dayCounter));
        }
        for (Size idx = 0; idx < numSwaps; ++idx) {
            auto swapQuote = ext::make_shared<SimpleQuote>(swapRates[idx]);
            instruments.push_back(ext::make_shared<SwapRateHelper>(
                Handle<Quote>(swapQuote), swapTenors[idx],
                calendar, Annual, Unadjusted, Thirty360(Thirty360::BondBasis),
                euribor6m));
        }

        auto yieldCurve = ext::make_shared<PiecewiseYieldCurve<ZeroYield, Linear>>(
            settlementDate, instruments, dayCounter);
        yieldCurve->enableExtrapolation();

        curveDates.clear();
        zeroRates.clear();
        curveDates.push_back(settlementDate);
        zeroRates.push_back(yieldCurve->zeroRate(settlementDate, dayCounter, Continuous).rate());
        Date endDate = settlementDate + 6 * Years;
        curveDates.push_back(endDate);
        zeroRates.push_back(yieldCurve->zeroRate(endDate, dayCounter, Continuous).rate());
    };

    // Test cases: different path counts
    std::vector<Size> pathCounts = {10, 100, 1000, 10000};
    std::vector<std::string> testCaseNames = {"10", "100", "1K", "10K"};

    // Results storage
    struct ScalingResult {
        double xad_ql_total;
        double xad_rrs_total;
        double xad_ql_std;
        double xad_rrs_std;
    };
    std::vector<ScalingResult> results(pathCounts.size());

    Size warmupIterations = 5;
    Size benchmarkIterations = 10;

    std::cout << "  CONFIGURATION:\n";
    std::cout << "    Market inputs:    " << numMarketQuotes << " (4 deposits + 5 swaps)\n";
    std::cout << "    Forward rates:    " << size << " (semi-annual to 5Y)\n";
    std::cout << "    Time steps:       " << fullGridSteps << "\n";
    std::cout << "    MC paths tested:  10, 100, 1K, 10K\n";
    std::cout << "    Warmup/Bench:     " << warmupIterations << "/" << benchmarkIterations << " iterations\n";
    std::cout << std::endl;

    for (Size tc = 0; tc < pathCounts.size(); ++tc) {
        Size nrTrails = pathCounts[tc];
        std::cout << "  Running test case: " << testCaseNames[tc] << " (" << nrTrails << " paths)..." << std::flush;

        // Pre-generate random numbers for this path count
        rsg_type rsg_base = PseudoRandom::make_sequence_generator(fullGridRandoms, BigNatural(42));
        std::vector<std::vector<double>> allRandoms(nrTrails);
        for (Size n = 0; n < nrTrails; ++n) {
            allRandoms[n].resize(fullGridRandoms);
            const auto& seq = rsg_base.nextSequence();
            for (Size m = 0; m < fullGridRandoms; ++m) {
                allRandoms[n][m] = value(seq.value[m]);
            }
        }

        // Timing accumulators
        std::vector<double> xad_ql_times, xad_rrs_times;

        for (Size iter = 0; iter < warmupIterations + benchmarkIterations; ++iter) {
            bool recordTiming = (iter >= warmupIterations);

            // =================================================================
            // XAD (QuantLib) - MultiPathGenerator with XAD tape
            // =================================================================
            {
                auto t_start = Clock::now();

                using tape_type = Real::tape_type;
                tape_type tape;

                std::vector<Real> depositRates(numDeposits);
                std::vector<Real> swapRates(numSwaps);
                for (Size idx = 0; idx < numDeposits; ++idx) depositRates[idx] = depoRates_val[idx];
                for (Size idx = 0; idx < numSwaps; ++idx) swapRates[idx] = swapRates_val[idx];

                tape.registerInputs(depositRates);
                tape.registerInputs(swapRates);
                tape.newRecording();

                std::vector<Date> curveDates;
                std::vector<Rate> zeroRates;
                std::vector<double> depoRates_dbl(numDeposits), swapRates_dbl(numSwaps);
                for (Size idx = 0; idx < numDeposits; ++idx) depoRates_dbl[idx] = value(depositRates[idx]);
                for (Size idx = 0; idx < numSwaps; ++idx) swapRates_dbl[idx] = value(swapRates[idx]);
                buildCurveAndProcess(depoRates_dbl, swapRates_dbl, curveDates, zeroRates);

                RelinkableHandle<YieldTermStructure> termStructure;
                ext::shared_ptr<IborIndex> index(new Euribor6M(termStructure));
                index->addFixing(Date(2, September, 2005), 0.04);

                std::vector<Real> zeroRates_real(zeroRates.begin(), zeroRates.end());
                termStructure.linkTo(ext::make_shared<ZeroCurve>(curveDates, zeroRates_real, dayCounter));

                ext::shared_ptr<LiborForwardModelProcess> process(
                    new LiborForwardModelProcess(size, index));
                process->setCovarParam(ext::shared_ptr<LfmCovarianceParameterization>(
                    new LfmCovarianceProxy(
                        ext::make_shared<LmLinearExponentialVolatilityModel>(process->fixingTimes(), 0.291, 1.483, 0.116, 0.00001),
                        ext::make_shared<LmExponentialCorrelationModel>(size, 0.5))));

                ext::shared_ptr<VanillaSwap> fwdSwap(
                    new VanillaSwap(Swap::Receiver, 1.0,
                                    schedule, 0.05, dayCounter,
                                    schedule, index, 0.0, index->dayCounter()));
                fwdSwap->setPricingEngine(ext::make_shared<DiscountingSwapEngine>(
                    index->forwardingTermStructure()));
                Real swapRate = fwdSwap->fairRate();

                rsg_type rsg_ql = PseudoRandom::make_sequence_generator(fullGridRandoms, BigNatural(42));
                MultiPathGenerator<rsg_type> generator_ql(process, grid, rsg_ql, false);

                Real price = Real(0.0);
                for (Size n = 0; n < nrTrails; ++n) {
                    sample_type path = (n % 2) != 0U ? generator_ql.antithetic() : generator_ql.next();

                    Array mcRates(size);
                    for (Size k = 0; k < size; ++k) {
                        mcRates[k] = path.value[k][exerciseStep];
                    }

                    Array dis(size);
                    Real df = Real(1.0);
                    for (Size k = 0; k < size; ++k) {
                        Real accrual = accrualEnd[k] - accrualStart[k];
                        df = df / (Real(1.0) + mcRates[k] * accrual);
                        dis[k] = df;
                    }

                    Real npv = Real(0.0);
                    for (Size m = i_opt; m < i_opt + j_opt; ++m) {
                        Real accrual = accrualEnd[m] - accrualStart[m];
                        npv += (swapRate - mcRates[m]) * accrual * dis[m];
                    }
                    price += max(npv, Real(0.0));
                }
                price /= Real(static_cast<double>(nrTrails));

                tape.registerOutput(price);
                derivative(price) = 1.0;
                tape.computeAdjoints();

                auto t_end = Clock::now();
                if (recordTiming) {
                    xad_ql_times.push_back(Duration(t_end - t_start).count());
                }
                tape.deactivate();
            }

            // =================================================================
            // XAD (RR) - Full grid with XAD tape
            // =================================================================
            {
                auto t_start = Clock::now();

                using tape_type = Real::tape_type;
                tape_type tape;

                std::vector<Real> depositRates(numDeposits);
                std::vector<Real> swapRates(numSwaps);
                for (Size idx = 0; idx < numDeposits; ++idx) depositRates[idx] = depoRates_val[idx];
                for (Size idx = 0; idx < numSwaps; ++idx) swapRates[idx] = swapRates_val[idx];

                tape.registerInputs(depositRates);
                tape.registerInputs(swapRates);
                tape.newRecording();

                RelinkableHandle<YieldTermStructure> euriborTS;
                auto euribor6m = ext::make_shared<Euribor6M>(euriborTS);
                euribor6m->addFixing(Date(2, September, 2005), 0.04);

                std::vector<ext::shared_ptr<RateHelper>> instruments;
                for (Size idx = 0; idx < numDeposits; ++idx) {
                    auto depoQuote = ext::make_shared<SimpleQuote>(depositRates[idx]);
                    instruments.push_back(ext::make_shared<DepositRateHelper>(
                        Handle<Quote>(depoQuote), depoTenors[idx], fixingDays,
                        calendar, ModifiedFollowing, true, dayCounter));
                }
                for (Size idx = 0; idx < numSwaps; ++idx) {
                    auto swapQuote = ext::make_shared<SimpleQuote>(swapRates[idx]);
                    instruments.push_back(ext::make_shared<SwapRateHelper>(
                        Handle<Quote>(swapQuote), swapTenors[idx],
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
                Date endDate = settlementDate + 6 * Years;
                curveDates.push_back(endDate);
                zeroRates.push_back(yieldCurve->zeroRate(endDate, dayCounter, Continuous).rate());

                std::vector<Rate> zeroRates_ql;
                for (const auto& r : zeroRates) zeroRates_ql.push_back(r);

                RelinkableHandle<YieldTermStructure> termStructure;
                ext::shared_ptr<IborIndex> index(new Euribor6M(termStructure));
                index->addFixing(Date(2, September, 2005), 0.04);
                termStructure.linkTo(ext::make_shared<ZeroCurve>(curveDates, zeroRates_ql, dayCounter));

                ext::shared_ptr<LiborForwardModelProcess> process(
                    new LiborForwardModelProcess(size, index));
                process->setCovarParam(ext::shared_ptr<LfmCovarianceParameterization>(
                    new LfmCovarianceProxy(
                        ext::make_shared<LmLinearExponentialVolatilityModel>(process->fixingTimes(), 0.291, 1.483, 0.116, 0.00001),
                        ext::make_shared<LmExponentialCorrelationModel>(size, 0.5))));

                ext::shared_ptr<VanillaSwap> fwdSwap(
                    new VanillaSwap(Swap::Receiver, 1.0,
                                    schedule, 0.05, dayCounter,
                                    schedule, index, 0.0, index->dayCounter()));
                fwdSwap->setPricingEngine(ext::make_shared<DiscountingSwapEngine>(
                    index->forwardingTermStructure()));
                Real swapRate = fwdSwap->fairRate();

                Array initRates = process->initialValues();

                Real price = Real(0.0);
                for (Size n = 0; n < nrTrails; ++n) {
                    Array asset(size);
                    for (Size k = 0; k < size; ++k) asset[k] = initRates[k];

                    Array assetAtExercise(size);
                    for (Size step = 1; step <= fullGridSteps; ++step) {
                        Size offset = (step - 1) * numFactors;
                        Time t = grid[step - 1];
                        Time dt = grid.dt(step - 1);

                        Array dw(numFactors);
                        for (Size f = 0; f < numFactors; ++f) {
                            dw[f] = allRandoms[n][offset + f];
                        }

                        asset = process->evolve(t, asset, dt, dw);

                        if (step == exerciseStep) {
                            for (Size k = 0; k < size; ++k) assetAtExercise[k] = asset[k];
                        }
                    }

                    Array dis(size);
                    Real df = Real(1.0);
                    for (Size k = 0; k < size; ++k) {
                        Real accrual = accrualEnd[k] - accrualStart[k];
                        df = df / (Real(1.0) + assetAtExercise[k] * accrual);
                        dis[k] = df;
                    }

                    Real npv = Real(0.0);
                    for (Size m = i_opt; m < i_opt + j_opt; ++m) {
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
                if (recordTiming) {
                    xad_rrs_times.push_back(Duration(t_end - t_start).count());
                }
                tape.deactivate();
            }

        }

        // Compute averages
        auto avg = [](const std::vector<double>& v) {
            if (v.empty()) return 0.0;
            double sum = 0.0;
            for (double x : v) sum += x;
            return sum / static_cast<double>(v.size());
        };

        auto stddev = [&avg](const std::vector<double>& v) {
            if (v.size() <= 1) return 0.0;
            double mean = avg(v);
            double sq_sum = 0.0;
            for (double x : v) {
                double diff = x - mean;
                sq_sum += diff * diff;
            }
            return std::sqrt(sq_sum / (v.size() - 1));
        };

        results[tc].xad_ql_total = avg(xad_ql_times);
        results[tc].xad_rrs_total = avg(xad_rrs_times);
        results[tc].xad_ql_std = stddev(xad_ql_times);
        results[tc].xad_rrs_std = stddev(xad_rrs_times);

        std::cout << " Done." << std::endl;
    }

    // Print results table (compact format)
    std::cout << std::endl;
    std::cout << "  " << std::string(79, '=') << "\n";
    std::cout << "  RESULTS: Simple Swaption (times in ms)\n";
    std::cout << "  " << std::string(79, '=') << "\n";
    std::cout << std::endl;

    std::cout << "    Paths |  XAD(QL) |  XAD(RR)\n";
    std::cout << "   -------+----------+----------\n";

    for (Size tc = 0; tc < pathCounts.size(); ++tc) {
        std::cout << "  " << std::setw(6) << pathCounts[tc] << " |"
                  << std::fixed << std::setprecision(1) << std::setw(10) << results[tc].xad_ql_total << " |"
                  << std::fixed << std::setprecision(1) << std::setw(10) << results[tc].xad_rrs_total << "\n";
    }

    std::cout << std::endl;

    // Basic verification
    BOOST_CHECK(results[0].xad_ql_total > 0.0);
    BOOST_CHECK(results[0].xad_rrs_total > 0.0);
}

//////////////////////////////////////////////////////////////////////////////
// Larger Swaption Benchmark (5Y into 5Y)
//////////////////////////////////////////////////////////////////////////////

BOOST_AUTO_TEST_CASE(testBenchmark_LargerSwaptionScaling)
{
    BOOST_TEST_MESSAGE("Running Larger Swaption Scaling Benchmark...");

    std::cout << "\n";
    std::cout << "=============================================================================\n";
    std::cout << "  XAD BASELINE BENCHMARK: Larger Swaption (5Y into 5Y)\n";
    std::cout << "=============================================================================\n";
    std::cout << std::endl;
    std::cout << "  This benchmark attempts to approximate a more realistic scenario:\n";
    std::cout << "  larger computation graph, more forward rates, longer simulation.\n";
    std::cout << std::endl;
    std::cout << "  APPROACH TESTED:\n";
    std::cout << "    XAD - XAD tape + direct evolve\n";
    std::cout << std::endl;
    std::cout << "  INSTRUMENT:\n";
    std::cout << "    European receiver swaption: 5Y option into 5Y swap (10Y total)\n";
    std::cout << "    Model: LIBOR Market Model (LMM) with 20 forward rates\n";
    std::cout << "    Sensitivities: dPrice/dMarketQuotes (14 inputs)\n";
    std::cout << std::endl;

    using Clock = std::chrono::high_resolution_clock;
    using Duration = std::chrono::duration<double, std::milli>;

    // Market data setup - PRODUCTION-LIKE CONFIGURATION
    Calendar calendar = TARGET();
    Date todaysDate(4, September, 2005);
    Settings::instance().evaluationDate() = todaysDate;
    Integer fixingDays = 2;
    Date settlementDate = calendar.adjust(calendar.advance(todaysDate, fixingDays, Days));
    DayCounter dayCounter = Actual360();

    // Larger setup: 4 deposits + 10 swaps = 14 market inputs (up to 10Y)
    Size numDeposits = 4;
    Size numSwaps = 10;
    std::vector<Period> depoTenors = {1 * Days, 1 * Months, 3 * Months, 6 * Months};
    std::vector<Period> swapTenors = {1 * Years, 2 * Years, 3 * Years, 4 * Years, 5 * Years,
                                      6 * Years, 7 * Years, 8 * Years, 9 * Years, 10 * Years};

    // Realistic deposit rates (upward sloping short end)
    std::vector<double> depoRates_val = {0.0320, 0.0335, 0.0355, 0.0375};
    // Realistic swap rates (upward sloping curve to 10Y)
    std::vector<double> swapRates_val = {0.0400, 0.0435, 0.0460, 0.0480, 0.0495,
                                         0.0505, 0.0515, 0.0522, 0.0528, 0.0532};

    Size numMarketQuotes = numDeposits + numSwaps;

    // LMM parameters: 20 forward rates (semi-annual to 10Y)
    // Swaption: 5Y option into 5Y swap (exercises at 5Y, underlying matures at 10Y)
    Size size = 20;      // Semi-annual rates to 10Y
    Size i_opt = 10;     // Option starts at 5Y (10 × 6 months)
    Size j_opt = 10;     // 5Y swap (10 × 6 months)
    Size steps = 20;     // Time steps for 5Y simulation

    // Build base curve and process setup - to 12Y (extra buffer for accrual periods)
    std::vector<Rate> baseZeroRates = {0.0320, 0.0535};
    std::vector<Date> baseDates = {settlementDate, settlementDate + 12 * Years};
    auto baseIndex = makeIndex(baseDates, baseZeroRates);

    ext::shared_ptr<LiborForwardModelProcess> baseProcess(
        new LiborForwardModelProcess(size, baseIndex));
    ext::shared_ptr<LmCorrelationModel> baseCorrModel(
        new LmExponentialCorrelationModel(size, 0.5));
    ext::shared_ptr<LmVolatilityModel> baseVolaModel(
        new LmLinearExponentialVolatilityModel(baseProcess->fixingTimes(),
                                               0.291, 1.483, 0.116, 0.00001));
    baseProcess->setCovarParam(ext::shared_ptr<LfmCovarianceParameterization>(
        new LfmCovarianceProxy(baseVolaModel, baseCorrModel)));

    // Grid and timing setup
    std::vector<Time> fixingTimes = baseProcess->fixingTimes();
    TimeGrid grid(fixingTimes.begin(), fixingTimes.end(), steps);

    std::vector<Size> location;
    for (Size idx = 0; idx < fixingTimes.size(); ++idx) {
        location.push_back(
            std::find(grid.begin(), grid.end(), fixingTimes[idx]) - grid.begin());
    }

    Size numFactors = baseProcess->factors();
    Size exerciseStep = location[i_opt];
    Size fullGridSteps = grid.size() - 1;
    Size fullGridRandoms = fullGridSteps * numFactors;

    // Types for MC generators
    typedef PseudoRandom::rsg_type rsg_type;
    typedef MultiPathGenerator<rsg_type>::sample_type sample_type;

    // Get fair swap rate
    BusinessDayConvention convention = baseIndex->businessDayConvention();
    Date settlement = baseIndex->forwardingTermStructure()->referenceDate();
    Date fwdStart = settlement + Period(6 * i_opt, Months);
    Date fwdMaturity = fwdStart + Period(6 * j_opt, Months);

    Schedule schedule(fwdStart, fwdMaturity, baseIndex->tenor(), calendar,
                      convention, convention, DateGeneration::Forward, false);

    // Accrual periods
    std::vector<double> accrualStart(size), accrualEnd(size);
    for (Size k = 0; k < size; ++k) {
        accrualStart[k] = value(baseProcess->accrualStartTimes()[k]);
        accrualEnd[k] = value(baseProcess->accrualEndTimes()[k]);
    }

    Size numIntermediates = size + 1;

    // Helper lambda for curve building
    auto buildCurveAndProcess = [&](const std::vector<double>& depoRates,
                                    const std::vector<double>& swapRates,
                                    std::vector<Date>& curveDates,
                                    std::vector<Rate>& zeroRates) {
        RelinkableHandle<YieldTermStructure> euriborTS;
        auto euribor6m = ext::make_shared<Euribor6M>(euriborTS);
        euribor6m->addFixing(Date(2, September, 2005), 0.04);

        std::vector<ext::shared_ptr<RateHelper>> instruments;
        for (Size idx = 0; idx < numDeposits; ++idx) {
            auto depoQuote = ext::make_shared<SimpleQuote>(depoRates[idx]);
            instruments.push_back(ext::make_shared<DepositRateHelper>(
                Handle<Quote>(depoQuote), depoTenors[idx], fixingDays,
                calendar, ModifiedFollowing, true, dayCounter));
        }
        for (Size idx = 0; idx < numSwaps; ++idx) {
            auto swapQuote = ext::make_shared<SimpleQuote>(swapRates[idx]);
            instruments.push_back(ext::make_shared<SwapRateHelper>(
                Handle<Quote>(swapQuote), swapTenors[idx],
                calendar, Annual, Unadjusted, Thirty360(Thirty360::BondBasis),
                euribor6m));
        }

        auto yieldCurve = ext::make_shared<PiecewiseYieldCurve<ZeroYield, Linear>>(
            settlementDate, instruments, dayCounter);
        yieldCurve->enableExtrapolation();

        curveDates.clear();
        zeroRates.clear();
        curveDates.push_back(settlementDate);
        zeroRates.push_back(yieldCurve->zeroRate(settlementDate, dayCounter, Continuous).rate());
        Date endDate = settlementDate + 12 * Years;  // Extended for 10Y instrument + buffer
        curveDates.push_back(endDate);
        zeroRates.push_back(yieldCurve->zeroRate(endDate, dayCounter, Continuous).rate());
    };

    // Test configuration - scaling over different path counts
    std::vector<Size> pathCounts = {10, 100, 1000, 10000};
    Size maxPaths = pathCounts.back();
    Size warmupIterations = 2;
    Size benchmarkIterations = 3;

    std::cout << "  CONFIGURATION:\n";
    std::cout << "    Market inputs:    " << numMarketQuotes << " (" << numDeposits << " deposits + " << numSwaps << " swaps)\n";
    std::cout << "    Forward rates:    " << size << " (semi-annual to 10Y)\n";
    std::cout << "    Time steps:       " << fullGridSteps << "\n";
    std::cout << "    MC paths tested:  10, 100, 1K, 10K\n";
    std::cout << "    Warmup/Bench:     " << warmupIterations << "/" << benchmarkIterations << " iterations\n";
    std::cout << std::endl;

    // Pre-generate random numbers for max paths
    std::cout << "  Generating " << maxPaths << " x " << fullGridRandoms << " random numbers..." << std::flush;
    rsg_type rsg_base = PseudoRandom::make_sequence_generator(fullGridRandoms, BigNatural(42));
    std::vector<std::vector<double>> allRandoms(maxPaths);
    for (Size n = 0; n < maxPaths; ++n) {
        allRandoms[n].resize(fullGridRandoms);
        const auto& seq = rsg_base.nextSequence();
        for (Size m = 0; m < fullGridRandoms; ++m) {
            allRandoms[n][m] = value(seq.value[m]);
        }
    }
    std::cout << " Done." << std::endl;

    // Results storage for scaling table
    struct ScalingResult {
        double xad_time;
    };
    std::vector<ScalingResult> scalingResults(pathCounts.size());

    // Store final prices/derivs from largest path count
    double xad_price = 0.0;
    std::vector<double> xad_derivs;

    // Run benchmarks for each path count
    for (Size pathIdx = 0; pathIdx < pathCounts.size(); ++pathIdx) {
        Size nrTrails = pathCounts[pathIdx];
        std::cout << "\n  --- Running with " << nrTrails << " paths ---\n";

        // Timing accumulators for this path count
        std::vector<double> xad_rrs_times;

        for (Size iter = 0; iter < warmupIterations + benchmarkIterations; ++iter) {
            bool recordTiming = (iter >= warmupIterations);
            bool isLast = (iter == warmupIterations + benchmarkIterations - 1) &&
                          (pathIdx == pathCounts.size() - 1);  // Only save details for largest path count

            std::cout << "  [" << (iter < warmupIterations ? "Warmup" : "Bench") << " "
                      << (iter < warmupIterations ? iter + 1 : iter - warmupIterations + 1) << "/"
                      << (iter < warmupIterations ? warmupIterations : benchmarkIterations) << "] ";

        // =================================================================
        // XAD (RR) - Full grid with XAD tape (with granular timing)
        // =================================================================
        std::cout << "XAD..." << std::flush;
        {
            auto t_start = Clock::now();
            auto t_phase = Clock::now();

            // Phase timing accumulators (only for last iteration)
            double t_bootstrap_fwd = 0, t_mc_fwd = 0, t_backward = 0;

            using tape_type = Real::tape_type;
            tape_type tape;

            std::vector<Real> depositRates(numDeposits);
            std::vector<Real> swapRates(numSwaps);
            for (Size idx = 0; idx < numDeposits; ++idx) depositRates[idx] = depoRates_val[idx];
            for (Size idx = 0; idx < numSwaps; ++idx) swapRates[idx] = swapRates_val[idx];

            tape.registerInputs(depositRates);
            tape.registerInputs(swapRates);
            tape.newRecording();

            RelinkableHandle<YieldTermStructure> euriborTS;
            auto euribor6m = ext::make_shared<Euribor6M>(euriborTS);
            euribor6m->addFixing(Date(2, September, 2005), 0.04);

            std::vector<ext::shared_ptr<RateHelper>> instruments;
            for (Size idx = 0; idx < numDeposits; ++idx) {
                auto depoQuote = ext::make_shared<SimpleQuote>(depositRates[idx]);
                instruments.push_back(ext::make_shared<DepositRateHelper>(
                    Handle<Quote>(depoQuote), depoTenors[idx], fixingDays,
                    calendar, ModifiedFollowing, true, dayCounter));
            }
            for (Size idx = 0; idx < numSwaps; ++idx) {
                auto swapQuote = ext::make_shared<SimpleQuote>(swapRates[idx]);
                instruments.push_back(ext::make_shared<SwapRateHelper>(
                    Handle<Quote>(swapQuote), swapTenors[idx],
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
            Date endDate = settlementDate + 12 * Years;  // Extended for 10Y instrument + buffer
            curveDates.push_back(endDate);
            zeroRates.push_back(yieldCurve->zeroRate(endDate, dayCounter, Continuous).rate());

            std::vector<Rate> zeroRates_ql;
            for (const auto& r : zeroRates) zeroRates_ql.push_back(r);

            RelinkableHandle<YieldTermStructure> termStructure;
            ext::shared_ptr<IborIndex> index(new Euribor6M(termStructure));
            index->addFixing(Date(2, September, 2005), 0.04);
            termStructure.linkTo(ext::make_shared<ZeroCurve>(curveDates, zeroRates_ql, dayCounter));

            ext::shared_ptr<LiborForwardModelProcess> process(
                new LiborForwardModelProcess(size, index));
            process->setCovarParam(ext::shared_ptr<LfmCovarianceParameterization>(
                new LfmCovarianceProxy(
                    ext::make_shared<LmLinearExponentialVolatilityModel>(process->fixingTimes(), 0.291, 1.483, 0.116, 0.00001),
                    ext::make_shared<LmExponentialCorrelationModel>(size, 0.5))));

            ext::shared_ptr<VanillaSwap> fwdSwap(
                new VanillaSwap(Swap::Receiver, 1.0,
                                schedule, 0.05, dayCounter,
                                schedule, index, 0.0, index->dayCounter()));
            fwdSwap->setPricingEngine(ext::make_shared<DiscountingSwapEngine>(
                index->forwardingTermStructure()));
            Real swapRate = fwdSwap->fairRate();

            Array initRates = process->initialValues();

            if (isLast) {
                t_bootstrap_fwd = Duration(Clock::now() - t_phase).count();
                t_phase = Clock::now();
            }

            // MC pricing with full grid
            Real price = Real(0.0);
            for (Size n = 0; n < nrTrails; ++n) {
                Array asset(size);
                for (Size k = 0; k < size; ++k) asset[k] = initRates[k];

                Array assetAtExercise(size);
                for (Size step = 1; step <= fullGridSteps; ++step) {
                    Size offset = (step - 1) * numFactors;
                    Time t = grid[step - 1];
                    Time dt = grid.dt(step - 1);

                    Array dw(numFactors);
                    for (Size f = 0; f < numFactors; ++f)
                        dw[f] = Real(allRandoms[n][offset + f]);

                    asset = process->evolve(t, asset, dt, dw);

                    if (step == exerciseStep) {
                        for (Size k = 0; k < size; ++k) assetAtExercise[k] = asset[k];
                    }
                }

                Array dis(size);
                Real df = Real(1.0);
                for (Size k = 0; k < size; ++k) {
                    Real accrual = accrualEnd[k] - accrualStart[k];
                    df = df / (Real(1.0) + assetAtExercise[k] * accrual);
                    dis[k] = df;
                }

                Real npv = Real(0.0);
                for (Size m = i_opt; m < i_opt + j_opt; ++m) {
                    Real accrual = accrualEnd[m] - accrualStart[m];
                    npv += (swapRate - assetAtExercise[m]) * accrual * dis[m];
                }
                price += max(npv, Real(0.0));
            }
            price /= Real(static_cast<double>(nrTrails));

            if (isLast) {
                t_mc_fwd = Duration(Clock::now() - t_phase).count();
                t_phase = Clock::now();
            }

            tape.registerOutput(price);
            derivative(price) = 1.0;
            tape.computeAdjoints();

            if (isLast) {
                t_backward = Duration(Clock::now() - t_phase).count();

                // Report XAD timing breakdown
                std::cout << "\n    [XAD Details: " << tape.getNumStatements() << " tape statements]\n";
                std::cout << "      Bootstrap fwd:   " << std::fixed << std::setprecision(1) << t_bootstrap_fwd << " ms\n";
                std::cout << "      MC fwd (" << nrTrails << " paths): " << std::fixed << std::setprecision(1) << t_mc_fwd << " ms\n";
                std::cout << "      Backward:        " << std::fixed << std::setprecision(1) << t_backward << " ms\n";

                xad_price = value(price);
                xad_derivs.resize(numMarketQuotes);
                for (Size m = 0; m < numDeposits; ++m)
                    xad_derivs[m] = derivative(depositRates[m]);
                for (Size m = 0; m < numSwaps; ++m)
                    xad_derivs[numDeposits + m] = derivative(swapRates[m]);
            }

            auto t_end = Clock::now();
            if (recordTiming) {
                xad_rrs_times.push_back(Duration(t_end - t_start).count());
            }
            tape.deactivate();
        }


        std::cout << " Done." << std::endl;
        }  // end iteration loop

        // Compute averages for this path count
        auto avg = [](const std::vector<double>& v) {
            if (v.empty()) return 0.0;
            double sum = 0.0;
            for (double x : v) sum += x;
            return sum / static_cast<double>(v.size());
        };

        scalingResults[pathIdx].xad_time = avg(xad_rrs_times);

        std::cout << "    Avg time: XAD=" << std::fixed << std::setprecision(1) << scalingResults[pathIdx].xad_time << "ms\n";
    }  // end path count loop

    // Print scaling table
    std::cout << std::endl;
    std::cout << "  " << std::string(79, '=') << "\n";
    std::cout << "  RESULTS: Larger Swaption (times in ms)\n";
    std::cout << "  " << std::string(79, '=') << "\n";
    std::cout << std::endl;

    std::cout << "    Paths |       XAD\n";
    std::cout << "   -------+-----------\n";
    for (Size i = 0; i < pathCounts.size(); ++i) {
        std::string pathStr;
        if (pathCounts[i] >= 1000) pathStr = std::to_string(pathCounts[i]/1000) + "K";
        else pathStr = std::to_string(pathCounts[i]);

        std::cout << "   " << std::setw(6) << pathStr << " |"
                  << std::fixed << std::setprecision(1) << std::setw(10) << scalingResults[i].xad_time << "\n";
    }
    std::cout << std::endl;

    std::cout << "  MC PRICE (10K paths):\n";
    std::cout << "    XAD: " << std::fixed << std::setprecision(8) << xad_price << "\n";
    std::cout << std::endl;

    std::cout << "  DERIVATIVES (first 5 market quotes, 10K paths):\n";
    std::cout << "    " << std::setw(12) << "Quote" << " | " << std::setw(14) << "XAD" << "\n";
    std::cout << "    " << std::string(30, '-') << "\n";
    for (Size i = 0; i < std::min(Size(5), numMarketQuotes); ++i) {
        std::string label = (i < numDeposits) ? "Depo " + std::to_string(i) : "Swap " + std::to_string(i - numDeposits);
        std::cout << "    " << std::setw(12) << label << " | "
                  << std::scientific << std::setprecision(4) << std::setw(14) << xad_derivs[i] << "\n";
    }
    std::cout << std::endl;

    // Verification
    BOOST_CHECK(scalingResults.back().xad_time > 0.0);
    BOOST_CHECK(xad_price > 0.0);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE_END()

