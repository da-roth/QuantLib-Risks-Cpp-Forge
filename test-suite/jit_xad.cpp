/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*
 This file is part of QuantLib-Risks, a C++ library for AAD-enabled
 quantitative finance using QuantLib and XAD.

 QuantLib-Risks is free software: you can redistribute it and/or modify it
 under the terms of the QuantLib-Risks license.  You should have received a
 copy of the license along with this program; if not, please visit
 <https://github.com/auto-differentiation/QuantLib-Risks-Cpp>.

 This file tests the XAD JIT compilation infrastructure.
*/

#include "toplevelfixture.hpp"
#include "utilities_xad.hpp"
#include <ql/quantlib.hpp>
#include <XAD/XAD.hpp>
#include <boost/test/unit_test.hpp>
#include <cmath>
#include <vector>

using namespace QuantLib;
using namespace boost::unit_test_framework;

BOOST_FIXTURE_TEST_SUITE(QuantLibRisksJITTests, TopLevelFixture)

BOOST_AUTO_TEST_SUITE(JITTests)

namespace {

// f1: Simple linear function
// f(x) = x * 3 + 2, f'(x) = 3
template <class T>
T f1(const T& x)
{
    return x * 3.0 + 2.0;
}

// f2: Function with supported math operations
// Uses: sin, cos, exp, log, sqrt, abs
template <class T>
T f2(const T& x)
{
    using std::sin; using std::cos; using std::exp; using std::log;
    using std::sqrt; using std::abs;

    T result = sin(x) + cos(x) * 2.0;
    result = result + exp(x / 10.0) + log(x + 5.0);
    result = result + sqrt(x + 1.0);
    result = result + abs(x - 1.0) + x * x;
    result = result + 1.0 / (x + 2.0);
    return result;
}

// Helper to get value for both double and AD types
inline double getValue(double x) { return x; }
template <class T>
double getValue(const T& x) { return value(x); }

// f3: Branching function (regular if/else - JIT records one branch)
template <class T>
T f3(const T& x)
{
    if (getValue(x) < 2.0)
        return 2.0 * x;
    else
        return 10.0 * x;
}

// f3ABool: Branching with ABool::If (JIT records both branches)
xad::AD f3ABool(const xad::AD& x)
{
    return xad::less(x, 2.0).If(2.0 * x, 10.0 * x);
}

double f3ABool_double(double x)
{
    return (x < 2.0) ? 2.0 * x : 10.0 * x;
}

} // anonymous namespace

BOOST_AUTO_TEST_CASE(testJITLinearFunction)
{
    BOOST_TEST_MESSAGE("Testing JIT compilation with linear function...");
    std::cout << "\n=== JIT Linear Function Test: f(x) = 3x + 2 ===" << std::endl;

    std::vector<double> inputs = {2.0, 0.5, -1.0};

    // Compute with Tape
    std::vector<double> tapeOutputs, tapeDerivatives;
    {
        xad::Tape<double> tape;
        for (double input : inputs)
        {
            xad::AD x(input);
            tape.registerInput(x);
            tape.newRecording();
            xad::AD y = f1(x);
            tape.registerOutput(y);
            derivative(y) = 1.0;
            tape.computeAdjoints();
            tapeOutputs.push_back(value(y));
            tapeDerivatives.push_back(derivative(x));
            tape.clearAll();
        }
    }

    // Compute with JIT (record once, reuse)
    std::vector<double> jitOutputs, jitDerivatives;
    {
        xad::JITCompiler<double> jit;

        xad::AD x(inputs[0]);
        jit.registerInput(x);
        jit.newRecording();
        xad::AD y = f1(x);
        jit.registerOutput(y);
        jit.compile();  // Compile before forward

        for (double input : inputs)
        {
            value(x) = input;
            double output;
            jit.forward(&output, 1);
            jitOutputs.push_back(output);

            jit.clearDerivatives();
            derivative(y) = 1.0;
            jit.computeAdjoints();
            jitDerivatives.push_back(derivative(x));
        }
    }

    // Compare results
    for (std::size_t i = 0; i < inputs.size(); ++i)
    {
        double expected = f1(inputs[i]);
        std::cout << "  x=" << inputs[i]
                  << ": tape=" << tapeOutputs[i] << " (deriv=" << tapeDerivatives[i] << ")"
                  << ", jit=" << jitOutputs[i] << " (deriv=" << jitDerivatives[i] << ")"
                  << std::endl;
        BOOST_CHECK_CLOSE(expected, tapeOutputs[i], 1e-10);
        BOOST_CHECK_CLOSE(expected, jitOutputs[i], 1e-10);
        BOOST_CHECK_CLOSE(tapeDerivatives[i], jitDerivatives[i], 1e-10);
    }
}

BOOST_AUTO_TEST_CASE(testJITMathFunctions)
{
    BOOST_TEST_MESSAGE("Testing JIT compilation with math functions...");
    std::cout << "\n=== JIT Math Functions Test: sin, cos, exp, log, sqrt, abs ===" << std::endl;

    std::vector<double> inputs = {2.0, 0.5};

    // Compute with Tape
    std::vector<double> tapeOutputs, tapeDerivatives;
    {
        xad::Tape<double> tape;
        for (double input : inputs)
        {
            xad::AD x(input);
            tape.registerInput(x);
            tape.newRecording();
            xad::AD y = f2(x);
            tape.registerOutput(y);
            derivative(y) = 1.0;
            tape.computeAdjoints();
            tapeOutputs.push_back(value(y));
            tapeDerivatives.push_back(derivative(x));
            tape.clearAll();
        }
    }

    // Compute with JIT
    std::vector<double> jitOutputs, jitDerivatives;
    {
        xad::JITCompiler<double> jit;

        xad::AD x(inputs[0]);
        jit.registerInput(x);
        jit.newRecording();
        xad::AD y = f2(x);
        jit.registerOutput(y);
        jit.compile();  // Compile before forward

        for (double input : inputs)
        {
            value(x) = input;
            double output;
            jit.forward(&output, 1);
            jitOutputs.push_back(output);

            jit.clearDerivatives();
            derivative(y) = 1.0;
            jit.computeAdjoints();
            jitDerivatives.push_back(derivative(x));
        }
    }

    // Compare results
    for (std::size_t i = 0; i < inputs.size(); ++i)
    {
        double expected = f2(inputs[i]);
        std::cout << "  x=" << inputs[i]
                  << ": tape=" << tapeOutputs[i] << " (deriv=" << tapeDerivatives[i] << ")"
                  << ", jit=" << jitOutputs[i] << " (deriv=" << jitDerivatives[i] << ")"
                  << std::endl;
        BOOST_CHECK_CLOSE(expected, tapeOutputs[i], 1e-10);
        BOOST_CHECK_CLOSE(expected, jitOutputs[i], 1e-10);
        BOOST_CHECK_CLOSE(tapeDerivatives[i], jitDerivatives[i], 1e-10);
    }
}

BOOST_AUTO_TEST_CASE(testJITBranchingRegularIf)
{
    BOOST_TEST_MESSAGE("Testing JIT with regular if/else (graph reuse behavior)...");
    std::cout << "\n=== JIT Branching Test: Regular if/else (demonstrates graph reuse) ===" << std::endl;
    std::cout << "  Formula: if (x < 2) 2*x else 10*x" << std::endl;

    // With regular if/else, JIT records the branch taken during recording
    // and will use that branch for all subsequent evaluations
    std::vector<double> inputs = {1.0, 3.0};

    // Compute with Tape (re-records each time, follows actual branch)
    std::vector<double> tapeOutputs;
    {
        xad::Tape<double> tape;
        for (double input : inputs)
        {
            xad::AD x(input);
            tape.registerInput(x);
            tape.newRecording();
            xad::AD y = f3(x);
            tape.registerOutput(y);
            tapeOutputs.push_back(value(y));
            tape.clearAll();
        }
    }

    // Compute with JIT (records branch at x=1, uses it for all)
    std::vector<double> jitOutputs;
    {
        xad::JITCompiler<double> jit;

        xad::AD x(inputs[0]);  // x=1 -> takes first branch (2*x)
        jit.registerInput(x);
        jit.newRecording();
        xad::AD y = f3(x);
        jit.registerOutput(y);
        jit.compile();  // Compile before forward

        for (double input : inputs)
        {
            value(x) = input;
            double output;
            jit.forward(&output, 1);
            jitOutputs.push_back(output);
        }
    }

    std::cout << "  x=1: tape=" << tapeOutputs[0] << " (correct: 2*1=2), jit=" << jitOutputs[0] << std::endl;
    std::cout << "  x=3: tape=" << tapeOutputs[1] << " (correct: 10*3=30), jit=" << jitOutputs[1] << " (uses recorded 2*x branch!)" << std::endl;

    // Tape follows actual branches: f(1)=2, f(3)=30
    BOOST_CHECK_CLOSE(tapeOutputs[0], 2.0, 1e-10);   // 2*1 = 2
    BOOST_CHECK_CLOSE(tapeOutputs[1], 30.0, 1e-10); // 10*3 = 30

    // JIT recorded first branch (2*x), uses it for both: f(1)=2, f(3)=6
    BOOST_CHECK_CLOSE(jitOutputs[0], 2.0, 1e-10);  // 2*1 = 2
    BOOST_CHECK_CLOSE(jitOutputs[1], 6.0, 1e-10);  // 2*3 = 6 (uses recorded branch!)
}

BOOST_AUTO_TEST_CASE(testJITBranchingABool)
{
    BOOST_TEST_MESSAGE("Testing JIT with ABool::If (tracks both branches)...");
    std::cout << "\n=== JIT Branching Test: ABool::If (tracks both branches correctly) ===" << std::endl;
    std::cout << "  Formula: ABool::If(x < 2, 2*x, 10*x)" << std::endl;

    // With ABool::If, JIT records both branches and selects at runtime
    std::vector<double> inputs = {1.0, 3.0};

    // Compute with Tape
    std::vector<double> tapeOutputs, tapeDerivatives;
    {
        xad::Tape<double> tape;
        for (double input : inputs)
        {
            xad::AD x(input);
            tape.registerInput(x);
            tape.newRecording();
            xad::AD y = f3ABool(x);
            tape.registerOutput(y);
            derivative(y) = 1.0;
            tape.computeAdjoints();
            tapeOutputs.push_back(value(y));
            tapeDerivatives.push_back(derivative(x));
            tape.clearAll();
        }
    }

    // Compute with JIT
    std::vector<double> jitOutputs, jitDerivatives;
    {
        xad::JITCompiler<double> jit;

        xad::AD x(inputs[0]);
        jit.registerInput(x);
        jit.newRecording();
        xad::AD y = f3ABool(x);
        jit.registerOutput(y);
        jit.compile();  // Compile before forward

        for (double input : inputs)
        {
            value(x) = input;
            double output;
            jit.forward(&output, 1);
            jitOutputs.push_back(output);

            jit.clearDerivatives();
            derivative(y) = 1.0;
            jit.computeAdjoints();
            jitDerivatives.push_back(derivative(x));
        }
    }

    // Both should match: ABool::If allows JIT to track both branches
    for (std::size_t i = 0; i < inputs.size(); ++i)
    {
        double expected = f3ABool_double(inputs[i]);
        std::cout << "  x=" << inputs[i]
                  << ": tape=" << tapeOutputs[i] << " (deriv=" << tapeDerivatives[i] << ")"
                  << ", jit=" << jitOutputs[i] << " (deriv=" << jitDerivatives[i] << ")"
                  << " - MATCH!" << std::endl;
        BOOST_CHECK_CLOSE(expected, tapeOutputs[i], 1e-10);
        BOOST_CHECK_CLOSE(expected, jitOutputs[i], 1e-10);
        BOOST_CHECK_CLOSE(tapeDerivatives[i], jitDerivatives[i], 1e-10);
    }
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE_END()
