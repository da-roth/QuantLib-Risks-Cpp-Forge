/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*
 This file is part of QuantLib-Risks, a C++ library for AAD-enabled
 quantitative finance using QuantLib and XAD.

 This file tests the Forge JIT backend integration with XAD.
 Unlike jit_xad.cpp which uses the C++ interpreter, this tests
 the actual native code generation via Forge.
*/

#include "toplevelfixture.hpp"
#include "utilities_xad.hpp"
#include <ql/quantlib.hpp>
#include <XAD/XAD.hpp>
#include <qlrisks-forge/ForgeBackend.hpp>
#include <boost/test/unit_test.hpp>
#include <cmath>
#include <vector>

using namespace QuantLib;
using namespace boost::unit_test_framework;

BOOST_FIXTURE_TEST_SUITE(QuantLibRisksForgeTests, TopLevelFixture)

BOOST_AUTO_TEST_SUITE(ForgeBackendTests)

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

// f3ABool: Branching with ABool::If for trackable branches
xad::AD f3ABool(const xad::AD& x)
{
    return xad::less(x, 2.0).If(2.0 * x, 10.0 * x);
}

double f3ABool_double(double x)
{
    return (x < 2.0) ? 2.0 * x : 10.0 * x;
}

} // anonymous namespace

BOOST_AUTO_TEST_CASE(testForgeBackendLinearFunction)
{
    BOOST_TEST_MESSAGE("Testing ForgeBackend with linear function...");
    std::cout << "\n=== ForgeBackend Linear Function Test: f(x) = 3x + 2 ===" << std::endl;

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

    // Compute with ForgeBackend (native JIT)
    std::vector<double> forgeOutputs, forgeDerivatives;
    {
        auto jit = xad::JITCompiler<double, 1>::withBackend<qlrisks::forge::ForgeBackend>();

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
            forgeOutputs.push_back(output);

            jit.clearDerivatives();
            derivative(y) = 1.0;
            jit.computeAdjoints();
            forgeDerivatives.push_back(derivative(x));
        }
    }

    // Compare results
    for (std::size_t i = 0; i < inputs.size(); ++i)
    {
        double expected = f1(inputs[i]);
        std::cout << "  x=" << inputs[i]
                  << ": tape=" << tapeOutputs[i] << " (deriv=" << tapeDerivatives[i] << ")"
                  << ", forge=" << forgeOutputs[i] << " (deriv=" << forgeDerivatives[i] << ")"
                  << std::endl;
        BOOST_CHECK_CLOSE(expected, tapeOutputs[i], 1e-10);
        BOOST_CHECK_CLOSE(expected, forgeOutputs[i], 1e-10);
        BOOST_CHECK_CLOSE(tapeDerivatives[i], forgeDerivatives[i], 1e-10);
    }
}

BOOST_AUTO_TEST_CASE(testForgeBackendMathFunctions)
{
    BOOST_TEST_MESSAGE("Testing ForgeBackend with math functions...");
    std::cout << "\n=== ForgeBackend Math Functions Test: sin, cos, exp, log, sqrt, abs ===" << std::endl;

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

    // Compute with ForgeBackend
    std::vector<double> forgeOutputs, forgeDerivatives;
    {
        auto jit = xad::JITCompiler<double, 1>::withBackend<qlrisks::forge::ForgeBackend>();

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
            forgeOutputs.push_back(output);

            jit.clearDerivatives();
            derivative(y) = 1.0;
            jit.computeAdjoints();
            forgeDerivatives.push_back(derivative(x));
        }
    }

    // Compare results
    for (std::size_t i = 0; i < inputs.size(); ++i)
    {
        double expected = f2(inputs[i]);
        std::cout << "  x=" << inputs[i]
                  << ": tape=" << tapeOutputs[i] << " (deriv=" << tapeDerivatives[i] << ")"
                  << ", forge=" << forgeOutputs[i] << " (deriv=" << forgeDerivatives[i] << ")"
                  << std::endl;
        BOOST_CHECK_CLOSE(expected, tapeOutputs[i], 1e-10);
        BOOST_CHECK_CLOSE(expected, forgeOutputs[i], 1e-10);
        BOOST_CHECK_CLOSE(tapeDerivatives[i], forgeDerivatives[i], 1e-10);
    }
}

BOOST_AUTO_TEST_CASE(testForgeBackendABoolBranching)
{
    BOOST_TEST_MESSAGE("Testing ForgeBackend with ABool::If branching...");
    std::cout << "\n=== ForgeBackend ABool::If Test: if(x<2) 2*x else 10*x ===" << std::endl;

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

    // Compute with ForgeBackend
    std::vector<double> forgeOutputs, forgeDerivatives;
    {
        auto jit = xad::JITCompiler<double, 1>::withBackend<qlrisks::forge::ForgeBackend>();

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
            forgeOutputs.push_back(output);

            jit.clearDerivatives();
            derivative(y) = 1.0;
            jit.computeAdjoints();
            forgeDerivatives.push_back(derivative(x));
        }
    }

    // Compare results - ABool::If should track both branches
    for (std::size_t i = 0; i < inputs.size(); ++i)
    {
        double expected = f3ABool_double(inputs[i]);
        std::cout << "  x=" << inputs[i]
                  << ": tape=" << tapeOutputs[i] << " (deriv=" << tapeDerivatives[i] << ")"
                  << ", forge=" << forgeOutputs[i] << " (deriv=" << forgeDerivatives[i] << ")"
                  << " - MATCH!" << std::endl;
        BOOST_CHECK_CLOSE(expected, tapeOutputs[i], 1e-10);
        BOOST_CHECK_CLOSE(expected, forgeOutputs[i], 1e-10);
        BOOST_CHECK_CLOSE(tapeDerivatives[i], forgeDerivatives[i], 1e-10);
    }
}

BOOST_AUTO_TEST_CASE(testForgeBackendBasicInstantiation)
{
    BOOST_TEST_MESSAGE("Testing ForgeBackend basic instantiation...");
    std::cout << "\n=== ForgeBackend Basic Test: f(x) = x^2 + 3x ===" << std::endl;

    auto jit = xad::JITCompiler<double, 1>::withBackend<qlrisks::forge::ForgeBackend>();

    xad::AD x(2.0);
    jit.registerInput(x);
    jit.newRecording();
    xad::AD y = x * x + 3.0 * x;  // f(x) = x^2 + 3x, f'(x) = 2x + 3
    jit.registerOutput(y);
    jit.compile();  // Compile before forward

    double output;
    jit.forward(&output, 1);
    std::cout << "  f(2) = " << output << " (expected: 10)" << std::endl;
    BOOST_CHECK_CLOSE(10.0, output, 1e-10);  // f(2) = 4 + 6 = 10

    value(x) = 5.0;
    jit.forward(&output, 1);
    std::cout << "  f(5) = " << output << " (expected: 40)" << std::endl;
    BOOST_CHECK_CLOSE(40.0, output, 1e-10);  // f(5) = 25 + 15 = 40

    jit.clearDerivatives();
    derivative(y) = 1.0;
    jit.computeAdjoints();
    std::cout << "  f'(5) = " << derivative(x) << " (expected: 13)" << std::endl;
    BOOST_CHECK_CLOSE(13.0, derivative(x), 1e-10);  // f'(5) = 10 + 3 = 13
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE_END()
