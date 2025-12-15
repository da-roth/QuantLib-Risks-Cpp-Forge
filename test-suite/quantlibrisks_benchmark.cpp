/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*
 Copyright (C) 2025 Xcelerit Computing Limited

 This file is part of QuantLib-Risks / XAD / Forge integration.

 Main entry point for the QuantLib-Risks benchmark suite.
*/

#define BOOST_TEST_MODULE QuantLibRisksBenchmark

#include <boost/test/included/unit_test.hpp>

/* Use BOOST_MSVC instead of _MSC_VER since some other vendors (Metrowerks,
   for example) also #define _MSC_VER
*/
#if !defined(BOOST_ALL_NO_LIB) && defined(BOOST_MSVC)
#    include <ql/auto_link.hpp>
#endif
