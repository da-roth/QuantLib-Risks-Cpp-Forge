option(QLRISKS_BUILD_TEST_SUITE "Build the QuantLib-Risks test suite" OFF)
option(QLRISKS_ENABLE_FORGE_TESTS "Enable Forge JIT tests (requires Forge)" OFF)
option(QLRISKS_BUILD_BENCHMARK "Build the QuantLib-Risks benchmark suite (Boost.Test)" OFF)
option(QLRISKS_BUILD_BENCHMARK_STANDALONE "Build standalone benchmark executable (no Boost.Test)" OFF)

set(QLRISKS_TEST_SOURCES
    americanoption_xad.cpp
    barrieroption_xad.cpp
    batesmodel_xad.cpp
    bermudanswaption_xad.cpp
    bonds_xad.cpp
    creditdefaultswap_xad.cpp
    europeanoption_xad.cpp
    forwardrateagreement_xad.cpp
    hestonmodel_xad.cpp
    swap_xad.cpp

    utilities_xad.cpp
    quantlibtestsuite_xad.cpp
)

# Forge JIT tests - require Forge's native code backend
# Tests QuantLib-specific JIT functionality (swaption pricing pipeline)
set(QLRISKS_FORGE_TESTS_ENABLED FALSE)
if(QLRISKS_ENABLE_FORGE_TESTS AND TARGET QLRisks::forge)
    message(STATUS "QLRisks test-suite: Adding Forge JIT tests (native code backend)")
    list(APPEND QLRISKS_TEST_SOURCES swaption_jit_pipeline_xad.cpp)
    set(QLRISKS_FORGE_TESTS_ENABLED TRUE)
elseif(QLRISKS_ENABLE_FORGE_TESTS)
    message(WARNING "QLRisks test-suite: QLRISKS_ENABLE_FORGE_TESTS=ON but QLRisks::forge not available")
endif()

# Benchmark sources - Boost.Test based (requires Forge for JIT benchmarks)
set(QLRISKS_BENCHMARK_SOURCES
    quantlibrisks_benchmark.cpp
    swaption_benchmark.cpp
    utilities_xad.cpp
)

# Baseline benchmark sources - Boost.Test based (no Forge required)
set(QLRISKS_BENCHMARK_BASELINE_SOURCES
    quantlibrisks_benchmark.cpp
    swaption_benchmark_baseline.cpp
    utilities_xad.cpp
)

set(QLRISKS_TEST_HEADERS utilities_xad.hpp)

if(QL_BUILD_TEST_SUITE OR QLRISKS_BUILD_TEST_SUITE)
    add_executable(QuantLib-Risks_test_suite ${QLRISKS_TEST_SOURCES} ${QLRISKS_TEST_HEADERS})
    set_target_properties(QuantLib-Risks_test_suite PROPERTIES OUTPUT_NAME "quantlib-risks-test-suite")
    if (NOT Boost_USE_STATIC_LIBS)
        target_compile_definitions(QuantLib-Risks_test_suite PRIVATE BOOST_ALL_DYN_LINK)
    endif()
    target_link_libraries(QuantLib-Risks_test_suite PRIVATE
        ql_library
        ${QL_THREAD_LIBRARIES})

    # ONLY link to Forge if Forge tests are actually enabled
    # This is important because linking Forge brings in AVX2-compiled code
    # which can cause ODR violations if not properly isolated
    if(QLRISKS_FORGE_TESTS_ENABLED)
        message(STATUS "QLRisks test-suite: Linking QLRisks::forge (Forge tests enabled)")
        target_link_libraries(QuantLib-Risks_test_suite PRIVATE QLRisks::forge)
        target_compile_definitions(QuantLib-Risks_test_suite PRIVATE QLRISKS_HAS_FORGE=1)
    else()
        message(STATUS "QLRisks test-suite: NOT linking Forge (Forge tests disabled)")
    endif()

    if (QL_INSTALL_TEST_SUITE)
        install(TARGETS QuantLib-Risks_test_suite RUNTIME DESTINATION ${QL_INSTALL_BINDIR})
    endif()
    add_test(NAME QuantLib-Risks_test_suite COMMAND QuantLib-Risks_test_suite --log_level=message)
endif()

# Benchmark suite with Forge JIT - Boost.Test based
# Requires Forge for JIT scalar and JIT-AVX benchmarks
if(QLRISKS_BUILD_BENCHMARK AND TARGET QLRisks::forge)
    message(STATUS "QLRisks: Building benchmark suite (with Forge JIT)")
    add_executable(QuantLib-Risks_benchmark ${QLRISKS_BENCHMARK_SOURCES} ${QLRISKS_TEST_HEADERS})
    set_target_properties(QuantLib-Risks_benchmark PROPERTIES OUTPUT_NAME "quantlib-risks-benchmark")
    if (NOT Boost_USE_STATIC_LIBS)
        target_compile_definitions(QuantLib-Risks_benchmark PRIVATE BOOST_ALL_DYN_LINK)
    endif()
    target_link_libraries(QuantLib-Risks_benchmark PRIVATE
        ql_library
        QLRisks::forge
        ${QL_THREAD_LIBRARIES})
    target_compile_definitions(QuantLib-Risks_benchmark PRIVATE QLRISKS_HAS_FORGE=1)

    if (QL_INSTALL_TEST_SUITE)
        install(TARGETS QuantLib-Risks_benchmark RUNTIME DESTINATION ${QL_INSTALL_BINDIR})
    endif()
    add_test(NAME QuantLib-Risks_benchmark COMMAND QuantLib-Risks_benchmark --log_level=message)
elseif(QLRISKS_BUILD_BENCHMARK)
    message(WARNING "QLRisks: QLRISKS_BUILD_BENCHMARK=ON but QLRisks::forge not available. Use QLRISKS_BUILD_BENCHMARK_BASELINE for non-Forge benchmarks.")
endif()

# Baseline benchmark suite - Boost.Test based, NO Forge required
# Compares XAD tape-based AD only (useful for baseline measurements)
option(QLRISKS_BUILD_BENCHMARK_BASELINE "Build baseline benchmark suite (Boost.Test, no Forge)" OFF)
if(QLRISKS_BUILD_BENCHMARK_BASELINE)
    message(STATUS "QLRisks: Building baseline benchmark suite (XAD tape only)")
    add_executable(QuantLib-Risks_benchmark_baseline ${QLRISKS_BENCHMARK_BASELINE_SOURCES} ${QLRISKS_TEST_HEADERS})
    set_target_properties(QuantLib-Risks_benchmark_baseline PROPERTIES OUTPUT_NAME "quantlib-risks-benchmark-baseline")
    if (NOT Boost_USE_STATIC_LIBS)
        target_compile_definitions(QuantLib-Risks_benchmark_baseline PRIVATE BOOST_ALL_DYN_LINK)
    endif()
    target_link_libraries(QuantLib-Risks_benchmark_baseline PRIVATE
        ql_library
        ${QL_THREAD_LIBRARIES})

    if (QL_INSTALL_TEST_SUITE)
        install(TARGETS QuantLib-Risks_benchmark_baseline RUNTIME DESTINATION ${QL_INSTALL_BINDIR})
    endif()
    add_test(NAME QuantLib-Risks_benchmark_baseline COMMAND QuantLib-Risks_benchmark_baseline --log_level=message)
endif()

# Standalone benchmark executable - no Boost.Test dependency
# Can be built with or without Forge
if(QLRISKS_BUILD_BENCHMARK_STANDALONE)
    message(STATUS "QLRisks: Building standalone benchmark executable")
    add_executable(QuantLib-Risks_benchmark_standalone benchmark_main.cpp)
    set_target_properties(QuantLib-Risks_benchmark_standalone PROPERTIES
        OUTPUT_NAME "quantlib-risks-benchmark-standalone")
    target_link_libraries(QuantLib-Risks_benchmark_standalone PRIVATE
        ql_library
        ${QL_THREAD_LIBRARIES})

    # Optionally link Forge if available
    if(TARGET QLRisks::forge)
        message(STATUS "QLRisks standalone benchmark: Linking QLRisks::forge")
        target_link_libraries(QuantLib-Risks_benchmark_standalone PRIVATE QLRisks::forge)
        target_compile_definitions(QuantLib-Risks_benchmark_standalone PRIVATE QLRISKS_HAS_FORGE=1)
    else()
        message(STATUS "QLRisks standalone benchmark: Building without Forge (XAD tape only)")
    endif()

    if (QL_INSTALL_TEST_SUITE)
        install(TARGETS QuantLib-Risks_benchmark_standalone RUNTIME DESTINATION ${QL_INSTALL_BINDIR})
    endif()
endif()

# =============================================================================
# Benchmark V2 - Split FD/AAD executables for fair comparison
# =============================================================================

# V2 FD benchmark - Finite Differences using plain double QuantLib
# This is built WITHOUT XAD to ensure fair FD vs AAD comparison
option(QLRISKS_BUILD_BENCHMARK_V2_FD "Build V2 FD benchmark (plain double, no XAD)" OFF)
if(QLRISKS_BUILD_BENCHMARK_V2_FD)
    message(STATUS "QLRisks: Building V2 FD benchmark (plain double)")
    add_executable(benchmark_fd_v2
        benchmark_v2_fd.cpp
        benchmark_v2_common.hpp
        benchmark_v2_pricing.hpp
        PlatformInfo.hpp
    )
    set_target_properties(benchmark_fd_v2 PROPERTIES
        OUTPUT_NAME "benchmark-fd-v2")
    target_link_libraries(benchmark_fd_v2 PRIVATE
        ql_library
        ${QL_THREAD_LIBRARIES})

    if (QL_INSTALL_TEST_SUITE)
        install(TARGETS benchmark_fd_v2 RUNTIME DESTINATION ${QL_INSTALL_BINDIR})
    endif()
endif()

# V2 AAD benchmark - XAD tape-based AAD and optionally Forge JIT
# Built with XAD-enabled QuantLib, optionally links Forge for JIT benchmarks
option(QLRISKS_BUILD_BENCHMARK_V2_AAD "Build V2 AAD benchmark (XAD tape, optionally Forge JIT)" OFF)
if(QLRISKS_BUILD_BENCHMARK_V2_AAD)
    message(STATUS "QLRisks: Building V2 AAD benchmark")
    add_executable(benchmark_aad_v2
        benchmark_v2_aad.cpp
        benchmark_v2_common.hpp
        benchmark_v2_pricing.hpp
        PlatformInfo.hpp
    )
    set_target_properties(benchmark_aad_v2 PROPERTIES
        OUTPUT_NAME "benchmark-aad-v2")
    target_link_libraries(benchmark_aad_v2 PRIVATE
        ql_library
        ${QL_THREAD_LIBRARIES})

    # Link Forge if available for JIT benchmarks
    if(TARGET QLRisks::forge)
        message(STATUS "QLRisks V2 AAD benchmark: Linking QLRisks::forge (JIT benchmarks enabled)")
        target_link_libraries(benchmark_aad_v2 PRIVATE QLRisks::forge)
        target_compile_definitions(benchmark_aad_v2 PRIVATE QLRISKS_HAS_FORGE=1)
    else()
        message(STATUS "QLRisks V2 AAD benchmark: Building without Forge (XAD tape only)")
    endif()

    if (QL_INSTALL_TEST_SUITE)
        install(TARGETS benchmark_aad_v2 RUNTIME DESTINATION ${QL_INSTALL_BINDIR})
    endif()
endif()