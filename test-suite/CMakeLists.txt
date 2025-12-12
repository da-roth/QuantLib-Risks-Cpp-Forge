option(QLRISKS_BUILD_TEST_SUITE "Build the QuantLib-Risks test suite" OFF)
option(QLRISKS_ENABLE_FORGE_TESTS "Enable Forge JIT tests (requires Forge)" OFF)
option(QLRISKS_BUILD_BENCHMARK "Build the QuantLib-Risks benchmark suite" OFF)

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
    swaption_jit_pipeline_xad.cpp

    utilities_xad.cpp
    quantlibtestsuite_xad.cpp
)

# Benchmark sources (separate from test suite)
set(QLRISKS_BENCHMARK_SOURCES
    quantlibrisks_benchmark.cpp
    swaption_benchmark.cpp
    utilities_xad.cpp
)

# Forge JIT tests - only added when explicitly enabled AND Forge is available
# These tests require Forge's JIT compiler for AAD
set(QLRISKS_FORGE_TESTS_ENABLED FALSE)
if(QLRISKS_ENABLE_FORGE_TESTS AND TARGET QLRisks::forge)
    message(STATUS "QLRisks test-suite: Adding Forge JIT tests")
    list(APPEND QLRISKS_TEST_SOURCES jit_xad.cpp forgebackend_xad.cpp)
    set(QLRISKS_FORGE_TESTS_ENABLED TRUE)
elseif(QLRISKS_ENABLE_FORGE_TESTS)
    message(WARNING "QLRisks test-suite: QLRISKS_ENABLE_FORGE_TESTS=ON but QLRisks::forge not available")
endif()

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

# Benchmark suite - separate executable for CI benchmarks
# Requires Forge to be available
if(QLRISKS_BUILD_BENCHMARK AND TARGET QLRisks::forge)
    message(STATUS "QLRisks: Building benchmark suite")
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
    message(WARNING "QLRisks: QLRISKS_BUILD_BENCHMARK=ON but QLRisks::forge not available")
endif()