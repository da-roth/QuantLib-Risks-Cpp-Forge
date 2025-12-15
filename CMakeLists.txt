##############################################################################
#
#
#  This file is part of QuantLib-Risks, an adaptor module to enable using XAD with
#  QuantLib. XAD is a fast and comprehensive C++ library for
#  automatic differentiation.
#
#  Copyright (C) 2010-2024 Xcelerit Computing Ltd.
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published
#  by the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
##############################################################################

option(QLRISKS_DISABLE_AAD "Disable using XAD for QuantLib's Real, allowing to run samples with double" OFF)
option(QLRISKS_USE_FORGE_CAPI "Use Forge C API instead of C++ API for binary compatibility" OFF)

add_subdirectory(ql)
if(MSVC)
	set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()
add_subdirectory(Examples)

##############################################################################
# QLRisks-Forge interface library (ForgeBackend adapter)
# NOTE: This must be defined BEFORE test-suite so tests can link to it
##############################################################################

# Find Forge - either as a subdirectory target or as a pre-built package
# Pre-built package is PREFERRED because it isolates Forge's AVX2 compiler flags
# from QuantLib/XAD, preventing ODR violations and ABI incompatibilities.
message(STATUS "QLRisks-Forge: Looking for Forge...")

set(FORGE_FOUND FALSE)

if(QLRISKS_USE_FORGE_CAPI)
    ##########################################################################
    # C API Mode: Use forge_capi shared library for binary compatibility
    ##########################################################################
    message(STATUS "QLRisks-Forge: Using C API mode for binary compatibility")

    # Option 1: Check if forge_capi was added as subdirectory
    if(TARGET forge_capi)
        message(STATUS "QLRisks-Forge: Found forge_capi target (subdirectory mode)")
        set(FORGE_FOUND TRUE)
        set(FORGE_TARGET forge_capi)
        # Get the capi directory for headers
        get_target_property(FORGE_CAPI_SOURCE_DIR forge_capi SOURCE_DIR)
    endif()

    # Option 2: Try find_package for pre-built ForgeCAPI
    if(NOT FORGE_FOUND)
        find_package(ForgeCAPI CONFIG QUIET)
        if(ForgeCAPI_FOUND)
            message(STATUS "QLRisks-Forge: Found ForgeCAPI package (pre-built mode)")
            set(FORGE_FOUND TRUE)
            set(FORGE_TARGET Forge::forge_capi)
        endif()
    endif()

    if(FORGE_FOUND)
        add_library(qlrisks-forge INTERFACE)
        add_library(QLRisks::forge ALIAS qlrisks-forge)

        target_include_directories(qlrisks-forge INTERFACE
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        )

        # Add C API header directory for subdirectory mode
        if(FORGE_CAPI_SOURCE_DIR)
            target_include_directories(qlrisks-forge INTERFACE
                $<BUILD_INTERFACE:${FORGE_CAPI_SOURCE_DIR}>
            )
        endif()

        target_compile_definitions(qlrisks-forge INTERFACE QLRISKS_USE_FORGE_CAPI=1)

        target_link_libraries(qlrisks-forge INTERFACE
            XAD::xad
            ${FORGE_TARGET}
        )

        message(STATUS "QLRisks-Forge: Configured with C API target ${FORGE_TARGET}")
    else()
        message(STATUS "QLRisks-Forge: forge_capi not found - ForgeBackend will not be available")
        message(STATUS "QLRisks-Forge: To enable C API mode, add forge/tools/capi as subdirectory")
    endif()

else()
    ##########################################################################
    # C++ API Mode: Use original Forge library (requires matching compiler)
    ##########################################################################

    # Option 1: Check if forge was added via QL_EXTERNAL_SUBDIRECTORIES (legacy)
    if(TARGET forge)
        message(STATUS "QLRisks-Forge: Found forge target (subdirectory mode)")
        set(FORGE_FOUND TRUE)
        set(FORGE_TARGET forge)
    endif()

    # Option 2: Try find_package for pre-built Forge (preferred)
    if(NOT FORGE_FOUND)
        find_package(Forge CONFIG QUIET)
        if(Forge_FOUND)
            message(STATUS "QLRisks-Forge: Found Forge package (pre-built mode) version ${Forge_VERSION}")
            set(FORGE_FOUND TRUE)
            set(FORGE_TARGET Forge::forge)
        endif()
    endif()

    if(FORGE_FOUND)
        add_library(qlrisks-forge INTERFACE)
        add_library(QLRisks::forge ALIAS qlrisks-forge)

        target_include_directories(qlrisks-forge INTERFACE
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        )

        # For subdirectory mode, add Forge source directories
        if(TARGET forge)
            get_target_property(FORGE_SOURCE_DIR forge SOURCE_DIR)
            if(FORGE_SOURCE_DIR)
                target_include_directories(qlrisks-forge INTERFACE
                    $<BUILD_INTERFACE:${FORGE_SOURCE_DIR}/src>
                    $<BUILD_INTERFACE:${FORGE_SOURCE_DIR}/tools>
                )
            endif()
        endif()

        target_link_libraries(qlrisks-forge INTERFACE
            XAD::xad
            ${FORGE_TARGET}
        )

        message(STATUS "QLRisks-Forge: Configured with C++ API target ${FORGE_TARGET}")
    else()
        message(STATUS "QLRisks-Forge: Forge not found - ForgeBackend will not be available")
        message(STATUS "QLRisks-Forge: To enable, either:")
        message(STATUS "  1. Add forge to QL_EXTERNAL_SUBDIRECTORIES (not recommended)")
        message(STATUS "  2. Set CMAKE_PREFIX_PATH to pre-built Forge installation (recommended)")
    endif()
endif()

##############################################################################
# Test suite (after QLRisks::forge is defined)
##############################################################################

if(NOT QLRISKS_DISABLE_AAD)
    # the test suite is not supporting double
	add_subdirectory(test-suite)
endif()