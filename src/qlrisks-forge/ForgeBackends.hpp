#pragma once

//////////////////////////////////////////////////////////////////////////////
//
//  ForgeBackends.hpp - Backend type selection based on API mode
//
//  This header provides unified type aliases that automatically select
//  the appropriate backend implementation:
//
//    QLRISKS_USE_FORGE_CAPI=1: Uses C API backends (binary compatible)
//    QLRISKS_USE_FORGE_CAPI=0: Uses C++ API backends (requires matching compiler)
//
//  Usage:
//    #include <qlrisks-forge/ForgeBackends.hpp>
//    auto backend = std::make_unique<qlrisks::forge::ScalarBackend>();
//    qlrisks::forge::AVXBackend avxBackend;
//
//////////////////////////////////////////////////////////////////////////////

#ifdef QLRISKS_USE_FORGE_CAPI

// C API mode - binary compatible across compilers
#include <qlrisks-forge/ForgeBackendCAPI.hpp>
#include <qlrisks-forge/ForgeBackendAVX_CAPI.hpp>

namespace qlrisks
{
namespace forge
{

using ScalarBackend = ForgeBackendCAPI;
using AVXBackend = ForgeBackendAVX_CAPI;

}  // namespace forge
}  // namespace qlrisks

#else

// C++ API mode - requires matching compiler/ABI
#include <qlrisks-forge/ForgeBackend.hpp>
#include <qlrisks-forge/ForgeBackendAVX.hpp>

namespace qlrisks
{
namespace forge
{

using ScalarBackend = ForgeBackend;
using AVXBackend = ForgeBackendAVX;

}  // namespace forge
}  // namespace qlrisks

#endif
