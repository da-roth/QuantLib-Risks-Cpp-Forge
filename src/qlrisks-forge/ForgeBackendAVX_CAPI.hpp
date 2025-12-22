#pragma once

//////////////////////////////////////////////////////////////////////////////
//
//  ForgeBackendAVX_CAPI.hpp - Backward compatibility header
//
//  This header provides backward compatibility for code using the old
//  qlrisks::forge namespace. New code should use xad::forge directly.
//
//  The actual implementation has moved to xad-forge:
//  https://github.com/da-roth/xad-forge
//
//////////////////////////////////////////////////////////////////////////////

#include <xad-forge/ForgeBackendAVX_CAPI.hpp>

namespace qlrisks
{
namespace forge
{

using ForgeBackendAVX_CAPI = xad::forge::ForgeBackendAVX_CAPI;

}  // namespace forge
}  // namespace qlrisks
