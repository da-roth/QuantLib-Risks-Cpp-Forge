#pragma once

//////////////////////////////////////////////////////////////////////////////
//
//  ForgeBackendCAPI.hpp - Backward compatibility header
//
//  This header provides backward compatibility for code using the old
//  qlrisks::forge namespace. New code should use xad::forge directly.
//
//  The actual implementation has moved to xad-forge:
//  https://github.com/da-roth/xad-forge
//
//////////////////////////////////////////////////////////////////////////////

#include <xad-forge/ForgeBackendCAPI.hpp>

namespace qlrisks
{
namespace forge
{

using ForgeBackendCAPI = xad::forge::ForgeBackendCAPI;

}  // namespace forge
}  // namespace qlrisks
