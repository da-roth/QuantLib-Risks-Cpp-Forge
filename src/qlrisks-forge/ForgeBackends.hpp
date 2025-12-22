#pragma once

//////////////////////////////////////////////////////////////////////////////
//
//  ForgeBackends.hpp - Backward compatibility header
//
//  This header provides backward compatibility for code using the old
//  qlrisks::forge namespace. New code should use xad::forge directly.
//
//  The actual implementation has moved to xad-forge:
//  https://github.com/da-roth/xad-forge
//
//////////////////////////////////////////////////////////////////////////////

#include <xad-forge/ForgeBackends.hpp>

namespace qlrisks
{
namespace forge
{

using ScalarBackend = xad::forge::ScalarBackend;
using AVXBackend = xad::forge::AVXBackend;

}  // namespace forge
}  // namespace qlrisks
