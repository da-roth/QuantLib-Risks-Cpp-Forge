/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*
 Copyright (C) 2025 Xcelerit Computing Limited

 Benchmark utilities for QuantLib-Risks / XAD / Forge benchmarks.
 Provides environment detection and output formatting similar to BenchmarkDotNet.
*/

#pragma once

#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <intrin.h>
#include <windows.h>
#else
#include <sys/utsname.h>
#include <unistd.h>
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <cpuid.h>
#endif
#endif

#include <cstring>

namespace BenchmarkUtils {

inline std::string getCpuInfo()
{
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    char brand[49] = {0};
    unsigned int regs[4];

#ifdef _WIN32
    __cpuid(reinterpret_cast<int*>(regs), 0x80000000);
#else
    __get_cpuid(0x80000000, &regs[0], &regs[1], &regs[2], &regs[3]);
#endif

    if (regs[0] >= 0x80000004)
    {
        for (unsigned int i = 0; i < 3; ++i)
        {
#ifdef _WIN32
            __cpuid(reinterpret_cast<int*>(regs), 0x80000002 + i);
#else
            __get_cpuid(0x80000002 + i, &regs[0], &regs[1], &regs[2], &regs[3]);
#endif
            std::memcpy(brand + i * 16, regs, 16);
        }
        std::string result(brand);
        size_t start = result.find_first_not_of(' ');
        if (start != std::string::npos)
            result = result.substr(start);
        return result;
    }
#endif
    return "Unknown CPU";
}

inline std::string getPlatformInfo()
{
#ifdef _WIN32
    typedef LONG(WINAPI * RtlGetVersionPtr)(PRTL_OSVERSIONINFOW);
    HMODULE hMod = GetModuleHandleW(L"ntdll.dll");
    if (hMod)
    {
        auto RtlGetVersion = (RtlGetVersionPtr)GetProcAddress(hMod, "RtlGetVersion");
        if (RtlGetVersion)
        {
            RTL_OSVERSIONINFOW rovi = {0};
            rovi.dwOSVersionInfoSize = sizeof(rovi);
            if (RtlGetVersion(&rovi) == 0)
            {
                std::ostringstream oss;
                oss << "Windows " << rovi.dwMajorVersion << "." << rovi.dwMinorVersion
                    << " (Build " << rovi.dwBuildNumber << ")";
                return oss.str();
            }
        }
    }
    return "Windows";
#else
    struct utsname buf;
    if (uname(&buf) == 0)
    {
        std::ostringstream oss;
        oss << buf.sysname << " " << buf.release;
        return oss.str();
    }
    return "Unknown";
#endif
}

inline std::string getMemoryInfo()
{
#ifdef _WIN32
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    if (GlobalMemoryStatusEx(&memInfo))
    {
        double gb = static_cast<double>(memInfo.ullTotalPhys) / (1024.0 * 1024.0 * 1024.0);
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(0) << gb << " GB";
        return oss.str();
    }
#else
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && page_size > 0)
    {
        double gb = static_cast<double>(pages) * page_size / (1024.0 * 1024.0 * 1024.0);
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(0) << gb << " GB";
        return oss.str();
    }
#endif
    return "Unknown";
}

inline std::string getCompilerInfo()
{
#if defined(_MSC_VER)
    std::ostringstream oss;
    oss << "MSVC " << _MSC_VER / 100 << "." << _MSC_VER % 100;
#if defined(_DEBUG)
    oss << " Debug";
#else
    oss << " Release";
#endif
    return oss.str();
#elif defined(__clang__)
    std::ostringstream oss;
    oss << "Clang " << __clang_major__ << "." << __clang_minor__ << "." << __clang_patchlevel__;
    return oss.str();
#elif defined(__GNUC__)
    std::ostringstream oss;
    oss << "GCC " << __GNUC__ << "." << __GNUC_MINOR__ << "." << __GNUC_PATCHLEVEL__;
    return oss.str();
#else
    return "Unknown Compiler";
#endif
}

inline std::string getSimdInfo()
{
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    unsigned int regs[4];
    std::vector<std::string> features;

#ifdef _WIN32
    __cpuid(reinterpret_cast<int*>(regs), 1);
#else
    __get_cpuid(1, &regs[0], &regs[1], &regs[2], &regs[3]);
#endif

    if (regs[2] & (1 << 0)) features.push_back("SSE3");
    if (regs[2] & (1 << 19)) features.push_back("SSE4.1");
    if (regs[2] & (1 << 20)) features.push_back("SSE4.2");
    if (regs[2] & (1 << 28)) features.push_back("AVX");

#ifdef _WIN32
    __cpuidex(reinterpret_cast<int*>(regs), 7, 0);
#else
    __get_cpuid_count(7, 0, &regs[0], &regs[1], &regs[2], &regs[3]);
#endif

    if (regs[1] & (1 << 5)) features.push_back("AVX2");
    if (regs[1] & (1 << 16)) features.push_back("AVX512F");
    if (regs[1] & (1 << 30)) features.push_back("AVX512BW");

    if (features.empty())
        return "None detected";

    std::ostringstream oss;
    for (size_t i = 0; i < features.size(); ++i)
    {
        if (i > 0) oss << ", ";
        oss << features[i];
    }
    return oss.str();
#else
    return "N/A (non-x86)";
#endif
}

/// Print BenchmarkDotNet-style environment header
inline void printEnvironmentHeader()
{
    std::cout << "\n";
    std::cout << "// * Environment *\n";
    std::cout << "Platform=" << getPlatformInfo() << "\n";
    std::cout << "CPU=" << getCpuInfo() << "\n";
    std::cout << "RAM=" << getMemoryInfo() << "\n";
    std::cout << "SIMD=" << getSimdInfo() << "\n";
    std::cout << "Compiler=" << getCompilerInfo() << "\n";
    std::cout << "\n";
}

/// Format path count for display (e.g., 10000 -> "10K")
inline std::string formatPathCount(int paths)
{
    if (paths >= 1000)
        return std::to_string(paths / 1000) + "K";
    return std::to_string(paths);
}

} // namespace BenchmarkUtils
