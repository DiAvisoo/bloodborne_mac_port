// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: Metal back end M1 check (docs/DESIGN_METAL_BACKEND.md): every SPIR-V module of a shader
// cache directory (<user>/cache/<title>/*.spv) through SPIRV-Cross -> MSL -> MTLFunction.
// `bbgpu-host --msl-check <dir> [msl-cache-dir]`.

#import <Metal/Metal.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "video_core/renderer_metal/mtl_device.h"
#include "video_core/renderer_metal/mtl_shader.h"

namespace {

struct Totals {
    std::atomic<unsigned> ok[7]{}, failed[7]{}, skipped[7]{};
};

} // namespace

extern "C" int bbgpu_msl_check(const char* spv_dir, const char* msl_cache_dir) {
    namespace fs = std::filesystem;
    std::vector<fs::path> files;
    std::error_code ec;
    for (const auto& entry : fs::directory_iterator(spv_dir, ec)) {
        if (entry.path().extension() == ".spv") {
            files.push_back(entry.path());
        }
    }
    if (ec || files.empty()) {
        std::printf("MSL check: no .spv files in %s\n", spv_dir);
        return 2;
    }
    std::sort(files.begin(), files.end());
    Metal::Device device;
    fs::path cache = msl_cache_dir && *msl_cache_dir ? fs::path{msl_cache_dir}
                                                      : fs::temp_directory_path() / "bb-msl-check";
    fs::remove_all(cache, ec); // measure real compiles, not cache hits
    fs::create_directories(cache, ec);
    Metal::ShaderCompiler compiler{device, cache};
    Totals totals;
    std::atomic<size_t> next{0};
    std::mutex print_mutex;
    unsigned printed = 0;
    const auto start = std::chrono::steady_clock::now();
    auto worker = [&] {
        for (size_t i; (i = next.fetch_add(1)) < files.size();) {
            std::ifstream in(files[i], std::ios::binary);
            std::vector<char> bytes((std::istreambuf_iterator<char>(in)), {});
            std::vector<u32> words(bytes.size() / 4);
            std::memcpy(words.data(), bytes.data(), words.size() * 4);
            const auto stage = Metal::ReadShaderStage(words);
            const unsigned s = unsigned(stage);
            // Tessellation stages and geometry shaders compile as variants of a draw (M4); the
            // plain vertex, fragment and compute stages are what M1/M2 need first.
            if (stage == Metal::ShaderStage::TessellationControl ||
                stage == Metal::ShaderStage::TessellationEval ||
                stage == Metal::ShaderStage::Geometry) {
                ++totals.skipped[s];
                continue;
            }
            const std::string name = files[i].stem().string();
            const u64 hash = std::strtoull(name.c_str(), nullptr, 16);
            const auto result = compiler.Compile(words, hash, name);
            if (result.status == Metal::CompileStatus::Ok) {
                ++totals.ok[s];
            } else {
                ++totals.failed[s];
                std::scoped_lock lock{print_mutex};
                if (printed++ < 20) {
                    std::printf("MSL check: FAILED %s (%s): %s\n", name.c_str(),
                                Metal::ShaderStageName(stage), result.error.c_str());
                }
            }
        }
    };
    std::vector<std::thread> threads;
    const unsigned n = std::max(1u, std::thread::hardware_concurrency() / 2);
    for (unsigned t = 0; t < n; ++t) {
        threads.emplace_back(worker);
    }
    for (auto& t : threads) {
        t.join();
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    unsigned ok = 0, failed = 0, skipped = 0;
    std::printf("MSL check on %s: %zu modules, %.1f s on %u threads\n", device.Name().c_str(),
                files.size(), seconds, n);
    for (unsigned s = 0; s < 7; ++s) {
        if (totals.ok[s] + totals.failed[s] + totals.skipped[s] == 0) {
            continue;
        }
        std::printf("  %-12s compiled %4u, failed %4u, not yet (variants) %4u\n",
                    Metal::ShaderStageName(Metal::ShaderStage(s)), totals.ok[s].load(),
                    totals.failed[s].load(), totals.skipped[s].load());
        ok += totals.ok[s];
        failed += totals.failed[s];
        skipped += totals.skipped[s];
    }
    std::printf("%s: MSL check, %u compiled, %u failed, %u left for tessellation/geometry variants\n",
                failed ? "FAIL" : "PASS", ok, failed, skipped);
    return failed ? 1 : 0;
}
