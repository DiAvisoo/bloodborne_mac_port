// SPDX-License-Identifier: GPL-2.0-or-later
// Real Metal cache + CPU DMA fast paths, with an unprotected host mirror as in split mode.
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <setjmp.h>
#include <sys/mman.h>
#include "common/elf_info.h"
#include "../gpu/bbgpu.h"
#include "video_core/texture_cache/blit_helper.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_runtime.h"

namespace Core {
class Emulator {
public:
    static void FillElfInfo(const BbGpuConfig& config) {
        auto& info = Common::ElfInfo::Instance();
        info.initialized = true;
        info.game_serial = config.serial ? config.serial : "UNKNOWN";
        info.title = config.title ? config.title : "";
        info.sdk_ver = config.sdk_version;
        info.psf_attributes.raw = config.psf_attributes;
    }
};
}

namespace {
constexpr uintptr_t Base = 0x7000000000ULL;
constexpr size_t MappedSize = 4 * 1024 * 1024;
constexpr u32 Size = 64 * 1024; // above the small, uncached stream-buffer threshold
bool Covered(uintptr_t p, uint64_t n) { return p >= Base && p <= Base + MappedSize && n <= Base + MappedSize - p; }
int failures = 0;
void Check(bool ok, const char* message) {
    std::printf("%s: %s\n", ok ? "PASS" : "FAIL", message);
    failures += !ok;
}
}
#pragma GCC visibility push(default)
extern "C" {
__thread sigjmp_buf* runtime_fault_recover = nullptr;
uint64_t runtime_disabled_optimizations = 0;
uint64_t runtime_tsc_frequency() { return 1000000000; }
uint64_t runtime_process_time_us() { return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
uint64_t runtime_process_time_counter() { return runtime_process_time_us() * 1000; }
void runtime_thread_attach_host(const char*) {}
void runtime_thread_set_blocked(const char*, uint64_t) {}
void runtime_thread_clear_blocked() {}
uint64_t runtime_thread_cpu_ns(int) { return 0; }
void runtime_game_cpu_since_last(uint64_t* a, uint64_t* b, char n[32]) { *a = *b = 0; n[0] = 0; }
int runtime_file_translate(const char*, char*, size_t) { std::abort(); }
int32_t* runtime_errno() { std::abort(); }
void runtime_restart() { std::abort(); }
uint64_t runtime_memory_clamp(uintptr_t p, uint64_t n) { return Covered(p, n) ? n : 0; }
int runtime_memory_region(uintptr_t p, uintptr_t* a, uintptr_t* b, int* mapped) {
    if (!Covered(p, 1)) return 0;
    *a = Base; *b = Base + MappedSize; *mapped = 1; return 1;
}
void runtime_memory_set_gpu_hooks(void (*map)(uintptr_t, uint64_t), void (*)(uintptr_t, uint64_t), void (*)(uintptr_t, uint64_t)) {
    map(Base, MappedSize);
}
// Production split mode protects the separate guest process, never this host mirror.
void runtime_memory_gpu_protect(uintptr_t p, uint64_t n, int, int) { assert(Covered(p,n)); }
int runtime_memory_write_backing(uintptr_t p, const void* data, uint64_t n) {
    if (!Covered(p,n)) return 0;
    std::memcpy(reinterpret_cast<void*>(p), data, n); return 1;
}
int runtime_memory_read_backing(uintptr_t p, void* data, uint64_t n) {
    if (!Covered(p,n)) return 0;
    std::memcpy(data, reinterpret_cast<void*>(p), n); return 1;
}
}
#pragma GCC visibility pop
int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    setenv("BB_NULL_GPU", "1", 1);
    setenv("BB_DRAW_PIPE", "0", 1);
    setenv("BB_PIPELINE_CACHE", "0", 1);
    setenv("BB_READBACKS", "2", 1);
    void* mapped = mmap(reinterpret_cast<void*>(Base), MappedSize, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANON, -1, 0);
    assert(mapped == reinterpret_cast<void*>(Base));
    const BbGpuConfig config = {"Coherency test", "TEST", nullptr, 0x05050031, 0, 640, 360};
    Core::Emulator::FillElfInfo(config);
    Metal::Device device;
    Vulkan::Scheduler scheduler(device, false);
    Vulkan::Runtime runtime(device, scheduler);
    auto liverpool = std::make_unique<AmdGpu::Liverpool>();
    auto owner = std::make_unique<Vulkan::Rasterizer>(device, scheduler, runtime, liverpool.get());
    auto& rasterizer = *owner;
    auto& cache = rasterizer.GetBufferCache();
    VideoCore::Buffer readback(device, VideoCore::MemoryType::HostCached, 0, Size, "coherency result");
    const auto contents_are = [&](uintptr_t address, u32 expected) {
        const auto [buffer, offset] = cache.ObtainBuffer(address, Size, false);
        const VideoCore::BufferCopy copy{offset, 0, Size};
        runtime.CopyBuffer(*buffer, readback, std::span{&copy, 1});
        scheduler.Finish();
        const auto* words = reinterpret_cast<const u32*>(readback.mapped_data.data());
        return std::all_of(words, words + Size / 4, [&](u32 x) { return x == expected; });
    };
    const auto prime = [&](uintptr_t address) {
        std::fill_n(reinterpret_cast<u32*>(address), Size/4, 0x11111111u);
        Check(contents_are(address, 0x11111111), "initial GPU upload");
        Check(!cache.IsRegionCpuModified(address, Size), "uploaded region is clean");
    };
    const uintptr_t filled = Base, copied = Base + Size, source = Base + 2 * Size;
    prime(filled);
    rasterizer.FillBuffer(filled, Size, 0x22222222, false);
    Check(*reinterpret_cast<u32*>(filled) == 0x22222222, "CPU DMA fill changed guest mirror");
    Check(contents_are(filled, 0x22222222), "GPU sees CPU DMA fill after buffer reuse");
    prime(copied);
    std::fill_n(reinterpret_cast<u32*>(source), Size/4, 0x33333333u);
    rasterizer.CopyBuffer(copied, source, Size, false, false);
    Check(*reinterpret_cast<u32*>(copied) == 0x33333333, "CPU DMA copy changed guest mirror");
    Check(contents_are(copied, 0x33333333), "GPU sees CPU DMA copy after buffer reuse");
    const uintptr_t written = Base + 3 * Size;
    prime(written);
    rasterizer.WriteMemory(written, reinterpret_cast<void*>(source), Size);
    Check(contents_are(written, 0x33333333), "GPU sees WRITE_DATA/constant dump into reused memory");

    // The CPU overwrite must not change an earlier, still-queued upload's snapshot.
    const uintptr_t queued = Base + 4 * Size;
    std::fill_n(reinterpret_cast<u32*>(queued), Size/4, 0x44444444u);
    const auto [old_buffer, old_offset] = cache.ObtainBuffer(queued, Size, false);
    const VideoCore::BufferCopy earlier{old_offset, 0, Size};
    runtime.CopyBuffer(*old_buffer, readback, std::span{&earlier, 1});
    rasterizer.FillBuffer(queued, Size, 0x55555555, false);
    scheduler.Finish();
    const auto* words = reinterpret_cast<const u32*>(readback.mapped_data.data());
    Check(std::all_of(words, words + Size/4, [](u32 x) { return x == 0x44444444; }),
          "CPU overwrite preserves earlier queued upload contents");
    Check(contents_are(queued, 0x55555555), "later use sees the overwritten contents");

    // A small CPU write must first recover GPU-written neighbours on the same page.
    const uintptr_t partial = Base + 5 * Size;
    const auto [gpu_buffer, gpu_offset] = cache.ObtainBuffer(partial, Size, true);
    runtime.FillBuffer(*gpu_buffer, gpu_offset, Size, 0x66666666);
    const u32 replacement = 0x77777777;
    rasterizer.WriteMemory(partial + 4, &replacement, sizeof replacement);
    const auto [result_buffer, result_offset] = cache.ObtainBuffer(partial, Size, false);
    const VideoCore::BufferCopy result{result_offset, 0, Size};
    runtime.CopyBuffer(*result_buffer, readback, std::span{&result, 1});
    scheduler.Finish();
    bool preserved = true;
    for (u32 i = 0; i < Size/4; ++i) {
        preserved &= words[i] == (i == 1 ? replacement : 0x66666666);
    }
    Check(preserved, "partial CPU write preserves neighbouring GPU output");
    std::printf("%d failures\n", failures);
    return failures ? 1 : 0;
}
