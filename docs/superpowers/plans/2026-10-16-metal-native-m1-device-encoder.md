# Metal-native M1: Device, ticks and Encoder — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The foundation every later milestone builds on: Metal objects held from plain C++, a
Metal device with submission ticks and limits, and a `Metal::Encoder` with native types (render
passes, state, draws, dispatches, blits, keep-alive, submit), all unit-tested on the GPU.

**Architecture:** New files in `gpu/shadps4/video_core/renderer_metal/` compiled into the arm64
host library (`out/gpu-arm64/libbbgpu.dylib`; `gpu/CMakeLists.txt` already globs that directory
and compiles `.mm` files with ARC). The Encoder re-implements `mtl_frame.mm`'s pass/state logic
with Metal-native arguments; `mtl_frame.mm`, the Vulkan layer and the rest of the renderer stay
untouched in M1 (they are replaced in M3), so the existing game path keeps building and running.

**Tech Stack:** Objective-C++ (ARC), Metal, C++20, the project's `common/assert.h` and
`common/logging/log.h`; tests are a standalone arm64 executable linked against the library.

**Spec:** `docs/superpowers/specs/2026-10-16-metal-native-renderer-design.md` (components 1 and 2).
**Roadmap:** `docs/superpowers/plans/2026-10-16-metal-native-roadmap.md`.

## Global Constraints

- Work in the worktree `~/Developer/GitHub/bb-metal-wt`, branch `metal-native`.
- Plain C++ headers (`.h` used by `.cpp`) must not import Objective-C; Metal objects cross as
  `void*` or `Metal::Object`. Only `*_impl.h` and `.mm` files may `#import <Metal/Metal.h>`.
- Limits keep MoltenVK's values: uniform/storage/texel alignment 16, non-coherent atom 16,
  max anisotropy 16, max LOD bias 16, max samplers 500000; max buffer length from
  `MTLDevice.maxBufferLength`.
- Ticks: start at 1, +1 per commit, `IsFree(t)` ⇔ GPU finished every commit up to `t`.
- Command buffers use unretained references (`commandBufferWithUnretainedReferences`): owners keep
  resources alive; `KeepAlive` covers temporaries.
- File headers: `// SPDX-License-Identifier: GPL-2.0-or-later` then a `// bbport:` line saying
  what the file is.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

**Build and test command** (used by every task; `$WT` = `~/Developer/GitHub/bb-metal-wt`):

```bash
cd ~/Developer/GitHub/bb-metal-wt && bash scripts/macos/build_gpu_arm64.sh >/tmp/m1-build.log 2>&1 \
  && /usr/bin/clang++ -arch arm64 -mmacosx-version-min=15.0 -std=c++20 -fobjc-arc -O2 -g -Wall -Wextra -Werror \
     -Igpu/shadps4 tests/test_mtl_encoder.mm -Lout/gpu-arm64 -lbbgpu -Wl,-rpath,$PWD/out/gpu-arm64 \
     -framework Metal -framework Foundation -o out/mtl-encoder-test \
  && out/mtl-encoder-test; echo "exit $?"
```

If the library build fails, read `/tmp/m1-build.log` (`grep -n "error:" /tmp/m1-build.log`).

## Review Focus

1. A render pass that clears but draws nothing must still clear (the game clears targets with
   empty passes) — Task 5, `TestClearOnlyPass`.
2. A pass interrupted by a copy or dispatch must keep what was drawn before it (reopen with Load,
   never Clear) — Task 5, `TestPassReopenKeepsDrawing`.
3. Render state set in one pass must still apply in the next pass without being set again (Vulkan
   dynamic state persists across passes in a command buffer) — Task 5, `TestStatePersistsAcrossPasses`.
4. Constant data larger than Metal's 4 KiB `setBytes` limit must still arrive — Task 4,
   `TestLargeBytes`.
5. A `Submit` with nothing recorded must still produce a tick that completes (the scheduler submits
   every frame), and keep-alive objects must be released once the GPU is done — Task 2
   `TestTicks`, Task 3 `TestKeepAlive`.

---

### Task 1: `Metal::Object` and the test harness

**Files:**
- Create: `gpu/shadps4/video_core/renderer_metal/mtl_object.h`
- Create: `gpu/shadps4/video_core/renderer_metal/mtl_object.cpp`
- Create: `tests/test_mtl_encoder.mm`

**Interfaces:**
- Consumes: nothing.
- Produces: `Metal::Object` (move-only owner of one retained Objective-C object:
  `static Object Adopt(void* retained)`, `static Object Retain(void* object)`, `void* Get() const`,
  `explicit operator bool() const`, `void Reset()`); `Metal::BufferRef { void* buffer; u64 offset; }`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_mtl_encoder.mm`:

```objc
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: unit test for the Metal-native foundation (docs/superpowers/plans/
// 2026-10-16-metal-native-m1-device-encoder.md): Metal::Object, Metal::Device ticks and limits,
// Metal::Encoder blits, dispatches, render passes and draws, on the system's GPU. No window.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <vector>
#include "video_core/renderer_metal/mtl_object.h"

namespace {

int failures = 0;

void Expect(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++failures;
    }
}

std::atomic<int> probes_alive{0};

} // namespace

/// Counts live instances, to see when an Object releases what it holds.
@interface Probe : NSObject
@end
@implementation Probe
- (instancetype)init {
    if ((self = [super init])) {
        probes_alive.fetch_add(1);
    }
    return self;
}
- (void)dealloc {
    probes_alive.fetch_sub(1);
}
@end

namespace {

void TestObject() {
    @autoreleasepool {
        {
            Metal::Object a = Metal::Object::Adopt((__bridge_retained void*)[Probe new]);
            Expect(bool(a) && probes_alive == 1, "Object::Adopt holds the object");
            Metal::Object b = std::move(a);
            Expect(!a && bool(b) && probes_alive == 1, "moving an Object moves the reference");
            Metal::Object c = Metal::Object::Retain(b.Get());
            b.Reset();
            Expect(!b && probes_alive == 1, "Object::Retain keeps the object after the first Reset");
        }
        Expect(probes_alive == 0, "the last Object releases the object");
    }
}

} // namespace

int main() {
    @autoreleasepool {
        TestObject();
    }
    if (failures) {
        std::printf("FAILED: %d check(s)\n", failures);
        return 1;
    }
    std::printf("PASS: Metal-native foundation (object, device ticks, encoder)\n");
    return 0;
}
```

- [ ] **Step 2: Run it to see it fail**

Run the build and test command (Global Constraints).
Expected: compile error `'video_core/renderer_metal/mtl_object.h' file not found`.

- [ ] **Step 3: Implement `Metal::Object`**

Create `gpu/shadps4/video_core/renderer_metal/mtl_object.h`:

```cpp
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: Metal objects held from plain C++ (Metal-native renderer, spec component 1).

#pragma once

#include <utility>
#include "common/types.h"

namespace Metal {

/// Owns one retained Objective-C object (an `id`) for code that is plain C++. Objective-C++
/// makes one with Object::Adopt((__bridge_retained void*)obj) and reads it back with
/// (__bridge id<...>)object.Get() (unretained). Move-only; the last owner releases it.
class Object {
public:
    Object() = default;
    ~Object() {
        Reset();
    }
    Object(Object&& other) noexcept : ptr{std::exchange(other.ptr, nullptr)} {}
    Object& operator=(Object&& other) noexcept {
        if (this != &other) {
            Reset();
            ptr = std::exchange(other.ptr, nullptr);
        }
        return *this;
    }
    Object(const Object&) = delete;
    Object& operator=(const Object&) = delete;

    /// Takes over a +1 reference (from __bridge_retained or a `new…` method).
    static Object Adopt(void* retained) {
        Object object;
        object.ptr = retained;
        return object;
    }
    /// Adds a reference to `object` (an unretained pointer).
    static Object Retain(void* object);

    [[nodiscard]] void* Get() const noexcept {
        return ptr;
    }
    explicit operator bool() const noexcept {
        return ptr != nullptr;
    }
    /// Releases the object (if any).
    void Reset();

private:
    void* ptr{};
};

/// A buffer (id<MTLBuffer>, not retained) and a byte offset into it.
struct BufferRef {
    void* buffer{};
    u64 offset{};
};

} // namespace Metal
```

Create `gpu/shadps4/video_core/renderer_metal/mtl_object.cpp`:

```cpp
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: Metal::Object reference counting through CoreFoundation (plain C++).

#include <CoreFoundation/CoreFoundation.h>
#include "video_core/renderer_metal/mtl_object.h"

namespace Metal {

Object Object::Retain(void* object) {
    if (object) {
        CFRetain(object);
    }
    return Adopt(object);
}

void Object::Reset() {
    if (ptr) {
        CFRelease(ptr);
        ptr = nullptr;
    }
}

} // namespace Metal
```

- [ ] **Step 4: Run it to see it pass**

Run the build and test command. Expected: `PASS: Metal-native foundation (object, device ticks, encoder)` and `exit 0`.

- [ ] **Step 5: Commit**

```bash
git add gpu/shadps4/video_core/renderer_metal/mtl_object.h gpu/shadps4/video_core/renderer_metal/mtl_object.cpp tests/test_mtl_encoder.mm
git commit -m "Metal-native M1: Metal::Object (retained Objective-C objects from plain C++), with a GPU unit test

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Device ticks and limits

**Files:**
- Modify: `gpu/shadps4/video_core/renderer_metal/mtl_device.h` (whole file below)
- Modify: `gpu/shadps4/video_core/renderer_metal/mtl_device_impl.h` (add the event)
- Modify: `gpu/shadps4/video_core/renderer_metal/mtl_device.mm` (whole file below)
- Test: `tests/test_mtl_encoder.mm`

**Interfaces:**
- Consumes: nothing new.
- Produces: `Metal::Limits` (fields below); `Metal::Device` keeps its constructor
  `Device(void* mtl_device = nullptr, void* mtl_queue = nullptr)` (the `metal-backend` layer still
  passes MoltenVK's objects) and adds `GetLimits()`, `NativeDevice()`, `NativeQueue()`,
  `CurrentTick()`, `KnownGpuTick()`, `IsFree(u64)`, `Wait(u64)`, `Finish()`,
  `u64 Commit(void* command_buffer)`, `ErrorCount()`.

- [ ] **Step 1: Write the failing test**

In `tests/test_mtl_encoder.mm` add `#include "video_core/renderer_metal/mtl_device.h"` after the
`mtl_object.h` include, add this function before `main` (inside the second anonymous namespace),
and call `TestTicks();` in `main` after `TestObject();`:

```objc
void TestTicks() {
    Metal::Device device;
    const Metal::Limits& limits = device.GetLimits();
    Expect(limits.max_buffer_length >= (u64(1) << 30), "max buffer length read from the device");
    Expect(limits.uniform_alignment == 16 && limits.storage_alignment == 16 &&
               limits.max_samplers == 500000,
           "limits keep MoltenVK's values");
    Expect(device.CurrentTick() == 1 && device.KnownGpuTick() == 0, "ticks start at 1");

    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)device.NativeQueue();
    id<MTLCommandBuffer> empty = [queue commandBuffer];
    const u64 first = device.Commit((__bridge void*)empty);
    Expect(first == 1 && device.CurrentTick() == 2, "Commit signals the current tick and advances");
    device.Wait(first);
    Expect(device.IsFree(first), "an empty command buffer completes");
    Expect(!device.IsFree(2), "an uncommitted tick is not free");

    for (int i = 0; i < 3; ++i) {
        device.Commit((__bridge void*)[queue commandBuffer]);
    }
    device.Finish();
    Expect(device.IsFree(4) && device.KnownGpuTick() >= 4, "Finish waits for every commit");
    Expect(device.ErrorCount() == 0, "no command buffer failed");
}
```

- [ ] **Step 2: Run it to see it fail**

Run the build and test command. Expected: compile errors (`no type named 'Limits'`,
`no member named 'GetLimits'`, …).

- [ ] **Step 3: Implement**

Replace `gpu/shadps4/video_core/renderer_metal/mtl_device.h` with:

```cpp
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the Metal device, queue and submission ticks (Metal-native renderer, spec component 1).
// The interface began as Huax-1839/bloodborne-macos's Metal::Device (GPL-2.0-or-later). On the
// metal-backend path the device and queue are MoltenVK's (VK_EXT_metal_objects).

#pragma once

#include <atomic>
#include <memory>
#include <string>
#include "common/types.h"

namespace Metal {

/// Limits the caches use. MoltenVK's values, so the caches behave as with MoltenVK.
struct Limits {
    u64 max_buffer_length{};       ///< MTLDevice.maxBufferLength
    u64 recommended_working_set{}; ///< MTLDevice.recommendedMaxWorkingSetSize
    u32 uniform_alignment{16};
    u32 storage_alignment{16};
    u32 texel_buffer_alignment{16};
    u32 non_coherent_atom{16};
    float max_anisotropy{16.0f};
    float max_lod_bias{16.0f};
    u32 max_samplers{500000};
};

class Device {
public:
    /// `mtl_device`/`mtl_queue`: id<MTLDevice>/id<MTLCommandQueue> to use (MoltenVK's); null: the
    /// system default device and a queue of our own.
    explicit Device(void* mtl_device = nullptr, void* mtl_queue = nullptr);
    /// Waits for every committed command buffer.
    ~Device();

    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    struct Impl;
    Impl& GetImpl() const {
        return *impl;
    }

    const std::string& Name() const {
        return name;
    }
    const Limits& GetLimits() const noexcept {
        return limits;
    }
    /// id<MTLDevice> and id<MTLCommandQueue>, unretained.
    void* NativeDevice() const noexcept;
    void* NativeQueue() const noexcept;

    /// The tick the next Commit() signals. Ticks start at 1 and grow by one per commit.
    [[nodiscard]] u64 CurrentTick() const noexcept {
        return current_tick.load(std::memory_order_acquire);
    }
    /// The highest tick the GPU is known to have finished.
    [[nodiscard]] u64 KnownGpuTick() const noexcept {
        return gpu_tick.load(std::memory_order_acquire);
    }
    /// True once the GPU finished every command buffer up to `tick`.
    [[nodiscard]] bool IsFree(u64 tick) const;
    /// Blocks until `tick` is done; also returns for command buffers that failed. `tick` must have
    /// been committed (asserted).
    void Wait(u64 tick);
    /// Waits for every command buffer committed so far.
    void Finish();
    /// Commits `command_buffer` (id<MTLCommandBuffer>) so that it signals CurrentTick() when done,
    /// and returns that tick. One thread commits at a time, so ticks follow commit order.
    u64 Commit(void* command_buffer);
    /// Command buffers that completed with an error (each one is logged).
    [[nodiscard]] u64 ErrorCount() const noexcept {
        return errors.load(std::memory_order_relaxed);
    }

private:
    /// Raises the known GPU tick to the shared event's value.
    void Refresh() const;

    std::unique_ptr<Impl> impl;
    std::string name;
    Limits limits;
    std::atomic<u64> current_tick{1};
    mutable std::atomic<u64> gpu_tick{0};
    std::atomic<u64> errors{0};
};

} // namespace Metal
```

In `gpu/shadps4/video_core/renderer_metal/mtl_device_impl.h` change the struct to:

```objc
struct Device::Impl {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLSharedEvent> event; ///< signalled with each command buffer's tick
};
```

Replace `gpu/shadps4/video_core/renderer_metal/mtl_device.mm` with:

```objc
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: Metal::Device (Metal-native renderer, spec component 1).

#include <atomic>
#include "common/assert.h"
#include "common/logging/log.h"
#include "video_core/renderer_metal/mtl_device_impl.h"

namespace Metal {

namespace {

void RaiseTo(std::atomic<u64>& value, u64 tick) {
    u64 known = value.load(std::memory_order_acquire);
    while (known < tick &&
           !value.compare_exchange_weak(known, tick, std::memory_order_acq_rel)) {
    }
}

} // namespace

Device::Device(void* mtl_device, void* mtl_queue) : impl{std::make_unique<Impl>()} {
    @autoreleasepool {
        impl->device = mtl_device ? (__bridge id<MTLDevice>)mtl_device : MTLCreateSystemDefaultDevice();
        ASSERT_MSG(impl->device, "No Metal device");
        impl->queue = mtl_queue ? (__bridge id<MTLCommandQueue>)mtl_queue : [impl->device newCommandQueue];
        ASSERT_MSG(impl->queue, "No Metal command queue");
        impl->event = [impl->device newSharedEvent];
        ASSERT_MSG(impl->event, "No Metal shared event");
        name = impl->device.name.UTF8String;
        limits.max_buffer_length = impl->device.maxBufferLength;
        limits.recommended_working_set = impl->device.recommendedMaxWorkingSetSize;
    }
}

Device::~Device() {
    Finish();
}

void* Device::NativeDevice() const noexcept {
    return (__bridge void*)impl->device;
}

void* Device::NativeQueue() const noexcept {
    return (__bridge void*)impl->queue;
}

void Device::Refresh() const {
    RaiseTo(gpu_tick, impl->event.signaledValue);
}

bool Device::IsFree(u64 tick) const {
    if (KnownGpuTick() >= tick) {
        return true;
    }
    Refresh();
    return KnownGpuTick() >= tick;
}

void Device::Wait(u64 tick) {
    ASSERT_MSG(tick < CurrentTick(), "Waiting for tick {} that was never committed", tick);
    while (!IsFree(tick)) {
        // A failed command buffer may never signal the event; its completion handler raises the
        // known tick instead, so wait in slices and look again.
        [impl->event waitUntilSignaledValue:tick timeoutMS:100];
    }
}

void Device::Finish() {
    const u64 last = CurrentTick() - 1;
    if (last != 0) {
        Wait(last);
    }
}

u64 Device::Commit(void* command_buffer) {
    id<MTLCommandBuffer> cmd = (__bridge id<MTLCommandBuffer>)command_buffer;
    const u64 tick = current_tick.fetch_add(1, std::memory_order_acq_rel);
    [cmd encodeSignalEvent:impl->event value:tick];
    std::atomic<u64>* known = &gpu_tick;
    std::atomic<u64>* failed = &errors;
    [cmd addCompletedHandler:^(id<MTLCommandBuffer> done) {
        if (done.status == MTLCommandBufferStatusError) {
            failed->fetch_add(1, std::memory_order_relaxed);
            LOG_ERROR(Render, "Metal command buffer of tick {} failed: {}", tick,
                      done.error ? done.error.localizedDescription.UTF8String : "unknown error");
        }
        RaiseTo(*known, tick);
    }];
    [cmd commit];
    return tick;
}

} // namespace Metal
```

- [ ] **Step 4: Run it to see it pass**

Run the build and test command. Expected: `PASS: …` and `exit 0`.

- [ ] **Step 5: Check the old path still builds**

Run: `grep -n "Metal::Device\|Device(" gpu/shadps4/video_core/renderer_metal/mtl_vk_layer.mm | head`
Expected: the layer constructs `Device` with two pointers, which still matches. The library build
in Step 4 succeeded, so it compiles.

- [ ] **Step 6: Commit**

```bash
git add gpu/shadps4/video_core/renderer_metal/mtl_device.h gpu/shadps4/video_core/renderer_metal/mtl_device_impl.h gpu/shadps4/video_core/renderer_metal/mtl_device.mm tests/test_mtl_encoder.mm
git commit -m "Metal-native M1: Device ticks (MTLSharedEvent), limits with MoltenVK's values, error count

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Encoder skeleton — blits, keep-alive, submit

**Files:**
- Create: `gpu/shadps4/video_core/renderer_metal/mtl_encoder.h` (complete interface, all tasks)
- Create: `gpu/shadps4/video_core/renderer_metal/mtl_encoder.mm` (blit, keep-alive, submit now;
  compute in Task 4; passes and draws in Task 5)
- Test: `tests/test_mtl_encoder.mm`

**Interfaces:**
- Consumes: `Metal::Device` (`NativeDevice`, `NativeQueue`, `Commit`), `Metal::Object`,
  `Metal::BufferRef`.
- Produces: the whole `Metal::Encoder` interface below (Tasks 4–5 fill in the rest of the bodies).

- [ ] **Step 1: Write the failing test**

In `tests/test_mtl_encoder.mm` add `#include "video_core/renderer_metal/mtl_encoder.h"`, these
helpers and tests (second anonymous namespace), and call `TestBlits(); TestKeepAlive();` in `main`
after `TestTicks();`:

```objc
id<MTLBuffer> SharedBuffer(Metal::Device& device, NSUInteger length) {
    id<MTLDevice> dev = (__bridge id<MTLDevice>)device.NativeDevice();
    id<MTLBuffer> buffer = [dev newBufferWithLength:length options:MTLResourceStorageModeShared];
    std::memset(buffer.contents, 0, length);
    return buffer;
}

Metal::BufferRef Ref(id<MTLBuffer> buffer, u64 offset = 0) {
    return {(__bridge void*)buffer, offset};
}

id<MTLTexture> Texture(Metal::Device& device, u32 width, u32 height) {
    id<MTLDevice> dev = (__bridge id<MTLDevice>)device.NativeDevice();
    MTLTextureDescriptor* desc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                           width:width
                                                          height:height
                                                       mipmapped:NO];
    desc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    desc.storageMode = MTLStorageModePrivate;
    return [dev newTextureWithDescriptor:desc];
}

u32 Pixel(id<MTLBuffer> readback, u32 width, u32 x, u32 y) {
    u32 value;
    std::memcpy(&value, static_cast<const u8*>(readback.contents) + (y * width + x) * 4, 4);
    return value;
}

void TestBlits() {
    Metal::Device device;
    Metal::Encoder encoder{device};
    id<MTLBuffer> src = SharedBuffer(device, 256);
    id<MTLBuffer> dst = SharedBuffer(device, 256);
    for (u32 i = 0; i < 256; ++i) {
        static_cast<u8*>(src.contents)[i] = u8(i);
    }
    encoder.FillBuffer(Ref(dst), 64, 0x7f7f7f7fu);
    encoder.CopyBuffer(Ref(src, 16), Ref(dst, 128), 32);
    // Buffer -> texture -> texture -> buffer round trip, 4x4 RGBA8.
    id<MTLTexture> a = Texture(device, 4, 4);
    id<MTLTexture> b = Texture(device, 4, 4);
    id<MTLBuffer> pixels = SharedBuffer(device, 64);
    for (u32 i = 0; i < 64; ++i) {
        static_cast<u8*>(pixels.contents)[i] = u8(200 - i);
    }
    id<MTLBuffer> back = SharedBuffer(device, 64);
    encoder.CopyBufferToTexture(Ref(pixels), 16, 64, (__bridge void*)a, 0, 0, {}, {4, 4, 1});
    encoder.CopyTexture((__bridge void*)a, 0, 0, {}, (__bridge void*)b, 0, 0, {}, {4, 4, 1});
    encoder.CopyTextureToBuffer((__bridge void*)b, 0, 0, {}, {4, 4, 1}, Ref(back), 16, 64);
    const u64 tick = encoder.Submit();
    device.Wait(tick);

    const u8* d = static_cast<const u8*>(dst.contents);
    bool filled = true, copied = true;
    for (u32 i = 0; i < 64; ++i) {
        filled &= d[i] == 0x7f;
    }
    for (u32 i = 0; i < 32; ++i) {
        copied &= d[128 + i] == u8(16 + i);
    }
    Expect(filled, "FillBuffer fills the range");
    Expect(copied && d[127] == 0 && d[160] == 0, "CopyBuffer copies exactly the range");
    Expect(std::memcmp(pixels.contents, back.contents, 64) == 0,
           "buffer -> texture -> texture -> buffer keeps the bytes");
    Expect(device.ErrorCount() == 0, "blits ran without errors");
}

void TestKeepAlive() {
    Metal::Device device;
    Metal::Encoder encoder{device};
    @autoreleasepool {
        encoder.KeepAlive(Metal::Object::Adopt((__bridge_retained void*)[Probe new]));
    }
    Expect(probes_alive == 1, "KeepAlive holds the object while recording");
    const u64 empty = encoder.Submit(); // nothing recorded: still a tick that completes
    device.Wait(empty);
    for (int i = 0; i < 100 && probes_alive != 0; ++i) {
        [NSThread sleepForTimeInterval:0.01]; // the completion handler may run just after the event
    }
    Expect(probes_alive == 0, "KeepAlive releases the object once the GPU is done");
    Expect(encoder.Submit() == empty + 1, "every Submit takes the next tick");
}
```

- [ ] **Step 2: Run it to see it fail**

Run the build and test command. Expected: `'video_core/renderer_metal/mtl_encoder.h' file not found`.

- [ ] **Step 3: Write the full header**

Create `gpu/shadps4/video_core/renderer_metal/mtl_encoder.h`:

```cpp
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: Metal::Encoder records the renderer's commands into Metal command buffers (Metal-native
// renderer, spec component 2). It is the state machine of mtl_frame.mm (metal-backend) with
// native arguments: a render pass opens its Metal encoder at the first draw (a pass with neither
// draws nor clears costs nothing), blits and dispatches inside a pass end the render encoder and
// the pass reopens with Load actions, and state reaches the Metal encoder only when it changed.
// One thread uses an Encoder at a time.

#pragma once

#include <array>
#include <memory>
#include "common/types.h"
#include "video_core/renderer_metal/mtl_object.h"

namespace Metal {

class Device;

enum class Stage : u8 { Vertex, Fragment, Compute };
enum class Primitive : u8 { Point, Line, LineStrip, Triangle, TriangleStrip };
enum class IndexType : u8 { U16, U32 };
enum class CullMode : u8 { None, Front, Back };
enum class Winding : u8 { Clockwise, CounterClockwise };

struct Origin3 {
    u32 x{}, y{}, z{};
};
struct Size3 {
    u32 width{1}, height{1}, depth{1};
};

struct ColorAttachment {
    void* texture{}; ///< id<MTLTexture>; null: slot unused
    u32 level{}, slice{};
    bool clear{};
    std::array<double, 4> clear_color{};
};

struct DepthAttachment {
    void* texture{}; ///< id<MTLTexture>; null: no depth
    u32 level{}, slice{};
    bool has_stencil{}; ///< the format has a stencil aspect (bound as the stencil attachment too)
    bool clear_depth{}, clear_stencil{};
    double depth{1.0};
    u32 stencil{};
};

struct RenderPassDesc {
    static constexpr u32 MaxColors = 8;
    std::array<ColorAttachment, MaxColors> colors{};
    DepthAttachment depth{};
    u32 width{}, height{}; ///< render area, used when there are no attachments
};

struct Viewport {
    float x{}, y{}, width{}, height{}, min_depth{}, max_depth{1.0f};
};
struct Scissor {
    u32 x{}, y{}, width{}, height{};
};

class Encoder {
public:
    explicit Encoder(Device& device);
    ~Encoder();

    Encoder(const Encoder&) = delete;
    Encoder& operator=(const Encoder&) = delete;

    // Render passes. A pass must be ended before Submit.
    void BeginRenderPass(const RenderPassDesc& desc);
    void EndRenderPass();
    [[nodiscard]] bool InRenderPass() const noexcept;

    // Render state: kept across passes, applied at the next draw.
    void SetRenderPipeline(void* pipeline); ///< id<MTLRenderPipelineState>
    void SetDepthStencil(void* state);      ///< id<MTLDepthStencilState>
    void SetViewport(const Viewport& viewport);
    void SetScissor(const Scissor& scissor);
    void SetCullMode(CullMode mode);
    void SetFrontFace(Winding winding);
    void SetDepthBias(float constant, float slope, float clamp);
    void SetStencilReference(u32 front, u32 back);
    void SetBlendColor(float r, float g, float b, float a);

    // Resources by stage and slot: render stages apply at the next draw, compute at the next
    // dispatch. Slots: buffers 0-30, textures 0-31, samplers 0-15.
    void SetBuffer(Stage stage, u32 slot, BufferRef buffer);
    /// Copies `size` bytes now. Up to 4 KiB they go inline (setBytes); more go through a
    /// temporary shared buffer kept until the command buffer completes.
    void SetBytes(Stage stage, u32 slot, const void* data, u32 size);
    void SetTexture(Stage stage, u32 slot, void* texture); ///< id<MTLTexture>
    void SetSampler(Stage stage, u32 slot, void* sampler); ///< id<MTLSamplerState>

    void Draw(Primitive primitive, u32 vertex_count, u32 instance_count = 1, u32 first_vertex = 0,
              u32 first_instance = 0);
    void DrawIndexed(Primitive primitive, u32 index_count, IndexType type, BufferRef indices,
                     u32 instance_count = 1, s32 base_vertex = 0, u32 first_instance = 0);

    // Compute.
    void SetComputePipeline(void* pipeline); ///< id<MTLComputePipelineState>
    void Dispatch(Size3 groups, Size3 threads_per_group);

    // Blits. Inside a render pass they end the render encoder; the pass continues afterwards.
    void CopyBuffer(BufferRef src, BufferRef dst, u64 size);
    /// Fills with the low byte of `pattern` (as mtl_frame did; logged once if its bytes differ).
    void FillBuffer(BufferRef dst, u64 size, u32 pattern);
    void CopyBufferToTexture(BufferRef src, u32 bytes_per_row, u32 bytes_per_image, void* texture,
                             u32 level, u32 slice, Origin3 origin, Size3 size);
    void CopyTextureToBuffer(void* texture, u32 level, u32 slice, Origin3 origin, Size3 size,
                             BufferRef dst, u32 bytes_per_row, u32 bytes_per_image);
    void CopyTexture(void* src, u32 src_level, u32 src_slice, Origin3 src_origin, void* dst,
                     u32 dst_level, u32 dst_slice, Origin3 dst_origin, Size3 size);

    /// Keeps `object` alive until the current command buffer completes.
    void KeepAlive(Object object);
    /// Ends the open render, compute or blit encoder (a pass in progress reopens with Load).
    void EndEncoding();
    /// The command buffer being recorded (id<MTLCommandBuffer>, created if needed), for work
    /// encoded outside the Encoder (MetalFX). Call EndEncoding() first.
    void* CommandBuffer();
    /// Ends encoding, commits with the device's next tick and returns it. Starts a new command
    /// buffer on the next command. Asserts that no render pass is in progress.
    u64 Submit();

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace Metal
```

- [ ] **Step 4: Implement blits, keep-alive and submit**

Create `gpu/shadps4/video_core/renderer_metal/mtl_encoder.mm` with the skeleton below. The
compute and render functions are filled in by Tasks 4 and 5; for now they assert, so nothing
calls them by accident.

```objc
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: Metal::Encoder (Metal-native renderer, spec component 2).

#import <Metal/Metal.h>

#include <array>
#include <atomic>
#include <bit>
#include <cstring>
#include <vector>
#include "common/assert.h"
#include "common/logging/log.h"
#include "video_core/renderer_metal/mtl_device.h"
#include "video_core/renderer_metal/mtl_encoder.h"

namespace Metal {

namespace {

constexpr u32 MaxBuffers = 31, MaxTextures = 32, MaxSamplers = 16;
constexpr u32 MaxInlineBytes = 4096;

/// One stage's resources: what the caller set, and which slots changed since the Metal encoder
/// last saw them (bit per slot). A new Metal encoder starts empty, so it gets every used slot.
struct StageState {
    struct Slot {
        void* buffer{};
        u64 offset{};
        std::vector<u8> bytes; ///< non-empty: inline bytes instead of the buffer
    };
    std::array<Slot, MaxBuffers> buffers{};
    std::array<void*, MaxTextures> textures{};
    std::array<void*, MaxSamplers> samplers{};
    u32 buffers_dirty{}, buffers_used{};
    u32 textures_dirty{}, textures_used{};
    u32 samplers_dirty{}, samplers_used{};

    void Reapply() {
        buffers_dirty = buffers_used;
        textures_dirty = textures_used;
        samplers_dirty = samplers_used;
    }
};

/// Calls `f(slot)` for every set bit of `mask`.
template <typename F>
void ForEachBit(u32 mask, F&& f) {
    while (mask) {
        const u32 slot = u32(std::countr_zero(mask));
        f(slot);
        mask &= mask - 1;
    }
}

} // namespace

struct Encoder::Impl {
    explicit Impl(Device& device_)
        : device{device_}, dev{(__bridge id<MTLDevice>)device_.NativeDevice()},
          queue{(__bridge id<MTLCommandQueue>)device_.NativeQueue()} {}

    Device& device;
    id<MTLDevice> dev;
    id<MTLCommandQueue> queue;
    id<MTLCommandBuffer> cmd;
    id<MTLRenderCommandEncoder> render;
    id<MTLComputeCommandEncoder> compute;
    id<MTLBlitCommandEncoder> blit;
    std::vector<Object> keep;

    // Render pass (Task 5).
    bool in_pass{}, pass_opened{}, pass_clears{};
    MTLRenderPassDescriptor* pass_first; ///< as begun, with its clears
    MTLRenderPassDescriptor* pass_load;  ///< the same with Load actions (reopening)

    // Render state (Task 5): what the caller set, and what changed since the encoder saw it.
    enum RenderBit : u32 {
        BitPipeline = 1u << 0,
        BitDepthStencil = 1u << 1,
        BitViewport = 1u << 2,
        BitScissor = 1u << 3,
        BitCull = 1u << 4,
        BitWinding = 1u << 5,
        BitBias = 1u << 6,
        BitStencilRef = 1u << 7,
        BitBlend = 1u << 8,
    };
    u32 render_dirty{}, render_used{};
    void* render_pipeline{};
    void* depth_stencil{};
    Viewport viewport{};
    Scissor scissor{};
    CullMode cull{CullMode::None};
    Winding winding{Winding::CounterClockwise};
    float bias[3]{}; ///< constant, slope, clamp
    u32 stencil_front{}, stencil_back{};
    float blend[4]{};
    StageState vertex, fragment;

    // Compute state (Task 4).
    void* compute_pipeline{};
    bool compute_pipeline_dirty{};
    StageState compute_stage;

    std::atomic<bool> fill_pattern_logged{};

    id<MTLCommandBuffer> Cmd() {
        if (!cmd) {
            @autoreleasepool {
                cmd = [queue commandBufferWithUnretainedReferences];
            }
        }
        return cmd;
    }

    void EndRender() {
        if (render) {
            [render endEncoding];
            render = nil;
        }
    }
    void EndCompute() {
        if (compute) {
            [compute endEncoding];
            compute = nil;
        }
    }
    void EndBlit() {
        if (blit) {
            [blit endEncoding];
            blit = nil;
        }
    }

    id<MTLBlitCommandEncoder> Blit() {
        if (!blit) {
            EndRender();
            EndCompute();
            @autoreleasepool {
                blit = [Cmd() blitCommandEncoder];
            }
        }
        return blit;
    }

    StageState& StateOf(Stage stage) {
        switch (stage) {
        case Stage::Vertex:
            return vertex;
        case Stage::Fragment:
            return fragment;
        case Stage::Compute:
            return compute_stage;
        }
        UNREACHABLE();
    }
};

Encoder::Encoder(Device& device) : impl{std::make_unique<Impl>(device)} {}

Encoder::~Encoder() {
    if (impl->cmd) {
        if (impl->in_pass) {
            EndRenderPass();
        }
        Submit(); // nothing recorded is lost, and the keep-alive list is released after the GPU
    }
}

// ---- Blits ----

void Encoder::CopyBuffer(BufferRef src, BufferRef dst, u64 size) {
    [impl->Blit() copyFromBuffer:(__bridge id<MTLBuffer>)src.buffer
                    sourceOffset:src.offset
                        toBuffer:(__bridge id<MTLBuffer>)dst.buffer
               destinationOffset:dst.offset
                            size:size];
}

void Encoder::FillBuffer(BufferRef dst, u64 size, u32 pattern) {
    const u8 byte = u8(pattern);
    if (pattern != u32(byte) * 0x01010101u && !impl->fill_pattern_logged.exchange(true)) {
        LOG_WARNING(Render, "FillBuffer pattern {:#x} has different bytes; filled with its low byte",
                    pattern);
    }
    [impl->Blit() fillBuffer:(__bridge id<MTLBuffer>)dst.buffer
                       range:NSMakeRange(dst.offset, size)
                       value:byte];
}

void Encoder::CopyBufferToTexture(BufferRef src, u32 bytes_per_row, u32 bytes_per_image,
                                  void* texture, u32 level, u32 slice, Origin3 origin, Size3 size) {
    [impl->Blit() copyFromBuffer:(__bridge id<MTLBuffer>)src.buffer
                    sourceOffset:src.offset
               sourceBytesPerRow:bytes_per_row
             sourceBytesPerImage:bytes_per_image
                      sourceSize:MTLSizeMake(size.width, size.height, size.depth)
                       toTexture:(__bridge id<MTLTexture>)texture
                destinationSlice:slice
                destinationLevel:level
               destinationOrigin:MTLOriginMake(origin.x, origin.y, origin.z)];
}

void Encoder::CopyTextureToBuffer(void* texture, u32 level, u32 slice, Origin3 origin, Size3 size,
                                  BufferRef dst, u32 bytes_per_row, u32 bytes_per_image) {
    [impl->Blit() copyFromTexture:(__bridge id<MTLTexture>)texture
                      sourceSlice:slice
                      sourceLevel:level
                     sourceOrigin:MTLOriginMake(origin.x, origin.y, origin.z)
                       sourceSize:MTLSizeMake(size.width, size.height, size.depth)
                         toBuffer:(__bridge id<MTLBuffer>)dst.buffer
                destinationOffset:dst.offset
           destinationBytesPerRow:bytes_per_row
         destinationBytesPerImage:bytes_per_image];
}

void Encoder::CopyTexture(void* src, u32 src_level, u32 src_slice, Origin3 src_origin, void* dst,
                          u32 dst_level, u32 dst_slice, Origin3 dst_origin, Size3 size) {
    [impl->Blit() copyFromTexture:(__bridge id<MTLTexture>)src
                      sourceSlice:src_slice
                      sourceLevel:src_level
                     sourceOrigin:MTLOriginMake(src_origin.x, src_origin.y, src_origin.z)
                       sourceSize:MTLSizeMake(size.width, size.height, size.depth)
                        toTexture:(__bridge id<MTLTexture>)dst
                 destinationSlice:dst_slice
                 destinationLevel:dst_level
                destinationOrigin:MTLOriginMake(dst_origin.x, dst_origin.y, dst_origin.z)];
}

// ---- Command buffer ----

void Encoder::KeepAlive(Object object) {
    impl->keep.push_back(std::move(object));
}

void Encoder::EndEncoding() {
    impl->EndRender();
    impl->EndCompute();
    impl->EndBlit();
}

void* Encoder::CommandBuffer() {
    return (__bridge void*)impl->Cmd();
}

u64 Encoder::Submit() {
    ASSERT_MSG(!impl->in_pass, "Submit inside a render pass");
    EndEncoding();
    id<MTLCommandBuffer> cmd = impl->Cmd();
    auto* held = new std::vector<Object>(std::move(impl->keep));
    impl->keep.clear();
    [cmd addCompletedHandler:^(id<MTLCommandBuffer>) {
        delete held;
    }];
    const u64 tick = impl->device.Commit((__bridge void*)cmd);
    impl->cmd = nil;
    return tick;
}

// ---- Compute (Task 4) ----

void Encoder::SetComputePipeline(void*) {
    UNREACHABLE_MSG("Metal::Encoder compute: Task 4");
}
void Encoder::Dispatch(Size3, Size3) {
    UNREACHABLE_MSG("Metal::Encoder compute: Task 4");
}
void Encoder::SetBuffer(Stage, u32, BufferRef) {
    UNREACHABLE_MSG("Metal::Encoder resources: Task 4");
}
void Encoder::SetBytes(Stage, u32, const void*, u32) {
    UNREACHABLE_MSG("Metal::Encoder resources: Task 4");
}
void Encoder::SetTexture(Stage, u32, void*) {
    UNREACHABLE_MSG("Metal::Encoder resources: Task 4");
}
void Encoder::SetSampler(Stage, u32, void*) {
    UNREACHABLE_MSG("Metal::Encoder resources: Task 4");
}

// ---- Render passes and draws (Task 5) ----

void Encoder::BeginRenderPass(const RenderPassDesc&) {
    UNREACHABLE_MSG("Metal::Encoder render passes: Task 5");
}
void Encoder::EndRenderPass() {
    UNREACHABLE_MSG("Metal::Encoder render passes: Task 5");
}
bool Encoder::InRenderPass() const noexcept {
    return impl->in_pass;
}
void Encoder::SetRenderPipeline(void*) {
    UNREACHABLE_MSG("Metal::Encoder render state: Task 5");
}
void Encoder::SetDepthStencil(void*) {
    UNREACHABLE_MSG("Metal::Encoder render state: Task 5");
}
void Encoder::SetViewport(const Viewport&) {
    UNREACHABLE_MSG("Metal::Encoder render state: Task 5");
}
void Encoder::SetScissor(const Scissor&) {
    UNREACHABLE_MSG("Metal::Encoder render state: Task 5");
}
void Encoder::SetCullMode(CullMode) {
    UNREACHABLE_MSG("Metal::Encoder render state: Task 5");
}
void Encoder::SetFrontFace(Winding) {
    UNREACHABLE_MSG("Metal::Encoder render state: Task 5");
}
void Encoder::SetDepthBias(float, float, float) {
    UNREACHABLE_MSG("Metal::Encoder render state: Task 5");
}
void Encoder::SetStencilReference(u32, u32) {
    UNREACHABLE_MSG("Metal::Encoder render state: Task 5");
}
void Encoder::SetBlendColor(float, float, float, float) {
    UNREACHABLE_MSG("Metal::Encoder render state: Task 5");
}
void Encoder::Draw(Primitive, u32, u32, u32, u32) {
    UNREACHABLE_MSG("Metal::Encoder draws: Task 5");
}
void Encoder::DrawIndexed(Primitive, u32, IndexType, BufferRef, u32, s32, u32) {
    UNREACHABLE_MSG("Metal::Encoder draws: Task 5");
}

} // namespace Metal
```

- [ ] **Step 5: Run it to see it pass**

Run the build and test command. Expected: `PASS: …` and `exit 0`.

- [ ] **Step 6: Commit**

```bash
git add gpu/shadps4/video_core/renderer_metal/mtl_encoder.h gpu/shadps4/video_core/renderer_metal/mtl_encoder.mm tests/test_mtl_encoder.mm
git commit -m "Metal-native M1: Metal::Encoder interface; blits, keep-alive and submit with ticks

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Encoder resources and compute

**Files:**
- Modify: `gpu/shadps4/video_core/renderer_metal/mtl_encoder.mm` (replace the six Task 4 stubs)
- Test: `tests/test_mtl_encoder.mm`

**Interfaces:**
- Consumes: Task 3's `Impl` (`StateOf`, `StageState`, `ForEachBit`, `compute_*` fields,
  `EndRender`, `EndBlit`, `Cmd`, `keep`).
- Produces: working `SetBuffer`, `SetBytes`, `SetTexture`, `SetSampler`, `SetComputePipeline`,
  `Dispatch`; `Impl::ApplyCompute()` and `Impl::Compute()` (used only inside this file).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_mtl_encoder.mm` (second anonymous namespace) and call
`TestCompute(); TestLargeBytes();` in `main` after `TestKeepAlive();`:

```objc
constexpr const char* kTestShaders = R"(
#include <metal_stdlib>
using namespace metal;

kernel void doubles(device uint* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    out[i] = i * 2u;
}

kernel void copy_constants(device uint* out [[buffer(0)]], constant uint* in [[buffer(1)]],
                           uint i [[thread_position_in_grid]]) {
    out[i] = in[i];
}

struct VOut { float4 pos [[position]]; };

vertex VOut fullscreen(uint vid [[vertex_id]]) {
    const float2 uv = float2((vid << 1) & 2, vid & 2);
    VOut o;
    o.pos = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    return o;
}

vertex VOut indexed_quad(uint vid [[vertex_id]], constant float2* corners [[buffer(0)]]) {
    VOut o;
    o.pos = float4(corners[vid], 0.0, 1.0);
    return o;
}

fragment float4 solid(constant float4& color [[buffer(0)]]) {
    return color;
}
)";

id<MTLLibrary> TestLibrary(Metal::Device& device) {
    id<MTLDevice> dev = (__bridge id<MTLDevice>)device.NativeDevice();
    NSError* error = nil;
    id<MTLLibrary> library = [dev newLibraryWithSource:@(kTestShaders) options:nil error:&error];
    if (!library) {
        std::printf("FAIL: test shaders: %s\n", error.localizedDescription.UTF8String);
        std::exit(1);
    }
    return library;
}

id<MTLComputePipelineState> Kernel(Metal::Device& device, id<MTLLibrary> library, NSString* name) {
    id<MTLDevice> dev = (__bridge id<MTLDevice>)device.NativeDevice();
    NSError* error = nil;
    id<MTLComputePipelineState> pipeline =
        [dev newComputePipelineStateWithFunction:[library newFunctionWithName:name] error:&error];
    if (!pipeline) {
        std::printf("FAIL: kernel: %s\n", error.localizedDescription.UTF8String);
        std::exit(1);
    }
    return pipeline;
}

void TestCompute() {
    Metal::Device device;
    Metal::Encoder encoder{device};
    id<MTLLibrary> library = TestLibrary(device);
    id<MTLComputePipelineState> doubles = Kernel(device, library, @"doubles");
    id<MTLBuffer> out = SharedBuffer(device, 64 * 4);
    encoder.SetComputePipeline((__bridge void*)doubles);
    encoder.SetBuffer(Metal::Stage::Compute, 0, Ref(out));
    encoder.Dispatch({4, 1, 1}, {16, 1, 1});
    // A blit between two dispatches ends the compute encoder; the second dispatch must get its
    // pipeline and buffer again on the new encoder.
    id<MTLBuffer> scratch = SharedBuffer(device, 16);
    encoder.FillBuffer(Ref(scratch), 16, 0);
    id<MTLBuffer> out2 = SharedBuffer(device, 64 * 4);
    encoder.SetBuffer(Metal::Stage::Compute, 0, Ref(out2));
    encoder.Dispatch({4, 1, 1}, {16, 1, 1});
    device.Wait(encoder.Submit());

    bool ok = true, ok2 = true;
    for (u32 i = 0; i < 64; ++i) {
        ok &= static_cast<const u32*>(out.contents)[i] == i * 2;
        ok2 &= static_cast<const u32*>(out2.contents)[i] == i * 2;
    }
    Expect(ok, "Dispatch runs the kernel over the grid");
    Expect(ok2, "state is reapplied on a compute encoder opened after a blit");
}

void TestLargeBytes() {
    Metal::Device device;
    Metal::Encoder encoder{device};
    id<MTLLibrary> library = TestLibrary(device);
    id<MTLComputePipelineState> copy = Kernel(device, library, @"copy_constants");
    std::vector<u32> small(64), large(2048); // 256 B inline; 8 KiB over the setBytes limit
    for (u32 i = 0; i < small.size(); ++i) {
        small[i] = 1000 + i;
    }
    for (u32 i = 0; i < large.size(); ++i) {
        large[i] = 50000 + i;
    }
    id<MTLBuffer> out_small = SharedBuffer(device, small.size() * 4);
    id<MTLBuffer> out_large = SharedBuffer(device, large.size() * 4);
    encoder.SetComputePipeline((__bridge void*)copy);
    encoder.SetBuffer(Metal::Stage::Compute, 0, Ref(out_small));
    encoder.SetBytes(Metal::Stage::Compute, 1, small.data(), u32(small.size() * 4));
    encoder.Dispatch({1, 1, 1}, {64, 1, 1});
    encoder.SetBuffer(Metal::Stage::Compute, 0, Ref(out_large));
    encoder.SetBytes(Metal::Stage::Compute, 1, large.data(), u32(large.size() * 4));
    std::fill(large.begin(), large.end(), 0u); // the encoder must have copied the bytes already
    encoder.Dispatch({32, 1, 1}, {64, 1, 1});
    device.Wait(encoder.Submit());

    Expect(std::memcmp(out_small.contents, small.data(), small.size() * 4) == 0,
           "SetBytes up to 4 KiB arrives");
    bool ok = true;
    for (u32 i = 0; i < 2048; ++i) {
        ok &= static_cast<const u32*>(out_large.contents)[i] == 50000 + i;
    }
    Expect(ok, "SetBytes over 4 KiB arrives (temporary buffer)");
    Expect(device.ErrorCount() == 0, "compute ran without errors");
}
```

Also add `#include <cstdlib>` and `#include <algorithm>` to the test's includes.

- [ ] **Step 2: Run them to see them fail**

Run the build and test command. Expected: the test aborts in `UNREACHABLE_MSG("Metal::Encoder
compute: Task 4")` (non-zero exit).

- [ ] **Step 3: Implement**

In `mtl_encoder.mm`, add to `struct Encoder::Impl` (after `Blit()`):

```objc
    id<MTLComputeCommandEncoder> Compute() {
        if (!compute) {
            EndRender();
            EndBlit();
            @autoreleasepool {
                compute = [Cmd() computeCommandEncoder];
            }
            compute_stage.Reapply();
            compute_pipeline_dirty = compute_pipeline != nullptr;
        }
        return compute;
    }

    void ApplyCompute() {
        id<MTLComputeCommandEncoder> e = Compute();
        if (compute_pipeline_dirty) {
            [e setComputePipelineState:(__bridge id<MTLComputePipelineState>)compute_pipeline];
            compute_pipeline_dirty = false;
        }
        StageState& s = compute_stage;
        ForEachBit(s.buffers_dirty, [&](u32 slot) {
            const StageState::Slot& b = s.buffers[slot];
            if (!b.bytes.empty()) {
                [e setBytes:b.bytes.data() length:b.bytes.size() atIndex:slot];
            } else {
                [e setBuffer:(__bridge id<MTLBuffer>)b.buffer offset:b.offset atIndex:slot];
            }
        });
        ForEachBit(s.textures_dirty, [&](u32 slot) {
            [e setTexture:(__bridge id<MTLTexture>)s.textures[slot] atIndex:slot];
        });
        ForEachBit(s.samplers_dirty, [&](u32 slot) {
            [e setSamplerState:(__bridge id<MTLSamplerState>)s.samplers[slot] atIndex:slot];
        });
        s.buffers_dirty = s.textures_dirty = s.samplers_dirty = 0;
    }
```

Replace the six Task 4 stubs with:

```objc
void Encoder::SetBuffer(Stage stage, u32 slot, BufferRef buffer) {
    ASSERT(slot < MaxBuffers);
    StageState& s = impl->StateOf(stage);
    StageState::Slot& b = s.buffers[slot];
    b.buffer = buffer.buffer;
    b.offset = buffer.offset;
    b.bytes.clear();
    s.buffers_dirty |= 1u << slot;
    s.buffers_used |= 1u << slot;
}

void Encoder::SetBytes(Stage stage, u32 slot, const void* data, u32 size) {
    ASSERT(slot < MaxBuffers && size > 0);
    if (size > MaxInlineBytes) {
        id<MTLBuffer> temp;
        @autoreleasepool {
            temp = [impl->dev newBufferWithBytes:data
                                          length:size
                                         options:MTLResourceStorageModeShared];
        }
        SetBuffer(stage, slot, {(__bridge void*)temp, 0});
        KeepAlive(Object::Retain((__bridge void*)temp));
        return;
    }
    StageState& s = impl->StateOf(stage);
    StageState::Slot& b = s.buffers[slot];
    b.buffer = nullptr;
    b.offset = 0;
    b.bytes.assign(static_cast<const u8*>(data), static_cast<const u8*>(data) + size);
    s.buffers_dirty |= 1u << slot;
    s.buffers_used |= 1u << slot;
}

void Encoder::SetTexture(Stage stage, u32 slot, void* texture) {
    ASSERT(slot < MaxTextures);
    StageState& s = impl->StateOf(stage);
    s.textures[slot] = texture;
    s.textures_dirty |= 1u << slot;
    s.textures_used |= 1u << slot;
}

void Encoder::SetSampler(Stage stage, u32 slot, void* sampler) {
    ASSERT(slot < MaxSamplers);
    StageState& s = impl->StateOf(stage);
    s.samplers[slot] = sampler;
    s.samplers_dirty |= 1u << slot;
    s.samplers_used |= 1u << slot;
}

void Encoder::SetComputePipeline(void* pipeline) {
    if (pipeline != impl->compute_pipeline) {
        impl->compute_pipeline = pipeline;
        impl->compute_pipeline_dirty = true;
    }
}

void Encoder::Dispatch(Size3 groups, Size3 threads_per_group) {
    ASSERT_MSG(impl->compute_pipeline, "Dispatch without a compute pipeline");
    impl->ApplyCompute();
    [impl->compute dispatchThreadgroups:MTLSizeMake(groups.width, groups.height, groups.depth)
                  threadsPerThreadgroup:MTLSizeMake(threads_per_group.width,
                                                    threads_per_group.height,
                                                    threads_per_group.depth)];
}
```

- [ ] **Step 4: Run them to see them pass**

Run the build and test command. Expected: `PASS: …` and `exit 0`.

- [ ] **Step 5: Commit**

```bash
git add gpu/shadps4/video_core/renderer_metal/mtl_encoder.mm tests/test_mtl_encoder.mm
git commit -m "Metal-native M1: Encoder resources (slots set on change, bytes over 4 KiB via a temporary buffer) and compute dispatches

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Encoder render passes and draws

**Files:**
- Modify: `gpu/shadps4/video_core/renderer_metal/mtl_encoder.mm` (replace the Task 5 stubs)
- Test: `tests/test_mtl_encoder.mm`

**Interfaces:**
- Consumes: Task 3/4 `Impl` members (`pass_*`, render state fields, `vertex`, `fragment`,
  `ForEachBit`, `EndCompute`, `EndBlit`, `Cmd`).
- Produces: working `BeginRenderPass`, `EndRenderPass`, all render-state setters, `Draw`,
  `DrawIndexed`; `Impl::Render()` and `Impl::ApplyRender()` (file-local).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_mtl_encoder.mm` (second anonymous namespace) and call
`TestClearOnlyPass(); TestPassReopenKeepsDrawing(); TestStatePersistsAcrossPasses(); TestIndexedDraw();`
in `main` after `TestLargeBytes();`:

```objc
constexpr u32 kSize = 64;
constexpr u32 kRed = 0xff0000ffu, kGreen = 0xff00ff00u, kBlue = 0xffff0000u, kBlack = 0xff000000u,
              kYellow = 0xff00ffffu; // RGBA8 bytes read as a little-endian u32

id<MTLRenderPipelineState> SolidPipeline(Metal::Device& device, id<MTLLibrary> library,
                                         NSString* vertex) {
    id<MTLDevice> dev = (__bridge id<MTLDevice>)device.NativeDevice();
    MTLRenderPipelineDescriptor* desc = [MTLRenderPipelineDescriptor new];
    desc.vertexFunction = [library newFunctionWithName:vertex];
    desc.fragmentFunction = [library newFunctionWithName:@"solid"];
    desc.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
    NSError* error = nil;
    id<MTLRenderPipelineState> pipeline = [dev newRenderPipelineStateWithDescriptor:desc
                                                                              error:&error];
    if (!pipeline) {
        std::printf("FAIL: render pipeline: %s\n", error.localizedDescription.UTF8String);
        std::exit(1);
    }
    return pipeline;
}

Metal::RenderPassDesc ColorPass(id<MTLTexture> target, bool clear, u32 rgba) {
    Metal::RenderPassDesc desc;
    desc.colors[0].texture = (__bridge void*)target;
    desc.colors[0].clear = clear;
    desc.colors[0].clear_color = {double(rgba & 0xff) / 255.0, double((rgba >> 8) & 0xff) / 255.0,
                                  double((rgba >> 16) & 0xff) / 255.0, 1.0};
    desc.width = kSize;
    desc.height = kSize;
    return desc;
}

/// Records a copy of `texture` into a new shared buffer (kSize x kSize RGBA8).
id<MTLBuffer> Readback(Metal::Device& device, Metal::Encoder& encoder, id<MTLTexture> texture) {
    id<MTLBuffer> out = SharedBuffer(device, kSize * kSize * 4);
    encoder.CopyTextureToBuffer((__bridge void*)texture, 0, 0, {}, {kSize, kSize, 1}, Ref(out),
                                kSize * 4, kSize * kSize * 4);
    return out;
}

void FillTexture(Metal::Device& device, Metal::Encoder& encoder, id<MTLTexture> texture, u32 rgba) {
    id<MTLBuffer> pixels = SharedBuffer(device, kSize * kSize * 4);
    for (u32 i = 0; i < kSize * kSize; ++i) {
        static_cast<u32*>(pixels.contents)[i] = rgba;
    }
    encoder.CopyBufferToTexture(Ref(pixels), kSize * 4, kSize * kSize * 4, (__bridge void*)texture,
                                0, 0, {}, {kSize, kSize, 1});
    encoder.KeepAlive(Metal::Object::Retain((__bridge void*)pixels));
}

void SetColor(Metal::Encoder& encoder, u32 rgba) {
    const float color[4] = {float(rgba & 0xff) / 255.0f, float((rgba >> 8) & 0xff) / 255.0f,
                            float((rgba >> 16) & 0xff) / 255.0f, 1.0f};
    encoder.SetBytes(Metal::Stage::Fragment, 0, color, sizeof(color));
}

void TestClearOnlyPass() {
    Metal::Device device;
    Metal::Encoder encoder{device};
    id<MTLTexture> cleared = Texture(device, kSize, kSize);
    id<MTLTexture> untouched = Texture(device, kSize, kSize);
    FillTexture(device, encoder, cleared, kBlue);
    FillTexture(device, encoder, untouched, kBlue);
    encoder.BeginRenderPass(ColorPass(cleared, true, kYellow));
    encoder.EndRenderPass();
    encoder.BeginRenderPass(ColorPass(untouched, false, 0));
    encoder.EndRenderPass();
    id<MTLBuffer> a = Readback(device, encoder, cleared);
    id<MTLBuffer> b = Readback(device, encoder, untouched);
    device.Wait(encoder.Submit());
    Expect(Pixel(a, kSize, 5, 5) == kYellow && Pixel(a, kSize, 63, 63) == kYellow,
           "a pass that only clears still clears");
    Expect(Pixel(b, kSize, 5, 5) == kBlue, "a pass with no draws and no clears leaves the target");
}

void TestPassReopenKeepsDrawing() {
    Metal::Device device;
    Metal::Encoder encoder{device};
    id<MTLLibrary> library = TestLibrary(device);
    id<MTLRenderPipelineState> solid = SolidPipeline(device, library, @"fullscreen");
    id<MTLTexture> target = Texture(device, kSize, kSize);
    id<MTLBuffer> scratch = SharedBuffer(device, 16);

    encoder.BeginRenderPass(ColorPass(target, true, kBlack));
    encoder.SetRenderPipeline((__bridge void*)solid);
    encoder.SetViewport({0, 0, float(kSize), float(kSize), 0, 1});
    encoder.SetScissor({0, 0, kSize / 2, kSize});
    SetColor(encoder, kRed);
    encoder.Draw(Metal::Primitive::Triangle, 3);
    encoder.FillBuffer(Ref(scratch), 16, 0); // interrupts the pass
    encoder.SetScissor({kSize / 2, 0, kSize / 2, kSize});
    SetColor(encoder, kGreen);
    encoder.Draw(Metal::Primitive::Triangle, 3); // reopens the pass
    encoder.EndRenderPass();
    id<MTLBuffer> out = Readback(device, encoder, target);
    device.Wait(encoder.Submit());
    Expect(Pixel(out, kSize, 1, 1) == kRed, "the draw before the interruption is kept (Load)");
    Expect(Pixel(out, kSize, kSize - 2, 1) == kGreen, "the draw after the interruption lands");
    Expect(device.ErrorCount() == 0, "no errors reopening a pass");
}

void TestStatePersistsAcrossPasses() {
    Metal::Device device;
    Metal::Encoder encoder{device};
    id<MTLLibrary> library = TestLibrary(device);
    id<MTLRenderPipelineState> solid = SolidPipeline(device, library, @"fullscreen");
    id<MTLTexture> first = Texture(device, kSize, kSize);
    id<MTLTexture> second = Texture(device, kSize, kSize);

    encoder.BeginRenderPass(ColorPass(first, true, kBlack));
    encoder.SetRenderPipeline((__bridge void*)solid);
    encoder.SetViewport({0, 0, float(kSize), float(kSize), 0, 1});
    encoder.SetScissor({0, 0, kSize, kSize / 2}); // top half only
    SetColor(encoder, kGreen);
    encoder.Draw(Metal::Primitive::Triangle, 3);
    encoder.EndRenderPass();
    // Second pass: nothing set again; pipeline, viewport, scissor and the color must carry over.
    encoder.BeginRenderPass(ColorPass(second, true, kBlack));
    encoder.Draw(Metal::Primitive::Triangle, 3);
    encoder.EndRenderPass();
    id<MTLBuffer> out = Readback(device, encoder, second);
    device.Wait(encoder.Submit());
    Expect(Pixel(out, kSize, 3, 3) == kGreen, "state set in one pass applies in the next");
    Expect(Pixel(out, kSize, 3, kSize - 3) == kBlack, "including the scissor");
}

void TestIndexedDraw() {
    Metal::Device device;
    Metal::Encoder encoder{device};
    id<MTLLibrary> library = TestLibrary(device);
    id<MTLRenderPipelineState> quad = SolidPipeline(device, library, @"indexed_quad");
    id<MTLTexture> target = Texture(device, kSize, kSize);
    // Left half of clip space as two triangles from four corners.
    const float corners[8] = {-1, -1, 0, -1, -1, 1, 0, 1};
    id<MTLBuffer> indices = SharedBuffer(device, 12);
    const u16 index_data[6] = {0, 1, 2, 2, 1, 3};
    std::memcpy(indices.contents, index_data, sizeof(index_data));

    encoder.BeginRenderPass(ColorPass(target, true, kBlack));
    encoder.SetRenderPipeline((__bridge void*)quad);
    encoder.SetViewport({0, 0, float(kSize), float(kSize), 0, 1});
    encoder.SetScissor({0, 0, kSize, kSize});
    encoder.SetBytes(Metal::Stage::Vertex, 0, corners, sizeof(corners));
    SetColor(encoder, kBlue);
    encoder.DrawIndexed(Metal::Primitive::Triangle, 6, Metal::IndexType::U16, Ref(indices));
    encoder.EndRenderPass();
    id<MTLBuffer> out = Readback(device, encoder, target);
    device.Wait(encoder.Submit());
    Expect(Pixel(out, kSize, 2, kSize / 2) == kBlue, "DrawIndexed draws the left half");
    Expect(Pixel(out, kSize, kSize - 2, kSize / 2) == kBlack, "and not the right half");
}
```

- [ ] **Step 2: Run them to see them fail**

Run the build and test command. Expected: abort in
`UNREACHABLE_MSG("Metal::Encoder render passes: Task 5")`.

- [ ] **Step 3: Implement**

In `mtl_encoder.mm`, add to the anonymous namespace (after `ForEachBit`):

```objc
MTLPrimitiveType ToMtl(Primitive primitive) {
    switch (primitive) {
    case Primitive::Point:
        return MTLPrimitiveTypePoint;
    case Primitive::Line:
        return MTLPrimitiveTypeLine;
    case Primitive::LineStrip:
        return MTLPrimitiveTypeLineStrip;
    case Primitive::Triangle:
        return MTLPrimitiveTypeTriangle;
    case Primitive::TriangleStrip:
        return MTLPrimitiveTypeTriangleStrip;
    }
    UNREACHABLE();
}

MTLCullMode ToMtl(CullMode mode) {
    switch (mode) {
    case CullMode::None:
        return MTLCullModeNone;
    case CullMode::Front:
        return MTLCullModeFront;
    case CullMode::Back:
        return MTLCullModeBack;
    }
    UNREACHABLE();
}
```

Add to `struct Encoder::Impl` (after `ApplyCompute()`):

```objc
    id<MTLRenderCommandEncoder> Render() {
        if (!render) {
            ASSERT_MSG(in_pass, "Draw outside a render pass");
            EndCompute();
            EndBlit();
            @autoreleasepool {
                render = [Cmd() renderCommandEncoderWithDescriptor:pass_opened ? pass_load
                                                                               : pass_first];
            }
            pass_opened = true;
            render_dirty = render_used;
            vertex.Reapply();
            fragment.Reapply();
        }
        return render;
    }

    void ApplyStage(id<MTLRenderCommandEncoder> e, StageState& s, bool is_vertex) {
        ForEachBit(s.buffers_dirty, [&](u32 slot) {
            const StageState::Slot& b = s.buffers[slot];
            if (!b.bytes.empty()) {
                if (is_vertex) {
                    [e setVertexBytes:b.bytes.data() length:b.bytes.size() atIndex:slot];
                } else {
                    [e setFragmentBytes:b.bytes.data() length:b.bytes.size() atIndex:slot];
                }
            } else if (is_vertex) {
                [e setVertexBuffer:(__bridge id<MTLBuffer>)b.buffer offset:b.offset atIndex:slot];
            } else {
                [e setFragmentBuffer:(__bridge id<MTLBuffer>)b.buffer offset:b.offset atIndex:slot];
            }
        });
        ForEachBit(s.textures_dirty, [&](u32 slot) {
            id<MTLTexture> t = (__bridge id<MTLTexture>)s.textures[slot];
            if (is_vertex) {
                [e setVertexTexture:t atIndex:slot];
            } else {
                [e setFragmentTexture:t atIndex:slot];
            }
        });
        ForEachBit(s.samplers_dirty, [&](u32 slot) {
            id<MTLSamplerState> st = (__bridge id<MTLSamplerState>)s.samplers[slot];
            if (is_vertex) {
                [e setVertexSamplerState:st atIndex:slot];
            } else {
                [e setFragmentSamplerState:st atIndex:slot];
            }
        });
        s.buffers_dirty = s.textures_dirty = s.samplers_dirty = 0;
    }

    void ApplyRender() {
        id<MTLRenderCommandEncoder> e = Render();
        const u32 d = render_dirty;
        if (d & BitPipeline) {
            [e setRenderPipelineState:(__bridge id<MTLRenderPipelineState>)render_pipeline];
        }
        if (d & BitDepthStencil) {
            [e setDepthStencilState:(__bridge id<MTLDepthStencilState>)depth_stencil];
        }
        if (d & BitViewport) {
            [e setViewport:(MTLViewport){viewport.x, viewport.y, viewport.width, viewport.height,
                                         viewport.min_depth, viewport.max_depth}];
        }
        if (d & BitScissor) {
            [e setScissorRect:(MTLScissorRect){scissor.x, scissor.y, scissor.width,
                                               scissor.height}];
        }
        if (d & BitCull) {
            [e setCullMode:ToMtl(cull)];
        }
        if (d & BitWinding) {
            [e setFrontFacingWinding:winding == Winding::Clockwise ? MTLWindingClockwise
                                                                    : MTLWindingCounterClockwise];
        }
        if (d & BitBias) {
            [e setDepthBias:bias[0] slopeScale:bias[1] clamp:bias[2]];
        }
        if (d & BitStencilRef) {
            [e setStencilFrontReferenceValue:stencil_front backReferenceValue:stencil_back];
        }
        if (d & BitBlend) {
            [e setBlendColorRed:blend[0] green:blend[1] blue:blend[2] alpha:blend[3]];
        }
        render_dirty = 0;
        ApplyStage(e, vertex, true);
        ApplyStage(e, fragment, false);
    }

    void MarkRender(u32 bit) {
        render_dirty |= bit;
        render_used |= bit;
    }
```

Replace the Task 5 stubs (keep `InRenderPass`) with:

```objc
void Encoder::BeginRenderPass(const RenderPassDesc& desc) {
    ASSERT_MSG(!impl->in_pass, "BeginRenderPass inside a render pass");
    EndEncoding();
    MTLRenderPassDescriptor* first = [MTLRenderPassDescriptor renderPassDescriptor];
    bool clears = false, attachments = false;
    for (u32 i = 0; i < RenderPassDesc::MaxColors; ++i) {
        const ColorAttachment& c = desc.colors[i];
        if (!c.texture) {
            continue;
        }
        attachments = true;
        MTLRenderPassColorAttachmentDescriptor* a = first.colorAttachments[i];
        a.texture = (__bridge id<MTLTexture>)c.texture;
        a.level = c.level;
        a.slice = c.slice;
        a.loadAction = c.clear ? MTLLoadActionClear : MTLLoadActionLoad;
        a.storeAction = MTLStoreActionStore;
        a.clearColor = MTLClearColorMake(c.clear_color[0], c.clear_color[1], c.clear_color[2],
                                         c.clear_color[3]);
        clears |= c.clear;
    }
    const DepthAttachment& d = desc.depth;
    if (d.texture) {
        attachments = true;
        MTLRenderPassDepthAttachmentDescriptor* a = first.depthAttachment;
        a.texture = (__bridge id<MTLTexture>)d.texture;
        a.level = d.level;
        a.slice = d.slice;
        a.loadAction = d.clear_depth ? MTLLoadActionClear : MTLLoadActionLoad;
        a.storeAction = MTLStoreActionStore;
        a.clearDepth = d.depth;
        clears |= d.clear_depth;
        if (d.has_stencil) {
            MTLRenderPassStencilAttachmentDescriptor* s = first.stencilAttachment;
            s.texture = a.texture;
            s.level = d.level;
            s.slice = d.slice;
            s.loadAction = d.clear_stencil ? MTLLoadActionClear : MTLLoadActionLoad;
            s.storeAction = MTLStoreActionStore;
            s.clearStencil = d.stencil;
            clears |= d.clear_stencil;
        }
    }
    if (!attachments) {
        first.renderTargetWidth = desc.width;
        first.renderTargetHeight = desc.height;
        first.defaultRasterSampleCount = 1;
    }
    MTLRenderPassDescriptor* load = [first copy];
    for (u32 i = 0; i < RenderPassDesc::MaxColors; ++i) {
        load.colorAttachments[i].loadAction = MTLLoadActionLoad;
    }
    load.depthAttachment.loadAction = MTLLoadActionLoad;
    load.stencilAttachment.loadAction = MTLLoadActionLoad;
    impl->pass_first = first;
    impl->pass_load = load;
    impl->pass_clears = clears;
    impl->pass_opened = false;
    impl->in_pass = true;
}

void Encoder::EndRenderPass() {
    ASSERT_MSG(impl->in_pass, "EndRenderPass outside a render pass");
    if (!impl->pass_opened && impl->pass_clears) {
        impl->Render(); // a pass that only clears still has to clear
    }
    impl->EndRender();
    impl->in_pass = false;
    impl->pass_first = nil;
    impl->pass_load = nil;
}

void Encoder::SetRenderPipeline(void* pipeline) {
    if (pipeline != impl->render_pipeline) {
        impl->render_pipeline = pipeline;
        impl->MarkRender(Impl::BitPipeline);
    }
}

void Encoder::SetDepthStencil(void* state) {
    if (state != impl->depth_stencil) {
        impl->depth_stencil = state;
        impl->MarkRender(Impl::BitDepthStencil);
    }
}

void Encoder::SetViewport(const Viewport& viewport) {
    impl->viewport = viewport;
    impl->MarkRender(Impl::BitViewport);
}

void Encoder::SetScissor(const Scissor& scissor) {
    impl->scissor = scissor;
    impl->MarkRender(Impl::BitScissor);
}

void Encoder::SetCullMode(CullMode mode) {
    impl->cull = mode;
    impl->MarkRender(Impl::BitCull);
}

void Encoder::SetFrontFace(Winding winding) {
    impl->winding = winding;
    impl->MarkRender(Impl::BitWinding);
}

void Encoder::SetDepthBias(float constant, float slope, float clamp) {
    impl->bias[0] = constant;
    impl->bias[1] = slope;
    impl->bias[2] = clamp;
    impl->MarkRender(Impl::BitBias);
}

void Encoder::SetStencilReference(u32 front, u32 back) {
    impl->stencil_front = front;
    impl->stencil_back = back;
    impl->MarkRender(Impl::BitStencilRef);
}

void Encoder::SetBlendColor(float r, float g, float b, float a) {
    impl->blend[0] = r;
    impl->blend[1] = g;
    impl->blend[2] = b;
    impl->blend[3] = a;
    impl->MarkRender(Impl::BitBlend);
}

void Encoder::Draw(Primitive primitive, u32 vertex_count, u32 instance_count, u32 first_vertex,
                   u32 first_instance) {
    ASSERT_MSG(impl->render_pipeline, "Draw without a render pipeline");
    impl->ApplyRender();
    [impl->render drawPrimitives:ToMtl(primitive)
                     vertexStart:first_vertex
                     vertexCount:vertex_count
                   instanceCount:instance_count
                    baseInstance:first_instance];
}

void Encoder::DrawIndexed(Primitive primitive, u32 index_count, IndexType type, BufferRef indices,
                          u32 instance_count, s32 base_vertex, u32 first_instance) {
    ASSERT_MSG(impl->render_pipeline, "Draw without a render pipeline");
    impl->ApplyRender();
    [impl->render drawIndexedPrimitives:ToMtl(primitive)
                             indexCount:index_count
                              indexType:type == IndexType::U32 ? MTLIndexTypeUInt32
                                                               : MTLIndexTypeUInt16
                            indexBuffer:(__bridge id<MTLBuffer>)indices.buffer
                      indexBufferOffset:indices.offset
                          instanceCount:instance_count
                             baseVertex:base_vertex
                           baseInstance:first_instance];
}
```

- [ ] **Step 4: Run them to see them pass**

Run the build and test command. Expected: `PASS: …` and `exit 0`. If
`TestPassReopenKeepsDrawing` fails with the left half black, the reopen used `pass_first`
(Clear) instead of `pass_load`: check `Render()`.

- [ ] **Step 5: Commit**

```bash
git add gpu/shadps4/video_core/renderer_metal/mtl_encoder.mm tests/test_mtl_encoder.mm
git commit -m "Metal-native M1: Encoder render passes (opened at the first draw, clear-only passes, reopen with Load) and draws; state persists across passes

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Test suite wiring and milestone check

**Files:**
- Modify: `build_macos.sh` (the `--test` section, after the "Metal back end M0" block, ~line 125)
- Modify: `docs/superpowers/plans/2026-10-16-metal-native-roadmap.md` (M1 status)

**Interfaces:**
- Consumes: everything above.
- Produces: `out/mtl-encoder-test` built and run by `bash build_macos.sh --test`.

- [ ] **Step 1: Add the test to the suite**

In `build_macos.sh`, after the `fi` that closes the `metal-interop-test` block, add:

```bash
    # Metal-native M1: Metal::Object, Device ticks and limits, Metal::Encoder (arm64, no window).
    if [[ -f out/gpu-arm64/libbbgpu.dylib ]]; then
        /usr/bin/clang++ -arch arm64 -mmacosx-version-min=$MACOSX_DEPLOYMENT_TARGET -std=c++20 -fobjc-arc -O2 -g \
            -Wall -Wextra -Werror -Igpu/shadps4 tests/test_mtl_encoder.mm -Lout/gpu-arm64 -lbbgpu \
            -Wl,-rpath,"$ROOT/out/gpu-arm64" -framework Metal -framework Foundation -o out/mtl-encoder-test
        out/mtl-encoder-test
    fi
```

- [ ] **Step 2: Run the whole suite**

Run: `cd ~/Developer/GitHub/bb-metal-wt && bash build_macos.sh --test 2>&1 | grep -E "PASS|FAIL"`
Expected: every existing PASS line as before (runtime, split, launcher, settings, SRT walker,
Metal interop) plus `PASS: Metal-native foundation (object, device ticks, encoder)`; no `FAIL`.

- [ ] **Step 3: Check the existing game path still runs**

The old Metal path (layer over MoltenVK) must be unaffected in M1. Run the title-screen check
with test data (never the user's save):

```bash
cd ~/Developer/GitHub/bb-metal-wt && MAIN=~/Developer/GitHub/bloodborne_mac_port
sed -e 's/^renderer=.*/renderer=metal/' -e 's/^fullscreen=.*/fullscreen=0/' $MAIN/out/macdata/bbport.ini > /tmp/m1-check.ini
grep -q '^renderer=' /tmp/m1-check.ini || echo renderer=metal >> /tmp/m1-check.ini
(BB_CONFIG=/tmp/m1-check.ini BB_DATA_DIR=$MAIN/out/macdata BB_GAME_DIR=/Users/markhall/Bloodborne/CUSA03173 \
  BB_FRAME_STATS=1 bash run.sh > /tmp/m1-check.log 2>&1 &) ; sleep 60
grep -E "^Renderer|^Frame stats" /tmp/m1-check.log | tail -2
pkill -TERM -f "$PWD/out/bb-probe"; pkill -TERM -f "$PWD/out/bbgpu-host"
```

Expected: `Renderer: native Metal (renderer=metal / BB_METAL_FRAME=1)` and a `Frame stats` line
near 60 FPS (the title screen).

- [ ] **Step 4: Mark M1 done in the roadmap**

In `docs/superpowers/plans/2026-10-16-metal-native-roadmap.md`, change the M1 row's Gate cell to
`Unit test passes; old path still builds and runs — **done** (YYYY-MM-DD)` with today's date.

- [ ] **Step 5: Commit and push**

```bash
git add build_macos.sh docs/superpowers/plans/2026-10-16-metal-native-roadmap.md
git commit -m "Metal-native M1 done: encoder test in build_macos.sh --test

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin metal-native
```
