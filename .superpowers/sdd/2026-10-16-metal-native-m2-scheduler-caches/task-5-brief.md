### Task 5: Buffers, stream buffer, staging pool, constant ring

**Files:**
- Modify: `buffer_cache/buffer.{h,cpp}`, `renderer_vulkan/vk_staging_buffer_pool.{h,cpp}`,
  `renderer_vulkan/vk_constant_ring.h`

**Interfaces (produces):**

```cpp
enum class MemoryType { DeviceLocal, HostUncached, HostCached, Stream };  // Sparse removed
class UniqueBuffer {            // buffer.h
public:
    void Create(Metal::Device& device, u64 size, MemoryType type);
    void Destroy();
    void* Handle() const noexcept;          // id<MTLBuffer>
    u64 GpuAddress() const noexcept;
    u8* Contents() const noexcept;          // null for DeviceLocal (private storage)
private:
    Metal::Object buffer;
    u64 gpu_address{};
};
class Buffer {                  // ctor takes Metal::Device& instead of const Vulkan::Instance&
public:
    Buffer(Metal::Device& device, MemoryType type, VAddr cpu_addr, u64 size_bytes);
    void* Handle() const noexcept;
    Metal::BufferRef Ref(u64 offset = 0) const noexcept { return {Handle(), offset}; }
    u64 GpuAddress() const noexcept;
    std::span<u8> mapped_data;              // shared storage only
    // Flush/Invalidate/is_coherent/BufferDeviceAddress removed
};
class StreamBuffer;             // ctor: (Metal::Device&, MemoryType, u64 size); ticks from Device
class StagingBufferPool;        // ctor: (Metal::Device&); StagingBufferRef keeps Buffer* / offset / mapped; Flush/Invalidate removed
class ConstantRing;             // Handle() -> Metal::BufferRef; Stamp/Retire use device.CurrentTick/IsFree
```

- [ ] **Step 1: Convert** per the buffer inventory:
  - `buffer.cpp`: `MemoryType` → `MTLResourceOptions` (`DeviceLocal`: `MTLResourceStorageModePrivate`;
    `HostUncached`: `MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined`;
    `HostCached`, `Stream`: `MTLResourceStorageModeShared`). `Create`:
    `[device newBufferWithLength:size options:opts]` (keep the "Failed allocating buffer" assert),
    `label` from the debug name, `gpu_address = buf.gpuAddress`. Since `buffer.cpp` becomes
    Objective-C++, rename it `buffer.mm` and list it as `.mm` (the CMake glob picks `.mm` only in
    renderer_metal — add it explicitly with the ARC flags) — **or** keep `buffer.cpp` plain C++ and
    put the three Metal calls (create, label, address) in `renderer_metal/mtl_buffer.mm`
    (`Metal::Object CreateBuffer(Device&, u64 size, MemoryType, std::string_view label, u64* gpu_address, u8** contents)`)
    in `bbmetal_core`. Choose the second: caches stay plain C++.
  - Delete VMA includes and flag mapping (`buffer.cpp:34-64`), the sparse branch (:107-111),
    `ReadFlags`/`AllFlags` (`buffer.h:38-44`), `non_coherent_atom_size`, `StreamBuffer::Commit`'s
    flush (keep `AdvanceAndWatch`), `StagingBufferRef::Flush/Invalidate` and all their callers
    (grep `\.Flush()`/`Invalidate()` on staging refs in buffer_cache, texture_cache, runtime).
  - Staging pool and constant ring: replace `Scheduler&`/`Semaphore` tick reads with
    `Metal::Device&` (`CurrentTick`, `IsFree`, `KnownGpuTick`).
- [ ] **Step 2: Compile** the objects of `buffer.cpp`, `vk_staging_buffer_pool.cpp`; core command
  (if `mtl_buffer.mm` was added to the core, add a quick test: a 64-byte Stream buffer has
  `Contents()`, a DeviceLocal one does not, both have a non-zero GPU address). Handle check on
  the converted files: no output.
- [ ] **Step 3: Commit**: "Metal-native M2: buffers, stream buffer, staging pool and constant ring on MTLBuffer".

---

