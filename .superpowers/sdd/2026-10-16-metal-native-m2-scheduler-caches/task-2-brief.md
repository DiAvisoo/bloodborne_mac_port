### Task 2: Encoder and Device additions the caches need

**Files:**
- Modify: `gpu/shadps4/video_core/renderer_metal/mtl_device.{h,mm}`, `mtl_device_impl.h`,
  `mtl_encoder.{h,mm}`
- Test: `tests/test_mtl_encoder.mm`

**Interfaces (produces — exact):**

```cpp
// mtl_device.h, class Device (public):
    /// Commits `command_buffer` with no tick of its own. It runs before every later commit on the
    /// queue and is covered by the next tracked tick (one queue completes in order).
    void CommitUntracked(void* command_buffer);
    /// Buffers shaders reach only by GPU address (arena, page table, fault buffer): kept resident
    /// through the queue's residency set.
    void MakeResident(void* buffer);
    void Evict(void* buffer);

// mtl_encoder.h:
enum class BlitAspect : u8 { All, Depth, Stencil };
struct ColorAttachment {           // fields added (existing ones unchanged)
    u32 depth_plane{};             ///< 3D targets
    void* resolve_texture{};       ///< MSAA resolve target (StoreAndMultisampleResolve)
    u32 resolve_level{}, resolve_slice{};
};
struct DepthAttachment {           // field added
    bool stencil_only{};           ///< Stencil8-type formats: bound as the stencil attachment only
};
class Encoder {                     // members added / changed
    /// Fills `size` bytes (offset and size multiples of 4) with the 32-bit `pattern`: a blit when
    /// all four bytes are equal, else a compute kernel. Replaces the M1 low-byte fill.
    void FillBuffer(BufferRef dst, u64 size, u32 pattern);
    void CopyBufferToTexture(BufferRef src, u32 bytes_per_row, u32 bytes_per_image, void* texture,
                             u32 level, u32 slice, Origin3 origin, Size3 size,
                             BlitAspect aspect = BlitAspect::All);
    void CopyTextureToBuffer(void* texture, u32 level, u32 slice, Origin3 origin, Size3 size,
                             BufferRef dst, u32 bytes_per_row, u32 bytes_per_image,
                             BlitAspect aspect = BlitAspect::All);
    /// Buffers shaders reach by GPU address, declared on every Metal encoder this Encoder opens
    /// from now on (render encoders: read; compute: read/write) so Metal orders them.
    void DeclareIndirect(void* buffer);
    /// True while any slot, pipeline or render state is set (Submit clears all of it).
    [[nodiscard]] bool HasBindings() const noexcept;
    /// Ends encoding and commits with no tick (Device::CommitUntracked); keep-alive objects are
    /// released when it completes. For the scheduler's hoisted-upload command buffer.
    void SubmitUntracked();
};
```

Behaviour changes: `Submit()` and `SubmitUntracked()` clear every binding, pipeline and render
state (resolves the roadmap's "binding lifetime" item: a new command buffer starts unbound, as in
Vulkan); `DeclareIndirect` survives Submit. `~Encoder` ends an open pass before checking for a
command buffer (M1 minor).

- [ ] **Step 1: Write the failing tests** — add to `tests/test_mtl_encoder.mm` and call from `main`:

```objc
void TestFillPattern() {
    Metal::Device device;
    Metal::Encoder encoder{device};
    id<MTLBuffer> b = SharedBuffer(device, 64);
    encoder.FillBuffer(Ref(b), 32, 0x0000000Fu);       // HTile-style pattern: bytes differ
    encoder.FillBuffer(Ref(b, 32), 32, 0x5a5a5a5au);   // uniform bytes: blit
    device.Wait(encoder.Submit());
    const u32* w = static_cast<const u32*>(b.contents);
    bool ok = true;
    for (int i = 0; i < 8; ++i) ok &= w[i] == 0x0000000Fu;
    for (int i = 8; i < 16; ++i) ok &= w[i] == 0x5a5a5a5au;
    Expect(ok, "FillBuffer writes the exact 32-bit pattern");
}

void TestSubmitResetsBindings() {
    Metal::Device device;
    Metal::Encoder encoder{device};
    id<MTLBuffer> b = SharedBuffer(device, 16);
    encoder.SetBuffer(Metal::Stage::Compute, 3, Ref(b));
    Expect(encoder.HasBindings(), "a binding is recorded");
    device.Wait(encoder.Submit());
    Expect(!encoder.HasBindings(), "Submit clears bindings (objects may be freed after it)");
}

void TestUntrackedRunsFirst() {
    Metal::Device device;
    Metal::Encoder upload{device}, main_encoder{device};
    id<MTLBuffer> src = SharedBuffer(device, 64), mid = SharedBuffer(device, 64),
                  dst = SharedBuffer(device, 64);
    std::memset(src.contents, 0x11, 64);
    upload.CopyBuffer(Ref(src), Ref(mid), 64);   // "hoisted" upload
    main_encoder.CopyBuffer(Ref(mid), Ref(dst), 64);
    upload.SubmitUntracked();
    const u64 tick = main_encoder.Submit();
    Expect(tick == 1, "an untracked commit takes no tick");
    device.Wait(tick);
    Expect(static_cast<const u8*>(dst.contents)[63] == 0x11,
           "the untracked command buffer ran before the tracked one");
}

void TestIndirectResidency() {
    // A kernel reads a buffer only through its GPU address (as shaders read the arena).
    Metal::Device device;
    Metal::Encoder encoder{device};
    id<MTLDevice> dev = (__bridge id<MTLDevice>)device.NativeDevice();
    NSError* error = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:@R"(
#include <metal_stdlib>
using namespace metal;
kernel void through_address(device uint* out [[buffer(0)]], constant ulong& address [[buffer(1)]],
                            uint i [[thread_position_in_grid]]) {
    out[i] = ((device const uint*)address)[i];
}
)" options:nil error:&error];
    id<MTLComputePipelineState> k =
        [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"through_address"] error:&error];
    id<MTLBuffer> hidden = [dev newBufferWithLength:256 options:MTLResourceStorageModePrivate];
    id<MTLBuffer> source = SharedBuffer(device, 256), out = SharedBuffer(device, 256);
    for (u32 i = 0; i < 64; ++i) static_cast<u32*>(source.contents)[i] = 300 + i;
    device.MakeResident((__bridge void*)hidden);
    encoder.DeclareIndirect((__bridge void*)hidden);
    encoder.CopyBuffer(Ref(source), Ref(hidden), 256);  // written by a blit, read by address
    const u64 address = hidden.gpuAddress;
    encoder.SetComputePipeline((__bridge void*)k);
    encoder.SetBuffer(Metal::Stage::Compute, 0, Ref(out));
    encoder.SetBytes(Metal::Stage::Compute, 1, &address, sizeof(address));
    encoder.Dispatch({1, 1, 1}, {64, 1, 1});
    device.Wait(encoder.Submit());
    Expect(static_cast<const u32*>(out.contents)[63] == 363, "a resident buffer reads by address");
    device.Evict((__bridge void*)hidden);
}

void TestDepthAspectCopy() {
    Metal::Device device;
    Metal::Encoder encoder{device};
    id<MTLDevice> dev = (__bridge id<MTLDevice>)device.NativeDevice();
    MTLTextureDescriptor* d = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float_Stencil8 width:4 height:4 mipmapped:NO];
    d.usage = MTLTextureUsageRenderTarget;
    d.storageMode = MTLStorageModePrivate;
    id<MTLTexture> ds = [dev newTextureWithDescriptor:d];
    Metal::RenderPassDesc pass;
    pass.depth.texture = (__bridge void*)ds;
    pass.depth.has_stencil = true;
    pass.depth.clear_depth = pass.depth.clear_stencil = true;
    pass.depth.depth = 0.5;
    pass.depth.stencil = 7;
    encoder.BeginRenderPass(pass);
    encoder.EndRenderPass();
    id<MTLBuffer> depth = SharedBuffer(device, 64), stencil = SharedBuffer(device, 16);
    encoder.CopyTextureToBuffer((__bridge void*)ds, 0, 0, {}, {4, 4, 1}, Ref(depth), 16, 64,
                                Metal::BlitAspect::Depth);
    encoder.CopyTextureToBuffer((__bridge void*)ds, 0, 0, {}, {4, 4, 1}, Ref(stencil), 4, 16,
                                Metal::BlitAspect::Stencil);
    device.Wait(encoder.Submit());
    Expect(static_cast<const float*>(depth.contents)[5] == 0.5f, "depth aspect copied");
    Expect(static_cast<const u8*>(stencil.contents)[5] == 7, "stencil aspect copied");
}
```

- [ ] **Step 2: Run** the core command. Expected: compile errors (`no member named 'SubmitUntracked'`,
  `'BlitAspect'`, …).

- [ ] **Step 3: Implement.**
  - `mtl_device_impl.h`: add `id<MTLResidencySet> residency;` to `Device::Impl`.
  - `mtl_device.mm` constructor (after the event): create the set and attach it to the queue:

```objc
        MTLResidencySetDescriptor* rd = [MTLResidencySetDescriptor new];
        rd.label = @"bbport arena and address-reached buffers";
        NSError* error = nil;
        impl->residency = [impl->device newResidencySetWithDescriptor:rd error:&error];
        ASSERT_MSG(impl->residency, "No Metal residency set: {}",
                   error ? error.localizedDescription.UTF8String : "unknown");
        [impl->queue addResidencySet:impl->residency];
```

  and the new members:

```objc
void Device::CommitUntracked(void* command_buffer) {
    [(__bridge id<MTLCommandBuffer>)command_buffer commit];
}

void Device::MakeResident(void* buffer) {
    std::scoped_lock lock{residency_mutex};
    [impl->residency addAllocation:(__bridge id<MTLBuffer>)buffer];
    [impl->residency commit];
}

void Device::Evict(void* buffer) {
    std::scoped_lock lock{residency_mutex};
    [impl->residency removeAllocation:(__bridge id<MTLBuffer>)buffer];
    [impl->residency commit];
}
```

  (`std::mutex residency_mutex;` private member in `mtl_device.h`, `#include <mutex>`.)
  - `mtl_encoder.mm`:
    - `Impl` gains `std::vector<void*> indirect;` and `id<MTLComputePipelineState> fill_u32;`.
      When `Render()` opens an encoder, after `pass_opened = true;` add
      `for (void* b : indirect) [render useResource:(__bridge id<MTLBuffer>)b usage:MTLResourceUsageRead stages:MTLRenderStageVertex | MTLRenderStageFragment];`
      and when `Compute()` opens one:
      `for (void* b : indirect) [compute useResource:(__bridge id<MTLBuffer>)b usage:MTLResourceUsageRead | MTLResourceUsageWrite];`
    - `FillBuffer`: keep the blit when `pattern == u32(u8(pattern)) * 0x01010101u`; otherwise
      `ASSERT((dst.offset | size) % 4 == 0)`, build `fill_u32` once from this MSL with
      `newLibraryWithSource`:

```metal
#include <metal_stdlib>
using namespace metal;
kernel void fill_u32(device uint* dst [[buffer(0)]], constant uint2& args [[buffer(1)]],
                     uint i [[thread_position_in_grid]]) {
    if (i < args.y) dst[i] = args.x;
}
```

      then dispatch it through the Encoder's own compute path **without disturbing the caller's
      compute state**: end the current encoders, open a dedicated compute encoder
      (`[Cmd() computeCommandEncoder]`), set `fill_u32`, `setBuffer:dst offset:dst.offset atIndex:0`,
      `setBytes:{pattern, u32(size/4)} length:8 atIndex:1`, `dispatchThreads:MTLSizeMake(size/4,1,1)
      threadsPerThreadgroup:MTLSizeMake(64,1,1)`, `endEncoding`. Remove the M1 warning.
    - `CopyBufferToTexture` / `CopyTextureToBuffer`: add the `aspect` parameter and use the
      `…options:` variants with `MTLBlitOptionNone` / `MTLBlitOptionDepthFromDepthStencil` /
      `MTLBlitOptionStencilFromDepthStencil`.
    - `BeginRenderPass`: color `a.depthPlane = c.depth_plane;`; when `c.resolve_texture`:
      `a.resolveTexture = …; a.resolveLevel; a.resolveSlice; a.storeAction = MTLStoreActionStoreAndMultisampleResolve;`.
      Depth: when `d.stencil_only`, configure only `first.stencilAttachment` (texture, level, slice,
      clear from `clear_stencil`/`stencil`) and skip `depthAttachment`.
    - `DeclareIndirect(void* b)`: append if absent; set the bit so the next encoder declares it
      (an already open encoder: call `useResource` on it now too).
    - `HasBindings()`: any of `render_used`, `compute_pipeline`, or any stage's `*_used` mask set.
    - `Submit()`: after committing, reset: `impl->vertex = {}; impl->fragment = {}; impl->compute_stage = {};
      impl->render_pipeline = impl->depth_stencil = impl->compute_pipeline = nullptr;
      impl->render_used = impl->render_dirty = 0; impl->compute_pipeline_dirty = false;`
      (`viewport/scissor/...` back to defaults). `SubmitUntracked()`: same ending, but
      `impl->device.CommitUntracked(cmd)` instead of `Commit`, with the same keep-alive handler.
    - `~Encoder`: `if (impl->in_pass) EndRenderPass(); if (impl->cmd) Submit();`.

- [ ] **Step 4: Run** the core command. Expected: `PASS` twice (normal and validation layer).
  Existing tests that set state, Submit and then rely on it in a new command buffer must rebind;
  `TestCompute`/`TestLargeBytes` already rebind per command buffer — fix any that do not.

- [ ] **Step 5: Commit**: "Metal-native M2: encoder fills with 32-bit patterns, depth/stencil blit
  aspects, 3D/resolve/stencil-only attachments, residency and indirect declarations, untracked
  commits; Submit clears bindings".

---

