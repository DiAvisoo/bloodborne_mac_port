// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the texture cache's blit programs in MSL (Metal-native renderer), translated by hand
// from host_shaders/fs_tri.vert, color_to_ms_depth.frag, ms_image_blit.frag and
// depth_resample.frag. One library; inputs at textures 0/1, push constants at buffer 0.

#pragma once

namespace Metal::BlitShaders {

// Function names in Source.
inline constexpr const char* FullscreenTriangle = "fs_tri_vs";
/// color_to_ms_depth.frag: sample N of the depth target takes channel N of the colour texel.
inline constexpr const char* ColorToMsDepth = "color_to_ms_depth_fs";
/// ms_image_blit.frag: non-MSAA source (texture2d) and MSAA source (texture2d_ms, per sample),
/// in float, uint and int flavours (Metal wants the shader's type to match the formats').
inline constexpr const char* MsImageBlit[3] = {"ms_image_blit_fs", "ms_image_blit_uint_fs",
                                               "ms_image_blit_int_fs"};
inline constexpr const char* MsImageBlitMsaa[3] = {
    "ms_image_blit_msaa_fs", "ms_image_blit_msaa_uint_fs", "ms_image_blit_msaa_int_fs"};
/// depth_resample.frag: depth only; HAS_STENCIL (stencil exported from texture 1);
/// STENCIL_BITS (keeps pixels whose stencil (texture 1) has the bit at buffer 0; bit 0 keeps all).
inline constexpr const char* DepthResample = "depth_resample_fs";
inline constexpr const char* DepthResampleStencil = "depth_resample_stencil_fs";
inline constexpr const char* DepthResampleStencilBits = "depth_resample_stencil_bits_fs";

inline constexpr const char* Source = R"msl(
#include <metal_stdlib>
using namespace metal;

struct FsTriOut {
    float4 position [[position]];
    float2 uv;
};

// fs_tri.vert: one triangle covering the target. Metal's clip space has y up, Vulkan's y down:
// y is negated so that uv (0, 0) is still the top-left texel.
vertex FsTriOut fs_tri_vs(uint vertex_id [[vertex_id]]) {
    const float2 pos = float2(float((vertex_id & 1u) << 2u), float((vertex_id & 2u) << 1u));
    FsTriOut out;
    out.position = float4(pos.x - 1.0, 1.0 - pos.y, 0.0, 1.0);
    out.uv = pos * 0.5;
    return out;
}

struct DepthOut {
    float depth [[depth(any)]];
};

struct DepthStencilOut {
    float depth [[depth(any)]];
    uint stencil [[stencil]];
};

static uint2 TexelOf(float2 uv, uint2 size) {
    return min(uint2(uv * float2(size)), size - 1u);
}

// color_to_ms_depth.frag: runs per sample ([[sample_id]]); a colour texel holds up to four
// samples' depths (one per channel).
fragment DepthOut color_to_ms_depth_fs(FsTriOut in [[stage_in]], uint sample_id [[sample_id]],
                                       texture2d<float> color [[texture(0)]]) {
    const uint2 coord = TexelOf(in.uv, uint2(color.get_width(), color.get_height()));
    DepthOut out;
    out.depth = color.read(coord, 0)[min(sample_id, 3u)];
    return out;
}

// ms_image_blit.frag, without and with SRC_MSAA.
#define MS_IMAGE_BLIT(T, SUFFIX)                                                                  \
    fragment vec<T, 4> ms_image_blit##SUFFIX##_fs(float4 position [[position]],                  \
                                                  texture2d<T> in_tex [[texture(0)]]) {           \
        return in_tex.read(uint2(position.xy), 0);                                               \
    }                                                                                            \
    fragment vec<T, 4> ms_image_blit_msaa##SUFFIX##_fs(float4 position [[position]],             \
                                                       uint sample_id [[sample_id]],              \
                                                       texture2d_ms<T> in_tex [[texture(0)]]) {   \
        return in_tex.read(uint2(position.xy), sample_id);                                       \
    }
MS_IMAGE_BLIT(float, )
MS_IMAGE_BLIT(uint, _uint)
MS_IMAGE_BLIT(int, _int)

// depth_resample.frag: nearest resample of depth (and stencil) between native and reduced scene
// targets.
fragment DepthOut depth_resample_fs(FsTriOut in [[stage_in]],
                                    depth2d<float> src_depth [[texture(0)]]) {
    const uint2 coord = TexelOf(in.uv, uint2(src_depth.get_width(), src_depth.get_height()));
    DepthOut out;
    out.depth = src_depth.read(coord, 0);
    return out;
}

fragment DepthStencilOut depth_resample_stencil_fs(FsTriOut in [[stage_in]],
                                                   depth2d<float> src_depth [[texture(0)]],
                                                   texture2d<uint> src_stencil [[texture(1)]]) {
    const uint2 coord = TexelOf(in.uv, uint2(src_depth.get_width(), src_depth.get_height()));
    DepthStencilOut out;
    out.depth = src_depth.read(coord, 0);
    out.stencil = src_stencil.read(coord, 0).r;
    return out;
}

// Fixed-function stencil replacement writes one bit per draw. The first draw (bit 0, write mask
// 0) copies depth everywhere, including zero-stencil pixels.
fragment DepthOut depth_resample_stencil_bits_fs(FsTriOut in [[stage_in]],
                                                 depth2d<float> src_depth [[texture(0)]],
                                                 texture2d<uint> src_stencil [[texture(1)]],
                                                 constant uint& bit [[buffer(0)]]) {
    const uint2 coord = TexelOf(in.uv, uint2(src_depth.get_width(), src_depth.get_height()));
    if (bit != 0u && (src_stencil.read(coord, 0).r & bit) == 0u) {
        discard_fragment();
    }
    DepthOut out;
    out.depth = src_depth.read(coord, 0);
    return out;
}
)msl";

} // namespace Metal::BlitShaders
