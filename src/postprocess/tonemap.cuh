/*
 * tonemap.cuh — ACES Film Tone Mapping & sRGB Gamma Correction
 *              ACES 电影级色调映射与 sRGB 伽马校正
 * ====================================================================
 * Post-processing step: transforms accumulated HDR radiance into
 * display-ready LDR values using the ACES (Academy Color Encoding
 * System) filmic tone mapping pipeline.
 *
 * 后处理步骤：将累积的 HDR 辐射度数据变换为可供显示器直接输出的
 * LDR [0,1] 值，使用 ACES（学院色彩编码系统）的电影级色调映射管线。
 *
 * Core concepts / 核心概念：
 *   - ACES (Academy Color Encoding System):
 *     美国电影艺术与科学学院制定的色彩管理框架，统一不同摄影机、
 *     渲染器和显示设备的色彩流水线。
 *   - Film Tone Mapping (电影级色调映射):
 *     通过 RRT + ODT 管线将 HDR 压缩到 LDR，采用 S 形曲线保留
 *     高光细节和暗部层次，避免高光过曝死白。
 *   - sRGB Gamma Correction (sRGB 伽马校正):
 *     将线性光值编码为 sRGB 非线性格式，匹配显示器的 EOTF 特性。
 *     采用分段函数：暗部线性段 (linear toe) + 亮部幂函数段。
 *
 * Pipeline per pixel / 每条像素的处理管线:
 *   1. Average:   pix = accumulatedHDR / numSamples     // 多采样取均值
 *   2. Tone map:  pix = ACESFitted(pix)                 // 电影级 S 曲线
 *   3. Gamma:     pix = LinearToSRGB(pix)                // 线性光→sRGB 编码
 *   4. Clamp:     pix = saturate(pix)                    // 截断到 [0,1]
 *
 * References / 参考文献:
 *   Stephen Hill / MJP's BakingLab ACES.hlsl
 *     (fitted approximation of the full ACES RRT+ODT pipeline)
 *     (完整 ACES RRT+ODT 管线的多项式拟合逼近)
 *   Krzysztof Narkowicz, "ACES Filmic Tone Mapping Curve" (2016)
 *   IEC 61966-2-1:1999 — sRGB color space specification / sRGB 色彩空间规范
 * ====================================================================
 */

#pragma once

#include "glm/glm.hpp"

// ---------------------------------------------------------------------------
// ACESInputMat — ACES Input Matrix / ACES 输入矩阵变换
//
// Converts linear sRGB / Rec.709 primaries into the ACES AP1 working
// colour space.  This single 3×3 matrix bakes together:
//   sRGB → XYZ (Rec.709 primaries)
//   D65 → D60 white-point adaptation (chromatic adaptation)
//   AP0 → AP1 (archival space to working space)
// Coefficients from Hill's BakingLab fit.
//
// 将线性 sRGB / Rec.709 原色转换到 ACES AP1 工作色彩空间。
// 该矩阵内部融合了 sRGB→XYZ、D65→D60 白点适应、AP0→AP1 三个步骤。
//
// ACES colour space hierarchy / ACES 色彩空间层级:
//   AP0 (ACES2065-1) — archival primary space, "imaginary" primaries
//                      covering the entire visible spectrum
//                      存档主空间，使用"虚原色"覆盖整个可见光谱
//   AP1 (ACEScg)     — working space with real primaries where the
//                      RRT operates internally
//                      工作空间，使用真实原色，RRT 在此空间内运算
// ---------------------------------------------------------------------------
__device__ inline glm::vec3 ACESInputMat(glm::vec3 color)
{
    // GLSL: mat3(col0, col1, col2) * vec3 → result = r*col0 + g*col1 + b*col2
    //       列主序：构造函数参数按列依次填充
    // Explicit per-component form matching GLSL M*v / HLSL mul(M,v):
    return glm::vec3(
        0.59719f * color.r + 0.35458f * color.g + 0.04823f * color.b,
        0.07600f * color.r + 0.90834f * color.g + 0.01566f * color.b,
        0.02840f * color.r + 0.13383f * color.g + 0.83777f * color.b);
}

// ---------------------------------------------------------------------------
// ACESOutputMat — ACES Output Matrix / ACES 输出矩阵变换
//
// Converts from AP1 working space back to linear sRGB / Rec.709 primaries.
// Inverse of ACESInputMat: AP1→AP0, D60→D65 white-point adaptation, XYZ→sRGB.
//
// 将 AP1 工作空间转换回线性 sRGB / Rec.709 原色。
// 是 ACESInputMat 的逆过程：AP1→AP0 → D60→D65 白点适应 → XYZ→sRGB。
// ---------------------------------------------------------------------------
__device__ inline glm::vec3 ACESOutputMat(glm::vec3 color)
{
    // GLSL: mat3(col0, col1, col2) * vec3 → result = r*col0 + g*col1 + b*col2
    //       列主序：构造函数参数按列依次填充
    // Explicit per-component form matching GLSL M*v / HLSL mul(M,v):
    return glm::vec3(
         1.60475f * color.r - 0.53108f * color.g - 0.07367f * color.b,
        -0.10208f * color.r + 1.10813f * color.g - 0.00605f * color.b,
        -0.00327f * color.r - 0.07276f * color.g + 1.07602f * color.b);
}

// ---------------------------------------------------------------------------
// RRTAndODTFit — rational polynomial fit of the combined ACES RRT + ODT
//               RRT + ODT 联合有理多项式拟合 (电影级 S 曲线核心)
//
// The Reference Rendering Transform (RRT) and Output Device Transform
// (ODT) together form an S-shaped curve that gracefully rolls off
// highlights while preserving shadow and mid-tone contrast.
// RRT (参考渲染变换) 和 ODT (输出设备变换) 共同构成 S 形曲线，
// 在保留暗部和中间调对比度的同时，平滑地压缩高光。
//
// The full ACES reference pipeline uses segmented cubic splines; this
// is a per-channel rational approximation / 完整 ACES 使用分段三次样条，
// 此处用逐通道有理多项式逼近：
//
//     f(x) = (x² + 0.0245786*x - 0.000090537)
//          / (0.983729*x² + 0.4329510*x + 0.238081)
//
// Applied per-channel (not on luminance) so it naturally desaturates
// bright colours — the "filmic highlight roll-off" characteristic.
// 在 RGB 三通道上分别独立计算（而非仅对亮度），使高光颜色自然趋向灰白
// ——即"电影级高光衰减" 的特征效果。
// ---------------------------------------------------------------------------
__device__ inline glm::vec3 RRTAndODTFit(glm::vec3 v)
{
    // Numerator / 分子:   v² + 0.0245786*v - 0.000090537
    // Denominator / 分母: 0.983729*v² + 0.4329510*v + 0.238081
    glm::vec3 num = v * (v + 0.0245786f) - glm::vec3(0.000090537f);
    glm::vec3 den = v * (0.983729f * v + 0.4329510f) + glm::vec3(0.238081f);
    return num / den;
}

// ---------------------------------------------------------------------------
// ACESFilm (Narkowicz) — Krzysztof Narkowicz's "ACES Approx" (2016)
//                        Narkowicz 简化 ACES 逼近 (2016)
//
// Simpler fit that skips colour-space matrices entirely and operates
// directly on sRGB / Rec.709 primaries.  The rational curve:
//     f(x) = (x * (2.51*x + 0.03)) / (x * (2.43*x + 0.59) + 0.14)
//
// 无色彩空间矩阵的最简版本，直接在 sRGB / Rec.709 原色上运行。
//
// Pros / 优点: zero matrix confusion / 无矩阵困扰, well-tested / 久经考验
// Cons / 缺点: oversaturates bright highlights / 高光过饱和 (没有 filmic desaturation)
//
// Reference / 参考:
//   https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
// ---------------------------------------------------------------------------
__device__ inline glm::vec3 ACESFilm_Narkowicz(glm::vec3 x)
{
    const float a = 2.51f;
    const float b = 0.03f;
    const float c = 2.43f;
    const float d = 0.59f;
    const float e = 0.14f;
    return glm::clamp((x * (a * x + b)) / (x * (c * x + d) + e),
                      glm::vec3(0.0f), glm::vec3(1.0f));
}

// ---------------------------------------------------------------------------
// ACESFitted — Full Stephen Hill / MJP BakingLab ACES pipeline
//              完整的 Stephen Hill / MJP BakingLab ACES 管线
//
// 1. Convert from rendering colour space to AP1 working space.
// 2. Apply the RRT+ODT rational fit (filmic S-curve).
// 3. Convert back from AP1 to rendering colour space.
// 4. Saturate to [0, 1].
//
// 1. 将渲染色彩空间转换到 AP1 工作空间
// 2. 应用 RRT+ODT 有理拟合 (电影级 S 曲线)
// 3. 从 AP1 工作空间转换回渲染色彩空间
// 4. 钳制到 [0, 1]
//
// Output is still linear light — apply LinearToSRGB() before writing
// to an 8-bit framebuffer.
// 输出仍为线性光 —— 写入 8-bit 帧缓冲前必须调用 LinearToSRGB()。
//
// Why Hill's fit instead of the full ACES? / 为什么选 Hill 拟合而非完整 ACES?
//   - Full ACES requires LUT texture lookups + segmented splines
//     完整 ACES 需要 LUT 纹理查表 + 分段样条，不适合纯 CUDA kernel
//   - Hill's fit: 1% complexity for 99% of the visual result
//     Hill 拟合用 1% 复杂度达到 99% 视觉效果
//   - Used by Unreal Engine, Unity HDRP, and most real-time renderers
//     Unreal、Unity HDRP 等工业引擎均使用此版本
// ---------------------------------------------------------------------------
__device__ inline glm::vec3 ACESFitted(glm::vec3 color)
{
    color = ACESInputMat(color);
    color = RRTAndODTFit(color);
    color = ACESOutputMat(color);
    return glm::clamp(color, glm::vec3(0.0f), glm::vec3(1.0f));
}

// ---------------------------------------------------------------------------
// LinearToSRGB — IEC 61966-2-1 piecewise sRGB transfer function
//                IEC 61966-2-1 标准分段 sRGB 传递函数 (伽马编码)
//
// Encodes linear-light values for an sRGB display.  The piecewise form
// has a linear toe (preserves shadow detail in 8-bit quantisation) and
// a power segment with exponent 1/2.4 for mid-tones and highlights.
//
// 将线性光值编码为 sRGB 显示器所需的非线性信号。分段形式具有：
//   暗部线性段 (linear toe) — 在 8-bit 量化时保留暗部细节
//   亮部幂函数段 (power segment) — 指数 1/2.4，处理中间调和高光
//
//   Linear [0.0031308, 1] → 1.055 * x^(1/2.4) - 0.055   (power / 幂函数)
//   Linear [0, 0.0031308) → 12.92 * x                    (linear toe / 线性段)
//
// This is only needed when writing to 8-bit integer pixel formats.
// For floating-point or HDR displays, skip this step.
// 仅在输出到 8-bit 整数像素格式时需要此步骤。浮点 / HDR 显示可跳过。
// ---------------------------------------------------------------------------
__device__ inline glm::vec3 LinearToSRGB(glm::vec3 c)
{
    const float threshold = 0.0031308f;   // piecewise threshold / 分段阈值
    const float slope     = 12.92f;        // linear-segment slope / 线性段斜率
    const float offset    = 0.055f;        // power-segment offset / 幂函数段偏移
    const float exponent  = 1.0f / 2.4f;   // gamma exponent ≈ 0.4167 / 伽马指数

    glm::vec3 lo = c * slope;              // linear toe / 线性段 (暗部保留细节)
    glm::vec3 hi = 1.055f * glm::pow(c, glm::vec3(exponent)) - glm::vec3(offset); // power / 幂函数段
    glm::bvec3 mask = glm::greaterThan(c, glm::vec3(threshold));

    return glm::vec3(
        mask.x ? hi.x : lo.x,
        mask.y ? hi.y : lo.y,
        mask.z ? hi.z : lo.z);
}

// ---------------------------------------------------------------------------
// tonemapKernel — Post-processing kernel entry point / 后处理 kernel 入口
//
// Reads the raw accumulated HDR radiance from g_dev.image (sum of all
// path samples for the current frame), averages by the iteration count,
// applies ACES filmic tone mapping, encodes to sRGB, clamps, and writes
// the LDR [0,1] result to the display output buffer.
//
// 从 g_dev.image (当前帧所有路径采样的原始 HDR 累加和) 读取数据，
// 除以迭代次数取平均 → ACES 电影级色调映射 → sRGB 伽马编码 → 钳制，
// 最终将 LDR [0,1] 结果写入 g_dev.imageDisplay 显示缓冲。
//
// Launched as a 2-D grid matching the framebuffer resolution.
// 以 2-D 线程网格启动，与帧缓冲分辨率对齐。
//
// g_dev.image is left untouched so the cudaMemcpy D2H (line ~936)
// still pulls raw HDR for saveImage(), and sendImageToPBO can stay
// completely unchanged as starter code.
// g_dev.image 保持不变：cudaMemcpy D2H 仍获得原始 HDR 供 saveImage(),
// sendImageToPBO starter code 也无需任何修改。
// ---------------------------------------------------------------------------
__global__ void tonemapKernel(
    const glm::vec3* __restrict__ inputImage,   // accumulated HDR / 原始 HDR 累加 (g_dev.image)
    glm::vec3*       __restrict__ outputImage,   // LDR [0,1] display output / LDR 显示输出 (g_dev.imageDisplay)
    glm::ivec2 resolution,
    int iter,                                    // iteration count (= accumulated samples) / 当前迭代数
    int debugMode)                               // 0 = Hill ACES / 1 = linear bypass / 2 = Narkowicz ACES
{                                                // 0: Hill ACES   1: 线性旁路   2: Narkowicz ACES
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int idx = x + y * resolution.x;

        // 1. Average accumulated HDR samples for this pixel
        //    将累积的 HDR 样本取平均，得到该像素的平均辐射度
        glm::vec3 pix = inputImage[idx] / (float)iter;

        // Guard against negative pixels: floating-point accumulation error
        // can produce tiny negative values (~ -1e-7).  The ACES rational
        // functions are undefined for x < 0 (Narkowicz maps -∞ → ~1.03).
        // Clamping to zero prevents hot-pixel artifacts and NaN propagation.
        // 防御负值像素：浮点累加误差可能产生极小负值 (~ -1e-7)。
        // ACES 有理函数对 x < 0 行为不定 (Narkowicz 将 -∞ 映射到 ~1.03)。
        // 截断到零防止亮斑伪影和 NaN 传播。
        pix = glm::max(pix, glm::vec3(0.0f));

        if (debugMode == 0)
        {
            // Hill ACES: colour-space matrices + RRT/ODT fit
            // Hill ACES: 色彩空间矩阵 + RRT/ODT 拟合 (高光去饱和)
            pix = ACESFitted(pix);
        }
        else if (debugMode == 2)
        {
            // Narkowicz ACES: no matrices, pure S-curve on sRGB primaries
            // Narkowicz ACES: 无矩阵, 直接在 sRGB 原色上的 S 曲线
            pix = ACESFilm_Narkowicz(pix);
        }
        else
        {
            // DEBUG BYPASS (debugMode==1): simple linear clamp
            //    调试旁路: 简单线性钳制 (复现旧 sendImageToPBO 行为)
            pix = glm::clamp(pix, glm::vec3(0.0f), glm::vec3(1.0f));
        }

        // 3. Linear-light → sRGB transfer function (gamma encoding)
        //    线性光 → sRGB 编码 (伽马校正, 匹配显示器 EOTF)
        pix = LinearToSRGB(pix);

        // 4. Clamp to valid LDR range and write to display buffer
        //    钳制到 LDR 有效范围并写入显示缓冲
        pix = glm::clamp(pix, glm::vec3(0.0f), glm::vec3(1.0f));
        outputImage[idx] = pix;
    }
}
