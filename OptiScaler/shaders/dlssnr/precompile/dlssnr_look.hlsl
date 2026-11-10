// The look stage: reshapes what the model did before the resolve sees it.
//
// The model's answer N and the proxy P it was shown are both at the model's working resolution and in the
// same encoding (sRGB over the white point, or the frame as it was in passthrough). The model's EDIT is
// e = log2(N / P) per channel, in stops; this stage rewrites it to e' and writes N' = N * 2^(t * (e' - e)),
// which the resolve, the JBU and the edit cache then take in place of N. With every gain at its identity
// e' == e and N' is N, which is why the pass is skipped altogether then.
//
// The edit is split three ways, all in stops:
//
//   a   the achromatic edit, the luminance-weighted mean of e -- how much lighter or darker
//   c   the opponent edit, e - a -- the colour change, which sums to zero over the channels
//   aB  the broad part of a (lighting): a guided filter of a, guided by the proxy's log luminance, with
//       its coefficients at half resolution over a box of the detail radius. Guided rather than blurred so
//       the broad part follows the picture's own edges: a blur would spread a lit wall's brightening onto
//       the dark doorway beside it, and the difference would come back as a halo in the detail.
//   aD  the fine part, a - aB (texture)
//
// Then every gain acts in stops: tone and detail on the two parts, brightening and darkening on the sign,
// colour on the radial (toward or away from grey) and hue on the tangential part of c, shadows, midtones
// and highlights on the proxy's luminance zone, strength on all of it, and three soft knees limit the
// result.
//
// Taken from RenoDX's DLSS 5 addon (clshortfuse, https://github.com/clshortfuse/renodx, as maintained in
// https://github.com/williamstosh/RenoDX-DLSSNR-Nexus): its look stage's edit split, guided-filter band
// split and gain shaping (look_common.hlsli ShapeEdit, look_band.cs). Its halo suppression and temporal
// stabilisation are not taken. MIT, see Licenses/RenoDX_ATTRIBUTION.txt.
//
// Passes, chosen by gMode, each separated by a UAV barrier on the host:
//   0  down     gBandA = 2x2 means of (I, a, I*I, I*a)                                  half resolution
//   1  H        gBandB = horizontal box of gBandA
//   2  V+coef   vertical box, then the guided filter's coefficients A = cov(I, a) / (var(I) + eps) and
//               B = mean(a) - A * mean(I) into gBandA.xy
//   3  H        gBandB.xy = horizontal box of the coefficients
//   4  V        gBandA.xy = vertical box of the coefficients
//   5  compose  gTarget = N'                                                            working resolution
// Boxes are clamped at the border and normalised by the in-bounds count, so a constant field stays constant.

cbuffer LookParams : register(b0)
{
    uint  gMode;
    uint  gWidth;        // the model's working resolution: P, N and the output
    uint  gHeight;
    uint  gHalfWidth;    // the band grid, (working + 1) / 2
    uint  gHalfHeight;
    uint  gRadius;       // the box radius in band pixels, from the detail radius
    uint  gPassthrough;  // P and N are not sRGB-encoded
    uint  gBands;        // tone or detail is not 1: split the edit
    float gStrength;     // the whole edit
    float gBrighten;     // the achromatic edit where it lightens
    float gDarken;       // and where it darkens
    float gMaxBrighten;  // soft limits in stops, 0 = none
    float gMaxDarken;
    float gColour;       // the colour edit toward or away from grey
    float gHue;          // the colour edit across, the tint
    float gMaxColour;    // soft limit on the colour edit, 0 = none
    float gShadows;      // by the proxy's luminance zone
    float gMidtones;
    float gHighlights;
    float gTone;         // the broad part of the achromatic edit
    float gDetail;       // the fine part
    float gGfEps;        // the guided filter's regularisation, stops squared
    float gPad0;
    float gPad1;
};

Texture2D<float4>   gProxy  : register(t0);  // P: what the model was shown
Texture2D<float4>   gModel  : register(t1);  // N: what it returned
RWTexture2D<float4> gTarget : register(u0);  // N', at the working resolution
RWTexture2D<float4> gBandA  : register(u1);  // half resolution
RWTexture2D<float4> gBandB  : register(u2);

static const float  kRatioFloor = 1.0 / 1024.0;
static const float3 kW = float3(0.212639, 0.715169, 0.072192);
static const float  kLog2Grey = -2.4739312; // log2(0.18)

float3 SrgbToLinear(float3 v)
{
    v = saturate(v);
    return lerp(v / 12.92, pow((v + 0.055) / 1.055, 2.4), step(0.04045, v));
}

float3 LinearToSrgb(float3 v)
{
    v = saturate(v);
    return lerp(v * 12.92, 1.055 * pow(max(v, 1e-8), 1.0 / 2.4) - 0.055, step(0.0031308, v));
}

float3 Decode(float3 v) { return gPassthrough != 0 ? max(v, 0.0) : SrgbToLinear(v); }
float3 Encode(float3 v) { return gPassthrough != 0 ? max(v, 0.0) : LinearToSrgb(v); }

// How far the edit at this pixel can be trusted: nothing at black, where the ratio is a ratio of noise.
float FloorTrust(float3 p)
{
    const float m = max(p.x, max(p.y, p.z));
    return smoothstep(1.0 / 1024.0, 1.0 / 64.0, m);
}

struct Edit
{
    float3 n;   // the model's answer, linear
    float3 lp;  // log2 of the proxy, floored
    float3 e;   // the edit per channel, stops
    float  a;   // achromatic: the weighted mean of e
    float3 c;   // opponent: e - a, dot(kW, c) = 0
    float  I;   // log2 of the proxy's luminance
    float  t;   // floor trust
    float  alpha;
};

Edit ReadEdit(int2 pixel)
{
    Edit x;
    const float3 p = Decode(gProxy.Load(int3(pixel, 0)).rgb);
    const float4 n = gModel.Load(int3(pixel, 0));
    x.n = Decode(n.rgb);
    x.alpha = n.a;
    x.lp = log2(max(p, kRatioFloor));
    x.e = log2(max(x.n, kRatioFloor)) - x.lp;
    x.a = dot(kW, x.e);
    x.c = x.e - x.a;
    x.I = log2(max(dot(kW, p), kRatioFloor));
    x.t = FloorTrust(p);
    return x;
}

// The colour edit split into its part along the proxy's own log-opponent direction o (radial: more or less
// saturated) and the rest (tangential: a tint). Near grey o has no direction, so s fades the split out there
// and the colour gain alone applies.
void ColourSplit(float3 c, float3 lp, out float3 cR, out float3 cT, out float s)
{
    const float3 o = lp - dot(kW, lp);
    const float oo = dot(kW, o * o);
    cR = (dot(kW, c * o) / max(oo, 1e-6)) * o;
    cT = c - cR;
    s = smoothstep(0.05, 0.25, sqrt(oo));
}

// Shadows below about 3.5 stops under mid grey, highlights from about 2.5 above, midtones between, with
// smooth two-stop crossovers.
float ZoneGain(float I)
{
    const float L = I - kLog2Grey;
    const float wSh = 1.0 - smoothstep(-3.5, -1.5, L);
    const float wHi = smoothstep(0.5, 2.5, L);
    return wSh * gShadows + (1.0 - wSh - wHi) * gMidtones + wHi * gHighlights;
}

// Every gain acts on stops, and every operator is continuous and monotonic: the sign split is piecewise
// linear rather than a smooth blend, which would not be monotonic when the two gains differ.
float3 ShapeEdit(Edit x, float aB, float aD)
{
    float a = gTone * aB + gDetail * aD;
    a = gBrighten * max(a, 0.0) + gDarken * min(a, 0.0);

    float3 cR;
    float3 cT;
    float s;
    ColourSplit(x.c, x.lp, cR, cT, s);
    float3 c = s * (gColour * cR + gHue * cT) + (1.0 - s) * gColour * x.c;

    const float g = gStrength * ZoneGain(x.I);
    a *= g;
    c *= g;

    // The argument is clamped because tanh can be expanded as (e^x - e^-x) / (e^x + e^-x), which is
    // Inf / Inf past |x| ~ 88; tanh(10) is already 1.0 in float32, so nothing else changes.
    if (a > 0.0 && gMaxBrighten > 0.0)
        a = gMaxBrighten * tanh(min(a / gMaxBrighten, 10.0));

    if (a < 0.0 && gMaxDarken > 0.0)
        a = gMaxDarken * tanh(max(a / gMaxDarken, -10.0));

    if (gMaxColour > 0.0)
    {
        const float norm = sqrt(dot(kW, c * c));

        if (norm > 1e-6)
            c *= gMaxColour * tanh(min(norm / gMaxColour, 10.0)) / norm;
    }

    return a + c;
}

float4 Box(int2 pixel, bool horizontal)
{
    const int last = horizontal ? (int) gHalfWidth - 1 : (int) gHalfHeight - 1;
    const int r = (int) gRadius;
    const int centre = horizontal ? pixel.x : pixel.y;
    const int lo = max(centre - r, 0);
    const int hi = min(centre + r, last);
    float4 sum = 0.0;

    for (int i = lo; i <= hi; ++i)
        sum += horizontal ? gBandA[int2(i, pixel.y)] : gBandB[int2(pixel.x, i)];

    return sum / (float) (hi - lo + 1);
}

// The guided filter's coefficients at a working-resolution pixel, bilinear from the half-resolution grid at
// (x + 0.5) * 0.5 - 0.5.
float2 Coefficients(int2 pixel)
{
    const float2 pos = clamp((float2(pixel) + 0.5) * 0.5 - 0.5, 0.0,
                             float2(gHalfWidth - 1, gHalfHeight - 1));
    const int2 b0 = int2(floor(pos));
    const int2 b1 = min(b0 + 1, int2(gHalfWidth, gHalfHeight) - 1);
    const float2 f = pos - float2(b0);
    const float2 q00 = gBandA[b0].xy;
    const float2 q10 = gBandA[int2(b1.x, b0.y)].xy;
    const float2 q01 = gBandA[int2(b0.x, b1.y)].xy;
    const float2 q11 = gBandA[b1].xy;
    return lerp(lerp(q00, q10, f.x), lerp(q01, q11, f.x), f.y);
}

[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    const int2 pixel = int2(id.xy);

    if (gMode == 5)
    {
        if (id.x >= gWidth || id.y >= gHeight)
            return;

        const Edit x = ReadEdit(pixel);
        float aB = x.a;
        float aD = 0.0;

        if (gBands != 0)
        {
            const float2 ab = Coefficients(pixel);
            aB = ab.x * x.I + ab.y;
            aD = x.a - aB;
        }

        const float3 shaped = ShapeEdit(x, aB, aD);
        gTarget[pixel] = float4(Encode(x.n * exp2(x.t * (shaped - x.e))), x.alpha);
        return;
    }

    if (id.x >= gHalfWidth || id.y >= gHalfHeight)
        return;

    if (gMode == 0)
    {
        float4 moments = 0.0;
        float count = 0.0;

        for (int dy = 0; dy <= 1; ++dy)
        {
            for (int dx = 0; dx <= 1; ++dx)
            {
                const int2 source = pixel * 2 + int2(dx, dy);

                if (source.x >= (int) gWidth || source.y >= (int) gHeight)
                    continue;

                const Edit x = ReadEdit(source);
                moments += float4(x.I, x.a, x.I * x.I, x.I * x.a);
                count += 1.0;
            }
        }

        gBandA[pixel] = moments / max(count, 1.0);
    }
    else if (gMode == 1)
    {
        gBandB[pixel] = Box(pixel, true);
    }
    else if (gMode == 2)
    {
        const float4 m = Box(pixel, false);
        const float variance = max(m.z - m.x * m.x, 0.0);
        const float covariance = m.w - m.x * m.y;
        const float slope = covariance / (variance + gGfEps);
        gBandA[pixel] = float4(slope, m.y - slope * m.x, 0.0, 0.0);
    }
    else if (gMode == 3)
    {
        gBandB[pixel] = float4(Box(pixel, true).xy, 0.0, 0.0);
    }
    else
    {
        gBandA[pixel] = float4(Box(pixel, false).xy, 0.0, 0.0);
    }
}
