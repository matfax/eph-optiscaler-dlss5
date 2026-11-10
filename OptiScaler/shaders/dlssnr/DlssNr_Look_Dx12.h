#pragma once

// The look stage (precompile/dlssnr_look.hlsl): reshapes the model's edit before the resolve, the JBU and the
// edit cache see it. Ported from RenoDX's DLSS 5 addon, see the shader and Licenses/RenoDX_ATTRIBUTION.txt.

#include <d3d12.h>
#include <d3dx/d3dx12.h>
#include <shaders/Shader_Dx12.h>
#include <shaders/Shader_Dx12Utils.h>

#include <vector>

class Config;

// The player's values. With every one at its identity the stage does not run; the default shadow limits
// (Config) are not, so it normally does.
struct DlssNrLookSettings
{
    float strength = 1.0f;
    float brighten = 1.0f;
    float darken = 1.0f;
    float maxBrighten = 0.0f; // stops, 0 = no limit
    float maxDarken = 0.0f;
    float colour = 1.0f;
    float hue = 1.0f;
    float maxColour = 0.0f;
    float shadows = 1.0f;
    float midtones = 1.0f;
    float highlights = 1.0f;
    float tone = 1.0f;
    float detail = 1.0f;
    float detailRadius = 1.0f;      // percent of the model's working height
    float shadowMaxBrighten = 0.0f; // soft limits on the brightness change in shadows, stops, 0 = none
    float shadowMaxDarken = 0.0f;
    uint32_t blackRestore = 0; // 0 off, 1 near-black blocks, 2 near-black blocks with a smooth taper

    static DlssNrLookSettings FromConfig(const Config& cfg);

    bool Neutral() const;
    bool Bands() const { return tone != 1.0f || detail != 1.0f; }
};

// Mirrors the shader's LookParams.
struct alignas(256) DlssNrLookConstants
{
    uint32_t Mode;
    uint32_t Width;
    uint32_t Height;
    uint32_t HalfWidth;
    uint32_t HalfHeight;
    uint32_t Radius;
    uint32_t Passthrough;
    uint32_t Bands;
    float Strength;
    float Brighten;
    float Darken;
    float MaxBrighten;
    float MaxDarken;
    float Colour;
    float Hue;
    float MaxColour;
    float Shadows;
    float Midtones;
    float Highlights;
    float Tone;
    float Detail;
    float GfEps;
    float ShadowMaxBrighten;
    float ShadowMaxDarken;
    uint32_t BlackRestore;
    uint32_t BlocksX;
    uint32_t BlocksY;
    float Pad0;
    float Pad1;
};

#define DLSSNR_LOOK_NUM_OF_HEAPS 64

class DlssNrLook_Dx12 : public Shader_Dx12
{
  public:
    explicit DlssNrLook_Dx12(ID3D12Device* device);
    ~DlssNrLook_Dx12();

    // `proxy` (what the model was shown) must be in NON_PIXEL_SHADER_RESOURCE and `model` (its answer) in
    // UNORDERED_ACCESS, both at width x height. Returns the reshaped answer, in UNORDERED_ACCESS, in the
    // model's own format; or nullptr when the settings are the identity or something failed, and the caller
    // keeps `model`. `model` is back in UNORDERED_ACCESS either way.
    ID3D12Resource* Apply(ID3D12GraphicsCommandList* cmd, ID3D12Device* device, const DlssNrLookSettings& look,
                          ID3D12Resource* proxy, ID3D12Resource* model, unsigned int width, unsigned int height,
                          bool passthrough);

    static constexpr float kGfEps = 0.01f;         // RenoDX's guided-filter regularisation, stops squared
    static constexpr unsigned int kMaxRadius = 64; // band pixels

    static unsigned int BandRadius(float detailRadiusPercent, unsigned int height);

  private:
    static constexpr uint32_t kSrvCount = 2;
    static constexpr uint32_t kUavCount = 4;
    static constexpr unsigned int kBlock = 32; // the black restore's block, as in the shader

    bool EnsureResources(ID3D12Device* device, DXGI_FORMAT format, unsigned int width, unsigned int height, bool bands);
    bool Pass(ID3D12GraphicsCommandList* cmd, const DlssNrLookConstants& constants, ID3D12Resource* proxy,
              ID3D12Resource* model, unsigned int groupsX, unsigned int groupsY);
    void Retire(ID3D12Resource*& res);
    void TickRetired();
    void ReleaseResources();

    FrameDescriptorHeap _frameHeaps[DLSSNR_LOOK_NUM_OF_HEAPS];
    ID3D12Resource* _constantBuffers[DLSSNR_LOOK_NUM_OF_HEAPS] = {};
    uint32_t _heapIndex = 0;

    ID3D12Resource* _output = nullptr; // N', working resolution, the model's format
    ID3D12Resource* _bandA = nullptr;  // half resolution, 32-bit float: moments, then coefficients
    ID3D12Resource* _bandB = nullptr;
    ID3D12Resource* _blocks = nullptr; // one texel a 32x32 block: the black restore's measurement
    unsigned int _width = 0;
    unsigned int _height = 0;
    DXGI_FORMAT _format = DXGI_FORMAT_UNKNOWN;
    bool _bandsFull = false;
    unsigned int _failedWidth = 0;
    unsigned int _failedHeight = 0;
    DXGI_FORMAT _failedFormat = DXGI_FORMAT_UNKNOWN;

    // Replaced textures may still be read by frames in flight, so they are released some calls later.
    struct Retired
    {
        ID3D12Resource* res = nullptr;
        unsigned int age = 0;
    };
    std::vector<Retired> _retired;

    bool _loggedFirst = false;
};
