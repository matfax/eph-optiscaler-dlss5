#include "pch.h"

#include "DlssNr_Look_Dx12.h"

#include <Config.h>

#include <algorithm>
#include <cmath>

#include "precompile/DlssNr_Look_Shader.h"

namespace
{
constexpr D3D12_RESOURCE_STATES kUav = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
constexpr D3D12_RESOURCE_STATES kSrv = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

// Frames a replaced texture is kept for, so nothing still in flight reads a released one.
constexpr unsigned int kRetireAfter = 8;

void Transition(ID3D12GraphicsCommandList* cmd, ID3D12Resource* res, D3D12_RESOURCE_STATES from,
                D3D12_RESOURCE_STATES to)
{
    if (res == nullptr || from == to)
        return;

    D3D12_RESOURCE_BARRIER b {};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = res;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = from;
    b.Transition.StateAfter = to;
    cmd->ResourceBarrier(1, &b);
}

// Every pass reads what the one before it wrote, through UAVs only.
void UavBarrierAll(ID3D12GraphicsCommandList* cmd)
{
    D3D12_RESOURCE_BARRIER b {};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    b.UAV.pResource = nullptr;
    cmd->ResourceBarrier(1, &b);
}

ID3D12Resource* CreateTexture(ID3D12Device* device, DXGI_FORMAT format, unsigned int width, unsigned int height)
{
    D3D12_HEAP_PROPERTIES heap {};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;

    D3D12_RESOURCE_DESC desc {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = std::max(width, 1u);
    desc.Height = std::max(height, 1u);
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    ID3D12Resource* res = nullptr;

    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, kUav, nullptr, IID_PPV_ARGS(&res))))
        return nullptr;

    return res;
}

unsigned int Groups(unsigned int n) { return (n + 7u) / 8u; }
} // namespace

DlssNrLookSettings DlssNrLookSettings::FromConfig(const Config& cfg)
{
    DlssNrLookSettings s;
    s.strength = std::clamp(cfg.DlssNrLookStrength.value_or_default(), 0.0f, 2.0f);
    s.brighten = std::clamp(cfg.DlssNrLookBrighten.value_or_default(), 0.0f, 2.0f);
    s.darken = std::clamp(cfg.DlssNrLookDarken.value_or_default(), 0.0f, 2.0f);
    s.maxBrighten = std::clamp(cfg.DlssNrLookMaxBrighten.value_or_default(), 0.0f, 4.0f);
    s.maxDarken = std::clamp(cfg.DlssNrLookMaxDarken.value_or_default(), 0.0f, 4.0f);
    s.colour = std::clamp(cfg.DlssNrLookColour.value_or_default(), 0.0f, 2.0f);
    s.hue = std::clamp(cfg.DlssNrLookHue.value_or_default(), 0.0f, 2.0f);
    s.maxColour = std::clamp(cfg.DlssNrLookMaxColour.value_or_default(), 0.0f, 2.0f);
    s.shadows = std::clamp(cfg.DlssNrLookShadows.value_or_default(), 0.0f, 2.0f);
    s.midtones = std::clamp(cfg.DlssNrLookMidtones.value_or_default(), 0.0f, 2.0f);
    s.highlights = std::clamp(cfg.DlssNrLookHighlights.value_or_default(), 0.0f, 2.0f);
    s.tone = std::clamp(cfg.DlssNrLookTone.value_or_default(), 0.0f, 2.0f);
    s.detail = std::clamp(cfg.DlssNrLookDetail.value_or_default(), 0.0f, 2.0f);
    s.detailRadius = std::clamp(cfg.DlssNrLookDetailRadius.value_or_default(), 0.25f, 4.0f);
    s.shadowMaxBrighten = std::clamp(cfg.DlssNrLookShadowMaxBrighten.value_or_default(), 0.0f, 4.0f);
    s.shadowMaxDarken = std::clamp(cfg.DlssNrLookShadowMaxDarken.value_or_default(), 0.0f, 4.0f);
    s.blackRestore = std::min(cfg.DlssNrLookBlackRestore.value_or_default(), 2u);
    return s;
}

bool DlssNrLookSettings::Neutral() const
{
    return strength == 1.0f && brighten == 1.0f && darken == 1.0f && maxBrighten <= 0.0f && maxDarken <= 0.0f &&
           colour == 1.0f && hue == 1.0f && maxColour <= 0.0f && shadows == 1.0f && midtones == 1.0f &&
           highlights == 1.0f && tone == 1.0f && detail == 1.0f && shadowMaxBrighten <= 0.0f &&
           shadowMaxDarken <= 0.0f && blackRestore == 0;
}

// The detail radius is a share of the model's height, and the box runs on the half-resolution grid.
unsigned int DlssNrLook_Dx12::BandRadius(float detailRadiusPercent, unsigned int height)
{
    const float halfPixels = detailRadiusPercent * 0.01f * (float) height * 0.5f;
    return std::clamp((unsigned int) std::lround(halfPixels), 1u, kMaxRadius);
}

DlssNrLook_Dx12::DlssNrLook_Dx12(ID3D12Device* device) : Shader_Dx12("DLSS-NR look", device)
{
    if (device == nullptr)
        return;

    if (!SetupRootSignature(device, kSrvCount, kUavCount, 1))
    {
        LOG_ERROR("[{0}] Failed to set up the root signature", _name);
        return;
    }

    D3D12_RESOURCE_DESC desc = CD3DX12_RESOURCE_DESC::Buffer(sizeof(DlssNrLookConstants));
    auto heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);

    for (auto& cb : _constantBuffers)
    {
        if (FAILED(device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &desc,
                                                   D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&cb))))
        {
            LOG_ERROR("[{0}] Could not create a constant buffer", _name);
            return;
        }
    }

    if (!CreateComputePipeline(device, &_pipelineState, DlssNrLook_cso, sizeof(DlssNrLook_cso), nullptr))
    {
        LOG_ERROR("[{0}] Failed to create the compute pipeline", _name);
        return;
    }

    _init = InitHeaps(device, _frameHeaps, DLSSNR_LOOK_NUM_OF_HEAPS);

    if (_init)
        LOG_INFO("DLSS-NR look: shader ready");
}

DlssNrLook_Dx12::~DlssNrLook_Dx12()
{
    ReleaseResources();

    for (auto& r : _retired)
        SAFE_RELEASE(r.res);

    _retired.clear();

    for (auto& cb : _constantBuffers)
        SAFE_RELEASE(cb);
}

void DlssNrLook_Dx12::Retire(ID3D12Resource*& res)
{
    if (res != nullptr)
        _retired.push_back({ res, 0 });

    res = nullptr;
}

void DlssNrLook_Dx12::TickRetired()
{
    for (auto& r : _retired)
    {
        if (++r.age > kRetireAfter)
            SAFE_RELEASE(r.res);
    }

    _retired.erase(std::remove_if(_retired.begin(), _retired.end(), [](const Retired& r) { return r.res == nullptr; }),
                   _retired.end());
}

void DlssNrLook_Dx12::ReleaseResources()
{
    SAFE_RELEASE(_output);
    SAFE_RELEASE(_bandA);
    SAFE_RELEASE(_bandB);
    SAFE_RELEASE(_blocks);
    _width = 0;
    _height = 0;
    _format = DXGI_FORMAT_UNKNOWN;
}

bool DlssNrLook_Dx12::EnsureResources(ID3D12Device* device, DXGI_FORMAT format, unsigned int width, unsigned int height,
                                      bool bands)
{
    if (_width != width || _height != height || _format != format)
    {
        // A size and format that could not be allocated is not retried every frame.
        if (_failedWidth == width && _failedHeight == height && _failedFormat == format)
            return false;

        if (_width != 0)
            LOG_INFO("DLSS-NR look: {}x{} -> {}x{}, rebuilding", _width, _height, width, height);

        Retire(_output);
        Retire(_blocks);
        Retire(_bandA);
        Retire(_bandB);

        _output = CreateTexture(device, format, width, height);
        _blocks = CreateTexture(device, DXGI_FORMAT_R32G32B32A32_FLOAT, (width + kBlock - 1) / kBlock,
                                (height + kBlock - 1) / kBlock);
        _width = width;
        _height = height;
        _format = format;
    }

    // The band textures are full size only while tone or detail needs them, and a stand-in texel otherwise. 32-bit,
    // because the variance is a difference of two nearly equal means of squares.
    if (_output != nullptr && _blocks != nullptr && (_bandA == nullptr || _bandsFull != bands))
    {
        Retire(_bandA);
        Retire(_bandB);
        const unsigned int bandWidth = bands ? (width + 1) / 2 : 1;
        const unsigned int bandHeight = bands ? (height + 1) / 2 : 1;
        _bandA = CreateTexture(device, DXGI_FORMAT_R32G32B32A32_FLOAT, bandWidth, bandHeight);
        _bandB = CreateTexture(device, DXGI_FORMAT_R32G32B32A32_FLOAT, bandWidth, bandHeight);
        _bandsFull = bands;
    }

    if (_output == nullptr || _blocks == nullptr || _bandA == nullptr || _bandB == nullptr)
    {
        LOG_ERROR("DLSS-NR look: could not allocate its textures at {}x{} (format {}), the model's answer is used as "
                  "it is",
                  width, height, (int) format);
        Retire(_output);
        Retire(_blocks);
        Retire(_bandA);
        Retire(_bandB);
        _failedWidth = width;
        _failedHeight = height;
        _failedFormat = format;
        _width = 0;
        _height = 0;
        _format = DXGI_FORMAT_UNKNOWN;
        return false;
    }

    _failedWidth = 0;
    _failedHeight = 0;
    _failedFormat = DXGI_FORMAT_UNKNOWN;
    return true;
}

bool DlssNrLook_Dx12::Pass(ID3D12GraphicsCommandList* cmd, const DlssNrLookConstants& constants, ID3D12Resource* proxy,
                           ID3D12Resource* model, unsigned int groupsX, unsigned int groupsY)
{
    const uint32_t slot = _heapIndex;
    _heapIndex = (_heapIndex + 1) % DLSSNR_LOOK_NUM_OF_HEAPS;
    FrameDescriptorHeap& heap = _frameHeaps[slot];

    CreateShaderResourceView(_device, proxy, heap.GetSrvCPU(0));
    CreateShaderResourceView(_device, model, heap.GetSrvCPU(1));
    CreateUnorderedAccessView(_device, _output, heap.GetUavCPU(0), 0);
    CreateUnorderedAccessView(_device, _bandA, heap.GetUavCPU(1), 0);
    CreateUnorderedAccessView(_device, _bandB, heap.GetUavCPU(2), 0);
    CreateUnorderedAccessView(_device, _blocks, heap.GetUavCPU(3), 0);

    if (!CreateConstantsBuffer(_device, _constantBuffers[slot], constants, heap.GetCbvCPU(0)))
        return false;

    ID3D12DescriptorHeap* heaps[] = { heap.GetHeapCSU() };
    cmd->SetDescriptorHeaps(_countof(heaps), heaps);
    cmd->SetComputeRootSignature(_rootSignature);
    cmd->SetPipelineState(_pipelineState);
    cmd->SetComputeRootDescriptorTable(0, heap.GetTableGPUStart());
    cmd->Dispatch(std::max(groupsX, 1u), std::max(groupsY, 1u), 1);
    return true;
}

ID3D12Resource* DlssNrLook_Dx12::Apply(ID3D12GraphicsCommandList* cmd, ID3D12Device* device,
                                       const DlssNrLookSettings& look, ID3D12Resource* proxy, ID3D12Resource* model,
                                       unsigned int width, unsigned int height, bool passthrough)
{
    TickRetired();

    // The black restore is an SDR pass, as in RenoDX: on a passthrough (already tone-mapped) frame only.
    DlssNrLookSettings settings = look;

    if (!passthrough)
        settings.blackRestore = 0;

    if (!_init || cmd == nullptr || device == nullptr || proxy == nullptr || model == nullptr || width == 0 ||
        height == 0 || settings.Neutral())
        return nullptr;

    // Both are read pixel for pixel at the working resolution.
    const D3D12_RESOURCE_DESC pd = proxy->GetDesc();
    const D3D12_RESOURCE_DESC md = model->GetDesc();

    if (pd.Width != width || pd.Height != height || md.Width != width || md.Height != height)
        return nullptr;

    if (!EnsureResources(device, md.Format, width, height, settings.Bands()))
        return nullptr;

    DlssNrLookConstants c {};
    c.Width = width;
    c.Height = height;
    c.HalfWidth = (width + 1) / 2;
    c.HalfHeight = (height + 1) / 2;
    c.Radius = BandRadius(settings.detailRadius, height);
    c.Passthrough = passthrough ? 1 : 0;
    c.Bands = settings.Bands() ? 1 : 0;
    c.Strength = settings.strength;
    c.Brighten = settings.brighten;
    c.Darken = settings.darken;
    c.MaxBrighten = settings.maxBrighten;
    c.MaxDarken = settings.maxDarken;
    c.Colour = settings.colour;
    c.Hue = settings.hue;
    c.MaxColour = settings.maxColour;
    c.Shadows = settings.shadows;
    c.Midtones = settings.midtones;
    c.Highlights = settings.highlights;
    c.Tone = settings.tone;
    c.Detail = settings.detail;
    c.GfEps = kGfEps;
    c.ShadowMaxBrighten = settings.shadowMaxBrighten;
    c.ShadowMaxDarken = settings.shadowMaxDarken;
    c.BlackRestore = settings.blackRestore;
    c.BlocksX = (width + kBlock - 1) / kBlock;
    c.BlocksY = (height + kBlock - 1) / kBlock;

    if (!_loggedFirst)
    {
        _loggedFirst = true;
        LOG_INFO("DLSS-NR look: active at {}x{}, band radius {} (detail radius {:.2f}%), tone {:.2f}, detail {:.2f}, "
                 "shadow limits +{:.2f}/-{:.2f} stops, black restore {}",
                 width, height, c.Radius, settings.detailRadius, settings.tone, settings.detail,
                 settings.shadowMaxBrighten, settings.shadowMaxDarken, settings.blackRestore);
    }

    Transition(cmd, model, kUav, kSrv);

    bool ok = true;

    if (c.Bands != 0)
    {
        for (uint32_t mode = 0; mode < 5 && ok; ++mode)
        {
            c.Mode = mode;
            ok = Pass(cmd, c, proxy, model, Groups(c.HalfWidth), Groups(c.HalfHeight));
            UavBarrierAll(cmd);
        }
    }

    // The black lift is measured on the answer as shaped (after the band passes, which it reads), so a lift the
    // shadow limits already held back is not taken off twice.
    if (ok && c.BlackRestore != 0)
    {
        c.Mode = 6;
        ok = Pass(cmd, c, proxy, model, c.BlocksX, c.BlocksY);
        UavBarrierAll(cmd);
    }

    if (ok)
    {
        c.Mode = 5;
        ok = Pass(cmd, c, proxy, model, Groups(width), Groups(height));
        UavBarrierAll(cmd);
    }

    Transition(cmd, model, kSrv, kUav);

    if (!ok)
    {
        LOG_ERROR("DLSS-NR look: a pass could not be recorded, the model's answer is used as it is");
        return nullptr;
    }

    return _output;
}
