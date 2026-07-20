// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "RenderCore.h"

#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <memory>
#include <new>
#include <vector>

namespace {

constexpr uint32_t kSpectroStreamBins = 480;
constexpr uint32_t kSpectroStreamChannels = 4;
constexpr uint32_t kAudioChannels = 2;
constexpr uint32_t kFftLength = 960;
constexpr uint32_t kFrameLength = 960;
constexpr uint32_t kFrameStep = 480;
constexpr float kPi = 3.14159265358979323846f;

}  // namespace

struct RenderCore {
  explicit RenderCore(uint32_t channelCount, uint32_t requestedCapacityFrames)
      : channels(std::max<uint32_t>(1, channelCount)) {
    const size_t requestedSamples =
        static_cast<size_t>(std::max<uint32_t>(1, requestedCapacityFrames)) * channels + 1;
    capacitySamples = 1;
    while (capacitySamples < requestedSamples) {
      capacitySamples <<= 1;
    }
    mask = capacitySamples - 1;
    capacityFrames = static_cast<uint32_t>((capacitySamples - 1) / channels);
    minFillFrames.store(capacityFrames, std::memory_order_relaxed);
    samples.reset(new (std::nothrow) float[capacitySamples]);
    if (samples) {
      std::fill(samples.get(), samples.get() + capacitySamples, 0.0f);
    }
    overlap.assign(static_cast<size_t>(kAudioChannels) * kFrameStep, 0.0f);
    inverseWindow.resize(kFrameLength);
    for (uint32_t n = 0; n < kFrameLength; ++n) {
      // Match sequence_layers InverseSTFT exactly: the decoder's complex
      // frames receive one periodic Hann synthesis window, then overlap-add.
      // A dual-window normalization is appropriate only when a matching
      // analysis window is also present; applying it here imposed a periodic
      // gain envelope that is absent from the trained SpectroStream decoder.
      inverseWindow[n] =
          0.5f - 0.5f * std::cos((2.0f * kPi * static_cast<float>(n)) / kFrameLength);
    }
#if defined(CROSSFADE_LEGACY_DUAL_SYNTHESIS_WINDOW)
    // Reproducibility-only build of the pre-fix renderer used by the paper's
    // crossover. Production builds never define this macro.
    std::vector<float> denominator(kFrameStep, 0.0f);
    for (uint32_t n = 0; n < kFrameLength; ++n) {
      denominator[n % kFrameStep] += inverseWindow[n] * inverseWindow[n];
    }
    for (uint32_t n = 0; n < kFrameLength; ++n) {
      const float denom = denominator[n % kFrameStep];
      inverseWindow[n] = denom == 0.0f ? 0.0f : inverseWindow[n] / denom;
    }
#endif
    dftInputReal.assign(kFftLength, 0.0f);
    dftInputImag.assign(kFftLength, 0.0f);
    dftOutputReal.assign(kFftLength, 0.0f);
    dftOutputImag.assign(kFftLength, 0.0f);
    inverseDft = vDSP_DFT_zop_CreateSetup(nullptr, kFftLength, vDSP_DFT_INVERSE);
  }

  ~RenderCore() {
    if (inverseDft) {
      vDSP_DFT_DestroySetup(inverseDft);
    }
  }

  uint32_t channels = 2;
  uint32_t capacityFrames = 0;
  size_t capacitySamples = 0;
  size_t mask = 0;
  std::unique_ptr<float[]> samples;
  std::vector<float> overlap;
  std::vector<float> inverseWindow;
  std::vector<float> dftInputReal;
  std::vector<float> dftInputImag;
  std::vector<float> dftOutputReal;
  std::vector<float> dftOutputImag;
  float lastFrame[8] = {0.0f};
  vDSP_DFT_Setup inverseDft = nullptr;

  alignas(64) std::atomic<size_t> head{0};
  alignas(64) std::atomic<size_t> tail{0};

  std::atomic<uint64_t> pushedFrames{0};
  std::atomic<uint64_t> pulledFrames{0};
  std::atomic<uint64_t> droppedFrames{0};
  std::atomic<uint64_t> discardedFrames{0};
  std::atomic<uint64_t> underrunFrames{0};
  std::atomic<uint64_t> underrunEvents{0};
  std::atomic<uint64_t> renderCallbacks{0};
  std::atomic<uint32_t> minFillFrames{0};
  std::atomic<uint32_t> maxFillFrames{0};
};

static uint32_t available_frames(const RenderCore *core) {
  const size_t h = core->head.load(std::memory_order_acquire);
  const size_t t = core->tail.load(std::memory_order_relaxed);
  return static_cast<uint32_t>(((h - t) & core->mask) / core->channels);
}

static void update_min_fill(RenderCore *core, uint32_t fillFrames) {
  uint32_t current = core->minFillFrames.load(std::memory_order_relaxed);
  while (fillFrames < current &&
         !core->minFillFrames.compare_exchange_weak(
             current, fillFrames, std::memory_order_relaxed, std::memory_order_relaxed)) {
  }
}

static void update_max_fill(RenderCore *core, uint32_t fillFrames) {
  uint32_t current = core->maxFillFrames.load(std::memory_order_relaxed);
  while (fillFrames > current &&
         !core->maxFillFrames.compare_exchange_weak(
             current, fillFrames, std::memory_order_relaxed, std::memory_order_relaxed)) {
  }
}

static void apply_fade_in(RenderCore *core, size_t startSample, uint32_t frames) {
  if (!core || frames == 0) {
    return;
  }
  const uint32_t channels = core->channels;
  for (uint32_t frameIndex = 0; frameIndex < frames; ++frameIndex) {
    const float gain = static_cast<float>(frameIndex + 1) / static_cast<float>(frames);
    const size_t frameStart = startSample + static_cast<size_t>(frameIndex) * channels;
    for (uint32_t channel = 0; channel < channels; ++channel) {
      const size_t sampleIndex = (frameStart + channel) & core->mask;
      core->samples[sampleIndex] *= gain;
    }
  }
}

static uint32_t write_frame(RenderCore *core, const float *frame) {
  size_t h = core->head.load(std::memory_order_relaxed);
  const size_t t = core->tail.load(std::memory_order_acquire);
  const size_t used = (h - t) & core->mask;
  const size_t freeSamples = core->capacitySamples - 1 - used;
  if (freeSamples < core->channels) {
    core->droppedFrames.fetch_add(1, std::memory_order_relaxed);
    return 0;
  }
  for (uint32_t channel = 0; channel < core->channels; ++channel) {
    core->samples[(h + channel) & core->mask] = frame[channel];
    if (channel < 8) {
      core->lastFrame[channel] = frame[channel];
    }
  }
  core->head.store(h + core->channels, std::memory_order_release);
  core->pushedFrames.fetch_add(1, std::memory_order_relaxed);
  update_max_fill(core, static_cast<uint32_t>(((used + core->channels) & core->mask) / core->channels));
  return 1;
}

static void inverse_stft_frame(RenderCore *core, const float *stftFrame, float *left, float *right) {
  if (!core->inverseDft) {
    std::fill(left, left + kFrameLength, 0.0f);
    std::fill(right, right + kFrameLength, 0.0f);
    return;
  }

  float *outputs[kAudioChannels] = {left, right};
  for (uint32_t channel = 0; channel < kAudioChannels; ++channel) {
    std::fill(core->dftInputReal.begin(), core->dftInputReal.end(), 0.0f);
    std::fill(core->dftInputImag.begin(), core->dftInputImag.end(), 0.0f);

    for (uint32_t bin = 0; bin < kSpectroStreamBins; ++bin) {
      const float *binValues = stftFrame + static_cast<size_t>(bin) * kSpectroStreamChannels;
      const float real = binValues[channel * 2];
      const float imag = binValues[channel * 2 + 1];
      core->dftInputReal[bin] = real;
      core->dftInputImag[bin] = imag;
      if (bin > 0) {
        const uint32_t mirror = kFftLength - bin;
        core->dftInputReal[mirror] = real;
        core->dftInputImag[mirror] = -imag;
      }
    }
    vDSP_DFT_Execute(
        core->inverseDft,
        core->dftInputReal.data(),
        core->dftInputImag.data(),
        core->dftOutputReal.data(),
        core->dftOutputImag.data());
    for (uint32_t n = 0; n < kFrameLength; ++n) {
      outputs[channel][n] =
          (core->dftOutputReal[n] / static_cast<float>(kFftLength)) * core->inverseWindow[n];
    }
  }
}

RenderCore *render_core_create(uint32_t channels, uint32_t capacityFrames) {
  std::unique_ptr<RenderCore> core(new (std::nothrow) RenderCore(channels, capacityFrames));
  if (!core || !core->samples) {
    return nullptr;
  }
  return core.release();
}

void render_core_destroy(RenderCore *core) {
  delete core;
}

uint32_t render_core_prime_silence(RenderCore *core, uint32_t frames) {
  if (!core) {
    return 0;
  }
  float frame[8] = {0.0f};
  uint32_t written = 0;
  for (uint32_t frameIndex = 0; frameIndex < frames; ++frameIndex) {
    written += write_frame(core, frame);
  }
  update_min_fill(core, available_frames(core));
  return written;
}

uint32_t render_core_push_deterministic(RenderCore *core, uint32_t frames, uint32_t seed) {
  if (!core) {
    return 0;
  }
  float frame[8] = {0.0f};
  const uint32_t channels = std::min<uint32_t>(core->channels, 8);
  uint32_t written = 0;
  uint32_t state = seed == 0 ? 1 : seed;
  for (uint32_t frameIndex = 0; frameIndex < frames; ++frameIndex) {
    state = state * 1664525u + 1013904223u;
    const float value = (static_cast<float>((state >> 8) & 0xffffu) / 32768.0f - 1.0f) * 0.02f;
    for (uint32_t channel = 0; channel < channels; ++channel) {
      frame[channel] = value;
    }
    written += write_frame(core, frame);
  }
  update_min_fill(core, available_frames(core));
  return written;
}

uint32_t render_core_render_decoder_stft(
    RenderCore *core,
    const float *stft,
    uint32_t stftFrames,
    float *interleavedPcm,
    uint32_t capacityFrames,
    uint32_t channels) {
  if (!core || !stft || !interleavedPcm || channels < kAudioChannels) {
    return 0;
  }

  std::vector<float> left(kFrameLength, 0.0f);
  std::vector<float> right(kFrameLength, 0.0f);
  uint32_t rendered = 0;
  for (uint32_t stftFrame = 0; stftFrame < stftFrames; ++stftFrame) {
    if (rendered + kFrameStep > capacityFrames) {
      break;
    }
    const float *stftBase =
        stft + static_cast<size_t>(stftFrame) * kSpectroStreamBins * kSpectroStreamChannels;
    inverse_stft_frame(core, stftBase, left.data(), right.data());

    for (uint32_t sample = 0; sample < kFrameStep; ++sample) {
      const size_t outputIndex = static_cast<size_t>(rendered + sample) * channels;
      interleavedPcm[outputIndex] = core->overlap[sample] + left[sample];
      interleavedPcm[outputIndex + 1] = core->overlap[kFrameStep + sample] + right[sample];
      for (uint32_t channel = kAudioChannels; channel < channels; ++channel) {
        interleavedPcm[outputIndex + channel] = interleavedPcm[outputIndex + 1];
      }
    }

    for (uint32_t sample = 0; sample < kFrameStep; ++sample) {
      core->overlap[sample] = left[kFrameStep + sample];
      core->overlap[kFrameStep + sample] = right[kFrameStep + sample];
    }
    rendered += kFrameStep;
  }
  return rendered;
}

uint32_t render_core_push_decoder_stft(RenderCore *core, const float *stft, uint32_t stftFrames) {
  if (!core || !stft || core->channels < kAudioChannels) {
    return 0;
  }

  const uint32_t pcmFrames = stftFrames * kFrameStep;
  std::vector<float> pcm(static_cast<size_t>(pcmFrames) * core->channels, 0.0f);
  const uint32_t rendered = render_core_render_decoder_stft(
      core,
      stft,
      stftFrames,
      pcm.data(),
      pcmFrames,
      core->channels);
  float frame[8] = {0.0f};
  uint32_t written = 0;
  for (uint32_t frameIndex = 0; frameIndex < rendered; ++frameIndex) {
    const size_t inputIndex = static_cast<size_t>(frameIndex) * core->channels;
    for (uint32_t channel = 0; channel < std::min<uint32_t>(core->channels, 8); ++channel) {
      frame[channel] = pcm[inputIndex + channel];
    }
    written += write_frame(core, frame);
  }
  update_min_fill(core, available_frames(core));
  return written;
}

uint32_t render_core_push_last_frame(RenderCore *core, uint32_t frames) {
  if (!core) {
    return 0;
  }
  uint32_t written = 0;
  for (uint32_t frameIndex = 0; frameIndex < frames; ++frameIndex) {
    written += write_frame(core, core->lastFrame);
  }
  update_min_fill(core, available_frames(core));
  return written;
}

uint32_t render_core_pull_abl(RenderCore *core, AudioBufferList *audioBufferList, uint32_t frames) {
  if (!core || !audioBufferList) {
    return 0;
  }

  core->renderCallbacks.fetch_add(1, std::memory_order_relaxed);
  uint32_t pulled = 0;
  for (uint32_t frameIndex = 0; frameIndex < frames; ++frameIndex) {
    const size_t h = core->head.load(std::memory_order_acquire);
    size_t t = core->tail.load(std::memory_order_relaxed);
    const size_t used = (h - t) & core->mask;
    if (used < core->channels) {
      core->underrunEvents.fetch_add(1, std::memory_order_relaxed);
      core->underrunFrames.fetch_add(frames - frameIndex, std::memory_order_relaxed);
      for (uint32_t rest = frameIndex; rest < frames; ++rest) {
        for (uint32_t channel = 0; channel < audioBufferList->mNumberBuffers; ++channel) {
          auto *destination = static_cast<float *>(audioBufferList->mBuffers[channel].mData);
          if (destination) {
            destination[rest] = 0.0f;
          }
        }
      }
      update_min_fill(core, 0);
      return pulled;
    }

    for (uint32_t channel = 0; channel < audioBufferList->mNumberBuffers; ++channel) {
      auto *destination = static_cast<float *>(audioBufferList->mBuffers[channel].mData);
      if (!destination) {
        continue;
      }
      const uint32_t sourceChannel = std::min<uint32_t>(channel, core->channels - 1);
      destination[frameIndex] = core->samples[(t + sourceChannel) & core->mask];
    }
    core->tail.store(t + core->channels, std::memory_order_release);
    ++pulled;
  }

  core->pulledFrames.fetch_add(pulled, std::memory_order_relaxed);
  update_min_fill(core, available_frames(core));
  return pulled;
}

RenderCoreReport render_core_report(RenderCore *core) {
  RenderCoreReport report = {};
  if (!core) {
    return report;
  }
  report.channels = core->channels;
  report.capacityFrames = core->capacityFrames;
  report.availableFrames = available_frames(core);
  report.pushedFrames = core->pushedFrames.load(std::memory_order_relaxed);
  report.pulledFrames = core->pulledFrames.load(std::memory_order_relaxed);
  report.droppedFrames = core->droppedFrames.load(std::memory_order_relaxed);
  report.discardedFrames = core->discardedFrames.load(std::memory_order_relaxed);
  report.underrunFrames = core->underrunFrames.load(std::memory_order_relaxed);
  report.underrunEvents = core->underrunEvents.load(std::memory_order_relaxed);
  report.renderCallbacks = core->renderCallbacks.load(std::memory_order_relaxed);
  report.minFillFrames = core->minFillFrames.load(std::memory_order_relaxed);
  report.maxFillFrames = core->maxFillFrames.load(std::memory_order_relaxed);
  return report;
}

uint32_t render_core_available_frames(RenderCore *core) {
  if (!core) {
    return 0;
  }
  return available_frames(core);
}

uint32_t render_core_capacity_frames(RenderCore *core) {
  if (!core) {
    return 0;
  }
  return core->capacityFrames;
}

uint32_t render_core_discard_to_frames(RenderCore *core, uint32_t targetFrames) {
  return render_core_discard_to_frames_with_fade(core, targetFrames, 0);
}

uint32_t render_core_discard_to_frames_with_fade(
    RenderCore *core,
    uint32_t targetFrames,
    uint32_t fadeInFrames) {
  if (!core) {
    return 0;
  }
  const uint32_t boundedTarget = std::min<uint32_t>(targetFrames, core->capacityFrames);
  while (true) {
    const size_t h = core->head.load(std::memory_order_acquire);
    size_t t = core->tail.load(std::memory_order_acquire);
    const uint32_t available = static_cast<uint32_t>(((h - t) & core->mask) / core->channels);
    if (available <= boundedTarget) {
      update_min_fill(core, available);
      return 0;
    }
    const uint32_t discardFrames = available - boundedTarget;
    const size_t desiredTail = t + static_cast<size_t>(discardFrames) * core->channels;
    const uint32_t retainedFadeFrames = std::min<uint32_t>(fadeInFrames, boundedTarget);
    apply_fade_in(core, desiredTail, retainedFadeFrames);
    if (core->tail.compare_exchange_weak(
            t, desiredTail, std::memory_order_release, std::memory_order_acquire)) {
      core->discardedFrames.fetch_add(discardFrames, std::memory_order_relaxed);
      update_min_fill(core, boundedTarget);
      return discardFrames;
    }
  }
}

void render_core_reset_fill_watermarks(RenderCore *core) {
  if (!core) {
    return;
  }
  const uint32_t fillFrames = available_frames(core);
  core->minFillFrames.store(fillFrames, std::memory_order_relaxed);
  core->maxFillFrames.store(fillFrames, std::memory_order_relaxed);
}
