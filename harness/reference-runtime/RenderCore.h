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

#pragma once

#include <AudioToolbox/AudioToolbox.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RenderCore RenderCore;

typedef struct RenderCoreReport {
  uint32_t channels;
  uint32_t capacityFrames;
  uint32_t availableFrames;
  uint64_t pushedFrames;
  uint64_t pulledFrames;
  uint64_t droppedFrames;
  uint64_t discardedFrames;
  uint64_t underrunFrames;
  uint64_t underrunEvents;
  uint64_t renderCallbacks;
  uint32_t minFillFrames;
  uint32_t maxFillFrames;
} RenderCoreReport;

RenderCore *render_core_create(uint32_t channels, uint32_t capacityFrames);
void render_core_destroy(RenderCore *core);

uint32_t render_core_prime_silence(RenderCore *core, uint32_t frames);
uint32_t render_core_push_deterministic(RenderCore *core, uint32_t frames, uint32_t seed);
uint32_t render_core_render_decoder_stft(
    RenderCore *core,
    const float *stft,
    uint32_t stftFrames,
    float *interleavedPcm,
    uint32_t capacityFrames,
    uint32_t channels);
uint32_t render_core_push_decoder_stft(RenderCore *core, const float *stft, uint32_t stftFrames);
uint32_t render_core_push_last_frame(RenderCore *core, uint32_t frames);
uint32_t render_core_pull_abl(RenderCore *core, AudioBufferList *audioBufferList, uint32_t frames);
uint32_t render_core_available_frames(RenderCore *core);
uint32_t render_core_capacity_frames(RenderCore *core);
uint32_t render_core_discard_to_frames(RenderCore *core, uint32_t targetFrames);
uint32_t render_core_discard_to_frames_with_fade(
    RenderCore *core,
    uint32_t targetFrames,
    uint32_t fadeInFrames);
void render_core_reset_fill_watermarks(RenderCore *core);

RenderCoreReport render_core_report(RenderCore *core);

#ifdef __cplusplus
}
#endif
