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

import CoreML
import Foundation

/// Result of one full MRT2 generation burst (25 temporal frames of audio).
///
/// `checksum` is a deterministic token/STFT digest used by warmup paths and
/// proof harnesses to detect silent numerical drift. `timing` carries
/// producer-side per-stage latency for p50/p99 analysis; it is never measured
/// on the Core Audio render callback.
public struct CrossfadeGenerationResult: Sendable {
  public let checksum: Int
  public let timing: CrossfadeGenerationTiming

  public init(checksum: Int, timing: CrossfadeGenerationTiming) {
    self.checksum = checksum
    self.timing = timing
  }
}

/// Canonical Core ML model and resource names for the Crossfade runtime.
///
/// `CrossfadeAudioEngine.preflight` (CrossfadeAudioEngine.swift) and
/// `CrossfadeGenerationRuntime` share this single source of truth so bundle
/// validation and model loading can never disagree about which `.mlmodelc`
/// directories a configuration requires.
enum CrossfadeRuntimeResources {
  /// Rolling stateful temporal step (exported by
  /// `scripts/convert_mrt2_temporal_body_rolling_coreml.py`). Replaces
  /// `mrt2_temporal_body_unrolled_01`, whose `slot_index=0` attention bias
  /// masked all 41 history slots: its K/V state was written every prediction
  /// but never read — feeding a frame on a fresh `MLState` vs after prior
  /// frames produced bit-identical output, so the temporal transformer ran
  /// history-blind on every device build (the residual "texture roughness vs
  /// MLX"). The rolling graph keeps a true sliding 41-frame window: K/V
  /// caches roll left one frame per prediction and a `[1, 1]` step-counter
  /// state masks slots not yet written. `model.makeState()` covers the extra
  /// counter state automatically; the host must NEVER pre-fill state buffers
  /// (zeroed state + counter-derived masking IS the correct cold start).
  static let statefulOneFrameTemporalModelName = "mrt2_temporal_body_rolling_01"
  /// Pure-function one-frame temporal step. All 48 K/V arrays are ordinary
  /// inputs, all 48 updates are ordinary outputs, and Swift owns the rolling
  /// 41-frame cache. Exported by the public
  /// `exporters/convert_temporal_body_carry.py --streaming --frames 1` path.
  static let streamingCarryTemporalModelName = "mrt2_temporal_body_streaming_carry_01"
  static let temporalSchedule: [(resourceName: String, frames: Int, history: Int)] = [
    ("mrt2_temporal_body_carry_04", 4, 0),
    ("mrt2_temporal_body_carry_04_h04", 4, 4),
    ("mrt2_temporal_body_carry_04_h08", 4, 8),
    ("mrt2_temporal_body_carry_04_h12", 4, 12),
    ("mrt2_temporal_body_carry_04_h16", 4, 16),
    ("mrt2_temporal_body_carry_04_h20", 4, 20),
    ("mrt2_temporal_body_carry_01_h24", 1, 24),
  ]
  /// Whole-frame in-graph depth rollout (exported by
  /// `scripts/convert_mrt2_depth_body_rollout_coreml.py`, FLOAT16): ONE
  /// prediction per frame samples all 12 RVQ levels inside the graph via
  /// Gumbel-max over a static top-40 set, with host-supplied noise and
  /// inverse temperature keeping RNG/seed host-owned. Replaced both
  /// 12-predictions-per-frame designs (`mrt2_depth_body_logits` 12x full
  /// pass, `mrt2_depth_body_step` 12x stateful step): the depth path is
  /// weight-BANDWIDTH-bound on phones, so every prediction streams the full
  /// ~97 MB weight set from DRAM and ANY 12-call scheme costs ~40 ms/frame —
  /// the build-20-era stutter. One FLOAT16 prediction measures 12.7 ms/frame
  /// on iPhone 12 Pro and 8.4 ms on iPhone 15 Pro Max (zero composed
  /// underruns). The FLOAT32 export of the same graph is token-for-token
  /// exact vs the reference chain (0/900 across argmax and noisy temperature
  /// arms, `scripts/validate_mrt2_depth_body_rollout_coreml.py`); FLOAT16
  /// flips fp16 near-tie tokens (distribution preserved) and passed the
  /// 2026-06-10 device quality gate.
  static let depthModelName = "mrt2_depth_body_rollout"
  static let codebookResourceStem = "spectrostream_rvq_codebooks_12_f32"
  static let codebookResourceExtension = "bin"
  static let codebookLevels = 12
  static let codebookSize = 1_024
  static let codebookEmbeddingDim = 256
  static let expectedCodebookCount = codebookLevels * codebookSize * codebookEmbeddingDim
  static let expectedCodebookBytes = expectedCodebookCount * MemoryLayout<Float>.stride
  /// Depthformer token-embedding table (exported by
  /// `scripts/export_mrt2_depth_embedder_for_ios.py`, sqrt(1024) scale baked
  /// in). Vocabulary: ids 0..5 reserved, acoustic id = 6 + level*1024 + code.
  /// Required for both autoregressive feedback paths: temporal inputs (mean of
  /// the previous frame's 12 token embeddings) and depth inputs (embeddings of
  /// the current frame's previously sampled levels).
  static let depthEmbedderResourceStem = "mrt2_depth_embedder_f32"
  static let depthEmbedderResourceExtension = "bin"
  static let depthEmbedderVocab = 6 + codebookLevels * codebookSize
  static let depthEmbedderDim = 1_024
  static let expectedDepthEmbedderBytes =
    depthEmbedderVocab * depthEmbedderDim * MemoryLayout<Float>.stride

  static var codebookResourceName: String {
    "\(codebookResourceStem).\(codebookResourceExtension)"
  }

  static var depthEmbedderResourceName: String {
    "\(depthEmbedderResourceStem).\(depthEmbedderResourceExtension)"
  }

  static func modelNames(temporalMode: CrossfadeTemporalMode, decoderInputFrames: Int) -> [String] {
    let temporalModels: [String]
    switch temporalMode {
    case .carry:
      temporalModels = temporalSchedule.map(\.resourceName)
    case .streamingCarry:
      temporalModels = [streamingCarryTemporalModelName]
    case .statefulOneFrame:
      temporalModels = [statefulOneFrameTemporalModelName]
    }
    return temporalModels + [
      depthModelName,
      decoderResourceName(inputFrames: decoderInputFrames),
    ]
  }

  static func decoderResourceName(inputFrames: Int) -> String {
    if inputFrames == 25 {
      return "spectrostream_decoder_conv_nchw"
    }
    if inputFrames == 5 {
      return "spectrostream_decoder_conv_nchw_05"
    }
    return String(format: "spectrostream_decoder_conv_gpu_%02d", inputFrames)
  }
}

/// Returns the dense (stride-collapsed) float32 values of a Core ML output.
///
/// Core ML pads some output rows for alignment — `depth_logits` ships as
/// `[1, 12, 12294]` with a row stride of 12304 — so reading `dataPointer`
/// as a contiguous buffer silently shifts every row after the first (the
/// build-19 misaligned-sampling incident; see `readSampledCodes`). This
/// helper memcpy-fast-paths dense arrays and walks `strides` otherwise.
func crossfadeDenseFloats(from array: MLMultiArray) -> [Float] {
  let shape = array.shape.map(\.intValue)
  let strides = array.strides.map(\.intValue)
  let count = shape.reduce(1, *)
  var denseStrides = [Int](repeating: 1, count: shape.count)
  for dimension in stride(from: shape.count - 2, through: 0, by: -1) {
    denseStrides[dimension] = denseStrides[dimension + 1] * shape[dimension + 1]
  }
  let isDense = strides == denseStrides
  let capacity = zip(shape, strides).reduce(1) { $0 + ($1.0 - 1) * $1.1 }
  var values = [Float](repeating: 0, count: count)
  func bufferOffset(_ flat: Int) -> Int {
    var remainder = flat
    var offset = 0
    for dimension in 0..<shape.count {
      offset += (remainder / denseStrides[dimension]) * strides[dimension]
      remainder %= denseStrides[dimension]
    }
    return offset
  }
  switch array.dataType {
  case .float32:
    let pointer = array.dataPointer.bindMemory(to: Float.self, capacity: capacity)
    if isDense {
      for index in 0..<count { values[index] = pointer[index] }
    } else {
      for index in 0..<count { values[index] = pointer[bufferOffset(index)] }
    }
  case .float16:
    let pointer = array.dataPointer.bindMemory(to: Float16.self, capacity: capacity)
    if isDense {
      for index in 0..<count { values[index] = Float(pointer[index]) }
    } else {
      for index in 0..<count { values[index] = Float(pointer[bufferOffset(index)]) }
    }
  default:
    break
  }
  return values
}

private struct CrossfadeSeededGenerator {
  private var state: UInt64

  init(seed: UInt64) {
    state = seed == 0 ? 0x9E37_79B9_7F4A_7C15 : seed
  }

  mutating func nextUnitFloat() -> Float {
    state &+= 0x9E37_79B9_7F4A_7C15
    var value = state
    value = (value ^ (value >> 30)) &* 0xBF58_476D_1CE4_E5B9
    value = (value ^ (value >> 27)) &* 0x94D0_49BB_1331_11EB
    value = value ^ (value >> 31)
    return Float(value >> 40) / Float(1 << 24)
  }
}

/// Core ML generation pipeline for one MRT2 stream: temporal transformer →
/// depth transformer → token sampling → RVQ embedding → SpectroStream decoder.
///
/// This class owns model loading, KV-cache state, and the producer-side loop
/// that turns prompt control into decoder STFT frames. It contains no audio
/// I/O: callers pass an `audioSink` closure (see `CrossfadeAudioEngine`) or
/// consume the returned checksum/timing directly (see the CoreMLMRT2Probe
/// proof harness). Everything here runs on a producer task, never on the Core
/// Audio render callback.
public final class CrossfadeGenerationRuntime {
  private struct Burst {
    let frames: Int
    let historyLength: Int
    let model: MLModel
    let sourceEncoded: MLMultiArray
    let provider: MLDictionaryFeatureProvider
  }

  private struct TemporalFixtureMetrics {
    let frames: Int
    let wrappedFrames: Int
    let correlation: Double
    let maxAbsoluteError: Double
    let meanAbsoluteError: Double
    let finiteRatio: Double
    let passed: Bool
  }

  private static let updateElementsPerFrame = 8 * 128
  private static let temporalHeads = 8
  private static let temporalHeadDim = 128
  private static let modelDim = 1024
  private static let decoderEmbeddingDim = 256
  private static let temporalFrameCount = 25
  private static let temporalWindowFrames = 41
  private static let temporalAttentionExtent = 43
  private static let invalidAttentionBias = Float16(-10_000)
  private static let tokenLevels = 12
  private static let reservedTokens = 6
  private static let codebookSize = 1_024
  private static let temporalFixtureFrames = 64
  private static let temporalFixtureInputStem =
    "temporal_streaming_carry_64_temporal_inputs_f32"
  private static let temporalFixtureSourceStem =
    "temporal_streaming_carry_64_source_encoded_f32"
  private static let temporalFixtureReferenceStem =
    "temporal_streaming_carry_64_reference_outputs_f32"

  private let configuration: CrossfadeAudioConfiguration
  private let modelBundle: Bundle
  private let resourceBundle: Bundle
  private let cacheNames: [String]
  private let cacheInputNames: [String]
  private let cacheUpdateNames: [String]
  private let cacheArrays: [MLMultiArray]
  private let bursts: [Burst]
  private let streamingCarryTemporalModel: MLModel?
  private let streamingCarryTemporalProvider: MLDictionaryFeatureProvider?
  private let streamingCarryTemporalInput: MLMultiArray?
  private let streamingCarrySourceEncoded: MLMultiArray?
  private let streamingCarryValidBias: MLMultiArray?
  private let streamingCarryModelURL: URL?
  private let statefulTemporalModel: MLModel?
  private let statefulTemporalProvider: MLDictionaryFeatureProvider?
  private let statefulTemporalState: MLState?
  private let statefulSourceEncoded: MLMultiArray?
  /// Retained handle to the stateful path's `temporal_inputs` array so the
  /// loop can write the previous frame's mean token embedding before every
  /// temporal step. (This array used to be created inline and never written —
  /// the temporal transformer ran on all-zero inputs forever, one of the two
  /// disconnected feedback loops behind the device-vs-MLX quality gap.)
  private let statefulTemporalInput: MLMultiArray?
  private let depthModel: MLModel
  /// `[1, 1, 1024]` temporal frame input for the in-graph depth rollout (the
  /// old `depth_inputs` position 0). Levels 1..11 are fed in-graph from the
  /// embedded sampled tokens; the host never writes them.
  private let depthFrameInput: MLMultiArray
  /// `[12, 1024]` per-level, per-code Gumbel(0,1) noise. The host fills this
  /// from `samplingRNG` every frame (`writeDepthGumbelNoise`), which keeps
  /// determinism host-owned: the graph is pure math, so the same seed
  /// reproduces the same token stream.
  private let depthGumbelNoise: MLMultiArray
  /// `[1]` reciprocal of the clamped sampling temperature. The graph
  /// multiplies logits by this before adding the Gumbel noise.
  private let depthInverseTemperature: MLMultiArray
  private let depthProvider: MLDictionaryFeatureProvider
  private let decoderModel: MLModel
  private let decoderInput: MLMultiArray
  private let decoderInputFrames: Int
  private let decoderContextFrames: Int
  private let decoderStrideFrames: Int
  private let decoderProvider: MLDictionaryFeatureProvider
  private var croppedDecoderSTFT: MLMultiArray?
  private var hasDecodedWindow = false
  private let rvqCodebooks: CrossfadeRVQCodebooks
  private let depthEmbedder: CrossfadeDepthEmbedder
  private let controlLock = NSLock()
  private var runtimeControl = CrossfadePromptControl()
  private var samplingRNG = CrossfadeSeededGenerator(seed: 0)
  private var pendingDecoderEmbeddings: [Float] = []
  /// Mean token embedding of the most recently generated frame — the next
  /// frame's `temporal_inputs` row, produced IN-GRAPH by the depth rollout's
  /// `temporal_feedback` output (mean of the 12 sampled token embeddings,
  /// x32 scale baked in). `nil` until the first frame: the MLX sampler's
  /// initial `previous_frame` is twelve id-0 reserved tokens with a VALID
  /// mask, so the cold-start temporal input is mean(12 x embedder row 0)
  /// from `CrossfadeDepthEmbedder` — not zeros.
  private var previousTemporalFeedback: [Float]?
  private var streamingCarryValidHistory = 0
  private var generatedTrajectoryFrames = 0
  /// Parity tap: receives each frame's 12 codebook-local tokens right after
  /// sampling. Offline harnesses use it to compare the Swift path
  /// token-for-token against the Python/Core ML reference. Sampling now runs
  /// in-graph, so determinism comes from the seeded RNG: a fixed
  /// `control.seed` reproduces the Gumbel noise stream exactly, and a Python
  /// reference must replicate `CrossfadeSeededGenerator` to predict tokens
  /// (`topK=1` no longer forces argmax — top-k is baked at 40 in the graph).
  /// Never set on the audio path.
  public var tokenTap: (([Int]) -> Void)?

  public init(
    configuration: CrossfadeAudioConfiguration,
    modelBundle: Bundle,
    resourceBundle: Bundle
  ) throws {
    Self.logAdmission("runtimeInitStarted temporalMode=\(configuration.temporalMode)")
    guard configuration.decoderInputFrames > 1 else {
      throw CrossfadeRuntimeError.invalidArgument(
        "decoderInputFrames must exceed the 1-frame SpectroStream lookahead"
      )
    }
    guard configuration.decoderContextFrames >= 0,
      configuration.decoderContextFrames < configuration.decoderInputFrames - 1
    else {
      throw CrossfadeRuntimeError.invalidArgument(
        "decoderContextFrames must be in 0..<(decoderInputFrames - 1)"
      )
    }
    self.configuration = configuration
    self.modelBundle = modelBundle
    self.resourceBundle = resourceBundle
    decoderInputFrames = configuration.decoderInputFrames
    decoderContextFrames = configuration.decoderContextFrames
    decoderStrideFrames = configuration.decoderInputFrames - 1
      - configuration.decoderContextFrames

    switch configuration.temporalMode {
    case .carry, .streamingCarry:
      cacheNames = Self.makeCacheNames()
    case .statefulOneFrame:
      cacheNames = []
    }
    cacheInputNames = cacheNames.map { "\($0)_in" }
    cacheUpdateNames = cacheNames.map { "\($0)_updates" }
    cacheArrays = try cacheNames.map { _ in
      try Self.makeZeroArray(shape: [1, 41, 8, 128], dataType: .float16)
    }
    Self.logAdmission("cacheAllocationCompleted arrays=\(cacheArrays.count)")

    var builtBursts: [Burst] = []
    var loadedStreamingCarryTemporalModel: MLModel?
    var loadedStreamingCarryTemporalProvider: MLDictionaryFeatureProvider?
    var loadedStreamingCarryTemporalInput: MLMultiArray?
    var loadedStreamingCarrySourceEncoded: MLMultiArray?
    var loadedStreamingCarryValidBias: MLMultiArray?
    var loadedStreamingCarryModelURL: URL?
    var loadedStatefulTemporalModel: MLModel?
    var loadedStatefulTemporalProvider: MLDictionaryFeatureProvider?
    var loadedStatefulTemporalState: MLState?
    var loadedStatefulSourceEncoded: MLMultiArray?
    var loadedStatefulTemporalInput: MLMultiArray?
    switch configuration.temporalMode {
    case .carry:
      for item in CrossfadeRuntimeResources.temporalSchedule {
        let model = try Self.loadModel(
          resourceName: item.resourceName,
          computeUnits: configuration.temporalComputeUnits,
          bundle: modelBundle
        )
        let sourceEncoded = try Self.makeZeroArray(
          shape: [1, item.frames, 256], dataType: .float32)
        var dictionary: [String: MLFeatureValue] = [
          "temporal_inputs": MLFeatureValue(
            multiArray: try Self.makeZeroArray(
              shape: [1, item.frames, Self.modelDim], dataType: .float32)
          ),
          "source_encoded": MLFeatureValue(multiArray: sourceEncoded),
        ]
        for (name, array) in zip(cacheInputNames, cacheArrays) {
          dictionary[name] = MLFeatureValue(multiArray: array)
        }
        builtBursts.append(
          Burst(
            frames: item.frames,
            historyLength: item.history,
            model: model,
            sourceEncoded: sourceEncoded,
            provider: try MLDictionaryFeatureProvider(dictionary: dictionary)
          )
        )
      }
    case .streamingCarry:
      let modelURL = try Self.modelURL(
        resourceName: CrossfadeRuntimeResources.streamingCarryTemporalModelName,
        bundle: modelBundle
      )
      let model = try Self.loadModel(
        url: modelURL,
        computeUnits: configuration.temporalComputeUnits
      )
      let sourceEncoded = try Self.makeZeroArray(
        shape: [1, 1, Self.decoderEmbeddingDim],
        dataType: .float32
      )
      let temporalInput = try Self.makeZeroArray(
        shape: [1, 1, Self.modelDim],
        dataType: .float32
      )
      let validBias = try Self.makeZeroArray(
        shape: [1, 1, 1, Self.temporalAttentionExtent],
        dataType: .float16
      )
      Self.writeStreamingCarryValidBias(validHistory: 0, into: validBias)
      var dictionary: [String: MLFeatureValue] = [
        "temporal_inputs": MLFeatureValue(multiArray: temporalInput),
        "source_encoded": MLFeatureValue(multiArray: sourceEncoded),
        "cache_valid_bias": MLFeatureValue(multiArray: validBias),
      ]
      for (name, array) in zip(cacheInputNames, cacheArrays) {
        dictionary[name] = MLFeatureValue(multiArray: array)
      }
      loadedStreamingCarryTemporalModel = model
      loadedStreamingCarryTemporalProvider = try MLDictionaryFeatureProvider(
        dictionary: dictionary
      )
      loadedStreamingCarryTemporalInput = temporalInput
      loadedStreamingCarrySourceEncoded = sourceEncoded
      loadedStreamingCarryValidBias = validBias
      loadedStreamingCarryModelURL = modelURL
    case .statefulOneFrame:
      let model = try Self.loadModel(
        resourceName: CrossfadeRuntimeResources.statefulOneFrameTemporalModelName,
        computeUnits: configuration.temporalComputeUnits,
        bundle: modelBundle
      )
      let sourceEncoded = try Self.makeZeroArray(
        shape: [1, 1, Self.decoderEmbeddingDim], dataType: .float32)
      let temporalInput = try Self.makeZeroArray(shape: [1, 1, Self.modelDim], dataType: .float32)
      loadedStatefulTemporalModel = model
      loadedStatefulTemporalState = model.makeState()
      loadedStatefulSourceEncoded = sourceEncoded
      loadedStatefulTemporalInput = temporalInput
      loadedStatefulTemporalProvider = try MLDictionaryFeatureProvider(
        dictionary: [
          "temporal_inputs": MLFeatureValue(multiArray: temporalInput),
          "source_encoded": MLFeatureValue(multiArray: sourceEncoded),
        ]
      )
    }
    bursts = builtBursts
    streamingCarryTemporalModel = loadedStreamingCarryTemporalModel
    streamingCarryTemporalProvider = loadedStreamingCarryTemporalProvider
    streamingCarryTemporalInput = loadedStreamingCarryTemporalInput
    streamingCarrySourceEncoded = loadedStreamingCarrySourceEncoded
    streamingCarryValidBias = loadedStreamingCarryValidBias
    streamingCarryModelURL = loadedStreamingCarryModelURL
    statefulTemporalModel = loadedStatefulTemporalModel
    statefulTemporalProvider = loadedStatefulTemporalProvider
    statefulTemporalState = loadedStatefulTemporalState
    statefulSourceEncoded = loadedStatefulSourceEncoded
    statefulTemporalInput = loadedStatefulTemporalInput

    depthModel = try Self.loadModel(
      resourceName: CrossfadeRuntimeResources.depthModelName,
      computeUnits: configuration.depthComputeUnits,
      bundle: modelBundle
    )
    depthFrameInput = try Self.makeZeroArray(shape: [1, 1, Self.modelDim], dataType: .float32)
    depthGumbelNoise = try Self.makeZeroArray(
      shape: [Self.tokenLevels, Self.codebookSize],
      dataType: .float32
    )
    depthInverseTemperature = try Self.makeZeroArray(shape: [1], dataType: .float32)
    depthProvider = try MLDictionaryFeatureProvider(
      dictionary: [
        "temporal_frame": MLFeatureValue(multiArray: depthFrameInput),
        "gumbel_noise": MLFeatureValue(multiArray: depthGumbelNoise),
        "inverse_temperature": MLFeatureValue(multiArray: depthInverseTemperature),
      ]
    )

    decoderModel = try Self.loadModel(
      resourceName: CrossfadeRuntimeResources.decoderResourceName(
        inputFrames: configuration.decoderInputFrames),
      computeUnits: configuration.decoderComputeUnits,
      bundle: modelBundle
    )
    decoderInput = try Self.makeZeroArray(
      shape: [1, configuration.decoderInputFrames, Self.decoderEmbeddingDim],
      dataType: .float32
    )
    decoderProvider = try MLDictionaryFeatureProvider(
      dictionary: ["decoder_embeddings": MLFeatureValue(multiArray: decoderInput)]
    )
    rvqCodebooks = try CrossfadeRVQCodebooks(bundle: resourceBundle)
    depthEmbedder = try CrossfadeDepthEmbedder(bundle: resourceBundle)
    pendingDecoderEmbeddings.reserveCapacity(
      (Self.temporalFrameCount + configuration.decoderInputFrames) * Self.decoderEmbeddingDim
    )
  }

  public func setSourceConditioning(_ conditioning: CrossfadeSourceConditioning?) {
    controlLock.lock()
    runtimeControl.sourceConditioning = conditioning
    controlLock.unlock()
  }

  public func setPromptControl(_ control: CrossfadePromptControl) {
    controlLock.lock()
    runtimeControl = control
    samplingRNG = CrossfadeSeededGenerator(seed: control.seed ?? Self.seed(from: control))
    controlLock.unlock()
  }

  public func generateOnce(
    options: MLPredictionOptions,
    beforeAudioGeneration: (() async throws -> Void)? = nil,
    audioSink: ((MLMultiArray) -> Void)? = nil
  ) async throws -> CrossfadeGenerationResult {
    zeroFloat32(decoderInput)
    let control = currentRuntimeControl()
    switch configuration.temporalMode {
    case .carry:
      return try await generateCarryOnce(
        control: control,
        options: options,
        beforeAudioGeneration: beforeAudioGeneration,
        audioSink: audioSink
      )
    case .streamingCarry:
      return try await generateStreamingCarryOnce(
        control: control,
        options: options,
        beforeAudioGeneration: beforeAudioGeneration,
        audioSink: audioSink
      )
    case .statefulOneFrame:
      return try await generateStatefulOneFrameOnce(
        control: control,
        options: options,
        beforeAudioGeneration: beforeAudioGeneration,
        audioSink: audioSink
      )
    }
  }

  /// Inspect the exact temporal artifact loaded by this app session.
  ///
  /// The report is a startup rejection gate for CPU/GPU-only plans. Runtime
  /// ANE intervals and app GPU absence are still proven by the external Core
  /// ML Instruments trace captured from this same process.
  public func verifyTemporalPlacement() async throws -> CrossfadeTemporalPlacementReport? {
    guard configuration.temporalMode == .streamingCarry,
      let modelURL = streamingCarryModelURL
    else {
      return nil
    }
    guard #available(iOS 17.4, *) else {
      throw CrossfadeRuntimeError.invalidArgument(
        "Temporal placement gate requires iOS 17.4 or newer"
      )
    }
    let modelConfiguration = MLModelConfiguration()
    modelConfiguration.computeUnits = configuration.temporalComputeUnits
    let plan = try await MLComputePlan.load(
      contentsOf: modelURL,
      configuration: modelConfiguration
    )
    guard case .program(let program) = plan.modelStructure,
      let function = program.functions["main"] ?? program.functions.values.first
    else {
      throw CrossfadeRuntimeError.invalidArgument(
        "Temporal placement gate could not inspect the ML Program"
      )
    }
    let operations = Self.flattenOperations(function.block.operations)
    var cpuOperations = 0
    var gpuOperations = 0
    var aneOperations = 0
    var cpuCost = 0.0
    var gpuCost = 0.0
    var aneCost = 0.0
    var missingUsage = 0
    for operation in operations {
      guard let usage = plan.deviceUsage(for: operation) else {
        missingUsage += 1
        continue
      }
      let weight = plan.estimatedCost(of: operation)?.weight ?? 0
      switch usage.preferred {
      case .cpu:
        cpuOperations += 1
        cpuCost += weight
      case .gpu:
        gpuOperations += 1
        gpuCost += weight
      case .neuralEngine:
        aneOperations += 1
        aneCost += weight
      @unknown default:
        missingUsage += 1
      }
    }
    let passed = aneOperations > 0 && aneCost >= 0.95 && gpuOperations == 0 && gpuCost == 0
    return CrossfadeTemporalPlacementReport(
      modelName: CrossfadeRuntimeResources.streamingCarryTemporalModelName,
      operationCount: operations.count,
      cpuOperationCount: cpuOperations,
      gpuOperationCount: gpuOperations,
      aneOperationCount: aneOperations,
      cpuEstimatedCostWeight: cpuCost,
      gpuEstimatedCostWeight: gpuCost,
      aneEstimatedCostWeight: aneCost,
      missingUsageCount: missingUsage,
      passed: passed
    )
  }

  /// Prove that ordinary cache outputs are copied into, and then read from,
  /// the next temporal prediction. This deliberately resets temporal state and
  /// therefore belongs at session startup before any user audio is generated.
  public func runTemporalStateProof(
    options: MLPredictionOptions
  ) async throws -> CrossfadeTemporalStateProofReport? {
    guard configuration.temporalMode == .streamingCarry,
      let model = streamingCarryTemporalModel,
      let provider = streamingCarryTemporalProvider,
      let temporalInput = streamingCarryTemporalInput,
      let sourceEncoded = streamingCarrySourceEncoded,
      let validBias = streamingCarryValidBias
    else {
      return nil
    }
    resetStreamingCarryState()
    Self.writeDeterministicProofValues(into: temporalInput, modulus: 31, scale: 0.01)
    Self.writeDeterministicProofValues(into: sourceEncoded, modulus: 17, scale: 0.02)
    Self.writeStreamingCarryValidBias(validHistory: 0, into: validBias)
    let freshResult = try await model.prediction(from: provider, options: options)
    guard let freshOutput = freshResult.featureValue(for: "temporal_outputs")?.multiArrayValue
    else {
      throw CrossfadeRuntimeError.missingOutput("temporal_outputs")
    }
    let freshValues = crossfadeDenseFloats(from: freshOutput)
    for (index, updateName) in cacheUpdateNames.enumerated() {
      guard let update = freshResult.featureValue(for: updateName)?.multiArrayValue else {
        throw CrossfadeRuntimeError.missingOutput(updateName)
      }
      try appendStreamingCacheUpdate(update, into: cacheArrays[index], validHistory: 0)
    }
    streamingCarryValidHistory = 1
    Self.writeStreamingCarryValidBias(validHistory: 1, into: validBias)
    let warmedResult = try await model.prediction(from: provider, options: options)
    guard let warmedOutput = warmedResult.featureValue(for: "temporal_outputs")?.multiArrayValue
    else {
      throw CrossfadeRuntimeError.missingOutput("temporal_outputs")
    }
    let warmedValues = crossfadeDenseFloats(from: warmedOutput)
    let maxDifference = zip(freshValues, warmedValues).reduce(0.0) {
      max($0, Double(abs($1.0 - $1.1)))
    }
    let mutatedCaches = cacheArrays.filter(Self.hasNonzeroFloat16).count
    let fixture = try await runTemporalFixtureProof(
      model: model,
      provider: provider,
      temporalInput: temporalInput,
      sourceEncoded: sourceEncoded,
      validBias: validBias,
      options: options
    )
    let shortStatePassed = maxDifference > 0 && mutatedCaches == cacheArrays.count
    let report = CrossfadeTemporalStateProofReport(
      modelName: CrossfadeRuntimeResources.streamingCarryTemporalModelName,
      freshChecksum: Self.floatChecksum(freshValues),
      warmedChecksum: Self.floatChecksum(warmedValues),
      maxAbsoluteDifference: maxDifference,
      freshAndWarmedDiverged: maxDifference > 0,
      cacheArraysMutated: mutatedCaches,
      fixtureFrames: fixture?.frames ?? 0,
      fixtureWrappedFrames: fixture?.wrappedFrames ?? 0,
      fixtureCorrelation: fixture?.correlation,
      fixtureMaxAbsoluteError: fixture?.maxAbsoluteError,
      fixtureMeanAbsoluteError: fixture?.meanAbsoluteError,
      fixtureFiniteRatio: fixture?.finiteRatio,
      fixturePassed: fixture?.passed,
      passed: shortStatePassed && (fixture?.passed ?? true)
    )
    resetStreamingCarryState()
    zeroFloat32(temporalInput)
    zeroFloat32(sourceEncoded)
    return report
  }

  /// Compare 64 consecutive on-device predictions with the frozen readable
  /// PyTorch reference. The fixture crosses the 41-frame cache window, so the
  /// comparison exercises both the append and ring-shift branches in Swift.
  /// Proof resources are optional in product bundles and mandatory in the
  /// paper's signed host build.
  private func runTemporalFixtureProof(
    model: MLModel,
    provider: MLDictionaryFeatureProvider,
    temporalInput: MLMultiArray,
    sourceEncoded: MLMultiArray,
    validBias: MLMultiArray,
    options: MLPredictionOptions
  ) async throws -> TemporalFixtureMetrics? {
    let urls = [
      resourceBundle.url(
        forResource: Self.temporalFixtureInputStem,
        withExtension: "bin"
      ),
      resourceBundle.url(
        forResource: Self.temporalFixtureSourceStem,
        withExtension: "bin"
      ),
      resourceBundle.url(
        forResource: Self.temporalFixtureReferenceStem,
        withExtension: "bin"
      ),
    ]
    if urls.allSatisfy({ $0 == nil }) {
      return nil
    }
    guard let temporalURL = urls[0], let sourceURL = urls[1], let referenceURL = urls[2]
    else {
      throw CrossfadeRuntimeError.invalidArgument(
        "Temporal streaming fixture is incomplete; bundle all three Float32 arrays"
      )
    }
    let frames = Self.temporalFixtureFrames
    let temporalValues = try Self.readFloat32Fixture(
      temporalURL,
      expectedCount: frames * Self.modelDim
    )
    let sourceValues = try Self.readFloat32Fixture(
      sourceURL,
      expectedCount: frames * Self.decoderEmbeddingDim
    )
    let referenceValues = try Self.readFloat32Fixture(
      referenceURL,
      expectedCount: frames * Self.modelDim
    )

    resetStreamingCarryState()
    defer {
      resetStreamingCarryState()
      zeroFloat32(temporalInput)
      zeroFloat32(sourceEncoded)
    }
    var sampleCount = 0
    var finiteCount = 0
    var absoluteErrorSum = 0.0
    var maxAbsoluteError = 0.0
    var candidateSum = 0.0
    var referenceSum = 0.0
    var candidateSquareSum = 0.0
    var referenceSquareSum = 0.0
    var productSum = 0.0

    for frame in 0..<frames {
      Self.copyFloat32FixtureFrame(
        temporalValues,
        frame: frame,
        width: Self.modelDim,
        into: temporalInput
      )
      Self.copyFloat32FixtureFrame(
        sourceValues,
        frame: frame,
        width: Self.decoderEmbeddingDim,
        into: sourceEncoded
      )
      Self.writeStreamingCarryValidBias(
        validHistory: streamingCarryValidHistory,
        into: validBias
      )
      let result = try await model.prediction(from: provider, options: options)
      guard let output = result.featureValue(for: "temporal_outputs")?.multiArrayValue else {
        throw CrossfadeRuntimeError.missingOutput("temporal_outputs")
      }
      let candidate = crossfadeDenseFloats(from: output)
      guard candidate.count == Self.modelDim else {
        throw CrossfadeRuntimeError.invalidArgument(
          "Unexpected temporal fixture output count=\(candidate.count)"
        )
      }
      let referenceOffset = frame * Self.modelDim
      for index in 0..<Self.modelDim {
        let x = Double(candidate[index])
        let y = Double(referenceValues[referenceOffset + index])
        sampleCount += 1
        if x.isFinite {
          finiteCount += 1
        }
        let error = abs(x - y)
        absoluteErrorSum += error
        maxAbsoluteError = max(maxAbsoluteError, error)
        candidateSum += x
        referenceSum += y
        candidateSquareSum += x * x
        referenceSquareSum += y * y
        productSum += x * y
      }
      for (index, updateName) in cacheUpdateNames.enumerated() {
        guard let update = result.featureValue(for: updateName)?.multiArrayValue else {
          throw CrossfadeRuntimeError.missingOutput(updateName)
        }
        try appendStreamingCacheUpdate(
          update,
          into: cacheArrays[index],
          validHistory: streamingCarryValidHistory
        )
      }
      streamingCarryValidHistory = min(
        streamingCarryValidHistory + 1,
        Self.temporalWindowFrames
      )
    }
    let count = Double(sampleCount)
    let covariance = count * productSum - candidateSum * referenceSum
    let candidateVariance = count * candidateSquareSum - candidateSum * candidateSum
    let referenceVariance = count * referenceSquareSum - referenceSum * referenceSum
    let denominator = sqrt(max(0, candidateVariance) * max(0, referenceVariance))
    let correlation = denominator > 0 ? covariance / denominator : 0
    let finiteRatio = Double(finiteCount) / count
    let meanAbsoluteError = absoluteErrorSum / count
    let wrappedFrames = frames - Self.temporalWindowFrames
    let passed =
      wrappedFrames > 0
      && finiteRatio == 1
      && correlation >= 0.999
      && maxAbsoluteError <= 2.5
    return TemporalFixtureMetrics(
      frames: frames,
      wrappedFrames: wrappedFrames,
      correlation: correlation,
      maxAbsoluteError: maxAbsoluteError,
      meanAbsoluteError: meanAbsoluteError,
      finiteRatio: finiteRatio,
      passed: passed
    )
  }

  private func generateCarryOnce(
    control: CrossfadePromptControl,
    options: MLPredictionOptions,
    beforeAudioGeneration: (() async throws -> Void)?,
    audioSink: ((MLMultiArray) -> Void)?
  ) async throws -> CrossfadeGenerationResult {
    let totalStart = ContinuousClock.now
    resetCaches()
    var checksum = 0
    var sourceFrameOffset = 0
    var temporalMs = 0.0
    var depthMs = 0.0
    var samplingMs = 0.0
    var decoderMs = 0.0
    var audioBackpressureMs = 0.0
    var decoderCalls = 0
    for burst in bursts {
      copySourceConditioning(
        control.sourceConditioning,
        into: burst.sourceEncoded,
        startFrame: sourceFrameOffset,
        frameCount: burst.frames
      )
      sourceFrameOffset += burst.frames
      let temporalStart = ContinuousClock.now
      let temporalOutput = try await burst.model.prediction(from: burst.provider, options: options)
      temporalMs += Self.milliseconds(since: temporalStart)
      for (index, updateName) in cacheUpdateNames.enumerated() {
        guard let updateArray = temporalOutput.featureValue(for: updateName)?.multiArrayValue else {
          throw CrossfadeRuntimeError.missingOutput(updateName)
        }
        copyCacheUpdate(
          updateArray,
          into: cacheArrays[index],
          historyLength: burst.historyLength,
          frames: burst.frames
        )
      }
      guard
        let temporalArray = temporalOutput.featureValue(for: "temporal_outputs")?.multiArrayValue
      else {
        throw CrossfadeRuntimeError.missingOutput("temporal_outputs")
      }
      // KNOWN LIMITATION (carry escape hatch only — not the shipped path):
      // this mode runs the temporal transformer over multi-frame bursts whose
      // `temporal_inputs` stay zero, because frames 2..N of a burst would need
      // token embeddings sampled AFTER the burst's temporal pass already ran.
      // Within-frame depth feedback below is correct; cross-frame temporal
      // feedback requires the 1-frame stateful path (`.statefulOneFrame`).
      for frameIndex in 0..<burst.frames {
        copyTemporalFrameToDepthStepInput(temporalArray, frameIndex: frameIndex)
        let rollout = try await runDepthRollout(options: options, sampling: control.sampling)
        depthMs += rollout.depthMs
        samplingMs += rollout.samplingMs
        checksum &+= appendDecoderEmbedding(rawTokens: rollout.rawTokens)
        while pendingDecoderEmbeddings.count >= decoderInputFrames * Self.decoderEmbeddingDim {
          let audioBackpressureStart = ContinuousClock.now
          if audioSink != nil {
            try await beforeAudioGeneration?()
          }
          audioBackpressureMs += Self.milliseconds(since: audioBackpressureStart)
          let decoderStart = ContinuousClock.now
          let decoderResult = try await runDecoderWindow(options: options)
          decoderMs += Self.milliseconds(since: decoderStart)
          decoderCalls += 1
          checksum &+= decoderResult.checksum
          audioSink?(decoderResult.stft)
          discardDecodedEmbeddingStride()
        }
      }
    }
    return CrossfadeGenerationResult(
      checksum: checksum,
      timing: CrossfadeGenerationTiming(
        totalMs: Self.milliseconds(since: totalStart),
        temporalMs: temporalMs,
        depthMs: depthMs,
        samplingMs: samplingMs,
        decoderMs: decoderMs,
        audioBackpressureMs: audioBackpressureMs,
        decoderCalls: decoderCalls
      )
    )
  }

  private func generateStreamingCarryOnce(
    control: CrossfadePromptControl,
    options: MLPredictionOptions,
    beforeAudioGeneration: (() async throws -> Void)?,
    audioSink: ((MLMultiArray) -> Void)?
  ) async throws -> CrossfadeGenerationResult {
    let totalStart = ContinuousClock.now
    guard let temporalModel = streamingCarryTemporalModel,
      let temporalProvider = streamingCarryTemporalProvider,
      let temporalInput = streamingCarryTemporalInput,
      let sourceEncoded = streamingCarrySourceEncoded,
      let validBias = streamingCarryValidBias
    else {
      throw CrossfadeRuntimeError.invalidArgument("Streaming carry temporal model is not loaded")
    }
    var checksum = 0
    var temporalMs = 0.0
    var depthMs = 0.0
    var samplingMs = 0.0
    var decoderMs = 0.0
    var audioBackpressureMs = 0.0
    var decoderCalls = 0
    for frameIndex in 0..<Self.temporalFrameCount {
      if let refreshFrames = configuration.trajectoryRefreshFrames,
        refreshFrames > 0,
        generatedTrajectoryFrames > 0,
        generatedTrajectoryFrames.isMultiple(of: refreshFrames)
      {
        resetStreamingCarryState()
        Self.logAdmission(
          "trajectoryRefresh frame=\(generatedTrajectoryFrames) intervalFrames=\(refreshFrames)"
        )
      }
      copySourceConditioning(
        control.sourceConditioning,
        into: sourceEncoded,
        startFrame: frameIndex,
        frameCount: 1
      )
      writeTemporalMeanEmbedding(into: temporalInput)
      Self.writeStreamingCarryValidBias(
        validHistory: streamingCarryValidHistory,
        into: validBias
      )
      let temporalStart = ContinuousClock.now
      let temporalOutput = try await temporalModel.prediction(
        from: temporalProvider,
        options: options
      )
      temporalMs += Self.milliseconds(since: temporalStart)
      for (index, updateName) in cacheUpdateNames.enumerated() {
        guard let update = temporalOutput.featureValue(for: updateName)?.multiArrayValue else {
          throw CrossfadeRuntimeError.missingOutput(updateName)
        }
        try appendStreamingCacheUpdate(
          update,
          into: cacheArrays[index],
          validHistory: streamingCarryValidHistory
        )
      }
      streamingCarryValidHistory = min(
        streamingCarryValidHistory + 1,
        Self.temporalWindowFrames
      )
      guard
        let temporalArray = temporalOutput.featureValue(
          for: "temporal_outputs"
        )?.multiArrayValue
      else {
        throw CrossfadeRuntimeError.missingOutput("temporal_outputs")
      }
      copyTemporalFrameToDepthStepInput(temporalArray, frameIndex: 0)
      let rollout = try await runDepthRollout(options: options, sampling: control.sampling)
      depthMs += rollout.depthMs
      samplingMs += rollout.samplingMs
      generatedTrajectoryFrames += 1
      checksum &+= appendDecoderEmbedding(rawTokens: rollout.rawTokens)
      while pendingDecoderEmbeddings.count >= decoderInputFrames * Self.decoderEmbeddingDim {
        let audioBackpressureStart = ContinuousClock.now
        if audioSink != nil {
          try await beforeAudioGeneration?()
        }
        audioBackpressureMs += Self.milliseconds(since: audioBackpressureStart)
        let decoderStart = ContinuousClock.now
        let decoderResult = try await runDecoderWindow(options: options)
        decoderMs += Self.milliseconds(since: decoderStart)
        decoderCalls += 1
        checksum &+= decoderResult.checksum
        audioSink?(decoderResult.stft)
        discardDecodedEmbeddingStride()
      }
    }
    return CrossfadeGenerationResult(
      checksum: checksum,
      timing: CrossfadeGenerationTiming(
        totalMs: Self.milliseconds(since: totalStart),
        temporalMs: temporalMs,
        depthMs: depthMs,
        samplingMs: samplingMs,
        decoderMs: decoderMs,
        audioBackpressureMs: audioBackpressureMs,
        decoderCalls: decoderCalls
      )
    )
  }

  private func generateStatefulOneFrameOnce(
    control: CrossfadePromptControl,
    options: MLPredictionOptions,
    beforeAudioGeneration: (() async throws -> Void)?,
    audioSink: ((MLMultiArray) -> Void)?
  ) async throws -> CrossfadeGenerationResult {
    let totalStart = ContinuousClock.now
    guard let temporalModel = statefulTemporalModel,
      let temporalProvider = statefulTemporalProvider,
      let temporalState = statefulTemporalState,
      let sourceEncoded = statefulSourceEncoded,
      let temporalInput = statefulTemporalInput
    else {
      throw CrossfadeRuntimeError.invalidArgument("Stateful one-frame temporal model is not loaded")
    }
    var checksum = 0
    var temporalMs = 0.0
    var depthMs = 0.0
    var samplingMs = 0.0
    var decoderMs = 0.0
    var audioBackpressureMs = 0.0
    var decoderCalls = 0
    for frameIndex in 0..<Self.temporalFrameCount {
      copySourceConditioning(
        control.sourceConditioning,
        into: sourceEncoded,
        startFrame: frameIndex,
        frameCount: 1
      )
      // Temporal feedback: the transformer's input for this frame is the mean
      // of the previous frame's 12 token embeddings (MLX sampler contract).
      writeTemporalMeanEmbedding(into: temporalInput)
      let temporalStart = ContinuousClock.now
      let temporalOutput = try await temporalModel.prediction(
        from: temporalProvider,
        using: temporalState,
        options: options
      )
      temporalMs += Self.milliseconds(since: temporalStart)
      guard
        let temporalArray = temporalOutput.featureValue(for: "temporal_outputs")?.multiArrayValue
      else {
        throw CrossfadeRuntimeError.missingOutput("temporal_outputs")
      }
      copyTemporalFrameToDepthStepInput(temporalArray, frameIndex: 0)
      let rollout = try await runDepthRollout(options: options, sampling: control.sampling)
      depthMs += rollout.depthMs
      samplingMs += rollout.samplingMs
      checksum &+= appendDecoderEmbedding(rawTokens: rollout.rawTokens)
      while pendingDecoderEmbeddings.count >= decoderInputFrames * Self.decoderEmbeddingDim {
        let audioBackpressureStart = ContinuousClock.now
        if audioSink != nil {
          try await beforeAudioGeneration?()
        }
        audioBackpressureMs += Self.milliseconds(since: audioBackpressureStart)
        let decoderStart = ContinuousClock.now
        let decoderResult = try await runDecoderWindow(options: options)
        decoderMs += Self.milliseconds(since: decoderStart)
        decoderCalls += 1
        checksum &+= decoderResult.checksum
        audioSink?(decoderResult.stft)
        discardDecodedEmbeddingStride()
      }
    }
    return CrossfadeGenerationResult(
      checksum: checksum,
      timing: CrossfadeGenerationTiming(
        totalMs: Self.milliseconds(since: totalStart),
        temporalMs: temporalMs,
        depthMs: depthMs,
        samplingMs: samplingMs,
        decoderMs: decoderMs,
        audioBackpressureMs: audioBackpressureMs,
        decoderCalls: decoderCalls
      )
    )
  }

  private func currentRuntimeControl() -> CrossfadePromptControl {
    controlLock.lock()
    let control = runtimeControl
    controlLock.unlock()
    return control
  }

  private func copySourceConditioning(
    _ conditioning: CrossfadeSourceConditioning?,
    into sourceEncoded: MLMultiArray,
    startFrame: Int,
    frameCount: Int
  ) {
    let destination = sourceEncoded.dataPointer.bindMemory(
      to: Float.self, capacity: sourceEncoded.count)
    guard let conditioning, conditioning.frameCount > 0 else {
      for index in 0..<sourceEncoded.count {
        destination[index] = 0
      }
      return
    }
    let values = conditioning.values
    let sourceDimension = CrossfadeSourceConditioning.sourceDimension
    for frameIndex in 0..<frameCount {
      let sourceFrame = min(startFrame + frameIndex, conditioning.frameCount - 1)
      let sourceStart = sourceFrame * sourceDimension
      let destinationStart = frameIndex * sourceDimension
      for dimension in 0..<sourceDimension {
        destination[destinationStart + dimension] = values[sourceStart + dimension]
      }
    }
  }

  private func resetCaches() {
    for array in cacheArrays {
      zeroFloat16(array)
    }
  }

  private func resetStreamingCarryState() {
    resetCaches()
    streamingCarryValidHistory = 0
    previousTemporalFeedback = nil
    if let validBias = streamingCarryValidBias {
      Self.writeStreamingCarryValidBias(validHistory: 0, into: validBias)
    }
  }

  private static func writeStreamingCarryValidBias(
    validHistory: Int,
    into bias: MLMultiArray
  ) {
    let boundedHistory = min(max(validHistory, 0), temporalWindowFrames)
    let pointer = bias.dataPointer.bindMemory(to: Float16.self, capacity: bias.count)
    for index in 0..<bias.count {
      pointer[index] = invalidAttentionBias
    }
    pointer[0] = 0  // attention sink
    if boundedHistory > 0 {
      for index in 1...boundedHistory {
        pointer[index] = 0
      }
    }
    pointer[temporalAttentionExtent - 1] = 0  // current frame
  }

  /// Append one `[1,1,8,128]` Core ML update to a dense chronological host
  /// cache. Reads honor the output array's strides; the preallocated host
  /// array is shifted in place only after the 41-frame window fills.
  private func appendStreamingCacheUpdate(
    _ update: MLMultiArray,
    into cache: MLMultiArray,
    validHistory: Int
  ) throws {
    let updateShape = update.shape.map(\.intValue)
    let updateStrides = update.strides.map(\.intValue)
    guard updateShape == [1, 1, 8, 128], updateStrides.count == 4 else {
      throw CrossfadeRuntimeError.invalidArgument(
        "Unexpected temporal cache update shape=\(updateShape) strides=\(updateStrides)"
      )
    }
    guard cache.dataType == .float16 else {
      throw CrossfadeRuntimeError.unsupportedDataType(
        "Temporal host cache must be Float16, got \(cache.dataType.rawValue)"
      )
    }
    let cachePointer = cache.dataPointer.bindMemory(to: Float16.self, capacity: cache.count)
    let elementsPerFrame = Self.updateElementsPerFrame
    if validHistory >= Self.temporalWindowFrames {
      memmove(
        cachePointer,
        cachePointer.advanced(by: elementsPerFrame),
        (Self.temporalWindowFrames - 1) * elementsPerFrame * MemoryLayout<Float16>.stride
      )
    }
    let destinationFrame = min(validHistory, Self.temporalWindowFrames - 1)
    let destinationStart = destinationFrame * elementsPerFrame
    let updateCapacity = zip(updateShape, updateStrides).reduce(1) {
      $0 + ($1.0 - 1) * $1.1
    }
    switch update.dataType {
    case .float16:
      let source = update.dataPointer.bindMemory(to: Float16.self, capacity: updateCapacity)
      for head in 0..<Self.temporalHeads {
        for dimension in 0..<Self.temporalHeadDim {
          let sourceOffset = head * updateStrides[2] + dimension * updateStrides[3]
          cachePointer[destinationStart + head * Self.temporalHeadDim + dimension] =
            source[sourceOffset]
        }
      }
    case .float32:
      let source = update.dataPointer.bindMemory(to: Float.self, capacity: updateCapacity)
      for head in 0..<Self.temporalHeads {
        for dimension in 0..<Self.temporalHeadDim {
          let sourceOffset = head * updateStrides[2] + dimension * updateStrides[3]
          cachePointer[destinationStart + head * Self.temporalHeadDim + dimension] = Float16(
            source[sourceOffset])
        }
      }
    default:
      throw CrossfadeRuntimeError.unsupportedDataType(
        "Unsupported temporal cache update type: \(update.dataType.rawValue)"
      )
    }
  }

  private func copyCacheUpdate(
    _ update: MLMultiArray,
    into cache: MLMultiArray,
    historyLength: Int,
    frames: Int
  ) {
    let updatePointer = update.dataPointer.bindMemory(to: Float16.self, capacity: update.count)
    let cachePointer = cache.dataPointer.bindMemory(to: Float16.self, capacity: cache.count)
    let destinationStart = historyLength * Self.updateElementsPerFrame
    let elementCount = frames * Self.updateElementsPerFrame
    for index in 0..<elementCount {
      cachePointer[destinationStart + index] = updatePointer[index]
    }
  }

  /// Writes one temporal output frame into the depth rollout's frame input
  /// (level 0's per-position contract: position 0 is the temporal frame
  /// embedding; levels 1..11 are produced in-graph).
  private func copyTemporalFrameToDepthStepInput(_ temporalOutput: MLMultiArray, frameIndex: Int) {
    let destination = depthFrameInput.dataPointer.bindMemory(
      to: Float.self, capacity: depthFrameInput.count)
    // Stride-aware read: Core ML output rows may be padded for alignment, so
    // never compute the frame offset from the dense element count (see
    // crossfadeDenseFloats for the depth_logits incident this guards against).
    let dimensions = temporalOutput.shape.count
    let frameStride =
      dimensions >= 2
      ? temporalOutput.strides[dimensions - 2].intValue
      : Self.modelDim
    let elementStride = temporalOutput.strides[dimensions - 1].intValue
    let sourceStart = frameIndex * frameStride
    let capacity = sourceStart + (Self.modelDim - 1) * elementStride + 1
    switch temporalOutput.dataType {
    case .float32:
      let source = temporalOutput.dataPointer.bindMemory(to: Float.self, capacity: capacity)
      for index in 0..<Self.modelDim {
        destination[index] = source[sourceStart + index * elementStride]
      }
    case .float16:
      let source = temporalOutput.dataPointer.bindMemory(to: Float16.self, capacity: capacity)
      for index in 0..<Self.modelDim {
        destination[index] = Float(source[sourceStart + index * elementStride])
      }
    default:
      return
    }
  }

  private struct DepthRolloutResult {
    /// Per-level codebook-local tokens (0..1023), for the RVQ decoder embedding.
    let rawTokens: [Int]
    let depthMs: Double
    let samplingMs: Double
  }

  /// Samples one frame's 12 RVQ levels with a SINGLE depth prediction.
  ///
  /// The exported `mrt2_depth_body_rollout` graph runs the whole
  /// autoregressive rollout IN-GRAPH: sample level k via Gumbel-max over the
  /// static top-40 set, embed the sampled token, feed it to level k+1. The
  /// host contributes the per-frame Gumbel noise (from `samplingRNG`, so
  /// seed determinism is preserved) and the inverse temperature, and reads
  /// back the 12 codebook-local codes plus the `temporal_feedback` mean
  /// embedding for the next frame's temporal input.
  ///
  /// (Every out-of-graph rollout — 12 full passes through build 19's graph,
  /// then 12 stateful step calls — made 12 depth PREDICTIONS per frame. The
  /// depth path is weight-bandwidth-bound on phones: each prediction streams
  /// the full ~97 MB weight set from DRAM, so any 12-call scheme costs
  /// ~40 ms of depth per frame, ~2x the 25 Hz budget once composed with the
  /// temporal step — the audible stutter. One prediction streams the weights
  /// once. See SEQUEL 2 in
  /// README/Notes/aperture-device-quality-root-cause-feedback-and-strides.md.)
  ///
  /// Expects `depthFrameInput` to already hold the temporal frame
  /// (`copyTemporalFrameToDepthStepInput`). Updates
  /// `previousTemporalFeedback` for the next frame's temporal input.
  private func runDepthRollout(
    options: MLPredictionOptions,
    sampling: CrossfadeSamplingControl
  ) async throws -> DepthRolloutResult {
    let noiseStart = ContinuousClock.now
    writeDepthGumbelNoise()
    writeDepthInverseTemperature(sampling: sampling)
    var samplingMs = Self.milliseconds(since: noiseStart)
    let depthStart = ContinuousClock.now
    let depthOutput = try await depthModel.prediction(from: depthProvider, options: options)
    let depthMs = Self.milliseconds(since: depthStart)
    guard let codes = depthOutput.featureValue(for: "sampled_codes")?.multiArrayValue else {
      throw CrossfadeRuntimeError.missingOutput("sampled_codes")
    }
    guard let feedback = depthOutput.featureValue(for: "temporal_feedback")?.multiArrayValue else {
      throw CrossfadeRuntimeError.missingOutput("temporal_feedback")
    }
    let readStart = ContinuousClock.now
    let rawTokens = try readSampledCodes(codes)
    previousTemporalFeedback = crossfadeDenseFloats(from: feedback)
    samplingMs += Self.milliseconds(since: readStart)
    tokenTap?(rawTokens)
    return DepthRolloutResult(rawTokens: rawTokens, depthMs: depthMs, samplingMs: samplingMs)
  }

  /// Fills the `[12, 1024]` per-frame Gumbel(0,1) noise from the seeded host
  /// RNG: `g = -log(-log(u))` with `u` clamped away from zero so both logs
  /// stay finite. Sampling from `softmax(logits/T)` over the top-k set is
  /// exactly `argmax(logits/T + g)` over that set (the Gumbel-max trick), so
  /// this preserves the sampling distribution of the old host-side sampler
  /// while the argmax itself runs in-graph.
  private func writeDepthGumbelNoise() {
    let destination = depthGumbelNoise.dataPointer.bindMemory(
      to: Float.self,
      capacity: depthGumbelNoise.count
    )
    for index in 0..<depthGumbelNoise.count {
      let uniform = max(samplingRNG.nextUnitFloat(), 1e-7)
      destination[index] = -logf(-logf(uniform))
    }
  }

  /// Writes `1 / max(0.05, temperature)` for the in-graph sampler. Top-k is
  /// BAKED into the exported graph at 40 (the MRT2 MLX default and the
  /// shipped control value); `sampling.topK` is intentionally ignored on
  /// this path.
  private func writeDepthInverseTemperature(sampling: CrossfadeSamplingControl) {
    let destination = depthInverseTemperature.dataPointer.bindMemory(to: Float.self, capacity: 1)
    destination[0] = 1.0 / max(0.05, sampling.temperature)
  }

  /// Reads the `[12]` int32 codebook-local codes, stride-aware (CRITICAL:
  /// never index a Core ML output by `count / dims`; see the depth_logits
  /// misaligned-sampling incident documented on `crossfadeDenseFloats`).
  private func readSampledCodes(_ codes: MLMultiArray) throws -> [Int] {
    guard codes.dataType == .int32 else {
      throw CrossfadeRuntimeError.unsupportedDataType(
        "sampled_codes expected int32, got \(codes.dataType)"
      )
    }
    let elementStride = codes.strides.last?.intValue ?? 1
    let capacity = (Self.tokenLevels - 1) * elementStride + 1
    let pointer = codes.dataPointer.bindMemory(to: Int32.self, capacity: capacity)
    var rawTokens = [Int](repeating: 0, count: Self.tokenLevels)
    for level in 0..<Self.tokenLevels {
      let code = Int(pointer[level * elementStride])
      guard code >= 0, code < Self.codebookSize else {
        throw CrossfadeRuntimeError.invalidArgument(
          "sampled_codes[\(level)] out of range: \(code)"
        )
      }
      rawTokens[level] = code
    }
    return rawTokens
  }

  private func appendDecoderEmbedding(rawTokens: [Int]) -> Int {
    let frame = rvqCodebooks.embedding(rawTokens: rawTokens)
    pendingDecoderEmbeddings.append(contentsOf: frame)
    return rawTokens.reduce(0, &+)
  }

  /// Writes the previous frame's mean token embedding into `temporal_inputs`.
  ///
  /// After the first frame this is the depth rollout's in-graph
  /// `temporal_feedback` output; the cold start is the embedder mean of
  /// twelve id-0 reserved tokens (the MLX sampler's initial state).
  private func writeTemporalMeanEmbedding(into temporalInput: MLMultiArray) {
    let destination = temporalInput.dataPointer.bindMemory(
      to: Float.self,
      capacity: temporalInput.count
    )
    if let feedback = previousTemporalFeedback, feedback.count >= Self.modelDim {
      for index in 0..<Self.modelDim {
        destination[index] = feedback[index]
      }
    } else {
      depthEmbedder.copyMeanEmbedding(
        uniqueTokens: [Int](repeating: 0, count: Self.tokenLevels),
        into: destination
      )
    }
  }

  private struct DecoderWindowResult {
    let checksum: Int
    let stft: MLMultiArray
  }

  private func runDecoderWindow(options: MLPredictionOptions) async throws -> DecoderWindowResult {
    copyPendingDecoderWindowToInput(decoderInput)
    let decoderOutput = try await decoderModel.prediction(from: decoderProvider, options: options)
    guard let stft = decoderOutput.featureValue(for: "decoder_stft")?.multiArrayValue else {
      throw CrossfadeRuntimeError.missingOutput("decoder_stft")
    }
    let emittedSTFT = try decoderOutputForEmission(stft)
    return DecoderWindowResult(checksum: stftChecksum(emittedSTFT), stft: emittedSTFT)
  }

  /// Preserve the decoder's causal convolution history without changing its
  /// static Core ML input shape. The first window establishes overlap-add
  /// state. Later windows advance by `decoderStrideFrames`, discard the STFT
  /// rows corresponding to retained token context, and emit only new audio.
  private func decoderOutputForEmission(_ fullSTFT: MLMultiArray) throws -> MLMultiArray {
    defer { hasDecodedWindow = true }
    guard hasDecodedWindow, decoderContextFrames > 0 else {
      return fullSTFT
    }
    let shape = fullSTFT.shape.map(\.intValue)
    guard shape.count == 4, shape[0] == 1 else {
      throw CrossfadeRuntimeError.invalidArgument(
        "decoder_stft must have shape [1, frames, bins, channels]"
      )
    }
    let cropFrames = decoderContextFrames * 4
    guard cropFrames < shape[1] else {
      throw CrossfadeRuntimeError.invalidArgument(
        "decoder context crop exceeds decoder_stft frame count"
      )
    }
    let emittedShape = [1, shape[1] - cropFrames, shape[2], shape[3]]
    if croppedDecoderSTFT?.shape.map(\.intValue) != emittedShape {
      croppedDecoderSTFT = try Self.makeZeroArray(
        shape: emittedShape,
        dataType: .float32
      )
    }
    guard let croppedDecoderSTFT else {
      throw CrossfadeRuntimeError.missingOutput("cropped decoder_stft")
    }
    let dense = crossfadeDenseFloats(from: fullSTFT)
    let valuesPerFrame = shape[2] * shape[3]
    let sourceOffset = cropFrames * valuesPerFrame
    let destination = croppedDecoderSTFT.dataPointer.bindMemory(
      to: Float.self,
      capacity: croppedDecoderSTFT.count
    )
    for index in 0..<croppedDecoderSTFT.count {
      destination[index] = dense[sourceOffset + index]
    }
    return croppedDecoderSTFT
  }

  private func copyPendingDecoderWindowToInput(_ decoderInput: MLMultiArray) {
    let destination = decoderInput.dataPointer.bindMemory(
      to: Float.self, capacity: decoderInput.count)
    let elementCount = decoderInputFrames * Self.decoderEmbeddingDim
    for index in 0..<elementCount {
      destination[index] = pendingDecoderEmbeddings[index]
    }
  }

  private func discardDecodedEmbeddingStride() {
    let elementCount = decoderStrideFrames * Self.decoderEmbeddingDim
    if pendingDecoderEmbeddings.count <= elementCount {
      pendingDecoderEmbeddings.removeAll(keepingCapacity: true)
    } else {
      pendingDecoderEmbeddings.removeFirst(elementCount)
    }
  }

  private static func seed(from control: CrossfadePromptControl) -> UInt64 {
    var hash: UInt64 = 14_695_981_039_346_656_037
    func mix(_ value: String) {
      for byte in value.utf8 {
        hash ^= UInt64(byte)
        hash = hash &* 1_099_511_628_211
      }
    }
    mix(control.text)
    mix(control.style ?? "")
    mix(control.activeMIDINotes.map(String.init).joined(separator: ","))
    mix("\(control.drumsEnabled)")
    mix(String(format: "%.3f", control.sampling.temperature))
    mix("\(control.sampling.topK)")
    mix(String(format: "%.3f", control.styleGuidance))
    mix(String(format: "%.3f", control.noteGuidance))
    mix(String(format: "%.3f", control.drumGuidance))
    return hash
  }

  private func stftChecksum(_ stft: MLMultiArray) -> Int {
    let shape = stft.shape.map(\.intValue)
    let strides = stft.strides.map(\.intValue)
    var denseStrides = [Int](repeating: 1, count: shape.count)
    if shape.count > 1 {
      for dimension in stride(from: shape.count - 2, through: 0, by: -1) {
        denseStrides[dimension] = denseStrides[dimension + 1] * shape[dimension + 1]
      }
    }
    let capacity = zip(shape, strides).reduce(1) { $0 + ($1.0 - 1) * $1.1 }
    func physicalOffset(for logicalIndex: Int) -> Int {
      var remainder = logicalIndex
      var offset = 0
      for dimension in shape.indices {
        offset += (remainder / denseStrides[dimension]) * strides[dimension]
        remainder %= denseStrides[dimension]
      }
      return offset
    }
    var checksum = 0
    switch stft.dataType {
    case .float32:
      let pointer = stft.dataPointer.bindMemory(to: Float.self, capacity: capacity)
      for index in stride(from: 0, to: stft.count, by: 257) {
        checksum &+= Int(pointer[physicalOffset(for: index)].bitPattern)
      }
    case .float16:
      let pointer = stft.dataPointer.bindMemory(to: Float16.self, capacity: capacity)
      for index in stride(from: 0, to: stft.count, by: 257) {
        checksum &+= Int(pointer[physicalOffset(for: index)].bitPattern)
      }
    default:
      break
    }
    return checksum
  }

  private static func makeCacheNames() -> [String] {
    var names: [String] = []
    for layerIndex in 0..<12 {
      for kind in ["self", "cross"] {
        for role in ["key", "value"] {
          names.append(String(format: "temporal_layer_%02d_%@_%@_cache", layerIndex, kind, role))
        }
      }
    }
    return names
  }

  private static func writeDeterministicProofValues(
    into array: MLMultiArray,
    modulus: Int,
    scale: Float
  ) {
    let pointer = array.dataPointer.bindMemory(to: Float.self, capacity: array.count)
    let midpoint = modulus / 2
    for index in 0..<array.count {
      pointer[index] = Float((index % modulus) - midpoint) * scale
    }
  }

  private static func readFloat32Fixture(
    _ url: URL,
    expectedCount: Int
  ) throws -> [Float] {
    let data = try Data(contentsOf: url, options: .mappedIfSafe)
    let expectedBytes = expectedCount * MemoryLayout<Float>.stride
    guard data.count == expectedBytes else {
      throw CrossfadeRuntimeError.invalidArgument(
        "Fixture \(url.lastPathComponent) bytes=\(data.count), expected=\(expectedBytes)"
      )
    }
    var values = [Float](repeating: 0, count: expectedCount)
    _ = values.withUnsafeMutableBytes { destination in
      data.copyBytes(to: destination)
    }
    return values
  }

  private static func copyFloat32FixtureFrame(
    _ values: [Float],
    frame: Int,
    width: Int,
    into array: MLMultiArray
  ) {
    precondition(array.dataType == .float32 && array.count == width)
    let destination = array.dataPointer.bindMemory(to: Float.self, capacity: width)
    values.withUnsafeBufferPointer { source in
      destination.update(from: source.baseAddress!.advanced(by: frame * width), count: width)
    }
  }

  private static func hasNonzeroFloat16(_ array: MLMultiArray) -> Bool {
    let pointer = array.dataPointer.bindMemory(to: Float16.self, capacity: array.count)
    for index in 0..<array.count where pointer[index] != 0 {
      return true
    }
    return false
  }

  private static func floatChecksum(_ values: [Float]) -> Int {
    values.enumerated().reduce(0) { checksum, item in
      item.offset.isMultiple(of: 257)
        ? checksum &+ Int(item.element * 1_000)
        : checksum
    }
  }

  private static func makeZeroArray(shape: [Int], dataType: MLMultiArrayDataType) throws
    -> MLMultiArray
  {
    let array = try MLMultiArray(shape: shape.map(NSNumber.init(value:)), dataType: dataType)
    switch dataType {
    case .float16:
      zeroFloat16(array)
    case .float32:
      zeroFloat32(array)
    default:
      throw CrossfadeRuntimeError.unsupportedDataType(
        "Unsupported array data type: \(dataType.rawValue)")
    }
    return array
  }

  private func zeroFloat16(_ array: MLMultiArray) {
    Self.zeroFloat16(array)
  }

  private static func zeroFloat16(_ array: MLMultiArray) {
    let pointer = array.dataPointer.bindMemory(to: Float16.self, capacity: array.count)
    for index in 0..<array.count {
      pointer[index] = 0
    }
  }

  private func zeroFloat32(_ array: MLMultiArray) {
    Self.zeroFloat32(array)
  }

  private static func zeroFloat32(_ array: MLMultiArray) {
    let pointer = array.dataPointer.bindMemory(to: Float.self, capacity: array.count)
    for index in 0..<array.count {
      pointer[index] = 0
    }
  }

  private static func loadModel(
    resourceName: String,
    computeUnits: MLComputeUnits,
    bundle: Bundle
  ) throws -> MLModel {
    let url = try modelURL(resourceName: resourceName, bundle: bundle)
    return try loadModel(url: url, computeUnits: computeUnits)
  }

  private static func modelURL(resourceName: String, bundle: Bundle) throws -> URL {
    guard let url = bundle.url(forResource: resourceName, withExtension: "mlmodelc") else {
      throw CrossfadeRuntimeError.missingResource("\(resourceName).mlmodelc")
    }
    return url
  }

  private static func loadModel(url: URL, computeUnits: MLComputeUnits) throws -> MLModel {
    let configuration = MLModelConfiguration()
    configuration.computeUnits = computeUnits
    let started = ContinuousClock.now
    let name = url.deletingPathExtension().lastPathComponent
    logAdmission("modelLoadStarted model=\(name) computeUnits=\(computeUnits.rawValue)")
    let model = try MLModel(contentsOf: url, configuration: configuration)
    logAdmission(
      "modelLoadCompleted model=\(name) elapsedMs=\(milliseconds(since: started))"
    )
    return model
  }

  private static func logAdmission(_ message: String) {
    FileHandle.standardOutput.write(Data("CFGEN \(message)\n".utf8))
  }

  @available(iOS 17.4, *)
  private static func flattenOperations(
    _ operations: [MLModelStructure.Program.Operation]
  ) -> [MLModelStructure.Program.Operation] {
    var flattened: [MLModelStructure.Program.Operation] = []
    flattened.reserveCapacity(operations.count)
    for operation in operations {
      flattened.append(operation)
      for block in operation.blocks {
        flattened.append(contentsOf: flattenOperations(block.operations))
      }
    }
    return flattened
  }

  private static func milliseconds(since start: ContinuousClock.Instant) -> Double {
    let duration = start.duration(to: ContinuousClock.now)
    let components = duration.components
    return Double(components.seconds) * 1_000.0 + Double(components.attoseconds) / 1.0e15
  }

}

/// SpectroStream RVQ codebook table: maps 12 sampled token levels to one
/// 256-dimensional decoder embedding frame by summing per-level codewords.
///
/// Loads `spectrostream_rvq_codebooks_12_f32.bin` (exported by
/// scripts/export_mrt2_depth_body_coreml.py tooling) and validates its exact
/// byte count so a stale or truncated codebook fails loudly at init.
public final class CrossfadeRVQCodebooks {
  private static let levels = CrossfadeRuntimeResources.codebookLevels
  private static let codebookSize = CrossfadeRuntimeResources.codebookSize
  private static let embeddingDim = CrossfadeRuntimeResources.codebookEmbeddingDim

  private let values: [Float]

  public init(bundle: Bundle) throws {
    guard
      let url = bundle.url(
        forResource: CrossfadeRuntimeResources.codebookResourceStem,
        withExtension: CrossfadeRuntimeResources.codebookResourceExtension
      )
    else {
      throw CrossfadeRuntimeError.missingResource(CrossfadeRuntimeResources.codebookResourceName)
    }
    let data = try Data(contentsOf: url, options: [.mappedIfSafe])
    guard data.count == CrossfadeRuntimeResources.expectedCodebookBytes else {
      throw CrossfadeRuntimeError.invalidArgument(
        "Unexpected SpectroStream RVQ codebook byte count \(data.count), expected "
          + "\(CrossfadeRuntimeResources.expectedCodebookBytes)"
      )
    }
    values = data.withUnsafeBytes { rawBuffer in
      let pointer = rawBuffer.bindMemory(to: Float.self)
      return Array(pointer)
    }
  }

  public func embedding(rawTokens: [Int]) -> [Float] {
    var frame = Array(repeating: Float(0), count: Self.embeddingDim)
    for dimension in 0..<Self.embeddingDim {
      var sum: Float = 0
      for level in 0..<Self.levels {
        let rawToken = min(max(rawTokens[level], 0), Self.codebookSize - 1)
        let index = ((level * Self.codebookSize + rawToken) * Self.embeddingDim) + dimension
        sum += values[index]
      }
      frame[dimension] = sum
    }
    return frame
  }
}

/// MRT2 depthformer token-embedding table for autoregressive feedback.
///
/// Loads `mrt2_depth_embedder_f32.bin` (exported by
/// `scripts/export_mrt2_depth_embedder_for_ios.py`): little-endian float32
/// `[12294, 1024]`, the checkpoint's
/// `decoder_embedding/embedding/embedding` rows with the sampler's fixed
/// sqrt(1024) = 32.0 output scale baked in, so a row IS the embedder output.
///
/// Two consumers inside `CrossfadeGenerationRuntime`:
/// - temporal feedback: `temporal_inputs[frame] = mean over 12 levels of
///   row(uniqueToken)` for the PREVIOUS frame's sampled tokens (the MLX
///   sampler contract; see `validate_mrt2_depthformer_logits_coreml.py`).
/// - depth feedback: the step input for level k+1 is `row(uniqueToken(k))`
///   while rolling the depth transformer out across RVQ levels within one
///   frame (the full-pass graph's `depth_inputs[level+1]` contract).
///
/// Token ids are UNIQUE (global) ids: `6 + level*1024 + code`. Rows 0..5 are
/// the reserved tokens; the sampler's initial `previous_frame` is twelve id-0
/// tokens, so row 0 is load-bearing for frame 0.
public final class CrossfadeDepthEmbedder {
  private static let vocab = CrossfadeRuntimeResources.depthEmbedderVocab
  private static let dim = CrossfadeRuntimeResources.depthEmbedderDim

  private let values: [Float]

  public init(bundle: Bundle) throws {
    guard
      let url = bundle.url(
        forResource: CrossfadeRuntimeResources.depthEmbedderResourceStem,
        withExtension: CrossfadeRuntimeResources.depthEmbedderResourceExtension
      )
    else {
      throw CrossfadeRuntimeError.missingResource(
        CrossfadeRuntimeResources.depthEmbedderResourceName
      )
    }
    let data = try Data(contentsOf: url, options: [.mappedIfSafe])
    guard data.count == CrossfadeRuntimeResources.expectedDepthEmbedderBytes else {
      throw CrossfadeRuntimeError.invalidArgument(
        "Unexpected depth embedder byte count \(data.count), expected "
          + "\(CrossfadeRuntimeResources.expectedDepthEmbedderBytes)"
      )
    }
    values = data.withUnsafeBytes { rawBuffer in
      let pointer = rawBuffer.bindMemory(to: Float.self)
      return Array(pointer)
    }
  }

  /// Copies one embedding row into `destination` (must hold `dim` floats).
  public func copyEmbedding(uniqueToken: Int, into destination: UnsafeMutablePointer<Float>) {
    let token = min(max(uniqueToken, 0), Self.vocab - 1)
    let start = token * Self.dim
    for index in 0..<Self.dim {
      destination[index] = values[start + index]
    }
  }

  /// Writes the mean of the rows for `uniqueTokens` into `destination`
  /// (must hold `dim` floats). This is the temporal-input contract: the MLX
  /// sampler averages the previous frame's 12 level embeddings.
  public func copyMeanEmbedding(uniqueTokens: [Int], into destination: UnsafeMutablePointer<Float>)
  {
    for index in 0..<Self.dim {
      destination[index] = 0
    }
    guard !uniqueTokens.isEmpty else { return }
    for uniqueToken in uniqueTokens {
      let token = min(max(uniqueToken, 0), Self.vocab - 1)
      let start = token * Self.dim
      for index in 0..<Self.dim {
        destination[index] += values[start + index]
      }
    }
    let scale = 1 / Float(uniqueTokens.count)
    for index in 0..<Self.dim {
      destination[index] *= scale
    }
  }
}
