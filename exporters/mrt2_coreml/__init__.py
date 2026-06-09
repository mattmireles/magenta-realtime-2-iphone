# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PyTorch conversion wrappers for the MRT2 Core ML / ANE port.

These modules re-express the MRT2 (Magenta RealTime 2) temporal transformer,
depth transformer, and SpectroStream decoder as trace-friendly PyTorch graphs
so ``coremltools`` can convert them to ``mlprogram`` packages. They load
weights directly from the ``mrt2_small.safetensors`` checkpoint published at
https://huggingface.co/google/magenta-realtime-2 — the MLX stack is NOT
required for conversion.
"""
