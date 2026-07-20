#!/usr/bin/env python3
"""Aggregate the frozen MRT2 Phase 2 evidence into a private G5 candidate.

The aggregate is intentionally strict: only a complete 4 prompt x 3 seed x 2
mode MLX matrix, 12 hash-linked pair reports, and three control-valid unanimous
votes for each prompt can pass. Incomplete or internally inconsistent evidence
is ``inconclusive``; an authenticated catastrophic flag or unanimous prompt
failure is ``fail``.

Normalized judge-vote input schema (``mrt2-liveness-judge-votes-v2``)::

  {
    "schema": "mrt2-liveness-judge-votes-v2",
    "protocolSha256": "<sha256>",
    "prompts": [{
      "promptId": "warm",
      "votes": [{
        "voteId": "warm-50101",
        "lineupSeed": 50101,
        "inputManifest": {"path": "...", "sha256": "..."},
        "sealedMapping": {"path": "...", "sha256": "..."},
        "perSeedVerdicts": [{
          "generationSeed": 20260718,
          "refreshedControlPass": true,
          "corruptedControlPass": false,
          "candidateVerdict": "pass"
        }],
        "provenance": "primary",
        "artifacts": {
          "input.json": {"path": "...", "sha256": "..."},
          "result.json": {"path": "...", "sha256": "..."},
          "report.md": {"path": "...", "sha256": "..."},
          "workerMetadata": {"path": "...", "sha256": "..."}
        }
      }]
    }]
  }

Direct-Gemini votes retain the same lineup and per-seed fields and require both
the result JSON and the hash-bound request sidecar written by the fallback. A prompt becomes
``inconclusive`` when any seed-level vote fails control calibration, the three
frozen lineup seeds are not represented exactly once, or any seed's verdicts
disagree.  No prompt-level scalar may pool away a failed seed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import random
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
FROZEN_PROTOCOL_SHA256 = (
    "bfd0c7240e07c0f61ed9ee03bde9f07ad6ff4795b7db2c05288c063f1b7253ae"
)
FROZEN_PROTOCOL_STATUS = "frozen-before-candidate-generation"
PROTOCOL_SCHEMA = "mrt2-liveness-protocol-v1"
RUN_SCHEMA = "mrt2-long-horizon-token-run-v1"
PAIR_SCHEMA = "mrt2-liveness-pair-analysis-v1"
DECODE_SCHEMA = "mrt2-crossover-decode-v1"
PUBLIC_LINEUP_SCHEMA = "mrt2-liveness-opaque-prompt-lineup-v2"
SEALED_MAPPING_SCHEMA = "mrt2-liveness-sealed-vote-mapping-v2"
JUDGE_SCHEMA = "mrt2-liveness-judge-votes-v2"
OUTPUT_SCHEMA = "mrt2-liveness-g5-candidate-v1"

EXPECTED_PROMPT_COUNT = 4
EXPECTED_SEED_COUNT = 3
EXPECTED_MODES = ("refresh10", "unrefreshed")
EXPECTED_SECONDS = 600
EXPECTED_TOKEN_FRAMES = 15_001
EXPECTED_AUDIO_FRAMES = 28_800_000
EXPECTED_DECODER_CONTEXT_FRAMES = 12
EXPECTED_REFRESH_SECONDS = {"refresh10": 10.0, "unrefreshed": 0.0}
EXPECTED_RESET_COUNTS = {"refresh10": 60, "unrefreshed": 0}
EXPECTED_REFRESH_MODES = {"periodic": "refresh10", "off": "unrefreshed"}
EXPECTED_VOTES_PER_PROMPT = 3
PRIMARY_VOTE_ARTIFACTS = (
    "input.json",
    "result.json",
    "report.md",
    "workerMetadata",
)
DIRECT_GEMINI_VOTE_ARTIFACTS = ("request.json", "result.json")
DIRECT_GEMINI_REQUEST_SCHEMA = "mrt2-direct-gemini-audio-judge-request-v1"
ALLOWED_VOTE_PROVENANCE = {"primary", "direct-gemini"}
REQUIRED_REFERENCE_FLAGS = (
    "refreshedNonFinite",
    "refreshedDurationMismatch",
    "refreshedCaptureDiscontinuity",
)
REQUIRED_CATASTROPHIC_FLAGS = (
    "unrefreshedNonFiniteToken",
    "unrefreshedNonFinite",
    "unrefreshedDurationMismatch",
    "unrefreshedShorterThan600Seconds",
    "unrefreshedCaptureDiscontinuity",
    "unrefreshedExactRepeated30SecondWindow",
    "unrefreshedIntroducesClipping",
    "unrefreshedExactTokenCycle",
    "unrefreshedChannelCollapseRelativeToRefreshed",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _load_object_or_error(
    path: Path, *, label: str, errors: list[str]
) -> dict[str, Any]:
    try:
        return _load_object(path)
    except (OSError, ValueError) as error:
        errors.append(f"{label}: invalid JSON object ({error})")
        return {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _is_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _resolve_record_path(raw_path: object, owner_path: Path) -> Path | None:
    if not isinstance(raw_path, str) or not raw_path:
        return None
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    repo_candidate = REPO_ROOT / candidate
    if repo_candidate.exists():
        return repo_candidate
    return owner_path.parent / candidate


def _verify_file_record(
    record: object,
    *,
    owner_path: Path,
    label: str,
    errors: list[str],
) -> Path | None:
    if not isinstance(record, dict):
        errors.append(f"{label}: missing path/SHA-256 record")
        return None
    path = _resolve_record_path(record.get("path"), owner_path)
    expected = record.get("sha256")
    if path is None or not path.is_file():
        errors.append(f"{label}: referenced file is missing")
        return path
    if not _is_sha256(expected):
        errors.append(f"{label}: invalid SHA-256")
        return path
    actual = _sha256(path)
    if actual != expected:
        errors.append(f"{label}: SHA-256 mismatch ({actual} != {expected})")
    return path


def _protocol_contract(
    protocol: dict[str, Any], protocol_path: Path, errors: list[str]
) -> tuple[
    dict[tuple[str, int, str], dict[str, Any]],
    dict[str, dict[str, Any]],
    list[int],
]:
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        errors.append(f"protocol: schema must be {PROTOCOL_SCHEMA}")
    if protocol.get("status") != FROZEN_PROTOCOL_STATUS:
        errors.append(f"protocol: status must be {FROZEN_PROTOCOL_STATUS}")

    generation = protocol.get("generation")
    generation = generation if isinstance(generation, dict) else {}
    if generation.get("audibleSeconds") != EXPECTED_SECONDS:
        errors.append(f"protocol: audibleSeconds must be {EXPECTED_SECONDS}")
    if generation.get("expectedTokenFramesIncludingLookahead") != EXPECTED_TOKEN_FRAMES:
        errors.append(
            f"protocol: expectedTokenFramesIncludingLookahead must be {EXPECTED_TOKEN_FRAMES}"
        )
    if generation.get("decoderContextFrames") != EXPECTED_DECODER_CONTEXT_FRAMES:
        errors.append(
            f"protocol: decoderContextFrames must be {EXPECTED_DECODER_CONTEXT_FRAMES}"
        )
    _verify_file_record(
        protocol.get("checkpoint"),
        owner_path=protocol_path,
        label="protocol checkpoint",
        errors=errors,
    )

    seeds = protocol.get("seeds")
    seeds = seeds if isinstance(seeds, list) else []
    if (
        len(seeds) != EXPECTED_SEED_COUNT
        or len(set(seeds)) != EXPECTED_SEED_COUNT
        or not all(
            isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds
        )
    ):
        errors.append("protocol: seeds must contain exactly three unique integers")

    fixtures = protocol.get("fixtures")
    fixtures = fixtures if isinstance(fixtures, list) else []
    fixture_map: dict[str, dict[str, Any]] = {}
    for fixture in fixtures:
        if not isinstance(fixture, dict) or not isinstance(fixture.get("id"), str):
            errors.append("protocol: every fixture must have a string id")
            continue
        prompt_id = fixture["id"]
        if prompt_id in fixture_map:
            errors.append(f"protocol: duplicate fixture {prompt_id}")
            continue
        fixture_map[prompt_id] = fixture
        source = fixture.get("sourceConditioning")
        if not isinstance(source, dict) or not _is_sha256(source.get("sha256")):
            errors.append(
                f"protocol fixture {prompt_id}: invalid source-conditioning hash"
            )
            continue
        fixture_path = _resolve_record_path(source.get("path"), protocol_path)
        if fixture_path is None or not fixture_path.is_file():
            errors.append(
                f"protocol fixture {prompt_id}: source-conditioning file is missing"
            )
        elif _sha256(fixture_path) != source["sha256"]:
            errors.append(
                f"protocol fixture {prompt_id}: source-conditioning hash mismatch"
            )
    if len(fixture_map) != EXPECTED_PROMPT_COUNT:
        errors.append(f"protocol: expected {EXPECTED_PROMPT_COUNT} unique fixtures")

    arms = protocol.get("arms")
    arms = arms if isinstance(arms, list) else []
    expected: dict[tuple[str, int, str], dict[str, Any]] = {}
    for arm in arms:
        if not isinstance(arm, dict):
            errors.append("protocol: every arm must be an object")
            continue
        prompt_id = arm.get("promptId")
        seed = arm.get("seed")
        mode = arm.get("mode")
        if (
            not isinstance(prompt_id, str)
            or not isinstance(seed, int)
            or isinstance(seed, bool)
            or mode not in EXPECTED_MODES
        ):
            errors.append(f"protocol: invalid arm identity {arm!r}")
            continue
        key = (prompt_id, seed, mode)
        if key in expected:
            errors.append(f"protocol: duplicate arm {key}")
            continue
        expected[key] = arm
        fixture = fixture_map.get(prompt_id, {})
        fixture_hash = fixture.get("sourceConditioning", {}).get("sha256")
        if arm.get("sourceConditioningSha256") != fixture_hash:
            errors.append(f"protocol arm {key}: fixture hash mismatch")
        if arm.get("refreshSeconds") != EXPECTED_REFRESH_SECONDS[mode]:
            errors.append(f"protocol arm {key}: wrong refresh interval")

    expected_count = EXPECTED_PROMPT_COUNT * EXPECTED_SEED_COUNT * len(EXPECTED_MODES)
    if len(expected) != expected_count:
        errors.append(
            f"protocol: expected {expected_count} unique arms, got {len(expected)}"
        )
    for prompt_id in fixture_map:
        for seed in seeds:
            for mode in EXPECTED_MODES:
                if (prompt_id, seed, mode) not in expected:
                    errors.append(f"protocol: missing arm {(prompt_id, seed, mode)}")
    return expected, fixture_map, seeds


def _arm_identity(
    manifest: dict[str, Any], fixture_map: dict[str, dict[str, Any]]
) -> tuple[str, int, str] | None:
    prompt_matches = [
        prompt_id
        for prompt_id, fixture in fixture_map.items()
        if manifest.get("prompt") == fixture.get("prompt")
    ]
    seed = manifest.get("seed")
    mode = EXPECTED_REFRESH_MODES.get(manifest.get("refreshMode"))
    if (
        len(prompt_matches) != 1
        or not isinstance(seed, int)
        or isinstance(seed, bool)
        or mode is None
    ):
        return None
    return prompt_matches[0], seed, mode


def _collect_arms(
    arm_paths: list[Path],
    expected: dict[tuple[str, int, str], dict[str, Any]],
    fixture_map: dict[str, dict[str, Any]],
    protocol: dict[str, Any],
    errors: list[str],
) -> tuple[
    dict[tuple[str, int, str], tuple[Path, dict[str, Any]]],
    dict[tuple[str, int, str], str],
]:
    found: dict[tuple[str, int, str], tuple[Path, dict[str, Any]]] = {}
    digests: dict[tuple[str, int, str], str] = {}
    checkpoint_hash = protocol.get("checkpoint", {}).get("sha256")
    for path in arm_paths:
        manifest = _load_object_or_error(
            path, label=f"arm manifest {path}", errors=errors
        )
        key = _arm_identity(manifest, fixture_map)
        if key is None:
            errors.append(f"{path}: cannot map manifest to a frozen prompt/seed/mode")
            continue
        if key in found:
            errors.append(f"duplicate arm manifest {key}: {found[key][0]} and {path}")
            continue
        found[key] = (path, manifest)
        digests[key] = _sha256(path)
        if key not in expected:
            errors.append(f"{path}: arm {key} is not in the frozen protocol")
            continue
        if manifest.get("schema") != RUN_SCHEMA:
            errors.append(f"arm {key}: schema must be {RUN_SCHEMA}")
        if manifest.get("tokenSource") != "mlx":
            errors.append(f"arm {key}: tokenSource must be mlx")
        if manifest.get("requestedAudibleSeconds") != EXPECTED_SECONDS:
            errors.append(
                f"arm {key}: requestedAudibleSeconds must be {EXPECTED_SECONDS}"
            )
        if manifest.get("tokenFrames") != EXPECTED_TOKEN_FRAMES:
            errors.append(f"arm {key}: tokenFrames must be {EXPECTED_TOKEN_FRAMES}")
        mode = key[2]
        if manifest.get("trajectoryRefreshSeconds") != EXPECTED_REFRESH_SECONDS[mode]:
            errors.append(f"arm {key}: wrong trajectoryRefreshSeconds")
        if manifest.get("temporalResetCount") != EXPECTED_RESET_COUNTS[mode]:
            errors.append(f"arm {key}: wrong temporalResetCount")
        if (
            not _is_finite_number(manifest.get("elapsedSeconds"))
            or manifest["elapsedSeconds"] <= 0
        ):
            errors.append(f"arm {key}: elapsedSeconds must be finite and positive")
        inputs = manifest.get("inputs")
        inputs = inputs if isinstance(inputs, dict) else {}
        expected_fixture_hash = expected[key].get("sourceConditioningSha256")
        if inputs.get("sourceConditioning", {}).get("sha256") != expected_fixture_hash:
            errors.append(
                f"arm {key}: source-conditioning hash does not match protocol"
            )
        if inputs.get("checkpoint", {}).get("sha256") != checkpoint_hash:
            errors.append(f"arm {key}: checkpoint hash does not match protocol")
        _verify_file_record(
            manifest.get("tokens"),
            owner_path=path,
            label=f"arm {key} tokens",
            errors=errors,
        )
        _verify_file_record(
            manifest.get("summary"),
            owner_path=path,
            label=f"arm {key} summary",
            errors=errors,
        )
    for key in expected:
        if key not in found:
            errors.append(f"missing arm manifest {key}")
    return found, digests


def _collect_pairs(
    pair_paths: list[Path],
    *,
    protocol_digest: str,
    expected_arms: dict[tuple[str, int, str], dict[str, Any]],
    fixture_map: dict[str, dict[str, Any]],
    arm_records: dict[tuple[str, int, str], tuple[Path, dict[str, Any]]],
    arm_digests: dict[tuple[str, int, str], str],
    checkpoint_sha256: object,
    errors: list[str],
) -> tuple[dict[tuple[str, int], tuple[Path, dict[str, Any]]], list[dict[str, Any]]]:
    expected_pairs = {(prompt_id, seed) for prompt_id, seed, _ in expected_arms}
    prompt_by_text = {
        fixture.get("prompt"): prompt_id for prompt_id, fixture in fixture_map.items()
    }
    found: dict[tuple[str, int], tuple[Path, dict[str, Any]]] = {}
    catastrophic: list[dict[str, Any]] = []
    for path in pair_paths:
        report = _load_object_or_error(path, label=f"pair report {path}", errors=errors)
        invariants = report.get("pairInvariants")
        invariants = invariants if isinstance(invariants, dict) else {}
        prompt_id = prompt_by_text.get(invariants.get("prompt"))
        seed = invariants.get("seed")
        key = (
            (prompt_id, seed)
            if isinstance(prompt_id, str) and isinstance(seed, int)
            else None
        )
        if key is None:
            errors.append(f"{path}: cannot map pair report to a frozen prompt/seed")
            continue
        if key in found:
            errors.append(f"duplicate pair report {key}: {found[key][0]} and {path}")
            continue
        found[key] = (path, report)
        if key not in expected_pairs:
            errors.append(f"{path}: pair {key} is not in the frozen protocol")
            continue
        if report.get("schema") != PAIR_SCHEMA:
            errors.append(f"pair {key}: schema must be {PAIR_SCHEMA}")
        if report.get("protocolSha256") != protocol_digest:
            errors.append(f"pair {key}: protocol hash mismatch")
        manifest_records = report.get("manifests")
        manifest_records = (
            manifest_records if isinstance(manifest_records, dict) else {}
        )
        wav_records = report.get("wavs")
        wav_records = wav_records if isinstance(wav_records, dict) else {}
        receipt_records = report.get("decodeReceipts")
        receipt_records = receipt_records if isinstance(receipt_records, dict) else {}
        for label, mode in (
            ("refreshed", "refresh10"),
            ("unrefreshed", "unrefreshed"),
        ):
            arm_key = (key[0], key[1], mode)
            record = manifest_records.get(label)
            expected_digest = arm_digests.get(arm_key)
            if not isinstance(record, dict) or record.get("sha256") != expected_digest:
                errors.append(
                    f"pair {key}: {label} manifest is not hash-bound to arm {arm_key}"
                )
            manifest_path = _verify_file_record(
                record,
                owner_path=path,
                label=f"pair {key} {label} manifest",
                errors=errors,
            )
            expected_manifest = arm_records.get(arm_key)
            if (
                manifest_path is not None
                and expected_manifest is not None
                and manifest_path.resolve() != expected_manifest[0].resolve()
            ):
                errors.append(f"pair {key}: {label} manifest path does not match arm")

            wav_path = _verify_file_record(
                wav_records.get(label),
                owner_path=path,
                label=f"pair {key} {label} WAV",
                errors=errors,
            )
            receipt_path = _verify_file_record(
                receipt_records.get(label),
                owner_path=path,
                label=f"pair {key} {label} decode receipt",
                errors=errors,
            )
            if receipt_path is None or not receipt_path.is_file():
                continue
            receipt = _load_object_or_error(
                receipt_path,
                label=f"pair {key} {label} decode receipt",
                errors=errors,
            )
            if receipt.get("schema") != DECODE_SCHEMA:
                errors.append(f"pair {key}: {label} decode schema must be {DECODE_SCHEMA}")
            if receipt.get("decoderContextFrames") != EXPECTED_DECODER_CONTEXT_FRAMES:
                errors.append(f"pair {key}: {label} decode is not context-12")
            if receipt.get("renderedFrames") != EXPECTED_AUDIO_FRAMES:
                errors.append(f"pair {key}: {label} decode frame count is wrong")
            if receipt.get("durationSeconds") != float(EXPECTED_SECONDS):
                errors.append(f"pair {key}: {label} decode duration is wrong")
            manifest = expected_manifest[1] if expected_manifest is not None else {}
            expected_token_sha256 = manifest.get("tokens", {}).get("sha256")
            if receipt.get("tokenSha256") != expected_token_sha256:
                errors.append(f"pair {key}: {label} decode token hash mismatch")
            receipt_token_path = _resolve_record_path(
                receipt.get("tokenPath"), receipt_path
            )
            manifest_token_path = _resolve_record_path(
                manifest.get("tokens", {}).get("path"), expected_manifest[0]
            ) if expected_manifest is not None else None
            if (
                receipt_token_path is None
                or manifest_token_path is None
                or not receipt_token_path.is_file()
                or receipt_token_path.resolve() != manifest_token_path.resolve()
            ):
                errors.append(f"pair {key}: {label} decode token path mismatch")
            if receipt.get("checkpointSha256") != checkpoint_sha256:
                errors.append(f"pair {key}: {label} decode checkpoint hash mismatch")
            receipt_wav_path = _verify_file_record(
                receipt.get("wav"),
                owner_path=receipt_path,
                label=f"pair {key} {label} receipt WAV",
                errors=errors,
            )
            if (
                wav_path is not None
                and receipt_wav_path is not None
                and wav_path.resolve() != receipt_wav_path.resolve()
            ):
                errors.append(f"pair {key}: {label} receipt WAV path mismatch")
            if isinstance(receipt.get("wav"), dict) and isinstance(
                wav_records.get(label), dict
            ):
                if receipt["wav"].get("sha256") != wav_records[label].get("sha256"):
                    errors.append(f"pair {key}: {label} receipt WAV hash mismatch")
        flags = report.get("catastrophicIntegrityFailures")
        flags = flags if isinstance(flags, dict) else {}
        missing_flags = [
            name for name in REQUIRED_CATASTROPHIC_FLAGS if name not in flags
        ]
        if missing_flags:
            errors.append(f"pair {key}: missing catastrophic flags {missing_flags}")
        for name, value in flags.items():
            if not isinstance(value, bool):
                errors.append(f"pair {key}: catastrophic flag {name} must be boolean")
            elif value:
                catastrophic.append({"promptId": key[0], "seed": key[1], "flag": name})
        reference_flags = report.get("refreshedReferenceIntegrityFailures")
        reference_flags = reference_flags if isinstance(reference_flags, dict) else {}
        missing_reference = [
            name for name in REQUIRED_REFERENCE_FLAGS if name not in reference_flags
        ]
        if missing_reference:
            errors.append(f"pair {key}: missing refreshed reference flags {missing_reference}")
        for name, value in reference_flags.items():
            if not isinstance(value, bool):
                errors.append(f"pair {key}: refreshed reference flag {name} must be boolean")
            elif value:
                errors.append(f"pair {key}: refreshed reference integrity failure {name}")
        status = report.get("catastrophicIntegrityStatus")
        true_flags = sorted(name for name, value in flags.items() if value is True)
        true_reference_flags = sorted(
            name for name, value in reference_flags.items() if value is True
        )
        all_failure_ids = true_reference_flags + true_flags
        if not isinstance(status, dict) or status.get("allChecksAssessed") is not True:
            errors.append(f"pair {key}: hardened catastrophic status is missing")
        elif (
            status.get("candidateFailureIds") != true_flags
            or status.get("referenceFailureIds") != true_reference_flags
            or status.get("failureIds") != all_failure_ids
            or status.get("verdict") != ("fail" if all_failure_ids else "pass")
            or status.get("passed") is not (not all_failure_ids)
        ):
            errors.append(f"pair {key}: catastrophic status contradicts its flags")
    for key in expected_pairs:
        if key not in found:
            errors.append(f"missing pair report {key}")
    return found, catastrophic


def _verify_lineup_evidence(
    vote: dict[str, Any],
    *,
    vote_index: int,
    prompt_id: str,
    protocol_digest: str,
    frozen_lineup_seed: int,
    frozen_condition_order: list[str],
    seeds: list[int],
    judge_votes_path: Path,
    issues: list[str],
) -> dict[str, Any] | None:
    prefix = f"vote {vote_index}"
    input_path = _verify_file_record(
        vote.get("inputManifest"),
        owner_path=judge_votes_path,
        label=f"{prefix} input manifest",
        errors=issues,
    )
    mapping_path = _verify_file_record(
        vote.get("sealedMapping"),
        owner_path=judge_votes_path,
        label=f"{prefix} sealed mapping",
        errors=issues,
    )
    if input_path is None or mapping_path is None:
        return None
    public = _load_object_or_error(
        input_path, label=f"{prefix} input manifest", errors=issues
    )
    sealed = _load_object_or_error(
        mapping_path, label=f"{prefix} sealed mapping", errors=issues
    )
    if public.get("schema") != PUBLIC_LINEUP_SCHEMA:
        issues.append(f"{prefix}: input manifest schema must be {PUBLIC_LINEUP_SCHEMA}")
    if sealed.get("schema") != SEALED_MAPPING_SCHEMA:
        issues.append(f"{prefix}: sealed mapping schema must be {SEALED_MAPPING_SCHEMA}")
    for label, value in (("input manifest", public), ("sealed mapping", sealed)):
        if value.get("protocolSha256") != protocol_digest:
            issues.append(f"{prefix}: {label} protocol hash mismatch")
        if value.get("promptId") != prompt_id:
            issues.append(f"{prefix}: {label} prompt mismatch")
        if value.get("voteIndex") != vote_index:
            issues.append(f"{prefix}: {label} vote index mismatch")
    if sealed.get("lineupSeed") != frozen_lineup_seed:
        issues.append(f"{prefix}: sealed lineup seed does not match protocol")
    if vote.get("lineupSeed") != frozen_lineup_seed:
        issues.append(f"{prefix}: vote lineup seed does not match protocol")
    if sealed.get("frozenConditionOrder") != frozen_condition_order:
        issues.append(f"{prefix}: sealed condition order does not match protocol")
    expected_seed_order = random.Random(frozen_lineup_seed).sample(seeds, k=len(seeds))
    if sealed.get("generationSeedOrder") != expected_seed_order:
        issues.append(f"{prefix}: generation seed order does not match frozen seed")

    sealed_input = sealed.get("inputManifest")
    if not isinstance(sealed_input, dict):
        issues.append(f"{prefix}: sealed input-manifest record is missing")
    else:
        sealed_input_path = _resolve_record_path(sealed_input.get("path"), mapping_path)
        if (
            sealed_input_path is None
            or sealed_input_path.resolve() != input_path.resolve()
            or sealed_input.get("sha256") != _sha256(input_path)
        ):
            issues.append(f"{prefix}: sealed input-manifest record mismatch")

    clips = public.get("clips")
    clips = clips if isinstance(clips, list) else []
    clip_by_label: dict[str, dict[str, Any]] = {}
    opaque_order = []
    for clip_index, clip in enumerate(clips):
        if not isinstance(clip, dict) or not isinstance(clip.get("label"), str):
            issues.append(f"{prefix}: clip {clip_index} is invalid")
            continue
        label = clip["label"]
        if label in clip_by_label:
            issues.append(f"{prefix}: duplicate clip label {label}")
            continue
        clip_by_label[label] = clip
        _verify_file_record(
            clip,
            owner_path=input_path,
            label=f"{prefix} clip {label}",
            errors=issues,
        )
        opaque_order.append(
            {
                "position": clip_index,
                "label": label,
                "seedGroup": clip.get("seedGroup"),
                "sha256": clip.get("sha256"),
            }
        )
    if len(clip_by_label) != len(seeds) * 4:
        issues.append(f"{prefix}: lineup must contain exactly 12 unique clips")
    opaque_digest = _json_sha256(opaque_order)
    if public.get("opaqueOrderSha256") != opaque_digest:
        issues.append(f"{prefix}: public opaque-order hash mismatch")
    if sealed.get("opaqueOrderSha256") != opaque_digest:
        issues.append(f"{prefix}: sealed opaque-order hash mismatch")

    mapping = sealed.get("mapping")
    mapping = mapping if isinstance(mapping, dict) else {}
    if sealed.get("mappingSha256") != _json_sha256(mapping):
        issues.append(f"{prefix}: sealed mapping hash mismatch")
    if set(mapping) != set(clip_by_label):
        issues.append(f"{prefix}: sealed mapping does not cover the public clips exactly")
    for label, entry in mapping.items():
        if not isinstance(entry, dict):
            issues.append(f"{prefix}: mapping for {label} must be an object")
            continue
        clip = clip_by_label.get(label, {})
        if entry.get("clipSha256") != clip.get("sha256"):
            issues.append(f"{prefix}: mapping clip hash mismatch for {label}")
        if entry.get("condition") not in {
            "refreshed",
            "unrefreshed",
            "context0",
            "corrupted",
        }:
            issues.append(f"{prefix}: mapping condition is invalid for {label}")
        if entry.get("generationSeed") not in seeds:
            issues.append(f"{prefix}: mapping seed is invalid for {label}")
        if entry.get("seedGroup") != clip.get("seedGroup"):
            issues.append(f"{prefix}: mapping seed group mismatch for {label}")

    per_seed = sealed.get("perSeedEntries")
    per_seed = per_seed if isinstance(per_seed, list) else []
    if [row.get("generationSeed") for row in per_seed if isinstance(row, dict)] != expected_seed_order:
        issues.append(f"{prefix}: sealed per-seed entries do not match frozen order")
    seed_entries: dict[int, dict[str, str]] = {}
    for row in per_seed:
        if not isinstance(row, dict):
            continue
        seed = row.get("generationSeed")
        labels = row.get("labelsByCondition")
        if not isinstance(seed, int) or not isinstance(labels, dict):
            issues.append(f"{prefix}: invalid sealed per-seed entry")
            continue
        if set(labels) != {"refreshed", "unrefreshed", "context0", "corrupted"}:
            issues.append(f"{prefix}: seed {seed} does not map all four conditions")
            continue
        if len(set(labels.values())) != 4 or not set(labels.values()).issubset(clip_by_label):
            issues.append(f"{prefix}: seed {seed} condition labels are invalid")
            continue
        group = row.get("groupLabel")
        if any(
            mapping.get(label, {}).get("generationSeed") != seed
            or mapping.get(label, {}).get("condition") != condition
            or mapping.get(label, {}).get("seedGroup") != group
            for condition, label in labels.items()
        ):
            issues.append(f"{prefix}: seed {seed} entry contradicts sealed mapping")
        seed_entries[seed] = labels
    if set(seed_entries) != set(seeds):
        issues.append(f"{prefix}: sealed per-seed entries do not cover frozen seeds")

    public_groups = public.get("seedGroups")
    public_groups = public_groups if isinstance(public_groups, list) else []
    public_group_map = {
        row.get("groupLabel"): row for row in public_groups if isinstance(row, dict)
    }
    if len(public_group_map) != len(seeds):
        issues.append(f"{prefix}: public seed groups are incomplete")
    for row in per_seed:
        if not isinstance(row, dict):
            continue
        group = public_group_map.get(row.get("groupLabel"), {})
        labels = row.get("labelsByCondition", {})
        if set(group.get("clipLabels", [])) != set(labels.values()):
            issues.append(f"{prefix}: public group clips contradict sealed mapping")
        if group.get("baselineLabel") != labels.get("refreshed"):
            issues.append(f"{prefix}: public baseline does not match refreshed control")
    expected_baselines = [
        row.get("labelsByCondition", {}).get("refreshed")
        for row in per_seed
        if isinstance(row, dict)
    ]
    if public.get("baselineLabels") != expected_baselines:
        issues.append(f"{prefix}: public baseline list contradicts sealed mapping")
    if expected_baselines and public.get("baselineLabel") != expected_baselines[0]:
        issues.append(f"{prefix}: primary public baseline is wrong")
    neutral_context_path = _verify_file_record(
        public.get("neutralContextFile"),
        owner_path=input_path,
        label=f"{prefix} neutral-context file",
        errors=issues,
    )
    neutral_context = public.get("neutralContext")
    if not isinstance(neutral_context, str):
        issues.append(f"{prefix}: public neutral context must be a string")
    elif neutral_context_path is not None and neutral_context_path.is_file():
        try:
            context_bytes = neutral_context_path.read_bytes()
            decoded_context = context_bytes.decode("utf-8").strip()
            if decoded_context != neutral_context:
                issues.append(
                    f"{prefix}: neutral-context file bytes do not equal public context"
                )
        except (OSError, UnicodeDecodeError) as error:
            issues.append(f"{prefix}: cannot read neutral-context bytes ({error})")
    return {"seedEntries": seed_entries, "public": public}


def _verify_vote_artifacts(
    vote: dict[str, Any],
    *,
    vote_index: int,
    judge_votes_path: Path,
    public: dict[str, Any],
    issues: list[str],
) -> dict[str, str]:
    provenance = vote.get("provenance")
    if provenance not in ALLOWED_VOTE_PROVENANCE:
        issues.append(f"vote {vote_index}: unsupported provenance")
        return {}
    required = (
        PRIMARY_VOTE_ARTIFACTS
        if provenance == "primary"
        else DIRECT_GEMINI_VOTE_ARTIFACTS
    )
    artifacts = vote.get("artifacts")
    artifacts = artifacts if isinstance(artifacts, dict) else {}
    if set(artifacts) != set(required):
        issues.append(
            f"vote {vote_index}: {provenance} artifacts must be exactly {list(required)}"
        )
    paths: dict[str, Path] = {}
    for name in required:
        issue_count = len(issues)
        path = _verify_file_record(
            artifacts.get(name),
            owner_path=judge_votes_path,
            label=f"vote {vote_index} artifact {name}",
            errors=issues,
        )
        if path is not None and len(issues) == issue_count:
            paths[name] = path

    if provenance == "primary" and "input.json" in paths:
        workflow_input = _load_object_or_error(
            paths["input.json"],
            label=f"vote {vote_index} primary input artifact",
            errors=issues,
        )
        workflow_clips = workflow_input.get("clips")
        workflow_clips = workflow_clips if isinstance(workflow_clips, list) else []
        public_clips = public.get("clips")
        public_clips = public_clips if isinstance(public_clips, list) else []
        workflow_sources = [
            (
                row.get("label"),
                row.get("source", {}).get("sha256")
                if isinstance(row, dict) and isinstance(row.get("source"), dict)
                else None,
            )
            for row in workflow_clips
            if isinstance(row, dict)
        ]
        public_sources = [
            (row.get("label"), row.get("sha256"))
            for row in public_clips
            if isinstance(row, dict)
        ]
        if workflow_sources != public_sources:
            issues.append(f"vote {vote_index}: primary input clips do not match lineup")
        if workflow_input.get("prompt") != public.get("prompt"):
            issues.append(f"vote {vote_index}: primary input prompt does not match lineup")
        if workflow_input.get("baselineLabel") != public.get("baselineLabel"):
            issues.append(f"vote {vote_index}: primary input baseline does not match lineup")
        if workflow_input.get("context") != public.get("neutralContext"):
            issues.append(f"vote {vote_index}: primary input context does not match lineup")
    if provenance == "primary" and "result.json" in paths:
        result = _load_object_or_error(
            paths["result.json"],
            label=f"vote {vote_index} primary result artifact",
            errors=issues,
        )
        if result.get("workflowId") != "audio_judge_v1":
            issues.append(
                f"vote {vote_index}: primary result workflowId must be audio_judge_v1"
            )
        if result.get("status") != "completed":
            issues.append(f"vote {vote_index}: primary result status must be completed")
        result_run_id = result.get("runId")
        if not isinstance(result_run_id, str) or not result_run_id:
            issues.append(f"vote {vote_index}: primary result runId is missing")
        if "workerMetadata" in paths:
            worker = _load_object_or_error(
                paths["workerMetadata"],
                label=f"vote {vote_index} worker metadata",
                errors=issues,
            )
            accepted = worker.get("acceptedRun")
            latest = worker.get("latestRun")
            accepted = accepted if isinstance(accepted, dict) else {}
            latest = latest if isinstance(latest, dict) else {}
            worker_run_ids = {accepted.get("runId"), latest.get("runId")}
            if result_run_id not in worker_run_ids:
                issues.append(
                    f"vote {vote_index}: primary result runId is not linked to worker metadata"
                )
            if latest.get("runId") != result_run_id or latest.get("status") != "completed":
                issues.append(
                    f"vote {vote_index}: worker latestRun must identify the completed result"
                )
    if provenance == "direct-gemini" and "request.json" in paths:
        request = _load_object_or_error(
            paths["request.json"],
            label=f"vote {vote_index} direct request artifact",
            errors=issues,
        )
        if request.get("schema") != DIRECT_GEMINI_REQUEST_SCHEMA:
            issues.append(
                f"vote {vote_index}: direct request schema must be "
                f"{DIRECT_GEMINI_REQUEST_SCHEMA}"
            )
        request_clips = request.get("clips")
        request_clips = request_clips if isinstance(request_clips, list) else []
        public_clips = public.get("clips")
        public_clips = public_clips if isinstance(public_clips, list) else []
        request_sources = [
            (row.get("label"), row.get("sha256"))
            for row in request_clips
            if isinstance(row, dict)
        ]
        public_sources = [
            (row.get("label"), row.get("sha256"))
            for row in public_clips
            if isinstance(row, dict)
        ]
        if request_sources != public_sources:
            issues.append(f"vote {vote_index}: direct request clips do not match lineup")
        if request.get("prompt") != public.get("prompt"):
            issues.append(f"vote {vote_index}: direct request prompt does not match lineup")
        if request.get("baselineLabel") != public.get("baselineLabel"):
            issues.append(f"vote {vote_index}: direct request baseline does not match lineup")
        if request.get("context") != public.get("neutralContext"):
            issues.append(f"vote {vote_index}: direct request context does not match lineup")
        if "result.json" in paths:
            direct_result = _load_object_or_error(
                paths["result.json"],
                label=f"vote {vote_index} direct result artifact",
                errors=issues,
            )
            if direct_result.get("requestSha256") != _sha256(paths["request.json"]):
                issues.append(
                    f"vote {vote_index}: direct result is not hash-linked to request"
                )
    return {name: str(path) for name, path in paths.items()}


def _artifact_clip_verdicts(
    *,
    provenance: object,
    result_path: Path | None,
    vote_index: int,
    expected_labels: set[str],
    issues: list[str],
) -> dict[str, str]:
    if result_path is None:
        return {}
    result = _load_object_or_error(
        result_path, label=f"vote {vote_index} result artifact", errors=issues
    )
    verdicts: dict[str, str] = {}
    if provenance == "primary":
        output = result.get("output")
        rows = output.get("clips") if isinstance(output, dict) else None
        rows = rows if isinstance(rows, list) else []
        for row in rows:
            if isinstance(row, dict) and isinstance(row.get("label"), str):
                label = row["label"]
                if label in verdicts:
                    issues.append(f"vote {vote_index}: duplicate artifact verdict for {label}")
                verdicts[label] = row.get("verdict")
    else:
        rows = result.get("clips")
        rows = rows if isinstance(rows, dict) else {}
        for label, row in rows.items():
            if isinstance(label, str) and isinstance(row, dict):
                verdicts[label] = row.get("verdict")
    if set(verdicts) != expected_labels:
        issues.append(f"vote {vote_index}: result verdicts do not cover lineup clips exactly")
    for label, verdict in verdicts.items():
        if verdict not in {"pass", "fail"}:
            issues.append(f"vote {vote_index}: invalid artifact verdict for {label}")
    return verdicts


def _judge_prompt_verdicts(
    judge: dict[str, Any],
    *,
    judge_votes_path: Path,
    protocol_digest: str,
    protocol: dict[str, Any],
    fixture_map: dict[str, dict[str, Any]],
    seeds: list[int],
    errors: list[str],
) -> list[dict[str, Any]]:
    if judge.get("schema") != JUDGE_SCHEMA:
        errors.append(f"judge votes: schema must be {JUDGE_SCHEMA}")
    if judge.get("protocolSha256") != protocol_digest:
        errors.append("judge votes: protocol hash mismatch")
    rows = judge.get("prompts")
    rows = rows if isinstance(rows, list) else []
    by_prompt: dict[str, dict[str, Any]] = {}
    for row in rows:
        prompt_id = row.get("promptId") if isinstance(row, dict) else None
        if not isinstance(prompt_id, str):
            errors.append("judge votes: every prompt row must have a string promptId")
            continue
        if prompt_id in by_prompt:
            errors.append(f"judge votes: duplicate prompt row {prompt_id}")
            continue
        by_prompt[prompt_id] = row
    unknown = sorted(set(by_prompt) - set(fixture_map))
    if unknown:
        errors.append(f"judge votes: unknown prompts {unknown}")

    audio_judge = protocol.get("audioJudge")
    audio_judge = audio_judge if isinstance(audio_judge, dict) else {}
    lineup_seeds = audio_judge.get("lineupSeedsByPrompt", {})
    lineup_orders = audio_judge.get("lineupOrdersByPrompt", {})
    results: list[dict[str, Any]] = []
    for prompt_id in fixture_map:
        issues: list[str] = []
        row = by_prompt.get(prompt_id, {})
        votes = row.get("votes") if isinstance(row, dict) else None
        votes = votes if isinstance(votes, list) else []
        expected_lineup_seeds = lineup_seeds.get(prompt_id, [])
        expected_lineup_orders = lineup_orders.get(prompt_id, [])
        if len(votes) != EXPECTED_VOTES_PER_PROMPT:
            issues.append(
                f"expected {EXPECTED_VOTES_PER_PROMPT} votes, got {len(votes)}"
            )
        vote_ids: set[str] = set()
        observed_lineup_seeds: list[int] = []
        valid_vote_count = 0
        verdicts_by_seed: dict[int, list[str]] = {seed: [] for seed in seeds}
        for index, vote in enumerate(votes):
            if not isinstance(vote, dict):
                issues.append(f"vote {index + 1}: must be an object")
                continue
            vote_number = index + 1
            issue_count_before_vote = len(issues)
            vote_id = vote.get("voteId")
            if not isinstance(vote_id, str) or not vote_id:
                issues.append(f"vote {vote_number}: missing voteId")
            elif vote_id in vote_ids:
                issues.append(f"vote {vote_number}: duplicate voteId {vote_id}")
            else:
                vote_ids.add(vote_id)
            if index >= len(expected_lineup_seeds) or index >= len(expected_lineup_orders):
                issues.append(f"vote {vote_number}: protocol lineup entry is missing")
                continue
            expected_lineup_seed = expected_lineup_seeds[index]
            observed_lineup_seeds.append(vote.get("lineupSeed"))
            lineup = _verify_lineup_evidence(
                vote,
                vote_index=vote_number,
                prompt_id=prompt_id,
                protocol_digest=protocol_digest,
                frozen_lineup_seed=expected_lineup_seed,
                frozen_condition_order=expected_lineup_orders[index],
                seeds=seeds,
                judge_votes_path=judge_votes_path,
                issues=issues,
            )
            if lineup is None:
                continue
            artifact_paths = _verify_vote_artifacts(
                vote,
                vote_index=vote_number,
                judge_votes_path=judge_votes_path,
                public=lineup["public"],
                issues=issues,
            )
            public_clips = lineup["public"].get("clips")
            public_clips = public_clips if isinstance(public_clips, list) else []
            expected_labels = {
                clip.get("label")
                for clip in public_clips
                if isinstance(clip, dict) and isinstance(clip.get("label"), str)
            }
            artifact_verdicts = _artifact_clip_verdicts(
                provenance=vote.get("provenance"),
                result_path=Path(artifact_paths["result.json"])
                if "result.json" in artifact_paths
                else None,
                vote_index=vote_number,
                expected_labels=expected_labels,
                issues=issues,
            )
            per_seed_rows = vote.get("perSeedVerdicts")
            per_seed_rows = per_seed_rows if isinstance(per_seed_rows, list) else []
            by_seed: dict[int, dict[str, Any]] = {}
            for seed_index, seed_row in enumerate(per_seed_rows):
                if not isinstance(seed_row, dict):
                    issues.append(
                        f"vote {vote_number}: per-seed row {seed_index} must be an object"
                    )
                    continue
                seed = seed_row.get("generationSeed")
                if seed in by_seed:
                    issues.append(f"vote {vote_number}: duplicate generation seed {seed}")
                elif seed not in seeds:
                    issues.append(f"vote {vote_number}: unknown generation seed {seed}")
                else:
                    by_seed[seed] = seed_row
            if set(by_seed) != set(seeds):
                issues.append(f"vote {vote_number}: per-seed verdicts must cover frozen seeds")
            for seed in seeds:
                seed_row = by_seed.get(seed, {})
                labels = lineup["seedEntries"].get(seed, {})
                expected_refreshed = artifact_verdicts.get(labels.get("refreshed")) == "pass"
                expected_corrupted = artifact_verdicts.get(labels.get("corrupted")) == "pass"
                expected_candidate = artifact_verdicts.get(labels.get("unrefreshed"))
                if seed_row.get("refreshedControlPass") is not expected_refreshed:
                    issues.append(
                        f"vote {vote_number} seed {seed}: refreshed control contradicts result"
                    )
                if seed_row.get("corruptedControlPass") is not expected_corrupted:
                    issues.append(
                        f"vote {vote_number} seed {seed}: corrupted control contradicts result"
                    )
                if seed_row.get("candidateVerdict") != expected_candidate:
                    issues.append(
                        f"vote {vote_number} seed {seed}: candidate verdict contradicts result"
                    )
                if seed_row.get("refreshedControlPass") is not True:
                    issues.append(
                        f"vote {vote_number} seed {seed}: refreshed control did not pass"
                    )
                if seed_row.get("corruptedControlPass") is not False:
                    issues.append(
                        f"vote {vote_number} seed {seed}: corrupted control did not fail"
                    )
                candidate = seed_row.get("candidateVerdict")
                if candidate not in {"pass", "fail"}:
                    issues.append(
                        f"vote {vote_number} seed {seed}: candidate verdict must be pass or fail"
                    )
                else:
                    verdicts_by_seed[seed].append(candidate)
            if len(issues) == issue_count_before_vote:
                valid_vote_count += 1
        if sorted(value for value in observed_lineup_seeds if isinstance(value, int)) != sorted(
            expected_lineup_seeds
        ):
            issues.append("lineup seeds do not match the frozen protocol exactly once")

        seed_results = []
        evidence_invalid = bool(issues)
        for seed in seeds:
            seed_verdicts = verdicts_by_seed[seed]
            if evidence_invalid or len(seed_verdicts) != EXPECTED_VOTES_PER_PROMPT:
                seed_verdict = "inconclusive"
            elif len(set(seed_verdicts)) != 1:
                seed_verdict = "inconclusive"
                issues.append(f"seed {seed}: candidate verdicts are not unanimous")
            else:
                seed_verdict = seed_verdicts[0]
            seed_results.append(
                {"generationSeed": seed, "verdict": seed_verdict, "votes": seed_verdicts}
            )
        if issues or any(row["verdict"] == "inconclusive" for row in seed_results):
            verdict = "inconclusive"
        elif any(row["verdict"] == "fail" for row in seed_results):
            verdict = "fail"
        else:
            verdict = "pass"
        results.append(
            {
                "promptId": prompt_id,
                "verdict": verdict,
                "validVoteCount": valid_vote_count,
                "seedVerdicts": seed_results,
                "issues": issues,
            }
        )
    return results


def aggregate(
    *,
    protocol_path: Path,
    arm_paths: list[Path],
    pair_report_paths: list[Path],
    judge_votes_path: Path,
) -> dict[str, Any]:
    """Return a deterministic private G5 candidate manifest."""
    errors: list[str] = []
    protocol = _load_object_or_error(protocol_path, label="protocol", errors=errors)
    protocol_digest = _sha256(protocol_path)
    if protocol_digest != FROZEN_PROTOCOL_SHA256:
        errors.append(
            "protocol: frozen SHA-256 mismatch "
            f"({protocol_digest} != {FROZEN_PROTOCOL_SHA256})"
        )
    expected_arms, fixture_map, seeds = _protocol_contract(
        protocol, protocol_path, errors
    )
    arms, arm_digests = _collect_arms(
        arm_paths, expected_arms, fixture_map, protocol, errors
    )
    pairs, catastrophic = _collect_pairs(
        pair_report_paths,
        protocol_digest=protocol_digest,
        expected_arms=expected_arms,
        fixture_map=fixture_map,
        arm_records=arms,
        arm_digests=arm_digests,
        checkpoint_sha256=protocol.get("checkpoint", {}).get("sha256"),
        errors=errors,
    )
    judge = _load_object_or_error(
        judge_votes_path, label="judge votes", errors=errors
    )
    prompt_verdicts = _judge_prompt_verdicts(
        judge,
        judge_votes_path=judge_votes_path,
        protocol_digest=protocol_digest,
        protocol=protocol,
        fixture_map=fixture_map,
        seeds=seeds,
        errors=errors,
    )

    if errors:
        verdict = "inconclusive"
    elif catastrophic:
        verdict = "fail"
    elif any(row["verdict"] == "inconclusive" for row in prompt_verdicts):
        verdict = "inconclusive"
    elif any(row["verdict"] == "fail" for row in prompt_verdicts):
        verdict = "fail"
    else:
        verdict = "pass"
    return {
        "schema": OUTPUT_SCHEMA,
        "gate": "G5",
        "verdict": verdict,
        "passed": verdict == "pass",
        "protocol": {"path": str(protocol_path), "sha256": protocol_digest},
        "judgeVotes": {
            "path": str(judge_votes_path),
            "sha256": _sha256(judge_votes_path),
        },
        "counts": {
            "expectedArms": len(expected_arms),
            "observedArms": len(arms),
            "expectedPairs": len(
                {(prompt_id, seed) for prompt_id, seed, _ in expected_arms}
            ),
            "observedPairs": len(pairs),
            "expectedPrompts": len(fixture_map),
            "observedPromptVerdicts": len(prompt_verdicts),
        },
        "promptVerdicts": prompt_verdicts,
        "catastrophicFailures": catastrophic,
        "integrityErrors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--arm-manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--pair-report", type=Path, nargs="+", required=True)
    parser.add_argument("--judge-votes", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = aggregate(
        protocol_path=args.protocol,
        arm_paths=args.arm_manifest,
        pair_report_paths=args.pair_report,
        judge_votes_path=args.judge_votes,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.output_json)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
