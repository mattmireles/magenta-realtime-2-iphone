#!/usr/bin/env python3
"""Produce authenticated ``mrt2-liveness-judge-votes-v2`` evidence.

The normalizer derives every seed-level verdict from the sealed lineup mapping
and the exact primary or direct-Gemini result artifact. It never accepts a
manually entered candidate or control verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
FROZEN_PROTOCOL_SHA256 = (
    "bfd0c7240e07c0f61ed9ee03bde9f07ad6ff4795b7db2c05288c063f1b7253ae"
)
PROTOCOL_SCHEMA = "mrt2-liveness-protocol-v1"
PROTOCOL_STATUS = "frozen-before-candidate-generation"
PUBLIC_SCHEMA = "mrt2-liveness-opaque-prompt-lineup-v2"
SEALED_SCHEMA = "mrt2-liveness-sealed-vote-mapping-v2"
OUTPUT_SCHEMA = "mrt2-liveness-judge-votes-v2"
DIRECT_REQUEST_SCHEMA = "mrt2-direct-gemini-audio-judge-request-v1"
PRIMARY_FILES = ("input.json", "result.json", "report.md", "run.json")
DIRECT_FILES = ("request.json", "result.json")
CONDITIONS = {"refreshed", "unrefreshed", "context0", "corrupted"}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


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


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _resolve_path(raw: object, owner: Path) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"invalid path in {owner}: {raw!r}")
    path = Path(raw)
    if path.is_absolute():
        return path
    repo_path = REPO_ROOT / path
    return repo_path if repo_path.exists() else owner.parent / path


def _record(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise ValueError(f"missing artifact: {path}")
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _verified_record(record: object, *, owner: Path, label: str) -> Path:
    if not isinstance(record, dict):
        raise ValueError(f"{label} must be a path/SHA-256 record")
    path = _resolve_path(record.get("path"), owner)
    expected = record.get("sha256")
    if not isinstance(expected, str) or SHA256_RE.fullmatch(expected) is None:
        raise ValueError(f"{label} has an invalid SHA-256")
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: {actual} != {expected}")
    return path


def _public_contract(
    sealed: dict[str, Any], mapping_path: Path, protocol_sha256: str
) -> tuple[dict[str, Any], Path, list[dict[str, Any]]]:
    if sealed.get("schema") != SEALED_SCHEMA:
        raise ValueError(f"sealed mapping schema must be {SEALED_SCHEMA}")
    if sealed.get("protocolSha256") != protocol_sha256:
        raise ValueError("sealed mapping protocol hash mismatch")
    input_path = _verified_record(
        sealed.get("inputManifest"), owner=mapping_path, label="input manifest"
    )
    public = _load_object(input_path)
    if public.get("schema") != PUBLIC_SCHEMA:
        raise ValueError(f"public lineup schema must be {PUBLIC_SCHEMA}")
    for key in ("protocolSha256", "promptId", "voteIndex", "opaqueOrderSha256"):
        if public.get(key) != sealed.get(key):
            raise ValueError(f"public and sealed lineup disagree on {key}")
    clips = public.get("clips")
    if not isinstance(clips, list) or not clips:
        raise ValueError("public lineup clips must be a nonempty list")
    labels: set[str] = set()
    opaque_order = []
    for position, clip in enumerate(clips):
        if not isinstance(clip, dict) or not isinstance(clip.get("label"), str):
            raise ValueError(f"invalid public clip at position {position}")
        label = clip["label"]
        if label in labels:
            raise ValueError(f"duplicate public clip label: {label}")
        labels.add(label)
        _verified_record(clip, owner=input_path, label=f"public clip {label}")
        opaque_order.append(
            {
                "position": position,
                "label": label,
                "seedGroup": clip.get("seedGroup"),
                "sha256": clip.get("sha256"),
            }
        )
    if _json_sha256(opaque_order) != public.get("opaqueOrderSha256"):
        raise ValueError("public opaque-order hash mismatch")
    context_path = _verified_record(
        public.get("neutralContextFile"),
        owner=input_path,
        label="neutral-context file",
    )
    context = public.get("neutralContext")
    if not isinstance(context, str) or context_path.read_text().strip() != context:
        raise ValueError("neutral-context file bytes do not equal public context")
    mapping = sealed.get("mapping")
    if not isinstance(mapping, dict) or set(mapping) != labels:
        raise ValueError("sealed mapping does not cover public clips exactly")
    if sealed.get("mappingSha256") != _json_sha256(mapping):
        raise ValueError("sealed mapping hash mismatch")
    per_seed = sealed.get("perSeedEntries")
    if not isinstance(per_seed, list) or not per_seed:
        raise ValueError("sealed per-seed entries are missing")
    observed_seeds: set[int] = set()
    for row in per_seed:
        labels_by_condition = row.get("labelsByCondition") if isinstance(row, dict) else None
        seed = row.get("generationSeed") if isinstance(row, dict) else None
        if not isinstance(seed, int) or seed in observed_seeds:
            raise ValueError(f"invalid or duplicate generation seed: {seed!r}")
        observed_seeds.add(seed)
        if not isinstance(labels_by_condition, dict) or set(labels_by_condition) != CONDITIONS:
            raise ValueError(f"seed {seed} does not map all four conditions")
        for condition, label in labels_by_condition.items():
            entry = mapping.get(label)
            if (
                not isinstance(entry, dict)
                or entry.get("condition") != condition
                or entry.get("generationSeed") != seed
                or entry.get("clipSha256")
                != next(clip["sha256"] for clip in clips if clip["label"] == label)
            ):
                raise ValueError(f"seed {seed} mapping is inconsistent for {condition}")
    return public, input_path, per_seed


def _clip_verdicts(result: dict[str, Any], provenance: str) -> dict[str, str]:
    verdicts: dict[str, str] = {}
    if provenance == "primary":
        output = result.get("output")
        rows = output.get("clips") if isinstance(output, dict) else None
        if not isinstance(rows, list):
            raise ValueError("primary result output.clips must be a list")
        entries = [
            (row.get("label"), row.get("verdict"))
            for row in rows
            if isinstance(row, dict)
        ]
    else:
        rows = result.get("clips")
        if not isinstance(rows, dict):
            raise ValueError("direct result clips must be an object")
        entries = [
            (label, row.get("verdict"))
            for label, row in rows.items()
            if isinstance(row, dict)
        ]
    for label, verdict in entries:
        if not isinstance(label, str) or verdict not in {"pass", "fail"}:
            raise ValueError(f"invalid result clip verdict: {label!r}={verdict!r}")
        if label in verdicts:
            raise ValueError(f"duplicate result clip verdict: {label}")
        verdicts[label] = verdict
    return verdicts


def _clip_sources(value: object, *, primary: bool) -> list[tuple[object, object]]:
    rows = value if isinstance(value, list) else []
    return [
        (
            row.get("label"),
            row.get("source", {}).get("sha256")
            if primary and isinstance(row.get("source"), dict)
            else row.get("sha256"),
        )
        for row in rows
        if isinstance(row, dict)
    ]


def _primary_contract(
    artifact_dir: Path, public: dict[str, Any]
) -> tuple[dict[str, Any], str, dict[str, dict[str, str]]]:
    paths = {name: artifact_dir / name for name in PRIMARY_FILES}
    records = {
        "input.json": _record(paths["input.json"]),
        "result.json": _record(paths["result.json"]),
        "report.md": _record(paths["report.md"]),
        "workerMetadata": _record(paths["run.json"]),
    }
    workflow_input = _load_object(paths["input.json"])
    expected_sources = _clip_sources(public.get("clips"), primary=False)
    if _clip_sources(workflow_input.get("clips"), primary=True) != expected_sources:
        raise ValueError("primary input clips do not match public lineup")
    for input_key, public_key in (
        ("prompt", "prompt"),
        ("baselineLabel", "baselineLabel"),
        ("context", "neutralContext"),
    ):
        if workflow_input.get(input_key) != public.get(public_key):
            raise ValueError(f"primary input {input_key} does not match public lineup")
    result = _load_object(paths["result.json"])
    if result.get("workflowId") != "audio_judge_v1":
        raise ValueError("primary result workflowId must be audio_judge_v1")
    if result.get("status") != "completed":
        raise ValueError("primary result status must be completed")
    run_id = result.get("runId")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("primary result runId is missing")
    worker = _load_object(paths["run.json"])
    accepted = worker.get("acceptedRun")
    latest = worker.get("latestRun")
    accepted = accepted if isinstance(accepted, dict) else {}
    latest = latest if isinstance(latest, dict) else {}
    if accepted.get("runId") != run_id:
        raise ValueError("primary acceptedRun is not linked to result runId")
    if latest.get("runId") != run_id or latest.get("status") != "completed":
        raise ValueError("primary latestRun is not the completed result run")
    return result, run_id, records


def _direct_contract(
    artifact_dir: Path, public: dict[str, Any]
) -> tuple[dict[str, Any], str, dict[str, dict[str, str]]]:
    request_path = artifact_dir / "request.json"
    result_path = artifact_dir / "result.json"
    records = {
        "request.json": _record(request_path),
        "result.json": _record(result_path),
    }
    request = _load_object(request_path)
    if request.get("schema") != DIRECT_REQUEST_SCHEMA:
        raise ValueError(f"direct request schema must be {DIRECT_REQUEST_SCHEMA}")
    expected_sources = _clip_sources(public.get("clips"), primary=False)
    if _clip_sources(request.get("clips"), primary=False) != expected_sources:
        raise ValueError("direct request clips do not match public lineup")
    for request_key, public_key in (
        ("prompt", "prompt"),
        ("baselineLabel", "baselineLabel"),
        ("context", "neutralContext"),
    ):
        if request.get(request_key) != public.get(public_key):
            raise ValueError(f"direct request {request_key} does not match public lineup")
    rendered_hash = request.get("renderedPromptSha256")
    if not isinstance(rendered_hash, str) or SHA256_RE.fullmatch(rendered_hash) is None:
        raise ValueError("direct request renderedPromptSha256 is invalid")
    result = _load_object(result_path)
    if result.get("requestSha256") != _sha256(request_path):
        raise ValueError("direct result is not hash-linked to request sidecar")
    vote_id = (
        f"direct-gemini-{public['promptId']}-{public['voteIndex']}-"
        f"{_sha256(result_path)[:16]}"
    )
    return result, vote_id, records


def normalize_vote(
    *,
    mapping_path: Path,
    provenance: str,
    artifact_dir: Path,
    protocol_sha256: str,
) -> tuple[str, int, dict[str, Any]]:
    """Normalize one sealed lineup and judge artifact directory into one vote."""
    mapping_path = mapping_path.resolve()
    artifact_dir = artifact_dir.resolve()
    sealed = _load_object(mapping_path)
    public, input_path, per_seed = _public_contract(
        sealed, mapping_path, protocol_sha256
    )
    if provenance == "primary":
        result, vote_id, artifacts = _primary_contract(artifact_dir, public)
    elif provenance == "direct-gemini":
        result, vote_id, artifacts = _direct_contract(artifact_dir, public)
    else:
        raise ValueError(f"unsupported provenance: {provenance}")
    verdicts = _clip_verdicts(result, provenance)
    expected_labels = {clip["label"] for clip in public["clips"]}
    if set(verdicts) != expected_labels:
        raise ValueError("result verdicts do not cover public lineup exactly")
    per_seed_verdicts = []
    for row in per_seed:
        labels = row["labelsByCondition"]
        per_seed_verdicts.append(
            {
                "generationSeed": row["generationSeed"],
                "refreshedControlPass": verdicts[labels["refreshed"]] == "pass",
                "corruptedControlPass": verdicts[labels["corrupted"]] == "pass",
                "candidateVerdict": verdicts[labels["unrefreshed"]],
            }
        )
    vote = {
        "voteId": vote_id,
        "lineupSeed": sealed.get("lineupSeed"),
        "inputManifest": _record(input_path),
        "sealedMapping": _record(mapping_path),
        "perSeedVerdicts": per_seed_verdicts,
        "provenance": provenance,
        "artifacts": artifacts,
    }
    return public["promptId"], public["voteIndex"], vote


def _parse_spec(value: str) -> tuple[Path, Path]:
    mapping, separator, artifact_dir = value.partition("=")
    if not separator or not mapping or not artifact_dir:
        raise argparse.ArgumentTypeError(
            "vote must be SEALED_MAPPING_PATH=ARTIFACT_DIRECTORY"
        )
    return Path(mapping), Path(artifact_dir)


def normalize_campaign(
    *,
    protocol_path: Path,
    primary_specs: list[tuple[Path, Path]],
    direct_specs: list[tuple[Path, Path]],
) -> dict[str, Any]:
    """Normalize and require the complete frozen prompt/vote campaign."""
    protocol_path = protocol_path.resolve()
    digest = _sha256(protocol_path)
    if digest != FROZEN_PROTOCOL_SHA256:
        raise ValueError(
            f"frozen protocol SHA-256 mismatch: {digest} != {FROZEN_PROTOCOL_SHA256}"
        )
    protocol = _load_object(protocol_path)
    if protocol.get("schema") != PROTOCOL_SCHEMA or protocol.get("status") != PROTOCOL_STATUS:
        raise ValueError("protocol schema/status is not the frozen liveness contract")
    fixtures = protocol.get("fixtures")
    audio_judge = protocol.get("audioJudge")
    if not isinstance(fixtures, list) or not isinstance(audio_judge, dict):
        raise ValueError("protocol fixtures/audioJudge contract is missing")
    prompt_ids = [row.get("id") for row in fixtures if isinstance(row, dict)]
    lineup_seeds = audio_judge.get("lineupSeedsByPrompt")
    lineup_orders = audio_judge.get("lineupOrdersByPrompt")
    if not isinstance(lineup_seeds, dict) or not isinstance(lineup_orders, dict):
        raise ValueError("protocol frozen lineup campaign is missing")

    rows: dict[str, dict[int, dict[str, Any]]] = {prompt_id: {} for prompt_id in prompt_ids}
    all_specs = [
        ("primary", mapping, directory) for mapping, directory in primary_specs
    ]
    all_specs.extend(
        ("direct-gemini", mapping, directory) for mapping, directory in direct_specs
    )
    for provenance, mapping_path, artifact_dir in all_specs:
        prompt_id, vote_index, vote = normalize_vote(
            mapping_path=mapping_path,
            provenance=provenance,
            artifact_dir=artifact_dir,
            protocol_sha256=digest,
        )
        if prompt_id not in rows:
            raise ValueError(f"vote has unknown promptId: {prompt_id}")
        if vote_index in rows[prompt_id]:
            raise ValueError(f"duplicate vote {prompt_id}/{vote_index}")
        expected_seeds = lineup_seeds.get(prompt_id)
        expected_orders = lineup_orders.get(prompt_id)
        sealed = _load_object(mapping_path)
        if (
            not isinstance(expected_seeds, list)
            or not isinstance(expected_orders, list)
            or vote_index < 1
            or vote_index > len(expected_seeds)
            or vote["lineupSeed"] != expected_seeds[vote_index - 1]
            or sealed.get("frozenConditionOrder") != expected_orders[vote_index - 1]
        ):
            raise ValueError(f"vote {prompt_id}/{vote_index} contradicts frozen lineup")
        rows[prompt_id][vote_index] = vote
    prompts = []
    for prompt_id in prompt_ids:
        expected_count = len(lineup_seeds.get(prompt_id, []))
        if set(rows[prompt_id]) != set(range(1, expected_count + 1)):
            raise ValueError(f"prompt {prompt_id} does not contain every frozen vote")
        prompts.append(
            {
                "promptId": prompt_id,
                "votes": [rows[prompt_id][index] for index in range(1, expected_count + 1)],
            }
        )
    return {"schema": OUTPUT_SCHEMA, "protocolSha256": digest, "prompts": prompts}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument(
        "--primary-vote",
        action="append",
        default=[],
        type=_parse_spec,
        metavar="SEALED_MAPPING=ARTIFACT_DIR",
    )
    parser.add_argument(
        "--direct-vote",
        action="append",
        default=[],
        type=_parse_spec,
        metavar="SEALED_MAPPING=ARTIFACT_DIR",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = normalize_campaign(
        protocol_path=args.protocol,
        primary_specs=args.primary_vote,
        direct_specs=args.direct_vote,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
