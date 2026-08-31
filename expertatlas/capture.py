"""Router capture: turn a prompt into RoutingTrace rows (PLAN.md §3, WS-A).

Primary path: load with output_router_logits=True and read raw router
logits off the model output. We then independently recompute the top-k
selection and gate weights from those logits ourselves, rather than trust
an internal path we can't directly observe — same numbers, but auditable.

Fallback path: if a model/architecture doesn't support
output_router_logits, register forward hooks on every module whose name
matches a configurable pattern (default: modules ending in ".gate") and
capture their raw output tensor instead. Recorded in meta.json as
capture_method so it's never silently ambiguous which path produced a trace.
"""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

from expertatlas.schemas import ROUTING_TRACE_SCHEMA, ModelShape, RoutingTraceMeta

DEFAULT_GATE_MODULE_SUFFIX = ".gate"
MANIFEST_NAME = "manifest.json"


@dataclass
class LoadedModel:
    model: object
    tokenizer: object
    shape: ModelShape
    capture_method: str  # "output_router_logits" | "forward_hook"
    model_id: str
    model_revision: str


def _git_commit(cwd: str | None = None) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _host_fingerprint() -> str:
    return f"{platform.node()}|{platform.machine()}|{platform.processor()}"


def load_model(
    model_id: str,
    revision: str = "main",
    device: str = "cpu",
    dtype: str = "bfloat16",
) -> LoadedModel:
    """Load a MoE causal LM and confirm we can extract router logits.

    Raises RuntimeError with a clear message if neither path works — per
    PLAN.md Checkpoint 0, that means "stop and re-select the model", not
    "silently degrade the science".
    """
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    torch_dtype = getattr(torch, dtype)
    config = AutoConfig.from_pretrained(model_id, revision=revision)

    n_layers = getattr(config, "num_hidden_layers", None)
    n_experts = getattr(config, "num_experts", None)
    top_k = getattr(config, "num_experts_per_tok", None)
    norm_topk_prob = bool(getattr(config, "norm_topk_prob", False))

    if n_layers is None or n_experts is None or top_k is None:
        raise RuntimeError(
            f"{model_id}: config is missing num_hidden_layers/num_experts/"
            "num_experts_per_tok — this doesn't look like a supported MoE "
            "config. Re-select the model (PLAN.md Checkpoint 0)."
        )

    shape = ModelShape(
        n_layers=n_layers, n_experts=n_experts, top_k=top_k, norm_topk_prob=norm_topk_prob
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

    capture_method = "output_router_logits"
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=torch_dtype,
            output_router_logits=True,
        )
    except TypeError:
        # constructor doesn't accept the kwarg at all -> definitely no
        # native support, go straight to hook fallback
        model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, torch_dtype=torch_dtype
        )
        capture_method = "forward_hook"

    model = model.to(device)
    model.eval()

    if capture_method == "output_router_logits":
        # Prove it actually works with a tiny real forward pass rather than
        # trusting that accepting the kwarg means it's honored.
        probe = tokenizer("def foo():", return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**probe, output_router_logits=True)
        router_logits = getattr(out, "router_logits", None)
        if not router_logits or len(router_logits) == 0:
            capture_method = "forward_hook"

    if capture_method == "forward_hook":
        has_gate = any(name.endswith(DEFAULT_GATE_MODULE_SUFFIX) for name, _ in model.named_modules())
        if not has_gate:
            raise RuntimeError(
                f"{model_id}: output_router_logits unsupported AND no module "
                f"name ends in '{DEFAULT_GATE_MODULE_SUFFIX}' for the hook "
                "fallback. Re-select the model (PLAN.md Checkpoint 0)."
            )

    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        shape=shape,
        capture_method=capture_method,
        model_id=model_id,
        model_revision=revision,
    )


def build_meta(loaded: LoadedModel, device: str, dtype: str, seed: int, repo_root: str) -> RoutingTraceMeta:
    import torch as _torch
    import transformers as _transformers

    return RoutingTraceMeta(
        model_id=loaded.model_id,
        model_revision=loaded.model_revision,
        dtype=dtype,
        device=device,
        top_k=loaded.shape.top_k,
        n_layers=loaded.shape.n_layers,
        n_experts=loaded.shape.n_experts,
        transformers_version=_transformers.__version__,
        torch_version=_torch.__version__,
        seed=seed,
        git_commit=_git_commit(repo_root),
        timestamp=datetime.now(timezone.utc).isoformat(),
        host_fingerprint=_host_fingerprint(),
        capture_method=loaded.capture_method,
    )


def route_from_logits(router_logits: torch.Tensor, top_k: int, norm_topk_prob: bool):
    """Given raw router logits for one layer, shape (n_tokens, n_experts),
    return (expert_ids, gate_weights, topk_mass) each shape (n_tokens, top_k)
    / (n_tokens,) exactly reproducing what the model's own MoE block would
    select — softmax over ALL experts first, THEN top-k, matching HF's
    OlmoeSparseMoeBlock (and the Mixtral-family convention).

    norm_topk_prob controls whether the selected top-k weights are
    renormalised to sum to 1. OLMoE sets this False; do not assume True.
    """
    routing_weights = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(routing_weights, top_k, dim=-1)
    topk_mass = topk_weights.sum(dim=-1)
    if norm_topk_prob:
        topk_weights = topk_weights / topk_mass.unsqueeze(-1)
    return topk_ids, topk_weights, topk_mass


# ---------------------------------------------------------------------------
# WS-A: batched, streaming, resumable capture (PLAN.md §5 Phase 1)
# ---------------------------------------------------------------------------


class _GateHookCapture:
    """Registers forward hooks on every module named '*<suffix>' and records
    each call's raw output tensor, in registration (= layer) order.

    Used only when output_router_logits isn't available (load_model() falls
    back to this and sets capture_method="forward_hook"). Untested against a
    real such model in this repo -- OLMoE uses output_router_logits natively
    -- but implemented for real per PLAN.md §5's robustness requirement, not
    left as a stub.
    """

    def __init__(self, model, suffix: str = DEFAULT_GATE_MODULE_SUFFIX):
        self.captured: list[torch.Tensor] = []
        self._handles = []
        for name, module in model.named_modules():
            if name.endswith(suffix):
                self._handles.append(module.register_forward_hook(self._hook))

    def _hook(self, module, inputs, output):
        # Gate modules conventionally return raw logits, or a tuple whose
        # first element is. Either way we want a (n_tokens, n_experts) tensor.
        tensor = output[0] if isinstance(output, tuple) else output
        self.captured.append(tensor.detach())

    def reset(self):
        self.captured = []

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


def get_router_logits_for_prompt(loaded: LoadedModel, prompt: str, device: str) -> list[torch.Tensor]:
    """Run one forward pass and return a list of (n_tokens, n_experts) raw
    router-logit tensors, one per MoE layer, regardless of which capture
    path is active. This is the single place both CLI dry-run and the real
    batched writer go through, so they can never silently diverge."""
    inputs = loaded.tokenizer(prompt, return_tensors="pt").to(device)

    if loaded.capture_method == "output_router_logits":
        with torch.no_grad():
            out = loaded.model(**inputs, output_router_logits=True)
        router_logits = list(out.router_logits)
    else:
        hook = _GateHookCapture(loaded.model)
        try:
            with torch.no_grad():
                loaded.model(**inputs)
            router_logits = list(hook.captured)
        finally:
            hook.remove()

    if len(router_logits) != loaded.shape.n_layers:
        raise RuntimeError(
            f"expected {loaded.shape.n_layers} layers of router logits, got "
            f"{len(router_logits)} via capture_method={loaded.capture_method} "
            "-- a layer was silently skipped (PLAN.md §6.1 "
            "test_no_silent_layer_skips guards exactly this)"
        )
    return router_logits


def _read_manifest(out_dir: Path) -> dict:
    path = out_dir / MANIFEST_NAME
    if not path.exists():
        return {"done_prompt_ids": []}
    return json.loads(path.read_text())


def _write_manifest_atomic(out_dir: Path, manifest: dict) -> None:
    # Write-then-rename so a kill mid-write never leaves a corrupt manifest
    # for the next resume attempt to trust.
    tmp = out_dir / f".{MANIFEST_NAME}.tmp"
    tmp.write_text(json.dumps(manifest, indent=2))
    tmp.replace(out_dir / MANIFEST_NAME)


def _trace_shard_path(out_dir: Path, prompt_id: int) -> Path:
    return out_dir / f"trace_{prompt_id:06d}.parquet"


def prompt_rows_to_table(
    prompt_id: int,
    input_ids: torch.Tensor,  # (n_tokens,)
    router_logits: list[torch.Tensor],  # n_layers x (n_tokens, n_experts)
    top_k: int,
    norm_topk_prob: bool,
):
    """Build a pyarrow Table matching ROUTING_TRACE_SCHEMA for one prompt's
    full trace (every layer, every token) -- kept separate from the writer
    so it's independently testable without a model or disk I/O."""
    import pyarrow as pa

    n_tokens = input_ids.shape[0]
    prompt_ids_col, token_pos_col, token_id_col = [], [], []
    layer_col, expert_ids_col, gate_weights_col, topk_mass_col = [], [], [], []

    for layer, logits in enumerate(router_logits):
        ids, weights, mass = route_from_logits(logits, top_k, norm_topk_prob)
        for t in range(n_tokens):
            prompt_ids_col.append(prompt_id)
            token_pos_col.append(t)
            token_id_col.append(int(input_ids[t]))
            layer_col.append(layer)
            expert_ids_col.append(ids[t].tolist())
            gate_weights_col.append([float(w) for w in weights[t].tolist()])
            topk_mass_col.append(float(mass[t]))

    return pa.table(
        {
            "prompt_id": pa.array(prompt_ids_col, type=pa.int32()),
            "token_pos": pa.array(token_pos_col, type=pa.int32()),
            "token_id": pa.array(token_id_col, type=pa.int32()),
            "layer": pa.array(layer_col, type=pa.int16()),
            "expert_ids": pa.array(expert_ids_col, type=pa.list_(pa.int16())),
            "gate_weights": pa.array(gate_weights_col, type=pa.list_(pa.float32())),
            "topk_mass": pa.array(topk_mass_col, type=pa.float32()),
        },
        schema=ROUTING_TRACE_SCHEMA,
    )


def capture_to_dir(
    loaded: LoadedModel,
    prompts: list[tuple[int, str]],  # (prompt_id, text), deterministic order
    out_dir: str | Path,
    device: str = "cpu",
    dtype: str = "bfloat16",
    seed: int = 0,
    limit: int | None = None,
    resume: bool = True,
) -> dict:
    """Stream RoutingTrace to one parquet shard per prompt under out_dir,
    resumable via a manifest of completed prompt_ids.

    Streaming: at most one prompt's trace rows are ever held in memory at
    once (PLAN §6.4 test_capture_memory_bounded) -- nothing is accumulated
    across prompts before writing.

    Resumable: prompts already listed in manifest.json are skipped entirely
    on re-run, so a kill-and-restart produces the same set of shard files as
    an uninterrupted run (PLAN §6.4 test_capture_resumable). resume=False
    ignores any existing manifest and starts clean (still won't clobber
    already-written shards for prompt_ids not in `prompts`).

    Greedy-only: this only ever runs a single forward pass per prompt to
    capture routing on the prompt's own tokens (no autoregressive sampling
    happens here at all), so there is no decoding strategy to fix -- PLAN's
    "greedy decode only" requirement is satisfied by construction, not by a
    sampling flag.
    """
    import pyarrow.parquet as pq

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)

    manifest = _read_manifest(out_dir) if resume else {"done_prompt_ids": []}
    done = set(manifest["done_prompt_ids"])

    ordered = sorted(prompts, key=lambda p: p[0])  # deterministic ordering
    if limit is not None:
        ordered = ordered[:limit]

    n_written, n_skipped = 0, 0
    for prompt_id, text in ordered:
        if prompt_id in done:
            n_skipped += 1
            continue

        router_logits = get_router_logits_for_prompt(loaded, text, device)
        inputs = loaded.tokenizer(text, return_tensors="pt")
        table = prompt_rows_to_table(
            prompt_id,
            inputs["input_ids"][0],
            router_logits,
            loaded.shape.top_k,
            loaded.shape.norm_topk_prob,
        )
        pq.write_table(table, _trace_shard_path(out_dir, prompt_id))

        done.add(prompt_id)
        _write_manifest_atomic(out_dir, {"done_prompt_ids": sorted(done)})
        n_written += 1

    meta = build_meta(loaded, device=device, dtype=dtype, seed=seed, repo_root=str(out_dir.parent))
    (out_dir / "meta.json").write_text(meta.model_dump_json(indent=2))

    return {
        "out_dir": str(out_dir),
        "n_prompts_written_this_run": n_written,
        "n_prompts_skipped_already_done": n_skipped,
        "n_prompts_total_done": len(done),
    }
