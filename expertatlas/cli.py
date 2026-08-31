"""`atlas capture|analyze|export|serve` — CLI entrypoint (PLAN.md §3).

Phase 0 implements enough of `capture` to satisfy Checkpoint 0
("model loads and one forward pass emits router logits"). analyze/export/serve
are WS-C/WS-D/packaging territory — stubbed here so the CLI shape is frozen
even though the implementations land later.
"""

from __future__ import annotations

import json

import typer

app = typer.Typer(help="Expert Atlas — capture, analyze, export, serve.")

DEFAULT_MODEL_ID = "allenai/OLMoE-1B-7B-0924"


@app.command()
def capture(
    prompt: str = typer.Option(None, "--prompt", help="Single prompt to capture routing for (dry-run debug path)."),
    prompts_file: str = typer.Option(None, "--prompts-file", help="Path to a file, one prompt per line, for a real batched capture run."),
    out: str = typer.Option("data/traces", "--out", help="Output directory for parquet shards + manifest.json (real run only)."),
    model_id: str = typer.Option(DEFAULT_MODEL_ID, "--model"),
    device: str = typer.Option("cpu", "--device"),
    dtype: str = typer.Option("bfloat16", "--dtype"),
    seed: int = typer.Option(0, "--seed"),
    limit: int = typer.Option(None, "--limit", help="Cap the number of prompts captured this run."),
    layers: str = typer.Option(None, "--layers", help="Comma-separated layer indices to keep in the printed dry-run summary (real capture always writes all layers; §3 contract has no per-layer filtering)."),
    no_resume: bool = typer.Option(False, "--no-resume", help="Ignore any existing manifest.json and start clean (does not delete existing shards)."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Load, capture one prompt, print the trace shape. Do not write parquet."),
):
    """Capture router activations. --prompt --dry-run for a single-prompt
    debug summary; --prompts-file --out for a real batched, streaming,
    resumable capture run (PLAN.md §5 WS-A)."""
    from expertatlas.capture import build_meta, capture_to_dir, get_router_logits_for_prompt, load_model, route_from_logits

    if not prompt and not prompts_file:
        typer.echo("error: pass either --prompt (debug) or --prompts-file (real run)", err=True)
        raise typer.Exit(code=2)

    typer.echo(f"loading {model_id} (device={device}, dtype={dtype})...")
    loaded = load_model(model_id, device=device, dtype=dtype)
    typer.echo(
        f"loaded. capture_method={loaded.capture_method} "
        f"n_layers={loaded.shape.n_layers} n_experts={loaded.shape.n_experts} "
        f"top_k={loaded.shape.top_k} norm_topk_prob={loaded.shape.norm_topk_prob}"
    )

    if prompt:
        router_logits = get_router_logits_for_prompt(loaded, prompt, device)
        layer_filter = {int(x) for x in layers.split(",")} if layers else None

        n_tokens = router_logits[0].shape[0]
        trace_summary = []
        for layer, logits in enumerate(router_logits):
            if layer_filter is not None and layer not in layer_filter:
                continue
            ids, weights, mass = route_from_logits(
                logits, loaded.shape.top_k, loaded.shape.norm_topk_prob
            )
            trace_summary.append(
                {"layer": layer, "shape": list(ids.shape), "mean_topk_mass": float(mass.mean())}
            )

        meta = build_meta(loaded, device=device, dtype=dtype, seed=seed, repo_root=".")
        result = {
            "prompt": prompt,
            "n_tokens": n_tokens,
            "n_layers_captured": len(router_logits),
            "trace_summary": trace_summary,
            "meta": meta.model_dump(),
        }
        typer.echo(json.dumps(result, indent=2, default=str))
        return

    # Real batched/streaming/resumable run.
    lines = [l.strip() for l in open(prompts_file, encoding="utf-8") if l.strip()]
    prompts = list(enumerate(lines))
    typer.echo(f"capturing {len(prompts)} prompts (limit={limit}) -> {out}")
    result = capture_to_dir(
        loaded,
        prompts,
        out_dir=out,
        device=device,
        dtype=dtype,
        seed=seed,
        limit=limit,
        resume=not no_resume,
    )
    typer.echo(json.dumps(result, indent=2, default=str))


@app.command()
def analyze():
    """Compute lift/significance/communities from captured traces. WS-C scope."""
    typer.echo("not implemented yet — Phase 2, WS-C")
    raise typer.Exit(code=1)


@app.command()
def export():
    """Write atlas.json from analysis results. WS-C scope."""
    typer.echo("not implemented yet — Phase 2, WS-C")
    raise typer.Exit(code=1)


@app.command()
def serve():
    """Serve the visualiser locally. WS-D/packaging scope."""
    typer.echo("not implemented yet — Phase 4")
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
