"""``monorl`` command line (``python -m monorl.cli``).

Torch-free at import time: every backend (orchestrator, patch engine, data sources, env
server) is imported inside the command that needs it, so ``monorl --help`` and ``monorl serve``
stay instant.
"""

# ruff: noqa: B008  (typer.Option/Argument in defaults is the typer idiom)
from __future__ import annotations

import json
import sys
import threading
import webbrowser
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from monorl.config.loader import apply_overrides, load_yaml
from monorl.config.pipeline import PipelineConfig, is_pipeline_payload
from monorl.config.schema import RunConfig
from monorl.core.errors import ConfigError, MonorlError
from monorl.core.reporting import LOG_FILE
from monorl.core.types import MEDIA_PART_TYPES, message_media, message_text

app = typer.Typer(
    name="monorl",
    help="Unified RL harness: train, serve the web UI, inspect data, check environments.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)
patches_app = typer.Typer(
    help="Inspect and apply versioned monkeypatches for upstream frameworks.", no_args_is_help=True
)
data_app = typer.Typer(help="Inspect datasets (local files, hf://, s3://).", no_args_is_help=True)
env_app = typer.Typer(help="Check or serve MEP/1 environments.", no_args_is_help=True)
recipes_app = typer.Typer(help="Browse example run configurations.", no_args_is_help=True)
app.add_typer(patches_app, name="patches")
app.add_typer(data_app, name="data")
app.add_typer(env_app, name="env")
app.add_typer(recipes_app, name="recipes")

console = Console()
err_console = Console(stderr=True)

EXIT_BY_STATE: dict[str, int] = {"succeeded": 0, "failed": 1, "cancelled": 130}


def _fail(message: str, code: int = 1) -> None:
    err_console.print(f"[bold red]error:[/bold red] {escape(message)}")
    raise typer.Exit(code)


def _recipe_hint(config: Path) -> str:
    """For a ``--config`` that is no file but names a bundled recipe: where that recipe is."""
    from monorl.config.recipes import get_recipe

    if config.exists():
        return ""
    try:
        return f" (the recipe of that name is passed by its path: --config {get_recipe(str(config)).path})"
    except ConfigError:
        return ""


def _print_run_error(error: str, run_dir: str) -> None:
    """The error of a run that did not succeed, once. A failed run reports its message followed by
    the traceback, which the run log has printed already and keeps in ``run.log``."""
    message, cut, _ = error.partition("\nTraceback (most recent call last)")
    if cut:
        message += f" (traceback in {Path(run_dir) / LOG_FILE})"
    err_console.print(f"[red]{escape(message)}[/red]")


def _yes_no(flag: bool | None) -> str:
    if flag is None:
        return "[dim]-[/dim]"
    return "[green]yes[/green]" if flag else "[red]no[/red]"


def _message_line(message: dict[str, Any], limit: int) -> str:
    """One console line per chat message: media parts as ``<kind: ref>`` markers (inline ``data:``
    URLs cut to their header), then its text."""
    media = [
        f"<{kind}: {ref.partition(',')[0] + ',…' if ref.startswith('data:') else ref}>"
        for kind in MEDIA_PART_TYPES
        for ref in message_media(message, kind)  # type: ignore[arg-type]
    ]
    text = message_text(message)[:limit]  # type: ignore[arg-type]
    return f"  [{message.get('role')}] {' '.join([*media, text])}"


# -- train -----------------------------------------------------------------------------------


@app.command()
def train(
    config: Path = typer.Option(..., "--config", "-c", help="RunConfig YAML (see `monorl recipes list`)."),
    set_: list[str] | None = typer.Option(
        None, "--set", "-s", help="Override: key.path=value (YAML-coerced)."
    ),
    run_dir: Path | None = typer.Option(
        None, "--run-dir", help="Run directory (default <run_root>/<run_name>)."
    ),
    resume: bool = typer.Option(False, "--resume", help="Resume from the latest checkpoint in the run dir."),
) -> None:
    """Run a training job from a RunConfig YAML (a pipeline YAML with `base:` + `stages:` is
    dispatched to the pipeline runner)."""
    try:
        data = apply_overrides(load_yaml(config), set_)
    except ConfigError as e:
        _fail(str(e) + _recipe_hint(config))
        return
    if is_pipeline_payload(data):
        _run_pipeline(config, data, run_dir, resume)
        return
    try:
        cfg = RunConfig.from_dict(data)
    except ConfigError as e:
        _fail(f"{config}: {e}")
        return
    from monorl.orchestrator import run

    try:
        result = run(cfg, run_dir, resume=resume)
    except MonorlError as e:
        _fail(str(e))
        return
    except OSError as e:  # what a run does not record as its failure: its directory cannot be written
        _fail(f"cannot write the run directory: {e}")
        return
    console.print(f"run dir: {escape(result.run_dir)}")
    console.print(f"state: {result.state}  final step: {result.final_step}")
    if result.error:
        _print_run_error(result.error, result.run_dir)
    if result.metrics:
        table = Table(title="final metrics", show_lines=False)
        table.add_column("metric")
        table.add_column("value", justify="right")
        for key in sorted(result.metrics):
            table.add_row(key, f"{result.metrics[key]:.6g}")
        console.print(table)
    if result.final_model_path:
        console.print(f"final model: {escape(result.final_model_path)}")
    raise typer.Exit(EXIT_BY_STATE.get(result.state, 1))


# -- pipeline --------------------------------------------------------------------------------


@app.command()
def pipeline(
    config: Path = typer.Option(..., "--config", "-c", help="PipelineConfig YAML (`base:` + `stages:`)."),
    set_: list[str] | None = typer.Option(
        None, "--set", "-s", help="Override: key.path=value (YAML-coerced), e.g. base.trainer.device=cpu."
    ),
    run_dir: Path | None = typer.Option(
        None, "--run-dir", help="Pipeline directory (default <base.logging.run_root>/<name>)."
    ),
    resume: bool = typer.Option(False, "--resume", help="Skip stages that already succeeded in the run dir."),
) -> None:
    """Run a multi-stage pipeline (SFT → RL → rejection sampling → SFT → RL → eval)."""
    try:
        data = apply_overrides(load_yaml(config), set_)
    except ConfigError as e:
        _fail(str(e) + _recipe_hint(config))
        return
    if not is_pipeline_payload(data):
        _fail(f"{config} is not a pipeline config (needs top-level `base:` and `stages:`)")
        return
    _run_pipeline(config, data, run_dir, resume)


def _run_pipeline(config: Path, data: dict[str, Any], run_dir: Path | None, resume: bool) -> None:
    try:
        cfg = PipelineConfig.from_dict(data)
    except ConfigError as e:
        _fail(f"{config}: {e}")
        return
    from monorl.pipeline import run_pipeline

    try:
        result = run_pipeline(cfg, run_dir, resume=resume)
    except MonorlError as e:
        _fail(str(e))
        return
    except OSError as e:  # what a pipeline does not record as its failure: its directory cannot be written
        _fail(f"cannot write the pipeline directory: {e}")
        return
    console.print(f"pipeline dir: {escape(result.run_dir)}")
    table = Table(title="stages", show_lines=False)
    table.add_column("#", justify="right")
    table.add_column("name")
    table.add_column("type")
    table.add_column("state")
    table.add_column("step", justify="right")
    for stage in result.stages:
        table.add_row(
            str(stage["index"] + 1),
            escape(stage["name"]),
            stage["type"],
            stage["state"],
            str(stage.get("step", 0)),
        )
    console.print(table)
    console.print(f"state: {result.state}")
    if result.error:
        _print_run_error(result.error, result.run_dir)
    if result.final_model_path:
        console.print(f"final model: {escape(result.final_model_path)}")
    raise typer.Exit(EXIT_BY_STATE.get(result.state, 1))


# -- serve -----------------------------------------------------------------------------------


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Open the UI in a browser."),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on source changes (development)."),
    token: str | None = typer.Option(
        None,
        "--token",
        help="Access token of the API (default: MONORL_SERVER_TOKEN, else a new random one). "
        "Prefer the variable: an option shows in the process list.",
    ),
    allowed_host: list[str] = typer.Option(
        [],
        "--allowed-host",
        help="Accept this name in the Host header besides localhost, 127.0.0.1 and [::1] "
        "(repeatable; MONORL_ALLOWED_HOSTS takes a comma-separated list).",
    ),
) -> None:
    """Start the local web UI + API server. Every API request needs the access token: open the
    sign-in URL this prints, or read the token from $MONORL_HOME/server_token."""
    import uvicorn

    from monorl.server import auth

    try:
        token = auth.configure(host, token, allowed_host)
    except ConfigError as e:
        _fail(str(e))
        return
    for line in auth.startup_lines(host, port, token):
        console.print(line, markup=False, highlight=False, soft_wrap=True)
    if open_browser:
        threading.Timer(1.0, webbrowser.open, args=(auth.signin_url(host, port, token),)).start()
    uvicorn.run("monorl.server.app:create_app", factory=True, host=host, port=port, reload=reload)


# -- doctor ----------------------------------------------------------------------------------


@app.command()
def doctor(as_json: bool = typer.Option(False, "--json", help="Emit the report as JSON.")) -> None:
    """Report Python/torch/toolchains, installed frameworks, patches and registered components."""
    from monorl.doctor import detect

    report = detect()
    if as_json:
        typer.echo(json.dumps(report.to_dict(), indent=2, default=str))
        return

    sysinfo = Table(title="system", show_header=False)
    sysinfo.add_column("key")
    sysinfo.add_column("value")
    sysinfo.add_row("python", f"{report.python['version']} ({report.python['executable']})")
    sysinfo.add_row(
        "platform", f"{report.platform['system']} {report.platform['release']} {report.platform['machine']}"
    )
    t = report.torch
    torch_line = t["version"] or "[red]not installed[/red]"
    if t.get("cuda_available"):
        torch_line += (
            f"  cuda {t['cuda_version']} × {t['cuda_device_count']} ({', '.join(t['cuda_devices'])})"
        )
    elif t.get("mps_available"):
        torch_line += "  mps"
    else:
        torch_line += "  cpu only"
    sysinfo.add_row("torch", torch_line)
    for name, tool in report.toolchains.items():
        sysinfo.add_row(name, tool["version"] or "[dim]not found[/dim]")
    console.print(sysinfo)

    fw = Table(title="frameworks")
    fw.add_column("framework")
    fw.add_column("installed")
    fw.add_column("version")
    fw.add_column("patches (applicable/applied/total)")
    for name, info in report.frameworks.items():
        patches = info["patches"]
        applicable = sum(1 for p in patches if p.get("applicable"))
        applied = sum(1 for p in patches if p.get("applied"))
        fw.add_row(
            name,
            _yes_no(info["installed"]),
            info["version"] or "[dim]-[/dim]",
            f"{applicable}/{applied}/{len(patches)}" if patches else "[dim]-[/dim]",
        )
    console.print(fw)

    comps = Table(title="components")
    comps.add_column("namespace")
    comps.add_column("names")
    for ns, names in report.components.items():
        comps.add_row(ns, ", ".join(names) or "[dim](none)[/dim]")
    console.print(comps)

    if report.errors:
        err_console.print("[yellow]warnings:[/yellow]")
        for line in report.errors:
            err_console.print(f"  - {escape(line)}")


# -- patches ---------------------------------------------------------------------------------


def _patching_module() -> Any:
    try:
        import monorl.patching as patching
    except ImportError as e:
        _fail(f"patch engine unavailable: {e}")
    return patching


def _load_adapters() -> None:
    """Import adapter packages so their ``PatchSet``s are registered (missing ones are fine)."""
    import importlib

    from monorl.adapters import PROVIDERS

    for module_path in PROVIDERS.values():
        try:
            importlib.import_module(module_path)
        except Exception:  # noqa: BLE001 - adapter not built yet or framework-specific import error
            continue


def _status_table(framework: str, statuses: list[Any]) -> Table:
    table = Table(title=f"patches: {framework}")
    for col in ("id", "target", "versions", "installed", "applicable", "applied", "verified", "reason"):
        table.add_column(col)
    for s in statuses:
        table.add_row(
            s.id,
            s.target,
            s.version_spec or "*",
            s.installed_version or "[dim]-[/dim]",
            _yes_no(s.applicable),
            _yes_no(s.applied),
            _yes_no(s.verified),
            escape(s.reason or ""),
        )
    return table


@patches_app.command("list")
def patches_list(as_json: bool = typer.Option(False, "--json")) -> None:
    """List every registered patch per framework."""
    patching = _patching_module()
    _load_adapters()
    frameworks = patching.registered_frameworks()
    if as_json:
        typer.echo(
            json.dumps({fw: [s.to_dict() for s in patching.status(fw)] for fw in frameworks}, indent=2)
        )
        return
    if not frameworks:
        console.print("no patchsets registered")
        return
    for fw in frameworks:
        console.print(_status_table(fw, patching.status(fw)))


@patches_app.command("status")
def patches_status(
    framework: str | None = typer.Argument(None, help="Framework name; all when omitted."),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Show applicability / applied / verified state of patches against installed versions."""
    patching = _patching_module()
    _load_adapters()
    registered = patching.registered_frameworks()
    if framework is not None and framework not in registered:
        _fail(
            f"no patches are registered for {framework!r}; "
            f"frameworks with patches: {', '.join(registered) or '(none)'}"
        )
    frameworks = [framework] if framework else registered
    if as_json:
        typer.echo(
            json.dumps({fw: [s.to_dict() for s in patching.status(fw)] for fw in frameworks}, indent=2)
        )
        return
    for fw in frameworks:
        statuses = patching.status(fw)
        if not statuses:
            console.print(f"{fw}: no patches registered")
            continue
        console.print(_status_table(fw, statuses))


@patches_app.command("apply")
def patches_apply(
    framework: str = typer.Argument(..., help="Framework whose patchset to apply."),
    force: bool = typer.Option(False, "--force", help="Re-apply even if already applied."),
) -> None:
    """Apply the patchset of FRAMEWORK in this process and report which patches took. Nothing is
    written to the installed package: a run applies the patches itself when it builds a component
    of the framework, so this is a check that they apply to the installed version."""
    patching = _patching_module()
    _load_adapters()
    try:
        report = patching.apply(framework, force=force)
    except MonorlError as e:
        _fail(str(e))
        return
    console.print(f"{framework} {report.installed_version or '(not installed)'}")
    for pid in report.applied:
        console.print(f"  [green]applied[/green]  {pid}")
    for pid, reason in report.skipped:
        console.print(f"  [yellow]skipped[/yellow]  {pid}: {escape(reason)}")
    for pid, error in report.failed:
        console.print(f"  [red]failed[/red]   {pid}: {escape(error)}")
    if report.failed:
        raise typer.Exit(1)


# -- data ------------------------------------------------------------------------------------


@data_app.command("inspect")
def data_inspect(
    uri: str = typer.Argument(
        ...,
        help="Local path, hf://org/name, s3://bucket/key, http(s)://host/file, or the scheme of a --provider's data source.",
    ),
    limit: int = typer.Option(5, "--limit", "-n", min=1, help="Rows / samples to show."),
    split: str | None = typer.Option(None, "--split", help="Split for Hub datasets."),
    hf_config_name: str | None = typer.Option(None, "--config-name", help="Hub dataset config/subset."),
    providers: list[str] = typer.Option(
        [], "--provider", help="Module to import first, as in a config's `providers` (repeatable)."
    ),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Preview a dataset: columns, inferred format, first rows and the Samples monorl would build."""
    from itertools import islice

    from monorl.config.schema import DataConfig
    from monorl.core.registry import registry
    from monorl.data import open_source
    from monorl.data.browse import preview_uri

    try:
        registry.load_providers(providers)
        preview = preview_uri(uri, split=split, hf_config_name=hf_config_name, limit=limit)
        data_config = DataConfig(uri=uri, hf_config_name=hf_config_name, shuffle=False, streaming=True)
        if split:
            data_config.split = split
        source = open_source(data_config, split=split)
        samples = [
            {"id": s.id, "messages": s.messages, "reference": s.reference, "metadata": s.metadata}
            for s in islice(iter(source), limit)
        ]
    except FileNotFoundError as e:
        # A string without a scheme is a path, also one that reads like a Hub repository.
        looks_like_repo = "://" not in uri and uri.count("/") == 1 and not Path(uri).suffix
        _fail(f"{e}" + (f" (a Hub dataset is written hf://{uri})" if looks_like_repo else ""))
        return
    except MonorlError as e:
        _fail(str(e))
        return
    if as_json:
        typer.echo(json.dumps({**preview, "samples": samples}, indent=2, default=str))
        return
    console.print(f"[bold]{escape(uri)}[/bold]")
    inferred = preview.get("inferred") or {}
    console.print(
        escape(
            "format: {format}  prompt: {prompt_field}  answer: {answer_field}  "
            "messages: {messages_field}".format(
                **{k: inferred.get(k) for k in ("format", "prompt_field", "answer_field", "messages_field")}
            )
        )
    )
    if preview.get("num_rows") is not None:
        console.print(f"rows: {preview['num_rows']}")
    columns = preview.get("columns") or {}
    table = Table(title="rows")
    for name in columns:
        table.add_column(escape(f"{name} ({columns[name]})"), overflow="fold", max_width=60)
    for row in preview.get("rows") or []:
        table.add_row(*(escape(str(row.get(name, ""))[:200]) for name in columns))
    console.print(table)
    for s in samples:
        console.print(f"[bold]sample {escape(str(s['id']))}[/bold] reference={escape(repr(s['reference']))}")
        for m in s["messages"]:
            console.print(escape(_message_line(m, 300)))


# -- env -------------------------------------------------------------------------------------


@env_app.command("check")
def env_check(
    command: list[str] | None = typer.Option(
        None, "--command", "-c", help="Server argv for stdio (repeatable)."
    ),
    url: str | None = typer.Option(None, "--url", help="Base URL of an http MEP/1 server."),
    turns: int = typer.Option(1, "--turns", min=1, help="Steps to run after reset."),
    cwd: Path | None = typer.Option(None, "--cwd", help="Working directory for the stdio server."),
    timeout: float = typer.Option(30.0, "--timeout", help="Seconds to wait for each response."),
    sample: Path | None = typer.Option(
        None,
        "--sample",
        help="JSON file with the sample to reset with: an object with any of id, messages, reference, "
        "metadata. Keys left out keep the built-in sample's value.",
    ),
    reference: str | None = typer.Option(
        None,
        "--reference",
        help="Reference of the sample, read like a --set value (7 is an integer, '\"7\"' a string). "
        "Built-in sample: the string hello.",
    ),
    action: str | None = typer.Option(
        None, "--action", help="Completion text sent with every step. Default: hello."
    ),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Drive a MEP/1 server through spec → health → reset → step×N → close and report conformance."""
    from monorl.cli.envcheck import CANNED_ACTION, check_sample, run_check
    from monorl.core.types import Completion

    if bool(command) == bool(url):
        _fail("pass exactly one of --command or --url", code=2)
    try:
        reset_with = check_sample(sample, reference)
    except ValueError as e:
        _fail(str(e), code=2)
        return
    step_with = CANNED_ACTION
    if action is not None:
        step_with = Completion(text=action, token_ids=[], finish_reason="stop")
    report = run_check(
        command=list(command) if command else None,
        url=url,
        turns=turns,
        cwd=str(cwd) if cwd else None,
        timeout=timeout,
        sample=reset_with,
        action=step_with,
    )
    if as_json:
        typer.echo(json.dumps(report.to_dict(), indent=2, default=str))
    else:
        console.print(f"[bold]{report.transport}[/bold] {escape(report.target)}")
        if report.spec:
            spec = report.spec
            console.print(
                escape(
                    f"spec: name={spec['name']} version={spec['version']} protocol={spec['protocol']} "
                    f"max_turns={spec['max_turns']} parallel_episodes={spec['parallel_episodes']} "
                    f"tools={len(spec.get('tools') or [])}"
                )
            )
            if spec.get("description"):
                console.print(escape(f"  {spec['description']}"))
        if report.health is not None:
            console.print("health: ok")
        if report.reset_messages:
            console.print(f"reset: {len(report.reset_messages)} message(s)")
            for m in report.reset_messages:
                console.print(escape(_message_line(m, 200)))
        for s in report.steps:
            console.print(
                escape(
                    f"step {s.turn}: reward={s.reward} done={s.done} messages={len(s.messages)} info={s.info}"
                )
            )
        if report.closed:
            console.print("close: ok")
        if report.error:
            err_console.print(f"[bold red]protocol error:[/bold red] {escape(report.error)}")
        else:
            console.print("[green]MEP/1 conformance: OK[/green]")
    if not report.ok:
        raise typer.Exit(1)


@env_app.command("serve")
def env_serve(
    env_type: str = typer.Argument(
        ..., help="Registered env type (single_turn, tool_calling, openenv, ...)."
    ),
    transport: str = typer.Option("stdio", "--transport", help="stdio or http."),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8080, "--port"),
    config: Path | None = typer.Option(None, "--config", "-c", help="RunConfig YAML providing env.params."),
    param: list[str] | None = typer.Option(
        None, "--param", "-p", help="env.params override key=value (repeatable)."
    ),
    parallel: int = typer.Option(8, "--parallel", min=1, help="Concurrent episodes the server accepts."),
    allowed_host: list[str] | None = typer.Option(
        None,
        "--allowed-host",
        help=(
            "With --transport http: a DNS name clients may address the server by, besides localhost "
            "and its IP addresses (repeatable; '*' accepts any). Needed when other machines reach "
            "the server under a name, as in http://env-1:8080."
        ),
    ),
) -> None:
    """Expose a Python environment as a MEP/1 server (stdio or http)."""
    if transport not in ("stdio", "http"):
        _fail("--transport must be stdio or http", code=2)
    if allowed_host and transport != "http":
        _fail("--allowed-host applies to --transport http; the stdio transport has no Host names", code=2)
    from monorl.core.registry import ensure_providers, registry
    from monorl.envs.server import allowed_host_names
    from monorl.envs.server import serve as serve_env

    try:
        allowed_host_names(allowed_host or ())
    except ValueError as e:
        _fail(f"--allowed-host: {e}", code=2)
    try:
        if config is not None:
            data = load_yaml(config)
        else:
            data = {
                "model": {"name_or_path": "unused"},
                "data": {"uri": "unused"},
                "rewards": [{"type": "exact_match"}],
            }
        data.setdefault("env", {})["type"] = env_type
        if param:
            data["env"] = apply_overrides(data["env"], [f"params.{p}" for p in param])
        cfg = RunConfig.from_dict(data)
        ensure_providers(cfg)
        factory = registry.get("env", env_type)
    except MonorlError as e:
        _fail(str(e))
        return
    serve_env(
        lambda: factory(cfg),
        transport=transport,
        host=host,
        port=port,
        name=env_type,
        parallel_episodes=parallel,
        allowed_hosts=allowed_host or (),
    )


# -- bench -----------------------------------------------------------------------------------


@app.command(
    "bench",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def bench(ctx: typer.Context) -> None:
    """Throughput tooling (rollout | env | sync | plan); same as `python -m monorl.bench`."""
    from monorl.bench.cli import app as bench_app

    bench_app(args=list(ctx.args), prog_name="monorl bench")


# -- recipes ---------------------------------------------------------------------------------


@recipes_app.command("list")
def recipes_list(as_json: bool = typer.Option(False, "--json")) -> None:
    """List bundled recipes."""
    from monorl.config.recipes import list_recipes, recipes_dir

    try:
        recipes = list_recipes()
    except ConfigError as e:
        _fail(str(e))
        return
    if as_json:
        typer.echo(
            json.dumps(
                [
                    {"name": r.name, "kind": r.kind, "path": str(r.path), "description": r.description}
                    for r in recipes
                ]
            )
        )
        return
    if not recipes:
        console.print(f"no recipes found in {recipes_dir()}")
        return
    table = Table(title=f"recipes ({recipes_dir()})")
    table.add_column("name")
    table.add_column("kind")
    table.add_column("description", overflow="fold")
    for r in recipes:
        table.add_row(escape(r.name), r.kind, escape(r.description))
    console.print(table)


@recipes_app.command("show")
def recipes_show(
    name: str = typer.Argument(..., help="Recipe name (file stem)."),
    as_json: bool = typer.Option(False, "--json", help="Print the validated config as JSON."),
) -> None:
    """Print a recipe (validated against RunConfig, or PipelineConfig for `base:` + `stages:` files)."""
    from monorl.config.recipes import get_recipe

    try:
        recipe = get_recipe(name)
        data = recipe.load()
        kind = "pipeline" if is_pipeline_payload(data) else "run"
        cfg = PipelineConfig.from_dict(data) if kind == "pipeline" else RunConfig.from_dict(data)
    except ConfigError as e:
        _fail(str(e))
        return
    if as_json:
        typer.echo(
            json.dumps(
                {
                    "name": recipe.name,
                    "description": recipe.description,
                    "kind": kind,
                    "config": cfg.to_dict(),
                },
                indent=2,
            )
        )
        return
    typer.echo(recipe.path.read_text(), nl=False)


def main(argv: list[str] | None = None) -> None:
    app(args=argv if argv is not None else sys.argv[1:])


__all__ = ["app", "main"]
