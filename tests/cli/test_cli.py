from __future__ import annotations

import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from monorl.cli.main import app
from monorl.config.recipes import get_recipe, list_recipes, read_description, recipe_files, recipes_dir
from monorl.config.schema import RunConfig
from monorl.core.errors import ConfigError
from monorl.core.types import RunResult

# -- help ---------------------------------------------------------------------------------


def test_help_lists_all_commands(runner: CliRunner) -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("train", "serve", "doctor", "patches", "data", "env", "recipes"):
        assert cmd in result.output


def test_module_entrypoint_runs_help() -> None:
    import subprocess

    proc = subprocess.run(
        [sys.executable, "-m", "monorl.cli", "--help"], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
    assert "recipes" in proc.stdout

    from monorl.cli import app as exported

    assert exported is app


# -- doctor -------------------------------------------------------------------------------


def test_doctor_json_shape(runner: CliRunner) -> None:
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert set(report) == {"python", "platform", "torch", "toolchains", "frameworks", "components", "errors"}
    assert report["python"]["version"].startswith("3.")
    assert set(report["toolchains"]) == {"go", "clang++", "g++", "cmake"}
    for name in (
        "vllm",
        "sglang",
        "verl",
        "trl",
        "nemo_rl",
        "unsloth",
        "openenv",
        "hermes",
        "ray",
        "peft",
        "transformers",
    ):
        fw = report["frameworks"][name]
        assert set(fw) >= {"installed", "version", "patches"}
        assert isinstance(fw["patches"], list)
    assert report["frameworks"]["trl"]["installed"] is True
    assert report["frameworks"]["vllm"]["installed"] is False
    assert report["frameworks"]["vllm"]["version"] is None
    assert "hf" in report["components"]["rollout_engine"]
    assert "single_turn" in report["components"]["env"]
    assert isinstance(report["errors"], list)


def test_doctor_resilient_when_framework_missing(runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    """A framework whose metadata lookup and adapter import both explode is reported, not raised."""
    import monorl.patching as patching

    real_version = importlib.metadata.version

    def fake_version(dist: str) -> str:
        if dist == "trl":
            raise importlib.metadata.PackageNotFoundError(dist)
        return real_version(dist)

    monkeypatch.setattr(importlib.metadata, "version", fake_version)
    monkeypatch.setitem(sys.modules, "monorl.adapters.trl", None)  # import raises ImportError
    # Another test in this process may already have imported the trl adapter and registered its
    # patchset; hide it for the duration so the "framework absent" report is what we assert on.
    patchsets = patching.get_patchsets("trl") if patching.is_registered("trl") else []
    patching.unregister_patchset("trl")
    try:
        _assert_trl_reported_missing(runner)
    finally:
        for ps in patchsets:
            patching.register_patchset(ps)


def _assert_trl_reported_missing(runner: CliRunner) -> None:
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["frameworks"]["trl"] == {
        "installed": False,
        "version": None,
        "distribution": "trl",
        "patches": [],
    }
    assert any("trl" in e for e in report["errors"])


def test_doctor_table(runner: CliRunner) -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output
    assert "frameworks" in result.output and "components" in result.output


# -- recipes ------------------------------------------------------------------------------


def test_recipes_dir_and_descriptions() -> None:
    recipes = list_recipes()
    names = {r.name for r in recipes}
    assert names >= {
        "grpo_gsm8k_smollm_cpu",
        "grpo_gsm8k_qwen_vllm_1gpu",
        "dapo_math_vllm_8gpu_colocated",
        "grpo_tool_calling_sglang",
        "grpo_remote_env_go_wordle",
        "grpo_remote_env_cpp_number_guess",
        "trl_grpo_cpu",
        "verl_grpo_gsm8k_8gpu",
        "nemo_rl_grpo_math",
        "s3_dataset_example",
        "openenv_example",
        "hermes_agent_example",
    }
    for r in recipes:
        assert r.description, f"{r.name} has no leading '# ' description"
        assert r.path.read_text().startswith("# ")


@pytest.mark.parametrize("recipe", recipe_files(recipes_dir()), ids=lambda p: p.stem)
def test_every_recipe_validates(recipe: Path) -> None:
    from monorl.config.loader import load_yaml
    from monorl.config.pipeline import PipelineConfig, is_pipeline_payload

    data = load_yaml(recipe)
    if is_pipeline_payload(data):
        cfg = PipelineConfig.from_dict(data).base
    else:
        cfg = RunConfig.from_dict(data)
    assert cfg.model.name_or_path


def test_cpu_recipe_values() -> None:
    cfg = RunConfig.from_yaml(recipes_dir() / "grpo_gsm8k_smollm_cpu.yaml")
    assert cfg.model.name_or_path == "HuggingFaceTB/SmolLM2-135M-Instruct"
    assert cfg.data.uri == "hf://openai/gsm8k" and cfg.data.hf_config_name == "main"
    assert cfg.rollout.backend == "hf" and cfg.trainer.backend == "reference"
    assert cfg.rollout.group_size == 4 and cfg.rollout.max_new_tokens == 128
    assert cfg.trainer.max_steps == 20 and cfg.trainer.device == "cpu"
    assert cfg.rewards[0].type == "exact_match" and cfg.rewards[0].params["extract"] == "last_number"


def test_recipes_list_and_show(runner: CliRunner) -> None:
    result = runner.invoke(app, ["recipes", "list"])
    assert result.exit_code == 0, result.output
    assert "grpo_gsm8k_smollm_cpu" in result.output

    result = runner.invoke(app, ["recipes", "list", "--json"])
    assert result.exit_code == 0
    rows = json.loads(result.stdout)
    assert {"name", "path", "description"} <= set(rows[0])

    result = runner.invoke(app, ["recipes", "show", "grpo_gsm8k_smollm_cpu"])
    assert result.exit_code == 0, result.output
    assert "HuggingFaceTB/SmolLM2-135M-Instruct" in result.output

    result = runner.invoke(app, ["recipes", "show", "grpo_gsm8k_smollm_cpu", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["config"]["rollout"]["group_size"] == 4

    result = runner.invoke(app, ["recipes", "show", "nope"])
    assert result.exit_code == 1
    assert "no recipe named" in result.output


def test_a_recipe_is_described_by_its_leading_comment_block(tmp_path: Path) -> None:
    recipe = tmp_path / "r.yaml"
    recipe.write_text(
        "#!/usr/bin/env monorl\n\n# GRPO on GSM8K\n#   with a tiny model\n\nrun_name: x\n# trailing comment\n"
    )
    assert read_description(recipe) == "GRPO on GSM8K with a tiny model"
    recipe.write_text("run_name: x\n# not a description\n")
    assert read_description(recipe) == ""


def test_recipes_dir_env_override(runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "custom.yaml").write_text(
        "# Custom recipe.\n# Second line.\nmodel: {name_or_path: x}\ndata: {uri: y}\nrewards: [{type: exact_match}]\n"
    )
    monkeypatch.setenv("MONORL_RECIPES_DIR", str(tmp_path))
    assert read_description(tmp_path / "custom.yaml") == "Custom recipe. Second line."
    result = runner.invoke(app, ["recipes", "list", "--json"])
    assert [r["name"] for r in json.loads(result.output)] == ["custom"]


RECIPE_BODY = "model: {name_or_path: x}\ndata: {uri: y}\nrewards: [{type: exact_match}]\n"


def test_yml_recipe_is_listed_and_shown(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "b.yaml").write_text("# Long extension.\n" + RECIPE_BODY)
    (tmp_path / "a.yml").write_text("# Short extension.\n" + RECIPE_BODY)
    (tmp_path / "c.yaml").mkdir()
    (tmp_path / "notes.txt").write_text("ignored")
    monkeypatch.setenv("MONORL_RECIPES_DIR", str(tmp_path))
    assert [p.name for p in recipe_files(tmp_path)] == ["a.yml", "b.yaml"]

    result = runner.invoke(app, ["recipes", "list", "--json"])
    assert result.exit_code == 0, result.output
    rows = json.loads(result.stdout)
    assert [(r["name"], r["description"]) for r in rows] == [
        ("a", "Short extension."),
        ("b", "Long extension."),
    ]

    for name in ("a", "a.yml"):
        result = runner.invoke(app, ["recipes", "show", name, "--json"])
        assert result.exit_code == 0, result.output
        shown = json.loads(result.stdout)
        assert shown["name"] == "a" and shown["config"]["model"]["name_or_path"] == "x"
    assert get_recipe("b.yaml").name == "b"

    result = runner.invoke(app, ["recipes", "show", "a.yaml"])
    assert result.exit_code == 1
    assert "available: a, b" in result.output.replace("\n", " ")


def test_recipe_with_both_extensions_is_an_error(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "twin.yaml").write_text("# One.\n" + RECIPE_BODY)
    (tmp_path / "twin.yml").write_text("# Other.\n" + RECIPE_BODY)
    (tmp_path / "solo.yaml").write_text("# Alone.\n" + RECIPE_BODY)
    monkeypatch.setenv("MONORL_RECIPES_DIR", str(tmp_path))
    with pytest.raises(ConfigError, match=r"'twin'.*twin\.yaml and twin\.yml"):
        recipe_files(tmp_path)
    for args in (["recipes", "list"], ["recipes", "show", "twin"], ["recipes", "show", "solo"]):
        result = runner.invoke(app, args)
        assert result.exit_code == 1, result.output
        flat = " ".join(result.output.split())
        assert "twin.yaml" in flat and "twin.yml" in flat


# -- train --------------------------------------------------------------------------------


def test_train_invokes_orchestrator_with_overrides(
    runner: CliRunner, run_config_dict: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import monorl.orchestrator

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(run_config_dict))
    calls: list[tuple[RunConfig, Path | None, bool]] = []

    def fake_run(config: RunConfig, run_dir: Path | None = None, *, resume: bool = False) -> RunResult:
        calls.append((config, run_dir, resume))
        return RunResult(
            state="succeeded", final_step=7, run_dir=str(tmp_path / "rd"), metrics={"reward/mean": 0.5}
        )

    monkeypatch.setattr(monorl.orchestrator, "run", fake_run)
    result = runner.invoke(
        app,
        [
            "train",
            "--config",
            str(cfg_path),
            "--set",
            "trainer.max_steps=7",
            "--set",
            "rollout.temperature=0.5",
            "--run-dir",
            str(tmp_path / "rd"),
            "--resume",
        ],
    )
    assert result.exit_code == 0, result.output
    config, run_dir, resume = calls[0]
    assert config.trainer.max_steps == 7 and config.rollout.temperature == 0.5
    assert config.model.name_or_path == run_config_dict["model"]["name_or_path"]
    assert run_dir == tmp_path / "rd" and resume is True
    assert "reward/mean" in result.output and "final step: 7" in result.output


@pytest.mark.parametrize("state,code", [("failed", 1), ("cancelled", 130)])
def test_train_exit_codes(
    runner: CliRunner,
    run_config_dict: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state: str,
    code: int,
) -> None:
    import monorl.orchestrator

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(run_config_dict))
    monkeypatch.setattr(
        monorl.orchestrator,
        "run",
        lambda config, run_dir=None, *, resume=False: RunResult(
            state=state, final_step=1, run_dir="x", error="boom"
        ),
    )
    result = runner.invoke(app, ["train", "--config", str(cfg_path)])
    assert result.exit_code == code


def test_train_rejects_invalid_config(runner: CliRunner, tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("model: {name_or_path: x}\n")  # missing data
    result = runner.invoke(app, ["train", "--config", str(cfg_path)])
    assert result.exit_code == 1
    assert "error" in result.output

    result = runner.invoke(app, ["train", "--config", str(tmp_path / "missing.yaml")])
    assert result.exit_code == 1
    assert "not found" in result.output


def test_train_names_the_file_and_the_key_of_an_invalid_config(
    runner: CliRunner, run_config_dict: dict[str, Any], tmp_path: Path
) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(run_config_dict))
    result = runner.invoke(
        app,
        [
            "train",
            "--config",
            str(cfg_path),
            "--set",
            "trainer.max_stps=1",
            "--set",
            "rollout.group_size=two",
        ],
        env={"COLUMNS": "1000"},
    )
    assert result.exit_code == 1
    assert [line.strip() for line in result.output.strip().splitlines()] == [
        f"error: {cfg_path}: invalid RunConfig, 2 problems:",
        "rollout.group_size: Input should be a valid integer, unable to parse string as an integer (got 'two')",
        "trainer.max_stps: unknown key; did you mean max_steps?",
    ]


# -- env check ----------------------------------------------------------------------------


def test_env_check_stdio_ok(runner: CliRunner, mep_server_script: Path) -> None:
    result = runner.invoke(
        app,
        ["env", "check", "--command", sys.executable, "--command", str(mep_server_script), "--turns", "2"],
    )
    assert result.exit_code == 0, result.output
    assert "name=echo" in result.output
    assert "step 1" in result.output and "step 2" in result.output
    assert "MEP/1 conformance: OK" in result.output


def test_env_check_json_and_done_stops_early(runner: CliRunner, mep_server_script: Path) -> None:
    result = runner.invoke(
        app,
        [
            "env",
            "check",
            "--json",
            "--command",
            sys.executable,
            "--command",
            str(mep_server_script),
            "--turns",
            "5",
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["ok"] and report["closed"]
    assert report["spec"]["name"] == "echo" and report["spec"]["protocol"] == "MEP/1"
    assert report["health"] == {"ok": True}
    assert report["reset"]["messages"][0]["role"] == "user"
    assert len(report["steps"]) == 3 and report["steps"][-1]["done"] is True
    assert report["steps"][0]["reward"] == 5.0  # len("hello")


def test_env_check_protocol_error_exit_1(runner: CliRunner, mep_server_script: Path) -> None:
    result = runner.invoke(
        app,
        [
            "env",
            "check",
            "--command",
            sys.executable,
            "--command",
            str(mep_server_script),
            "--command",
            "--bad-spec",
        ],
    )
    assert result.exit_code == 1
    assert "protocol error" in result.output and "name" in result.output


def test_env_check_server_death_exit_1(runner: CliRunner, mep_server_script: Path) -> None:
    result = runner.invoke(
        app,
        [
            "env",
            "check",
            "--command",
            sys.executable,
            "--command",
            str(mep_server_script),
            "--command",
            "--die-on-step",
        ],
    )
    assert result.exit_code == 1
    assert "closed stdout" in result.output


def _check_guessing_server(runner: CliRunner, script: Path, *options: str) -> Any:
    argv = ["env", "check", "--json", "--command", sys.executable, "--command", str(script)]
    return runner.invoke(app, [*argv, "--command", "--int-reference", *options])


def test_env_check_built_in_sample_rejected_by_reset_says_how_to_give_one(
    runner: CliRunner, mep_server_script: Path
) -> None:
    result = _check_guessing_server(runner, mep_server_script)
    assert result.exit_code == 1
    error = json.loads(result.stdout)["error"]
    assert "reference must be an integer, got 'hello'" in error
    assert "--sample" in error and "--reference" in error

    # a sample the caller chose is the caller's to fix: the server's error stands alone
    result = _check_guessing_server(runner, mep_server_script, "--reference", "seven")
    assert result.exit_code == 1
    assert json.loads(result.stdout)["error"] == "ValueError: reference must be an integer, got 'seven'"


def test_env_check_reference_and_action_options(runner: CliRunner, mep_server_script: Path) -> None:
    result = _check_guessing_server(runner, mep_server_script, "--reference", "7", "--turns", "2")
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    sent = report["reset"]["info"]["sample"]
    assert sent["reference"] == 7
    # everything else is still the built-in sample and the built-in action
    assert sent["id"] == "env-check-0" and report["reset"]["messages"] == sent["messages"]
    assert "monorl env check" in sent["messages"][0]["content"]
    assert [(s["reward"], s["done"]) for s in report["steps"]] == [(0.0, False), (0.0, False)]
    assert report["steps"][0]["messages"] == [{"role": "user", "content": "echo:hello"}]

    result = _check_guessing_server(
        runner, mep_server_script, "--reference", "7", "--action", "7", "--turns", "2"
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert [(s["reward"], s["done"]) for s in report["steps"]] == [(1.0, True)]


def test_env_check_sample_file(runner: CliRunner, mep_server_script: Path, tmp_path: Path) -> None:
    given = {
        "messages": [{"role": "system", "content": "Guess."}, {"role": "user", "content": "Go."}],
        "reference": 12,
        "metadata": {"range": 20},
    }
    path = tmp_path / "sample.json"
    path.write_text(json.dumps(given), encoding="utf-8")
    result = _check_guessing_server(runner, mep_server_script, "--sample", str(path), "--action", "12")
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["reset"]["info"]["sample"] == {"id": "env-check-0", **given}
    assert report["steps"][0]["reward"] == 1.0

    # --reference replaces the file's reference and nothing else
    result = _check_guessing_server(runner, mep_server_script, "--sample", str(path), "--reference", "3")
    assert json.loads(result.stdout)["reset"]["info"]["sample"] == {
        "id": "env-check-0",
        **given,
        "reference": 3,
    }


@pytest.mark.parametrize(
    "content,expected",
    [
        (None, "cannot read sample file"),
        ("{not json", "is not JSON"),
        ("[1, 2]", "must hold a JSON object"),
        ('{"refrence": 7}', "unknown keys ['refrence']"),
        ('{"messages": "Guess."}', "messages must be a list"),
        ('{"metadata": [1]}', "metadata must be an object"),
    ],
)
def test_env_check_rejects_a_bad_sample_file(
    runner: CliRunner, mep_server_script: Path, tmp_path: Path, content: str | None, expected: str
) -> None:
    path = tmp_path / "sample.json"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    result = _check_guessing_server(runner, mep_server_script, "--sample", str(path))
    assert result.exit_code == 2
    assert result.stdout == ""  # no report: the server was never asked anything
    assert expected in " ".join(result.stderr.split())


def test_env_check_requires_exactly_one_target(runner: CliRunner) -> None:
    assert runner.invoke(app, ["env", "check"]).exit_code == 2
    result = runner.invoke(app, ["env", "check", "--command", "x", "--url", "http://h"])
    assert result.exit_code == 2


def test_env_check_http(runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the http transport against an in-memory ASGI MEP/1 server via httpx's mock transport."""
    import httpx

    from monorl.cli import envcheck

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/spec":
            return httpx.Response(200, json={"name": "http-echo", "protocol": "MEP/1"})
        if path == "/health":
            return httpx.Response(200, json={"ok": True})
        body = json.loads(request.content)
        if path == "/reset":
            return httpx.Response(200, json={"messages": body["sample"]["messages"], "info": {}})
        if path == "/step":
            return httpx.Response(200, json={"messages": [], "reward": 1.0, "done": True})
        if path == "/close":
            return httpx.Response(200, json={})
        return httpx.Response(404, json={"error": {"type": "not_found", "message": path}})

    real_client = httpx.Client

    def client_factory(**kw: Any) -> httpx.Client:
        return real_client(transport=httpx.MockTransport(handler), **kw)

    monkeypatch.setattr(httpx, "Client", client_factory)
    report = envcheck.run_check(url="http://env.test", turns=3)
    assert report.ok, report.error
    assert report.spec["name"] == "http-echo"
    assert len(report.steps) == 1 and report.closed

    result = runner.invoke(app, ["env", "check", "--url", "http://env.test", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["spec"]["name"] == "http-echo"


# -- patches ------------------------------------------------------------------------------


def test_patches_list_and_status(runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    import monorl.patching as patching
    from monorl.patching import PatchStatus

    statuses = {
        "vllm": [
            PatchStatus(
                id="vllm.sleep_mode",
                target="vllm.LLM.sleep",
                version_spec=">=0.8",
                installed_version=None,
                applicable=False,
                applied=False,
                verified=None,
                reason="vllm not installed",
                framework="vllm",
                mode="wrap",
            ),
        ],
        "trl": [
            PatchStatus(
                id="trl.reporter",
                target="trl.GRPOTrainer.log",
                version_spec=">=1.0",
                installed_version="1.17.1",
                applicable=True,
                applied=True,
                verified=True,
                framework="trl",
                mode="wrap",
            ),
        ],
    }
    monkeypatch.setattr(patching, "registered_frameworks", lambda: sorted(statuses))
    monkeypatch.setattr(patching, "status", lambda fw, resolve=True: statuses.get(fw, []))

    result = runner.invoke(app, ["patches", "list"])
    assert result.exit_code == 0, result.output
    assert "patches: vllm" in result.output and "patches: trl" in result.output

    result = runner.invoke(app, ["patches", "list", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["trl"][0]["applied"] is True and data["vllm"][0]["installed_version"] is None

    result = runner.invoke(app, ["patches", "status", "trl"])
    assert result.exit_code == 0 and "patches: trl" in result.output and "patches: vllm" not in result.output

    result = runner.invoke(app, ["patches", "status", "nothing"])
    assert result.exit_code == 0 and "no patches registered" in result.output


def test_patches_apply_reports(runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    import monorl.patching as patching
    from monorl.core.errors import PatchError
    from monorl.patching import ApplyReport

    def fake_apply(framework: str, force: bool = False) -> ApplyReport:
        if framework == "unknown":
            raise PatchError("no patchset registered for 'unknown'")
        return ApplyReport(
            framework=framework,
            installed_version="1.17.1",
            applied=["trl.reporter"],
            skipped=[("trl.other", "version mismatch")],
            failed=[],
        )

    monkeypatch.setattr(patching, "apply", fake_apply)
    result = runner.invoke(app, ["patches", "apply", "trl"])
    assert result.exit_code == 0, result.output
    assert "applied" in result.output and "trl.reporter" in result.output and "skipped" in result.output

    result = runner.invoke(app, ["patches", "apply", "unknown"])
    assert result.exit_code == 1 and "no patchset" in result.output


# -- data ---------------------------------------------------------------------------------


def test_data_inspect_local_jsonl(runner: CliRunner, tmp_path: Path) -> None:
    pytest.importorskip("monorl.data.browse")
    f = tmp_path / "d.jsonl"
    f.write_text('{"prompt": "What is 2+2?", "answer": "4"}\n{"prompt": "What is 3+3?", "answer": "6"}\n')
    result = runner.invoke(app, ["data", "inspect", str(f), "--limit", "1", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert set(data["columns"]) == {"prompt", "answer"}
    assert data["inferred"]["prompt_field"] == "prompt" and data["inferred"]["answer_field"] == "answer"
    assert len(data["rows"]) == 1 and len(data["samples"]) == 1
    assert data["samples"][0]["messages"][-1]["content"] == "What is 2+2?"

    result = runner.invoke(app, ["data", "inspect", str(f)])
    assert result.exit_code == 0, result.output
    assert "What is 2+2?" in result.output

    result = runner.invoke(app, ["data", "inspect", str(tmp_path / "missing.jsonl")])
    assert result.exit_code == 1


def test_data_inspect_reports_a_missing_path_without_a_traceback(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(app, ["data", "inspect", "openai/gsm8k"])
    assert result.exit_code == 1 and not isinstance(result.exception, FileNotFoundError)
    flat = " ".join(result.output.split())
    assert "dataset path not found: openai/gsm8k" in flat and "hf://openai/gsm8k" in flat
    assert "Traceback" not in result.output


def test_data_inspect_reads_the_data_source_of_a_provider(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = f"inspect_provider_{tmp_path.name.replace('-', '_')}"
    (tmp_path / f"{name}.py").write_text(
        "from monorl.data.rows import RowSource\n\n\n"
        "class Squares(RowSource):\n"
        "    def __init__(self, uri):\n"
        "        self.uri = uri.raw\n"
        "        self.rows = [(f'{uri.raw}#{i}', {'question': f'{i} squared?', 'answer': str(i * i)}) "
        "for i in range(int(uri.path))]\n\n"
        "    def columns(self):\n"
        "        return {'question': 'string', 'answer': 'string'}\n\n"
        "    def iter_rows(self, *, shuffle_seed=None):\n"
        "        return iter(self.rows)\n\n\n"
        "def register(registry):\n"
        "    registry.register('data_source', 'squares', lambda uri, **_: Squares(uri), overwrite=True)\n"
    )
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["data", "inspect", "squares://4", "--provider", name, "--json"])
    assert result.exit_code == 0, result.output
    out = json.loads(result.output)
    assert [s["reference"] for s in out["samples"]] == ["0", "1", "4", "9"]
    without = runner.invoke(app, ["data", "inspect", "nosuchscheme://4"])
    assert without.exit_code == 1 and "no data source for the scheme 'nosuchscheme'" in " ".join(
        without.output.split()
    )


def test_data_inspect_renders_media_parts_as_markers(runner: CliRunner, tmp_path: Path) -> None:
    image_module = pytest.importorskip("PIL.Image")
    image_module.new("RGB", (4, 4), (255, 0, 0)).save(tmp_path / "red.png")
    user = [{"type": "image", "image": str(tmp_path / "red.png")}, {"type": "text", "text": "What colour?"}]
    row = {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": "Red"}]}
    f = tmp_path / "chat.jsonl"
    f.write_text(json.dumps(row) + "\n")

    result = runner.invoke(app, ["data", "inspect", str(f), "--limit", "1"], env={"COLUMNS": "1000"})
    assert result.exit_code == 0, result.output
    sample_lines = [line for line in result.output.splitlines() if line.lstrip().startswith("[user]")]
    assert [" ".join(line.split()) for line in sample_lines] == [
        f"[user] <image: {tmp_path / 'red.png'}> What colour?"
    ]
