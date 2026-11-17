"""Every shipped recipe can be placed, and gets the placement its `cluster` block asks for.

A recipe that validates as a config can still fail to start: the rules that tie `cluster.mode`,
`placement` and `weight_sync` together run when the runtime is built, which no test did for the
recipes (`weight_sync: broadcast` with `mode: local` sat in one of them from the first commit
on). Planning needs no GPU: a recipe names how many it wants.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from monorl.config.loader import load_yaml
from monorl.config.pipeline import PipelineConfig, is_pipeline_payload
from monorl.config.recipes import recipe_files, recipes_dir
from monorl.config.schema import RunConfig
from monorl.distributed import plan_placement
from monorl.distributed.cluster import detect_cluster
from monorl.pipeline.stages import build_sampling_config, build_train_config


def run_configs(recipe: Path, stages_root: Path) -> list[tuple[str, RunConfig]]:
    """The runs a recipe starts: itself, or for a pipeline the config of every stage as the
    stage runner builds it (a stage that takes its data from the one before gets a path)."""
    data = load_yaml(recipe)
    if not is_pipeline_payload(data):
        return [(recipe.stem, RunConfig.from_dict(data))]
    pipeline = PipelineConfig.from_dict(data)
    configs = []
    for index, stage in enumerate(pipeline.stages):
        if stage.type == "train":
            config = build_train_config(
                pipeline,
                index,
                stages_root=stages_root,
                previous_model=None,
                previous_dataset=str(stages_root / "dataset.jsonl"),
            )
        else:
            config = build_sampling_config(pipeline, index, previous_model=None)
        configs.append((f"{recipe.stem}/{stage.name}", config))
    return configs


@pytest.mark.parametrize("recipe", recipe_files(recipes_dir()), ids=lambda p: p.stem)
def test_every_recipe_plans_as_its_cluster_block_says(recipe: Path, tmp_path: Path) -> None:
    cluster = detect_cluster(use_ray=False)
    configs = run_configs(recipe, tmp_path)
    assert configs, f"{recipe.name} starts no run"
    for name, config in configs:
        plan = plan_placement(config, cluster)
        asked = config.cluster
        assert plan.mode == asked.mode, name
        if asked.placement != "auto":
            assert plan.placement == asked.placement, name
        if asked.weight_sync != "auto":
            assert plan.weight_sync == asked.weight_sync, name
        if plan.total_gpus:
            assert plan.engine_gpus <= plan.total_gpus and plan.trainer_gpus <= plan.total_gpus, name
            if plan.placement == "disaggregated":
                assert plan.engine_gpus + plan.trainer_gpus <= plan.total_gpus, name
