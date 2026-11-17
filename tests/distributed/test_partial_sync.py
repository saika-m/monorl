"""A sync leaves out what an engine already holds (``monorl.distributed.weight_sync.SentWeights``):
whatever is left out, an engine's weights have to be the trainer's after every sync."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from distributed_fakes import AdapterTrainer, FakeEngine, FakeTrainer, FlakyEngine, PatchingEngine, make_batch

from monorl.distributed import InProcessWeightSync
from monorl.mtp import WIRE_PREFIX

EMBED, PROJ = "model.embed.weight", "model.layers.0.proj.weight"


def _bytes(trainer: FakeTrainer, names: list[str]) -> int:
    return sum(trainer.weights[name].numel() * trainer.weights[name].element_size() for name in names)


def _assert_holds(engine: FakeEngine, trainer: FakeTrainer) -> None:
    for name, tensor in trainer.export_weights():
        assert torch.equal(engine.weights[name], tensor), name


def test_after_the_first_sync_only_trained_tensors_are_sent():
    trainer, engines = AdapterTrainer(seed=5), [PatchingEngine(seed=0), PatchingEngine(seed=3)]
    sync = InProcessWeightSync()

    first = sync.sync(trainer, engines)
    assert [e.received for e in engines] == [[EMBED, PROJ]] * 2
    assert first["weight_sync/bytes"] == 2 * _bytes(trainer, [EMBED, PROJ])

    for _ in range(3):
        trainer.train_step(make_batch())
        metrics = sync.sync(trainer, engines)
        assert [e.received for e in engines] == [[PROJ]] * 2
        assert metrics["weight_sync/bytes"] == 2 * _bytes(trainer, [PROJ])
        for engine in engines:
            _assert_holds(engine, trainer)


def test_frozen_tensor_that_was_written_or_replaced_is_sent_again():
    trainer, engine = AdapterTrainer(seed=5), PatchingEngine(seed=0)
    sync = InProcessWeightSync()
    sync.sync(trainer, [engine])

    trainer.weights[EMBED].mul_(2.0)  # in place, as a checkpoint is loaded
    sync.sync(trainer, [engine])
    assert engine.received == [EMBED, PROJ]
    _assert_holds(engine, trainer)

    # A new tensor each time, as merged weights are: the old one is gone when the new one is made,
    # so the new one may well lie where the old one was.
    for i in range(3):
        values = (trainer.weights.pop(EMBED) + i + 1).tolist()
        trainer.weights = {EMBED: torch.tensor(values), **trainer.weights}
        sync.sync(trainer, [engine])
        assert engine.received == [EMBED, PROJ]
        _assert_holds(engine, trainer)

    sync.sync(trainer, [engine])
    assert engine.received == [PROJ]


@pytest.mark.parametrize(
    ("trainer", "engine"),
    [(AdapterTrainer, FakeEngine), (FakeTrainer, PatchingEngine)],
    ids=["engine does not take partial pushes", "trainer does not name what it trains"],
)
def test_everything_is_sent_unless_trainer_and_engine_both_allow_less(trainer, engine):
    trainer, engine, sync = trainer(seed=5), engine(seed=0), InProcessWeightSync()
    for _ in range(3):
        metrics = sync.sync(trainer, [engine])
        assert engine.received == [EMBED, PROJ]
        assert metrics["weight_sync/bytes"] == _bytes(trainer, [EMBED, PROJ])
        _assert_holds(engine, trainer)
        trainer.train_step(make_batch())


def test_failed_sync_is_followed_by_a_full_one():
    trainer, steady, flaky = AdapterTrainer(seed=5), PatchingEngine(seed=0), FlakyEngine(seed=3)
    sync = InProcessWeightSync()
    sync.sync(trainer, [steady, flaky])
    trainer.train_step(make_batch())

    flaky.failing = True
    with pytest.raises(RuntimeError, match="broke down"):
        sync.sync(trainer, [steady, flaky])
    flaky.failing = False
    trainer.train_step(make_batch())
    sync.sync(trainer, [steady, flaky])

    assert flaky.received == [EMBED, PROJ] and steady.received == [PROJ]
    for engine in (steady, flaky):
        _assert_holds(engine, trainer)


def test_engine_that_replaces_another_gets_everything():
    trainer, kept, old = AdapterTrainer(seed=5), PatchingEngine(seed=0), PatchingEngine(seed=3)
    sync = InProcessWeightSync()
    sync.sync(trainer, [kept, old])
    trainer.train_step(make_batch())

    new = PatchingEngine(seed=7)
    del old
    sync.sync(trainer, [kept, new])
    assert new.received == [EMBED, PROJ] and kept.received == [PROJ]
    _assert_holds(new, trainer)


def test_mtp_weights_are_sent_all_together_or_not_at_all():
    """An engine replaces its drafter's weights from a complete set only (``MtpLayers.load_checkpoint_weights``)."""
    head, norm = WIRE_PREFIX + "mtp.head.weight", WIRE_PREFIX + "mtp.norm.weight"

    def with_mtp(holder: FakeEngine | FakeTrainer, seed: int) -> None:
        g = torch.Generator().manual_seed(seed)
        holder.weights.update({head: torch.randn(4, 4, generator=g), norm: torch.randn(4, generator=g)})

    trainer, engine = AdapterTrainer(seed=5, trained=(PROJ,)), PatchingEngine(seed=0)
    with_mtp(trainer, 1)
    with_mtp(engine, 2)
    sync = InProcessWeightSync()
    sync.sync(trainer, [engine])
    assert engine.received == [EMBED, PROJ, head, norm]

    trainer.train_step(make_batch())  # frozen MTP layers: none of them
    sync.sync(trainer, [engine])
    assert engine.received == [PROJ]

    trainer.trained = (PROJ, head)  # one of them trained: all of them
    trainer.train_step(make_batch())
    sync.sync(trainer, [engine])
    assert engine.received == [PROJ, head, norm]
    _assert_holds(engine, trainer)


# -- the reference trainer and the hf engine ---------------------------------------------------


def _hf_engine(tiny_model_dir: Path, device: str = "cpu"):
    """An engine whose weights are not the trainer's, so that a tensor it is not sent shows."""
    from monorl.rollout.hf_engine import HFRolloutEngine
    from tests.trainer.conftest import make_config

    config = make_config(tiny_model_dir)
    config.rollout.device = device
    engine = HFRolloutEngine(config)
    with torch.no_grad():
        for parameter in engine.model.parameters():
            parameter.add_(1.0)
    return engine


def _train_batch(tiny_model_dir: Path):
    from transformers import AutoTokenizer

    from tests.trainer.conftest import make_batch as make_train_batch

    return make_train_batch(AutoTokenizer.from_pretrained(tiny_model_dir))


def _assert_hf_engines_hold(engines, trainer) -> None:
    exported = dict(trainer.export_weights())
    for engine in engines:
        state = engine.model.state_dict()
        assert set(state) == set(exported)
        for name, tensor in exported.items():
            assert torch.equal(state[name].cpu(), tensor), name


@pytest.mark.parametrize("lora", [True, False], ids=["lora", "full"])
def test_hf_engines_hold_the_trainers_weights_after_every_sync(tiny_model_dir: Path, tmp_path: Path, lora):
    from monorl.trainer.reference import ReferenceTrainer
    from tests.trainer.conftest import make_config, reinforce_loss

    def synced(engines) -> float:
        metrics = sync.sync(trainer, engines)
        _assert_hf_engines_hold(engines, trainer)
        return metrics["weight_sync/bytes"]

    batch = _train_batch(tiny_model_dir)
    trainer = ReferenceTrainer(make_config(tiny_model_dir, lora=lora), loss_fn=reinforce_loss)
    sync = InProcessWeightSync()
    engines = [_hf_engine(tiny_model_dir), _hf_engine(tiny_model_dir)]
    try:
        everything = sum(t.numel() * t.element_size() for _, t in trainer.export_weights())
        assert synced(engines) == 2 * everything
        for _ in range(3):
            trainer.train_step(batch)
            sent = synced(engines)
            if lora:  # the embedding is most of the model and no adapter touches it
                assert 0 < sent < 2 * everything / 10
            else:
                assert sent == 2 * everything
        if lora:
            trainer.save_checkpoint(tmp_path / "ckpt")
            trainer.train_step(batch)
            synced(engines)
            trainer.load_checkpoint(tmp_path / "ckpt")
            synced(engines)
        before = synced(engines)
        engines.append(_hf_engine(tiny_model_dir))
        assert synced(engines) == before + everything
    finally:
        for engine in engines:
            engine.shutdown()


@pytest.mark.parametrize(
    "device",
    [
        "cpu",  # where sleeping releases nothing
        pytest.param(
            "cuda",
            marks=[pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")],
        ),
    ],
)
def test_hf_engine_that_slept_holds_the_trainers_weights(tiny_model_dir: Path, device):
    """Partial pushes count on the engine keeping its weights while it sleeps, and reach it asleep."""
    from monorl.trainer.reference import ReferenceTrainer
    from tests.trainer.conftest import make_config, reinforce_loss

    batch = _train_batch(tiny_model_dir)
    trainer = ReferenceTrainer(make_config(tiny_model_dir, lora=True), loss_fn=reinforce_loss)
    sync = InProcessWeightSync()
    engine = _hf_engine(tiny_model_dir, device)
    try:
        everything = sync.sync(trainer, [engine])["weight_sync/bytes"]
        for sync_asleep in (True, False):
            engine.sleep()
            trainer.train_step(batch)
            if not sync_asleep:
                engine.wake_up()
            assert sync.sync(trainer, [engine])["weight_sync/bytes"] < everything / 10
            engine.wake_up()
            _assert_hf_engines_hold([engine], trainer)
    finally:
        engine.shutdown()
