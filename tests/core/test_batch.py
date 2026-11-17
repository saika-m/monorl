"""``TrainBatch.narrow`` / ``trimmed``: a micro-batch drops the padding it does not need."""

from __future__ import annotations

import torch

from monorl.core.batch import TrainBatch


def _batch() -> TrainBatch:
    lengths = [3, 5, 2]
    width = 8
    attention = torch.zeros((3, width), dtype=torch.long)
    for row, n in enumerate(lengths):
        attention[row, :n] = 1
    grid = torch.arange(3 * width).reshape(3, width)
    return TrainBatch(
        input_ids=grid.clone(),
        attention_mask=attention,
        response_mask=attention.clone(),
        advantages=grid.float(),
        rewards=torch.tensor([1.0, 2.0, 3.0]),
        group_ids=["a", "a", "b"],
        old_logprobs=grid.float() * -1,
        behavior_top_ids=torch.zeros((3, width, 2), dtype=torch.long),
        behavior_top_logprobs=torch.zeros((3, width, 2)),
        extra={
            "mm_token_type_ids": grid.clone(),  # aligned with the tokens
            "per_sequence": [None, None, None],  # one entry per row
            "patch_size": 14,  # shared
        },
    )


def test_trimmed_is_as_wide_as_the_longest_sequence_and_keeps_every_token() -> None:
    batch = _batch()
    trimmed = batch.trimmed()
    assert trimmed.input_ids.shape == (3, 5)
    for name in ("input_ids", "attention_mask", "response_mask", "advantages", "old_logprobs"):
        assert torch.equal(getattr(trimmed, name), getattr(batch, name)[:, :5]), name
    assert trimmed.behavior_top_ids.shape == (3, 5, 2) and trimmed.behavior_top_logprobs.shape == (3, 5, 2)
    assert int(trimmed.attention_mask.sum()) == int(batch.attention_mask.sum())
    # what is per row or shared is not cut
    assert torch.equal(trimmed.rewards, batch.rewards) and trimmed.group_ids == batch.group_ids
    assert trimmed.extra["mm_token_type_ids"].shape == (3, 5)
    assert trimmed.extra["per_sequence"] == [None, None, None] and trimmed.extra["patch_size"] == 14


def test_a_micro_batch_is_trimmed_to_its_own_rows() -> None:
    batch = _batch()
    assert batch.slice(0, 1).trimmed().input_ids.shape == (1, 3)
    assert batch.slice(2, 3).trimmed().input_ids.shape == (1, 2)
    assert batch.narrow(8) is batch and batch.narrow(99) is batch
