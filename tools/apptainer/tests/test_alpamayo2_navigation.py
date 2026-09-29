# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Check navigation prompt/processor wiring without loading Torch or weights."""

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from alpasim_driver.alpamayo2_inputs import prepare_navigation_inputs


@pytest.fixture
def backend(monkeypatch):
    calls = {}
    torch = ModuleType("torch")
    torch.uint8 = "uint8"
    monkeypatch.setitem(sys.modules, "torch", torch)
    conversation = ModuleType("alpamayo2_super.chat_template.conversation")

    def build_conversation(**kwargs):
        calls["conversation"] = kwargs
        return [
            dict(role="user", content=kwargs["data"]["nav_text"][0]),
            dict(role="assistant", content=""),
        ]

    conversation.build_conversation = build_conversation
    monkeypatch.setitem(sys.modules, conversation.__name__, conversation)

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            calls["template"] = kwargs
            assert len(messages) == 1
            return messages[0]["content"]

        def __call__(self, **kwargs):
            calls["processor"] = kwargs
            return dict(input_ids=np.ones((1, 5), dtype=int))

    helper = ModuleType("alpamayo2_super.helper")
    helper.get_processor = lambda tokenizer, config: Processor()
    monkeypatch.setitem(sys.modules, helper.__name__, helper)
    return calls


def inputs():
    class Frames:
        dtype = "uint8"

        def flatten(self, start, end):
            assert (start, end) == (0, 1)
            return self

        def float(self):
            return np.full((4, 3, 2, 2), 255.0)

    data = dict(
        image_frames=Frames(),
        camera_indices=[1],
        ego_history_xyz=object(),
        ego_history_rot=object(),
    )
    config = SimpleNamespace(
        tokens_per_history_traj=16,
        tokens_per_future_traj=64,
        include_camera_ids=True,
        frame_label="frame_num",
    )
    return data, config


def test_navigation_reaches_tokenizer_and_preserves_history(backend):
    data, config = inputs()
    instruction = "Turn right at the upcoming intersection, then continue straight."
    result = prepare_navigation_inputs(data, config, object(), instruction)
    assert backend["conversation"]["components_order"] == [
        "image",
        "traj_history",
        "nav_instruction",
        "prompt",
    ]
    assert backend["conversation"]["data"]["nav_text"] == [instruction]
    assert "nav_text" not in data
    assert backend["processor"]["text"] == instruction
    assert backend["processor"]["do_rescale"] is False
    np.testing.assert_array_equal(backend["processor"]["images"], np.ones((4, 3, 2, 2)))
    assert result["ego_history_xyz"] is data["ego_history_xyz"]
    assert result["ego_history_rot"] is data["ego_history_rot"]
    assert "unguided_tokenized_data" not in result


@pytest.mark.parametrize("instruction", [None, "", "  ", 123])
def test_empty_or_nontext_navigation_is_rejected(backend, instruction):
    data, config = inputs()
    with pytest.raises(ValueError, match="nonempty text"):
        prepare_navigation_inputs(data, config, object(), instruction)
    assert backend == {}
