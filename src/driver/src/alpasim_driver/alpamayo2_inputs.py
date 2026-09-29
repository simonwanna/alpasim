# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Navigation-conditioned single-prompt inputs for Alpamayo 2 (without CFG)."""


def prepare_navigation_inputs(data, config, tokenizer, instruction):
    """Include the native navigation component in the model's tokenized prompt.

    Uses the navigation template from the upstream two_gpu_nav_cfg_demo, but
    prepares only the conditioned prompt for ordinary single-prompt inference.
    Instruction following without CFG must be evaluated in the driving run.
    """
    import torch
    from alpamayo2_super.chat_template.conversation import build_conversation
    from alpamayo2_super.helper import get_processor

    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Navigation instruction must be nonempty text")
    processor = get_processor(tokenizer, config)
    messages = build_conversation(
        data=dict(data, nav_text=[instruction]),
        num_tokens_per_history_traj=config.tokens_per_history_traj,
        num_tokens_per_future_traj=config.tokens_per_future_traj,
        components_order=["image", "traj_history", "nav_instruction", "prompt"],
        components_prompt=["cot", "traj_future"],
        generation_mode=True,
        include_camera_ids=config.include_camera_ids,
        camera_ids=data["camera_indices"],
        include_frame_nums=config.frame_label == "frame_num",
    )
    if messages[-1]["role"] == "assistant" and not messages[-1]["content"]:
        messages = messages[:-1]
    has_assistant_content = messages[-1]["role"] == "assistant" and bool(
        messages[-1]["content"]
    )
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=not has_assistant_content,
        add_vision_id=False,
        continue_final_message=has_assistant_content,
    )
    images = data["image_frames"].flatten(0, 1)
    images = images.float() / 255.0 if images.dtype == torch.uint8 else images.float()
    tokenized_data = dict(
        processor(
            text=text,
            images=images,
            videos=None,
            padding=False,
            return_tensors="pt",
            do_rescale=False,
        )
    )
    if tokenized_data["input_ids"].shape[0] != 1:
        raise ValueError("Navigation inputs expect one sample at a time")
    return {
        "tokenized_data": tokenized_data,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
