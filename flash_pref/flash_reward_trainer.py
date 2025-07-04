from importlib.util import find_spec
from typing import Any, Union

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedModel

from flash_pref.shared_prefix import shared_prefix

is_trl_available = find_spec("trl") is not None

if is_trl_available:
    from trl import RewardTrainer
else:
    RewardTrainer = object


class FlashRewardTrainer(RewardTrainer):
    r"""A drop-in replacement for trl.RewardTrainer with prefix sharing acceleration."""

    def __init__(self, *args, use_shared_prefix: bool = True, **kwargs):
        if not is_trl_available:
            raise RuntimeError(f"{self.__class__.__name__} requires trl. Run `pip install trl` to install it.")

        super().__init__(*args, **kwargs)

        self.use_shared_prefix = use_shared_prefix

        if not self.use_shared_prefix:
            return

        assert not self.use_apex, f"{self.__class__.__name__} does not support use_apex"

        self.last_inputs = {}

        def patch_accelerator_backward(accelerator):
            def accelerator_backward_wrapper(accelerator, *args, **kwargs):
                with shared_prefix(self.model, **self.last_inputs, interleaved=True, enabled=self.use_shared_prefix):
                    return old_accelerator_backward(*args, **kwargs)

            old_accelerator_backward = accelerator.backward
            accelerator.backward = accelerator_backward_wrapper.__get__(accelerator)

        patch_accelerator_backward(self.accelerator)

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
        num_items_in_batch=None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, dict[str, torch.Tensor]]]:

        if not self.use_shared_prefix:
            return super().compute_loss(
                model=model, inputs=inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
            )

        input_ids = pad_sequence(
            [*inputs["input_ids_chosen"], *inputs["input_ids_rejected"]],
            batch_first=True,
            padding_value=model.config.pad_token_id,
        )
        attention_mask = pad_sequence(
            [*inputs["attention_mask_chosen"], *inputs["attention_mask_rejected"]], batch_first=True
        )

        self.last_inputs.update(input_ids=input_ids, attention_mask=attention_mask)

        with shared_prefix(model, **self.last_inputs, interleaved=True, enabled=self.use_shared_prefix):
            rewards = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )["logits"]

        rewards_chosen, rewards_rejected = rewards.chunk(2)

        # calculate loss, optionally modulate with margin
        if "margin" in inputs:
            loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected - inputs["margin"]).mean()
        else:
            loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()

        if self.args.center_rewards_coefficient is not None:
            loss += self.args.center_rewards_coefficient * torch.mean((rewards_chosen + rewards_rejected) ** 2)

        if return_outputs:
            return loss, {
                "rewards_chosen": rewards_chosen,
                "rewards_rejected": rewards_rejected,
            }
        return loss
