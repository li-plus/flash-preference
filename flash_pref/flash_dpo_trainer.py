from importlib.util import find_spec

import torch
import torch.nn as nn

from flash_pref.shared_prefix import shared_prefix

is_trl_available = find_spec("trl") is not None

if is_trl_available:
    from trl import DPOTrainer
else:
    DPOTrainer = object


class FlashDPOTrainer(DPOTrainer):
    r"""A drop-in replacement for trl.DPOTrainer with prefix sharing acceleration."""

    def __init__(self, *args, use_shared_prefix: bool = True, **kwargs):
        if not is_trl_available:
            raise RuntimeError(f"{self.__class__.__name__} requires trl. Run `pip install trl` to install it.")

        super().__init__(*args, **kwargs)

        self.use_shared_prefix = use_shared_prefix

        if not self.use_shared_prefix:
            return

        assert not self.is_encoder_decoder, f"{self.__class__.__name__} does not support encoder-decoder models"
        assert not self.use_apex, f"{self.__class__.__name__} does not support use_apex"
        assert not self.padding_free, f"{self.__class__.__name__} does not support padding_free"

        self.last_inputs = {}

        def patch_model_forward(model: nn.Module):
            def model_forward_wrapper(
                model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor, *args, **kwargs
            ):
                self.last_inputs.update(input_ids=input_ids, attention_mask=attention_mask)
                with shared_prefix(model, **self.last_inputs, interleaved=True, enabled=self.use_shared_prefix):
                    output = old_model_forward(input_ids=input_ids, attention_mask=attention_mask, *args, **kwargs)
                return output

            old_model_forward = model.forward
            model.forward = model_forward_wrapper.__get__(model)

        patch_model_forward(self.model)

        if self.ref_model is not None:
            patch_model_forward(self.ref_model)

        def patch_accelerator_backward(accelerator):
            def accelerator_backward_wrapper(accelerator, *args, **kwargs):
                with shared_prefix(self.model, **self.last_inputs, interleaved=True, enabled=self.use_shared_prefix):
                    return old_accelerator_backward(*args, **kwargs)

            old_accelerator_backward = accelerator.backward
            accelerator.backward = accelerator_backward_wrapper.__get__(accelerator)

        patch_accelerator_backward(self.accelerator)
