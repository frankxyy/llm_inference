import torch
import torch.distributed
from transformers import AutoConfig
from transformers.models.llama import LlamaTokenizer
from typing import Optional, List
from pathlib import Path

from flash_casual_lm import FlashCausalLM

from utils import (
    initialize_torch_distributed,
    weight_files,
    download_weights,
    weight_hub_files,
    LocalEntryNotFoundError,
)

from accelerate import init_empty_weights

class FlashLlama(FlashCausalLM):
    def __init__(self, model_id: str, revision: Optional[str] = None, quantize=False):
        self.past_pad = None
        if torch.cuda.is_available():
            device = torch.device("cuda")
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            raise NotImplementedError("FlashLlama is only available on GPU")

        tokenizer = LlamaTokenizer.from_pretrained(
            model_id,
            revision=revision,
            padding_side="left",
            truncation_side="left",
        )

        config = AutoConfig.from_pretrained(
            model_id,
            revision=revision,
        )

        # We do not use from_pretrained as we modified the model internal module layout
        try:
            filenames = weight_files(model_id, revision, ".bin")
        # Local files not found
        except LocalEntryNotFoundError:
            hub_files = weight_hub_files(model_id, revision, ".bin")
            filenames = download_weights(hub_files, model_id, revision)

        with init_empty_weights():
            model = FlashLlamaForCausalLM(config)

        self.load_weights(model, filenames, quantize, device, dtype)
        self.model = model.eval().to(device)

        super(FlashCausalLM, self).__init__(
            tokenizer=tokenizer,
            requires_padding=False,
            dtype=dtype,
            device=device,
        )

    @staticmethod
    def load_weights(
        model,
        filenames: List[Path],
        quantize: bool,
        device: torch.device,
        dtype: torch.dtype,
    ):
        for filename in filenames:
            state_dict = torch.load(filename, map_location="cpu")
            for key, value in state_dict.items():
                value = value.to(device if not quantize else "cpu").to(dtype)

                layer_name = ".".join(key.split(".")[:4])

                # Fused qkv
                if "q_proj" in key or "k_proj" in key or "v_proj" in key:
                    final_key = layer_name + ".query_key_value.weight"

                # Fused gate and up projs
                elif "gate_proj" in key or "up_proj" in key:
                    final_key = layer_name + ".gate_up_proj.weight"
                else:
                    final_key = key

                module_name, param_name = final_key.rsplit(".", 1)
                module = model.get_submodule(module_name)

                try:
                    current_parameter_tensor = module._parameters[param_name]
                except KeyError:
                    current_parameter_tensor = None

                if current_parameter_tensor is not None:
                    if current_parameter_tensor.device == torch.device("meta"):
                        # Init qkv
                        if "query_key_value" in final_key:
                            module._parameters[param_name] = value.new_empty(
                                (value.shape[0] * 3, value.shape[1])
                            )
                        # Init gate and up proj
                        elif "gate_up_proj" in final_key:
                            module._parameters[param_name] = value.new_empty(
                                (value.shape[0] * 2, value.shape[1])
                            )

                    # Copy to correct slice
                    if "q_proj" in key:
                        module._parameters[param_name][: value.shape[0]] = value
                    elif "k_proj" in key:
                        module._parameters[param_name][
                            value.shape[0] : value.shape[0] * 2
                        ] = value
                    elif "v_proj" in key:
                        module._parameters[param_name][value.shape[0] * 2 :] = value
                    elif "gate_proj" in key:
                        module._parameters[param_name][: value.shape[0]] = value
                    elif "up_proj" in key:
                        module._parameters[param_name][value.shape[0] :] = value
                    else:
                        if current_parameter_tensor.shape != value.shape:
                            raise ValueError(
                                f"Name {final_key} -- Current {current_parameter_tensor.shape} and got {value.shape}"
                            )
                        module._parameters[param_name] = value
                else:
                    module._buffers[param_name] = value

                del value

        uninitialized_parameters = []
        for n, p in model.named_parameters():
            if p.data.device == torch.device("meta"):
                uninitialized_parameters.append(n)
        if uninitialized_parameters:
            raise RuntimeError(
                f"found uninitialized parameters in model: {uninitialized_parameters}"
            )

        torch.cuda.empty_cache()
        model.post_load_weights(quantize)