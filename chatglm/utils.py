import os
from typing import Dict, Tuple, Union, Optional

import torch
from torch.nn import Module
from transformers import AutoModel
from accelerate import load_checkpoint_and_dispatch

import sys
sys.path.append("../")
from llm_utils.modeling import load_checkpoint_in_model


def auto_configure_device_map(num_gpus: int) -> Dict[str, int]:
    # transformer.word_embeddings 占用1层
    # transformer.final_layernorm 和 lm_head 占用1层
    # transformer.layers 占用 28 层
    # 总共30层分配到num_gpus张卡上
    num_trans_layers = 28
    per_gpu_layers = 30 / num_gpus

    # bugfix: 在linux中调用torch.embedding传入的weight,input不在同一device上,导致RuntimeError
    # windows下 model.device 会被设置成 transformer.word_embeddings.device
    # linux下 model.device 会被设置成 lm_head.device
    # 在调用chat或者stream_chat时,input_ids会被放到model.device上
    # 如果transformer.word_embeddings.device和model.device不同,则会导致RuntimeError
    # 因此这里将transformer.word_embeddings,transformer.final_layernorm,lm_head都放到第一张卡上
    device_map = {'transformer.word_embeddings': 0,
                  'transformer.final_layernorm': 0, 'lm_head': 0}

    used = 2
    gpu_target = 0
    for i in range(num_trans_layers):
        if used >= per_gpu_layers:
            gpu_target += 1
            used = 0
        assert gpu_target < num_gpus
        device_map[f'transformer.layers.{i}'] = gpu_target
        used += 1

    return device_map

def auto_configure_device_map_llama(num_gpus: int) -> Dict[str, int]:
    # embed_tokens 占用1层
    # LlamaRMSNorm 和 lm_head 占用1层
    # LlamaDecoderLayer 占用 32 层
    # 总共34层分配到num_gpus张卡上
    num_trans_layers = 32
    per_gpu_layers = 34 / num_gpus

    # bugfix: 在linux中调用torch.embedding传入的weight,input不在同一device上,导致RuntimeError
    # windows下 model.device 会被设置成 transformer.word_embeddings.device
    # linux下 model.device 会被设置成 lm_head.device
    # 在调用chat或者stream_chat时,input_ids会被放到model.device上
    # 如果transformer.word_embeddings.device和model.device不同,则会导致RuntimeError
    # 因此这里将transformer.word_embeddings,transformer.final_layernorm,lm_head都放到第一张卡上
    device_map = {'embed_tokens': 0,
                  'norm': 0}

    used = 2
    gpu_target = 0
    for i in range(num_trans_layers):
        if used >= per_gpu_layers:
            gpu_target += 1
            used = 0
        assert gpu_target < num_gpus
        device_map[f'layers.{i}'] = gpu_target
        used += 1

    return device_map


def load_model_on_gpus(model_name, model, num_gpus: int = 2,
                       device_map: Optional[Dict[str, int]] = None, **kwargs) -> Module:
    from accelerate import dispatch_model
    if device_map == 'auto':
        from accelerate import infer_auto_device_map
        device_map = infer_auto_device_map(model, max_memory={0: "10GiB", 1: "10GiB", "cpu": "30GiB"},
                                            dtype=torch.float16)
        
    if device_map is None:
        if model_name == 'THUDM/chatglm-6b':
            device_map = auto_configure_device_map(num_gpus)
        elif model_name == 'decapoda-research/llama-7b-hf':
            device_map = auto_configure_device_map_llama(num_gpus)
        else:
            raise Exception('no device mapping func available for model')

    print("device_map = {}".format(device_map))
    # model_dispatched = dispatch_model(model, device_map=device_map)
    from huggingface_hub import snapshot_download
    repo_root = snapshot_download(model_name,
                                    allow_patterns=["*"],
                                    cache_dir=os.getenv("TRANSFORMERS_CACHE", None),
                                    ignore_patterns=["*.safetensors"],
                                    local_files_only=False,
                                    revision=None)
    if model_name == 'decapoda-research/llama-7b-hf':
        full_model_device_map = {f"model.{k}": v for k, v in device_map.items()}
        full_model_device_map["lm_head"] = 0
    else:
        full_model_device_map = device_map
    load_checkpoint_in_model(
        model, 
        repo_root, 
        device_map=full_model_device_map, 
        offload_folder=None, 
        dtype='float16', 
        offload_state_dict=True
    )
    model.tie_weights()
    
    dispatch_model = dispatch_model(model, device_map=full_model_device_map)

    return dispatch_model

