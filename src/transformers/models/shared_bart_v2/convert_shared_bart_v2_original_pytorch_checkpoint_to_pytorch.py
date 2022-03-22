# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert SHARED_BART_V2 checkpoint."""


import argparse
import os
from pathlib import Path

import fairseq
import torch
from packaging import version
from torch import nn

from transformers import (
    BartSharedV2Config,
    BartSharedV2ForConditionalGeneration,
    BartSharedV2ForSequenceClassification,
    BartSharedV2Model,
    BartTokenizer,
)
from transformers.utils import logging


FAIRSEQ_MODELS = ["shared_bart_v2.large", "shared_bart_v2.large.mnli", "shared_bart_v2.large.cnn", "shared_bart_v2_xsum/model.pt"]
extra_arch = {"shared_bart_v2.large": BartSharedV2Model, "shared_bart_v2.large.mnli": BartSharedV2ForSequenceClassification}
if version.parse(fairseq.__version__) < version.parse("0.9.0"):
    raise Exception("requires fairseq >= 0.9.0")


logging.set_verbosity_info()
logger = logging.get_logger(__name__)

SAMPLE_TEXT = " Hello world! cécé herlolip"

mnli_rename_keys = [
    ("model.classification_heads.mnli.dense.weight", "classification_head.dense.weight"),
    ("model.classification_heads.mnli.dense.bias", "classification_head.dense.bias"),
    ("model.classification_heads.mnli.out_proj.weight", "classification_head.out_proj.weight"),
    ("model.classification_heads.mnli.out_proj.bias", "classification_head.out_proj.bias"),
]


def remove_ignore_keys_(state_dict):
    ignore_keys = [
        "encoder.version",
        "decoder.version",
        "model.encoder.version",
        "model.decoder.version",
        "_float_tensor",
    ]
    for k in ignore_keys:
        state_dict.pop(k, None)


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


def load_xsum_checkpoint(checkpoint_path):
    """Checkpoint path should end in model.pt"""
    sd = torch.load(checkpoint_path, map_location="cpu")
    hub_interface = torch.hub.load("pytorch/fairseq", "shared_bart_v2.large.cnn").eval()
    hub_interface.model.load_state_dict(sd["model"])
    return hub_interface


def make_linear_from_emb(emb):
    vocab_size, emb_size = emb.weight.shape
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
    lin_layer.weight.data = emb.weight.data
    return lin_layer


@torch.no_grad()
def convert_shared_bart_v2_checkpoint(checkpoint_path, pytorch_dump_folder_path, hf_checkpoint_name=None):
    """
    Copy/paste/tweak model's weights to our BERT structure.
    """
    if not os.path.exists(checkpoint_path):
        shared_bart_v2 = torch.hub.load("pytorch/fairseq", checkpoint_path).eval()
    else:
        shared_bart_v2 = load_xsum_checkpoint(checkpoint_path)

    shared_bart_v2.model.upgrade_state_dict(shared_bart_v2.model.state_dict())
    if hf_checkpoint_name is None:
        hf_checkpoint_name = checkpoint_path.replace(".", "-")
    config = BartSharedV2Config.from_pretrained(hf_checkpoint_name)
    tokens = shared_bart_v2.encode(SAMPLE_TEXT).unsqueeze(0)
    tokens2 = BartTokenizer.from_pretrained(hf_checkpoint_name).encode(SAMPLE_TEXT, return_tensors="pt").unsqueeze(0)
    assert torch.eq(tokens, tokens2).all()

    if checkpoint_path == "shared_bart_v2.large.mnli":
        state_dict = shared_bart_v2.state_dict()
        remove_ignore_keys_(state_dict)
        state_dict["model.shared.weight"] = state_dict["model.decoder.embed_tokens.weight"]
        for src, dest in mnli_rename_keys:
            rename_key(state_dict, src, dest)
        model = BartSharedV2ForSequenceClassification(config).eval()
        model.load_state_dict(state_dict)
        fairseq_output = shared_bart_v2.predict("mnli", tokens, return_logits=True)
        new_model_outputs = model(tokens)[0]  # logits
    else:  # no classification heads to worry about
        state_dict = shared_bart_v2.model.state_dict()
        remove_ignore_keys_(state_dict)
        state_dict["shared.weight"] = state_dict["decoder.embed_tokens.weight"]
        fairseq_output = shared_bart_v2.extract_features(tokens)
        if hf_checkpoint_name == "facebook/shared_bart_v2-large":
            model = BartSharedV2Model(config).eval()
            model.load_state_dict(state_dict)
            new_model_outputs = model(tokens).model[0]
        else:
            model = BartSharedV2ForConditionalGeneration(config).eval()  # an existing summarization ckpt
            model.model.load_state_dict(state_dict)
            if hasattr(model, "lm_head"):
                model.lm_head = make_linear_from_emb(model.model.shared)
            new_model_outputs = model.model(tokens)[0]

    # Check results
    assert fairseq_output.shape == new_model_outputs.shape
    assert (fairseq_output == new_model_outputs).all().item()
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    model.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "fairseq_path", type=str, help="shared_bart_v2.large, shared_bart_v2.large.cnn or a path to a model.pt on local filesystem."
    )
    parser.add_argument("pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument(
        "--hf_config", default=None, type=str, help="Which huggingface architecture to use: shared_bart_v2-large-xsum"
    )
    args = parser.parse_args()
    convert_shared_bart_v2_checkpoint(args.fairseq_path, args.pytorch_dump_folder_path, hf_checkpoint_name=args.hf_config)
