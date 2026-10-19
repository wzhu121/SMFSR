#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import argparse
import logging
import math
import os
import sys
sys.path.append(os.getcwd())
from pathlib import Path
from einops import rearrange
from PIL import Image
from contextlib import nullcontext

import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo
from tqdm.auto import tqdm
import torch.nn.functional as F
import diffusers
from models.transformer_sd3 import SD3Transformer2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import free_memory, cast_training_params
from diffusers.utils import check_min_version
from diffusers.utils.torch_utils import is_compiled_module
from models.transformer_sd3_interval import SD3Transformer2DModel_interval, LogitNormalCosineScheduler
from dataloaders.realsr_dataset import PairedCaptionDataset
from models.load_pipeline import load_smfsr_pipeline


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.30.0.dev0")

logger = get_logger(__name__)


def log_validation(args, reference_step, accelerator, weight_dtype, step, sd3_transformer=None, is_final_validation=False):
    logging.info("Running validation... ")

    if accelerator.is_main_process:
        save_dir = os.path.join(args.output_dir, f"validation_image_inference_{reference_step}", f"steps_{step}")
        os.makedirs(save_dir, exist_ok=True)

        pipeline = load_smfsr_pipeline(args, accelerator, sd3_transformer)

        pipeline.to(accelerator.device)
        pipeline.set_progress_bar_config(disable=True)

        if args.seed is None:
            generator = None
        else:
            generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
        
        img_names = sorted([f for f in os.listdir(args.validation_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        img_paths = [os.path.join(args.validation_images_dir, f) for f in img_names]
        txt_paths = [os.path.join(args.validation_txts_dir, os.path.splitext(os.path.basename(f))[0] + '.txt') for f in img_paths]

        assert len(img_paths) == len(txt_paths)

        if is_final_validation or torch.backends.mps.is_available():
            autocast_ctx = nullcontext()
        else:
            autocast_ctx = torch.autocast(accelerator.device.type, weight_dtype)

        for img_path, txt_path in zip(img_paths, txt_paths):
            image_name = os.path.splitext(os.path.basename(img_path))[0]

            validation_image = Image.open(img_path).convert("RGB")
            ori_width, ori_height = validation_image.size
            validation_image = validation_image.resize((int(4*ori_width), int(4*ori_height)),Image.BICUBIC)
            width, height = validation_image.size

            with open(txt_path, 'r', encoding='utf-8') as f:
                validation_prompt = f.read().strip()
                print(f"----------prompt:{validation_prompt}")

            with autocast_ctx:
                image = pipeline(
                            prompt=validation_prompt, control_image=validation_image, num_inference_steps=reference_step, generator=generator, height=height, width=width,
                            guidance_scale=1.0, negative_prompt=args.negative_prompt,
                            start_point="noise", latent_tiled_size=64, latent_tiled_overlap=24,
                            args=args,
                        ).images[0]
            image = image.resize((ori_width*4, ori_height*4))
            image.save(os.path.join(save_dir, f"{image_name}.png"))


        del pipeline
        free_memory()
    

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--transformer_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--transformer_model_name_or_path_student",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="controlnet-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="Save student weights and training state every X optimizer updates.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Resume from a checkpoint directory in --output_dir, an absolute path, or 'latest'.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="uniform_endclamp",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "uniform_endclamp"],
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--precondition_outputs",
        type=int,
        default=1,
        help="Flag indicating if we are preconditioning the model outputs or not as done in EDM. This affects how "
        "model `target` is calculated.",
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "swandb"],
        help="Log metrics with TensorBoard (default) or SwanLab.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default='NOTHING',
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing the target image."
    )
    parser.add_argument(
        "--conditioning_image_column",
        type=str,
        default="conditioning_image",
        help="The column of the dataset containing the controlnet conditioning image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=77,
        help="Maximum sequence length to use with with the T5 text encoder",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--tracker_project_name", "--wandb_project",
        type=str,
        default="split_meanflow_sr",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument("--negative_prompt", type=str, default='motion blur, noisy, dotted, bokeh, pointed, '
                            'CG Style, 3D render, unreal engine, blurring, dirty, messy, '
                            'worst quality, low quality, frames, watermark, signature, jpeg artifacts, '
                            'deformed, lowres, chaotic')
    
    parser.add_argument('--validation_images_dir', type=str, default="", help='validation image dir')
    parser.add_argument('--validation_txts_dir', type=str, default="", help='validation txt dir')
    parser.add_argument("--root_folders",  type=str , default='' )
    parser.add_argument("--null_text_ratio", type=float, default=0.0)
    parser.add_argument("--flow_ratio", type=float, default=0.6)
    parser.add_argument("--cfg_scale", type=float, default=4.5)
    parser.add_argument('--trainable_modules', nargs='*', type=str, default=["control", "time_text_embed"])
    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--train_data_dir`")

    if args.dataset_name is not None and args.train_data_dir is not None:
        raise ValueError("Specify only one of `--dataset_name` or `--train_data_dir`")

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    if args.validation_prompt is not None and args.validation_image is None:
        raise ValueError("`--validation_image` must be set if `--validation_prompt` is set")

    if args.validation_prompt is None and args.validation_image is not None:
        raise ValueError("`--validation_prompt` must be set if `--validation_image` is set")

    if (
        args.validation_image is not None
        and args.validation_prompt is not None
        and len(args.validation_image) != 1
        and len(args.validation_prompt) != 1
        and len(args.validation_image) != len(args.validation_prompt)
    ):
        raise ValueError(
            "Must provide either 1 `--validation_image`, 1 `--validation_prompt`,"
            " or the same number of `--validation_prompt`s and `--validation_image`s"
        )

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )

    return args

def main(args):

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )
    
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    if args.report_to == "swandb":
        from swanlab.integration.accelerate import SwanLabTracker
        tracker = SwanLabTracker(args.tracker_project_name)
    else:
        tracker = "tensorboard"
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=tracker,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load scheduler and models
    noise_scheduler = LogitNormalCosineScheduler(
                loc=0.0,
                scale=1.0,
                min_t=1e-8,
                max_t=1.0-1e-8
            )
    if args.transformer_model_name_or_path is not None:
        transformer_teacher = SD3Transformer2DModel.from_pretrained_local(
            args.transformer_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant
        )
        transformer_student = SD3Transformer2DModel_interval.from_pretrained_local(
            args.transformer_model_name_or_path_student, subfolder="transformer", revision=args.revision, variant=args.variant
        )

    else:
        transformer_teacher = SD3Transformer2DModel.from_pretrained_local(
            args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant
        )
        transformer_student = SD3Transformer2DModel_interval.from_pretrained_local(
            args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant
        )

    transformer_teacher.requires_grad_(False)
    transformer_student.requires_grad_(False)

    for name, params in transformer_student.named_parameters():
        if any(trainable_modules in name for trainable_modules in tuple(args.trainable_modules)):
            print(f'{name} in <transformer_student> will be optimized.' )
            params.requires_grad = True
    

    # Taken from [Sayak Paul's Diffusers PR #6511](https://github.com/huggingface/diffusers/pull/6511/files)
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = list(filter(lambda p: p.requires_grad, transformer_student.parameters()))
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    # Keep frozen teacher weights in the selected inference dtype.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move teacher and student to the training device.
    transformer_teacher.to(accelerator.device, dtype=weight_dtype)
    transformer_student.to(accelerator.device, dtype=weight_dtype)

    # Make sure the trainable params are in float32.
    if accelerator.mixed_precision in ("fp16", "bf16"):
        models = [transformer_student]
        cast_training_params(models, dtype=torch.float32)

    train_dataset = PairedCaptionDataset(root_folder=args.root_folders, 
                                         null_text_ratio=args.null_text_ratio,
)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    # Prepare everything with our `accelerator`.
    transformer_teacher, transformer_student, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer_teacher, transformer_student, optimizer, train_dataloader, lr_scheduler
    )

    def save_model_hook(models, weights, output_dir):
        weights.clear()  # The frozen teacher is not part of the checkpoint.
        if accelerator.is_main_process:
            unwrap_model(transformer_student).save_pretrained(os.path.join(output_dir, "transformer"))

    def load_model_hook(models, input_dir):
        loaded_student = SD3Transformer2DModel_interval.from_pretrained_local(input_dir, subfolder="transformer")
        student = unwrap_model(transformer_student)
        student.register_to_config(**loaded_student.config)
        student.load_state_dict(loaded_student.state_dict())
        del loaded_student
        models.clear()  # The teacher stays frozen; the student was loaded above.

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        # tensorboard cannot handle list types for config
        tracker_config.pop("validation_prompt")
        tracker_config.pop("validation_image")

        for k, v in list(tracker_config.items()):
            if not isinstance(v, (int, float, str, bool, torch.Tensor)):
                tracker_config[k] = str(v)

        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    resume_batch = 0

    if args.resume_from_checkpoint is not None:
        if args.resume_from_checkpoint == "latest":
            checkpoints = [
                name for name in os.listdir(args.output_dir)
                if name.startswith("checkpoint-") and name[11:].isdigit()
                and os.path.isfile(os.path.join(args.output_dir, name, "optimizer.bin"))
            ]
            if not checkpoints:
                raise FileNotFoundError(f"No resumable checkpoint found in {args.output_dir}")
            checkpoint_dir = os.path.join(args.output_dir, max(checkpoints, key=lambda name: int(name[11:])))
        else:
            checkpoint_dir = args.resume_from_checkpoint
            if not os.path.isabs(checkpoint_dir):
                checkpoint_dir = os.path.join(args.output_dir, checkpoint_dir)

        checkpoint_name = os.path.basename(os.path.normpath(checkpoint_dir))
        if not checkpoint_name.startswith("checkpoint-") or not checkpoint_name[11:].isdigit():
            raise ValueError(f"Invalid checkpoint directory name: {checkpoint_dir}")
        if not os.path.isfile(os.path.join(checkpoint_dir, "optimizer.bin")):
            raise FileNotFoundError(f"{checkpoint_dir} contains student weights but no optimizer state")
        global_step = int(checkpoint_name[11:])
        if global_step >= args.max_train_steps:
            raise ValueError("Checkpoint has already reached --max_train_steps")
        first_epoch = global_step // num_update_steps_per_epoch
        resume_batch = global_step % num_update_steps_per_epoch * args.gradient_accumulation_steps
        accelerator.print(f"Resuming from {checkpoint_dir} at step {global_step}")
        accelerator.load_state(checkpoint_dir)

    progress_bar = tqdm(
        range(args.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    free_memory()

    for epoch in range(first_epoch, args.num_train_epochs):
        if epoch == first_epoch and resume_batch:
            epoch_dataloader = accelerator.skip_first_batches(train_dataloader, resume_batch)
        else:
            epoch_dataloader = train_dataloader
        for step, batch in enumerate(epoch_dataloader, start=resume_batch if epoch == first_epoch else 0):
            log_dict = {}
            with accelerator.accumulate(transformer_student):
                model_input = batch["pixel_values"].to(dtype=weight_dtype)
                controlnet_image = batch["conditioning_pixel_values"].to(dtype=weight_dtype)
                prompt_embeds = batch["prompt_embeds"].to(dtype=model_input.dtype)
                pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(dtype=model_input.dtype)
                # null prompt used for cfg
                null_prompt_embeds = batch["null_prompt_embeds"].to(dtype=model_input.dtype)
                null_pooled_prompt_embeds = batch["null_pooled_prompt_embeds"].to(dtype=model_input.dtype)

                rand_val = torch.rand(1, device=accelerator.device)
                
                if rand_val < args.flow_ratio:
            
                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(model_input)
                    bsz = model_input.shape[0]
                    # Sample a random timestep for each image
                    t = noise_scheduler.sample_timesteps(bsz, model_input.device)
                    sigma_t = t
                    alpha_t = 1 - (1 - 1e-6) * t
                    alpha_t = alpha_t.view(bsz, 1, 1, 1)
                    sigma_t = sigma_t.view(bsz, 1, 1, 1)
                    
                    noisy_model_input =  alpha_t * model_input + sigma_t * noise
               
                    # Predict the noise residual
                    model_pred = transformer_student(
                        hidden_states=noisy_model_input,
                        controlnet_image=controlnet_image,
                        r_timestep=t,
                        timestep=t,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        return_dict=False,
                    )[0]

                    if args.cfg_scale > 1.0:
                        noisy_model_input = torch.cat([noisy_model_input]*2)
                        controlnet_image = torch.cat([controlnet_image]*2)
                        t = torch.cat([t]*2)
                        prompt_embeds = torch.cat([null_prompt_embeds,prompt_embeds],dim=0)
                        pooled_prompt_embeds = torch.cat([null_pooled_prompt_embeds,pooled_prompt_embeds],dim=0)

                        target_pred_ = transformer_teacher(
                            hidden_states=noisy_model_input,
                            controlnet_image=controlnet_image,
                            timestep=t,
                            encoder_hidden_states=prompt_embeds,
                            pooled_projections=pooled_prompt_embeds,
                            return_dict=False,
                        )[0]
                        target_pred_uncond, target_pred_cond = target_pred_.chunk(2)
                        target_pred = target_pred_uncond + args.cfg_scale * (target_pred_cond - target_pred_uncond)
                    else:
                        target_pred = transformer_teacher(
                            hidden_states=noisy_model_input,
                            controlnet_image=controlnet_image,
                            timestep=t,
                            encoder_hidden_states=prompt_embeds,
                            pooled_projections=pooled_prompt_embeds,
                            return_dict=False,
                        )[0]

                    loss = F.mse_loss(model_pred, target_pred)

                    log_dict["loss/bound"] = loss.detach().item()

                elif rand_val < args.flow_ratio + 0.1:

                    noise = torch.randn_like(model_input)
                    bsz = model_input.shape[0]
                    sub_u = torch.rand(1, device=accelerator.device)

                    if sub_u < 0.33:
                        # Full interval [0, 1]
                        r = torch.zeros(bsz, device=model_input.device)
                        t = torch.ones(bsz, device=model_input.device)
                    elif sub_u < 0.66:
                        # First half [0, 0.5]
                        r = torch.zeros(bsz, device=model_input.device)
                        t = torch.full((bsz,), 0.5, device=model_input.device)
                    else:
                        # Second half [0.5, 1]
                        r = torch.full((bsz,), 0.5, device=model_input.device)
                        t = torch.ones(bsz, device=model_input.device)
                        
                    lam = torch.rand(bsz, device=model_input.device)
                    s = (1 - lam) * t + lam * r

                    sigma_t = t 
                    alpha_t = 1 - (1 - 1e-6) * t
                    alpha_t = alpha_t.view(bsz, 1, 1, 1)
                    sigma_t = sigma_t.view(bsz, 1, 1, 1)

                    noisy_model_input = alpha_t * model_input + sigma_t * noise
                    
                    with torch.no_grad():
                    # Forward pass 1: u(z_t, s, t)
                        u2 = transformer_student(
                            hidden_states=noisy_model_input,
                            controlnet_image=controlnet_image,
                            r_timestep=s,
                            timestep=t,
                            encoder_hidden_states=prompt_embeds,
                            pooled_projections=pooled_prompt_embeds,
                            return_dict=False,
                        )[0]

                    
                    # Compute z_s using the predicted velocity
                    t_s_diff = rearrange(torch.clamp(t - s, min=1e-5), "b -> b 1 1 1")
                    z_s = noisy_model_input - t_s_diff * u2

                    with torch.no_grad():
                    # Forward pass 2: u(z_s, r, s)
                        u1 = transformer_student(
                            hidden_states=z_s,
                            controlnet_image=controlnet_image,
                            r_timestep=r,
                            timestep=s,
                            encoder_hidden_states=prompt_embeds,
                            pooled_projections=pooled_prompt_embeds,
                            return_dict=False,
                        )[0]

                    # Construct target using interval splitting consistency
                    lam_ = rearrange(lam, "b -> b 1 1 1")
                    target = (1 - lam_) * u1 + lam_ * u2

                    # Forward pass 3: predict u(z_t, r, t)
                    u_pred = transformer_student(
                        hidden_states=noisy_model_input,
                        controlnet_image=controlnet_image,
                        r_timestep=r,
                        timestep=t,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        return_dict=False,
                    )[0]
                        
                    loss = F.mse_loss(u_pred, target.detach())

                    log_dict["loss/consistency_half"] = loss.detach().item()
                else:
                    # Sample r < s < t
                    noise = torch.randn_like(model_input)
                    bsz = model_input.shape[0]
                    # Sample a random timestep for each image

                    r_sample = noise_scheduler.sample_timesteps(bsz, model_input.device)
                    t_sample = noise_scheduler.sample_timesteps(bsz, model_input.device)
                    # ensure r < t
                    r = torch.minimum(r_sample, t_sample)
                    t = torch.maximum(r_sample, t_sample)

                    lam = torch.rand(bsz, device=model_input.device)
                    s = (1 - lam) * t + lam * r

                    sigma_t = t 
                    alpha_t = 1 - (1 - 1e-6) * t
                    alpha_t = alpha_t.view(bsz, 1, 1, 1)
                    sigma_t = sigma_t.view(bsz, 1, 1, 1)

                    noisy_model_input =  alpha_t * model_input + sigma_t * noise
                    
                    with torch.no_grad():
                    # Forward pass 1: u(z_t, s, t)
                        u2 = transformer_student(
                            hidden_states=noisy_model_input,
                            controlnet_image=controlnet_image,
                            r_timestep=s,
                            timestep=t,
                            encoder_hidden_states=prompt_embeds,
                            pooled_projections=pooled_prompt_embeds,
                            return_dict=False,
                        )[0]

                    
                    # Compute z_s using the predicted velocity
                    t_s_diff = rearrange(torch.clamp(t - s, min=1e-5), "b -> b 1 1 1")
                    z_s = noisy_model_input - t_s_diff * u2

                    with torch.no_grad():
                    # Forward pass 2: u(z_s, r, s)
                        u1 = transformer_student(
                            hidden_states=z_s,
                            controlnet_image=controlnet_image,
                            r_timestep=r,
                            timestep=s,
                            encoder_hidden_states=prompt_embeds,
                            pooled_projections=pooled_prompt_embeds,
                            return_dict=False,
                        )[0]

                    # Construct target using interval splitting consistency
                    lam_ = rearrange(lam, "b -> b 1 1 1")
                    target = (1 - lam_) * u1 + lam_ * u2

                    # Forward pass 3: predict u(z_t, r, t)
                    u_pred = transformer_student(
                        hidden_states=noisy_model_input,
                        controlnet_image=controlnet_image,
                        r_timestep=r,
                        timestep=t,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        return_dict=False,
                    )[0]
                        
                    loss = F.mse_loss(u_pred, target.detach())

                    log_dict["loss/consistency_random"] = loss.detach().item()
                loss_total = loss
                accelerator.backward(loss_total)
                if accelerator.sync_gradients:
                    params_to_clip = transformer_student.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpointing_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    checkpoint_exists = os.path.exists(save_path)
                    accelerator.wait_for_everyone()
                    if checkpoint_exists:
                        raise FileExistsError(f"Checkpoint already exists: {save_path}")
                    accelerator.save_state(save_path)
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        logger.info(f"Saved student weights and training state to {save_path}")

                if accelerator.is_main_process:
                    if args.validation_images_dir is not None and global_step % args.validation_steps == 0:
                        sd3_transformer = unwrap_model(transformer_student)
                        log_validation(
                            args, 1,
                            accelerator,
                            weight_dtype,
                            global_step,
                            sd3_transformer
                        )

            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            logs.update(log_dict)
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
