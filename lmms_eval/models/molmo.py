# molmo.py
from typing import List, Optional, Tuple, Union
import torch
import os
from tqdm import tqdm

from lmms_eval.api.model import lmms
from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from transformers import (
    AutoProcessor,      # or AutoTokenizer
    AutoModelForCausalLM,  # or your specific model class, e.g. MolmoForConditionalGeneration if it exists,
    GenerationConfig
)

# If your framework uses accelerator:
from accelerate import Accelerator, DistributedType
from accelerate.state import AcceleratorState
from lmms_eval.tasks.mmmu.utils_group_img import process_images

from loguru import logger as eval_logger
from lmms_eval.utils import Collator
from lmms_eval import utils

try:
    import flash_attn
    best_fit_attn_implementation = "flash_attention_2"
except ImportError:
    best_fit_attn_implementation = "eager"


@register_model("molmo")
class Molmo(lmms):
    """
    Molmo Model for Hugging Face Transformers:

    Example usage:
accelerate launch --num_processes=8 -m lmms_eval \
    --model molmo \
    --model_args pretrained=allenai/Molmo-7B-D-0924 \
    --tasks mme \
    --batch_size 1 \
    --output_path ./logs/ \
    --log_samples
    """

    def __init__(
        self,
        pretrained: str = "allenai/Molmo-7B-D-0924",
        device: Optional[str] = "cuda",
        batch_size: Optional[Union[int, str]] = 1,
        trust_remote_code: Optional[bool] = True,
        use_cache=True,
        **kwargs,
    ) -> None:
        super().__init__()
        # (1) Accelerator logic (if you're using multi-GPU, FSDP, etc.)
        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
        else:
            self._device = device

        # (2) Load your Molmo model + processor (or tokenizer) from Hugging Face
        #    Replace AutoModelForCausalLM with your actual model class if needed.

        # load the processor
        self._processor = AutoProcessor.from_pretrained(
            pretrained,
            trust_remote_code=True,
            device_map=self._device,
            use_fast=False,
        )
        
        # load the model
        self._model = AutoModelForCausalLM.from_pretrained(
            pretrained,
            trust_remote_code=True,
            device_map=self._device,
        ).eval()


        self._tokenizer = self._processor.tokenizer if hasattr(self._processor, "tokenizer") else self._processor
        
        # (3) Additional configuration
        self._config = self._model.config
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")
            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.model.to(self._device)
            self._rank = 0
            self._word_size = 1


        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")
            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.model.to(self._device)
            self._rank = 0
            self._word_size = 1
    # (4) Required properties and methods:
    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # Unwrap if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        return self._model

    @property
    def eot_token_id(self):
        # Usually eos_token_id or eot_token_id
        return self.tokenizer.eos_token_id

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=False) -> List[int]:
        tokens = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        if left_truncate_len is not None:
            tokens = tokens[-left_truncate_len:]
        return tokens

    def tok_decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """Generate text responses for a batch of requests."""
        res = []
        
        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            visuals = self.flatten(visuals)
            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]

            # Set default values for until and max_new_tokens
            until = [self.tok_decode(self.eot_token_id)]

            # Update values from gen_kwargs if present
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")
            assert self.batch_size_per_gpu == 1, "Do not support batch_size_per_gpu > 1 for now"
            context = contexts[0]
            if "<image>" in context:
                # instruct blip does not expect the <image> tag
                context = context.replace("<image>", "")
            # Set trunction equals true here, the max length for qformer tokenizer is 512
            # if not truncate, some questions will cause size mismatch
            # The transformer implementation can't handle multi images for blip
            # Concat it into one image
            if len(visuals) > 1:
                visuals = [process_images(visuals)]
            
            # Process inputs but don't move to device yet
            inputs = self._processor.process(images=visuals, text=context, return_tensors="pt", truncation=True)

            # move inputs to the correct device and make a batch of size 1
            inputs = {k: v.to(self._device).unsqueeze(0) for k, v in inputs.items()}        
            
            
            gen_kwargs["image_sizes"] = [visuals[idx].size for idx in range(len(visuals))]
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "min_new_tokens" not in gen_kwargs:
                gen_kwargs["min_new_tokens"] = 1
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1
            
            try:
                # Create GenerationConfig with all parameters from gen_kwargs
                generation_config = GenerationConfig(
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                    temperature=gen_kwargs["temperature"],
                    top_p=gen_kwargs["top_p"] if gen_kwargs["top_p"] is not None else 1.0,
                    num_beams=gen_kwargs["num_beams"],
                    do_sample=True if gen_kwargs["temperature"] > 0 else False,
                    stop_strings=["<|endoftext|>"]
                )
                cont = self.model.generate_from_batch(
                    inputs,
                    generation_config,
                    tokenizer=self._processor.tokenizer
                )
            # Error 'super' object has no attribute '_extract_past_from_model_output
            except Exception as e:
                eval_logger.error(f"Error {e} in generating")
                # print("Error in generating", e)
                cont = ""
            # print("gen_kwargs['min_new_tokens']", gen_kwargs["min_new_tokens"])
            # print("inputs['input_ids'].size(1):", inputs['input_ids'].size(1))
            # print("cont", cont)
            generated_tokens = cont[:, inputs['input_ids'].size(1):]
            # generated_text = self._processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            # text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()

            # This is what the model generated only, while the lmms-eval expects the whole output in the document
            # The generated input+output text from the model will then be returned.
            text_outputs = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0].strip()
            
            res.append(text_outputs)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text_outputs)
            pbar.update(1)
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res

    def flatten(self, input_list):
        """Flatten a nested list."""
        new_list = []
        for i in input_list:
            for j in i:
                new_list.append(j)
        return new_list

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("If your eval needs loglikelihood, implement it here.")

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for Molmo")
    

if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
    from PIL import Image
    import requests

    # load the processor
    processor = AutoProcessor.from_pretrained(
        'allenai/Molmo-7B-D-0924',
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto'
    )

    # load the model
    model = AutoModelForCausalLM.from_pretrained(
        'allenai/Molmo-7B-D-0924',
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto'
    )

    # process the image and text
    inputs = processor.process(
        images=[Image.open(requests.get("https://picsum.photos/id/237/536/354", stream=True).raw)],
        text="Describe this image."
    )
    images=[Image.open(requests.get("https://picsum.photos/id/237/536/354", stream=True).raw)]

    # move inputs to the correct device and make a batch of size 1
    processed_inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

    # generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
    output = model.generate_from_batch(
        processed_inputs,
        GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
        tokenizer=processor.tokenizer
    )

    # only get generated tokens; decode them to text
    generated_tokens = output[0,processed_inputs['input_ids'].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # print the generated text
    print(generated_text)

    # >>>  This image features an adorable black Labrador puppy, captured from a top-down
    #      perspective. The puppy is sitting on a wooden deck, which is composed ...
