from dataclasses import dataclass, field
import os
import torch
from accelerate import logging
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
from trl import (
    ModelConfig,
    SFTConfig,
    SFTTrainer,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
logger = logging.get_logger(__name__)



@dataclass
class SFTScriptArguments(ScriptArguments):
    """
    Args:
        dataset_path: Path to local JSONL file. Each line must be a JSON object:
            {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
        enable_thinking: Controls Qwen3 thinking mode via apply_chat_template.
            False (default): no <think> blocks, suitable for standard SFT.
            True: include <think> blocks, use if training on CoT reasoning data.
        merge_adapter: If True, merge LoRA weights into the base model after training
            and save to {output_dir}-merged. Load this with vLLM for evaluation.
    """
    dataset_path: str = field(
        default=None,                          # ← add default
        metadata={"help": "Path to local JSONL file with 'messages' column."}
    )
    enable_thinking: bool = field(
        default=False,
        metadata={"help": "Pass enable_thinking to Qwen3's apply_chat_template."},
    )
    merge_adapter: bool = field(
        default=True,
        metadata={"help": "Merge LoRA adapter into base model after training for vLLM inference."},
    )
    
    
    
def main(script_args: SFTScriptArguments, training_args: SFTConfig, model_args: ModelConfig):
    if script_args.dataset_path is None:
        raise ValueError("dataset_path must be set in config yaml.")
    
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        revision=model_args.model_revision,
    )
    
    dataset = load_dataset("json", data_files={"train": script_args.dataset_path})
    train_dataset = dataset["train"]
    logger.info(f"Loaded {len(train_dataset)} training examples from {script_args.dataset_path}")
    logger.info(f"Sample: {train_dataset[0]}")
    
    def format_with_chat_template(example):
        messages = example["messages"]

        # Full formatted text: both turns, naturally includes closing <|im_end|>
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=script_args.enable_thinking,
        )

        # Prompt only: user turn + open assistant tag
        prompt = tokenizer.apply_chat_template(
            messages[:-1],       # everything except the last (assistant) turn
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=script_args.enable_thinking,
        )

        # Completion: everything after the prompt — assistant text + <|im_end|>\n
        completion = full_text[len(prompt):]
        completion = completion.rstrip("\n")

        return {"prompt": prompt, "completion": completion}
    
    train_dataset = train_dataset.map(
        format_with_chat_template,
        remove_columns=train_dataset.column_names,
        desc="Applying chat template",
    )
    
    
    # dtype = (
    #     model_args.torch_dtype
    #     if model_args.torch_dtype in ["auto", None]
    #     else getattr(torch, model_args.torch_dtype)
    # )
    #print(vars(model_args))
    dtype = (
        model_args.dtype
        if model_args.dtype in ["auto", None]
        else getattr(torch, model_args.dtype)
    )
    
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = get_kbit_device_map()
    else:
        model_kwargs["device_map"] = "auto"

    training_args.model_init_kwargs = model_kwargs
    
    peft_config = get_peft_config(model_args)
    logger.info(f"PEFT config: {peft_config}")
    
    trainer = SFTTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        peft_config=peft_config,
        processing_class=tokenizer,
        # No dataset_text_field — SFTTrainer detects prompt+completion columns
        # and masks prompt tokens automatically
    )
    
    if trainer.accelerator.is_main_process:
        print(f"Formatted prompt:\n{train_dataset[0]['prompt']}")
        print(f"Formatted completion:\n{train_dataset[0]['completion']}")
    
    
    
    trainer.train()
    trainer.accelerator.print("✅ Training complete.")
    
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    trainer.accelerator.print(f"💾 LoRA adapter saved to {training_args.output_dir}")

    
        
    if training_args.push_to_hub:
        trainer.push_to_hub()
        
        
    if script_args.merge_adapter and trainer.accelerator.is_main_process:
        merged_path = training_args.output_dir.rstrip("/") + "-merged"
        trainer.accelerator.print(f"🔀 Merging LoRA adapter into base model → {merged_path}")

        base_model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="cpu",
        )

        merged_model = PeftModel.from_pretrained(base_model, training_args.output_dir)
        merged_model = merged_model.merge_and_unload()

        merged_model.save_pretrained(merged_path, safe_serialization=True)
        tokenizer.save_pretrained(merged_path)

        trainer.accelerator.print(f"✅ Merged model saved to {merged_path}")
        trainer.accelerator.print(
            f"   Load in vLLM with: LLM(model='{merged_path}', dtype='bfloat16')"
        )
    
    
if __name__ == "__main__":
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args, _ = parser.parse_args_and_config(
        return_remaining_strings=True
    )
    main(script_args, training_args, model_args)