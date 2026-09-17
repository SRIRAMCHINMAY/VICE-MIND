"""LoRA fine-tune the Phase 1 Tommy model.

Default model is intentionally small enough to be practical on a 4 GB GPU.
Override MODEL_NAME if you want to try a larger checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def make_text(example: dict, tokenizer) -> str:
    messages = [
        {
            "role": "user",
            "content": (
                "You are Tommy Vercetti making a decision inside a simulated Vice City.\n\n"
                "Current situation:\n"
                f"{example['input']}\n\n"
                "Available actions: steal_car, ignore, wait, flee, fight, kill, threaten\n\n"
                "Choose exactly one action. Choose a target_id only when relevant. "
                "Respond with ONLY valid JSON and no markdown:\n"
                '{"action":"<action>","target_id":"<entity id or null>",'
                '"justification":"<one short sentence>"}'
            ),
        },
        {
            "role": "assistant",
            "content": json.dumps(example["output"], ensure_ascii=False),
        },
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("dataset/phase1.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("models/tommy-phase1"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was not detected. For this Phase 1 setup, run training with "
            "a CUDA-enabled PyTorch installation."
        )

    args.out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("json", data_files=str(args.data), split="train")
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    def tokenize(example):
        text = make_text(example, tokenizer)
        tokens = tokenizer(
            text,
            truncation=True,
            max_length=args.max_length,
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized = dataset.map(
        tokenize,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing",
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
    )

    model.config.use_cache = False
    model.enable_input_require_grads()

    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    from peft import get_peft_model

    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=str(args.out / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=collator,
    )

    trainer.train()

    # Merge LoRA weights so agent.py can load one ordinary local model directory.
    merged = model.merge_and_unload()
    merged.save_pretrained(args.out, safe_serialization=True)
    tokenizer.save_pretrained(args.out)

    print(f"\nSaved merged Tommy model to: {args.out}")


if __name__ == "__main__":
    main()
