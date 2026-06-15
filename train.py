from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

# base model
max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-1.5B-Instruct",
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)

# lora
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# data (sharegpt)
dataset = load_dataset("json", data_files="dataset_train_sharegpt.jsonl", split="train")

def format_sharegpt(examples):
    texts = []
    for messages in examples["messages"]:
        text = ""
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            # Formats it exactly how Qwen likes it
            text += f"<|im_start|>{role}\n{content}\n<|im_end|>\n"
        # Adds the absolute stop token at the end
        texts.append(text + tokenizer.eos_token)
    return { "text" : texts }

dataset = dataset.map(format_sharegpt, batched = True)

# train
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60, # Running a quick 60-step test
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        seed = 3407,
        output_dir = "lora_outputs",
    ),
)

print("🚀 Starting Training...")
trainer.train()

# save
print("💾 Saving LoRA adapter...")
model.save_pretrained("qwen_1.5b_custom_lora")
tokenizer.save_pretrained("qwen_1.5b_custom_lora")
print("✅ Done!")
