from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, LogitsProcessor, default_data_collator
from huggingface_hub import login
import torch
from consts import PROMPT_FOR_GENERATION_FORMAT
from instruct_pipeline import InstructionTextGenerationPipeline
from datasets import load_dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import transformers
import os
import string


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    "RWKV/rwkv-raven-3b",
    #return_dict=True,
    #torch_dtype=torch.float16,
    #quantization_config=bnb_config,
    #rescale_every=0,
)

tokenizer = AutoTokenizer.from_pretrained("RWKV/rwkv-raven-3b")

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r = 8,
    lora_alpha = 32,
    target_modules = ["output"],
    lora_dropout = 0.05,
    bias = "none",
    task_type = "CAUSAL_LM"
    )
    
model = get_peft_model(model,config)

#Data for test
#dataset_name_test = "truthful_qa"
#dataset_test = load_dataset(dataset_name_test, 'generation',split='validation')
#test_dataset_train = dataset_test.shuffle(seed=42).select(range(int(0.01 * len(dataset_test))))
#tokenized_data_test = dataset.map(lambda samples: {'question_tokens': tokenizer(samples['question']), 'answer_tokens': tokenizer(samples['best_answer'])}, batched=True)

##################################################################
tokenizer.pad_token = tokenizer.eos_token
squad = load_dataset("squad")
test_dataset = squad['train'][100:200]
def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while idx < len(sequence_ids) and sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while idx < len(sequence_ids) and sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

tokenized_squad = squad.map(preprocess_function, batched=True, remove_columns=squad["train"].column_names)

trainer = transformers.Trainer(
    model = model,
    train_dataset=tokenized_squad["train"],
    eval_dataset=tokenized_squad["validation"],
    args = transformers.TrainingArguments(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        warmup_steps = 2,
        max_steps = 10,
        learning_rate = 2e-6,
        fp16 = True,
        logging_steps = 1,
        output_dir = "outputs",
        optim = "adamw_hf"
    ),
    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm = False)
)

model.config.use_cache = False

#trainer.train()

def calculate_model_size(model, debug=False):
    total_bytes = 0
    for name, param in model.named_parameters():
        param_size = param.nelement() * param.element_size()
        if debug:
          print(f"Name: {name}, Size: {param_size} bytes")
        total_bytes += param_size

    # Converting bytes to megabytes and gigabytes
    total_megabytes = total_bytes / (1024 ** 2)
    total_gigabytes = total_bytes / (1024 ** 3)
    return total_bytes, total_megabytes, total_gigabytes
    
_, _, base_size_gigabytes = calculate_model_size(trainer.model)
print(base_size_gigabytes)

# pipeline = InstructionTextGenerationPipeline(
#     model=model,
#     tokenizer=tokenizer,
#     top_p=0.92,
#     top_k=50,
#     temperature=1.0,
#     do_sample=True
# )

# def count_overlapping_words(sentence1, sentence2):
#     translator = str.maketrans('', '', string.punctuation)
#     cleaned_sentence1 = sentence1.translate(translator).lower()
#     cleaned_sentence2 = sentence2.translate(translator).lower()
    
#     words_set1 = set(cleaned_sentence1.split())
#     words_set2 = set(cleaned_sentence2.split())
    
#     overlapping_words = words_set1.intersection(words_set2)
#     return len(overlapping_words)
# #############################################################################
# print("Begin test")

# #############################################################
# sum = 0

# for i in range(len(test_dataset["question"])):
#     print(f"the {i} sentence")
#     prompt = str(PROMPT_FOR_GENERATION_FORMAT.format(instruction=test_dataset["question"][i]))
#     gen = pipeline(prompt, max_new_tokens=60)
#     first_generated_text = gen[0]['generated_text']
#     response_hat = first_generated_text.split('<|endoftext|>')[0]
#     print(response_hat)
#     print(test_dataset["answers"][i]["text"][0])
#     if count_overlapping_words(response_hat,test_dataset["answers"][i]["text"][0]) >= 1:
#         sum += 1
#         print("correct")

# print("Accuracy Rate:" + str(sum / len(test_dataset["question"])))

