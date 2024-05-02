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

model = AutoModelForCausalLM.from_pretrained("RWKV/rwkv-raven-3b", torch_dtype=torch.float16).to(0)
tokenizer = AutoTokenizer.from_pretrained("RWKV/rwkv-raven-3b")

tokenizer.pad_token = tokenizer.eos_token
squad = load_dataset("squad")

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
    
_, _, base_size_gigabytes = calculate_model_size(model)
print(base_size_gigabytes)

# def preprocess_function(examples):
#     questions = [q.strip() for q in examples["question"]]
#     inputs = tokenizer(
#         questions,
#         examples["context"],
#         max_length=384,
#         truncation="only_second",
#         return_offsets_mapping=True,
#         padding="max_length",
#     )

#     offset_mapping = inputs.pop("offset_mapping")
#     answers = examples["answers"]
#     start_positions = []
#     end_positions = []

#     for i, offset in enumerate(offset_mapping):
#         answer = answers[i]
#         start_char = answer["answer_start"][0]
#         end_char = answer["answer_start"][0] + len(answer["text"][0])
#         sequence_ids = inputs.sequence_ids(i)

#         # Find the start and end of the context
#         idx = 0
#         while idx < len(sequence_ids) and sequence_ids[idx] != 1:
#             idx += 1
#         context_start = idx
#         while idx < len(sequence_ids) and sequence_ids[idx] == 1:
#             idx += 1
#         context_end = idx - 1

#         # If the answer is not fully inside the context, label it (0, 0)
#         if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
#             start_positions.append(0)
#             end_positions.append(0)
#         else:
#             # Otherwise it's the start and end token positions
#             idx = context_start
#             while idx <= context_end and offset[idx][0] <= start_char:
#                 idx += 1
#             start_positions.append(idx - 1)

#             idx = context_end
#             while idx >= context_start and offset[idx][1] >= end_char:
#                 idx -= 1
#             end_positions.append(idx + 1)

#     inputs["start_positions"] = start_positions
#     inputs["end_positions"] = end_positions
#     return inputs


# tokenized_squad = squad.map(preprocess_function, batched=True, remove_columns=squad["train"].column_names)

# test_dataset = squad['train'][0:200]

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
# sum = 0
# for i in range(len(test_dataset["question"])):
#     print(f"the {i} sentence")
#     print(test_dataset["answers"][i]["text"][0])
#     prompt = str(test_dataset["question"][i])
#     inputs = tokenizer(prompt, return_tensors="pt").to(0)
#     output = model.generate(inputs["input_ids"], max_new_tokens=80)
#     response_hat = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
#     print(response_hat)
#     if count_overlapping_words(response_hat, test_dataset["answers"][i]["text"][0]) >= 1:
#         sum += 1
#         print("correct")

# print("Accuracy Rate:" + str(sum / len(test_dataset["question"])))

prompt = "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?"

inputs = tokenizer(prompt, return_tensors="pt").to(0)
output = model.generate(inputs["input_ids"], max_new_tokens=10)
print(tokenizer.decode(output[0].tolist(), skip_special_tokens=True))