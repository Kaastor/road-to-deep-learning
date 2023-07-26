import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# uses adapter-model.bin
peft_model_id = "Llama-finetuned"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, return_dict=True, load_in_8bit=True, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Load the Llama2 model
model = PeftModel.from_pretrained(model, peft_model_id)

# List of strings for inference
input_strings = ["What is ActiveID?",
                 "How does ActiveID work with Assignments?",
                 "Why is the ActiveID report incomplete without the ACTIVE_ID feature flag enabled?",
                 "How is the typing pattern collected in the customer's domain?",
                 "Why aren't there any results in the ActiveID section for the first 1-2 Assignments even after enabling the ACTIVE_ID feature flag?",
                 "What kind of data is displayed in the ActiveID charts section?",
                 "What are the five functionalities of the Teacher Assignments feature?"]

output_strings = []

with torch.cuda.amp.autocast():
    for input_string in input_strings:
        # Convert each string to tensors
        batch = tokenizer(input_string, return_tensors='pt')
        # Generate output tokens
        output_tokens = model.generate(**batch, max_new_tokens=50)
        # Decode output tokens and append to list
        output_strings.append(tokenizer.decode(output_tokens[0], skip_special_tokens=True))

# Print all output strings
for i, output_string in enumerate(output_strings):
    print(f'Input: {input_strings[i]}\nOutput: {output_string}\n\n')

