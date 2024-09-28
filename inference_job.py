from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", device_map="auto", token="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")

prompt = '''Write JSON sample with field values as per the schema given in JSON format below.

 {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "ethenoid": {},
            "veneralia": {
                "type": "string"
            },
            "pituicyte": {
                "type": "boolean"
            },
            "lopper": {
                "type": "string"
            },
            "flavaniline": {
                "type": "string",
                "minLength": 1,
                "maxLength": 4
            },
            "palatably": {
                "type": "boolean"
            }
        },
        "additionalProperties": true
    }
}

JSON sample:
```
'''

inputs = tokenizer(prompt, return_tensors="pt").to("cuda:1")
output = model.generate(**inputs, max_new_tokens=1000)
gen_tokens = output.reshape(1, -1, output.shape[-1])[0][0]
output_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
print(output_text)
# output_text = tokenizer.decode([128000,   8144,   4823,   6205,    449,   2115,   2819,    439,    824], skip_special_tokens=True)[0]

# print(output_text)
# print("model loaded")

# '''base_path = "./constrain-data-gen-eval/data_generator/data_generator/"
# xml_path = "xml_dataset_6_march/"
# yaml_path = "yaml_dataset_6_march/"
# json_path = "json_schema_dataset_28_feb/"

# xml_files = os.listdir(base_path + xml_path)
# yaml_files = os.listdir(base_path + yaml_path)
# json_files = os.listdir(base_path + json_path)


# for file in tqdm(json_files):
#     f = open(base_path+json_path+file, "r")
#     schema = f.read()
# #     prompt = '''[INST]''' + "\n" + '''Write JSON sample with field values as per the schema given in JSON format below''' + "\n\n" + schema + "\n" + '''[/INST]''' + "\n"
# #     prompt = '''Write a JSON sample with field values as per the JSON format schema given below.''' + "\n\n" + schema + "\n\nJSON sample:\n```\n"
# #     prompt = '''Question:\nWrite a JSON sample as per the JSON format schema given below.\n''' + schema + '''\nAnswer:\n```\n'''
#     prompt = '''[INST] Write JSON sample with field values as per the schema given in JSON format below.\n''' + '''The JSON sample must be between [SAMPLE] and [/SAMPLE] tags.\n\n''' + schema + '''\n[/INST]\n[SAMPLE]\n```\n'''

#     inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
#     output = model.generate(**inputs, max_new_tokens=1000, num_beams=3)
#     output_text = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
#     f_opt = open("./constrain-data-gen-eval/data_generator/data_generator/json_outputs/codellama/34B/prompt_5/" + file, "w")
#     f_opt.write(output_text)
#     f_opt.close()'''