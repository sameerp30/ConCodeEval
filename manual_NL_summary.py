from genai import Credentials, Client
from genai.schema import TextGenerationParameters, TextGenerationReturnOptions
from tqdm import tqdm
import os
import json
from tqdm import tqdm

base_path = "./constrain-data-gen-eval/data_generator/data_generator/"
xml_path = "xml_dataset_6_march/"
yaml_path = "yaml_dataset_6_march/"
json_path = "json_schema_dataset_28_feb/"
NL_path = "NL_summary/"

xml_files = os.listdir(base_path + xml_path)
yaml_files = os.listdir(base_path + yaml_path)
json_files = os.listdir(base_path + json_path)
NL_files = os.listdir(base_path + NL_path)

os.environ['GENAI_KEY'] = 'pak-OVzQSFKXE-IrzfvYM-jQTY6GZUDAk6dVJYBRYq4Zs7Q'
os.environ['GENAI_API'] = 'https://bam-api.res.ibm.com'

credentials = Credentials.from_env()
client = Client(credentials=credentials)

for file in tqdm(NL_files):
    if file == ".ipynb_checkpoints":
        continue
    f = open(base_path+NL_path+file, "r")
    prompt = f.read()
#     prompt = "[INST] " + prompt.split('''JSON sample:\n```''')[0].strip() + '''\nJSON sample:\n[/INST]\n```\n'''
    prompt = prompt.replace('JSON sample', 'XML sample')
    prompt = prompt.strip('\n') + "\n"
    response = list(
             client.text.generation.create(
            model_id="meta-llama/llama-3-8b-instruct",
            inputs=[prompt],
            parameters=TextGenerationParameters(
                temperature=0,
                max_new_tokens=1024,
                return_options=TextGenerationReturnOptions(input_text=True),
                ),
            )
        )
    result = response[0].results[0]
    f_opt = open('''./constrain-data-gen-eval/data_generator/data_generator/Task_1/x_to_XML/NL_2_XML/codellama/8B/zero_shot/without_inst_prompt_1''' + file.split(".")[0] + ".txt", "w")
    f_opt.write(result.input_text + result.generated_text)
    f_opt.close()
#     print(result.input_text + result.generated_text)