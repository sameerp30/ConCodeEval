from genai import Credentials, Client
from genai.schema import TextGenerationParameters, TextGenerationReturnOptions
from tqdm import tqdm
import os
import sys

input_repr = sys.argv[1]
output_repr = sys.argv[2]
model = sys.argv[3]
parameter = sys.argv[4]
model_id = sys.argv[5]

base_path = "/dccstor/ai4code-summ/benchmark-paper/backup/constrain-data-gen-eval/data_generator/data_generator/"

if input_repr == "JSON":
    input_path = "json_schema_dataset_28_feb/"
    
elif input_repr == "YAML":
    input_path = "yaml_dataset_6_march/"

elif input_repr == "Python":
    input_path = "python_schema_data_4th_june/"

elif input_repr == "XML":
    input_path = "xml_dataset_6_march/"

elif input_repr == "NL":
    input_path = "NL_summary/"
    
# xml_files = os.listdir(base_path + xml_path)
# yaml_files = os.listdir(base_path + yaml_path)
# json_files = os.listdir(base_path + json_path)
# python_files = os.listdir(base_path + python_path)

input_files = os.listdir(base_path + input_path)

# os.environ['GENAI_KEY'] = 'pak-OVzQSFKXE-IrzfvYM-jQTY6GZUDAk6dVJYBRYq4Zs7Q'
# os.environ['GENAI_KEY'] = 'pak-J06N56058ZDMh2Fb1gB24Wv8iq5CvkfH8J1pm-uf0B8'
os.environ['GENAI_KEY'] = 'pak-HjM28sEXNixooFy4kdQP--L3xHpd54xdHGzKEglPQFw'

os.environ['GENAI_API'] = 'https://bam-api.res.ibm.com'

credentials = Credentials.from_env()
client = Client(credentials=credentials)


for file in tqdm(input_files):
    
    f = open(base_path+input_path+file, "r")
    schema = f.read()
    f_schema = open(base_path+input_path+file, "r")
        
#     prompt = prompt_few_shot_YAML + schema.strip() + '''\n\nJSON sample:\n```\n'''
    # prompt = '''Write {output_repr} sample with field values as per the schema given in {input_repr} format below.\n\n{schema}\n\n{output_repr} sample:\n```\n'''.format(input_repr=input_repr, output_repr=output_repr, schema=schema.strip())
    prompt = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Assume that you are an expert in understanding {input_repr} and writing {output_repr} code.
<|eot_id|><|start_header_id|>user<|end_header_id|>

Write {output_repr} sample with field values as per the schema given in {input_repr} format below.\n\n{schema}\n\n{output_repr} sample:\n```
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

""".format(input_repr=input_repr, output_repr=output_repr, schema=schema.strip())
    
#     prompt = '''Question:\nWrite JSON sample with field values as per the schema given in JSON format below.\n\n''' + schema.strip() + '''\n\nAnswer:\n```\n'''

#     prompt = '''Your task is to write a YAML sample with field values as per the JSON format schema given below.\nThe JSON sample must be between [SAMPLE] and [/SAMPLE] tags.\n\n''' + schema.strip() + '''\n\nYAML sample:\n[SAMPLE]\n'''
#     prompt = '''You are an expert XML programmer. Your task is to convert the Python format schema given below into JSON format schema. Then, you should write a XML sample with field values that conform to your JSON format schema.\n\n''' + schema.strip() + '''\n\nJSON schema:\n```\n'''
#     prompt = '''You are an expert {output_repr} programmer. Your task is to write a {output_repr} sample with field values directly without converting to JSON format schema as per {input_repr} format schema given below.\n\n{schema}\n\n{output_repr} sample:\n```\n'''.format(input_repr=input_repr, output_repr=output_repr, schema=schema.strip())

#     prompt = '''[INST] You are an expert JSON programmer. Your task is to write a JSON sample with field values that conform to XML format schema. You should write a JSON sample without converting to JSON format schema.\n\n''' + schema.strip() + '''\n\nJSON sample:\n[/INST]\n```\n'''

    few_shot_prompt = '''Your task is to write a {output_repr} sample with field values as per {input_repr} format schema.
Please wrap {output_repr} sample using ```.
You are given a few examples demonstrating the same.

{input_repr} format schema:
```
from pydantic import BaseModel, Field

class BoolList(BaseModel):
    bool_list: list[bool] = Field(min_items=0)
```
JSON sample:
```
[
    "string1",
    "string2",
    "string3"
]
```

Python format schema:
{
    "type": "string",
    "format": "duration",
    "minLength": 0,
    "maxLength": 50
}
JSON sample:
```
"P1Y2M10DT2H30M"
```

Python format schema:\n{schema}\nJSON sample:```\n'''

    response = list(
    client.text.generation.create(
        model_id=model_id,
        inputs=[prompt],
        parameters=TextGenerationParameters(
            temperature=0,
            max_new_tokens=1024,
            return_options=TextGenerationReturnOptions(input_text=True),
            ),
        )
    )
    result = response[0].results[0]
    opt_path = "/dccstor/ai4code-summ/benchmark-paper/backup/constrain-data-gen-eval/data_generator/data_generator/Task_1/x_to_{output_repr}/{input_repr}_2_{output_repr}/{model}/{parameter}/zero_shot/prompt_2/".format(input_repr = input_repr, output_repr = output_repr, model=model, parameter=parameter)
    f_opt = open(opt_path + file.split(".")[0] + ".txt", "w")
    f_opt.write(result.input_text + result.generated_text)
    f_opt.close()
#     print(result.input_text)
#     print("#################################")
#     print(result.generated_text)