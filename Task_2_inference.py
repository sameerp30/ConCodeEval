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
    
# xml_path = "xml_dataset_6_march/"
# yaml_path = "yaml_dataset_6_march/"
# json_path = "json_schema_dataset_28_feb/"
# python_path = "python_schema_data_4th_june/"
task_2_path = "Task_2/" + output_repr.lower() + "/"

# xml_files = os.listdir(base_path + xml_path)
# yaml_files = os.listdir(base_path + yaml_path)
# json_files = os.listdir(base_path + json_path)
# python_files = os.listdir(base_path + python_path)

input_files = os.listdir(base_path + input_path)
task_2_files = os.listdir(base_path + task_2_path)

# os.environ['GENAI_KEY'] = 'pak-OVzQSFKXE-IrzfvYM-jQTY6GZUDAk6dVJYBRYq4Zs7Q'
os.environ['GENAI_KEY'] = 'pak-J06N56058ZDMh2Fb1gB24Wv8iq5CvkfH8J1pm-uf0B8'
# os.environ['GENAI_KEY'] = 'pak-HjM28sEXNixooFy4kdQP--L3xHpd54xdHGzKEglPQFw'


os.environ['GENAI_API'] = 'https://bam-api.res.ibm.com'

credentials = Credentials.from_env()
client = Client(credentials=credentials)


for file in tqdm(task_2_files):
    
    f_sample = open(base_path+task_2_path+file, "r")
    sample = f_sample.read()
    schema_file = "_".join(file.split("_")[1:])
    if input_repr not in ["Python", "NL"]:
        f_schema = open(base_path+input_path+schema_file.split(".")[0] + "." + input_repr.lower(), "r")
    elif input_repr == "NL":
        f_schema = open(base_path+input_path+schema_file.split(".")[0] + "." + "txt", "r")
    else:
        f_schema = open(base_path+input_path+schema_file.split(".")[0] + ".py", "r")
        
    schema = f_schema.read()
    
    if input_repr == "NL":
        schema = schema.split("Write a")[0].strip()
        
    system_prompt = '''System:\nYou are an intelligent AI programming assistant, utilizing a Granite code language model developed by IBM. Your primary function is to assist users in code explanation, code generation and other software engineering tasks. You MUST follow these guidelines: - Your responses must be factual. Do not assume the answer is "yes" when you do not know, and DO NOT SHARE FALSE INFORMATION. - You should give concise answers. You should follow the instruction and provide the answer in the specified format and DO NOT SHARE FALSE INFORMATION.\n\n'''
    
    prompt = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nAssume that you are an expert in understanding {input_repr} format schema and writing {output_repr} code.\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nQuestion:\nDoes the {output_repr} sample\n\n{sample}\n\nadhere to all the constraints defined in {input_repr} format schema given below.\n\n{schema}\n\nRespond to yes or no.\n\nAnswer:\n```\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>'''.format(input_repr=input_repr, output_repr=output_repr, schema=schema.strip(), sample=sample.strip())

    # prompt = '''Question:\nDoes the {output_repr} sample\n\n{sample}\n\nadhere to all the constraints defined in schema description given below.\n\n{schema}\n\nRespond to yes or no.\n\nAnswer:\n```\n'''.format(input_repr=input_repr, output_repr=output_repr, schema=schema.strip(), sample=sample.strip())
    
#     prompt = '''Question:\nYou are given a {input_repr} format schema and a {output_repr} sample. Validate if {output_repr} sample adheres to {input_repr} schema.\nWhile validating note that if any field defined as an empty object `{{}}` in the schema, then it means any value is allowed for that.\n\n{input_repr} schema:\n{schema}\n\n{output_repr} sample:\n{sample}\n\nSelect answer from two options "yes" or "no".\n\nIf {output_repr} sample adheres to {input_repr} schema for every constraint answer "yes" else answer "no".\n\nIf your answer is "no", list down attributes in {output_repr} sample that do not follow schema constraints.\n\nAnswer:\n```\n'''.format(input_repr=input_repr, output_repr=output_repr, schema=schema.strip(), sample=sample.strip())
    
#     prompt = '''System:\nYou are an intelligent AI programming assistant, utilizing a Granite code language model developed by IBM. Your primary function is to assist users in code explanation, code generation and other software engineering tasks. You MUST follow these guidelines: - Your responses must be factual. Do not assume the answer is "yes" when you do not know, and DO NOT SHARE FALSE INFORMATION. - You should give concise answers. You should follow the instruction and provide the answer in the specified format and DO NOT SHARE FALSE INFORMATION.\n\nQuestion:\nYou are given a {input_repr} format schema and a {output_repr} sample. Validate if {output_repr} sample adheres to {input_repr} schema.\nWhile validating note that if any field defined as an empty object `{{}}` in the schema, then it means any value is allowed for that.\n\n{input_repr} schema:\n{schema}\n\n{output_repr} sample:\n{sample}\n\nSelect answer from two options "yes" or "no".\n\nIf {output_repr} sample adheres to {input_repr} schema for every constraint answer "yes" else answer "no".\n\nAnswer:\n```\n'''.format(input_repr=input_repr, output_repr=output_repr, schema=schema.strip(), sample=sample.strip())

#     prompt = '''Question:\nYou are given a {input_repr} format schema and a {output_repr} sample. Validate if {output_repr} sample adheres to {input_repr} schema. While validating note that if any field defined as an empty object `{{}}` in the schema, then it means any value is allowed for that.\n\nSelect answer from two options "yes" or "no".\nIf {output_repr} sample adheres to {input_repr} schema for every constraint answer "yes" else answer "no".Below are examples demonstrating the same.

# {input_repr} schema:
# <?xml version="1.0" ?>
# <all>
# 	<type type="str">array</type>
# 	<contains type="dict">
# 		<type type="str">boolean</type>
# 	</contains>
# 	<minContains type="int">0</minContains>
# </all>

# {output_repr} sample:
# <?xml version="1.0" ?>
# <all>
# 	<item type="bool">true</item>
# 	<item type="bool">true</item>
# 	<item type="bool">false</item>
# </all>

# Answer:
# ```
# yes
# ```

# {input_repr} schema:
# <?xml version="1.0" ?>
# <all>
# 	<type type="str">array</type>
# 	<contains type="dict">
# 		<type type="str">string</type>
# 	</contains>
# </all>

# {output_repr} sample:
# <?xml version="1.0" ?>
# <all>
# 	<item type="str">abcd</item>
# 	<item type="str">bgfi</item>
# 	<item type="str">3.14</item>
# </all>

# Answer:
# ```
# no
# ```\n\n{input_repr} schema:\n{schema}\n\n{output_repr} sample:\n{sample}\n\nAnswer:\n```\n'''.format(input_repr=input_repr, output_repr=output_repr, schema=schema.strip(), sample=sample.strip())

    # prompt = '''You are given a {input_repr} format schema and a {output_repr} sample. Validate if {output_repr} sample adheres to {input_repr} schema. While validating note that if any field defined as an empty object `{{}}` in the schema, then it means any value is allowed for that.\n\nSelect answer from two options "yes" or "no".\nIf {output_repr} sample adheres to {input_repr} schema for every constraint answer "yes" else answer "no".Below are examples demonstrating the same.

# {input_repr} schema:
# {{
#     "type": "array",
#     "contains": {{
#         "type": "boolean"
#     }},
#     "minContains": 0
# }}

# {output_repr} sample:
# - true
# - true
# - false

# Answer:
# ```
# yes
# ```

# {input_repr} schema:
# {{
#     "type": "array",
#     "contains": {{
#         "type": "string"
#     }}
# }}

# {output_repr} sample:
# - abc
# - def
# - 13.69

# Answer:
# ```
# no
# ```\n\n{input_repr} schema:\n{schema}\n\n{output_repr} sample:\n{sample}\n\nAnswer:\n```\n'''.format(input_repr=input_repr, output_repr=output_repr, schema=schema.strip(), sample=sample.strip())
    
    if model == "granite":
        prompt = system_prompt + prompt        

    
    try:
        response = list(
            client.text.generation.create(
            model_id=model_id,
            inputs=[prompt],
            parameters=TextGenerationParameters(
                temperature=0,
                max_new_tokens=20,
                return_options=TextGenerationReturnOptions(input_text=True),
              ),
           )
        )
        result = response[0].results[0]
        opt_path = "/dccstor/ai4code-summ/benchmark-paper/backup/constrain-data-gen-eval/data_generator/data_generator/Task_2/x_to_{output_repr}/{input_repr}_2_{output_repr}/{model}/{parameter}/zero_shot/prompt_2/".format(input_repr = input_repr, output_repr = output_repr, model=model, parameter=parameter)
        
        f_opt = open(opt_path + file.split(".")[0] + ".txt", "w")

        f_opt.write(result.input_text + result.generated_text)
        f_opt.close()
    except:
        print(file)