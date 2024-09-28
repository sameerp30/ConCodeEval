from genai import Credentials, Client
from genai.schema import TextGenerationParameters, TextGenerationReturnOptions
from tqdm import tqdm
import os
import sys

# base_path = "./constrain-data-gen-eval/data_generator/data_generator/"
# xml_path = "xml_dataset_6_march/"
# yaml_path = "yaml_dataset_6_march/"
# json_path = "json_schema_dataset_28_feb/"
# python_path = "python_schema_data_4th_june/"

input_repr = sys.argv[1]
output_repr = sys.argv[2]

base_path = "./constrain-data-gen-eval/data_generator/data_generator/"

if input_repr == "JSON":
    input_path = "json_schema_dataset_28_feb/"
    
elif input_repr == "YAML":
    input_path = "yaml_dataset_6_march/"

elif input_repr == "Python":
    input_path = "python_schema_data_4th_june/"

elif input_repr == "XML":
    input_path = "xml_dataset_6_march/"
    
xml_files = os.listdir(base_path + xml_path)
yaml_files = os.listdir(base_path + yaml_path)
json_files = os.listdir(base_path + json_path)
python_files = os.listdir(base_path + python_path)

prompt_few_shot = '''Your task is to write a JSON sample with field values as per JSON format schema.
You are given a few examples demonstrating the same.

JSON format schema:
{
    "type": "array",
    "contains": {
        "type": "string"
    }
}
JSON sample:
```
[
    "string1",
    "string2",
    "string3"
]
```

JSON format schema:
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

JSON format schema:
{
    "type": "number",
    "exclusiveMinimum": 0,
    "exclusiveMaximum": 10
}
JSON sample:
```
67.89
```

JSON format schema:
'''

prompt_few_shot_xml = '''Your task is to write a JSON sample with field values as per XML format schema.
You are given a few examples demonstrating the same.

XML format schema:
<?xml version="1.0" ?>
<all>
    <type type="str">array</type>
    <contains type="dict">
        <type type="str">object</type>
        <properties type="dict">
            <replied type="dict">
                <type type="str">string</type>
            </replied>
            <scirenga type="dict"/>
            <medias type="dict">
                <type type="str">boolean</type>
            </medias>
            <pyrocinchonic type="dict">
                <type type="str">string</type>
            </pyrocinchonic>
            <thermoses type="dict"/>
            <unglue type="dict">
                <type type="str">string</type>
            </unglue>
            <emboss type="dict"/>
        </properties>
        <additionalProperties type="bool">true</additionalProperties>
        <required type="list">
        <item type="str">thermoses</item>
            <item type="str">unglue</item>
            <item type="str">scirenga</item>
            <item type="str">replied</item>
        </required>
    </contains>
    <minContains type="int">0</minContains>
</all>
JSON sample:
```
[
    {
        "replied": "string1",
        "scirenga": "string2",
        "medias": true,
        "pyrocinchonic": "string3",
        "thermoses": "string4",
        "unglue": "string5",
        "emboss": "string6"
    },
    {
        "replied": "string7",
        "scirenga": "string8",
        "medias": false,
        "pyrocinchonic": "string9",
        "thermoses": "string10",
        "unglue": "string11",
        "emboss": "string12"
    }
]
```

XML format schema:
<?xml version="1.0" ?>
<all>
    <type type="str">string</type>
    <format type="str">uri-reference</format>
    <minLength type="int">0</minLength>
    <maxLength type="int">50</maxLength>
</all>
JSON sample:
```
"www.example.com"
```

XML format schema:
<?xml version="1.0" ?>
<all>
    <type type="str">number</type>
    <multipleOf type="float">33.67</multipleOf>
    <minimum type="int">0</minimum>
    <exclusiveMaximum type="int">100</exclusiveMaximum>
</all>
JSON sample:
```
67.34
```

XML format schema:
'''

prompt_few_shot_YAML = '''Your task is to write a JSON sample with field values as per YAML format schema.
You are given a few examples demonstrating the same.

YAML format schema:

contains:
  additionalProperties: true
  properties:
    emboss: {}
    medias:
      type: boolean
    pyrocinchonic:
      type: string
    replied:
      type: string
    scirenga: {}
    thermoses: {}
    unglue:
      type: string
  required:
  - thermoses
  - unglue
  - scirenga
  - replied
  type: object
minContains: 0
type: array

JSON sample:
```
[
    {
        "replied": "string1",
        "scirenga": "string2",
        "medias": true,
        "pyrocinchonic": "string3",
        "thermoses": "string4",
        "unglue": "string5",
        "emboss": "string6"
    },
    {
        "replied": "string7",
        "scirenga": "string8",
        "medias": false,
        "pyrocinchonic": "string9",
        "thermoses": "string10",
        "unglue": "string11",
        "emboss": "string12"
    }
]
```

YAML format schema:

format: uri-reference
maxLength: 50
minLength: 0
type: string

JSON sample:
```
"www.example.com"
```

YAML format schema:

maximum: 100
minimum: 0
multipleOf: 33.67
type: number

JSON sample:
```
67.34
```

YAML format schema:

'''

os.environ['GENAI_KEY'] = 'pak-OVzQSFKXE-IrzfvYM-jQTY6GZUDAk6dVJYBRYq4Zs7Q'
os.environ['GENAI_API'] = 'https://bam-api.res.ibm.com'

credentials = Credentials.from_env()
client = Client(credentials=credentials)



for file in tqdm(json_files):
    
    f = open(base_path+json_path+file, "r")
    schema = f.read()
#     prompt = prompt_few_shot_YAML + schema.strip() + '''\n\nJSON sample:\n```\n'''
    prompt = '''Write {output_repr} sample with field values as per the schema given in {input_repr} format below.\n\n {schema}\n\n{output_repr} sample:\n```\n'''.format(input_repr=input_repr, output_repr=output_repr, schema=schema.strip())
#     prompt = '''Question:\nWrite JSON sample with field values as per the schema given in JSON format below.\n\n''' + schema.strip() + '''\n\nAnswer:\n```\n'''

#     prompt = '''Your task is to write a YAML sample with field values as per the JSON format schema given below.\nThe JSON sample must be between [SAMPLE] and [/SAMPLE] tags.\n\n''' + schema.strip() + '''\n\nYAML sample:\n[SAMPLE]\n'''
#     prompt = '''You are an expert XML programmer. Your task is to convert the Python format schema given below into JSON format schema. Then, you should write a XML sample with field values that conform to your JSON format schema.\n\n''' + schema.strip() + '''\n\nJSON schema:\n```\n'''
#     prompt = '''[INST] You are an expert JSON programmer. Your task is to write a JSON sample with field values directly without converting to JSON format schema as per YAML format schema given below.\n\n''' + schema.strip() + '''\n\nJSON sample:\n[/INST]\n```\n'''

#     prompt = '''[INST] You are an expert JSON programmer. Your task is to write a JSON sample with field values that conform to XML format schema. You should write a JSON sample without converting to JSON format schema.\n\n''' + schema.strip() + '''\n\nJSON sample:\n[/INST]\n```\n'''

    response = list(
    client.text.generation.create(
        model_id="codellama/codellama-34b-instruct",
        inputs=[prompt],
        parameters=TextGenerationParameters(
            temperature=0,
            max_new_tokens=1024,
            return_options=TextGenerationReturnOptions(input_text=True),
            ),
        )
    )
    result = response[0].results[0]
    f_opt = open("./constrain-data-gen-eval/data_generator/data_generator/Task_1/x_to_JSON/JSON_2_JSON/codellama/34B/zero_shot/prompt_1/" + file.split(".")[0] + ".txt", "w")
    f_opt.write(result.input_text + result.generated_text)
    f_opt.close()
#     print(result.input_text)
#     print("#################################")
#     print(result.generated_text)