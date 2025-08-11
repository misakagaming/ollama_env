import pandas as pd
import datasets
import huggingface_hub
from huggingface_hub import notebook_login
from datasets import Dataset, DatasetDict, load_dataset
import json
import numpy as np
import regex as re
import evaluate
from openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
import sys

key = sys.argv[1]

#0 = lamner #1 = lamner_only_codebert #2 = lamner_codebert
#3 = lam    #4 = ner                  #5 = static
#6 = tlcodesum #7 = codebert
#8 = rencos #9 = rencos_lamner

def get_preds(file_name, client):
    file_response = client.files.content('file-' + file_name)
    #print(file_response.text)
    results_filename = "gpt4-o-mini-results-with-ties-translate-9-9.jsonl"
    with open(results_filename, "w", encoding = "utf-8", errors = "ignore") as f:
        f.write(file_response.text)
        
    df = pd.read_json(results_filename, lines = True)
    df.head()

    df["prediction"] = df["response"].apply(lambda x: x["body"]["choices"][0]["message"]["content"])
    predictions = list(df["prediction"])

    return predictions
    


def create_batch(task = "summary", model_1 = 0, model_2 = 1, start=0, end=None, java=None, cs=None, error=None, lang=None, code=None, llm="gpt-4.1-nano-2025-04-14"):
    task_lines = []
    count = 1
    content_format = bigcodebench_format
    task_instruction = bigcodebench_format
    if task == "summary":
        content_format = summary_content_format
        task_instruction = summary_instruction
        model_1_predictions = mlsum_predictions[model_1]
        model_2_predictions = mlsum_predictions[model_2]
        inputs = mlsum_inputs
    elif task == "translation":
        content_format = translation_format
        task_instruction = translation_instruction
        model_1_predictions = trnews_predictions[model_1]
        inputs = mlsum_inputs
    elif task == "completion":
        content_format = completion_format
        task_instruction = completion_instruction
        outputs = main_outputs[start:start+100]
        inputs = coms[model_1][start:start+100]
    elif task == "completion_cs":
        content_format = completion_format
        task_instruction = completion_instruction_cs
        outputs = main_outputs
        model_1_predictions = trnews_predictions[model_1][:100]
        inputs = coms[model_1][:100]
    elif task == "unit_test":
        content_format = unit_test_format
        task_instruction = unit_test_instruction
        outputs = coms[model_1][:100]
        java = java[:100]
        cs = cs[:100]
    elif task == "unit_test_diff":
        if not end:
            end = start + 10
        content_format = unit_test_format
        task_instruction = unit_test_instruction
        outputs = coms[model_1][start:end]
        java = java[start:end]
        cs = cs[start:end]
    elif task == "error_fix":
        content_format = completion_fix_format
        if lang == "java":
          task_instruction = completion_instruction_fixed
        elif lang == "cs":
          task_instruction = completion_instruction_cs_fixed



    #sampled_input_ids = list(rng.choice(8714, 50, replace=False))
    sampled_input_ids = range(5000, 5200)
    #sampled_inst_nos = list(rng.choice(174, 50, replace = True))
    sampled_inst_nos = range(5000, 5200)
    if task == "summary":
        for input_id, inst_no in zip(sampled_input_ids, sampled_inst_nos):
            text = inputs[input_id]
            """if model_1 != 4:
                prediction_1 = model_1_predictions[inst_no*500+input_id]
            else:
                prediction_1 = model_1_predictions[input_id]"""
            prediction_1 = model_1_predictions[input_id]

            """if model_2 != 4:
                prediction_2 = model_2_predictions[inst_no*500+input_id]
            else:
                prediction_2 = model_2_predictions[input_id]"""
            prediction_2 = model_2_predictions[input_id]

            content_1 = content_format.format(instruction = task_instruction, text = text, output_1 = prediction_1, output_2 = prediction_2)
            content_2 = content_format.format(instruction = task_instruction, text = text, output_1 = prediction_2, output_2 = prediction_1)
            line_1 = {"custom_id": "{task}-{model_1}-{model_2}-{count}".format(task = task, model_1 = str(model_1), model_2 = str(model_2), count = count),
                      "method": "POST", "url": "/v1/chat/completions",
                      "body": {"model": "gpt-4o-mini", "messages": [{"role": "system", "content": "You are a helpful assistant."},
                                                                    {"role": "user", "content": content_1}],"max_tokens": 16}}
            task_lines.append(line_1)
            count += 1
            line_2 = {"custom_id": "{task}-{model_1}-{model_2}-{count}".format(task = task, model_1 = str(model_1), model_2 = str(model_2), count = count),
                      "method": "POST", "url": "/v1/chat/completions",
                      "body": {"model": "gpt-4o-mini", "messages": [{"role": "system", "content": "You are a helpful assistant."},
                                                                    {"role": "user", "content": content_2}],"max_tokens": 16}}
            task_lines.append(line_2)
            count += 1

        filename = "{task}-{model_1}-{model_2}-batch.jsonl".format(task = task, model_1 = str(model_1), model_2 = str(model_2))
    elif task == "translation":
        for input_id, inst_no in zip(sampled_input_ids, sampled_inst_nos):
            text = inputs[input_id]
            """if model_1 != 4:
                prediction_1 = model_1_predictions[inst_no*500+input_id]
            else:
                prediction_1 = model_1_predictions[input_id]"""
            prediction_1 = model_1_predictions[input_id]
            content_1 = content_format.format(instruction = task_instruction, text = text, output = prediction_1)
            line_1 = {"custom_id": "{task}-{model_1}-{count}".format(task = task, model_1 = str(model_1), count = count),
                      "method": "POST", "url": "/v1/chat/completions",
                      "body": {"model": "gpt-4o-mini", "messages": [{"role": "system", "content": "You are a helpful assistant."},
                                                                    {"role": "user", "content": content_1}],"max_tokens": 256}}
            task_lines.append(line_1)
            count += 1
        filename = "{task}-{model_1}-batch.jsonl".format(task = task, model_1 = str(model_1))
    elif task == "bigcodebench":
        for sample in iter(ds):
            complete_prompt = sample['complete_prompt']
            instruct_prompt = sample['instruct_prompt']
            canonical_solution = sample['canonical_solution']
            code_prompt = sample['code_prompt']
            test = sample['test']
            entry_point = sample['entry_point']
            doc_struct = sample['doc_struct']
            libs = sample['libs']
            content_1 = content_format.format(instruction = task_instruction,
                                              complete_prompt = complete_prompt,
                                              instruct_prompt = instruct_prompt,
                                              canonical_solution = canonical_solution,
                                              code_prompt = code_prompt,
                                              test = test,
                                              entry_point = entry_point,
                                              doc_struct = doc_struct,
                                              libs = libs)
            line_1 = {"custom_id": "{task}-{count}".format(task = task, count = count),
                      "method": "POST", "url": "/v1/chat/completions",
                      "body": {"model": "gpt-4o-mini", "messages": [{"role": "system", "content": "You are a helpful assistant."},
                                                                    {"role": "user", "content": content_1}],"max_tokens": 256}}
            task_lines.append(line_1)
            count += 1
        filename = "{task}-batch.jsonl".format(task = task)

    elif task == "completion":
        for input, output in zip(inputs, outputs):
            content_1 = content_format.format(instruction = task_instruction,
                                              text = input,
                                              output = output)
            line_1 = {"custom_id": "{task}-{count}".format(task = task, count = count),
                      "method": "POST", "url": "/v1/chat/completions",
                      "body": {"model": "gpt-4.1-2025-04-14", "messages": [{"role": "system", "content": "You are a helpful assistant."},
                                                                    {"role": "user", "content": content_1}],"max_tokens": 1024}}
            task_lines.append(line_1)
            count += 1
        filename = "{task}-batch.jsonl".format(task = task)

    elif task == "completion_cs":
        for input, output in zip(model_1_predictions, outputs):
            content_1 = content_format.format(instruction = task_instruction,
                                              text = input,
                                              output = output)
            line_1 = {"custom_id": "{task}-{count}".format(task = task, count = count),
                      "method": "POST", "url": "/v1/chat/completions",
                      "body": {"model": "gpt-4.1-2025-04-14", "messages": [{"role": "system", "content": "You are a helpful assistant."},
                                                                    {"role": "user", "content": content_1}],"max_tokens": 1024}}
            task_lines.append(line_1)
            count += 1
        filename = "{task}-batch.jsonl".format(task = task)
    elif task == "unit_test":
        for java, cs, output in zip(java, cs, outputs):
            content_1 = content_format.format(instruction = task_instruction,
                                              java = java,
                                              cs = cs,
                                              output = output)
            line_1 = {"custom_id": "{task}-{count}".format(task = task, count = count),
                      "method": "POST", "url": "/v1/chat/completions",
                      "body": {"model": llm, "messages": [{"role": "system", "content": "You are a helpful assistant."},
                                                                    {"role": "user", "content": content_1}],"max_tokens": 4096}}
            task_lines.append(line_1)
            count += 1
        filename = "{task}-batch.jsonl".format(task = task)
    elif task == "unit_test_diff":
        for java, cs, output in zip(java, cs, outputs):
            model = OllamaLLM(model=llm)
            prompt = ChatPromptTemplate.from_template(content_format)
            chain = prompt | model
            output = chain.invoke({"instruction": task_instruction,
                                    "java": java,
                                    "cs": cs,
                                    "output": output})
            task_lines.append(output)
            count += 1

        filename = "{task}-batch.jsonl".format(task = task)
    elif task == "error_fix":
        content_1 = content_format.format(instruction = task_instruction,
                                          text = code,
                                          error = error)
        line_1 = {"custom_id": "{task}-{count}".format(task = task, count = count),
                  "method": "POST", "url": "/v1/chat/completions",
                  "body": {"model": "gpt-4.1-2025-04-14", "messages": [{"role": "system", "content": "You are a helpful assistant."},
                                                                {"role": "user", "content": content_1}],"max_tokens": 4096}}
        task_lines.append(line_1)
        filename = "{task}-batch.jsonl".format(task = task)

    if task == "unit_test_diff":
        return task_lines

    with open(filename, "w", encoding = "utf-8", errors = "ignore") as f:
        for line in task_lines:
            f.write(json.dumps(line))
            f.write("\n")

    batch_input_file = client.files.create(
          file=open(filename, "rb"),
          purpose="batch")

    return batch_input_file





rng = np.random.default_rng(42)
client = OpenAI(api_key = key)

with open("competition/input.code", "r", encoding = "utf-8-sig" ) as f:
  codes = f.readlines()
with open("competition/input.comment", "r", encoding = "utf-8-sig" ) as f:
  comments = f.readlines()

print(len(codes), len(comments))

for i in range(len(codes)):
    codes[i] = "\t".join(codes[i].split("\t")[1:]).replace("\n", " ")
    comments[i] = "\t".join(comments[i].split("\t")[1:]).replace("\n", " ")
    
print(codes[:10])
print(comments[:10])


data = {"input": codes,
        'output': comments}

mlsum_df = pd.DataFrame(data)
trnews_df = pd.DataFrame(data)

mlsum_inputs = list(mlsum_df["input"])
trnews_inputs = list(trnews_df["input"])

print(len(mlsum_df), len(trnews_df))

main_inputs = mlsum_inputs
main_outputs = list(mlsum_df["output"])

print(len(main_inputs), len(main_outputs))

with open("competition/lamner/test-predictions.out.txt", "r", encoding = "utf-8-sig" ) as f:
  lamner_com_predictions = f.readlines()
with open("competition/lamner_only_codebert/test-predictions.out.txt", "r", encoding = "utf-8-sig" ) as f:
  lamner_only_codebert_com_predictions = f.readlines()
with open("competition/lamner_codebert/test-predictions.out.txt", "r", encoding = "utf-8-sig" ) as f:
  lamner_codebert_com_predictions = f.readlines()
with open("competition/lam/test-predictions.out.txt", "r", encoding = "utf-8-sig" ) as f:
  lam_com_predictions = f.readlines()
with open("competition/ner/test-predictions.out.txt", "r", encoding = "utf-8-sig" ) as f:
  ner_com_predictions = f.readlines()
with open("competition/static/test-predictions.out.txt", "r", encoding = "utf-8-sig" ) as f:
  static_com_predictions = f.readlines()
with open("competition/tl_codesum/test-predictions.out.txt", "r", encoding = "utf-8-sig" ) as f:
  tl_codesum_com_predictions = f.readlines()
with open("competition/codebert/test-predictions.out.txt", "r", encoding = "utf-8-sig" ) as f:
  codebert_com_predictions = f.readlines()
with open("competition/rencos/test-predictions.out.txt", "r", encoding = "utf-8-sig" ) as f:
  rencos_com_predictions = f.readlines()
with open("competition/rencos_lamner/test-predictions.out.txt", "r", encoding = "utf-8-sig" ) as f:
  rencos_lamner_com_predictions = f.readlines()

with open("translation_results/normal/translation-results-0.txt", "r", encoding = "utf-8-sig" ) as f:
  lamner_predictions = f.readlines()
with open("translation_results/normal/translation-results-1.txt", "r", encoding = "utf-8-sig" ) as f:
  lamner_only_codebert_predictions = f.readlines()
with open("translation_results/normal/translation-results-2.txt", "r", encoding = "utf-8-sig" ) as f:
  lamner_codebert_predictions = f.readlines()
with open("translation_results/normal/translation-results-3.txt", "r", encoding = "utf-8-sig" ) as f:
  lam_predictions = f.readlines()
with open("translation_results/normal/translation-results-4.txt", "r", encoding = "utf-8-sig" ) as f:
  ner_predictions = f.readlines()
with open("translation_results/normal/translation-results-5.txt", "r", encoding = "utf-8-sig" ) as f:
  static_predictions = f.readlines()
with open("translation_results/normal/translation-results-6.txt", "r", encoding = "utf-8-sig" ) as f:
  tl_codesum_predictions = f.readlines()
with open("translation_results/normal/translation-results-7.txt", "r", encoding = "utf-8-sig" ) as f:
  codebert_predictions = f.readlines()
with open("translation_results/normal/translation-results-8.txt", "r", encoding = "utf-8-sig" ) as f:
  rencos_predictions = f.readlines()
with open("translation_results/normal/translation-results-9.txt", "r", encoding = "utf-8-sig" ) as f:
  rencos_lamner_predictions = f.readlines()

for i in range(len(lamner_predictions)):
    lamner_predictions[i] = lamner_predictions[i].replace("\n", " ")
    lamner_only_codebert_predictions[i] = lamner_only_codebert_predictions[i].replace("\n", " ")
    lamner_codebert_predictions[i] = lamner_codebert_predictions[i].replace("\n", " ")
    lam_predictions[i] = lam_predictions[i].replace("\n", " ")
    ner_predictions[i] = ner_predictions[i].replace("\n", " ")
    static_predictions[i] = static_predictions[i].replace("\n", " ")
    tl_codesum_predictions[i] = tl_codesum_predictions[i].replace("\n", " ")
    codebert_predictions[i] = codebert_predictions[i].replace("\n", " ")
    rencos_predictions[i] = rencos_predictions[i].replace("\n", " ")
    rencos_lamner_predictions[i] = rencos_lamner_predictions[i].replace("\n", " ")


for i in range(len(lamner_com_predictions[:1500])):
    lamner_com_predictions[i] = lamner_com_predictions[i].replace("\n", " ")
    lamner_only_codebert_com_predictions[i] = lamner_only_codebert_com_predictions[i].replace("\n", " ")
    lamner_codebert_com_predictions[i] = lamner_codebert_com_predictions[i].replace("\n", " ")
    lam_com_predictions[i] = lam_com_predictions[i].replace("\n", " ")
    ner_com_predictions[i] = ner_com_predictions[i].replace("\n", " ")
    static_com_predictions[i] = static_com_predictions[i].replace("\n", " ")
    tl_codesum_com_predictions[i] = tl_codesum_com_predictions[i].replace("\n", " ")
    codebert_com_predictions[i] = codebert_com_predictions[i].replace("\n", " ")
    rencos_com_predictions[i] = rencos_com_predictions[i].replace("\n", " ")
    rencos_lamner_com_predictions[i] = rencos_lamner_com_predictions[i].replace("\n", " ")


print(lamner_predictions[:10])
print(lamner_com_predictions[:10])

mlsum_predictions=[lamner_predictions, lamner_only_codebert_predictions, lamner_codebert_predictions, lam_predictions, ner_predictions, static_predictions, tl_codesum_predictions, codebert_predictions, rencos_predictions, rencos_lamner_predictions]
trnews_predictions=[lamner_predictions, lamner_only_codebert_predictions, lamner_codebert_predictions, lam_predictions, ner_predictions, static_predictions, tl_codesum_predictions, codebert_predictions, rencos_predictions, rencos_lamner_predictions]
coms=[lamner_com_predictions, lamner_only_codebert_com_predictions, lamner_codebert_com_predictions, lam_com_predictions, ner_com_predictions, static_com_predictions, tl_codesum_com_predictions, codebert_com_predictions, rencos_com_predictions, rencos_lamner_com_predictions]

with open("competition/instruction.txt", "r", encoding = "utf-8-sig" ) as f:
    instruction_list = f.read()
    instruction_list = instruction_list.split("\n")
    instruction_list = [x for x in instruction_list if x != ""]

print(len(instruction_list))
print(instruction_list)

translation_instruction = instruction_list[0]
summary_instruction = instruction_list[1]

bigcodebench_instruction = "You are a code generator. You are given a list of prompts and a canonical solution. Generate a Python program that passes the given tests using the specified libraries. The answer should only consist of the resulting Python program including the imported libraries. Do not give an answer that is not the resulting Python program and the imported libraries."

completion_instruction = "You are a code completer. You are given a code snippet and a code comment of the code snippet. Generate a Java program that can run this code snippet, incuding all of the necessary class definitions and import statements. Your answer should consist ONLY of the resulting Java program. Do not put any descriptions in your answer other than the resulting Java program."
completion_instruction_cs = "You are a code completer. You are given a code snippet and a code comment of the code snippet. Generate a C# program that can run this code snippet, incuding all of the necessary class definitions and import statements. Your answer should consist ONLY of the resulting C# program. Do not put any descriptions in your answer other than the resulting C# program."

completion_instruction_fixed = "You are a code completer. You are given a faulty code with unit tests, and a compilation error. Generate a fixed Java program that addresses and fixes the compilation error. Your answer should consist ONLY of the resulting Java program. Do not put any descriptions in your answer other than the resulting Java program."
completion_instruction_cs_fixed = "You are a code completer. You are given a faulty code with unit tests, and a compilation error. Generate a fixed C# program that addresses and fixes the compilation error. Your answer should consist ONLY of the resulting C# program. Do not put any descriptions in your answer other than the resulting C# program."

unit_test_instruction = "You are a code tester. You are given a Java code, its corresponding C# code that performs the same task and a code comment description of the task. Generate 5 unit tests for the task along with the original code and run these unit tests on both the Java code and C# code. Preserve the import statements, as well as class and function definitions in the original programs. All unit tests should be performed in the 'main' function definition of the classes. The unit tests generated for both the Java and C# code should be the exact same with the exact same inputs and the exact same correct outputs. Return the unit tests, run these unit tests on both the Java and C# code, and return their outputs and the percentage of unit tests that pass for both the Java code and C# code. Your answer should consist ONLY of the unit tests along with the original program, their outputs and the percentages. Do not put any descriptions or explanations in your answer other than the unit tests with the original programs, their outputs and the percentages. Do not generate different unit tests for the Java and C# programs."

unit_test_instruction_old = "You are a code tester. You are given a Java code, its corresponding C# code that performs the same task and a code comment description of the task. Generate 5 unit tests for the task along with the original code and run these unit tests on both the Java code and C# code. Preserve the import statements, as well as class and function definitions in the original programs. All unit tests should be performed in the 'main' function definition of the classes. The unit tests generated for both the Java and C# code should be the exact same with the exact same inputs and the exact same correct outputs. Return the unit tests, run these unit tests on both the Java and C# code, and return their outputs and the percentage of unit tests that pass for both the Java code and C# code. Your answer should consist ONLY of the unit tests along with the original program, their outputs and the percentages. Do not put any descriptions in your answer other than the unit tests with the original programs, their outputs and the percentages. Do not generate different unit tests for the Java and C# programs."



summary_content_format = """{instruction}
Code: {text}
Summary 1: {output_1}
Summary 2: {output_2}"""

translation_format = """{instruction}
Code: {text}
Summary: {output}"""

bigcodebench_format = """ {instruction}
complete_prompt: {complete_prompt}
instruct_prompt: {instruct_prompt}
canonical_solution: {canonical_solution}
code_prompt: {code_prompt}
test: {test}
entry_point: {entry_point}
doc_struct: {doc_struct}
libs: {libs}"""

completion_format = """{instruction}
Code: {text}
Summary: {output}"""

completion_fix_format = """{instruction}
Code: {text}
Error: {error}"""

unit_test_format = """{instruction}
Java: {java}
C#: {cs}
Summary: {output}"""



#completion_new

#normal
#batch_id: "batch_687cef75d6d08190ac1f036935b60883" output_file file-48rHYAsQbky39RMBQhh15t

#alt
#batch_id: "" output_file file-

#completion_cs_new

#0 = lamner #1 = lamner_only_codebert #2 = lamner_codebert
#3 = lam    #4 = ner                  #5 = static
#6 = tlcodesum #7 = codebert
#8 = rencos #9 = rencos_lamner

#normal
#0 batch_id: "batch_687cefc774ec8190aa3aee041fbaed0f" output_file file-6K5KcSZfwLcU6c5kXCYDEo
#1 batch_id: "batch_687cefd48bbc8190a400fab18242c208" output_file file-4bsFsp8eSfkXYhEtZY1xqL
#2 batch_id: "batch_687cefe69d388190bb12fbb7b6e187d8" output_file file-9f4s3N1n3TwJTRTeWE2NTY
#3 batch_id: "batch_687ceff1e5008190915a9ad6cd467d06" output_file file-GNRojuNHKmXuuCBTdjV25n
#4 batch_id: "batch_687ceffcf8188190b11e383a737457b9" output_file file-VPsjkHYxmG5saqgm9heydp
#5 batch_id: "batch_687cf00801648190871953cae40226a3" output_file file-NRRn7eeGkbzSpQvTWLPHbg
#6 batch_id: "batch_687cf011ee9c81909f5ef50e6867db38" output_file file-MywQHThocYnogipesRe45f
#7 batch_id: "batch_687cf01a54848190b7751cf1376efe0b" output_file file-J1anWMwG2PRSYBS4Tov8WV
#8 batch_id: "batch_687cf02332748190886a5bef131ef1b3" output_file file-SR5yReZADFBKU54NiP9BXD
#9 batch_id: "batch_687cf031bb508190b2a8f3f9508a1d2f" output_file file-QjpeJxC1GbsFPEAYm97XFM

#alt
#0 batch_id: "" output_file file-
#1 batch_id: "" output_file file-
#2 batch_id: "" output_file file-
#3 batch_id: "" output_file file-
#4 batch_id: "" output_file file-
#5 batch_id: "" output_file file-
#6 batch_id: "" output_file file-
#7 batch_id: "" output_file file-
#8 batch_id: "" output_file file-
#9 batch_id: "" output_file file-


file_response = client.files.content('file-6K5KcSZfwLcU6c5kXCYDEo')
#print(file_response.text)
results_filename = "gpt4-o-mini-results-with-ties-translate-9-9.jsonl"
with open(results_filename, "w", encoding = "utf-8", errors = "ignore") as f:
    f.write(file_response.text)
    
df = pd.read_json(results_filename, lines = True)
df.head()

df["prediction"] = df["response"].apply(lambda x: x["body"]["choices"][0]["message"]["content"])
predictions = list(df["prediction"])

cs = get_preds("48rHYAsQbky39RMBQhh15t", client)
java = get_preds("6K5KcSZfwLcU6c5kXCYDEo", client)

batch_input_file = create_batch(task = "unit_test_diff", java=java, cs=cs, llm="mistral:latest", start=0, end=100)


results_filename = "unit-tests-1-0.txt"
count = 0
with open(results_filename, "w", encoding = "utf-8", errors = "ignore") as f:
    for line in batch_input_file:
        f.write(f"CODE COUNT: {count}\n\n")
        count+=1
        f.write(f"{line}\n")


