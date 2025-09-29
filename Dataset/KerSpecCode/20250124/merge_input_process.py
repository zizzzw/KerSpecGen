import json

def process_specification(data):
    output = data["spec"]
    instruction_parts = [data["property"], data["title"], data["chapter"], data["section"]]
    instruction = ". ".join(filter(None, instruction_parts))
    
    result = [{
        "instruction": instruction,
        "input": "",
        "output": output,
        "system": "",
        "history": []
    }]
    
    return result

def process_jsonl(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
    
    result = []
    for line in lines:
        data = json.loads(line.strip())
        formatted_data = process_specification(data)
        result.extend(formatted_data)
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(result, outfile, indent=2, ensure_ascii=False)

# 使用示例
input_file = '20250124\output-llama3.1-405b-instruct.jsonl'
output_file = '20250124\output-llama3.1-qa.jsonl'


# 1、处理随机划分train和test
train_file = '20250124\单独训练-按照随机分配/llama_raw_train_0210.jsonl'  # 训练集输出路径
test_file = '20250124\单独训练-按照随机分配/llama_raw_test_0210.jsonl'    # 测试集输出路径
train_output_file = '20250124\单独训练-按照随机分配/llama_qa_train_rando_0210.jsonl'
test_output_file = '20250124\单独训练-按照随机分配/llama_qa_test_rando_0210.jsonl'
process_jsonl(train_file, train_output_file)
process_jsonl(test_file, test_output_file)

# 2、处理按照title划分train和test
train_file = '20250124\按照title划分数据集/llama_raw_train_data_210.jsonl'       # 训练集输出文件
test_file = '20250124\按照title划分数据集/llama_raw_test_data_210.jsonl'         # 测试集输出文件
train_output_file = '20250124\按照title划分数据集/llama_qa_train_title_210.jsonl'
test_output_file = '20250124\按照title划分数据集/llama_qa_test_title_210.jsonl'
process_jsonl(train_file, train_output_file)
process_jsonl(test_file, test_output_file)


