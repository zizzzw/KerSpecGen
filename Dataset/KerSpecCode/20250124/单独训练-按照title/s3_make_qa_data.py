import json

def process_jsonl_to_json(input_file, output_file):
    output_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                # 解析每一行的JSON数据
                data = json.loads(line.strip())
                
                # 提取需要的字段，并整理成指定格式
                instruction = data.get('property', '')
                output = data.get('spec', '')
                
                formatted_entry = {
                    "instruction": instruction,
                    "input": "",  # 根据你的要求，"input"字段为空
                    "output": output,
                    "system": "",  # "system"字段为空
                    "history": []  # "history"字段为空
                }
                
                output_data.append(formatted_entry)
            
            except json.JSONDecodeError:
                print(f"Warning: Failed to decode line: {line}")
    
    # 将整理后的数据保存为JSON文件
    with open(output_file, 'w', encoding='utf-8') as out_f:
        json.dump(output_data, out_f, ensure_ascii=False, indent=2)

# 训练集变为问答数据集
# input_file = '20250124/单独训练/llama_raw_train_data.jsonl'  # 这里替换为你的原始文件路径
# output_file = '20250124/单独训练/llama_qa_train_data.json' # 输出的整理好的JSON文件
# input_file = '20250124/单独训练/qw_raw_train_data.jsonl'  # 这里替换为你的原始文件路径
# output_file = '20250124/单独训练/qw_qa_train_data.json' # 输出的整理好的JSON文件
#测试集变为问答数据集
# input_file = '20250124/单独训练/llama_raw_test_data.jsonl'  # 这里替换为你的原始文件路径
# output_file = '20250124/单独训练/llama_qa_test_data.jsonl' # 输出的整理好的JSON文件
input_file = '20250124/单独训练/qw_raw_test_data.jsonl'  # 这里替换为你的原始文件路径
output_file = '20250124/单独训练/qw_qa_test_data.json' # 输出的整理好的JSON文件

# 调用函数
process_jsonl_to_json(input_file, output_file)

print("处理完成，已保存为JSON文件。")
