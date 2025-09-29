import json

def process_jsonl(input_file, train_file, test_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        train_data = []
        test_data = []
        
        for line in f:
            try:
                # 加载每一行的JSON数据
                data = json.loads(line.strip())
                
                # 判断title是否包含'spec/sep-abstract'
                if 'spec/sep-abstract' in data.get('title', ''):
                    test_data.append(data)
                else:
                    train_data.append(data)
            
            except json.JSONDecodeError:
                print(f"Warning: Failed to decode line: {line}")
        
        # 保存训练集数据
        with open(train_file, 'w', encoding='utf-8') as train_f:
            for entry in train_data:
                train_f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        # 保存测试集数据
        with open(test_file, 'w', encoding='utf-8') as test_f:
            for entry in test_data:
                test_f.write(json.dumps(entry, ensure_ascii=False) + '\n')

# 文件路径
# input_file = '20250124/output-qwen-max.jsonl'  # 这里替换为你的原始文件路径
# train_file = '20250124/qw_raw_train_data.jsonl'       # 训练集输出文件
# test_file = '20250124/qw_raw_test_data.jsonl'         # 测试集输出文件

input_file = '20250124/output-llama3.1-405b-instruct.jsonl'  # 这里替换为你的原始文件路径
train_file = '20250124\按照title划分数据集/llama_raw_train_data_210.jsonl'       # 训练集输出文件
test_file = '20250124\按照title划分数据集/llama_raw_test_data_210.jsonl'         # 测试集输出文件
# 调用处理函数
process_jsonl(input_file, train_file, test_file)

print("处理完成，训练集和测试集已保存。")
