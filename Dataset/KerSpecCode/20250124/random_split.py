import json
import random

def split_jsonl(input_file, train_file, test_file, test_ratio=0.1):
    # 打开输入的jsonl文件
    with open(input_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    # 随机打乱数据
    random.shuffle(lines)

    # 计算测试集大小
    test_size = int(len(lines) * test_ratio)

    # 将数据分为训练集和测试集
    test_data = lines[:test_size]
    train_data = lines[test_size:]

    # 将训练集保存到文件
    with open(train_file, 'w', encoding='utf-8') as train_f:
        train_f.writelines(train_data)

    # 将测试集保存到文件
    with open(test_file, 'w', encoding='utf-8') as test_f:
        test_f.writelines(test_data)

    print(f"训练集保存为 {train_file}")
    print(f"测试集保存为 {test_file}")

# 使用示例
input_file = '20250124\output-llama3.1-405b-instruct.jsonl'  # 输入的jsonl文件路径
train_file = '20250124\单独训练-按照随机分配/llama_raw_train_0210.jsonl'  # 训练集输出路径
test_file = '20250124\单独训练-按照随机分配/llama_raw_test_0210.jsonl'    # 测试集输出路径

split_jsonl(input_file, train_file, test_file)
