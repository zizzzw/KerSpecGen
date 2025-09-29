import json

def merge_jsonl(file1, file2, output_file):
    # 打开两个输入文件和一个输出文件
    with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2, open(output_file, 'w', encoding='utf-8') as out:
        # 读取文件内容并解析为字典
        data1 = f1.readlines()
        data2 = f2.readlines()
        
        # 合并两个文件的内容
        merged_data = data1 + data2
        
        # 写入到输出文件
        for item in merged_data:
            out.write(item)


# 训练数据集合并
# merge_jsonl('20250124/qw_raw_test_data.jsonl', '20250124/llama_raw_test_data.jsonl', '20250124/merged_raw_test_data.jsonl')
# 测试数据集合并
merge_jsonl('20250124/output-llama3.1-405b-instruct.jsonl', '20250124/output-qwen-max.jsonl', '20250124/merged_raw_data.jsonl')
