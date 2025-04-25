import os
import csv
import re

# 定义文件目录
objective_dir = "test_data/subjective"

# 获取目录下所有的CSV文件
csv_files = [f for f in os.listdir(objective_dir) if f.endswith('.csv')]

# 处理每个CSV文件
for csv_file in csv_files:
    file_path = os.path.join(objective_dir, csv_file)
    print(f"处理文件: {file_path}")
    
    # 读取CSV文件内容
    rows = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows.append(header)
        
        # 确定Correct option列的索引
        correct_option_index = header.index('Correct option')
        
        # 处理每一行数据
        for row in reader:
            # 特殊处理Correct option列，将中文逗号替换回英文逗号
            if row[correct_option_index]:
                # 对多选题的选项进行处理，如"A， B， D" 替换为 "A, B, D"
                row[correct_option_index] = row[correct_option_index].replace('， ', ', ')
            
            rows.append(row)
    
    # 写回CSV文件
    with open(file_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    
    print(f"完成文件: {file_path}")

print("所有文件处理完成！") 