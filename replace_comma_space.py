import os
import re

# 定义文件目录
objective_dir = "test_data/subjective"

# 获取目录下所有的CSV文件
csv_files = [f for f in os.listdir(objective_dir) if f.endswith('.jsonl')]

# 处理每个CSV文件
for csv_file in csv_files:
    file_path = os.path.join(objective_dir, csv_file)
    print(f"处理文件: {file_path}")
    
    # 以字符串形式读取整个文件内容
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.split('\n')
    header = lines[0].split(',')
    
    
    # 处理每一行数据
    new_lines = []  # 保留原始标题行
    for i in range(0, len(lines)):
    
                # 新规则：将所有", "替换为"， "，不管后面跟什么字符
        lines[i] = re.sub(r', ([a-zA-Z])', r'， \1', lines[i])
        
        # 重新组合成一行
        new_lines.append(lines[i])
    
    # 重新组合成文件内容
    new_content = '\n'.join(new_lines)
    
    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"完成文件: {file_path}")

print("所有文件处理完成！") 