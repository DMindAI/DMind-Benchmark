import os
import re

# 定义文件目录
objective_dir = "test_data/objective"

# 获取目录下所有的CSV文件
csv_files = [f for f in os.listdir(objective_dir) if f.endswith('.csv')]

# 处理每个CSV文件
for csv_file in csv_files:
    file_path = os.path.join(objective_dir, csv_file)
    print(f"处理文件: {file_path}")
    
    # 以字符串形式读取整个文件内容
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # 使用正则表达式替换英文逗号+空格+字母为中文逗号+空格+字母
    # 注意：避免替换Correct option列中的内容
    lines = content.split('\n')
    header = lines[0].split(',')
    
    # 找到"Correct option"列的索引
    correct_option_index = -1
    for i, col in enumerate(header):
        if col.strip() == 'Correct option':
            correct_option_index = i
            break
    
    if correct_option_index == -1:
        print("警告：找不到'Correct option'列，将处理所有列")
    
    # 处理每一行数据
    new_lines = [lines[0]]  # 保留原始标题行
    for i in range(1, len(lines)):
        if not lines[i].strip():
            new_lines.append(lines[i])
            continue
            
        # 将行拆分为字段
        fields = lines[i].split(',')
        
        # 处理每个字段，除了Correct option列
        for j in range(len(fields)):
            if j != correct_option_index:
                # 将", X"替换为"， X"（其中X是任何字母）
                fields[j] = re.sub(r', ([a-zA-Z])', r'， \1', fields[j])
        
        # 重新组合成一行
        new_lines.append(','.join(fields))
    
    # 重新组合成文件内容
    new_content = '\n'.join(new_lines)
    
    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"完成文件: {file_path}")

print("所有文件处理完成！") 