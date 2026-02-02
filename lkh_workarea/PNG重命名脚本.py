import os
import re

def rename_files(folder_path, naming_rule):
    """
    按照指定规则重命名文件夹中的所有PNG文件
    
    参数:
        folder_path: 包含PNG文件的文件夹路径
        naming_rule: 命名规则（如'22_'、'5'或'50&'）
    """
    # 获取所有PNG文件
    png_files = [f for f in os.listdir(folder_path) 
                if os.path.isfile(os.path.join(folder_path, f)) 
                and f.lower().endswith('.png')]
    
    # 按文件名排序（自然排序）
    png_files.sort(key=lambda f: [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', f)])
    
    # 计数器
    count = 1
    
    # 重命名所有文件
    for filename in png_files:
        # 构建新文件名
        new_name = f"{naming_rule}{count}.png"
        
        # 源文件路径和目标文件路径
        src = os.path.join(folder_path, filename)
        dst = os.path.join(folder_path, new_name)
        
        # 重命名文件
        os.rename(src, dst)
        print(f"重命名: {filename} -> {new_name}")
        
        # 增加计数器
        count += 1

if __name__ == '__main__':
    # 获取用户输入
    folder_path = input("请输入包含PNG文件的文件夹路径: ").strip().strip('"')

    naming_rule = input("请输入命名规则（如'name_'则输出'name_1'、'name_2'...）\n")
    
    # "C:\Users\alanlee\Desktop\q"
    if not os.path.exists(folder_path):
        print(f"错误: 路径 '{folder_path}' 不存在")
        exit(1)
    
    # 执行重命名
    print(f"\n开始重命名 {folder_path} 中的PNG文件...")
    rename_files(folder_path, naming_rule)
    print("\n重命名完成！")
    