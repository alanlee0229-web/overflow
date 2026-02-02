import os
import shutil
import random

def split_two_level_dataset(root_dir, output_dir=None, train_ratio=0.8, seed=42):
    random.seed(seed)

    # 自动去除引号
    root_dir = root_dir.strip('"').strip("'")

    # 定义images和labels目录
    images_dir = os.path.join(root_dir, 'images')
    labels_dir = os.path.join(root_dir, 'labels')

    # 检查目录是否存在
    if not os.path.exists(images_dir):
        print(f"images目录 {images_dir} 不存在，请检查路径。")
        return False
    if not os.path.exists(labels_dir):
        print(f"labels目录 {labels_dir} 不存在，请检查路径。")
        return False

    # 获取所有图片文件，并提取文件名（不含扩展名）
    image_files = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
    image_names = [os.path.splitext(f)[0] for f in image_files]
    
    if not image_names:
        print("未找到任何图片文件")
        return False
    
    random.shuffle(image_names)

    # 计算分割点
    split_index = int(len(image_names) * train_ratio)
    train_names = image_names[:split_index]
    val_names = image_names[split_index:]

    # 根据训练集比例动态生成输出文件夹名称
    # 例如：0.8 → data82, 0.7 → data73
    train_percent = int(train_ratio * 10)  # 只取十位数字，0.8 → 8
    val_percent = 10 - train_percent       # 0.8 → 10-8=2
    output_folder_name = f'data{train_percent}{val_percent}'
    
    # 设置输出目录 - 在用户输入的数据集目录下
    if output_dir is None:
        output_dir = os.path.join(root_dir, output_folder_name)
    else:
        output_dir = output_dir.strip('"').strip("'")

    # 创建输出目录结构
    dirs_to_create = [
        os.path.join(output_dir, 'train', 'images'),
        os.path.join(output_dir, 'train', 'labels'),
        os.path.join(output_dir, 'val', 'images'),
        os.path.join(output_dir, 'val', 'labels')
    ]
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)

    # 复制文件的函数，避免重复代码
    def copy_files(names, subset_type):
        subset_dir = os.path.join(output_dir, subset_type)
        for name in names:
            # 复制图片
            found = False
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                img_src = os.path.join(images_dir, name + ext)
                if os.path.exists(img_src):
                    img_dst = os.path.join(subset_dir, 'images', name + ext)
                    shutil.copy2(img_src, img_dst)
                    found = True
                    break
            if not found:
                print(f"未找到图片文件: {name}")

            # 复制标签
            label_src = os.path.join(labels_dir, name + '.txt')
            if os.path.exists(label_src):
                label_dst = os.path.join(subset_dir, 'labels', name + '.txt')
                shutil.copy2(label_src, label_dst)
            else:
                print(f"未找到标签文件: {name}.txt")

    # 复制训练集文件
    copy_files(train_names, 'train')
    
    # 复制验证集文件
    copy_files(val_names, 'val')

    print(f"总样本数: {len(image_names)}")
    print(f"训练集: {len(train_names)} 个样本")
    print(f"验证集: {len(val_names)} 个样本")
    print(f"输出目录: {output_dir}")
    return True

if __name__ == '__main__':
    print("数据集分割工具")
    print("=" * 20)
    
    # 使用input函数获取输入
    root_dir = input("请输入数据集根目录路径（例如：F:/work_area/___overflow/train_data）: ").strip()
    
    # 可选参数，允许用户自定义
    try:
        train_ratio = float(input("请输入训练集比例（默认0.8）: ") or "0.8")
        if not 0 < train_ratio < 1:
            print("训练集比例必须在0-1之间，将使用默认值0.8")
            train_ratio = 0.8
    except ValueError:
        print("输入无效，将使用默认训练集比例0.8")
        train_ratio = 0.8
    
    try:
        seed = int(input("请输入随机种子（默认42）: ") or "42")
    except ValueError:
        print("输入无效，将使用默认随机种子42")
        seed = 42
    
    print("\n开始分割数据集...")
    success = split_two_level_dataset(
        root_dir=root_dir,
        train_ratio=train_ratio,
        seed=seed
    )
    
    if success:
        print("\n数据集分割完成！")
    else:
        print("\n数据集分割失败！")

'''
预期目录结构：
输入目录：
train_data/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ... (其他图片文件)
└── labels/
    ├── image1.txt
    ├── image2.txt
    └── ... (其他标签文件)

输出目录：
train_data/
├── images/
├── labels/
└── data82/
    ├── train/
    │   ├── images/
    │   │   ├── image1.jpg
    │   │   ├── image3.jpg
    │   │   └── ... (随机选择的80%图片)
    │   └── labels/
    │       ├── image1.txt
    │       ├── image3.txt
    │       └── ... (对应的标签文件)
    └── val/
        ├── images/
        │   ├── image2.jpg
        │   ├── image4.jpg
        │   └── ... (剩余的20%图片)
        └── labels/
            ├── image2.txt
            ├── image4.txt
            └── ... (对应的标签文件)

使用示例：
python dataset_single_class.py --input "F:/work_area/___overflow/train_data" --train_ratio 0.8 --seed 42
'''

