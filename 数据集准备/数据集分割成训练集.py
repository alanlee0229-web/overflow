import os
import shutil
import random

def split_dataset(root_dir, output_dir=None, train_ratio=0.8, seed=42):
    random.seed(seed)

    if output_dir is None:
        output_dir = os.path.join(root_dir, 'dataset')
    else:
        output_dir = os.path.abspath(output_dir)

    class_dirs = ['0', '1']
    for class_name in class_dirs:
        class_path = os.path.join(root_dir, class_name)
        if not os.path.exists(class_path):
            print(f"类文件夹 {class_path} 不存在，跳过。")
            continue

        images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
        random.shuffle(images)

        split_index = int(len(images) * train_ratio)
        train_files = images[:split_index]
        val_files = images[split_index:]

        for split, files in [('train', train_files), ('val', val_files)]:
            split_class_dir = os.path.join(output_dir, split, class_name)
            os.makedirs(split_class_dir, exist_ok=True)

            for f in files:
                src = os.path.join(class_path, f)
                dst = os.path.join(split_class_dir, f)
                shutil.copy2(src, dst)

        print(f"类 {class_name}: 训练集 {len(train_files)} 张，验证集 {len(val_files)} 张")

if __name__ == '__main__':
    input_dir = r"F:\work_area\___overflow\data_img\data_demo_done1&4\cut_data0&1"  # 源数据目录，包含0和1两个类
    output_dir = os.path.join(input_dir, 'dataset_Data' )  # 输出目录就在原始目录下
    split_dataset(input_dir, output_dir=output_dir, train_ratio=0.8)


'''
├── 0
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ... (其他图片文件)
├── 1
│   ├── image101.jpg
│   ├── image102.jpg
│   └── ... (其他图片文件)
└── dataset_Data
    ├── train
    │   ├── 0
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   └── ... (随机选择的80%图片)
    │   └── 1
    │       ├── image101.jpg
    │       ├── image102.jpg
    │       └── ... (随机选择的80%图片)
    └── val
        ├── 0
        │   ├── image3.jpg
        │   ├── image4.jpg
        │   └── ... (剩余的20%图片)
        └── 1
            ├── image103.jpg
            ├── image104.jpg
            └── ... (剩余的20%图片)'''