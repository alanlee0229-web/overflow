import os
from PIL import Image
import random

# 定义旋转角度（只包含90°、180°、270°）
ROTATION_ANGLES = [90, 180, 270]


def rotate_without_cropping(img, angle):
    """
    旋转图像但不裁剪，保持原始内容完整
    """
    # 旋转图像（expand=True确保所有内容都在图像内）
    rotated_img = img.rotate(angle, resample=Image.BICUBIC, expand=True)
    return rotated_img


def augment_images(input_dir, quadruple=True):
    """
    对目录中的所有JPG图像进行数据增强（旋转但不裁剪）

    参数:
        input_dir: 输入目录路径
        quadruple: 是否将图像数量翻四倍（默认为True）
    """
    # 遍历输入目录中的所有子文件夹和文件
    for root, dirs, files in os.walk(input_dir):
        print(f"正在处理文件夹: {root}")

        # 收集所有JPG文件
        jpg_files = [f for f in files if f.lower().endswith('.png')]
        print(f"找到 {len(jpg_files)} 个JPG文件")

        for file in jpg_files:
            # 构建输入文件路径
            img_path = os.path.join(root, file)

            print(f"处理文件: {img_path}")

            try:
                # 打开JPG文件
                image = Image.open(img_path).convert('RGB')

                if quadruple:
                    # 四倍增强模式：生成三个旋转版本（90°, 180°, 270°）
                    angles_to_process = ROTATION_ANGLES

                    # 为每个旋转角度生成新图像

                    for angle in angles_to_process:
                        # 旋转图像（不裁剪）
                        rotated_image = rotate_without_cropping(image, angle)

                        # 生成新文件名
                        name, ext = os.path.splitext(file)
                        new_filename = f"{name}_rot{angle}{ext}"
                        new_img_path = os.path.join(root, new_filename)

                        # 保存新图像
                        rotated_image.save(new_img_path)
                        print(f"已创建新图像: {new_img_path} (角度: {angle}°)")
                else:
                    # 单倍增强模式：覆盖原图
                    angle = random.choice(ROTATION_ANGLES)

                    # 旋转图像（不裁剪）
                    rotated_image = rotate_without_cropping(image, angle)

                    # 直接覆盖原图
                    rotated_image.save(img_path)
                    print(f"已覆盖原图: {img_path} (角度: {angle}°)")

            except Exception as e:
                print(f"处理文件 {img_path} 时出错: {str(e)}")
                continue


def get_user_input():
    """
    获取用户输入：目录路径和增强模式
    """
    print("=" * 50)
    print("图像数据增强工具")
    print("=" * 50)

    # 获取目录路径
    while True:
        input_dir = input("请输入包含JPG图像的目录路径: ")
        if os.path.exists(input_dir):
            break
        print(f"错误: 目录 '{input_dir}' 不存在，请重新输入")

    # 获取增强模式
    while True:
        mode = input("请选择增强模式 (1-四倍增强, 2-单倍增强): ")
        if mode in ['1', '2']:
            quadruple = (mode == '1')
            break
        print("错误: 请输入1或2")

    return input_dir, quadruple


# 主函数
if __name__ == "__main__":
    # 获取用户输入
    input_directory, quadruple_mode = get_user_input()
    # 显示配置信息
    print("\n配置信息:")
    print(f"输入目录: {input_directory}")
    print(f"增强模式: {'四倍增强（生成三个新图像）' if quadruple_mode else '单倍增强（覆盖原图）'}")

    # 确认操作（单倍增强会覆盖原图）
    if not quadruple_mode:
        confirm = input("\n警告: 单倍增强模式会覆盖原始图像！\n是否继续? (y/n): ")
        if confirm.lower() != 'y':
            print("操作已取消")
            exit()

    # 执行增强
    augment_images(input_directory, quadruple_mode)

    # 完成提示
    print("\n数据增强完成!")
    print("按Enter键退出...")
    input()

    #先dataset 再 数据增强