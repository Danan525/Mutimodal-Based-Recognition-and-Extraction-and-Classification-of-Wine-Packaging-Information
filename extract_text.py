import os
import easyocr

def extract_text_from_images(folder_path):
    # 初始化EasyOCR读取器，支持英文和简体中文
    reader = easyocr.Reader(['ch_sim', 'en'])  # 可以根据需要添加其他语言支持

    # 遍历文件夹中的所有图片文件
    for file in os.listdir(folder_path):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(folder_path, file)

            # 使用EasyOCR提取文本
            results = reader.readtext(image_path)

            # 输出图像文件名和提取的信息
            print(f"Image: {file}")
            print("Extracted Text:")
            for result in results:
                print(result[1])  # result[1] 是提取的文字内容
            print("\n" + "-"*50 + "\n")  # 分隔符，便于区分不同图片的输出

# 使用方法
folder_path = r"D:\package_detect\wine_labels_dataset\train"  # 替换为你的文件夹路径
extract_text_from_images(folder_path)
