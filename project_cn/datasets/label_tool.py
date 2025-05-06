import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# 图片目录路径和标注结果保存文件
folder_path = r'/media/ntu/volume1/home/s124md306_06/project_cn/datasets/images/train'
save_txt = '/media/ntu/volume1/home/s124md306_06/project_cn/datasets/coordinates.txt'  # 保存标注结果到 coordinates.txt

# 如果保存的坐标文件不存在，则创建文件
if not os.path.exists(save_txt):
    with open(save_txt, 'w') as f:
        pass

# 读取目录中的所有图片文件
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

# 初始化全局变量
coordinates = []

# 标注图片的函数
def annotate_image(img_file):
    img_path = os.path.join(folder_path, img_file)
    
    # 读取图片
    img = plt.imread(img_path)
    
    # 创建一个图形和坐标轴
    fig, ax = plt.subplots()
    ax.imshow(img)

    # 用于存储矩形框的列表
    boxes = []

    # 事件回调：鼠标点击事件，用于绘制矩形框
    def on_click(event):
        nonlocal boxes
        if event.dblclick:  # 双击鼠标来选择一个矩形框的起始和结束点
            start_x, start_y = event.xdata, event.ydata
            rect = patches.Rectangle((start_x, start_y), 50, 50, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            boxes.append((start_x, start_y, start_x+50, start_y+50))  # 保存矩形框的坐标
            plt.draw()

    # 连接鼠标点击事件
    cid = fig.canvas.mpl_connect('button_press_event', on_click)
    
    plt.title(f'标注图片: {img_file}')
    plt.show()
    
    return boxes

# 遍历所有图片，逐一进行标注
for img_file in image_files:
    print(f"标注图片: {img_file}")
    
    # 获取标注框
    boxes = annotate_image(img_file)
    
    # 保存标注结果到文本文件
    if boxes:
        with open(save_txt, 'a') as f:
            f.write(f"{img_file}:<box>")
            for box in boxes:
                f.write(f"({box[0]}, {box[1]}),({box[2]}, {box[3]}) ")
            f.write("</box>\n")
        print(f"标注结果已保存到 {save_txt}")

    # 清理标注框并继续到下一张图片
    coordinates.clear()
