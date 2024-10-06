import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from tkinter import filedialog, messagebox

def select_mat_file():
    """
    用来给用户选择一个mat信号文件
    """
    global mat_file_path
    mat_file_path = filedialog.askopenfilename(
        title="Select Electrical Signal .mat File",
        filetypes=[("MAT files", "*.mat")]
    )
    if not mat_file_path:
        messagebox.showerror("Error", "No MAT file selected. Exiting.")
        exit()

def select_image_folder():
    """
    用来给用户选择一个装有细胞图片的文件夹
    """
    global image_folder
    image_folder = filedialog.askdirectory(
        title="Select Optical Signal Image Folder"
    )
    if not image_folder:
        messagebox.showerror("Error", "No image folder selected. Exiting.")
        exit()

def onselect(eclick, erelease):
    """
    获取用户画的矩形框的大小、位置参数
    """
    global selected_region
    x1, y1 = int(eclick.xdata), int(eclick.ydata)
    x2, y2 = int(erelease.xdata), int(erelease.ydata)
    # Ensure correct coordinate order
    xmin, xmax = sorted([x1, x2])
    ymin, ymax = sorted([y1, y2])
    selected_region = (xmin, ymin, xmax, ymax)
    plt.close()

def get_user_selected_region(image):
    """
    给用户画一个矩形框，限定图片展示的范围
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image)
    ax.set_title("Please select a region on the image and then close the window.", fontsize=16)
    toggle_selector = RectangleSelector(
        ax, onselect, drawtype='box',
        useblit=True, button=[1],  # Only respond to left mouse button
        minspanx=5, minspany=5, spancoords='pixels',
        interactive=True
    )
    plt.show()
    return selected_region

def remove_horizontal_lines(data):
    """
    用于去除很横纹噪声
    """
    img = data.copy()
    if len(img.shape) == 2:  # Grayscale
        height, width = img.shape
        # Identify all identical rows
        identical_rows = np.all(img == img[:, [0]], axis=1)

        # 找到连续几行像素值一样的区域
        blocks = []
        in_block = False
        for y in range(height):
            if identical_rows[y]:
                if not in_block:
                    block_start = y
                    in_block = True
            else:
                if in_block:
                    block_end = y - 1
                    blocks.append((block_start, block_end))
                    in_block = False
        if in_block:
            blocks.append((block_start, height - 1))

        # 把像素值一样的区域用该区域以外的像素值替代
        for block_start, block_end in blocks:
            num_identical = block_end - block_start + 1
            # Determine the rows above and below the block
            upper_indices = np.arange(block_start - num_identical, block_start)
            lower_indices = np.arange(block_end + 1, block_end + 1 + num_identical)

            # Handle boundary conditions
            if upper_indices[0] < 0:
                upper_rows = np.tile(img[0, :], (num_identical, 1))
            else:
                upper_rows = img[upper_indices, :]

            if lower_indices[-1] >= height:
                lower_rows = np.tile(img[-1, :], (num_identical, 1))
            else:
                lower_rows = img[lower_indices, :]

            # 用平均值作为去噪后的像素值
            replacement = ((upper_rows.astype(np.float32) + lower_rows.astype(np.float32)) / 2).astype(img.dtype)
            img[block_start:block_end + 1, :] = replacement
    return img