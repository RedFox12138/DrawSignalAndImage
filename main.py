import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os
from matplotlib.widgets import Slider, Button, CheckButtons
import tkinter as tk
from tkinter import messagebox
from PIL import Image

from Utils import select_mat_file, select_image_folder, remove_horizontal_lines, get_user_selected_region

Image.MAX_IMAGE_PIXELS = None
root = tk.Tk()
root.withdraw()  # Hide the root window

# 全局设置
selected_region = None
mat_file_path = None
image_folder = None
spike_detection_enabled = False  # 尖峰检测开关，默认为关

def main():
    global selected_region, mat_file_path, image_folder, spike_detection_enabled

    # ==========================
    # 1. 选择电信号文件与光信号文件
    # ==========================
    select_mat_file()
    select_image_folder()

    # ==========================
    # 2. 读取电信号文件
    # ==========================
    if not os.path.exists(mat_file_path):
        messagebox.showerror("Error", f"MAT file not found: {mat_file_path}")
        exit()
    signal_data = scipy.io.loadmat(mat_file_path)
    if 'signal' not in signal_data:
        messagebox.showerror("Error", "'all_data' not found in the MAT file.")
        exit()
    signal = signal_data['signal'][0, :]  # Assuming 'all_data' is a 2D array

    # 计算信号的 RMS
    signal_rms = np.sqrt(np.mean(signal ** 2))
    print(f"Signal RMS: {signal_rms}")

    # 根据 RMS 设定 y 轴初始范围
    rms_multiplier = 3  # 可根据信号的实际大小调整倍数
    fixed_ymin = -rms_multiplier * signal_rms
    fixed_ymax = rms_multiplier * signal_rms
    print(f"Initial y-axis limits set to: ymin={fixed_ymin}, ymax={fixed_ymax}")

    Fs = 39065.5  # 采样率
    signal_length = len(signal)
    t = np.arange(signal_length) / Fs  # 时间轴长度
    window_size_initial = 1  # 展示窗长的初始值
    step_size = 1  # 展示步长的初始值

    # 双阈值法的初始值
    initial_positive_spike_thresh = 10
    initial_negative_spike_thresh = -10
    initial_positive_noise_thresh = 200
    initial_negative_noise_thresh = -200

    # ==============================
    # 3. 一次性读取所有光信号
    # ==============================
    if not os.path.exists(image_folder):
        messagebox.showerror("Error", f"Image folder not found: {image_folder}")
        exit()

    # 读取所有图像并处理
    image_paths = sorted(glob.glob(os.path.join(image_folder, "*.bmp")))
    num_images = len(image_paths)

    if num_images == 0:
        messagebox.showerror("Error",
                             "No BMP images found in the specified folder. Please check the path and file format.")
        exit()

    # ===================================
    # 4. 用户框选展示区域
    # ===================================
    first_image = mpimg.imread(image_paths[0])
    first_image = remove_horizontal_lines(first_image)
    selected_region = get_user_selected_region(first_image)
    if selected_region is None:
        messagebox.showerror("Error", "No region selected. The program will exit.")
        exit()

    xmin, ymin, xmax, ymax = selected_region
    print(f"Selected region coordinates: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}")

    # 预处理所有图像
    cropped_images = []
    for path in image_paths:
        img = mpimg.imread(path)
        img = remove_horizontal_lines(img[ymin:ymax, xmin:xmax])
        cropped_images.append(img)


    # ===================================
    # 5. 去噪相关的参数
    # ===================================
    noise_duration_threshold = 0.1  # 噪声持续时间
    samples_per_noise_duration = int(noise_duration_threshold * Fs)
    zero_duration_seconds = 25  # 如果碰到长度为0.1s的噪声，直接将后续25s的信号全部归零，这是拍摄引起的
    zero_duration_samples = int(zero_duration_seconds * Fs)

    # ===================================
    # 6. 绘图
    # ===================================
    plt.style.use('seaborn-darkgrid')
    plt.ion()  # Enable interactive mode
    fig = plt.figure(figsize=(18, 14), constrained_layout=True)  # 增大图形尺寸并使用约束布局
    gs = fig.add_gridspec(6, 2, height_ratios=[1.5, 2.5, 1, 0.5, 0.5, 0.5], hspace=0.5, width_ratios=[1, 1])

    # 使用GridSpec重新布局各个子图和控件
    ax_image = fig.add_subplot(gs[0:2, 0])  # 图像占据前两行
    ax_signal = fig.add_subplot(gs[0:2, 1])  # 信号图占据前两行

    # 初始化光信号图（只显示选定区域）
    current_image_index = 0
    img = mpimg.imread(image_paths[current_image_index])
    img = remove_horizontal_lines(img)  # Remove horizontal line interference
    cropped_img = img[ymin:ymax, xmin:xmax]
    img_display = ax_image.imshow(cropped_img, cmap='gray_r')
    ax_image.axis('off')
    ax_image.set_title(f'Optical Signal Image {current_image_index + 1}/{num_images}', fontsize=18)

    #current_image_index = -1

    # 初始化电信号图
    window_size = window_size_initial  # Current window size
    start_index = 0
    end_index = int(window_size * Fs)
    h_plot, = ax_signal.plot(t[start_index:end_index], signal[start_index:end_index], linewidth=2, color='#1f77b4', label='Electrical Signal')
    h_marks, = ax_signal.plot([], [], 'r*', markersize=10, label='Spikes')  # Initially empty markers
    h_upper_spike_line, = ax_signal.plot(t[start_index:end_index], initial_positive_spike_thresh * np.ones(end_index - start_index), 'g--', linewidth=1, label='Spike Thresholds')
    h_lower_spike_line, = ax_signal.plot(t[start_index:end_index], initial_negative_spike_thresh * np.ones(end_index - start_index), 'g--', linewidth=1)
    h_upper_noise_line, = ax_signal.plot(t[start_index:end_index], initial_positive_noise_thresh * np.ones(end_index - start_index), 'orange', linewidth=1, label='Noise Thresholds')
    h_lower_noise_line, = ax_signal.plot(t[start_index:end_index], initial_negative_noise_thresh * np.ones(end_index - start_index), 'orange', linewidth=1)

    ax_signal.set_xlim([t[start_index], t[end_index - 1]])
    # 初始化 y 轴范围为固定值
    ax_signal.set_ylim([fixed_ymin, fixed_ymax])
    ax_signal.set_xlabel('Time (seconds)', fontsize=16)
    ax_signal.set_ylabel('Signal/μV', fontsize=16)
    ax_signal.set_title('Dynamic Electrical Signal Display', fontsize=20)
    ax_signal.legend(loc='upper right', fontsize=12)

    # Set colors and line styles
    h_upper_spike_line.set_color('#2ca02c')  # Green dashed line
    h_lower_spike_line.set_color('#2ca02c')
    h_upper_noise_line.set_color('#ff7f0e')  # Orange solid line
    h_lower_noise_line.set_color('#ff7f0e')

    # 控件区域：滑块和按钮
    control_ax = fig.add_subplot(gs[2:, :])  # 控件区域占据剩余所有行
    control_ax.axis('off')  # 隐藏坐标轴

    # 定义滑块的位置和尺寸
    # 采用相对布局，分为两列，每列三个滑块
    slider_height = 0.03
    slider_spacing_y = 0.05
    slider_width = 0.35
    left_x = 0.05
    right_x = 0.55
    start_y = 0.15

    # 创建滑块
    slider_speed_ax = fig.add_axes([left_x, start_y + slider_spacing_y * 2, slider_width, slider_height], facecolor='#f0f0f0')
    slider_window_ax = fig.add_axes([right_x, start_y + slider_spacing_y * 2, slider_width, slider_height], facecolor='#f0f0f0')
    slider_positive_spike_ax = fig.add_axes([left_x, start_y + slider_spacing_y, slider_width, slider_height], facecolor='#f0f0f0')
    slider_negative_spike_ax = fig.add_axes([left_x, start_y, slider_width, slider_height], facecolor='#f0f0f0')
    slider_positive_noise_ax = fig.add_axes([right_x, start_y + slider_spacing_y, slider_width, slider_height], facecolor='#f0f0f0')
    slider_negative_noise_ax = fig.add_axes([right_x, start_y, slider_width, slider_height], facecolor='#f0f0f0')

    # 定义滑块
    slider_speed = Slider(
        ax=slider_speed_ax,
        label='Update Speed (s)',
        valmin=0.001,
        valmax=5,
        valinit=5,
        valstep=0.01,
        color='#3b5998'
    )

    slider_window = Slider(
        ax=slider_window_ax,
        label='Signal Window (s)',
        valmin=0.1,
        valmax=10,
        valinit=window_size_initial,
        valstep=1,
        color='#3b5998'
    )

    slider_positive_spike = Slider(
        ax=slider_positive_spike_ax,
        label='Positive Spike Thresh',
        valmin=1,
        valmax=1000,
        valinit=initial_positive_spike_thresh,
        valstep=1,
        color='#3b5998'
    )

    slider_negative_spike = Slider(
        ax=slider_negative_spike_ax,
        label='Negative Spike Thresh',
        valmin=-1000,
        valmax=-1,
        valinit=initial_negative_spike_thresh,
        valstep=1,
        color='#3b5998'
    )

    slider_positive_noise = Slider(
        ax=slider_positive_noise_ax,
        label='Positive Noise Thresh',
        valmin=10,
        valmax=300,
        valinit=initial_positive_noise_thresh,
        valstep=10,
        color='#3b5998'
    )

    slider_negative_noise = Slider(
        ax=slider_negative_noise_ax,
        label='Negative Noise Thresh',
        valmin=-300,
        valmax=-10,
        valinit=initial_negative_noise_thresh,
        valstep=10,
        color='#3b5998'
    )

    # 添加重置按钮
    # 将按钮放在控件区域的中心下方
    reset_button_ax = fig.add_axes([0.45, 0.02, 0.1, 0.04], facecolor='#f0f0f0')
    button_reset = Button(reset_button_ax, 'Reset', color='#4CAF50', hovercolor='#45a049')

    def reset(event):
        slider_speed.reset()
        slider_window.reset()
        slider_positive_spike.reset()
        slider_negative_spike.reset()
        slider_positive_noise.reset()
        slider_negative_noise.reset()
        spike_detection_toggle.set_active(False)  # 重置尖峰检测开关为关闭

    button_reset.on_clicked(reset)

    # 定义滑块更新函数
    def update_params(val):
        nonlocal window_size, start_index, end_index
        window_size = slider_window.val
        # 重新计算 end_index
        end_index = start_index + int(window_size * Fs)
        if end_index > signal_length:
            end_index = signal_length
        # 更新电信号图
        h_plot.set_data(t[start_index:end_index], signal[start_index:end_index])
        # 获取当前阈值
        pos_spike = slider_positive_spike.val
        neg_spike = slider_negative_spike.val
        pos_noise = slider_positive_noise.val
        neg_noise = slider_negative_noise.val
        # 更新阈值线
        h_upper_spike_line.set_data(t[start_index:end_index], pos_spike * np.ones(end_index - start_index))
        h_lower_spike_line.set_data(t[start_index:end_index], neg_spike * np.ones(end_index - start_index))
        h_upper_noise_line.set_data(t[start_index:end_index], pos_noise * np.ones(end_index - start_index))
        h_lower_noise_line.set_data(t[start_index:end_index], neg_noise * np.ones(end_index - start_index))
        ax_signal.set_xlim([t[start_index], t[end_index - 1]])
        ax_signal.figure.canvas.draw_idle()

    # 连接滑块到更新函数
    slider_speed.on_changed(update_params)
    slider_window.on_changed(update_params)
    slider_positive_spike.on_changed(update_params)
    slider_negative_spike.on_changed(update_params)
    slider_positive_noise.on_changed(update_params)
    slider_negative_noise.on_changed(update_params)

    # ============================
    # 7. 添加尖峰检测开关
    # ============================
    # 定义CheckButtons的位置和大小
    check_button_ax = fig.add_axes([0.85, 0.55, 0.12, 0.04], facecolor='white')
    spike_detection_toggle = CheckButtons(check_button_ax, ['Enable Spike Detection'], [spike_detection_enabled])

    def toggle_spike_detection(label):
        global spike_detection_enabled
        spike_detection_enabled = not spike_detection_enabled
        print(f"Spike Detection Enabled: {spike_detection_enabled}")
        # 当关闭尖峰检测时，清空现有的尖峰标记
        if not spike_detection_enabled:
            h_marks.set_data([], [])

    spike_detection_toggle.on_clicked(toggle_spike_detection)

    # 初始化 y 轴范围变量
    current_ylim = [fixed_ymin, fixed_ymax]

    def on_click(event):
        nonlocal current_ylim
        if event.inaxes != ax_signal:
            return  # 只响应信号图区域的点击

        if event.button == 1:  # 左键点击，拉伸 y 轴
            scale_factor = 1.1  # 每次拉伸 10%
            new_ylim = [current_ylim[0] * scale_factor, current_ylim[1] * scale_factor]
            # 可选：设置一个最大限制
            max_limit = 1e6
            if abs(new_ylim[0]) > max_limit or abs(new_ylim[1]) > max_limit:
                return
            current_ylim = new_ylim
            ax_signal.set_ylim(current_ylim)
            ax_signal.figure.canvas.draw_idle()
            print(f"Y轴范围拉伸到: {current_ylim}")

        elif event.button == 3:  # 右键点击，压缩 y 轴
            scale_factor = 1 / 1.1  # 每次压缩约 9%
            new_ylim = [current_ylim[0] * scale_factor, current_ylim[1] * scale_factor]
            # 可选：设置一个最小限制
            min_limit = 1e-3
            if abs(new_ylim[0]) < min_limit or abs(new_ylim[1]) < min_limit:
                return
            current_ylim = new_ylim
            ax_signal.set_ylim(current_ylim)
            ax_signal.figure.canvas.draw_idle()
            print(f"Y轴范围压缩到: {current_ylim}")

    # 连接鼠标点击事件到 on_click 函数
    fig.canvas.mpl_connect('button_press_event', on_click)

    # ===================================
    # 8. 更新光信号
    # ===================================
    # Initialize variables for noise duration detection
    #noise_counter = 0
    #image_switch_triggered = False
    #noise_event_count = 0  # 记录噪声事件次数
    zero_mask = np.zeros(signal_length, dtype=bool)  # 标记需要归零的样本
    #last_zero_end = -1  # 记录上一次归零结束的索引

    overlap = 2500  # 设定重叠部分的大小（例如2500个点）
    total_time = signal_length / Fs  # 信号总时长（秒）
    time_per_image = total_time / num_images  # 每张图片显示的时长
    # 初始化
    current_image_index = 0
    last_switch_time = 0  # 记录上次切换的时间

    while start_index < signal_length:
        current_data = signal[start_index:end_index]

        # 获取当前阈值
        pos_spike_thresh = slider_positive_spike.val
        neg_spike_thresh = slider_negative_spike.val
        pos_noise_thresh = slider_positive_noise.val
        neg_noise_thresh = slider_negative_noise.val

        # 噪声检测
        noise_mask_current = (current_data > pos_noise_thresh) | (current_data < neg_noise_thresh)
        absolute_start = start_index
        noise_indices = np.where(noise_mask_current)[0]

        # 计算当前显示时间
        current_time = start_index / Fs  # 当前显示的时间（秒）

        # 内层循环处理图片切换
        while current_time - last_switch_time >= time_per_image:
            last_switch_time += time_per_image  # 更新上次切换的时间
            current_image_index = (current_image_index + 1) % num_images
            img_display.set_data(cropped_images[current_image_index])
            ax_image.set_title(f'Optical Signal Image {current_image_index + 1}/{num_images}', fontsize=18)
            plt.pause(0.01)

        # 处理噪声
        for idx in noise_indices:
            absolute_idx = absolute_start + idx
            if not zero_mask[absolute_idx]:
                zero_start = absolute_idx
                zero_end = min(absolute_idx + zero_duration_samples, signal_length)
                zero_mask[zero_start:zero_end] = True

        filtered_data = current_data.copy()
        filtered_data[zero_mask[start_index:end_index]] = 0

        # 尖峰检测
        if spike_detection_enabled:
            spike_indices = np.concatenate((
                np.where(filtered_data > pos_spike_thresh)[0],
                np.where(filtered_data < neg_spike_thresh)[0]
            ))

            valid_spikes = []
            for spike_idx in spike_indices:
                window_start = spike_idx - 1000
                window_end = spike_idx + 1000
                window_spikes = spike_indices[(spike_indices >= window_start) & (spike_indices <= window_end)]
                if len(window_spikes) > 0:
                    max_spike_idx = window_spikes[np.argmax(np.abs(filtered_data[window_spikes]))]
                    if max_spike_idx not in valid_spikes:
                        valid_spikes.append(max_spike_idx)

            valid_spikes = np.array(valid_spikes).astype(int)
            h_marks.set_data(t[start_index + valid_spikes], filtered_data[valid_spikes])
        else:
            h_marks.set_data([], [])

        # 更新电信号图数据
        h_plot.set_data(t[start_index:end_index], filtered_data)
        h_upper_spike_line.set_data(t[start_index:end_index], pos_spike_thresh * np.ones(end_index - start_index))
        h_lower_spike_line.set_data(t[start_index:end_index], neg_spike_thresh * np.ones(end_index - start_index))
        h_upper_noise_line.set_data(t[start_index:end_index], pos_noise_thresh * np.ones(end_index - start_index))
        h_lower_noise_line.set_data(t[start_index:end_index], neg_noise_thresh * np.ones(end_index - start_index))

        ax_signal.set_xlim([t[start_index], t[end_index - 1]])

        # 刷新图形
        fig.canvas.draw_idle()
        plt.pause(slider_speed.val)

        # 更新索引，加入重叠部分
        start_index += int(step_size * Fs) - overlap
        end_index += int(step_size * Fs) - overlap
        if end_index > signal_length:
            end_index = signal_length

        # 确保索引不超过信号长度
        if start_index + window_size > signal_length:
            start_index = signal_length - window_size
            end_index = signal_length

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()

   # while start_index < signal_length and current_image_index < num_images:
    #     current_data = signal[start_index:end_index]
    #
    #     # 获取当前阈值
    #     pos_spike_thresh = slider_positive_spike.val
    #     neg_spike_thresh = slider_negative_spike.val
    #     pos_noise_thresh = slider_positive_noise.val
    #     neg_noise_thresh = slider_negative_noise.val
    #
    #     # 噪声检测和处理
    #     # noise_mask_current = (current_data > pos_noise_thresh) | (current_data < neg_noise_thresh)
    #     # absolute_start = start_index
    #     # noise_indices = np.where(noise_mask_current)[0]
    #     # for idx in noise_indices:
    #     #     absolute_idx = absolute_start + idx
    #     #     if not zero_mask[absolute_idx]:
    #     #         zero_start = absolute_idx
    #     #         zero_end = min(absolute_idx + zero_duration_samples, signal_length)
    #     #         zero_mask[zero_start:zero_end] = True
    #     #
    #     #         # 切换图像
    #     #         current_image_index = (current_image_index + 1) % num_images
    #     #         img_display.set_data(cropped_images[current_image_index])
    #     #         ax_image.set_title(f'Optical Signal Image {current_image_index + 1}/{num_images}', fontsize=18)
    #     #
    #     #         noise_event_count += 1
    #     #
    #     # filtered_data = current_data.copy()
    #     # filtered_data[zero_mask[start_index:end_index]] = 0
    #
    #
    #     noise_mask_current = (current_data > pos_noise_thresh) | (current_data < neg_noise_thresh)
    #     absolute_start = start_index
    #     noise_indices = np.where(noise_mask_current)[0]
    #
    #     # 计算当前显示时间
    #     current_time = start_index / Fs  # 当前显示的时间（秒）
    #
    #     # 检查是否需要切换图片
    #     if current_time - last_switch_time >= time_per_image:
    #         last_switch_time = current_time  # 更新上次切换的时间
    #         current_image_index = (current_image_index + 1) % num_images
    #         img_display.set_data(cropped_images[current_image_index])
    #         ax_image.set_title(f'Optical Signal Image {current_image_index + 1}/{num_images}', fontsize=18)
    #
    #
    #     for idx in noise_indices:
    #         absolute_idx = absolute_start + idx
    #         if not zero_mask[absolute_idx]:
    #             zero_start = absolute_idx
    #             zero_end = min(absolute_idx + zero_duration_samples, signal_length)
    #             zero_mask[zero_start:zero_end] = True
    #             noise_event_count += 1
    #
    #     filtered_data = current_data.copy()
    #     filtered_data[zero_mask[start_index:end_index]] = 0
    #
    #     # 尖峰检测
    #     if spike_detection_enabled:
    #         spike_indices = np.concatenate((
    #             np.where(filtered_data > pos_spike_thresh)[0],
    #             np.where(filtered_data < neg_spike_thresh)[0]
    #         ))
    #
    #         valid_spikes = []
    #         for spike_idx in spike_indices:
    #             window_start = spike_idx - 5000
    #             window_end = spike_idx + 5000
    #             window_spikes = spike_indices[(spike_indices >= window_start) & (spike_indices <= window_end)]
    #             if len(window_spikes) > 0:
    #                 max_spike_idx = window_spikes[np.argmax(np.abs(filtered_data[window_spikes]))]
    #                 if max_spike_idx not in valid_spikes:
    #                     valid_spikes.append(max_spike_idx)
    #
    #         valid_spikes = np.array(valid_spikes).astype(int)
    #         h_marks.set_data(t[start_index + valid_spikes], filtered_data[valid_spikes])
    #     else:
    #         h_marks.set_data([], [])
    #
    #     # 更新电信号图数据
    #     h_plot.set_data(t[start_index:end_index], filtered_data)
    #     h_upper_spike_line.set_data(t[start_index:end_index], pos_spike_thresh * np.ones(end_index - start_index))
    #     h_lower_spike_line.set_data(t[start_index:end_index], neg_spike_thresh * np.ones(end_index - start_index))
    #     h_upper_noise_line.set_data(t[start_index:end_index], pos_noise_thresh * np.ones(end_index - start_index))
    #     h_lower_noise_line.set_data(t[start_index:end_index], neg_noise_thresh * np.ones(end_index - start_index))
    #
    #     ax_signal.set_xlim([t[start_index], t[end_index - 1]])
    #
    #     # 刷新图形
    #     fig.canvas.draw_idle()
    #     plt.pause(slider_speed.val)
    #
    #     # 更新索引，加入重叠部分
    #     start_index += int(step_size * Fs) - overlap
    #     end_index += int(step_size * Fs) - overlap
    #     if end_index > signal_length:
    #         end_index = signal_length
    #
    #     # 确保索引不超过信号长度
    #     if start_index + window_size > signal_length:
    #         start_index = signal_length - window_size
    #         end_index = signal_length
    #
    # plt.ioff()
    # plt.show()
    # 初始化时间变量