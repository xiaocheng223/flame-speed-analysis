# 火焰图像分析计算非拉伸层流燃烧速度

## 概述

本Python脚本处理恒容燃烧弹 (CVB) 中球形火焰的高速相机图像（.tif 格式），以确定非拉伸层流燃烧速度 ($\text{S}_{L,0}$) 和 Markstein 长度 ($\text{L}$)。它执行图像处理以提取火焰半径，平滑数据，计算火焰传播速度和拉伸率，最后应用 Markstein 线性回归。

## 功能

-   **图像处理**: 将 .tif 图像转换为灰度，应用 Otsu 阈值分割，使用轮廓检测识别火焰边界，并拟合圆形以物理单位提取火焰半径。
-   **数据预处理**: 使用 Savitzky–Golay 滤波器平滑提取的半径-时间数据。
-   **火焰速度计算**: 计算表观火焰传播速度 ($\text{S}_b$) 并将其转换为相对于未燃气体的火焰速度 ($\text{S}_u$)。
-   **拉伸率计算**: 确定火焰拉伸率 ($\text{K}$)。
-   **Markstein 回归**: 对 $\text{S}_u$ 与 $\text{K}$ 进行线性回归，以获得 $\text{S}_{L,0}$ (截距) 和 -$\text{L}$ (斜率)。
-   **可视化**: 生成原始和平滑火焰半径随时间变化的曲线图、$\text{S}_b$ 随时间变化的曲线图以及 $\text{S}_u$ 与拉伸率 $\text{K}$ 带有回归线的曲线图。
-   **数据输出**: 将所有处理过的数据（时间、原始半径、平滑半径、$\text{S}_b$、$\text{S}_u$、$\text{K}$）保存到 CSV 文件。

## 要求

-   Python 3.8+
-   以下 Python 包:
    -   `numpy`
    -   `pandas`
    -   `scipy`
    -   `matplotlib`
    -   `opencv-python` (cv2)

## 设置

1.  **创建虚拟环境** (推荐):

    ```bash
    python -m venv lfs
    ```

2.  **激活虚拟环境**:

    ```bash
    source lfs/bin/activate  # 在 Linux/macOS 上
    # lfs\Scripts\activate   # 在 Windows (Command Prompt) 上
    # lfs\Scripts\Activate.ps1 # 在 Windows (PowerShell) 上
    ```

3.  **安装依赖项**:

    ```bash
    pip install -r requirements.txt
    ```

## 使用方法

1.  **准备图像数据**:
    -   在 `flame_image_analysis.py` 脚本所在的目录中创建一个名为 `flame_images` 的文件夹。
    -   将您的球形火焰的高速相机 `.tif` 图像序列放入 `flame_images` 文件夹中。请确保它们按时间顺序命名（例如，`frame_0001.tif`、`frame_0002.tif` 等），以确保正确的时间排序。

2.  **配置参数**:
    -   打开 `flame_image_analysis.py` 并导航到 `main()` 函数。
    -   根据您的实验设置调整以下参数:
        -   `image_folder`: 图像文件夹的路径 (默认值: `'flame_images'`)。
        -   `calibration_factor`: 像素到米的转换因子 (例如，如果 1 像素 = 10 微米，则为 `1e-5`)。
        -   `frame_rate`: 高速相机的帧率，单位为帧/秒 (Hz)。
        -   `Tu`: 未燃气体的初始温度，单位为开尔文 (K)。
        -   `Tb`: 绝热火焰温度，单位为开尔文 (K)，来自平衡计算。
        -   `window_length`: Savitzky–Golay 滤波器的窗口长度。必须是奇数整数 (例如，`11`)。
        -   `polyorder`: Savitzky–Golay 滤波器的多项式阶数 (例如，`3`)。
        -   `output_csv_filepath`: 用于保存处理后数据的 CSV 文件名 (默认值: `'processed_flame_data.csv'`)。

3.  **运行脚本**:

    ```bash
    python flame_image_analysis.py
    ```

## 输出

成功执行后，脚本将:

-   在控制台打印计算出的 `非拉伸层流燃烧速度 (S_L0)` 和 `Markstein 长度 (L)`。
-   显示三张图:
    1.  **火焰半径 vs. 时间**: 显示原始和 Savitzky–Golay 平滑后的半径数据。
    2.  **表观火焰传播速度 (Sb) vs. 时间**。
    3.  **未燃火焰速度 (Su) vs. 拉伸率 (K)**: 包含数据点和 Markstein 关系的线性回归线。
-   创建一个名为 `processed_flame_data.csv` (或您指定的 `output_csv_filepath`) 的 CSV 文件，其中包含 `time`、`raw_radius`、`smoothed_radius`、`Sb`、`Su` 和 `K` 列。

## 模块化函数

脚本被组织成几个模块化函数，以提高清晰度和可维护性:

-   `extract_radius_from_images(image_folder, calibration_factor, frame_rate)`: 处理图像加载、处理（灰度、阈值、轮廓检测、圆形拟合），并将像素半径转换为物理半径。
-   `smooth_and_differentiate(time_data, radius_data, window_length, polyorder)`: 对半径数据应用 Savitzky–Golay 平滑，并计算导数以找出 `Sb`。
-   `compute_speeds_and_stretch(sb_data, smoothed_radius_data, Tu, Tb)`: 根据导出的 `Sb` 和平滑半径计算 `Su` 和 `K`。
-   `fit_markstein_relation(su_data, k_data)`: 对 `Su` 与 `K` 进行线性回归，以确定 `S_L0` 和 `L`。
-   `plot_results(time_data, raw_radius_data, smoothed_radius_data, sb_data, su_data, k_data, regression_line)`: 生成并显示所有必需的图表。
-   `save_data_to_csv(filepath, time_data, raw_radius_data, smoothed_radius_data, sb_data, su_data, k_data)`: 将完整的处理后数据保存到 CSV 文件。
