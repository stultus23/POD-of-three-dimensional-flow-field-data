# POD Flow 🌀

POD Flow 是一个面向 2D/3D 流场 `.DAT` 批量快照的 POD（Proper Orthogonal Decomposition）分析工具，提供可视化能量谱、模态云图、时间系数与频谱分析，并支持低阶重构导出。

## ✨ 功能亮点
- 批量导入 `.DAT` 快照（每个文件 = 一个时间快照）
- 支持列名表头或 DynamicStudio 风格头部
- POD 分解：能量占比、累计能量、模态云图、时间系数
- FFT 频谱分析（支持切换模态）
- 低阶重构流场并导出 DAT
- 结果导出可选择（云图 / 能量等 / 空间模态 / 时间系数 / FFT / 重构）

## ✅ 环境要求
- Python 3.11+
- Windows/macOS/Linux（推荐 Windows）

## 📦 依赖库（含文档）
- numpy: https://numpy.org/doc/
- scipy: https://docs.scipy.org/doc/scipy/
- matplotlib: https://matplotlib.org/stable/
- pandas: https://pandas.pydata.org/docs/

安装依赖：
```powershell
python -m pip install -e .
```

## ▶️ 启动
推荐双击启动：`run_pod_flow.bat`

或命令行启动：
```powershell
python -m app
```

## 🧭 使用指南（完整）

### 1) 导入数据
- 点击 **Add DAT** 选择一个或多个 `.dat` 文件
- 列表中每个文件代表一个时间快照
- 选中后点击 **Remove** 可删除

### 2) 解析参数（Parse）
- **Delimiter**：分隔符（默认空格）
- **Header rows**：表头行数
  - 如果第一行就是列名，填 `1`
  - 如果有 `TITLE/VARIABLES/ZONE`，一般是 `3`
- **Data variables**：要参与 POD 的变量名（逗号分隔）
  - 例如：`P` 或 `p` 或 `U[m/s],V[m/s],W[m/s]`
  - 只做压力 POD 时，建议只填 `P` 或 `p_data`
- **Detect Header**：自动识别列名与网格
- **Preview**：弹窗预览前 10 行

### 3) POD 参数（Compute）
- **Subtract mean**：减去均值（建议勾选）
- **Normalize**：归一化（一般不勾）
- **Modes (0=auto)**：模态数
  - 0 = 自动（推荐）
- **Method**：`SVD` 或 `randomized`
  - 大规模数据推荐 randomized
- **Sample freq (Hz)**：采样频率（影响 FFT 频率轴）

### 4) 计算与重构
- 点击 **Run POD** 开始计算
- 低阶重构：
  - 在 **Recon modes** 输入重构模态数
  - 点击 **Reconstruct now**
  - 未重构前，导出重构相关选项不可选

### 5) 结果展示
- 能量谱 / 累积能量
- FFT 频谱（可切换模态）
- 云图（可切换模态/变量/切片）

### 6) 导出结果
- 在 **Export Options** 勾选需要导出的内容
- 选择导出目录后点击 **Export Results**

## 📁 导出文件结构
默认导出包括能量与云图，可选导出空间模态、时间系数与 FFT、重构流场。

- `metadata.json`
- `modal_metrics_wide.csv`
- `clouds/mode_###.dat`
- `per_mode/mode_###/mode.csv`（可选）
- `per_mode/mode_###/coefficients.csv`（可选）
- `per_mode/mode_###/coefficients_fft.csv`（可选）
- `recon/recon_###.dat`（可选）

## ❓ 常见问题

**Q1: Preview 报错 “too many indices for array”**  
A: 通常是分隔符或 Header 行数错误，导致解析成一维。请确认：  
- Delimiter 是否为“空格”  
- Header rows 是否为正确行数  

**Q2: 只做压力 POD 要怎么填？**  
A: `Data variables` 只填 `P` 或 `p_data`，勾选 `Subtract mean`。  

**Q3: FFT 频谱数值很大？**  
A: 需要填写真实采样频率，且 FFT 取决于时间系数的量纲。  
