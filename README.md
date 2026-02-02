# POD Flow

POD Flow 是一个面向 3D 流场 `.DAT` 批量快照的 POD 分析工具。

## 安装

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m ensurepip --upgrade
python -m pip install --upgrade pip
python -m pip install -e .
```

## 启动

双击 `run_pod_flow.bat`，或运行：

```powershell
python -m app
```

## Tk UI 操作指南（中文）

### 1) 载入数据
- Add DAT：选择一个或多个 `.dat` 文件（每个文件 = 一个快照）
- Remove：移除选中的文件

### 2) 解析参数
- Delimiter：分隔符（默认空格）
- Header rows：头部行数（默认 3）
- Has coordinates：是否包含 X/Y/Z 坐标列
- Data variables：变量名（逗号分隔，如 `U[m/s],V[m/s],W[m/s]`）
- Detect Header：检测变量名/网格并自动填 U/V/W
- Preview：预览首个文件前 10 行

### 3) POD 参数
- Subtract mean：是否减去均值
- Normalize：是否归一化
- Modes (0=auto)：模态数量（0 表示自动）
- Method：SVD / randomized

### 4) 计算与重构
- Run POD：只计算，不导出
- Recon modes：低阶重构模态数
- Reconstruct now：执行重构（完成后允许导出重构结果）

### 5) 导出
- Export Results：导出勾选项
- Export Options 中可勾选导出内容

### 6) 结果展示
- 能量谱 / 累积能量
- FFT 频谱（可切换模态）
- 云图（可切换模态/变量/切片）

## 导出文件
- `metadata.json`
- `modal_metrics_wide.csv`
- `clouds/mode_###.dat`
- `per_mode/mode_###/mode.csv`（可选）
- `per_mode/mode_###/coefficients.csv`（可选）
- `per_mode/mode_###/coefficients_fft.csv`（可选）
- `recon/recon_###.dat`（可选）

