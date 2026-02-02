import numpy as np
import os

# 定义文件路径
input_file = r"d:\projcect\POD of three-dimensional flow field data\tests\4-shao-经典工况-近场-减去时均\per_mode\mode_001\mode.csv"
output_file = r"d:\projcect\POD of three-dimensional flow field data\tests\4-shao-经典工况-近场-减去时均\per_mode\mode_001\mode.dat"

# 直接指定网格大小
grid_shape = (26, 26, 100)

# 检查网格大小是否匹配数据大小
data = np.loadtxt(input_file)
if np.prod(grid_shape) != data.size:
    raise ValueError(f"指定的网格大小 {grid_shape} 与数据大小 ({data.size}) 不匹配！")

# 使用指定网格大小
print(f"使用指定的网格大小: {grid_shape}")

def main():
    # 读取 CSV 文件
    with open(input_file, 'r') as f:
        data = np.loadtxt(f)

    # 检查数据大小是否匹配目标网格
    if data.size != np.prod(grid_shape):
        raise ValueError(f"数据大小 ({data.size}) 与目标网格大小 ({np.prod(grid_shape)}) 不匹配！")

    # 重组数据为目标网格
    reshaped_data = data.reshape(grid_shape)

    # 生成三维坐标轴
    x = np.linspace(0, 1, grid_shape[0])  # 假设 x 轴范围为 [0, 1]
    y = np.linspace(0, 1, grid_shape[1])  # 假设 y 轴范围为 [0, 1]
    z = np.linspace(0, 1, grid_shape[2])  # 假设 z 轴范围为 [0, 1]

    # 创建网格坐标
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

    # 展平坐标和数据
    coords = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel(), data.ravel()))

    # 保存为 .dat 文件
    with open(output_file, 'w') as f:
        f.write("# x, y, z, value\n")
        np.savetxt(f, coords, fmt='%.6e', delimiter=' ')

    print(f"数据已成功重组并保存为 {output_file}")

if __name__ == "__main__":
    main()