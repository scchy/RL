# python3
# Create date: 2025-05-20
# Author: Scc_hy
# ===================================================================

import numpy as np
import matplotlib.pyplot as plt

def gaussian_pdf(x, y, mu, sigma):
    """
    二维高斯分布的概率密度函数
    :param x: x 坐标
    :param y: y 坐标
    :param mu: 均值向量 [mu_x, mu_y]
    :param sigma: 协方差矩阵 [[sigma_x^2, sigma_xy], [sigma_xy, sigma_y^2]]
    :return: 高斯分布的概率密度值
    """
    det_sigma = np.linalg.det(sigma)
    inv_sigma = np.linalg.inv(sigma)
    norm_const = 1.0 / (2 * np.pi * np.sqrt(det_sigma))
    exponent = -0.5 * (np.dot(np.dot(np.array([x - mu[0], y - mu[1]]), inv_sigma), np.array([x - mu[0], y - mu[1]])))
    return norm_const * np.exp(exponent)


# 定义均值和协方差矩阵
mu = [0, -0.5]
sigma = np.array([[0.05, 0.04], [0.04, 0.1]])

# 创建网格点
x = np.linspace(-1, 1.5, 100)
y = np.linspace(-1, 1.5, 100)
X, Y = np.meshgrid(x, y)

# 计算每个点的概率密度值
Z = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = gaussian_pdf(X[i, j], Y[i, j], mu, sigma) + gaussian_pdf(X[i, j], Y[i, j], [0.7, 0.75], np.array([[0.1, 0.075], [0.075, 0.1]]))

# 绘制等高线图
plt.figure(figsize=(8, 6))
contour = plt.contour(X, Y, Z, levels=15, cmap='viridis')  # 绘制等高线图
plt.clabel(contour, inline=True, fontsize=8)  # 添加等高线标签
plt.title("Contour Plot of a 2D Multi-Gaussian Distribution")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.colorbar(contour)  # 添加颜色条
plt.show()




# 求解
# -----------------------------------------------------------
p = Z.flatten()/Z.sum()

q_org = np.random.uniform(0, 1, size=p.shape[0])
q = q_org/q_org.sum()
q.sum()

# p log p - p log q
d_o = (p * np.log(p/(q + 1e-15))).sum()
print(f'd_o D_forward_KL={d_o:3f}')
for i in range(15):
    q_partial = -p/q  
    q -= 0.0025 * q_partial 
    q = q/q.sum()
    d = (p * np.log(p/(q + 1e-15))).sum()
    print(f'{i=} D_forward_KL={d:3f}')

d_forward = d
p_std = (p - p.min())/(p.max() - p.min())
p_minmax = (p_std + Z.min()) * (Z.max() - Z.min())



# ----
q_org = np.random.uniform(0, 1, size=p.shape[0])
q_r = q_org/q_org.sum()

# q log q - q log p
d_o = (q_r * np.log((q_r + 1e-15)/(p + 1e-15))).sum()
print(f'd_o D_reverse_KL={d_o:3f}')
for i in range(50):
    q_partial = 1 * np.log(q_r + 1e-15) + 1 - np.log(p+1e-15)
    q_r -= 0.00075 * q_partial 
    q_r = np.clip(q_r/q_r.sum(), 1e-15, 9.999)
    d = (q_r * np.log((q_r + 1e-15)/(p + 1e-15))).sum()
    print(f'{i=} D_reverse_KL={d:3f}')

d_reverse = d
p_r_std = (q_r - q_r.min())/(q_r.max() - q_r.min())
p_r_minmax = (p_r_std + p_r_std.min()) * (Z.max() - Z.min())


# zero-force
fig, axis = plt.subplots(1, 2, figsize=(16, 6))
contour = axis[0].contour(X, Y, Z, levels=15, cmap='viridis')  # 绘制等高线图
contour1 = axis[0].contour(X, Y, p_r_minmax.reshape(Z.shape), levels=15, cmap='RdYlGn_r')  # 绘制等高线图
axis[0].clabel(contour, inline=True, fontsize=8)  # 添加等高线标签
axis[0].clabel(contour1, inline=True, fontsize=8)  # 添加等高线标签
axis[0].set_title(f"Contour Plot of reverse_KL (zero forcing)\n{d_reverse=:.3f}")
axis[0].set_xlabel("X-axis")
axis[0].set_ylabel("Y-axis")
plt.colorbar(contour, ax=axis[0])
plt.colorbar(contour1, ax=axis[0])

# 当 p 具有多个峰并且这些峰间隔很宽时
contour = axis[1].contour(X, Y, Z, levels=15, cmap='viridis')  # 绘制等高线图
contour1 = axis[1].contour(X, Y, p_minmax.reshape(Z.shape), levels=8, cmap='RdYlGn_r')  # 绘制等高线图
axis[1].clabel(contour, inline=True, fontsize=8)  # 添加等高线标签
axis[1].clabel(contour1, inline=True, fontsize=8)  # 添加等高线标签
axis[1].set_title(f"Contour Plot of a forward_KL (zero avoiding)\n{d_forward=:.3f}")
axis[1].set_xlabel("X-axis")
axis[1].set_ylabel("Y-axis")
plt.colorbar(contour, ax=axis[1])
plt.colorbar(contour1, ax=axis[1])
plt.show()

