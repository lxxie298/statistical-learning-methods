import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier

# 实例点
x = np.array(
    [[0.5, 0.9], [0.7, 2.8], [1.3, 4.6], [1.4, 2.8], [1.7, 0.8], [1.9, 1.4], [2, 4], [2.3, 3], [2.5, 2.5], [2.9, 2],
     [2.9, 3], [3, 4.5], [3.3, 1.1], [4, 3.7], [4, 2.2], [4.5, 2.5], [4.6, 1], [5, 4]])

# 类别
y = np.array([0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0])

# 设置二维网格的边界
x_min, x_max = 0, 6
y_min, y_max = 0, 6

# 设置不同类别区域的颜色
cmap_light = ListedColormap(['#FFFFFF', '#BDBDBD'])

# 生成二维网格
h = .01
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# 这里的参数n_neighbors就是k近邻法中的k
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x, y)

# ravel()实现扁平化，比如将形状为3*4的数组变成1*12
# np.c_()在列方向上连接数组，要求被连接数组的行数相同
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure()

# 设置坐标轴的刻度
plt.xticks(tuple([x for x in range(6)]))
plt.yticks(tuple([y for y in range(6) if y != 0]))

# 填充不同分类区域的颜色
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# 设置坐标轴标签
plt.xlabel('$x^{(1)}$')
plt.ylabel('$x^{(2)}$')

# 绘制实例点的散点图
plt.scatter(x[:, 0], x[:, 1], c=y)

plt.show()
