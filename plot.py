import matplotlib.pyplot as plt

# # 假设我们有以下两个列表
# list1 = [1, 2, 3, 4, 5]
# list2 = [10, 20, 30, 40, 50]

def draw_pic(list1, list2, name, color, x_name, y_name):
    # 使用matplotlib来绘制图形
    plt.figure()  # 创建一个新的图形

    # 绘制list1和list2
    plt.plot(list1, list2, marker='o', color=color)  # 'o'是点的样式

    # 添加标题和标签
    plt.title(name)
    plt.xlabel(x_name)
    plt.ylabel(y_name)

    # 保存图形到文件系统，选择一个路径和文件名
    plt.savefig('{}.png'.format(name))  # 这将保存图形为PNG文件

    # 显示图形
    plt.show()
