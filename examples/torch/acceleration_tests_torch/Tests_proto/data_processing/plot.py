import matplotlib.pyplot as plt
def draw_pic(list1, list2, name, color, x_name, y_name):
    plt.figure()
    plt.plot(list1, list2, marker='o', color=color)
    plt.title(name)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.savefig('{}.png'.format(name))
    plt.show()
