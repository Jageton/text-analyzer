from matplotlib import pyplot as plt
from pandas import DataFrame

colors = ['#3b4cc0', '#db2c2c', '#13e700', '#ffdd00', '#ff52d5',
          '#08c7c9', '#5e2594', '#a0ff45', '#f36c1f', '#B3B6B7']


def get_color(x):
    if x < 0:
        return '#000000'
    return colors[x % len(colors)]


def get_colors(predict):
    return list(map(get_color, predict))


def _dataframe_2d_visualization(dataframe, predict):
    plt.scatter(dataframe.values[:, 0], dataframe.values[:, 1], c=get_colors(predict),
                marker="o", linewidths=0.5, edgecolors='#000000', picker=True)
    plt.title('Clusterization result')
    plt.xlabel(dataframe.columns[0])
    plt.ylabel(dataframe.columns[1])
    plt.show()


def dataframe_2d_vis(dataframe, col1_idx, col2_idx, predict):
    plt.scatter(dataframe.values[:, col1_idx], dataframe.values[:, col2_idx], c=get_colors(predict),
                marker="o", linewidths=0.5, edgecolors='#000000', picker=True)
    plt.title('Clusterization result')
    plt.xlabel(dataframe.columns[col1_idx])
    plt.ylabel(dataframe.columns[col2_idx])
    plt.show()


def _dataframe_3d_visualization(dataframe, predict):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(dataframe.values[:, 0], dataframe.values[:, 1], dataframe.values[:, 2], c=get_colors(predict),
               marker="o", linewidths=0.5, edgecolors='#000000', picker=True)
    ax.set_xlabel(dataframe.columns[0])
    ax.set_ylabel(dataframe.columns[1])
    ax.set_zlabel(dataframe.columns[2])
    plt.show()


def _array_2d_visualization(array, predict):
    plt.scatter(array[:, 0], array[:, 1], c=get_colors(predict),
                marker="o", linewidths=0.5, edgecolors='#000000', picker=True)
    plt.title('Clusterization result')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


def _array_3d_visualization(array, predict):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(array[:, 0], array[:, 1], array[:, 2], c=get_colors(predict),
               marker="o", linewidths=0.5, edgecolors='#000000', picker=True)
    ax.set_xlabel('X')
    ax.set_xlabel('Y')
    ax.set_xlabel('Z')
    plt.show()


def data_2d_visualization(data, predict):
    is_data_frame = type(data) is DataFrame
    if is_data_frame:
        _dataframe_2d_visualization(data, predict)
    else:
        _array_2d_visualization(data, predict)


def data_3d_visualization(data, predict):
    is_data_frame = type(data) is DataFrame
    if is_data_frame:
        _dataframe_3d_visualization(data, predict)
    else:
        _array_3d_visualization(data, predict)
