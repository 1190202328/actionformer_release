import os
import sys

# add python path of PadleDetection to sys.path
from matplotlib.ticker import MultipleLocator
from mpl_toolkits import axisartist

parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 3)))
sys.path.insert(0, parent_path)
from libs.utils import postprocessing
import matplotlib.pyplot as plt
import numpy as np

to_visualize_name = {
    'video_test_0000006': [19, 18.8, 57.3, 'VolleyballSpiking'],
    'video_test_0000011': [15, 0.4, 12.4, 'Shotput'],
    'video_test_0000045': [18, 1.6, 8.7, 'ThrowDiscus']
}


def get_data():
    num_pred = 200
    topk = 2
    output_dir = '../../ckpt/thumos_i3d_reproduce/figures'
    if os.path.exists(output_dir):
        for name in os.listdir(output_dir):
            os.remove(f'{output_dir}/{name}')
        os.removedirs(output_dir)
    os.makedirs(output_dir)

    results = postprocessing.load_results_from_pkl('../../ckpt/thumos_i3d_reproduce/eval_results.pkl')
    # array -> dict
    results = postprocessing.results_to_array(results, num_pred=num_pred)

    for vid in to_visualize_name:
        # print(results[vid]['segment'][:5])
        # print(results[vid]['label'][:5])
        # print(results[vid]['score'][:5])
        # print(results[vid]['point'][:5])
        draw_regression_classification(results[vid], to_visualize_name[vid],
                                       f'../../ckpt/thumos_i3d_reproduce/figures/{vid}.jpg')
        print(f'{vid}.jpg generated!')


# def draw_regression_classification(result, info, output_file):
#     # 绘制直方图
#     bound = (info[2] - info[1]) / 3
#     start = info[1] - bound
#     end = info[2] + bound
#     label_id = info[0]
#     class_name = info[3]
#     #
#     total_num = 1000
#     gap = total_num / (end - start)
#     #
#     x = np.linspace(start, end, total_num)
#     y_onset = np.zeros(shape=(total_num,))
#     y_offset = np.zeros(shape=(total_num,))
#     for i in range(len(result['segment'])):
#         onset, offset = result['segment'][i]
#         score = result['score'][i]
#         label = int(result['label'][i])
#         if start <= onset <= end and label == label_id:
#             y_onset[int((onset - start) * gap)] += score
#         if start <= offset <= end and label == label_id:
#             y_offset[int((offset - start) * gap)] += score
#     y_point = np.zeros(shape=(total_num,))
#     for i in range(len(result['point'])):
#         point, label, score = result['point'][i]
#         if start <= point <= end and label == label_id:
#             y_point[int((point - start) * gap)] += score
#     max_y = max(max(y_onset), max(y_offset)) * 1.1
#     fig = plt.figure(figsize=(20, 8))
#
#     ax = axisartist.Subplot(fig, 211)
#     # 将绘图区对象添加到画布中
#     fig.add_axes(ax)
#     x_set = MultipleLocator((end - start) / total_num * 50)  # x轴的刻度
#     # 设置刻度间隔
#     ax.xaxis.set_major_locator(x_set)
#     gt_x_onset = np.ones(shape=(10,)) * info[1]
#     gt_x_offset = np.ones(shape=(10,)) * info[2]
#     gt_y = np.linspace(0, max_y, 10)
#     ax.plot(gt_x_onset, gt_y, c='black', linestyle='--')
#     ax.plot(gt_x_offset, gt_y, c='black', linestyle='--')
#     ax.bar(x, y_point, color='red', label=f'Classification Scores for "{class_name}"')
#     # # 设置图表参数
#     ax.axis["bottom"].set_axisline_style("->", size=1.5)
#     ax.axis["left"].set_axisline_style("->", size=1.5)
#     # 通过set_visible方法设置绘图区的顶部及右侧坐标轴隐藏
#     ax.axis["top"].set_visible(False)
#     ax.axis["right"].set_visible(False)
#     plt.xticks([])
#     plt.yticks([])
#     plt.legend(loc='upper right')
#
#     ax = axisartist.Subplot(fig, 212)
#     # 将绘图区对象添加到画布中
#     fig.add_axes(ax)
#     x_set = MultipleLocator((end - start) / total_num * 50)  # x轴的刻度
#     # 设置刻度间隔
#     ax.xaxis.set_major_locator(x_set)
#     gt_x_onset = np.ones(shape=(10,)) * info[1]
#     gt_x_offset = np.ones(shape=(10,)) * info[2]
#     gt_y = np.linspace(0, max_y, 10)
#     ax.plot(gt_x_onset, gt_y, c='black', linestyle='--')
#     ax.plot(gt_x_offset, gt_y, c='black', linestyle='--')
#     ax.bar(x, y_onset, color='steelblue', label='Predict Onset')
#     ax.bar(x, y_offset, color='coral', label='Predict Offset')
#     # # 设置图表参数
#     ax.axis["bottom"].set_axisline_style("->", size=1.5)
#     ax.axis["left"].set_axisline_style("->", size=1.5)
#     # 通过set_visible方法设置绘图区的顶部及右侧坐标轴隐藏
#     ax.axis["top"].set_visible(False)
#     ax.axis["right"].set_visible(False)
#     plt.xticks([])
#     plt.yticks([])
#     plt.legend(loc='upper right')
#
#     plt.savefig(output_file)

def draw_regression_classification(result, info, output_file):
    # 绘制直方图
    # TODO 需要归一化图像
    factor = 50 / (info[2] - info[1])
    bound = (info[2] - info[1]) / 3
    start = info[1] - bound
    end = info[2] + bound
    start *= factor
    end *= factor

    label_id = info[0]
    class_name = info[3]
    #
    total_num = 1000
    gap = total_num / (end - start)
    #
    x = np.linspace(start, end, total_num)
    y_onset = np.zeros(shape=(total_num,))
    y_offset = np.zeros(shape=(total_num,))
    for i in range(len(result['segment'])):
        onset, offset = result['segment'][i]
        score = result['score'][i]
        label = int(result['label'][i])
        if start <= onset <= end and label == label_id:
            y_onset[int((onset - start) * gap)] += score
        if start <= offset <= end and label == label_id:
            y_offset[int((offset - start) * gap)] += score
    y_point = np.zeros(shape=(total_num,))
    for i in range(len(result['point'])):
        point, label, score = result['point'][i]
        if start <= point <= end and label == label_id:
            y_point[int((point - start) * gap)] += score
    max_y = max(max(y_onset), max(y_offset)) * 1.1
    fig = plt.figure(figsize=(20, 8))

    ax = axisartist.Subplot(fig, 211)
    # 将绘图区对象添加到画布中
    fig.add_axes(ax)
    x_set = MultipleLocator((end - start) / total_num * 50)  # x轴的刻度
    # 设置刻度间隔
    ax.xaxis.set_major_locator(x_set)
    gt_x_onset = np.ones(shape=(10,)) * info[1]
    gt_x_offset = np.ones(shape=(10,)) * info[2]
    gt_y = np.linspace(0, max_y, 10)
    ax.plot(gt_x_onset, gt_y, c='black', linestyle='--')
    ax.plot(gt_x_offset, gt_y, c='black', linestyle='--')
    ax.bar(x, y_point, color='red', label=f'Classification Scores for "{class_name}"')
    # # 设置图表参数
    ax.axis["bottom"].set_axisline_style("->", size=1.5)
    ax.axis["left"].set_axisline_style("->", size=1.5)
    # 通过set_visible方法设置绘图区的顶部及右侧坐标轴隐藏
    ax.axis["top"].set_visible(False)
    ax.axis["right"].set_visible(False)
    plt.xticks([])
    plt.yticks([])
    plt.legend(loc='upper right')

    ax = axisartist.Subplot(fig, 212)
    # 将绘图区对象添加到画布中
    fig.add_axes(ax)
    x_set = MultipleLocator((end - start) / total_num * 50)  # x轴的刻度
    # 设置刻度间隔
    ax.xaxis.set_major_locator(x_set)
    gt_x_onset = np.ones(shape=(10,)) * info[1]
    gt_x_offset = np.ones(shape=(10,)) * info[2]
    gt_y = np.linspace(0, max_y, 10)
    ax.plot(gt_x_onset, gt_y, c='black', linestyle='--')
    ax.plot(gt_x_offset, gt_y, c='black', linestyle='--')
    ax.bar(x, y_onset, color='steelblue', label='Predict Onset')
    ax.bar(x, y_offset, color='coral', label='Predict Offset')
    # # 设置图表参数
    ax.axis["bottom"].set_axisline_style("->", size=1.5)
    ax.axis["left"].set_axisline_style("->", size=1.5)
    # 通过set_visible方法设置绘图区的顶部及右侧坐标轴隐藏
    ax.axis["top"].set_visible(False)
    ax.axis["right"].set_visible(False)
    plt.xticks([])
    plt.yticks([])
    plt.legend(loc='upper right')

    plt.savefig(output_file)


get_data()
