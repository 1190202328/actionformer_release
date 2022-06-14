import os
import sys

# add python path of PadleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 3)))
sys.path.insert(0, parent_path)
from libs.utils import postprocessing
import matplotlib.pyplot as plt
import numpy as np

to_visualize_name = {
    'video_test_0000004', 'video_test_0000006', 'video_test_0000007'
}


def get_data():
    num_pred = 200
    topk = 2
    results = postprocessing.load_results_from_pkl('../../ckpt/thumos_i3d_reproduce/eval_results.pkl')
    # array -> dict
    results = postprocessing.results_to_array(results, num_pred=num_pred)

    processed_results = {
        'video-id': [],
        't-start': [],
        't-end': [],
        'label': [],
        'score': []
    }
    print(results['video_test_0000004'])
    draw(results['video_test_0000004'])
    # for vid in to_visualize_name:
    #     result = results[vid]
    #     pred_score, pred_segment, pred_label = \
    #         result['score'], result['segment'], result['label']
    #     num_segs = min(num_pred, len(pred_score))
    #
    #     processed_results['video-id'].extend([vid] * num_segs * topk)
    #     processed_results['t-start'].append(pred_segment[:, 0])
    #     processed_results['t-end'].append(pred_segment[:, 1])
    #     processed_results['label'].append(pred_label)
    #     processed_results['score'].append(pred_score)
    # print(processed_results)


def draw(result):
    # 绘制直方图
    # figure = plt.figure()

    x = np.linspace(0, 50, 5000)

    y = []
    for i in range(result['segment']):
        start, end = result['segment'][i]
        score = result['score'][i]
        print(start, end)
        start = round(start, 2)
        end = round(end, 2)
        if 20 <= start <= 22 or 22 <= end <= 24:
            y.append(score)

    plt.bar(x, y, color='blue')
    # # 设置图表参数
    plt.xlabel('total_score', fontsize=15, color='black')
    plt.ylabel('interface_delta_B', fontsize=15, color='green')
    # plt.title('score', fontsize = 20) #设置标题
    # plt.axis([-1, 6, -2, 2])#可手动设置x轴y轴范围
    # plt.grid(True) #设置网格

    plt.show()


get_data()
