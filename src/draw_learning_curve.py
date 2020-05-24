'''
Created on Nov 3, 2016

draw a learning curve

@author: xiul
'''

import argparse, json
import matplotlib.pyplot as plt
import numpy as np
# import csv

plt_params = {'figure.figsize': (5.0, 3.5), 'xtick.labelsize': 6, 'ytick.labelsize': 6,
              'figure.dpi': 300, 'savefig.dpi': 300, 'font.family': 'Times New Roman'}
plt.rcParams.update(plt_params)

def read_performance_records(path):
    """ load the performance score (.json) file """
    
    data = json.load(open(path, 'rb'))
    for key in data['success_rate'].keys():
        if int(key) > -1:
            print("%s\t%s\t%s\t%s" % (key, data['success_rate'][key], data['ave_turns'][key], data['ave_reward'][key]))
            

def load_performance_file(path):
    """ load the performance score (.json) file """
    
    data = json.load(open(path, 'rb'))
    numbers = {'x': [], 'success_rate':[], 'ave_turns':[], 'ave_rewards':[]}
    keylist = [int(key) for key in data['success_rate'].keys()]
    keylist.sort()

    for key in keylist:
        if int(key) > -1:
            numbers['x'].append(int(key))
            numbers['success_rate'].append(data['success_rate'][str(key)])
            numbers['ave_turns'].append(data['ave_turns'][str(key)])
            numbers['ave_rewards'].append(data['ave_reward'][str(key)])
    return numbers

def draw_learning_curve(numbers):
    """ draw the learning curve """
    
    plt.xlabel('Simulation Epoch')
    plt.ylabel('Success Rate')
    plt.title('Learning Curve')
    plt.grid(True)

    plt.plot(numbers['x'], numbers['success_rate'], 'r', lw=1)
    plt.show()

    
# def main(params):
#     cmd = params['cmd']
    
#     if cmd == 0:
#         numbers = load_performance_file(params['result_file'])
#         draw_learning_curve(numbers)
#     elif cmd == 1:
#         read_performance_records(params['result_file'])

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
    
#     parser.add_argument('--cmd', dest='cmd', type=int, default=1, help='cmd')
    
#     parser.add_argument('--result_file', dest='result_file', type=str, default='./deep_dialog/checkpoints/runk20/agt_6_performance_records.json', help='path to the result file')
    
#     args = parser.parse_args()
#     params = vars(args)
#     print(json.dumps(params, indent=2))

#     main(params)
# if __name__ == "__main__":
#     path_list = [1, 10, 20]
    # color_list = ['DeepSkyBlue', 'DarkCyan', 'LimeGreen', 'OrangeRed', 'Olive']
    # sub_color = ['LightSkyBlue', 'Cyan', 'LightGreen', 'LightCoral', 'DarkKhaki']
    # plt.xlabel('Simulation Epoch')
    # plt.ylabel('Success Rate')
    # plt.title('Learning Curve')
    # plt.grid(True)

    # with open("res.csv","w") as csvfile: 
    #     writer = csv.writer(csvfile)
    #     writer.writerow(["Simulation Epoch","success_rate"])
    #     for i in range(len(path_list)):
    #         for j in range(5):
    #             path = './deep_dialog/checkpoints/DDQ_k' + str(path_list[i]) + '_run' + str(j+1) + '/agt_6_performance_records.json'
    #             numbers = load_performance_file(path)
    #             writer.writerows([numbers['x'], numbers['success_rate'], numbers['ave_turns']])

            # plt.plot(numbers['x'], numbers['success_rate'], color_list[i], lw=0.3)
    # plt.legend(loc="best")
    # plt.show()
def smooth(a, WSZ):
    # a:原始数据，NumPy 1-D array containing the data to be smoothed
    # 必须是1-D的，如果不是，请使用 np.ravel()或者np.squeeze()转化 
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    out0 = np.convolve(a,np.ones(WSZ,dtype=int),'valid')/WSZ
    r = np.arange(1,WSZ-1,2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((  start , out0, stop  ))

if __name__ == "__main__":
    path_list = [1, 10, 20]
    # color_list = ['DeepSkyBlue', 'DarkCyan', 'LimeGreen', 'OrangeRed', 'Olive']
    # sub_color = ['LightSkyBlue', 'Cyan', 'LightGreen', 'LightCoral', 'DarkKhaki']
    color_list = [[0.4660, 0.6740, 0.1880], [0.8500, 0.3250, 0.0980], [0, 0.4470, 0.7410]]
    sub_color = ['LightGreen', 'LightCoral', 'LightSkyBlue']
    sr_mean = []
    sr_std = []
    turn_mean = []
    turn_std = []
    
    fig = plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Average turns')
    plt.grid(True)
    for i in range(len(path_list)):
        srs = []
        turns = []
        for j in range(5):
            path = './deep_dialog/checkpoints/DDQ_k' + str(path_list[i]) + '_run' + str(j+1) + '/agt_6_performance_records.json'
            res = load_performance_file(path)
            x = res['x']
            srs.append(smooth(res['success_rate'], 9))
            turns.append(smooth(res['ave_turns'], 9))
        sr_mean.append(np.mean(srs, axis=0))
        sr_std.append(np.std(srs, axis=0))
        turn_mean.append(np.mean(turns, axis=0))
        turn_std.append(np.std(turns, axis=0))

        plt.plot(x[:-1], turn_mean[i][:-1], color=color_list[i], lw=0.6)
        plt.fill_between(x[:-1], turn_mean[i][:-1]+turn_std[i][:-1], turn_mean[i][:-1]-turn_std[i][:-1], color=sub_color[i], alpha=0.2)

    plt.legend(['DQN', 'DDQ(10)', "DDQ(20)"], loc='upper right', prop={'size': 6})
    plt.savefig('res_turn.png')
    # plt.show()

    


