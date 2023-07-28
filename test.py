# import torch
# gate_value = torch.tensor([[ 0.3744,  1.3059,  0.5642],
#         [-0.0132, -0.8948, -0.7674],
#         [ 1.3528, -0.3122, -0.7936],
#         [ 1.4298,  0.5266,  0.3011],
#         [-2.1607,  0.1379,  0.9104],
#         [-2.5439,  0.4604,  0.6687],
#         [-0.9078, -0.7056, -0.2652],
#         [ 0.7623,  1.9272,  1.3256],
#         [-0.4074, -0.2061,  1.1980],
#         [ 0.2467, -0.8713,  2.1926]])
# x = torch.tensor([[ 0.2218,  0.1708, -1.7542,  0.3792],
#         [-0.6737,  0.8634, -0.1898, -0.6905],
#         [-0.2160, -0.0644, -0.5811,  0.9310],
#         [ 0.9957,  1.0091, -1.0574, -1.2687],
#         [-0.0588,  0.6803, -0.1245,  1.6237],
#         [-1.1072,  1.1992, -0.8199, -1.0051],
#         [ 0.3339,  0.7643, -1.8591, -0.6082],
#         [ 0.6836,  0.0535, -0.5755, -2.0284],
#         [ 0.1638, -1.3508,  0.5188, -0.9345],
#         [-0.7793, -0.1039, -0.0675,  0.7540]])
# n = 3
# la = 7
# TOPK= 2
# mask = torch.full_like(gate_value,0)
# gate_value_tmp = torch.clone(gate_value)
# while not gate_value_tmp.isinf().all():
#     id_t = gate_value_tmp.argmax()
#     id_sample, id_op = id_t // n, id_t%n
#     if (mask[:,id_op]).sum()<la and (mask[id_sample]>0).sum()<TOPK:
#         mask[id_sample][id_op] = 1
#     gate_value_tmp[id_sample][id_op] = float('-inf')
# gate_value[mask==0] = float('-inf')
# gate_value = gate_value.softmax(dim=-1)

# import IPython; import sys;
# IPython.embed();sys.exit(0)


# plot Gaussian Function
# 注：正态分布也叫高斯分布
# import matplotlib.pyplot as plt
# import numpy as np
# import math
# def gauss(x,mu,sigma):
#     left = 1 / (np.sqrt(2 * math.pi) * sigma)
#     right = np.exp(-(x - mu)**2 / (2 * sigma**2))
#     return left * right
# x = np.arange(-5,5,0.1)
# plt.plot(x,gauss(x,1.,2.))
# plt.plot(x,gauss(x,1,1))
# # plt.plot(x,gauss(x,1,0.2))
# # plt.plot(x,gauss(x,1,0.001))
# plt.show()

# u1 = 0  # 第一个高斯分布的均值
# sigma1 = 1  # 第一个高斯分布的标准差

# u2 = 1  # 第二个高斯分布的均值
# sigma2 = 2  # 第二个高斯分布的标准差
# x = np.arange(-5, 5, 0.1)
# # 表示第一个高斯分布函数
# y1 = np.multiply(np.power(np.sqrt(2 * np.pi) * sigma1, -1), np.exp(-np.power(x - u1, 2) / 2 * sigma1 ** 2))
# # 表示第二个高斯分布函数
# y2 = np.multiply(np.power(np.sqrt(2 * np.pi) * sigma2, -1), np.exp(-np.power(x - u2, 2) / 2 * sigma2 ** 2))

# plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决pythonmatplotlib绘图无法显示中文的问题
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# plt.subplot(121)
# plt.plot(x, y1, 'b-', linewidth=2)
# plt.title("高斯分布函数图像")

# plt.subplot(122)
# plt.plot(x, y2, 'r-', linewidth=2)
# plt.title('高斯分布函数图像')
# plt.show()

## 矩阵图
# moedarts_wo_loss = [[0.1131,0.1221, 0.1221, 0.1218, 0.1406, 0.1112, 0.1212, 0.1479],
#        [0.0904, 0.1255, 0.1221, 0.1416, 0.1426, 0.1278, 0.1253, 0.1248],
#        [0.1617, 0.1088, 0.1072, 0.1092, 0.1086, 0.1845, 0.1108, 0.1092],
#        [0.1212, 0.145 , 0.1232, 0.1522, 0.1158, 0.1151, 0.1082, 0.1192],
#        [0.1622, 0.1799, 0.1088, 0.1147, 0.1017, 0.1097, 0.1132, 0.1097]]

# moedarts_w_loss = [[0.10,0.1068, 0.1175, 0.0776, 0.3146, 0.079 , 0.0795, 0.125 ],
#        [0.0723, 0.153 , 0.0594, 0.1324, 0.3292, 0.0595, 0.1371, 0.0571],
#        [0.0914, 0.0982, 0.0907, 0.0923, 0.1156, 0.3364, 0.1157, 0.0596],
#        [0.0712, 0.1918, 0.104 , 0.0832, 0.0815, 0.1054, 0.087 , 0.2859],
#        [0.0813, 0.1331, 0.0891, 0.0706, 0.1475, 0.1533, 0.25   , 0.0751]]
# import matplotlib
# import matplotlib.pyplot as plt
# fig,axes = plt.subplots(1,2)
# norm = matplotlib.colors.Normalize(vmin=0, vmax=0.4)
# mat1=axes[0].matshow(moedarts_wo_loss,cmap=plt.cm.Greens,norm=norm)
# mat2=axes[1].matshow(moedarts_w_loss,cmap=plt.cm.Greens,norm=norm)
# # 
# position = fig.add_axes([0.92, 0.12, 0.015, 0.78 ])
# fig.colorbar(mat2,cax=position)
# plt.show()

import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

# config = {
#     "font.family":'serif',
#     "font.size": 20,
#     "mathtext.fontset":'stix',
#     "font.serif": ['SimSun'],
# }
# rcParams.update(config)
fig = plt.figure(figsize=(10,6))
# plt.rcParams['font.sans-serif']=['SimHei']
scs = [0.23,1.5,4,1.5,0.4,1.6]
accs =[97.4,97,97.24,97.15,97.35,97.05]
labels = ["Gate-DARTS","DARTS(first order)","DARTS\n(second order)","SNAS","NoisyDARTS","R-DARTS"]
markers = ["*","^","8","x","p","+"]
for i in range(len(scs)):
       plt.scatter(scs[i],accs[i], label=labels[i],marker=markers[i],s=80)
       if i==2:
              plt.annotate(labels[i], xy = (scs[i], accs[i]), xytext = (scs[i]-0.4,accs[i]-0.03))
       elif i==1:
              plt.annotate(labels[i], xy = (scs[i], accs[i]), xytext = (scs[i]+0.04,accs[i]))
       else:
              plt.annotate(labels[i], xy = (scs[i], accs[i]), xytext = (scs[i]-0.06,accs[i]-0.02))
       
plt.xlabel("search cost (GPU days)")
plt.ylabel("CIFAR-10 accuracy")
plt.show()

