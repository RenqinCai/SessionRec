import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Thin color per time graphs
fig = plt.figure()
fig.suptitle('Performance vs. Popualrity Over Time (RNN Taobao)', fontsize=16)
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

recalls = [[0.1183, 0.1404, 0.1565, 0.1676, 0.1890, 0.2161, 0.2188, 0.2653],
           [0.1045, 0.1339, 0.1547, 0.1681, 0.1931, 0.2119, 0.2478, 0.3033],
           [0.1008, 0.1259, 0.1429, 0.1538, 0.1764, 0.2062, 0.2539, 0.2947],
           [0.0885, 0.1188, 0.1290, 0.1561, 0.1799, 0.2105, 0.2458, 0.2943],
           [0.0882, 0.1086, 0.1262, 0.1366, 0.1479, 0.1867, 0.2289, 0.2926]]
bin_num = 0
for c, z in zip(['r', 'g', 'b', 'y', 'c'], [[1]*8, [2]*8, [3]*8, [4]*8, [5]*8]):
    x = [1,2,3,4,5,6,7,8]
    y = recalls[bin_num]

    cs = [c] * len(x)
    ax1.bar(x, y, zs=z, zdir='y', color=cs, alpha=0.6)
    bin_num += 1

ax1.view_init(elev=20, azim=280)
ax1.set_title('Performance: Recall')
ax1.set_xlabel('Popularity')
ax1.set_ylabel('Time', labelpad=10)
ax1.set_zlabel('Recall@20', labelpad=10)


mrrs = [[0.0573, 0.0646, 0.0682, 0.0677, 0.0666, 0.0740, 0.0725, 0.0894],
        [0.0498, 0.0655, 0.0705, 0.0688, 0.0684, 0.0719, 0.0838, 0.1038],
        [0.0497, 0.0610, 0.0634, 0.0610, 0.0600, 0.0689, 0.0868, 0.1062],
        [0.0420, 0.0546, 0.0551, 0.0621, 0.0619, 0.0703, 0.0858, 0.1073],
        [0.0380, 0.0478, 0.0514, 0.0503, 0.0494, 0.0567, 0.0696, 0.1086]]
bin_num = 0
for c, z in zip(['r', 'g', 'b', 'y', 'c'], [[1]*8, [2]*8, [3]*8, [4]*8, [5]*8]):
    x = [1,2,3,4,5,6,7,8]
    y = mrrs[bin_num]

    cs = [c] * len(x)
    ax2.bar(x, y, zs=z, zdir='y', color=cs, alpha=0.6)
    bin_num += 1

ax2.view_init(elev=20, azim=280)
ax2.set_title('Performance: MRR')
ax2.set_xlabel('Popularity')
ax2.set_ylabel('Time', labelpad=10)
ax2.set_zlabel('MRR@20', labelpad=15)
fig.savefig('sequential_bias.png')

# Block bar graphs
fig_recall = plt.figure()
fig_recall.suptitle('Recall vs. Popualrity Over Time (RNN Taobao)', fontsize=16)
ax = fig_recall.add_subplot(111, projection='3d')
x = [1,2,3,4,5,6,7,8]*5
repeat = [1,2,3,4,5]
y = [element for element in repeat for i in range(8)]
z = 0
dx = 1
dy = 1
dz = [0.1183, 0.1404, 0.1565, 0.1676, 0.1890, 0.2161, 0.2188, 0.2653,
    0.1045, 0.1339, 0.1547, 0.1681, 0.1931, 0.2119, 0.2478, 0.3033,
    0.1008, 0.1259, 0.1429, 0.1538, 0.1764, 0.2062, 0.2539, 0.2947,
    0.0885, 0.1188, 0.1290, 0.1561, 0.1799, 0.2105, 0.2458, 0.2943,
    0.0882, 0.1086, 0.1262, 0.1366, 0.1479, 0.1867, 0.2289, 0.2926]
c = ['r','b','g','y','c']
cs = [color for color in c for i in range(8)]
ax.bar3d(x, y, z, dx, dy, dz, shade=True, color=cs)
ax.view_init(elev=10, azim=90)
ax.set_xlabel('Popularity')
ax.set_ylabel('Time', labelpad=10)
ax.set_zlabel('Recall@20', labelpad=10)
fig_recall.savefig('sequential_bias_recall.png')

fig_mrr = plt.figure()
fig_mrr.suptitle('MRR vs. Popualrity Over Time (RNN Taobao)', fontsize=16)
ax = fig_mrr.add_subplot(111, projection='3d')
x = [1,2,3,4,5,6,7,8]*5
repeat = [1,2,3,4,5]
y = [element for element in repeat for i in range(8)]
z = 0
dx = 1
dy = 1
dz = [0.0573, 0.0646, 0.0682, 0.0677, 0.0666, 0.0740, 0.0725, 0.0894,
    0.0498, 0.0655, 0.0705, 0.0688, 0.0684, 0.0719, 0.0838, 0.1038,
    0.0497, 0.0610, 0.0634, 0.0610, 0.0600, 0.0689, 0.0868, 0.1062,
    0.0420, 0.0546, 0.0551, 0.0621, 0.0619, 0.0703, 0.0858, 0.1073,
    0.0380, 0.0478, 0.0514, 0.0503, 0.0494, 0.0567, 0.0696, 0.1086]
c = ['r','b','g','y','c']
cs = [color for color in c for i in range(8)]
ax.bar3d(x, y, z, dx, dy, dz, shade=True, color=cs)
ax.view_init(elev=10, azim=90)
ax.set_xlabel('Popularity')
ax.set_ylabel('Time', labelpad=10)
ax.set_zlabel('MRR@20', labelpad=20)
fig_mrr.savefig('sequential_bias_mrr.png')


# Graphs for each popularity over 
recalls = np.array(recalls)

pop1_fig, ax = plt.subplots()
ax.bar([1,2,3,4,5], recalls[:,0])
ax.set(xlabel='Time', ylabel='Recall@20',
    title='Recall of Popularity Bin 1 Over Time (Taobao RNN)')
ax.grid()
pop1_fig.savefig("recall_pop1.png")

pop2_fig, ax = plt.subplots()
ax.bar([1,2,3,4,5], recalls[:,1])
ax.set(xlabel='Time', ylabel='Recall@20',
    title='Recall of Popularity Bin 2 Over Time (Taobao RNN)')
ax.grid()
pop2_fig.savefig("recall_pop2.png")

pop3_fig, ax = plt.subplots()
ax.bar([1,2,3,4,5], recalls[:,2])
ax.set(xlabel='Time', ylabel='Recall@20',
    title='Recall of Popularity Bin 3 Over Time (Taobao RNN)')
ax.grid()
pop3_fig.savefig("recall_pop3.png")

pop4_fig, ax = plt.subplots()
ax.bar([1,2,3,4,5], recalls[:,3])
ax.set(xlabel='Time', ylabel='Recall@20',
    title='Recall of Popularity Bin 4 Over Time (Taobao RNN)')
ax.grid()
pop4_fig.savefig("recall_pop4.png")

pop5_fig, ax = plt.subplots()
ax.bar([1,2,3,4,5], recalls[:,4])
ax.set(xlabel='Time', ylabel='Recall@20',
    title='Recall of Popularity Bin 5 Over Time (Taobao RNN)')
ax.grid()
pop5_fig.savefig("recall_pop5.png")

pop6_fig, ax = plt.subplots()
ax.bar([1,2,3,4,5], recalls[:,5])
ax.set(xlabel='Time', ylabel='Recall@20',
    title='Recall of Popularity Bin 6 Over Time (Taobao RNN)')
ax.grid()
pop6_fig.savefig("recall_pop6.png")

pop7_fig, ax = plt.subplots()
ax.bar([1,2,3,4,5], recalls[:,6])
ax.set(xlabel='Time', ylabel='Recall@20',
    title='Recall of Popularity Bin 7 Over Time (Taobao RNN)')
ax.grid()
pop7_fig.savefig("recall_pop7.png")

pop8_fig, ax = plt.subplots()
ax.bar([1,2,3,4,5], recalls[:,7])
ax.set(xlabel='Time', ylabel='Recall@20',
    title='Recall of Popularity Bin 8 Over Time (Taobao RNN)')
ax.grid()
pop8_fig.savefig("recall_pop8.png")


#MRR
mrrs = np.array(mrrs)
pop1_fig, ax = plt.subplots()
ax.bar([1,2,3,4,5], mrrs[:,0])
ax.set(xlabel='Time', ylabel='MRR@20',
    title='MRR of Popularity Bin 1 Over Time (Taobao RNN)')
ax.grid()
pop1_fig.savefig("mrr_pop1.png")

pop2_fig, ax = plt.subplots()
ax.bar([1,2,3,4,5], mrrs[:,1])
ax.set(xlabel='Time', ylabel='MRR@20',
    title='MRR of Popularity Bin 2 Over Time (Taobao RNN)')
ax.grid()
pop2_fig.savefig("mrr_pop2.png")

pop3_fig, ax = plt.subplots()
ax.bar([1,2,3,4,5], mrrs[:,2])
ax.set(xlabel='Time', ylabel='MRR@20',
    title='MRR of Popularity Bin 3 Over Time (Taobao RNN)')
ax.grid()
pop3_fig.savefig("mrr_pop3.png")

pop4_fig, ax = plt.subplots()
ax.bar([1,2,3,4,5], mrrs[:,3])
ax.set(xlabel='Time', ylabel='MRR@20',
    title='MRR of Popularity Bin 4 Over Time (Taobao RNN)')
ax.grid()
pop4_fig.savefig("mrr_pop4.png")

pop5_fig, ax = plt.subplots()
ax.bar([1,2,3,4,5], mrrs[:,4])
ax.set(xlabel='Time', ylabel='MRR@20',
    title='MRR of Popularity Bin 5 Over Time (Taobao RNN)')
ax.grid()
pop5_fig.savefig("mrr_pop5.png")

pop6_fig, ax = plt.subplots()
ax.bar([1,2,3,4,5], mrrs[:,5])
ax.set(xlabel='Time', ylabel='MRR@20',
    title='MRR of Popularity Bin 6 Over Time (Taobao RNN)')
ax.grid()
pop6_fig.savefig("mrr_pop6.png")

pop7_fig, ax = plt.subplots()
ax.bar([1,2,3,4,5], mrrs[:,6])
ax.set(xlabel='Time', ylabel='MRR@20',
    title='MRR of Popularity Bin 7 Over Time (Taobao RNN)')
ax.grid()
pop7_fig.savefig("mrr_pop7.png")

pop8_fig, ax = plt.subplots()
ax.bar([1,2,3,4,5], mrrs[:,7])
ax.set(xlabel='Time', ylabel='MRR@20',
    title='MRR of Popularity Bin 8 Over Time (Taobao RNN)')
ax.grid()
pop8_fig.savefig("mrr_pop8.png") 

