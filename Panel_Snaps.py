import numpy as np
import pandas as pd
np.random.seed(1977)
import matplotlib.pyplot as plt

BoundsOri = pd.read_csv('/Users/Lino/PycharmProjects/Classification/Snapshots/Scores_Ori.csv',header=None)
BoundsMVI = pd.read_csv('/Users/Lino/PycharmProjects/Classification/Snapshots/Scores_Ori.csv',header=None)

ScoresSnap = pd.read_csv('/Users/Lino/PycharmProjects/Classification/Snapshots/AUC_Test.txt')
Window = [90,180,365]



Min = min(BoundsOri.iloc[0:3,2])
Max = max(BoundsOri.iloc[0:4,1])

ScoreMin= min(Scores.iloc[0:len(Scores):2,2])
ScoreMax= max(Scores.iloc[0:len(Scores):2,2])

LabelMin = min(Min,ScoreMin)
LabelMax = max(Max,ScoreMax)

# Plot the results...
fig, ax = plt.subplots()
ax.fill_between(np.arange(0,3),BoundsOri.iloc[0:3,1] , BoundsOri.iloc[0:3,2], color='grey', alpha=0.5)
ax.plot(np.arange(0,3), Scores.iloc[0:len(Scores):2,2], 'o-', label='RF',linewidth=3, color = 'black')
plt.xticks(np.arange(0,3),['90','180','365'], size=8)
plt.xlim([0-0.1,2+0.1])
plt.yticks((np.round(LabelMin), np.round(LabelMax)),
           (str(int(np.round(LabelMin, 0))), str(int(np.round(LabelMax, 0)))), color='k', size=8)
plt.ylim([LabelMin - 2.5, LabelMax + 2.5])
ax.minorticks_off()
#leg=ax.legend(loc='upper center', ncol=2,bbox_to_anchor=((0.5, 1.10)))
#leg.get_frame().set_linewidth(0.0)
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(True)
plt.xlabel('Time Window (days)',size =8)
plt.ylabel('AUC', size = 8)
plt.savefig('Snap_Ori_NB.png')
plt.close()

Min = min(BoundsOri.iloc[0:3, 4])
Max = max(BoundsOri.iloc[0:4, 3])

ScoreMin = min(Scores.iloc[0:len(Scores):2, 3])
ScoreMax = max(Scores.iloc[0:len(Scores):2, 3])

LabelMin = min(Min, ScoreMin)
LabelMax = max(Max, ScoreMax)

# Plot the results...
fig, ax = plt.subplots()
ax.fill_between(np.arange(0, 3), BoundsOri.iloc[0:3, 3], BoundsOri.iloc[0:3, 4], color='grey', alpha=0.5)
ax.plot(np.arange(0, 3), Scores.iloc[0:len(Scores):2, 3], 'o-', label='RF', linewidth=3, color='black')
plt.xticks(np.arange(0, 3), ['90', '180', '365'], size=8)
plt.xlim([0 - 0.1, 2 + 0.1])
plt.tick_params(
    axis='x',  # changes apply to the x-axis
    which='both',  # both major and minor ticks are affected
    bottom='off',  # ticks along the bottom edge are off
    labelbottom='off')
plt.yticks((np.round(LabelMin), np.round(LabelMax)),
           (str(int(np.round(LabelMin, 0))), str(int(np.round(LabelMax, 0)))), color='k', size=8)
#plt.ylim([LabelMin - 5, LabelMax + 5])
ax.minorticks_off()
# leg=ax.legend(loc='upper center', ncol=2,bbox_to_anchor=((0.5, 1.10)))
# leg.get_frame().set_linewidth(0.0)
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.savefig('Snap_Ori_RF.png')
plt.close()