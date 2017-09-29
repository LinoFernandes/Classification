import numpy as np
import pandas as pd
np.random.seed(1977)
import matplotlib.pyplot as plt

BoundsOri = pd.read_csv('/Users/Lino/PycharmProjects/Classification/Snapshots/Scores_Ori.csv',header=None)
BoundsSlow = pd.read_csv('/Users/Lino/PycharmProjects/Classification/Snapshots/SlowBounds_Snap_Final.csv',header=None)
BoundsNeutral = pd.read_csv('/Users/Lino/PycharmProjects/Classification/Snapshots/NeutralBounds_Snap_Final.csv',header=None)
BoundsFast = pd.read_csv('/Users/Lino/PycharmProjects/Classification/Snapshots/FastBounds_Snap_Final.csv',header=None)

ScoresProg = pd.read_csv('/Users/Lino/PycharmProjects/Classification/Snapshots/ScoresSnapProg.txt')
ScoresSnap = pd.read_csv('/Users/Lino/PycharmProjects/Classification/Snapshots/AUC_Test.txt')


Window = [90,180,365]

#1a75ff Slow
#001f4d Slow line

MinOri = min(BoundsOri.iloc[0:3,2])
MaxOri = max(BoundsOri.iloc[0:3,1])
ScoresOri = ScoresSnap.iloc[0:len(ScoresSnap):2,2]

ScoresProg = ScoresProg.iloc[2:len(ScoresProg):3,2]

MinProg = min(BoundsSlow.iloc[:,2])
MaxProb = max(BoundsSlow.iloc[:,1])



LabelMin = min(MinOri,min(ScoresProg))
LabelMax = max(MaxOri,max(ScoresProg))

# Plot the results...ScoresOri
fig, ax = plt.subplots()
#ax.fill_between(np.arange(0,3),BoundsSlow.iloc[:,1] , BoundsSlow.iloc[:,2], color='#1a75ff', alpha=0.5)
ax.fill_between(np.arange(0,3),BoundsOri.iloc[0:3,1] , BoundsOri.iloc[0:3,2], color='grey', alpha=0.5)

ax.plot(np.arange(0,3), ScoresProg, 'o-', label='Neutral',linewidth=3, color = '#0066ff')
ax.plot(np.arange(0,3), ScoresOri, 'o-', label='Ori',linewidth=3, color = 'black')
plt.xticks(np.arange(0,3),['90','180','365'], size=8)
plt.xlim([0-0.1,2+0.1])
plt.yticks((np.round(LabelMin), np.round(LabelMax)),
           (str(int(np.round(LabelMin, 0))), str(int(np.round(LabelMax, 0)))), color='k', size=8)
plt.ylim([LabelMin - 2.5, LabelMax + 2.5])
ax.minorticks_off()
leg=ax.legend(loc='upper center', ncol=2,bbox_to_anchor=((0.5, 1.10)))
leg.get_frame().set_linewidth(0.0)
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(False)
plt.xlabel('Time Window (days)',size =8)
plt.ylabel('AUC', size = 8)
plt.savefig('Snap_Ori_Neutral.png')
plt.close()
