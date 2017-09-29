import numpy as np
import pandas as pd
np.random.seed(1977)
import matplotlib.pyplot as plt

BoundsLast = pd.read_csv('/Users/Lino/PycharmProjects/Classification/NTPClass/New2/NTPClass_BoundsLast.csv',header=None)
BoundsFirst = pd.read_csv('/Users/Lino/PycharmProjects/Classification/NTPClass/New2/NTPClass_BoundsFirst.csv',header=None)

ScoresLast = pd.read_csv('/Users/Lino/PycharmProjects/Classification/NTPClass/New/HeldOut/ScoresLast.csv',header=None)
ScoresFirst = pd.read_csv('/Users/Lino/PycharmProjects/Classification/NTPClass/New/HeldOut/Scores.csv',header=None)

Window = [90,180,365]

for i in np.arange(0,12,4):
    if i ==0:
        Min = min(BoundsFirst.iloc[0:4,4])
        Max = max(BoundsFirst.iloc[0:4,3])

        ScoreMin= min(ScoresFirst.iloc[0:4,2])
        ScoreMax= max(ScoresFirst.iloc[0:4,2])

        LabelMin = min(Min,ScoreMin)
        LabelMax = max(Max,ScoreMax)

        # Plot the results...
        fig, ax = plt.subplots()
        ax.fill_between(np.arange(2,6),BoundsFirst.iloc[0:4,3] , BoundsFirst.iloc[0:4,4], color='grey', alpha=0.5)
        ax.plot(np.arange(2,6), ScoresFirst.iloc[0:4,2], 'o-', label='RF',linewidth=3, color = 'black')
        plt.xticks([2,3,4,5], size=8)
        plt.xlim([2-0.1,5+0.1])
        plt.yticks((np.round(LabelMin), np.round(LabelMax)),
                   (str(int(np.round(LabelMin, 0))), str(int(np.round(LabelMax, 0)))), color='k', size=8)
        plt.ylim([LabelMin - 5, LabelMax + 5])
        ax.minorticks_off()
        #leg=ax.legend(loc='upper center', ncol=2,bbox_to_anchor=((0.5, 1.10)))
        #leg.get_frame().set_linewidth(0.0)
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(True)
        plt.xlabel('Timepoints')
        plt.ylabel('AUC', size = 8)
        plt.savefig('RF-' + str(Window[int(i / 4)]) + '_First.png')
        plt.close()

    else:
        Min = min(BoundsFirst.iloc[0+i:4+i, 4])
        Max = max(BoundsFirst.iloc[0+i:4+i, 3])

        ScoreMin = min(ScoresFirst.iloc[0+i:4+i, 2])
        ScoreMax = max(ScoresFirst.iloc[0+i:4+i, 2])

        LabelMin = min(Min, ScoreMin)
        LabelMax = max(Max, ScoreMax)

        # Plot the results...
        fig, ax = plt.subplots()
        ax.fill_between(np.arange(2, 6), BoundsFirst.iloc[0+i:4+i, 3], BoundsFirst.iloc[0+i:4+i, 4], color='grey', alpha=0.5)
        ax.plot(np.arange(2, 6), ScoresFirst.iloc[0+i:4+i, 2], 'o-', label='RF', linewidth=3, color='black')
        plt.xticks([2, 3, 4, 5], size=8)
        plt.xlim([2 - 0.1, 5 + 0.1])
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
        plt.savefig('RF-'+str(Window[int(i / 4)])+ '_First.png')
        plt.close()