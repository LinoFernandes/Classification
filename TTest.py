import pandas as pd
from scipy import stats

data = pd.read_csv('/Users/Lino/PycharmProjects/Classification/NTPClass/New/NTPClassSMOTE_FS_AUC.txt')
prec = pd.read_csv('/Users/Lino/PycharmProjects/Classification/NTPClass/New/NTPClassSMOTE_FS_Precision.txt')
rec = pd.read_csv('/Users/Lino/PycharmProjects/Classification/NTPClass/New/NTPClassSMOTE_FS_Recall.txt')
Window = [90,180,365]
#
# i=2
# for row in range(0,len(data),2):
#     if i in [2,6,10]:
#         print('-----------------------'+str(Window[int(i/4)])+'-----------------------\n')
#     print('%%%%%   ' + str(i) + '   %%%%%\n')
#     First = rec.iloc[row, 3:].values
#     Last = rec.iloc[row+1,3:].values
#     print(stats.ttest_rel(First,Last))
#     i+=1

#
#
# for beg in [0,8,16]:
#
#     print('-----------------------'+str(Window[int(beg/8)])+'-----------------------\n')
#
#     dataset=['2','3','4','5']
#     First = [i for j in prec.iloc[beg:beg+1,3:].values for i in j]
#     Last = [i for j in prec.iloc[beg+1:beg+2,3:].values for i in j]
#     print('%%%%% AUC %%%%\n')
#     i = 2
#     for col in range(1,len(dataset)):
#         print('%%%%%   '+dataset[0] +'vs'+ dataset[col]+'   %%%%%\n')
#         z = stats.ttest_rel(First,prec.iloc[int(beg) + i,3:].values)
#         z1 = stats.ttest_rel(Last,prec.iloc[int(beg) + i + 1,3:].values)
#         print('First:'+str(z)+'\n')
#         print('Last:'+str(z) + '\n')
#
#         #print(First)
#         #print(Last)
#
#         #print(data.iloc[beg+i ,3:].values)
#         #print(data.iloc[beg+i+1,3:].values)
#         i +=2

    # First = [i for j in prec.iloc[beg:beg + 1, 3:].values for i in j]
    # Last = [i for j in prec.iloc[beg + 1:beg + 2, 3:].values for i in j]
    # print('%%%%% PRECISION %%%%\n')
    # for col in range(1,len(dataset)):
    #     print('%%%%%   '+dataset[0] +'vs'+ dataset[col]+'   %%%%%\n')
    #     z = stats.ttest_rel(First, [i for j in prec.iloc[beg + int(col):beg + int(col) + 1, 3:].values for i in j])
    #     z1 = stats.ttest_rel(Last, [i for j in prec.iloc[beg + int(col) + 1:beg + int(col) + 2, 3:].values for i in j])
    #     print('First:' + str(z) + '\n')
    #     print('Last:' + str(z) + '\n')
    #
    # First = [i for j in rec.iloc[beg:beg + 1, 3:].values for i in j]
    # Last = [i for j in rec.iloc[beg + 1:beg + 2, 3:].values for i in j]
    # print('%%%%% Recall %%%%\n')
    # for col in range(1,len(dataset)):
    #     print('%%%%%   ' + dataset[0] + 'vs' + dataset[col] + '   %%%%%\n')
    #     z = stats.ttest_rel(First, [i for j in rec.iloc[beg + int(col):beg + int(col) + 1, 3:].values for i in j])
    #     z1 = stats.ttest_rel(Last, [i for j in rec.iloc[beg + int(col) + 1:beg + int(col) + 2, 3:].values for i in j])
    #     print('First:' + str(z) + '\n')
    #     print('Last:' + str(z) + '\n')

for i in range(4,6):
    print(stats.ttest_rel(data.iloc[:,3].values,data.iloc[:,i].values))
    print(stats.ttest_rel(prec.iloc[:, 2].values, prec.iloc[:,i-1].values))
    print(stats.ttest_rel(rec.iloc[:, 3].values, rec.iloc[:,i].values))

    print('%%%%%%%%%')

