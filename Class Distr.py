import pandas as pd
import numpy as np

Scores = open('/Users/Lino/PycharmProjects/Classification/ClassDistrNTPProg.txt', 'a')
for window in ['90', '180', '365']:
    for ntp in range(1,4):
        data = pd.read_csv('/Users/Lino/PycharmProjects/Preprocessing/NTP/New2/Slow/'+str(ntp)+'TP/'+str(window)+'d_'+str(ntp)+'.csv')
        aux = data.groupby('Evolution_'+str(ntp-1))['Evolution_'+str(ntp-1)].count().values
        Scores.write(str(window)+' & '+str(ntp)+'&'+str(len(data))+'&'+str(aux[0]) + '(' +str(np.round(aux[0]/len(data),4)*100) + '%)'+'&'+str(aux[1]) + '(' +str(np.round(aux[1]/len(data),4)*100) + '%)\n')