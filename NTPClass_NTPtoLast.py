import pandas as pd
import numpy as np
import sys
import weka.core.jvm as jvm
import os
import matplotlib.pyplot as plt

from weka.core.classes import from_commandline

os.environ["JAVA_HOME"] = "/Library/Java/JavaVirtualMachines/jdk1.8.0_131.jdk"
jvm.start(class_path=['/Users/Lino/wekafiles/packages/SMOTE/SMOTE.jar','/Users/Lino/wekafiles/packages/RerankingSearch/RerankingSearch.jar','/Applications/weka-3-9-1-oracle-jvm.app/Contents/Java/weka.jar'])

Window = np.array([90, 180, 365])

if sys.platform == "darwin":
    completeName = os.path.join('/Users/Lino/Desktop','PerformanceNB.txt')
elif sys.platform == "win32":
    completeName = os.path.join('C:\\Users\\Lino\\Desktop','PerformanceNB.txt')
Perf = open(completeName,'w')

counter = -1
NTPspNB = []
NTPspSVM = []
NTPspRF = []
BeginsTotal = []

window = 365
for ntp in range(2,6):
    Begins = np.array(range(0,ntp))
    counter += 1
    BeginsTotal.insert(counter, Begins)
    Begins = Begins[::-1]
    
    roc_NB = []
    roc_SVM = []
    roc_RF = []

    if sys.platform == "darwin":
        path = '/Users/Lino/PycharmProjects/Preprocessing/NTPtoLast/' + str(ntp) + 'TP'
        directory = '/Users/Lino/PycharmProjects/Classification/NTPtoLastClass/'
        if not os.path.exists(directory):
            os.makedirs(directory)
    elif sys.platform == "win32":
        path = 'C:\\Users\\Lino\\PycharmProjects\\Preprocessing\\NTP\\' + str(ntp) + 'TP'
    sys.path.append(path)
    for classifier in range(0,3):
        for begin in Begins:
            try:
                from weka.core.converters import Loader

                loader = Loader(classname="weka.core.converters.CSVLoader")
                data = loader.load_file(path + '/' + str(window) + 'd_' + str(begin) + 'to' + str(ntp-1) + '.csv')
                data.class_is_last()

                from weka.filters import Filter
                NominalToBinary = Filter(classname="weka.filters.unsupervised.attribute.NominalToBinary", options=["-R", "5,7,8"])
                NumericToNominal = Filter(classname="weka.filters.unsupervised.attribute.NumericToNominal")
                ReplaceMV = Filter(classname="weka.filters.unsupervised.attribute.ReplaceMissingValues")
                ReplaceMV.inputformat(data)
                data = ReplaceMV.filter(data)


                from weka.classifiers import Classifier
                if classifier == 0:
                    mapper = Classifier(classname="weka.classifiers.misc.InputMappedClassifier", options=["-W", "weka.classifiers.bayes.NaiveBayes", "--", "-K"])
                elif classifier == 1:
                    mapper = Classifier(classname="weka.classifiers.misc.InputMappedClassifier", options=["-W", "weka.classifiers.functions.SMO", "--", "-K","weka.classifiers.functions.supportVector.PolyKernel -E 2.0"])
                else:
                    mapper = Classifier(classname="weka.classifiers.misc.InputMappedClassifier", options=["-W", "weka.classifiers.trees.RandomForest", "--", "-I","20"])
                #mapper.build_classifier(data)
                # options = ["-K"]
                # cls = Classifier(classname="weka.classifiers.bayes.NaiveBayes")
                # cls.options=options
                # #print(cls.options)
                # cls.build_classifier(dataTrain)



                from weka.classifiers import Evaluation
                from weka.core.classes import Random
                #print("Evaluating NB classifier")
                evaluation = Evaluation(data)
                evaluation.crossvalidate_model(mapper, data, 10, Random(42))
                if classifier == 0:
                    roc_NB.append(100*evaluation.area_under_roc(1))
                elif classifier == 1:
                    roc_SVM.append(100*evaluation.area_under_roc(1))
                else:
                    roc_RF.append(100*evaluation.area_under_roc(1))


            except:
                continue

    NTPspNB.insert(counter,roc_NB)
    NTPspSVM.insert(counter,roc_SVM)
    NTPspRF.insert(counter,roc_RF)

jvm.stop()

fig, axs = plt.subplots(2,2, figsize=(15, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .8, wspace=.1)
axs = axs.flatten()
for i in range(0,4):
    labels = []
    aux = []
    Min=np.array([min(NTPspNB[i]),min(NTPspSVM[i]),min(NTPspRF[i])])
    Max=np.array([max(NTPspNB[i]),max(NTPspSVM[i]),max(NTPspRF[i])])

    axs[i].plot(BeginsTotal[i],NTPspNB[i],label='NB_'+str(window)+'d')
    axs[i].plot(BeginsTotal[i],NTPspSVM[i],label='SVM_'+str(window)+'d')
    axs[i].plot(BeginsTotal[i],NTPspRF[i],label='RF_'+str(window)+'d')
    axs[i].legend()
    for label in range(0,len(NTPspNB[i])):
        labels.append(str(label+1))
        aux.append(label+1)
    axs[i].set_xticks(range(0,len(BeginsTotal[i])))
    axs[i].set_xticklabels(labels)
    axs[i].set_xlabel('Número de Snapshots Usados')
    axs[i].set_ylabel('AUC')
    axs[i].set_title('Previsão ' + str(len(NTPspNB[i])) + 'º Snapshot')
    axs[i].set_xlim(-0.05,aux[len(aux)-1]-0.95)
    axs[i].set_ylim(min(Min)-0.5,max(Max)+0.5)

    major_ticks = np.arange(round(min(Min)), round(max(Max))+1, 2)
    minor_ticks = np.arange(round(min(Min)), round(max(Max))+1, 0.4)

    # if len(NTPspNB[i])<5:
    #     major_ticks = np.arange(round(min(Min)), round(max(Max))+1, 2)
    #     minor_ticks = np.arange(round(min(Min)), round(max(Max))+1, 0.4)
    # else:
    #     major_ticks = np.arange(round(min(Min)), round(max(Max))+1, 5)
    #     minor_ticks = np.arange(round(min(Min)), round(max(Max))+1, 1)

    axs[i].set_yticks(major_ticks)
    axs[i].set_yticks(minor_ticks, minor=True)
    axs[i].tick_params(labelsize='6')

    # and a corresponding grid

    axs[i].grid(which='both')

    # or if you want differnet settings for the grids:
    axs[i].grid(which='minor', alpha=0.2)
    axs[i].grid(which='major', alpha=0.5)
    axs[i].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=1, borderaxespad=0., prop={'size':6})
fig.suptitle(str(window) + 'd')
fig.savefig(directory+str(window) + 'd' +'_NTPtoLast-AUC.png')

# for ntp in range(2,6):
#     fig, axs = plt.subplots(2,2, figsize=(15, 6), facecolor='w', edgecolor='k')
#     fig.subplots_adjust(hspace = .5, wspace=.1)
#     counter = -1
#     for i in range(0,2):
#         for j in range(0,2):
#             counter += 1
#             labels = []
#             print(BeginsTotal[i])
#             axs[i,j].plot(BeginsTotal[i],NTPspNB[i],label='NB_90d')
#             axs[i,j].plot(BeginsTotal[i],NTPspSVM[i],label='NB_180d')
#             axs[i,j].plot(BeginsTotal[i],NTPspRF[i],label='NB_365d')
#             axs[i,j].legend()
#             for label in range(0,len(BeginsTotal[counter])):
#                 labels.append(str(Begins[label]) + 'to' + str(ntp))
#             axs[i,j].xticks(range(0,len(BeginsTotal[counter])))
#             axs[i,j].set_xlabel('nTPto'+str(ntp)+'_NB-AUC')
#             axs[i,j].set_xlabel('NB-AUC')
#             axs[i,j].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=2, mode="expand", borderaxespad=0., fontsize='11')
#     axs.savefig('SVM_nTPto'+str(ntp)+'-AUC.png')
#     axs.close()

# plt.plot(Begins, roc_90, label='SVM_90d')
# plt.plot(Begins, roc_180, label='SVM_180')
# plt.plot(Begins, roc_365, label='SVM_365')
# for label in range(0,len(Begins)):
#     labels.append(str(Begins[label]) + 'to' + str(ntp))
# plt.xticks(Begins,labels)
# plt.xlabel('nTPto'+str(ntp)+'_SVM-AUC')
# plt.ylabel('SVM-AUC')
# plt.grid()
# plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
#            ncol=2, mode="expand", borderaxespad=0., fontsize='11')
# plt.savefig('SVM_nTPto'+str(ntp)+'-AUC.png')
# plt.close()


