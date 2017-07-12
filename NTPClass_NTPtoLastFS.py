import pandas as pd
import numpy as np
import sys
import weka.core.jvm as jvm
import weka.core.packages as pk
import os
import matplotlib.pyplot as plt

from weka.core.classes import from_commandline

os.environ["JAVA_HOME"] = "/Library/Java/JavaVirtualMachines/jdk1.8.0_131.jdk"

jvm.start(class_path=['/Users/Lino/wekafiles/packages/RerankingSearch/RerankingSearch.jar','/Applications/weka-3-9-1-oracle-jvm.app/Contents/Java/weka.jar'])
print(pk.installed_packages())
Window = np.array([90, 180, 365])

import weka.core.classes as wcc
opt=wcc.split_options("weka.attributeSelection.CfsSubsetEval -P 1 -E 1 -S weka.attributeSelection.RerankingSearch -method 2 -blockSize 20 -rankingMeasure 0 -search \"weka.attributeSelection.BestFirst -D 1 -N 5\"")

counter = -1
NTPspNB = []
NTPspSVM = []
NTPspRF = []

NTPspNB_FS = []
NTPspSVM_FS = []
NTPspRF_FS = []
BeginsTotal = []

window = 90
for ntp in range(2, 6):
    Begins = np.array(range(0, ntp))
    counter += 1
    BeginsTotal.insert(counter, Begins)
    Begins = Begins[::-1]

    roc_NB_FS = []
    roc_SVM_FS = []
    roc_RF_FS = []

    roc_NB = []
    roc_SVM = []
    roc_RF = []

    if sys.platform == "darwin":
        path = '/Users/Lino/PycharmProjects/Preprocessing/NTPtoLast/' + str(ntp) + 'TP'
        directory = '/Users/Lino/PycharmProjects/Classification/FS/'
    elif sys.platform == "win32":
        path = 'C:\\Users\\Lino\\PycharmProjects\\Preprocessing\\NTP\\' + str(ntp) + 'TP'
    sys.path.append(path)
    for classifier in range(0, 3):
        for begin in Begins:
            try:
                from weka.core.converters import Loader

                loader = Loader(classname="weka.core.converters.CSVLoader")
                data = loader.load_file(path + '/' + str(window) + 'd_' + str(begin) + 'to' + str(ntp-1) + '.csv')
                data.class_is_last()

                import weka.core.classes as wcc

                #opt = wcc.split_options("weka.attributeSelection.RerankingSearch -method 2 -blockSize 20 -rankingMeasure 0 -search \"weka.attributeSelection.BestFirst -D 1 -N 5\"")

                #from weka.attribute_selection import ASEvaluation, ASSearch, AttributeSelection
                #evaluation = ASEvaluation(classname="weka.attributeSelection.CfsSubsetEval", options=['-P','1','-E','1'])
                #search = ASSearch(classname="weka.attributeSelection.RerankingSearch", options=opt)
                #search = ASSearch(classname="weka.attributeSelection.BestFirst", options=["-D", "1", "-N", "5"])
                # attsel = AttributeSelection()
                # attsel.search(search)
                # attsel.evaluator(evaluation)
                # attsel.select_attributes(data)
                # print("# attributes: " + str(attsel.number_attributes_selected))
                # print("attributes: " + str(attsel.selected_attributes))
                # print("result string:\n" + attsel.results_string)
                #
                # FullAttributes = np.array(range(34))
                # toBeDeleted = [i for i in range(0,33) if FullAttributes[i] not in attsel.selected_attributes]
                # toBeDeleted.append(33)
                # dataTrain = removeAttributes(dataTrain,toBeDeleted)
                # dataTest = removeAttributes(dataTest,toBeDeleted)
                #
                #
                # def removeAttributes(instaces, toBeDeleted):
                #     from weka.filters import Filter
                #     remove = Filter(classname="weka.filters.unsupervised.attribute.Remove",
                #                     options=["-R", ','.join(list(map(str, toBeDeleted)))])
                #     remove.inputformat(instaces)
                #     newInstaces = remove.filter(instaces)
                #     return newInstaces

                # from weka.filters import Filter
                # Split = Filter(classname='weka.filters.unsupervised.instance.RemovePercentage', options=['-P','30'])
                # Split.inputformat(data)
                # dataTrain = Split.filter(data)
                # Split = Filter(classname='weka.filters.unsupervised.instance.RemovePercentage', options=['-P','30','-V'])
                # Split.inputformat(data)
                # dataTest = Split.filter(data)
                # dataTrain.class_is_last()
                # dataTest.class_is_last()

                from weka.filters import Filter

                NominalToBinary = Filter(classname="weka.filters.unsupervised.attribute.NominalToBinary",
                                         options=["-R", "5,7,8"])
                NumericToNominal = Filter(classname="weka.filters.unsupervised.attribute.NumericToNominal")
                ReplaceMV = Filter(classname="weka.filters.unsupervised.attribute.ReplaceMissingValues")
                ReplaceMV.inputformat(data)
                data = ReplaceMV.filter(data)
                # ReplaceMV.inputformat(dataTest)
                # dataTest=ReplaceMV.filter(dataTest)

                import weka.core.classes as wcc
                opt = wcc.split_options("-S weka.attributeSelection.RerankingSearch -method 2 -blockSize 20 -rankingMeasure 0 -search \"weka.attributeSelection.BestFirst -D 1 -N 5\"")

                FS = Filter(classname="weka.filters.supervised.attribute.AttributeSelection", options=['-E', 'weka.attributeSelection.CfsSubsetEval -P 1 -E 1', '-S', 'weka.attributeSelection.RerankingSearch -method 2 -blockSize 20 -rankingMeasure 0 -search "weka.attributeSelection.GreedyStepwise -T -1.7976931348623157E308 -N -1 -num-slots 1"'])
                FS.inputformat(data)
                FSdata = FS.filter(data)


                from weka.classifiers import Classifier
                if classifier == 0:
                    mapper = Classifier(classname="weka.classifiers.misc.InputMappedClassifier", options=["-W", "weka.classifiers.bayes.NaiveBayes", "--", "-K"])
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    completeName = os.path.join(directory, 'Features_' + str(window) + 'd_' + str(ntp) +'TP.txt')
                    Attr = open(completeName,'a')
                    Attr.write(str(window) + 'd_' + str(begin) + 'to' + str(ntp) + '\n')
                    for attribute in FSdata.attributes():
                        Attr.write(str(attribute) + '\n')
                elif classifier == 1:
                    mapper = Classifier(classname="weka.classifiers.misc.InputMappedClassifier", options=["-W", "weka.classifiers.functions.SMO", "--", "-K","weka.classifiers.functions.supportVector.PolyKernel -E 2.0"])
                else:
                    mapper = Classifier(classname="weka.classifiers.misc.InputMappedClassifier", options=["-W", "weka.classifiers.trees.RandomForest", "--", "-I", "20"])
                # mapper.build_classifier(data)
                # options = ["-K"]
                # cls = Classifier(classname="weka.classifiers.bayes.NaiveBayes")
                # cls.options=options
                # #print(cls.options)
                # cls.build_classifier(dataTrain)



                from weka.classifiers import Evaluation
                from weka.core.classes import Random

                # print("Evaluating NB classifier")
                evaluationFS = Evaluation(data)
                evaluationFS.crossvalidate_model(mapper, FSdata, 10, Random(42))

                evaluation = Evaluation(data)
                evaluation.crossvalidate_model(mapper, data, 10, Random(42))
                if classifier == 0:
                    roc_NB.append(100 * evaluation.area_under_roc(1))
                    roc_NB_FS.append(100 * evaluationFS.area_under_roc(1))

                elif classifier == 1:
                    roc_SVM.append(100 * evaluation.area_under_roc(1))
                    roc_SVM_FS.append(100 * evaluationFS.area_under_roc(1))
                else:
                    roc_RF.append(100 * evaluation.area_under_roc(1))
                    roc_RF_FS.append(100 * evaluationFS.area_under_roc(1))



            except:
                continue

    NTPspNB.insert(counter, roc_NB)
    NTPspSVM.insert(counter, roc_SVM)
    NTPspRF.insert(counter, roc_RF)

    NTPspNB_FS.insert(counter, roc_NB_FS)
    NTPspSVM_FS.insert(counter, roc_SVM_FS)
    NTPspRF_FS.insert(counter, roc_RF_FS)

jvm.stop()

fig, axs = plt.subplots(2, 2, figsize=(15, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace=.8, wspace=.1)
axs = axs.flatten()
for i in range(0, 4):
    labels = []
    aux = []
    Min = np.array([min(NTPspNB[i]), min(NTPspSVM[i]), min(NTPspRF[i]),min(NTPspNB_FS[i]), min(NTPspSVM_FS[i]), min(NTPspRF_FS[i])])
    Max = np.array([max(NTPspNB[i]), max(NTPspSVM[i]), max(NTPspRF[i]),max(NTPspNB_FS[i]), max(NTPspSVM_FS[i]), max(NTPspRF_FS[i])])

    axs[i].plot(BeginsTotal[i], NTPspNB_FS[i], label='NB_' + str(window) + 'd' + ' c/ FS', color='b')
    axs[i].plot(BeginsTotal[i], NTPspSVM_FS[i], label='SVM_' + str(window) + 'd' + ' c/ FS', color='g')
    axs[i].plot(BeginsTotal[i], NTPspRF_FS[i], label='RF_' + str(window) + 'd' + ' c/ FS', color='r')

    axs[i].plot(BeginsTotal[i], NTPspNB[i], label='NB_' + str(window) + 'd', color='b', ls='dashed')
    axs[i].plot(BeginsTotal[i], NTPspSVM[i], label='SVM_' + str(window) + 'd', color='g', ls='dashed')
    axs[i].plot(BeginsTotal[i], NTPspRF[i], label='RF_' + str(window) + 'd', color='r', ls='dashed')
    axs[i].legend()
    for label in range(0, len(NTPspNB[i])):
        labels.append(str(label + 1))
        aux.append(label + 1)
    axs[i].set_xticks(range(0, len(BeginsTotal[i])))
    axs[i].set_xticklabels(labels)
    axs[i].set_xlabel('Número de Snapshots Usados')
    axs[i].set_ylabel('AUC')
    axs[i].set_title('Previsão ' + str(len(NTPspNB[i])) + 'º Snapshot')
    axs[i].set_xlim(-0.05, aux[len(aux) - 1] - 0.95)
    axs[i].set_ylim(min(Min) - 0.5, max(Max) + 0.5)

    major_ticks = np.arange(round(min(Min)), round(max(Max)) + 1, 2)
    minor_ticks = np.arange(round(min(Min)), round(max(Max)) + 1, 0.4)

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
    axs[i].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, borderaxespad=0., prop={'size': 6})
fig.suptitle(str(window) + 'd')
fig.savefig(directory+'FeatureSelection'+str(window) + 'd' + '_NTPtoLast-AUC.png')

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


