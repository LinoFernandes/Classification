import pandas as pd
import numpy as np
import sys
import weka.core.jvm as jvm
import weka.core.packages as pk
import os
import matplotlib.pyplot as plt

from weka.core.classes import from_commandline

os.environ["JAVA_HOME"] = "/Library/Java/JavaVirtualMachines/jdk1.8.0_131.jdk"

jvm.start(class_path=['/Users/Lino/wekafiles/packages/SMOTE/SMOTE.jar','/Users/Lino/wekafiles/packages/RerankingSearch/RerankingSearch.jar','/Applications/weka-3-9-1-oracle-jvm.app/Contents/Java/weka.jar'])
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

Perf = open('/Users/Lino/PycharmProjects/Classification/FS/' + 'NTPtoLastSMOTE_50_FS.txt', 'a')


for window in Window:
    counter = -1

    #roc_NB_K = []
    #roc_SVM_1 = []
    #roc_SVM_2 = []
    #roc_SVM_3 = []
    #roc_RF_5 = []
    #roc_RF_10 = []
    #roc_RF_15 = []
    #roc_RF_20 = []
    for ntp in range(2,7):
        roc = []
        title = [str(window)]
        title.append('#' + str(ntp))
        Begins = np.array(range(0,ntp))
        counter += 1
        BeginsTotal.insert(counter, Begins)
        Begins = Begins[::-1]

        if sys.platform == "darwin":
            path = '/Users/Lino/PycharmProjects/Preprocessing/NTPtoLast/' + str(ntp) + 'TP'
            directory = '/Users/Lino/PycharmProjects/Classification/FS/'
            if not os.path.exists(directory):
                os.makedirs(directory)
        elif sys.platform == "win32":
            path = 'C:\\Users\\Lino\\PycharmProjects\\Preprocessing\\NTP\\' + str(ntp) + 'TP'
        sys.path.append(path)
        for classifier in range(0,3):
            aux1 = []
            aux2 = []
            aux3 = []
            aux4 = []
            for begin in Begins:
                try:
                    from weka.core.converters import Loader

                    loader = Loader(classname="weka.core.converters.CSVLoader")
                    data = loader.load_file(path + '/' + str(window) + 'd_' + str(begin) + 'to' + str(ntp-1) + '.csv')
                    data.class_is_last()
                    for fold in range(1, 11):
                        from weka.filters import Filter

                        StratifiedCV = Filter(classname="weka.filters.supervised.instance.StratifiedRemoveFolds",
                                              options=['-S', '42', '-N', '10', '-F', str(fold)])
                        StratifiedCV.inputformat(data)
                        dataTest = StratifiedCV.filter(data)

                        StratifiedCV = Filter(classname="weka.filters.supervised.instance.StratifiedRemoveFolds",
                                              options=['-S', '42', '-V', '-N', '10', '-F', str(fold)])
                        StratifiedCV.inputformat(data)
                        dataTrain = StratifiedCV.filter(data)

                        toBeRemoved = []
                        for attribute in range(0, dataTrain.attributes().data.class_index):
                            if dataTrain.attribute_stats(
                                    attribute).missing_count == dataTrain.attributes().data.num_instances and dataTest.attribute_stats(
                                attribute).missing_count == dataTest.attributes().data.num_instances:
                                sys.exit("Fold has full missing column")
                            if (dataTrain.attribute_stats(
                                    attribute).missing_count / dataTrain.attributes().data.num_instances) > 0.5 and (
                                dataTest.attribute_stats(
                                        attribute).missing_count / dataTest.attributes().data.num_instances) > 0.5:
                                toBeRemoved.append(str(attribute))

                        Remove = Filter(classname="weka.filters.unsupervised.attribute.Remove",
                                        options=['-R', ','.join(toBeRemoved)])
                        Remove.inputformat(dataTrain)
                        dataTrain = Remove.filter(dataTrain)
                        dataTest = Remove.filter(dataTest)

                        import weka.core.classes as wcc

                        FS = Filter(classname="weka.filters.supervised.attribute.AttributeSelection",
                                    options=['-E', 'weka.attributeSelection.CfsSubsetEval -P 1 -E 1', '-S',
                                             'weka.attributeSelection.RerankingSearch -method 2 -blockSize 20 -rankingMeasure 0 -search "weka.attributeSelection.GreedyStepwise -T -1.7976931348623157E308 -N -1 -num-slots 1"'])
                        FS.inputformat(dataTrain)
                        dataTrain = FS.filter(dataTrain)
                        dataTest = FS.filter(dataTest)

                        ReplaceMV = Filter(classname="weka.filters.unsupervised.attribute.ReplaceMissingValues")
                        ReplaceMV.inputformat(dataTrain)
                        dataTrain = ReplaceMV.filter(dataTrain)
                        ReplaceMV.inputformat(dataTest)
                        dataTest = ReplaceMV.filter(dataTest)
                        # ReplaceMV.inputformat(dataTest)
                        # dataTest=ReplaceMV.filter(dataTest)

                        SMOTE = Filter(classname="weka.filters.supervised.instance.SMOTE", options=['-P', '50.0'])
                        SMOTE.inputformat(dataTrain)
                        dataTrain = SMOTE.filter(dataTrain)

                        dataTrain.class_is_last()
                        dataTest.class_is_last()

                        from weka.classifiers import Evaluation
                        from weka.core.classes import Random
                        from weka.classifiers import Classifier

                        if classifier == 0:
                            for kernel in range(0, 2):
                                if kernel == 0:
                                    mapper = Classifier(classname="weka.classifiers.misc.InputMappedClassifier",
                                                        options=["-W", "weka.classifiers.bayes.NaiveBayes"])
                                    Class = 'NaiveBayes'
                                    mapper.build_classifier(dataTrain)
                                    evaluation = Evaluation(dataTrain)
                                    evaluation.test_model(mapper, dataTest)
                                    aux1.append(evaluation.area_under_roc(1) * 100)
                                    if fold == 10:
                                        title.append('NB_' + str(begin) + 'to' + str(ntp))
                                        roc.append(str(round(np.mean(aux1),2)))
                                else:
                                    mapper = Classifier(classname="weka.classifiers.misc.InputMappedClassifier",
                                                        options=["-W", "weka.classifiers.bayes.NaiveBayes", "--", "-K"])
                                    Class = 'NaiveBayes'
                                    mapper.build_classifier(dataTrain)
                                    evaluation = Evaluation(dataTrain)
                                    evaluation.test_model(mapper, dataTest)
                                    aux2.append(evaluation.area_under_roc(1) * 100)
                                    if fold == 10:
                                        title.append('NB_K_' + str(begin) + 'to' + str(ntp))
                                        roc.append(str(round(np.mean(aux2),2)))
                        elif classifier == 1:
                            for degree in range(1, 4):
                                mapper = Classifier(classname="weka.classifiers.misc.InputMappedClassifier",
                                                    options=["-W", "weka.classifiers.functions.SMO", "--", "-K",
                                                             "weka.classifiers.functions.supportVector.PolyKernel -E " + str(
                                                                 degree)])
                                Class = 'SVM'
                                mapper.build_classifier(dataTrain)
                                evaluation = Evaluation(dataTrain)
                                evaluation.test_model(mapper, dataTest)
                                if degree == 1:
                                    aux1.append(evaluation.area_under_roc(1) * 100)
                                    if fold == 10:
                                        title.append('SVM_' + str(degree) + '_' + str(begin) + 'to' + str(ntp))
                                        roc.append(str(round(np.mean(aux1),2)))
                                elif degree == 2:
                                    aux2.append(evaluation.area_under_roc(1) * 100)
                                    if fold == 10:
                                        title.append('SVM_' + str(degree) + '_' + str(begin) + 'to' + str(ntp))
                                        roc.append(str(round(np.mean(aux2),2)))
                                else:
                                    aux3.append(evaluation.area_under_roc(1) * 100)
                                    if fold == 10:
                                        title.append('SVM_' + str(degree) + '_' + str(begin) + 'to' + str(ntp))
                                        roc.append(str(round(np.mean(aux3),2)))
                        else:
                            for numTrees in np.arange(5, 25, 5):
                                mapper = Classifier(classname="weka.classifiers.misc.InputMappedClassifier",
                                                    options=["-W", "weka.classifiers.trees.RandomForest", "--", "-I",
                                                             str(numTrees)])
                                Class = 'RF'
                                mapper.build_classifier(dataTrain)
                                evaluation = Evaluation(dataTrain)
                                evaluation.test_model(mapper, dataTest)
                                if numTrees == 5:
                                    aux1.append(evaluation.area_under_roc(1) * 100)
                                    if fold == 10:
                                        title.append('RF_' + str(numTrees) + '_' + str(begin) + 'to' + str(ntp))
                                        roc.append(str(round(np.mean(aux1),2)))
                                elif numTrees == 10:
                                    aux2.append(evaluation.area_under_roc(1) * 100)
                                    if fold == 10:
                                        title.append('RF_' + str(numTrees) + '_' + str(begin) + 'to' + str(ntp))
                                        roc.append(str(round(np.mean(aux2),2)))
                                elif numTrees == 15:
                                    aux3.append(evaluation.area_under_roc(1) * 100)
                                    if fold == 10:
                                        title.append('RF_' + str(numTrees) + '_' + str(begin) + 'to' + str(ntp))
                                        roc.append(str(round(np.mean(aux3),2)))
                                else:
                                    aux4.append(evaluation.area_under_roc(1) * 100)
                                    if fold == 10:
                                        title.append('RF_' + str(numTrees) + '_' + str(begin) + 'to' + str(ntp))
                                        roc.append(str(round(np.mean(aux4),2)))

                except:
                    continue
        fulltitle = ','.join(title)
        Perf.write(fulltitle + '\n')
        fullroc = ','.join(roc)
        Perf.write(str(window) + ',' + str(ntp) + ',' + fullroc + '\n')

jvm.stop()

# fig, axs = plt.subplots(2, 2, figsize=(15, 6), facecolor='w', edgecolor='k')
# fig.subplots_adjust(hspace=.8, wspace=.1)
# axs = axs.flatten()
# for i in range(0, 4):
#     labels = []
#     aux = []
#     Min = np.array([min(NTPspNB[i]), min(NTPspSVM[i]), min(NTPspRF[i]),min(NTPspNB_FS[i]), min(NTPspSVM_FS[i]), min(NTPspRF_FS[i])])
#     Max = np.array([max(NTPspNB[i]), max(NTPspSVM[i]), max(NTPspRF[i]),max(NTPspNB_FS[i]), max(NTPspSVM_FS[i]), max(NTPspRF_FS[i])])
#
#     axs[i].plot(BeginsTotal[i], NTPspNB_FS[i], label='NB_' + str(window) + 'd' + ' c/ FS', color='b')
#     axs[i].plot(BeginsTotal[i], NTPspSVM_FS[i], label='SVM_' + str(window) + 'd' + ' c/ FS', color='g')
#     axs[i].plot(BeginsTotal[i], NTPspRF_FS[i], label='RF_' + str(window) + 'd' + ' c/ FS', color='r')
#
#     axs[i].plot(BeginsTotal[i], NTPspNB[i], label='NB_' + str(window) + 'd', color='b', ls='dashed')
#     axs[i].plot(BeginsTotal[i], NTPspSVM[i], label='SVM_' + str(window) + 'd', color='g', ls='dashed')
#     axs[i].plot(BeginsTotal[i], NTPspRF[i], label='RF_' + str(window) + 'd', color='r', ls='dashed')
#     axs[i].legend()
#     for label in range(0, len(NTPspNB[i])):
#         labels.append(str(label + 1))
#         aux.append(label + 1)
#     axs[i].set_xticks(range(0, len(BeginsTotal[i])))
#     axs[i].set_xticklabels(labels)
#     axs[i].set_xlabel('Número de Snapshots Usados')
#     axs[i].set_ylabel('AUC')
#     axs[i].set_title('Previsão ' + str(len(NTPspNB[i])) + 'º Snapshot')
#     axs[i].set_xlim(-0.05, aux[len(aux) - 1] - 0.95)
#     axs[i].set_ylim(min(Min) - 0.5, max(Max) + 0.5)
#
#     major_ticks = np.arange(round(min(Min)), round(max(Max)) + 1, 2)
#     minor_ticks = np.arange(round(min(Min)), round(max(Max)) + 1, 0.4)
#
#     axs[i].set_yticks(major_ticks)
#     axs[i].set_yticks(minor_ticks, minor=True)
#     axs[i].tick_params(labelsize='6')
#
#     # and a corresponding grid
#
#     axs[i].grid(which='both')
#
#     # or if you want differnet settings for the grids:
#     axs[i].grid(which='minor', alpha=0.2)
#     axs[i].grid(which='major', alpha=0.5)
#     axs[i].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, borderaxespad=0., prop={'size': 6})
# fig.suptitle(str(window) + 'd')
# fig.savefig(directory+'FeatureSelection'+str(window) + 'd' + '_NTPtoLast-AUC.png')




                #
                # from weka.classifiers import Classifier
                # if classifier == 0:
                #     mapper = Classifier(classname="weka.classifiers.misc.InputMappedClassifier", options=["-W", "weka.classifiers.bayes.NaiveBayes", "--", "-K"])
                #     if not os.path.exists(directory):
                #         os.makedirs(directory)
                #     completeName = os.path.join(directory, 'Features_' + str(window) + 'd_' +str(begin)+'to'+ str(ntp) +'TP.txt')
                #     Attr = open(completeName,'a')
                #     Attr.write(str(window) + 'd_' + str(begin) + 'to' + str(ntp) + '\n')
                #     for attribute in FSdata.attributes():
                #         aux = str(attribute)
                #         aux = aux.split()
                #         Attr.write(aux[1] + '\n')
                # elif classifier == 1:
                #     mapper = Classifier(classname="weka.classifiers.misc.InputMappedClassifier", options=["-W", "weka.classifiers.functions.SMO", "--", "-K","weka.classifiers.functions.supportVector.PolyKernel -E 2.0"])
                # else:
                #     mapper = Classifier(classname="weka.classifiers.misc.InputMappedClassifier", options=["-W", "weka.classifiers.trees.RandomForest", "--", "-I", "20"])