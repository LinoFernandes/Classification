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
MV = []
# roc_90 = []
# spec_90 = []
# sens_90 = []
#
# roc_180 = []
# spec_180 = []
# sens_180 = []
#
# roc_365 = []
# spec_365 = []
# sens_365 = []
directory = '/Users/Lino/PycharmProjects/Classification/NTPClass/New/'
if not os.path.exists(directory):
    os.makedirs(directory)
#Perf = open(directory + 'NTPClassSMOTE_FS_AUC.txt', 'a')
#Recall = open(directory + 'NTPClassSMOTE_FS_Recall.txt', 'a')
#Precision = open(directory + 'NTPClassSMOTE_FS_Precision.txt', 'a')
Scores = open(directory + 'NTPClassSMOTE_FS_AUC_Scores.csv', 'a')
ScoresLast = open(directory + 'NTPClassSMOTE_FS_AUC_Scores_Last.csv', 'a')

for window in Window:
    for ntp in range(2, 7):
        #Perf.write('k,#TP,%SMOTE,NB,NB_K,SVM_1,SVM_2,SVM_3,RF_5,RF_10,RF_15,RF_20\n')
        #Recall.write('k,#TP,%SMOTE,NB,NB_K,SVM_1,SVM_2,SVM_3,RF_5,RF_10,RF_15,RF_20\n')
        #Precision.write('k,#TP,%SMOTE,NB,NB_K,SVM_1,SVM_2,SVM_3,RF_5,RF_10,RF_15,RF_20\n')
        for smote in [0, 50, 100, 150, 200]:
            roc_NB = []
            roc_NB_K = []
            roc_SVM_1 = []
            roc_SVM_2 = []
            roc_SVM_3 = []
            roc_RF_5 = []
            roc_RF_10 = []
            roc_RF_15 = []
            roc_RF_20 = []

            roc_NB_Last = []
            roc_NB_K_Last = []
            roc_SVM_1_Last = []
            roc_SVM_2_Last = []
            roc_SVM_3_Last = []
            roc_RF_5_Last = []
            roc_RF_10_Last = []
            roc_RF_15_Last = []
            roc_RF_20_Last = []

            recall_NB = []
            recall_NB_K = []
            recall_SVM_1 = []
            recall_SVM_2 = []
            recall_SVM_3 = []
            recall_RF_5 = []
            recall_RF_10 = []
            recall_RF_15 = []
            recall_RF_20 = []

            recall_NB_Last = []
            recall_NB_K_Last = []
            recall_SVM_1_Last = []
            recall_SVM_2_Last = []
            recall_SVM_3_Last = []
            recall_RF_5_Last = []
            recall_RF_10_Last = []
            recall_RF_15_Last = []
            recall_RF_20_Last = []

            precision_NB = []
            precision_NB_K = []
            precision_SVM_1 = []
            precision_SVM_2 = []
            precision_SVM_3 = []
            precision_RF_5 = []
            precision_RF_10 = []
            precision_RF_15 = []
            precision_RF_20 = []

            precision_NB_Last = []
            precision_NB_K_Last = []
            precision_SVM_1_Last = []
            precision_SVM_2_Last = []
            precision_SVM_3_Last = []
            precision_RF_5_Last = []
            precision_RF_10_Last = []
            precision_RF_15_Last = []
            precision_RF_20_Last = []
            for classifier in range(0, 1):
                if sys.platform == "darwin":
                    path = '/Users/Lino/PycharmProjects/Preprocessing/NTP/New/' + str(ntp) + 'TP'
                    lastpath = '/Users/Lino/PycharmProjects/Preprocessing/NTPtoLast/New/' + str(ntp) + 'TP'
                    directory = '/Users/Lino/PycharmProjects/Classification/NTPClass/'
                elif sys.platform == "win32":
                    path = 'C:\\Users\\Lino\\PycharmProjects\\Preprocessing\\NTP\\' + str(ntp) + 'TP'
                sys.path.append(path)
                #try:
                from weka.core.converters import Loader

                loader = Loader(classname="weka.core.converters.CSVLoader")
                data = loader.load_file(path + '/' + str(window) + 'd_' + str(ntp) + '.csv')
                dataLast = loader.load_file(lastpath + '/' + str(window) + 'd_' + str(ntp-1) + 'to' + str(ntp-1) +'.csv')
                data.class_is_last()
                dataLast.class_is_last()
                for fold in range(1,11):

                    from weka.filters import Filter
                    StratifiedCV = Filter(classname="weka.filters.supervised.instance.StratifiedRemoveFolds", options=['-S', '42', '-N' ,'10', '-F', str(fold)])
                    StratifiedCV.inputformat(data)
                    dataTest = StratifiedCV.filter(data)
                    StratifiedCV.inputformat(dataLast)
                    dataLastTest = StratifiedCV.filter(dataLast)

                    StratifiedCV = Filter(classname="weka.filters.supervised.instance.StratifiedRemoveFolds",
                                          options=['-S', '42', '-V' ,'-N', '10', '-F', str(fold)])
                    StratifiedCV.inputformat(data)
                    dataTrain = StratifiedCV.filter(data)
                    StratifiedCV.inputformat(dataLast)
                    dataLastTrain = StratifiedCV.filter(dataLast)

                    toBeRemoved = []
                    for attribute in range(0,dataTrain.attributes().data.class_index):
                        if dataTrain.attribute_stats(attribute).missing_count == dataTrain.attributes().data.num_instances and dataTest.attribute_stats(attribute).missing_count == dataTest.attributes().data.num_instances:
                            sys.exit("Fold has full missing column")
                        if (dataTrain.attribute_stats(attribute).missing_count/dataTrain.attributes().data.num_instances) > 0.5 and (dataTest.attribute_stats(attribute).missing_count/dataTest.attributes().data.num_instances) > 0.5:
                            toBeRemoved.append(str(attribute))

                    Remove = Filter(classname="weka.filters.unsupervised.attribute.Remove", options=['-R',','.join(toBeRemoved)])
                    Remove.inputformat(dataTrain)
                    dataTrain = Remove.filter(dataTrain)
                    dataTest = Remove.filter(dataTest)

                    toBeRemoved = []
                    for attribute in range(0, dataLastTrain.attributes().data.class_index):
                        if dataLastTrain.attribute_stats(
                                attribute).missing_count == dataLastTrain.attributes().data.num_instances and dataLastTest.attribute_stats(
                                attribute).missing_count == dataLastTest.attributes().data.num_instances:
                            sys.exit("Fold has full missing column")
                        if (dataLastTrain.attribute_stats(attribute).missing_count / dataLastTrain.attributes().data.num_instances) > 0.5 and (
                            dataLastTest.attribute_stats(attribute).missing_count / dataLastTest.attributes().data.num_instances) > 0.5:
                            toBeRemoved.append(str(attribute))

                    Remove = Filter(classname="weka.filters.unsupervised.attribute.Remove",
                                    options=['-R', ','.join(toBeRemoved)])
                    Remove.inputformat(dataLastTrain)
                    dataLastTrain = Remove.filter(dataLastTrain)
                    dataLastTest = Remove.filter(dataLastTest)

                    import weka.core.classes as wcc

                    FS = Filter(classname="weka.filters.supervised.attribute.AttributeSelection",
                                options=['-E', 'weka.attributeSelection.CfsSubsetEval -P 1 -E 1', '-S',
                                         'weka.attributeSelection.RerankingSearch -method 2 -blockSize 20 -rankingMeasure 0 -search "weka.attributeSelection.GreedyStepwise -T -1.7976931348623157E308 -N -1 -num-slots 1"'])
                    FS.inputformat(dataTrain)
                    dataTrain = FS.filter(dataTrain)
                    dataTest = FS.filter(dataTest)

                    FS.inputformat(dataLastTrain)
                    dataLastTrain = FS.filter(dataLastTrain)
                    dataLastTest = FS.filter(dataLastTest)

                    ReplaceMV = Filter(classname="weka.filters.unsupervised.attribute.ReplaceMissingValues")
                    ReplaceMV.inputformat(dataTrain)
                    dataTrain = ReplaceMV.filter(dataTrain)
                    ReplaceMV.inputformat(dataTest)
                    dataTest = ReplaceMV.filter(dataTest)

                    ReplaceMV.inputformat(dataLastTrain)
                    dataLastTrain = ReplaceMV.filter(dataLastTrain)
                    ReplaceMV.inputformat(dataLastTest)
                    dataLastTest = ReplaceMV.filter(dataLastTest)
                    # ReplaceMV.inputformat(dataTest)
                    # dataTest=ReplaceMV.filter(dataTest)

                    if smote != 0:
                        SMOTE = Filter(classname="weka.filters.supervised.instance.SMOTE", options=['-P', str(smote)])
                        SMOTE.inputformat(dataTrain)
                        dataTrain = SMOTE.filter(dataTrain)

                        SMOTE.inputformat(dataLastTrain)
                        dataLastTrain = SMOTE.filter(dataLastTrain)

                    dataTrain.class_is_last()
                    dataTest.class_is_last()

                    dataLastTrain.class_is_last()
                    dataLastTest.class_is_last()

                    from weka.classifiers import Evaluation
                    from weka.core.classes import Random
                    from weka.classifiers import Classifier
                    if classifier == 0:
                        for kernel in range(0,2):
                            if kernel == 0:
                                mapper = Classifier(classname="weka.classifiers.misc.InputMappedClassifier", options=["-M","-W", "weka.classifiers.bayes.NaiveBayes"])
                                Class = 'NaiveBayes'
                                mapper.build_classifier(dataTrain)
                                evaluation = Evaluation(dataTrain)
                                evaluation.test_model(mapper,dataTest)
                                Scores.write(str(evaluation.area_under_roc(1)*100) + ',')
                                recall_NB.append(evaluation.recall(1)*100)
                                precision_NB.append(evaluation.precision(1)*100)



                                mapper.build_classifier(dataLastTrain)
                                evaluation = Evaluation(dataLastTrain)
                                evaluation.test_model(mapper, dataLastTest)

                                ScoresLast.write(str(evaluation.area_under_roc(1) * 100)+',')

                            else:
                                mapper = Classifier(classname="weka.classifiers.misc.InputMappedClassifier",
                                                    options=["-M","-W", "weka.classifiers.bayes.NaiveBayes", "--", "-K"])
                                Class = 'NaiveBayes'
                                mapper.build_classifier(dataTrain)
                                evaluation = Evaluation(dataTrain)
                                evaluation.test_model(mapper, dataTest)
                                Scores.write(str(evaluation.area_under_roc(1) * 100) + '\n')



                                mapper.build_classifier(dataLastTrain)
                                evaluation = Evaluation(dataLastTrain)
                                evaluation.test_model(mapper, dataLastTest)
                                ScoresLast.write(str(evaluation.area_under_roc(1) * 100) + '\n')

                    elif classifier == 1:
                        for degree in range(1,4):
                            mapper = Classifier(classname="weka.classifiers.misc.InputMappedClassifier", options=["-M","-W", "weka.classifiers.functions.SMO", "--", "-K","weka.classifiers.functions.supportVector.PolyKernel -E " + str(degree)])
                            Class = 'SVM'
                            mapper.build_classifier(dataTrain)
                            evaluation = Evaluation(dataTrain)
                            evaluation.test_model(mapper, dataTest)
                            if degree == 1:
                                roc_SVM_1.append(evaluation.area_under_roc(1)*100)

                            elif degree == 2:
                                roc_SVM_2.append(evaluation.area_under_roc(1)*100)

                            else:
                                roc_SVM_3.append(evaluation.area_under_roc(1)*100)


                            mapper.build_classifier(dataLastTrain)
                            evaluation = Evaluation(dataLastTrain)
                            evaluation.test_model(mapper, dataLastTest)
                            if degree == 1:
                                roc_SVM_1_Last.append(evaluation.area_under_roc(1) * 100)

                            elif degree == 2:
                                roc_SVM_2_Last.append(evaluation.area_under_roc(1) * 100)

                            else:
                                roc_SVM_3_Last.append(evaluation.area_under_roc(1) * 100)

                    else:
                        for numTrees in np.arange(5,25,5):
                            mapper = Classifier(classname="weka.classifiers.misc.InputMappedClassifier", options=["-M","-W", "weka.classifiers.trees.RandomForest", "--", "-I", str(numTrees)])
                            Class = 'RF'
                            mapper.build_classifier(dataTrain)
                            evaluation = Evaluation(dataTrain)
                            evaluation.test_model(mapper, dataTest)
                            if numTrees == 5:
                                roc_RF_5.append(evaluation.area_under_roc(1)*100)
                            elif numTrees == 10:
                                roc_RF_10.append(evaluation.area_under_roc(1) * 100)

                            elif numTrees == 15:
                                roc_RF_15.append(evaluation.area_under_roc(1) * 100)

                            else:
                                roc_RF_20.append(evaluation.area_under_roc(1) * 100)

                            mapper.build_classifier(dataLastTrain)
                            evaluation = Evaluation(dataLastTrain)
                            evaluation.test_model(mapper, dataLastTest)
                            if numTrees == 5:
                                roc_RF_5_Last.append(evaluation.area_under_roc(1) * 100)

                            elif numTrees == 10:
                                roc_RF_10_Last.append(evaluation.area_under_roc(1) * 100)

                            elif numTrees == 15:
                                roc_RF_15_Last.append(evaluation.area_under_roc(1) * 100)

                            else:
                                roc_RF_20_Last.append(evaluation.area_under_roc(1) * 100)


                #except:
                #    continue


jvm.stop()
print(MV)