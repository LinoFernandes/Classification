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
Recall = open(directory + 'Progression_FS_Recall.txt', 'a')
Precision = open(directory + 'Progression_FS_Precision.txt', 'a')
Scores = open(directory + 'Progression_FS_AUC_Scores.txt', 'a')

Progression = ['Slow','Neutral','Fast']
Seeds = ['42','1313','653','575','789']

for window in Window:
    Scores.write('k,#TP,prog,NB,SVM_3,RF\n')
    Recall.write('k,#TP,prog,NB,SVM_3,RF\n')
    Precision.write('k,#TP,prog,NB,SVM_3,RF\n')
    for ntp in range(1,3):
        for prog in Progression:
            for smote in [50]:
                roc_NB = []
                recall_NB = []
                precision_NB = []
                recall_SVM_3 = []
                precision_SVM_3 = []
                roc_SVM_3 = []
                recall_RF_20 = []
                precision_RF_20 = []
                roc_RF_20 = []
                for seed in Seeds:
                    if sys.platform == "darwin":
                        path = '/Users/Lino/PycharmProjects/Preprocessing/NTP/' + str(ntp) + 'TP'
                        path2 = '/Users/Lino/PycharmProjects/Preprocessing/NTP/' + prog + '/' + str(ntp) + 'TP'
                        directory = '/Users/Lino/PycharmProjects/Classification/NTPClass/'
                    elif sys.platform == "win32":
                        path = 'C:\\Users\\Lino\\PycharmProjects\\Preprocessing\\NTP\\' + str(ntp) + 'TP'
                    sys.path.append(path)
                    from weka.core.converters import Loader

                    loader = Loader(classname="weka.core.converters.CSVLoader")
                    data = loader.load_file(path + '/' + str(window) + 'd_' + str(ntp) + '.csv')
                    data.class_is_last()
                    for classifier in range(0, 3):
                        roc_aux_NB=[]
                        precision_aux_NB=[]
                        recall_aux_NB=[]

                        roc_aux_SVM = []
                        precision_aux_SVM = []
                        recall_aux_SVM = []

                        roc_aux_RF = []
                        precision_aux_RF = []
                        recall_aux_RF = []
                        for fold in range(1,11):
                            from weka.filters import Filter

                            StratifiedCV = Filter(classname="weka.filters.supervised.instance.StratifiedRemoveFolds",
                                                  options=['-S', seed, '-N', '10', '-F', str(fold)])
                            StratifiedCV.inputformat(data)
                            dataTest = StratifiedCV.filter(data)
                            StratifiedCV = Filter(classname="weka.filters.supervised.instance.StratifiedRemoveFolds",
                                                  options=['-S', seed, '-V' ,'-N', '10', '-F', str(fold)])
                            StratifiedCV.inputformat(data)
                            dataTrain = StratifiedCV.filter(data)

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

                            if smote != 0:
                                SMOTE = Filter(classname="weka.filters.supervised.instance.SMOTE", options=['-P', str(smote)])
                                SMOTE.inputformat(dataTrain)
                                dataTrain = SMOTE.filter(dataTrain)

                            dataTrain.class_is_last()
                            dataTest.class_is_last()

                            from weka.classifiers import Evaluation
                            from weka.core.classes import Random
                            from weka.classifiers import Classifier
                            if classifier == 0:
                                for kernel in range(0,1):
                                    if kernel == 0:
                                        mapper = Classifier(classname="weka.classifiers.misc.InputMappedClassifier", options=['-M',"-W", "weka.classifiers.bayes.NaiveBayes"])
                                        Class = 'NaiveBayes'
                                        mapper.build_classifier(dataTrain)
                                        evaluation = Evaluation(dataTrain)
                                        evaluation.test_model(mapper,dataTest)
                                        roc_aux_NB.append(evaluation.area_under_roc(1) * 100)
                                        recall_aux_NB.append(evaluation.recall(1) * 100)
                                        precision_aux_NB.append(evaluation.precision(1) * 100)

                            elif classifier == 1:
                                for degree in range(3,4):
                                    mapper = Classifier(classname="weka.classifiers.misc.InputMappedClassifier", options=['-M',"-W", "weka.classifiers.functions.SMO", "--", "-K","weka.classifiers.functions.supportVector.PolyKernel -E " + str(degree)])
                                    Class = 'SVM'
                                    mapper.build_classifier(dataTrain)
                                    evaluation = Evaluation(dataTrain)
                                    evaluation.test_model(mapper, dataTest)
                                    roc_aux_SVM.append(evaluation.area_under_roc(1)*100)
                                    recall_aux_SVM.append(evaluation.recall(1)*100)
                                    precision_aux_SVM.append(evaluation.precision(1)*100)


                            else:
                                for numTrees in np.arange(20,25,5):
                                    mapper = Classifier(classname="weka.classifiers.misc.InputMappedClassifier", options=['-M',"-W", "weka.classifiers.trees.RandomForest", "--", "-I", str(numTrees)])
                                    Class = 'RF'
                                    mapper.build_classifier(dataTrain)
                                    evaluation = Evaluation(dataTrain)
                                    evaluation.test_model(mapper, dataTest)
                                    roc_aux_RF.append(evaluation.area_under_roc(1) * 100)
                                    precision_aux_RF.append(evaluation.precision(1) * 100)
                                    recall_aux_RF.append(evaluation.recall(1) * 100)
                    roc_NB = np.mean(roc_aux_NB)
                    roc_SVM_3 = np.mean(roc_aux_SVM)
                    roc_RF_20 = np.mean(roc_aux_RF)

                    recall_NB = np.mean(recall_aux_NB)
                    recall_SVM_3 = np.mean(recall_aux_SVM)
                    recall_RF_20 = np.mean(recall_aux_RF)

                    precision_NB = np.mean(precision_aux_NB)
                    precision_SVM_3 = np.mean(precision_SVM_3)
                    precision_RF_20 = np.mean(precision_RF_20)
            Scores.write(str(window) + ',' + str(ntp) + ',' + prog + ',' + str(round(np.mean(roc_NB),2)) + '%,' + str(round(np.mean(roc_SVM_3),2)) + '%,' + str(round(np.mean(roc_RF_20),2))+ '%\n')
            Recall.write(str(window) + ',' + str(ntp) + ',' + prog + ',' + str(round(np.mean(recall_NB),2)) + '%,' + str(round(np.mean(recall_SVM_3),2)) + '%,' + str(round(np.mean(recall_RF_20),2)) + '%\n')
            Precision.write(str(window) + ',' + str(ntp) + ',' + prog + ',' + str(round(np.mean(precision_NB),2)) + '%,' + str(round(np.mean(precision_SVM_3),2)) + '%,' + str(round(np.mean(precision_RF_20),2)) + '%\n')

jvm.stop()
print(MV)