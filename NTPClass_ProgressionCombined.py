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
directory = '/Users/Lino/PycharmProjects/Classification/NTPClass/New/Progression/'
if not os.path.exists(directory):
    os.makedirs(directory)


Progression = ['Slow','Neutral','Fast']
Seeds = ['42','1313','653','575','789']

#Window = [180]

# Recall = open(directory + 'Recall_'+str(Window[0])+'.txt', 'a')
# Precision = open(directory + 'Precision_'+str(Window[0])+'.txt', 'a')
# Scores = open(directory + 'AUC_'+str(Window[0])+'.txt', 'a')

Recall = open(directory + 'Recall_Comb.txt', 'a')
Precision = open(directory + 'Precision_Comb.txt', 'a')
Scores = open(directory + 'AUC_Comb.txt', 'a')


for window in Window:
    Scores.write('k,#TP,prog,NB,SVM_3,RF\n')
    Recall.write('k,#TP,prog,NB,SVM_3,RF\n')
    Precision.write('k,#TP,prog,NB,SVM_3,RF\n')
    for ntp in range(1,3):
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
            for classifier in range(0, 3):
                if sys.platform == "darwin":
                    pathSlow = '/Users/Lino/PycharmProjects/Preprocessing/NTP/New/Slow/' + str(ntp) + 'TP'
                    pathNeutral = '/Users/Lino/PycharmProjects/Preprocessing/NTP/New/Neutral/' + str(ntp) + 'TP'
                    pathFast = '/Users/Lino/PycharmProjects/Preprocessing/NTP/New/Fast/' + str(ntp) + 'TP'
                    pathTest = '/Users/Lino/PycharmProjects/Preprocessing/NTP/New/' + str(ntp) + 'TP'
                    #path2 = '/Users/Lino/PycharmProjects/Preprocessing/NTP/' + prog + '/' + str(ntp) + 'TP'
                    directory = '/Users/Lino/PycharmProjects/Classification/NTPClass/'
                elif sys.platform == "win32":
                    path = 'C:\\Users\\Lino\\PycharmProjects\\Preprocessing\\NTP\\' + str(ntp) + 'TP'
                #sys.path.append(path)
                from weka.core.converters import Loader

                loader = Loader(classname="weka.core.converters.CSVLoader")
                dataTrainSlow = loader.load_file(pathSlow + '/' + str(window) + 'd_' + str(ntp) + '.csv')
                dataTrainNeutral = loader.load_file(pathNeutral + '/' + str(window) + 'd_' + str(ntp) + '.csv')
                dataTrainFast = loader.load_file(pathFast + '/' + str(window) + 'd_' + str(ntp) + '.csv')
                dataTest = loader.load_file(pathTest + '/' + str(window) + 'd_' + str(ntp) + '_Test.csv')
                dataTrainSlow.class_is_last()
                dataTrainNeutral.class_is_last()
                dataTrainFast.class_is_last()
                dataTest.class_is_last()


                roc_aux_NB=[]
                precision_aux_NB=[]
                recall_aux_NB=[]

                roc_aux_SVM = []
                precision_aux_SVM = []
                recall_aux_SVM = []

                roc_aux_RF = []
                precision_aux_RF = []
                recall_aux_RF = []
                for fold in [1]:
                    from weka.filters import Filter
                    #
                    # StratifiedCV = Filter(classname="weka.filters.supervised.instance.StratifiedRemoveFolds",
                    #                       options=['-S', '42', '-N', '10', '-F', str(fold)])
                    # StratifiedCV.inputformat(dataSlow)
                    # dataTest = StratifiedCV.filter(dataSlow)
                    #
                    # StratifiedCV.inputformat(dataNeutral)
                    # dataTest = StratifiedCV.filter(dataNeutral)
                    #
                    # StratifiedCV.inputformat(dataFast)
                    # dataTest = StratifiedCV.filter(dataFast)

                    # StratifiedCV = Filter(classname="weka.filters.supervised.instance.StratifiedRemoveFolds",
                    #                       options=['-S', '42', '-V' ,'-N', '10', '-F', str(fold)])
                    # StratifiedCV.inputformat(dataSlow)
                    # dataTrainSlow = StratifiedCV.filter(dataSlow)
                    #
                    # StratifiedCV.inputformat(dataNeutral)
                    # dataTrainNeutral = StratifiedCV.filter(dataNeutral)
                    #
                    # StratifiedCV.inputformat(dataFast)
                    # dataTrainFast = StratifiedCV.filter(dataFast)

                    toBeRemoved = []
                    for attribute in range(0,dataTrainSlow.attributes().data.class_index):
                        if dataTrainSlow.attribute_stats(attribute).missing_count == dataTrainSlow.attributes().data.num_instances:
                            sys.exit("Fold has full missing column")
                        if (dataTrainSlow.attribute_stats(attribute).missing_count/dataTrainSlow.attributes().data.num_instances) > 0.5:
                            toBeRemoved.append(str(attribute))

                    Remove = Filter(classname="weka.filters.unsupervised.attribute.Remove", options=['-R',','.join(toBeRemoved)])
                    Remove.inputformat(dataTrainSlow)
                    dataTrainSlow = Remove.filter(dataTrainSlow)
                    Remove.inputformat(dataTest)
                    dataTestSlow = Remove.filter(dataTest)

                    toBeRemoved = []
                    for attribute in range(0, dataTrainNeutral.attributes().data.class_index):
                        if dataTrainNeutral.attribute_stats(
                                attribute).missing_count == dataTrainNeutral.attributes().data.num_instances:
                            sys.exit("Fold has full missing column")
                        if (dataTrainNeutral.attribute_stats(
                                attribute).missing_count / dataTrainNeutral.attributes().data.num_instances) > 0.5:
                            toBeRemoved.append(str(attribute))

                    Remove = Filter(classname="weka.filters.unsupervised.attribute.Remove",
                                    options=['-R', ','.join(toBeRemoved)])
                    Remove.inputformat(dataTrainNeutral)
                    dataTrainNeutral = Remove.filter(dataTrainNeutral)
                    Remove.inputformat(dataTest)
                    dataTestNeutral = Remove.filter(dataTest)



                    toBeRemoved = []
                    for attribute in range(0, dataTrainFast.attributes().data.class_index):
                        if dataTrainFast.attribute_stats(
                                attribute).missing_count == dataTrainFast.attributes().data.num_instances:
                            sys.exit("Fold has full missing column")
                        if (dataTrainFast.attribute_stats(
                                attribute).missing_count / dataTrainFast.attributes().data.num_instances) > 0.5:
                            toBeRemoved.append(str(attribute))

                    Remove = Filter(classname="weka.filters.unsupervised.attribute.Remove", options=['-R', ','.join(toBeRemoved)])
                    Remove.inputformat(dataTrainFast)
                    dataTrainFast = Remove.filter(dataTrainFast)
                    Remove.inputformat(dataTest)
                    dataTestFast = Remove.filter(dataTest)


                    import weka.core.classes as wcc

                    FS = Filter(classname="weka.filters.supervised.attribute.AttributeSelection",
                                options=['-E', 'weka.attributeSelection.CfsSubsetEval -P 1 -E 1', '-S',
                                         'weka.attributeSelection.RerankingSearch -method 2 -blockSize 20 -rankingMeasure 0 -search "weka.attributeSelection.GreedyStepwise -T -1.7976931348623157E308 -N 20 -num-slots 1"'])
                    FS.inputformat(dataTrainSlow)
                    dataTrainSlow = FS.filter(dataTrainSlow)
                    FS.inputformat(dataTestSlow)
                    dataTestSlow = FS.filter(dataTestSlow)

                    FS.inputformat(dataTrainNeutral)
                    dataTrainNeutral = FS.filter(dataTrainNeutral)
                    FS.inputformat(dataTestNeutral)
                    dataTestNeutral = FS.filter(dataTestNeutral)

                    FS.inputformat(dataTrainFast)
                    dataTrainFast = FS.filter(dataTrainFast)
                    FS.inputformat(dataTestFast)
                    dataTestFast = FS.filter(dataTestFast)


                    ReplaceMV = Filter(classname="weka.filters.unsupervised.attribute.ReplaceMissingValues")
                    ReplaceMV.inputformat(dataTrainSlow)
                    dataTrainSlow = ReplaceMV.filter(dataTrainSlow)


                    ReplaceMV.inputformat(dataTrainNeutral)
                    dataTrainNeutral = ReplaceMV.filter(dataTrainNeutral)
                    ReplaceMV.inputformat(dataTest)

                    ReplaceMV.inputformat(dataTrainFast)
                    dataTrainFast = ReplaceMV.filter(dataTrainFast)

                    ReplaceMV.inputformat(dataTestSlow)
                    dataTestSlow = ReplaceMV.filter(dataTestSlow)

                    ReplaceMV.inputformat(dataTestNeutral)
                    dataTestNeutral = ReplaceMV.filter(dataTestNeutral)

                    ReplaceMV.inputformat(dataTestFast)
                    dataTestFast = ReplaceMV.filter(dataTestFast)


                    from weka.classifiers import Evaluation
                    from weka.core.classes import Random
                    from weka.classifiers import Classifier, PredictionOutput
                    if classifier == 0:
                        for kernel in range(0,1):
                            if kernel == 0:
                                mapper = Classifier(classname="weka.classifiers.misc.InputMappedClassifier", options=['-M',"-W", "weka.classifiers.bayes.NaiveBayes","--", "-K"])
                                Class = 'NaiveBayes'
                                mapper.build_classifier(dataTrainSlow)
                                evaluation = Evaluation(dataTrainSlow)
                                evaluation.test_model(mapper,dataTestSlow)

                                aux = evaluation.predictions
                                NB_Slow = [i.distribution[0] for i in aux]

                                mapper.build_classifier(dataTrainNeutral)
                                evaluation = Evaluation(dataTrainNeutral)
                                evaluation.test_model(mapper, dataTestNeutral)
                                aux = evaluation.predictions
                                NB_Neutral = [i.distribution[0] for i in aux]


                                mapper.build_classifier(dataTrainFast)
                                evaluation = Evaluation(dataTrainFast)
                                evaluation.test_model(mapper, dataTestFast)
                                aux = evaluation.predictions
                                NB_Fast = [i.distribution[0] for i in aux]



                    elif classifier == 1:
                        SMOTE = Filter(classname="weka.filters.supervised.instance.SMOTE",
                                       options=['-P', str(smote)])
                        SMOTE.inputformat(dataTrainSlow)
                        dataTrainSlow = SMOTE.filter(dataTrainSlow)

                        SMOTE.inputformat(dataTrainNeutral)
                        dataTrainNeutral = SMOTE.filter(dataTrainNeutral)

                        SMOTE.inputformat(dataTrainFast)
                        dataTrainFast = SMOTE.filter(dataTrainFast)
                        for degree in range(3,4):
                            mapper = Classifier(classname="weka.classifiers.misc.InputMappedClassifier", options=['-M',"-W", "weka.classifiers.functions.SMO", "--", "-K","weka.classifiers.functions.supportVector.PolyKernel -E " + str(degree)])
                            Class = 'SVM'
                            mapper.build_classifier(dataTrainSlow)
                            evaluation = Evaluation(dataTrainSlow)
                            evaluation.test_model(mapper, dataTestSlow)

                            aux = evaluation.predictions
                            SVM_Slow = [i.distribution[0] for i in aux]

                            mapper.build_classifier(dataTrainNeutral)
                            evaluation = Evaluation(dataTrainNeutral)
                            evaluation.test_model(mapper, dataTestNeutral)
                            aux = evaluation.predictions
                            SVM_Neutral = [i.distribution[0] for i in aux]

                            mapper.build_classifier(dataTrainFast)
                            evaluation = Evaluation(dataTrainFast)
                            evaluation.test_model(mapper, dataTestFast)
                            aux = evaluation.predictions
                            SVM_Fast = [i.distribution[0] for i in aux]

                    else:
                        for numTrees in np.arange(20,25,5):
                            mapper = Classifier(classname="weka.classifiers.misc.InputMappedClassifier", options=['-M',"-W", "weka.classifiers.trees.RandomForest", "--", "-I", str(numTrees)])
                            Class = 'RF'
                            mapper.build_classifier(dataTrainSlow)
                            evaluation = Evaluation(dataTrainSlow)
                            evaluation.test_model(mapper, dataTestSlow)

                            aux = evaluation.predictions
                            RF_Slow = [i.distribution[0] for i in aux]

                            mapper.build_classifier(dataTrainNeutral)
                            evaluation = Evaluation(dataTrainNeutral)
                            evaluation.test_model(mapper, dataTestNeutral)
                            aux = evaluation.predictions
                            RF_Neutral = [i.distribution[0] for i in aux]

                            mapper.build_classifier(dataTrainFast)
                            evaluation = Evaluation(dataTrainFast)
                            evaluation.test_model(mapper, dataTestFast)
                            aux = evaluation.area_under_prc()
                            RF_Fast = [i.distribution[0] for i in aux]
        NB_Pred = []
        SVM_Pred = []
        RF_Pred = []

        aux = [NB_Slow, NB_Neutral, NB_Fast]
        aux = np.mean(aux, axis=0)
        for i in aux:
            if i > 0.5:
                NB_Pred.append(1)
            elif i < 0.5:
                NB_Pred.append(0)
            else:
                NB_Pred.append(np.random.randint(0,2))

        TrueVal = dataTest.values(dataTest.class_index)
        from sklearn.metrics import auc, roc_curve, recall_score, precision_score

        fpr, tpr, thresholds = roc_curve(TrueVal, NB_Pred)
        roc_NB = auc(fpr, tpr)*100
        recall_NB = recall_score(TrueVal, NB_Pred)*100
        precision_NB = precision_score(TrueVal, NB_Pred)*100

        aux = [SVM_Slow, SVM_Neutral, SVM_Fast]
        aux = np.mean(aux, axis=0)
        for i in aux:
            if i > 0.5:
                SVM_Pred.append(1)
            elif i < 0.5:
                SVM_Pred.append(0)
            else:
                SVM_Pred.append(np.random.randint(0, 2))


        fpr, tpr, thresholds = roc_curve(TrueVal, SVM_Pred)
        roc_SVM = auc(fpr, tpr)*100
        recall_SVM = recall_score(TrueVal, SVM_Pred)*100
        precision_SVM = precision_score(TrueVal, SVM_Pred)*100

        aux = [RF_Slow, RF_Neutral, RF_Fast]
        aux = np.mean(aux, axis=0)
        for i in aux:
            if i > 0.5:
                RF_Pred.append(1)
            elif i < 0.5:
                RF_Pred.append(0)
            else:
                RF_Pred.append(np.random.randint(0, 2))


        fpr, tpr, thresholds = roc_curve(TrueVal, RF_Pred)
        roc_RF=auc(fpr, tpr)*100
        recall_RF = recall_score(TrueVal,RF_Pred)*100
        precision_RF=precision_score(TrueVal,RF_Pred)*100
        Scores.write(str(window) + ',' + str(ntp) + ','+ str(round(roc_NB,2)) + ',' + str(round(roc_SVM,2)) + ',' + str(round(roc_RF,2))+ '\n')
        Recall.write(str(window) + ',' + str(ntp) + ','+ str(round(recall_NB,2)) + ',' + str(round(recall_SVM,2)) + ',' + str(round(recall_RF,2)) + '\n')
        Precision.write(str(window) + ',' + str(ntp) + ','+ str(round(precision_NB,2)) + ',' + str(round(precision_SVM,2)) + ',' + str(round(precision_RF,2)) + '\n')

jvm.stop()


