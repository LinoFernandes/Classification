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

directory = '/Users/Lino/PycharmProjects/Classification/NTPClass/New2/Prog/'
if not os.path.exists(directory):
    os.makedirs(directory)
Perf = open(directory + 'AUC_Final.txt', 'a')
Recall = open(directory + 'Recall_Final.txt', 'a')
Precision = open(directory + 'Precision_Final.txt', 'a')
Scores = open(directory + 'NTPClass_BoundsSlow_Final.csv', 'a')
ScoresLast = open(directory + 'NTPClass_BoundsFast_Final.csv', 'a')
ScoresNeutral = open(directory + 'NTPClass_BoundsNeutral_Final.csv', 'a')


Seeds = [42 , 467, 14, 9, 125]
smote=50

for window in Window:
    for ntp in range(2, 4):
        roc_NB = []
        roc_SVM_2 = []
        roc_RF_20 = []

        roc_NB_Last = []
        roc_SVM_2_Last = []
        roc_RF_20_Last = []

        recall_NB = []
        recall_SVM_2 = []
        recall_RF_20 = []

        recall_NB_Last = []
        recall_SVM_2_Last = []
        recall_RF_20_Last = []

        precision_NB = []
        precision_SVM_2 = []
        precision_RF_20 = []

        precision_NB_Last = []
        precision_SVM_2_Last = []
        precision_RF_20_Last = []

        recall_NB_Neutral = []
        recall_SVM_2_Neutral = []
        recall_RF_20_Neutral = []

        roc_NB_Neutral = []
        roc_SVM_2_Neutral = []
        roc_RF_20_Neutral = []

        precision_NB_Neutral = []
        precision_SVM_2_Neutral = []
        precision_RF_20_Neutral = []
        for classifier in range(0, 3):
            if sys.platform == "darwin":
                path = '/Users/Lino/PycharmProjects/Preprocessing/NTP/New2/Slow/' + str(ntp) + 'TP'
                fastpath = '/Users/Lino/PycharmProjects/Preprocessing/NTP/New2/Fast/' + str(ntp) + 'TP'
                neutralpath = '/Users/Lino/PycharmProjects/Preprocessing/NTP/New2/Neutral/' + str(ntp) + 'TP'
                directory = '/Users/Lino/PycharmProjects/Classification/NTPClass/'
            elif sys.platform == "win32":
                path = 'C:\\Users\\Lino\\PycharmProjects\\Preprocessing\\NTP\\' + str(ntp) + 'TP'
            sys.path.append(path)
            #try:
            from weka.core.converters import Loader

            loader = Loader(classname="weka.core.converters.CSVLoader")
            dataSlow = loader.load_file(path + '/' + str(window) + 'd_' + str(ntp) + '.csv')
            dataNeutral = loader.load_file(neutralpath + '/' + str(window) + 'd_' + str(ntp) + '.csv')
            dataFast = loader.load_file(fastpath + '/' + str(window) + 'd_' + str(ntp) + '.csv')
            
            dataSlow.class_is_last()
            dataNeutral.class_is_last()
            dataFast.class_is_last()

            for seed in Seeds:
                for fold in range(1,6):

                    from weka.filters import Filter
                    StratifiedCV = Filter(classname="weka.filters.supervised.instance.StratifiedRemoveFolds", options=['-S', str(seed), '-N' ,'5', '-F', str(fold)])
                    StratifiedCV.inputformat(dataSlow)
                    dataSlowTest = StratifiedCV.filter(dataSlow)
                    StratifiedCV.inputformat(dataFast)
                    dataFastTest = StratifiedCV.filter(dataFast)
                    StratifiedCV.inputformat(dataNeutral)
                    dataNeutralTest = StratifiedCV.filter(dataNeutral)

                    StratifiedCV = Filter(classname="weka.filters.supervised.instance.StratifiedRemoveFolds",
                                          options=['-S', str(seed), '-V' ,'-N', '5', '-F', str(fold)])
                    StratifiedCV.inputformat(dataSlow)
                    dataSlowTrain = StratifiedCV.filter(dataSlow)
                    StratifiedCV.inputformat(dataFast)
                    dataFastTrain = StratifiedCV.filter(dataFast)
                    StratifiedCV.inputformat(dataNeutral)
                    dataNeutralTrain = StratifiedCV.filter(dataNeutral)

                    toBeRemoved = []
                    for attribute in range(0,dataSlowTrain.attributes().data.class_index):
                        if dataSlowTrain.attribute_stats(attribute).missing_count == dataSlowTrain.attributes().data.num_instances and dataSlowTest.attribute_stats(attribute).missing_count == dataSlowTest.attributes().data.num_instances:
                            sys.exit("Fold has full missing column")
                        if (dataSlowTrain.attribute_stats(attribute).missing_count/dataSlowTrain.attributes().data.num_instances) > 0.5 and (dataSlowTest.attribute_stats(attribute).missing_count/dataSlowTest.attributes().data.num_instances) > 0.5:
                            toBeRemoved.append(str(attribute))

                    Remove = Filter(classname="weka.filters.unsupervised.attribute.Remove", options=['-R',','.join(toBeRemoved)])
                    Remove.inputformat(dataSlowTrain)
                    dataSlowTrain = Remove.filter(dataSlowTrain)
                    dataSlowTest = Remove.filter(dataSlowTest)

                    toBeRemoved = []
                    for attribute in range(0, dataFastTrain.attributes().data.class_index):
                        if dataFastTrain.attribute_stats(
                                attribute).missing_count == dataFastTrain.attributes().data.num_instances and dataFastTest.attribute_stats(
                                attribute).missing_count == dataFastTest.attributes().data.num_instances:
                            sys.exit("Fold has full missing column")
                        if (dataFastTrain.attribute_stats(attribute).missing_count / dataFastTrain.attributes().data.num_instances) > 0.5 and (
                            dataFastTest.attribute_stats(attribute).missing_count / dataFastTest.attributes().data.num_instances) > 0.5:
                            toBeRemoved.append(str(attribute))

                    Remove = Filter(classname="weka.filters.unsupervised.attribute.Remove",
                                    options=['-R', ','.join(toBeRemoved)])
                    Remove.inputformat(dataFastTrain)
                    dataFastTrain = Remove.filter(dataFastTrain)
                    dataFastTest = Remove.filter(dataFastTest)

                    toBeRemoved = []
                    for attribute in range(0, dataNeutralTrain.attributes().data.class_index):
                        if dataNeutralTrain.attribute_stats(
                                attribute).missing_count == dataNeutralTrain.attributes().data.num_instances and dataNeutralTest.attribute_stats(
                            attribute).missing_count == dataNeutralTest.attributes().data.num_instances:
                            sys.exit("Fold has full missing column")
                        if (dataNeutralTrain.attribute_stats(
                                attribute).missing_count / dataNeutralTrain.attributes().data.num_instances) > 0.5 and (
                                    dataNeutralTest.attribute_stats(
                                        attribute).missing_count / dataNeutralTest.attributes().data.num_instances) > 0.5:
                            toBeRemoved.append(str(attribute))

                    Remove = Filter(classname="weka.filters.unsupervised.attribute.Remove",
                                    options=['-R', ','.join(toBeRemoved)])
                    Remove.inputformat(dataNeutralTrain)
                    dataNeutralTrain = Remove.filter(dataNeutralTrain)
                    dataNeutralTest = Remove.filter(dataNeutralTest)

                    import weka.core.classes as wcc


                    #
                    # ReplaceMV = Filter(classname="weka.filters.unsupervised.attribute.ReplaceMissingValues")
                    # ReplaceMV.inputformat(dataSlowTrain)
                    # dataSlowTrain = ReplaceMV.filter(dataSlowTrain)
                    # ReplaceMV.inputformat(dataSlowTest)
                    # dataSlowTest = ReplaceMV.filter(dataSlowTest)
                    #
                    # ReplaceMV.inputformat(dataFastTrain)
                    # dataFastTrain = ReplaceMV.filter(dataFastTrain)
                    # ReplaceMV.inputformat(dataFastTest)
                    # dataFastTest = ReplaceMV.filter(dataFastTest)
                    #
                    # ReplaceMV.inputformat(dataNeutralTrain)
                    # dataNeutralTrain = ReplaceMV.filter(dataNeutralTrain)
                    # ReplaceMV.inputformat(dataNeutralTest)
                    # dataNeutralTest = ReplaceMV.filter(dataNeutralTest)

                    #
                    # FS = Filter(classname="weka.filters.supervised.attribute.AttributeSelection",
                    #             options=['-E', 'weka.attributeSelection.CfsSubsetEval -P 1 -E 1', '-S',
                    #                      'weka.attributeSelection.RerankingSearch -method 2 -blockSize 20 -rankingMeasure 0 -search "weka.attributeSelection.GreedyStepwise -T -1.7976931348623157E308 -N -1 -num-slots 1"'])

                    FS = Filter(classname="weka.filters.supervised.attribute.AttributeSelection",
                                options=['-E', 'weka.attributeSelection.CfsSubsetEval -P 1 -E 1', '-S',
                                         "weka.attributeSelection.GreedyStepwise -T -1.7976931348623157E308 -N -1 -num-slots 1"])

                    FS.inputformat(dataSlowTrain)
                    dataSlowTrain = FS.filter(dataSlowTrain)
                    dataSlowTest = FS.filter(dataSlowTest)

                    ClassIndex = dataSlowTrain.attribute(dataSlowTrain.class_index).values
                    yIndexSlow = ClassIndex.index('Y')

                    FS.inputformat(dataFastTrain)
                    dataFastTrain = FS.filter(dataFastTrain)
                    dataFastTest = FS.filter(dataFastTest)

                    ClassIndex = dataFastTrain.attribute(dataFastTrain.class_index).values
                    yIndexFast = ClassIndex.index('Y')

                    FS.inputformat(dataNeutralTrain)
                    dataNeutralTrain = FS.filter(dataNeutralTrain)
                    dataNeutralTest = FS.filter(dataNeutralTest)

                    ClassIndex = dataNeutralTrain.attribute(dataNeutralTrain.class_index).values
                    yIndexNeutral = ClassIndex.index('Y')


                    dataSlowTrain.class_is_last()
                    dataSlowTest.class_is_last()

                    dataFastTrain.class_is_last()
                    dataFastTest.class_is_last()

                    dataNeutralTrain.class_is_last()
                    dataNeutralTest.class_is_last()

                    from weka.classifiers import Evaluation
                    from weka.core.classes import Random
                    from weka.classifiers import Classifier
                    if classifier == 0:
                        SMOTE = Filter(classname="weka.filters.supervised.instance.SMOTE", options=['-P', str(smote)])
                        SMOTE.inputformat(dataSlowTrain)
                        dataSlowTrain = SMOTE.filter(dataSlowTrain)

                        SMOTE.inputformat(dataFastTrain)
                        dataFastTrain = SMOTE.filter(dataFastTrain)
                        for kernel in range(0,1):
                            if kernel == 0:
                                mapper = Classifier(classname="weka.classifiers.misc.InputMappedClassifier", options=["-M","-W", "weka.classifiers.bayes.NaiveBayes"])
                                Class = 'NaiveBayes'
                                mapper.build_classifier(dataSlowTrain)
                                evaluation = Evaluation(dataSlowTrain)
                                evaluation.test_model(mapper,dataSlowTest)
                                roc_NB.append(evaluation.area_under_roc(1)*100)
                                recall_NB.append(evaluation.recall(yIndexSlow)*100)
                                precision_NB.append(evaluation.precision(yIndexSlow)*100)

                                mapper.build_classifier(dataFastTrain)
                                evaluation = Evaluation(dataFastTrain)
                                evaluation.test_model(mapper, dataFastTest)

                                roc_NB_Last.append(evaluation.area_under_roc(1) * 100)
                                recall_NB_Last.append(evaluation.recall(yIndexFast) * 100)
                                precision_NB_Last.append(evaluation.precision(yIndexFast) * 100)

                                mapper.build_classifier(dataNeutralTrain)
                                evaluation = Evaluation(dataNeutralTrain)
                                evaluation.test_model(mapper, dataNeutralTest)

                                roc_NB_Neutral.append(evaluation.area_under_roc(1) * 100)
                                recall_NB_Neutral.append(evaluation.recall(yIndexNeutral) * 100)
                                precision_NB_Neutral.append(evaluation.precision(yIndexNeutral) * 100)
                    elif classifier == 1:
                        for degree in [2]:
                            mapper = Classifier(classname="weka.classifiers.misc.InputMappedClassifier", options=["-M","-W", "weka.classifiers.functions.SMO", "--", "-K","weka.classifiers.functions.supportVector.PolyKernel -E " + str(degree)])
                            Class = 'SVM'

                            if ((window == 90) and (ntp == 3 or ntp == 4)):

                                SMOTE = Filter(classname="weka.filters.supervised.instance.SMOTE", options=['-P', '50'])
                                SMOTE.inputformat(dataSlowTrain)
                                dataSlowTrain = SMOTE.filter(dataSlowTrain)

                                SMOTE.inputformat(dataFastTrain)
                                dataFastTrain = SMOTE.filter(dataFastTrain)

                                mapper.build_classifier(dataSlowTrain)
                                evaluation = Evaluation(dataSlowTrain)
                                evaluation.test_model(mapper, dataSlowTest)

                                roc_SVM_2.append(evaluation.area_under_roc(1)*100)
                                recall_SVM_2.append(evaluation.recall(yIndexSlow) * 100)
                                precision_SVM_2.append(evaluation.precision(yIndexSlow) * 100)

                                mapper.build_classifier(dataFastTrain)
                                evaluation = Evaluation(dataFastTrain)
                                evaluation.test_model(mapper, dataFastTest)

                                roc_SVM_2_Last.append(evaluation.area_under_roc(1) * 100)
                                recall_SVM_2_Last.append(evaluation.recall(yIndexFast) * 100)
                                precision_SVM_2_Last.append(evaluation.precision(yIndexFast) * 100)

                                mapper.build_classifier(dataNeutralTrain)
                                evaluation = Evaluation(dataNeutralTrain)
                                evaluation.test_model(mapper, dataNeutralTest)

                                roc_SVM_2_Neutral.append(evaluation.area_under_roc(1) * 100)
                                recall_SVM_2_Neutral.append(evaluation.recall(yIndexNeutral) * 100)
                                precision_SVM_2_Neutral.append(evaluation.precision(yIndexNeutral) * 100)

                            else:
                                SMOTE = Filter(classname="weka.filters.supervised.instance.SMOTE",
                                               options=['-P', str(smote)])
                                SMOTE.inputformat(dataSlowTrain)
                                dataSlowTrain = SMOTE.filter(dataSlowTrain)

                                SMOTE.inputformat(dataFastTrain)
                                dataFastTrain = SMOTE.filter(dataFastTrain)
                                mapper.build_classifier(dataSlowTrain)
                                evaluation = Evaluation(dataSlowTrain)
                                evaluation.test_model(mapper, dataSlowTest)

                                roc_SVM_2.append(evaluation.area_under_roc(1) * 100)
                                recall_SVM_2.append(evaluation.recall(1) * 100)
                                precision_SVM_2.append(evaluation.precision(1) * 100)

                                mapper.build_classifier(dataFastTrain)
                                evaluation = Evaluation(dataFastTrain)
                                evaluation.test_model(mapper, dataFastTest)

                                roc_SVM_2_Last.append(evaluation.area_under_roc(1) * 100)
                                recall_SVM_2_Last.append(evaluation.recall(1) * 100)
                                precision_SVM_2_Last.append(evaluation.precision(1) * 100)

                                mapper.build_classifier(dataNeutralTrain)
                                evaluation = Evaluation(dataNeutralTrain)
                                evaluation.test_model(mapper, dataNeutralTest)

                                roc_SVM_2_Neutral.append(evaluation.area_under_roc(1) * 100)
                                recall_SVM_2_Neutral.append(evaluation.recall(1) * 100)
                                precision_SVM_2_Neutral.append(evaluation.precision(1) * 100)

                    else:
                        for numTrees in [20]:
                            SMOTE = Filter(classname="weka.filters.supervised.instance.SMOTE",
                                           options=['-P', str(smote)])
                            SMOTE.inputformat(dataSlowTrain)
                            dataSlowTrain = SMOTE.filter(dataSlowTrain)

                            SMOTE.inputformat(dataFastTrain)
                            dataFastTrain = SMOTE.filter(dataFastTrain)

                            mapper = Classifier(classname="weka.classifiers.misc.InputMappedClassifier", options=["-M","-W", "weka.classifiers.trees.RandomForest", "--", "-I", str(numTrees)])
                            Class = 'RF'
                            mapper.build_classifier(dataSlowTrain)
                            evaluation = Evaluation(dataSlowTrain)
                            evaluation.test_model(mapper, dataSlowTest)

                            roc_RF_20.append(evaluation.area_under_roc(1) * 100)
                            recall_RF_20.append(evaluation.recall(yIndexSlow) * 100)
                            precision_RF_20.append(evaluation.precision(yIndexSlow) * 100)

                            mapper.build_classifier(dataFastTrain)
                            evaluation = Evaluation(dataFastTrain)
                            evaluation.test_model(mapper, dataFastTest)

                            roc_RF_20_Last.append(evaluation.area_under_roc(1) * 100)
                            recall_RF_20_Last.append(evaluation.recall(yIndexFast) * 100)
                            precision_RF_20_Last.append(evaluation.precision(yIndexFast) * 100)

                            mapper.build_classifier(dataNeutralTrain)
                            evaluation = Evaluation(dataNeutralTrain)
                            evaluation.test_model(mapper, dataNeutralTest)

                            print(evaluation.class_details())
                            print(evaluation.classname)
                            print(evaluation.recall(0))
                            print(evaluation.recall(1))
                            print(yIndexNeutral)

                            roc_RF_20_Neutral.append(evaluation.area_under_roc(1) * 100)
                            recall_RF_20_Neutral.append(evaluation.recall(yIndexNeutral) * 100)
                            precision_RF_20_Neutral.append(evaluation.precision(yIndexNeutral) * 100)
                            print(recall_RF_20_Neutral)
        Scores.write(str(ntp) + ',' + str(round(np.mean(roc_NB)+np.std(roc_NB),2)) + ',' + str(round(np.mean(roc_NB)-np.std(roc_NB),2))+ ',' + str(round(np.mean(roc_RF_20)+np.std(roc_RF_20),2)) + ',' + str(round(np.mean(roc_RF_20)-np.std(roc_RF_20),2)) + '\n')

        ScoresLast.write(str(ntp) + ',' + str(round(np.mean(roc_NB_Last)+np.std(roc_NB_Last),2)) + ',' + str(round(np.mean(roc_NB_Last)-np.std(roc_NB_Last),2)) + ',' + str(round(np.mean(roc_RF_20_Last)+np.std(roc_RF_20_Last),2)) +','+ str(round(np.mean(roc_RF_20_Last)-np.std(roc_RF_20_Last),2)) + '\n')

        ScoresNeutral.write(str(ntp) + ',' + str(round(np.mean(roc_NB_Neutral)+np.std(roc_NB_Neutral),2)) + ',' + str(round(np.mean(roc_NB_Neutral)-np.std(roc_NB_Neutral),2)) + ',' + str(round(np.mean(roc_RF_20_Neutral)+np.std(roc_RF_20_Neutral),2)) +','+ str(round(np.mean(roc_RF_20_Neutral)-np.std(roc_RF_20_Neutral),2)) + '\n')
        if ntp == 2:

            Perf.write('\multirow{6}{*}{'+str(window)+'d}' + ' & ' + '\multirow{3}{*}{'+str(ntp)+'d}' + ' & '+ 'Slow' + ' & ' + str(round(np.mean(roc_NB),2)) + ' $\pm$ ' + str(str(round(np.std(roc_NB),2))) + ' & ' + str(round(np.mean(roc_RF_20),2)) + ' $\pm$ ' + str(str(round(np.std(roc_RF_20),2))) + '\\\\\n')
            Perf.write( ' & ' + ' & '+ 'Neutral' + ' & ' + str(round(np.mean(roc_NB_Neutral),2)) + ' $\pm$ ' + str(str(round(np.std(roc_NB_Neutral),2))) + ' & ' + str(round(np.mean(roc_RF_20_Neutral),2)) + ' $\pm$ ' + str(str(round(np.std(roc_RF_20_Neutral),2))) + '\\\\\\cline{2-6}\n')
            Perf.write( ' & ' + ' & '+ 'Fast' + ' & ' + str(round(np.mean(roc_NB_Last),2)) + ' $\pm$ ' + str(str(round(np.std(roc_NB_Last),2))) + ' & ' + str(round(np.mean(roc_RF_20_Last),2)) + ' $\pm$ ' + str(str(round(np.std(roc_RF_20_Last),2))) + '\\\\\\cline{2-5}\n')



            Recall.write('\multirow{6}{*}{'+str(window)+'d}' + ' & ' + '\multirow{3}{*}{'+str(ntp)+'d}' + ' & '+ 'Slow' + ' & ' + str(round(np.mean(recall_NB),2)) + ' $\pm$ ' + str(str(round(np.std(recall_NB),2))) + ' & ' + str(round(np.mean(recall_RF_20),2)) + ' $\pm$ ' + str(str(round(np.std(recall_RF_20),2))) + '\\\\\\cline{2-5}\n')
            Recall.write( ' & ' + ' & '+ 'Neutral' + ' & ' + str(round(np.mean(recall_NB_Neutral),2)) + ' $\pm$ ' + str(str(round(np.std(recall_NB_Neutral),2))) + ' & ' + str(round(np.mean(recall_RF_20_Neutral),2)) + ' $\pm$ ' + str(str(round(np.std(recall_RF_20_Neutral),2))) + '\\\\\\cline{2-5}\n')
            Recall.write(' & ' + ' & '+ 'Fast' + ' & ' + str(round(np.mean(recall_NB_Last),2)) + ' $\pm$ ' + str(str(round(np.std(recall_NB_Last),2))) + ' & ' +str(round(np.mean(recall_RF_20_Last),2)) + ' $\pm$ ' + str(str(round(np.std(recall_RF_20_Last),2))) + '\\\\\\cline{2-5}\n')


            Precision.write('\multirow{6}{*}{'+str(window)+'d}' + ' & ' + '\multirow{3}{*}{'+str(ntp)+'d}'+ ' & ' + 'Slow' + ' & ' + str(round(np.mean(precision_NB),2)) + ' $\pm$ ' + str(str(round(np.std(precision_NB),2))) + ' & '  + str(round(np.mean(precision_RF_20),2)) + ' $\pm$ ' + str(str(round(np.std(precision_RF_20),2))) + '\\\\\n')
            Precision.write( ' & ' + ' & '+ 'Neutral' + ' & ' + str(round(np.mean(precision_NB_Neutral),2)) + ' $\pm$ ' + str(str(round(np.std(precision_NB_Neutral),2))) + ' & ' + str(round(np.mean(precision_RF_20_Neutral),2)) + ' $\pm$ ' + str(str(round(np.std(precision_RF_20_Neutral),2))) + '\\\\\\cline{2-5}\n')
            Precision.write(' & ' + ' & ' + 'Fast' + ' & ' + str(round(np.mean(precision_NB_Last),2)) + ' $\pm$ ' + str(str(round(np.std(precision_NB_Last),2))) + ' & ' +  str(round(np.mean(precision_RF_20_Last),2)) + ' $\pm$ ' + str(str(round(np.std(precision_RF_20_Last),2))) + '\\\\\\cline{2-5}\n')


        else:
            Perf.write( ' & ' + '\multirow{3}{*}{' + str(
                ntp) + 'd}' + ' & ' + 'Slow' + ' & ' + str(round(np.mean(roc_NB), 2)) + ' $\pm$ ' + str(
                str(round(np.std(roc_NB), 2))) + ' & ' + str(round(np.mean(roc_RF_20), 2)) + ' $\pm$ ' + str(
                str(round(np.std(roc_RF_20), 2))) + '\\\\\n')
            Perf.write(' & ' + ' & ' + 'Neutral' + ' & ' + str(round(np.mean(roc_NB_Neutral), 2)) + ' $\pm$ ' + str(
                str(round(np.std(roc_NB_Neutral), 2))) + ' & ' + str(
                round(np.mean(roc_RF_20_Neutral), 2)) + ' $\pm$ ' + str(
                str(round(np.std(roc_RF_20_Neutral), 2))) + '\\\\\\cline{2-5}\n')
            Perf.write(' & ' + ' & ' + 'Fast' + ' & ' + str(round(np.mean(roc_NB_Last), 2)) + ' $\pm$ ' + str(
                str(round(np.std(roc_NB_Last), 2))) + ' & '+ str(round(np.mean(roc_RF_20_Last), 2)) + ' $\pm$ ' + str(
                str(round(np.std(roc_RF_20_Last), 2))) + '\\\\\\cline{2-5}\n')



            Recall.write( ' & ' + '\multirow{3}{*}{' + str(
                ntp) + 'd}' + ' & ' + 'Slow' + ' & ' + str(round(np.mean(recall_NB), 2)) + ' $\pm$ ' + str(
                str(round(np.std(recall_NB), 2))) + ' & ' + str(round(np.mean(recall_RF_20), 2)) + ' $\pm$ ' + str(
                str(round(np.std(recall_RF_20), 2))) + '\\\\\n')
            Recall.write(
                ' & ' + ' & ' + 'Neutral' + ' & ' + str(round(np.mean(recall_NB_Neutral), 2)) + ' $\pm$ ' + str(
                    str(round(np.std(recall_NB_Neutral), 2))) + ' & ' + str(
                    round(np.mean(recall_RF_20_Neutral), 2)) + ' $\pm$ ' + str(
                    str(round(np.std(recall_RF_20_Neutral), 2))) + '\\\\\\cline{2-5}\n')
            Recall.write(' & ' + ' & ' + 'Fast' + ' & ' + str(round(np.mean(recall_NB_Last), 2)) + ' $\pm$ ' + str(
                str(round(np.std(recall_NB_Last), 2))) + ' & ' + str(
                round(np.mean(recall_RF_20_Last), 2)) + ' $\pm$ ' + str(
                str(round(np.std(recall_RF_20_Last), 2))) + '\\\\\\cline{2-5}\n')


            Precision.write( ' & ' + '\multirow{3}{*}{' + str(
                ntp) + 'd}' + ' & ' + 'Slow' + ' & ' + str(round(np.mean(precision_NB), 2)) + ' $\pm$ ' + str(
                str(round(np.std(precision_NB), 2))) +  ' & ' + str(round(np.mean(precision_RF_20), 2)) + ' $\pm$ ' + str(
                str(round(np.std(precision_RF_20), 2))) + '\\\\\n')
            Precision.write(
                ' & ' + ' & ' + 'Neutral' + ' & ' + str(round(np.mean(precision_NB_Neutral), 2)) + ' $\pm$ ' + str(
                    str(round(np.std(precision_NB_Neutral), 2))) + ' & ' + str(
                    round(np.mean(precision_RF_20_Neutral), 2)) + ' $\pm$ ' + str(
                    str(round(np.std(precision_RF_20_Neutral), 2))) + '\\\\\\cline{2-5}\n')
            Precision.write(' & ' + ' & ' + 'Fast' + ' & ' + str(round(np.mean(precision_NB_Last), 2)) + ' $\pm$ ' + str(
                str(round(np.std(precision_NB_Last), 2))) + ' & '  + str(
                round(np.mean(precision_RF_20_Last), 2)) + ' $\pm$ ' + str(
                str(round(np.std(precision_RF_20_Last), 2))) + '\\\\\\cline{2-5}\n')





jvm.stop()
