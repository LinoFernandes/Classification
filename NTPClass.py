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

directory = '/Users/Lino/PycharmProjects/Classification/NTPClass/New2/'
if not os.path.exists(directory):
    os.makedirs(directory)
Perf = open(directory + 'NTPClass_AUC.txt', 'a')
Recall = open(directory + 'NTPClassS_Recall.txt', 'a')
Precision = open(directory + 'NTPClass_Precision.txt', 'a')
Scores = open(directory + 'NTPClass_BoundsFirst_Final.csv', 'a')
ScoresLast = open(directory + 'NTPClass_BoundsLast_Final.csv', 'a')

Seeds = [42 , 467, 14, 9, 125]
smote=50

for window in Window:
    for ntp in range(2, 6):
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
        for classifier in range(0, 3):
            if sys.platform == "darwin":
                path = '/Users/Lino/PycharmProjects/Preprocessing/NTP/New2/' + str(ntp) + 'TP'
                lastpath = '/Users/Lino/PycharmProjects/Preprocessing/NTPtoLast/New2/' + str(ntp) + 'TP'
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
            for seed in Seeds:
                for fold in range(1,11):

                    from weka.filters import Filter
                    StratifiedCV = Filter(classname="weka.filters.supervised.instance.StratifiedRemoveFolds", options=['-S', str(seed), '-N' ,'10', '-F', str(fold)])
                    StratifiedCV.inputformat(data)
                    dataTest = StratifiedCV.filter(data)
                    StratifiedCV.inputformat(dataLast)
                    dataLastTest = StratifiedCV.filter(dataLast)

                    StratifiedCV = Filter(classname="weka.filters.supervised.instance.StratifiedRemoveFolds",
                                          options=['-S', str(seed), '-V' ,'-N', '10', '-F', str(fold)])
                    StratifiedCV.inputformat(data)
                    dataTrain = StratifiedCV.filter(data)
                    StratifiedCV.inputformat(dataLast)
                    dataLastTrain = StratifiedCV.filter(dataLast)

                    ClassIndex = dataTrain.attribute(dataTrain.class_index).values
                    yIndex = ClassIndex.index('Y')
                    
                    
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



                    # ReplaceMV = Filter(classname="weka.filters.unsupervised.attribute.ReplaceMissingValues")
                    # ReplaceMV.inputformat(dataTrain)
                    # dataTrain = ReplaceMV.filter(dataTrain)
                    # ReplaceMV.inputformat(dataTest)
                    # dataTest = ReplaceMV.filter(dataTest)
                    #
                    # ReplaceMV.inputformat(dataLastTrain)
                    # dataLastTrain = ReplaceMV.filter(dataLastTrain)
                    # ReplaceMV.inputformat(dataLastTest)
                    # dataLastTest = ReplaceMV.filter(dataLastTest)

                    #
                    # FS = Filter(classname="weka.filters.supervised.attribute.AttributeSelection",
                    #             options=['-E', 'weka.attributeSelection.CfsSubsetEval -P 1 -E 1', '-S',
                    #                      'weka.attributeSelection.RerankingSearch -method 2 -blockSize 20 -rankingMeasure 0 -search "weka.attributeSelection.GreedyStepwise -T -1.7976931348623157E308 -N -1 -num-slots 1"'])

                    FS = Filter(classname="weka.filters.supervised.attribute.AttributeSelection",
                                options=['-E', 'weka.attributeSelection.CfsSubsetEval -P 1 -E 1', '-S',
                                         "weka.attributeSelection.GreedyStepwise -T -1.7976931348623157E308 -N -1 -num-slots 1"])

                    FS.inputformat(dataTrain)
                    dataTrain = FS.filter(dataTrain)
                    dataTest = FS.filter(dataTest)

                    FS.inputformat(dataLastTrain)
                    dataLastTrain = FS.filter(dataLastTrain)
                    dataLastTest = FS.filter(dataLastTest)


                    dataTrain.class_is_last()
                    dataTest.class_is_last()

                    dataLastTrain.class_is_last()
                    dataLastTest.class_is_last()

                    from weka.classifiers import Evaluation
                    from weka.core.classes import Random
                    from weka.classifiers import Classifier
                    if classifier == 0:
                        SMOTE = Filter(classname="weka.filters.supervised.instance.SMOTE", options=['-P', str(smote)])
                        SMOTE.inputformat(dataTrain)
                        dataTrain = SMOTE.filter(dataTrain)

                        SMOTE.inputformat(dataLastTrain)
                        dataLastTrain = SMOTE.filter(dataLastTrain)
                        for kernel in range(0,1):
                            if kernel == 0:
                                mapper = Classifier(classname="weka.classifiers.misc.InputMappedClassifier", options=["-M","-W", "weka.classifiers.bayes.NaiveBayes"])
                                Class = 'NaiveBayes'
                                mapper.build_classifier(dataTrain)
                                evaluation = Evaluation(dataTrain)
                                evaluation.test_model(mapper,dataTest)
                                roc_NB.append(evaluation.area_under_roc(1)*100)
                                recall_NB.append(evaluation.recall(yIndex)*100)
                                precision_NB.append(evaluation.precision(yIndex)*100)

                                mapper.build_classifier(dataLastTrain)
                                evaluation = Evaluation(dataLastTrain)
                                evaluation.test_model(mapper, dataLastTest)

                                roc_NB_Last.append(evaluation.area_under_roc(1) * 100)
                                recall_NB_Last.append(evaluation.recall(yIndex) * 100)
                                precision_NB_Last.append(evaluation.precision(yIndex) * 100)
                    elif classifier == 1:
                        for degree in [2]:
                            mapper = Classifier(classname="weka.classifiers.misc.InputMappedClassifier", options=["-M","-W", "weka.classifiers.functions.SMO", "--", "-K","weka.classifiers.functions.supportVector.PolyKernel -E " + str(degree)])
                            Class = 'SVM'

                            if ((window == 90) and (ntp == 3 or ntp == 4)):

                                SMOTE = Filter(classname="weka.filters.supervised.instance.SMOTE", options=['-P', '200'])
                                SMOTE.inputformat(dataTrain)
                                dataTrain = SMOTE.filter(dataTrain)

                                SMOTE.inputformat(dataLastTrain)
                                dataLastTrain = SMOTE.filter(dataLastTrain)

                                mapper.build_classifier(dataTrain)
                                evaluation = Evaluation(dataTrain)
                                evaluation.test_model(mapper, dataTest)

                                roc_SVM_2.append(evaluation.area_under_roc(1)*100)
                                recall_SVM_2.append(evaluation.recall(yIndex) * 100)
                                precision_SVM_2.append(evaluation.precision(yIndex) * 100)

                                mapper.build_classifier(dataLastTrain)
                                evaluation = Evaluation(dataLastTrain)
                                evaluation.test_model(mapper, dataLastTest)

                                roc_SVM_2_Last.append(evaluation.area_under_roc(1) * 100)
                                recall_SVM_2_Last.append(evaluation.recall(yIndex) * 100)
                                precision_SVM_2_Last.append(evaluation.precision(yIndex) * 100)

                            elif ((window == 180) and (ntp == 2 or ntp == 5)):
                                SMOTE = Filter(classname="weka.filters.supervised.instance.SMOTE", options=['-P', '200'])
                                SMOTE.inputformat(dataTrain)
                                dataTrain = SMOTE.filter(dataTrain)

                                SMOTE.inputformat(dataLastTrain)
                                dataLastTrain = SMOTE.filter(dataLastTrain)
                                mapper.build_classifier(dataTrain)
                                evaluation = Evaluation(dataTrain)
                                evaluation.test_model(mapper, dataTest)

                                roc_SVM_2.append(evaluation.area_under_roc(1) * 100)
                                recall_SVM_2.append(evaluation.recall(yIndex) * 100)
                                precision_SVM_2.append(evaluation.precision(yIndex) * 100)

                                mapper.build_classifier(dataLastTrain)
                                evaluation = Evaluation(dataLastTrain)
                                evaluation.test_model(mapper, dataLastTest)

                                roc_SVM_2_Last.append(evaluation.area_under_roc(1) * 100)
                                recall_SVM_2_Last.append(evaluation.recall(yIndex) * 100)
                                precision_SVM_2_Last.append(evaluation.precision(yIndex) * 100)
                            else:
                                SMOTE = Filter(classname="weka.filters.supervised.instance.SMOTE",
                                               options=['-P', str(smote)])
                                SMOTE.inputformat(dataTrain)
                                dataTrain = SMOTE.filter(dataTrain)

                                SMOTE.inputformat(dataLastTrain)
                                dataLastTrain = SMOTE.filter(dataLastTrain)
                                mapper.build_classifier(dataTrain)
                                evaluation = Evaluation(dataTrain)
                                evaluation.test_model(mapper, dataTest)

                                roc_SVM_2.append(evaluation.area_under_roc(1) * 100)
                                recall_SVM_2.append(evaluation.recall(yIndex) * 100)
                                precision_SVM_2.append(evaluation.precision(yIndex) * 100)

                                mapper.build_classifier(dataLastTrain)
                                evaluation = Evaluation(dataLastTrain)
                                evaluation.test_model(mapper, dataLastTest)

                                roc_SVM_2_Last.append(evaluation.area_under_roc(1) * 100)
                                recall_SVM_2_Last.append(evaluation.recall(yIndex) * 100)
                                precision_SVM_2_Last.append(evaluation.precision(yIndex) * 100)

                    else:
                        for numTrees in [20]:
                            SMOTE = Filter(classname="weka.filters.supervised.instance.SMOTE",
                                           options=['-P', str(smote)])
                            SMOTE.inputformat(dataTrain)
                            dataTrain = SMOTE.filter(dataTrain)

                            SMOTE.inputformat(dataLastTrain)
                            dataLastTrain = SMOTE.filter(dataLastTrain)

                            mapper = Classifier(classname="weka.classifiers.misc.InputMappedClassifier", options=["-M","-W", "weka.classifiers.trees.RandomForest", "--", "-I", str(numTrees)])
                            Class = 'RF'
                            mapper.build_classifier(dataTrain)
                            evaluation = Evaluation(dataTrain)
                            evaluation.test_model(mapper, dataTest)

                            roc_RF_20.append(evaluation.area_under_roc(1) * 100)
                            recall_RF_20.append(evaluation.recall(yIndex) * 100)
                            precision_RF_20.append(evaluation.precision(yIndex) * 100)

                            mapper.build_classifier(dataLastTrain)
                            evaluation = Evaluation(dataLastTrain)
                            evaluation.test_model(mapper, dataLastTest)

                            roc_RF_20_Last.append(evaluation.area_under_roc(1) * 100)
                            recall_RF_20_Last.append(evaluation.recall(yIndex) * 100)
                            precision_RF_20_Last.append(evaluation.precision(yIndex) * 100)
        Scores.write(str(ntp) + ',' + str(round(np.mean(roc_NB)+np.std(roc_NB),2)) + ',' + str(round(np.mean(roc_NB)-np.std(roc_NB),2))+','+str(ntp)+ ',' + str(round(np.mean(roc_RF_20)+np.std(roc_RF_20),2)) + ',' + str(round(np.mean(roc_RF_20)-np.std(roc_RF_20),2)) + '\n')

        ScoresLast.write(str(ntp) + ',' + str(round(np.mean(roc_NB_Last)+np.std(roc_NB),2)) + ',' + str(round(np.mean(roc_NB_Last)-np.std(roc_NB_Last),2)) + ',' + str(round(np.mean(roc_RF_20_Last)+np.std(roc_RF_20_Last),2)) +','+ str(round(np.mean(roc_RF_20_Last)-np.std(roc_RF_20_Last),2)) + '\n')


        if ntp == 2:
            Perf.write('\\thickline\n')
            Precision.write('\\thickline\n')
            Recall.write('\\thickline\n')

            Perf.write('\multirow{8}{*}{'+str(window)+'d}' + ' & ' + '\multirow{2}{*}{'+str(ntp)+'d}' + ' & '+ 'FirstT' + ' & ' + str(round(np.mean(roc_NB),2)) + ' $\pm$ ' + str(str(round(np.std(roc_NB),2))) + ' & ' +  str(round(np.mean(roc_SVM_2),2)) + ' $\pm$ ' + str(str(round(np.std(roc_SVM_2),2))) + ' & ' + str(round(np.mean(roc_RF_20),2)) + ' $\pm$ ' + str(str(round(np.std(roc_RF_20),2))) + '\\\\\n')
            Perf.write( ' & ' + ' & '+ 'Last' + ' & ' + str(round(np.mean(roc_NB_Last),2)) + ' $\pm$ ' + str(str(round(np.std(roc_NB_Last),2))) + ' & ' +  str(round(np.mean(roc_SVM_2_Last),2)) + ' $\pm$ ' + str(str(round(np.std(roc_SVM_2_Last),2))) + ' & ' + str(round(np.mean(roc_RF_20_Last),2)) + ' $\pm$ ' + str(str(round(np.std(roc_RF_20_Last),2))) + '\\\\\\cline{2-6}\n')

            Recall.write('\multirow{8}{*}{'+str(window)+'d}' + ' & ' + '\multirow{2}{*}{'+str(ntp)+'d}' + ' & '+ 'FirstT' + ' & ' + str(round(np.mean(recall_NB),2)) + ' $\pm$ ' + str(str(round(np.std(recall_NB),2))) + ' & ' +  str(round(np.mean(recall_SVM_2),2)) + ' $\pm$ ' + str(str(round(np.std(recall_SVM_2),2))) + ' & ' + str(round(np.mean(recall_RF_20),2)) + ' $\pm$ ' + str(str(round(np.std(recall_RF_20),2))) + '\\\\\\cline{2-6}\n')
            Recall.write(' & ' + ' & '+ 'Last' + ' & ' + str(round(np.mean(recall_NB_Last),2)) + ' $\pm$ ' + str(str(round(np.std(recall_NB_Last),2))) + ' & ' +  str(round(np.mean(recall_SVM_2_Last),2)) + ' $\pm$ ' + str(str(round(np.std(recall_SVM_2_Last),2))) + ' & ' + str(round(np.mean(recall_RF_20_Last),2)) + ' $\pm$ ' + str(str(round(np.std(recall_RF_20_Last),2))) + '\\\\\\cline{2-6}\n')

            Precision.write('\multirow{8}{*}{'+str(window)+'d}' + ' & ' + '\multirow{2}{*}{'+str(ntp)+'d}'+ ' & ' + 'FirstT' + ' & ' + str(round(np.mean(precision_NB),2)) + ' $\pm$ ' + str(str(round(np.std(precision_NB),2))) + ' & ' +  str(round(np.mean(precision_SVM_2),2)) + ' $\pm$ ' + str(str(round(np.std(precision_SVM_2),2))) + ' & ' + str(round(np.mean(precision_RF_20),2)) + ' $\pm$ ' + str(str(round(np.std(precision_RF_20),2))) + '\\\\\n')
            Precision.write(' & ' + ' & ' + 'Last' + ' & ' + str(round(np.mean(precision_NB_Last),2)) + ' $\pm$ ' + str(str(round(np.std(precision_NB_Last),2))) + ' & ' +  str(round(np.mean(precision_SVM_2_Last),2)) + ' $\pm$ ' + str(str(round(np.std(precision_SVM_2_Last),2))) + ' & ' + str(round(np.mean(precision_RF_20_Last),2)) + ' $\pm$ ' + str(str(round(np.std(precision_RF_20_Last),2))) + '\\\\\\cline{2-6}\n')
        else:
            Perf.write( ' & ' + '\multirow{2}{*}{' + str(
                ntp) + 'd}' + ' & ' + 'FirstT' + ' & ' + str(round(np.mean(roc_NB), 2)) + ' $\pm$ ' + str(
                str(round(np.std(roc_NB), 2))) + ' & ' + str(round(np.mean(roc_SVM_2), 2)) + ' $\pm$ ' + str(
                str(round(np.std(roc_SVM_2), 2))) + ' & ' + str(round(np.mean(roc_RF_20), 2)) + ' $\pm$ ' + str(
                str(round(np.std(roc_RF_20), 2))) + '\\\\\n')
            Perf.write(' & ' + ' & ' + 'Last' + ' & ' + str(round(np.mean(roc_NB_Last), 2)) + ' $\pm$ ' + str(
                str(round(np.std(roc_NB_Last), 2))) + ' & ' + str(round(np.mean(roc_SVM_2_Last), 2)) + ' $\pm$ ' + str(
                str(round(np.std(roc_SVM_2_Last), 2))) + ' & ' + str(round(np.mean(roc_RF_20_Last), 2)) + ' $\pm$ ' + str(
                str(round(np.std(roc_RF_20_Last), 2))) + '\\\\\\cline{2-6}\n')

            Recall.write( ' & ' + '\multirow{2}{*}{' + str(
                ntp) + 'd}' + ' & ' + 'FirstT' + ' & ' + str(round(np.mean(recall_NB), 2)) + ' $\pm$ ' + str(
                str(round(np.std(recall_NB), 2))) + ' & ' + str(round(np.mean(recall_SVM_2), 2)) + ' $\pm$ ' + str(
                str(round(np.std(recall_SVM_2), 2))) + ' & ' + str(round(np.mean(recall_RF_20), 2)) + ' $\pm$ ' + str(
                str(round(np.std(recall_RF_20), 2))) + '\\\\\n')
            Recall.write(' & ' + ' & ' + 'Last' + ' & ' + str(round(np.mean(recall_NB_Last), 2)) + ' $\pm$ ' + str(
                str(round(np.std(recall_NB_Last), 2))) + ' & ' + str(
                round(np.mean(recall_SVM_2_Last), 2)) + ' $\pm$ ' + str(
                str(round(np.std(recall_SVM_2_Last), 2))) + ' & ' + str(
                round(np.mean(recall_RF_20_Last), 2)) + ' $\pm$ ' + str(
                str(round(np.std(recall_RF_20_Last), 2))) + '\\\\\\cline{2-6}\n')

            Precision.write( ' & ' + '\multirow{2}{*}{' + str(
                ntp) + 'd}' + ' & ' + 'FirstT' + ' & ' + str(round(np.mean(precision_NB), 2)) + ' $\pm$ ' + str(
                str(round(np.std(precision_NB), 2))) + ' & ' + str(round(np.mean(precision_SVM_2), 2)) + ' $\pm$ ' + str(
                str(round(np.std(precision_SVM_2), 2))) + ' & ' + str(round(np.mean(precision_RF_20), 2)) + ' $\pm$ ' + str(
                str(round(np.std(precision_RF_20), 2))) + '\\\\\n')
            Precision.write(' & ' + ' & ' + 'Last' + ' & ' + str(round(np.mean(precision_NB_Last), 2)) + ' $\pm$ ' + str(
                str(round(np.std(precision_NB_Last), 2))) + ' & ' + str(
                round(np.mean(precision_SVM_2_Last), 2)) + ' $\pm$ ' + str(
                str(round(np.std(precision_SVM_2_Last), 2))) + ' & ' + str(
                round(np.mean(precision_RF_20_Last), 2)) + ' $\pm$ ' + str(
                str(round(np.std(precision_RF_20_Last), 2))) + '\\\\\\cline{2-6}\n')




jvm.stop()
print(MV)