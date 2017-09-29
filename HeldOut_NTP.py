import pandas as pd
import numpy as np
import sys
import weka.core.jvm as jvm
import os
import matplotlib.pyplot as plt

from weka.core.classes import from_commandline

os.environ["JAVA_HOME"] = "/Library/Java/JavaVirtualMachines/jdk1.8.0_131.jdk"
jvm.start(class_path=['/Users/Lino/wekafiles/packages/SMOTE/SMOTE.jar',
                      '/Users/Lino/wekafiles/packages/RerankingSearch/RerankingSearch.jar',
                      '/Applications/weka-3-9-1-oracle-jvm.app/Contents/Java/weka.jar'])

Window = np.array([90, 180, 365])



directory = '/Users/Lino/PycharmProjects/Classification/NTPClass/New/HeldOut/'
directoryFirst = '/Users/Lino/PycharmProjects/Preprocessing/NTP/New2/'
directoryLast = '/Users/Lino/PycharmProjects/Preprocessing/NTPtoLast/New2/'
if not os.path.exists(directory):
    os.makedirs(directory)
Perf = open(directory + 'AUC_Test_Set.txt', 'a')
Recall = open(directory + 'Recall_Test_Set.txt', 'a')
Precision = open(directory + 'Precision_Test_Set.txt', 'a')
Scores = open(directory + 'Scores_Set.csv', 'a')
ScoresLast = open(directory + 'ScoresLast_Set.csv', 'a')

Perf.write('k,TP,Dataset,NB,RF\\\\\n')
Recall.write('k,TP,DS,NB,RF\\\\\n')
Precision.write('k,TP,DS,NB,RF\\\\\n')
for window in Window:
    for ntp in range(2,6):
        for dataset in ['First','Last']:
            if dataset == 'First':
                path = directoryFirst + str(ntp) + 'TP/' + str(window) + 'd_' + str(ntp) + '.csv'
                testpath = directoryFirst + str(ntp) + 'TP/' + str(window) + 'd_' + str(ntp) + '_Test.csv'
            else:
                path = directoryLast + str(ntp) + 'TP/' + str(window) + 'd_' + str(ntp-1) + 'to' + str(ntp-1) + '.csv'
                testpath = directoryLast + str(ntp) + 'TP/' + str(window) + 'd_' + str(ntp-1) + 'to' + str(ntp-1) + '_Test.csv'

            from weka.core.converters import Loader

            loader = Loader(classname="weka.core.converters.CSVLoader")
            dataTrain = loader.load_file(path)
            dataTest = loader.load_file(testpath)
            dataTrain.class_is_last()
            dataTest.class_is_last()

            ClassIndex = dataTrain.attribute(dataTrain.class_index).values
            yIndex = ClassIndex.index('Y')

            from weka.filters import Filter

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
            Remove.inputformat(dataTest)
            dataTest = Remove.filter(dataTest)

            # ReplaceMV = Filter(classname="weka.filters.unsupervised.attribute.ReplaceMissingValues")
            # ReplaceMV.inputformat(dataTrain)
            # dataTrain = ReplaceMV.filter(dataTrain)
            # ReplaceMV.inputformat(dataTest)
            # dataTest = ReplaceMV.filter(dataTest)

            FS = Filter(classname="weka.filters.supervised.attribute.AttributeSelection",
                        options=['-E', 'weka.attributeSelection.CfsSubsetEval -P 1 -E 1', '-S',
                                 "weka.attributeSelection.GreedyStepwise -R -T -1.7976931348623157E308 -N -1 -num-slots 1"])

            FS.inputformat(dataTrain)
            dataTrain = FS.filter(dataTrain)
            FS.inputformat(dataTest)
            dataTest = FS.filter(dataTest)

            SMOTE = Filter(classname="weka.filters.supervised.instance.SMOTE", options=['-P', '50'])
            SMOTE.inputformat(dataTrain)
            dataTrain = SMOTE.filter(dataTrain)


            dataTrain.class_is_last()
            dataTest.class_is_last()

            from weka.classifiers import Evaluation
            from weka.core.classes import Random
            from weka.classifiers import Classifier

            NB = Classifier(classname="weka.classifiers.misc.InputMappedClassifier",
                            options=["-M", "-W", "weka.classifiers.bayes.NaiveBayes"])
            Class = 'NaiveBayes'
            NB.build_classifier(dataTrain)
            evaluationNB = Evaluation(dataTrain)
            evaluationNB.test_model(NB, dataTest)

            RF = Classifier(classname="weka.classifiers.misc.InputMappedClassifier",
                            options=["-M", "-W", "weka.classifiers.trees.RandomForest", "--", "-I",
                                     '20'])
            Class = 'NaiveBayes'
            RF.build_classifier(dataTrain)
            evaluationRF = Evaluation(dataTrain)
            evaluationRF.test_model(RF, dataTest)

            if dataset == 'First':
                Scores.write(str(window)+','+str(np.round(evaluationNB.area_under_roc(1) * 100, 2)) +','+str(np.round(evaluationRF.area_under_roc(1) * 100, 2))+'\n')
            else:
                ScoresLast.write(str(window)+','+str(np.round(evaluationNB.area_under_roc(1) * 100, 2)) +','+str(np.round(evaluationRF.area_under_roc(1) * 100, 2))+'\n')

            if ntp == 2 and dataset == 'First':

                Perf.write(
                    '\multirow{8}{*}{' + str(window) + 'd}' + ' & ' + '\multirow{2}{*}{' + str(
                        ntp) + '}' + ' & ' + dataset + ' & ' + str(
                        np.round(evaluationNB.area_under_roc(1) * 100, 2)) + ' & ' + str(
                        np.round(evaluationRF.area_under_roc(1) * 100, 2)) + '\\\\\n')

                Precision.write(
                    '\multirow{8}{*}{' + str(window) + 'd}' + ' & ' + '\multirow{2}{*}{' + str(
                        ntp) + '}' + ' & ' + dataset + ' & ' + str(
                        np.round(evaluationNB.precision(yIndex) * 100, 2)) + ' & ' + str(
                        np.round(evaluationRF.precision(yIndex) * 100, 2)) + '\\\\\n')

                Recall.write('\multirow{8}{*}{' + str(window) + 'd}' + ' & ' + '\multirow{2}{*}{' + str(
                    ntp) + '}' + ' & ' + dataset + ' & ' + str(np.round(evaluationNB.recall(yIndex) * 100, 2)) + ' & ' + str(
                    np.round(evaluationRF.recall(yIndex) * 100, 2)) + '\\\\\n')

            else:
                Perf.write(
                     ' & ' +  ' & ' + dataset + ' & ' + str(np.round(evaluationNB.area_under_roc(1) * 100, 2)) + ' & ' + str(
                        np.round(evaluationRF.area_under_roc(1) * 100, 2)) + '\\\\\n')

                Precision.write(
                    ' & ' + ' & ' + dataset + ' & ' + str(np.round(evaluationNB.precision(yIndex) * 100, 2)) + ' & ' + str(
                        np.round(evaluationRF.precision(yIndex) * 100, 2)) + '\\\\\n')

                Recall.write(' & '  + ' & ' + dataset + ' & ' + str(np.round(evaluationNB.recall(1) * 100, 2)) + ' & ' + str(
                    np.round(evaluationRF.recall(yIndex) * 100, 2)) + '\\\\\n')
jvm.stop()

