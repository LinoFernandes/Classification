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



directory = '/Users/Lino/PycharmProjects/Classification/Snapshots/'
if not os.path.exists(directory):
    os.makedirs(directory)
Perf = open(directory + 'AUC_New_Final.txt', 'a')
Recall = open(directory + 'Recall_New_Final.txt', 'a')
Precision = open(directory + 'Precision_New_Final.txt', 'a')

ScoresMVI = open(directory + 'Scores_MVI_Final.csv', 'a')
ScoresOri = open(directory + 'Scores_Ori_Final.csv', 'a')
ScoresForProg = open(directory + 'Scores_ForProg_Final.csv', 'a')


for window in Window:
    Perf.write('k,DS,NB,SVM_2,RF_20\n')
    Recall.write('k,DS,NB,SVM_2,RF_20\n')
    Precision.write('k,DS,NB,SVM_2,RF_20\n')

    NB_AUC = np.zeros([5,10,3])
    RF_AUC = np.zeros([5,10,3])
    SVM_AUC = np.zeros([5,10,3])

    NB_Recall = np.zeros([5,10,3])
    RF_Recall = np.zeros([5,10,3])
    SVM_Recall = np.zeros([5,10,3])

    NB_Precision = np.zeros([5,10,3])
    RF_Precision = np.zeros([5,10,3])
    SVM_Precision = np.zeros([5,10,3])
    for seed in range(1, 6):
        for smote in [500]:
            # roc_NB = []
            # roc_NB_FS = []
            # roc_NB_MVI = []
            #
            # roc_SVM = []
            # roc_SVM_FS = []
            # roc_SVM_MVI = []
            #
            # roc_RF = []
            # roc_RF_FS = []
            # roc_RF_MVI = []
            #
            # recall_NB = []
            # recall_NB_FS = []
            # recall_NB_MVI = []
            #
            # recall_SVM = []
            # recall_SVM_FS = []
            # recall_SVM_MVI = []
            #
            # recall_RF = []
            # recall_RF_FS = []
            # recall_RF_MVI = []
            #
            # precision_NB = []
            # precision_NB_FS = []
            # precision_NB_MVI = []
            #
            # precision_SVM = []
            # precision_SVM_FS = []
            # precision_SVM_MVI = []
            #
            # precision_RF = []
            # precision_RF_FS = []
            # precision_RF_MVI = []

            for classifier in range(0, 3):
                for fold in range(1, 11):
                    if sys.platform == "darwin":
                        path = '/Users/Lino/PycharmProjects/Preprocessing/PreProcessedFolds/' + str(window) + 'd_FOLDS' +'/S' + str(seed) + '/' + str(window) + 'd_FOLDS_train_' + str(fold) +'.csv'
                        testpath = '/Users/Lino/PycharmProjects/Preprocessing/PreProcessedFolds/' + str(window) + 'd_FOLDS' +'/S' + str(seed) + '/' + str(window) + 'd_FOLDS_test_' + str(fold) +'.csv'
                        pathMV = '/Users/Lino/PycharmProjects/Preprocessing/PreProcessedFolds/' + str(
                            window) + 'd_FOLDS' + '/S' + str(seed) + '/' + str(window) + 'd_FOLDS_train_' + str(
                            fold) + '.csv'
                        testpathMV = '/Users/Lino/PycharmProjects/Preprocessing/PreProcessedFolds/' + str(
                            window) + 'd_FOLDS' + '/S' + str(seed) + '/' + str(window) + 'd_FOLDS_test_' + str(
                            fold) + '.csv'

                        directory = '/Users/Lino/PycharmProjects/Classification/NTPClass/'
                    #elif sys.platform == "win32":
                        #path = 'C:\\Users\\Lino\\PycharmProjects\\Preprocessing\\NTP\\' + str(ntp) + 'TP'
                    sys.path.append(path)
                    # try:
                    from weka.core.converters import Loader


                    
                    loader = Loader(classname="weka.core.converters.CSVLoader")
                    dataTrain = loader.load_file(path)
                    dataTest = loader.load_file(testpath)
                    dataTrain.class_is_last()
                    dataTest.class_is_last()
                    ClassIndex = dataTrain.attribute(33).values
                    yIndex = ClassIndex.index('Y')

                    dataTrainMV = loader.load_file(pathMV)
                    dataTestMV = loader.load_file(testpathMV)
                    dataTrainMV.class_is_last()
                    dataTestMV.class_is_last()

                    from weka.filters import Filter

                    toBeRemoved = []
                    for attribute in range(0, dataTrain.attributes().data.class_index):
                        if dataTrain.attribute_stats(
                                attribute).missing_count == dataTrain.attributes().data.num_instances and dataTest.attribute_stats(attribute).missing_count == dataTest.attributes().data.num_instances:
                            sys.exit("Fold has full missing column")
                        if (dataTrain.attribute_stats(
                                attribute).missing_count / dataTrain.attributes().data.num_instances) > 0.5 and (
                                    dataTest.attribute_stats(
                                        attribute).missing_count / dataTest.attributes().data.num_instances) > 0.5:
                            toBeRemoved.append(str(attribute))

                    Remove = Filter(classname="weka.filters.unsupervised.attribute.Remove", options=['-R', ','.join(toBeRemoved)])
                    Remove.inputformat(dataTrain)
                    dataTrain = Remove.filter(dataTrain)
                    Remove.inputformat(dataTest)
                    dataTest = Remove.filter(dataTest)

                    # Remove.inputformat(dataTrainMV)
                    # dataTrainMV = Remove.filter(dataTrainMV)
                    # Remove.inputformat(dataTestMV)
                    # dataTestMV = Remove.filter(dataTestMV)

                    import weka.core.classes as wcc

                    FS = Filter(classname="weka.filters.supervised.attribute.AttributeSelection",
                                options=['-E', 'weka.attributeSelection.CfsSubsetEval -P 1 -E 1', '-S',
                                         "weka.attributeSelection.GreedyStepwise -T -1.7976931348623157E308 -N -1 -num-slots 1"])
                    FS.inputformat(dataTrain)
                    dataTrainFS = FS.filter(dataTrain)
                    FS.inputformat(dataTest)
                    dataTestFS = FS.filter(dataTest)


                    ReplaceMV = Filter(classname="weka.filters.unsupervised.attribute.ReplaceMissingValues")
                    ReplaceMV.inputformat(dataTrain)
                    dataTrainMV = ReplaceMV.filter(dataTrain)
                    ReplaceMV.inputformat(dataTest)
                    dataTestMV = ReplaceMV.filter(dataTest)

                    # ReplaceMV.inputformat(dataTest)
                    # dataTest=ReplaceMV.filter(dataTest)


                    dataTrain.class_is_last()
                    dataTest.class_is_last()


                    from weka.classifiers import Evaluation
                    from weka.core.classes import Random
                    from weka.classifiers import Classifier

                    if classifier == 0:
                        for kernel in range(0, 1):
                            if kernel == 0:
                                mapper = Classifier(classname="weka.classifiers.misc.InputMappedClassifier",
                                                    options=["-M", "-W", "weka.classifiers.bayes.NaiveBayes"])
                                Class = 'NaiveBayes'
                                mapper.build_classifier(dataTrain)
                                evaluation = Evaluation(dataTrain)
                                evaluation.test_model(mapper, dataTest)

                                NB_AUC[seed-1,fold-1,0] = (evaluation.area_under_roc(1) * 100)
                                NB_Recall[seed-1,fold-1,0] =(evaluation.recall(yIndex) * 100)
                                NB_Precision[seed-1,fold-1,0] =(evaluation.precision(yIndex) * 100)

                                mapper.build_classifier(dataTrainFS)
                                evaluation = Evaluation(dataTrainFS)
                                evaluation.test_model(mapper, dataTestFS)

                                NB_AUC[seed - 1, fold - 1, 1] = (evaluation.area_under_roc(1) * 100)
                                NB_Recall[seed - 1, fold - 1, 1] = (evaluation.recall(yIndex) * 100)
                                NB_Precision[seed - 1, fold - 1, 1] = (evaluation.precision(yIndex) * 100)

                                mapper.build_classifier(dataTrainMV)
                                evaluation = Evaluation(dataTrainMV)
                                evaluation.test_model(mapper, dataTestMV)

                                NB_AUC[seed - 1, fold - 1, 2] = (evaluation.area_under_roc(1) * 100)
                                NB_Recall[seed - 1, fold - 1, 2] = (evaluation.recall(yIndex) * 100)
                                NB_Precision[seed - 1, fold - 1, 2] = (evaluation.precision(yIndex) * 100)
                    elif classifier == 1:
                        for degree in [1]:
                            mapper = Classifier(classname="weka.classifiers.misc.InputMappedClassifier",
                                                options=["-M", "-W", "weka.classifiers.functions.SMO", "--", "-K",
                                                         "weka.classifiers.functions.supportVector.PolyKernel -E " + str(
                                                             degree)])
                            Class = 'SVM'
                            mapper.build_classifier(dataTrain)
                            evaluation = Evaluation(dataTrain)
                            evaluation.test_model(mapper, dataTest)

                            SVM_AUC[seed-1,fold-1,0] = (evaluation.area_under_roc(1) * 100)
                            SVM_Recall[seed - 1, fold - 1, 0] = (evaluation.recall(yIndex) * 100)
                            SVM_Precision[seed - 1, fold - 1, 0] = (evaluation.precision(yIndex) * 100)

                            mapper.build_classifier(dataTrainFS)
                            evaluation = Evaluation(dataTrainFS)
                            evaluation.test_model(mapper, dataTestFS)

                            SVM_AUC[seed - 1, fold - 1, 1] = (evaluation.area_under_roc(1) * 100)
                            SVM_Recall[seed - 1, fold - 1, 1] = (evaluation.recall(yIndex) * 100)
                            SVM_Precision[seed - 1, fold - 1, 1] = (evaluation.precision(yIndex) * 100)

                            mapper.build_classifier(dataTrainMV)
                            evaluation = Evaluation(dataTrainMV)
                            evaluation.test_model(mapper, dataTestMV)

                            SVM_AUC[seed - 1, fold - 1, 2] = (evaluation.area_under_roc(1) * 100)
                            SVM_Recall[seed - 1, fold - 1, 2] = (evaluation.recall(yIndex) * 100)
                            SVM_Precision[seed - 1, fold - 1, 2] = (evaluation.precision(yIndex) * 100)

                    else:
                        for numTrees in [20]:

                            mapper = Classifier(classname="weka.classifiers.misc.InputMappedClassifier",
                                                options=["-M", "-W", "weka.classifiers.trees.RandomForest", "--", "-I",
                                                         str(numTrees)])
                            Class = 'RF'

                            mapper.build_classifier(dataTrain)
                            evaluation = Evaluation(dataTrain)
                            evaluation.test_model(mapper, dataTest)

                            RF_AUC[seed - 1, fold - 1, 0] = (evaluation.area_under_roc(1) * 100)
                            RF_Recall[seed - 1, fold - 1, 0] = (evaluation.recall(yIndex) * 100)
                            RF_Precision[seed - 1, fold - 1, 0] = (evaluation.precision(yIndex) * 100)

                            mapper.build_classifier(dataTrainFS)
                            evaluation = Evaluation(dataTrainFS)
                            evaluation.test_model(mapper, dataTestFS)

                            RF_AUC[seed - 1, fold - 1, 1] = (evaluation.area_under_roc(1) * 100)
                            RF_Recall[seed - 1, fold - 1, 1] = (evaluation.recall(yIndex) * 100)
                            RF_Precision[seed - 1, fold - 1, 1] = (evaluation.precision(yIndex) * 100)

                            mapper.build_classifier(dataTrainMV)
                            evaluation = Evaluation(dataTrainMV)
                            evaluation.test_model(mapper, dataTestMV)

                            RF_AUC[seed - 1, fold - 1, 2] = (evaluation.area_under_roc(1) * 100)
                            RF_Recall[seed - 1, fold - 1, 2] = (evaluation.recall(yIndex) * 100)
                            RF_Precision[seed - 1, fold - 1, 2] = (evaluation.precision(yIndex) * 100)

    ScoresOri.write('Ori,' + str(round(np.mean(np.mean(np.mean(NB_AUC, axis=0), axis=0)[0])+np.std(np.mean(NB_AUC, axis=0), axis=0)[0],2)) + ',' + str(round(np.mean(np.mean(np.mean(NB_AUC, axis=0), axis=0)[0])-np.std(np.mean(NB_AUC, axis=0), axis=0)[0],2))+ ',' + str(round(np.mean(np.mean(np.mean(RF_AUC, axis=0), axis=0)[0])+np.std(np.mean(RF_AUC, axis=0), axis=0)[0],2)) + ',' + str(round(np.mean(np.mean(np.mean(RF_AUC, axis=0), axis=0)[0])-np.std(np.mean(RF_AUC, axis=0), axis=0)[0],2)) + '\n')
    ScoresMVI.write('MVI,' + str(round(np.mean(np.mean(np.mean(NB_AUC, axis=0), axis=0)[2])+np.std(np.mean(NB_AUC, axis=0), axis=0)[2],2)) + ',' + str(round(np.mean(np.mean(np.mean(NB_AUC, axis=0), axis=0)[2])-np.std(np.mean(NB_AUC, axis=0), axis=0)[2],2))+ ',' + str(round(np.mean(np.mean(np.mean(RF_AUC, axis=0), axis=0)[2])+np.std(np.mean(RF_AUC, axis=0), axis=0)[2],2)) + ',' + str(round(np.mean(np.mean(np.mean(RF_AUC, axis=0), axis=0)[2])-np.std(np.mean(RF_AUC, axis=0), axis=0)[2],2)) + '\n')


    Perf.write('\multirow{4}{*}{' + str(window) + 'd}' + ' & ' + str('Ori') + ' & ' + str(
        np.round(np.mean(np.mean(NB_AUC, axis=0), axis=0)[0], 2)) + ' $\pm$ ' + str(
        np.round(np.std(np.mean(NB_AUC, axis=0), axis=0)[0], 2)) + ' & ' + str(
        np.round(np.mean(np.mean(SVM_AUC, axis=0), axis=0)[0], 2)) + ' $\pm$ ' + str(
        np.round(np.std(np.mean(SVM_AUC, axis=0), axis=0)[0], 2)) + ' & ' + str(
        np.round(np.mean(np.mean(RF_AUC, axis=0), axis=0)[0], 2)) + ' $\pm$ ' + str(
        np.round(np.std(np.mean(RF_AUC, axis=0), axis=0)[0], 2)) + '\\\\\n')
    Perf.write(
        ' & ' + str('FS') + ' & ' + str(np.round(np.mean(np.mean(NB_AUC, axis=0), axis=0)[1], 2)) + ' $\pm$ ' + str(
            np.round(np.std(np.mean(NB_AUC, axis=0), axis=0)[1], 2)) + ' & ' + str(
            np.round(np.mean(np.mean(SVM_AUC, axis=0), axis=0)[1], 2)) + ' $\pm$ ' + str(
            np.round(np.std(np.mean(SVM_AUC, axis=0), axis=0)[1], 2)) + ' & ' + str(
            np.round(np.mean(np.mean(RF_AUC, axis=0), axis=0)[1], 2)) + ' $\pm$ ' + str(
            np.round(np.std(np.mean(RF_AUC, axis=0), axis=0)[1], 2)) + '\\\\\n')
    Perf.write(
        ' & ' + str('MVI') + ' & ' + str(np.round(np.mean(np.mean(NB_AUC, axis=0), axis=0)[2], 2)) + ' $\pm$ ' + str(
            np.round(np.std(np.mean(NB_AUC, axis=0), axis=0)[2], 2)) + ' & ' + str(
            np.round(np.mean(np.mean(SVM_AUC, axis=0), axis=0)[2], 2)) + ' $\pm$ ' + str(
            np.round(np.std(np.mean(SVM_AUC, axis=0), axis=0)[2], 2)) + ' & ' + str(
            np.round(np.mean(np.mean(RF_AUC, axis=0), axis=0)[2], 2)) + ' $\pm$ ' + str(
            np.round(np.std(np.mean(RF_AUC, axis=0), axis=0)[2], 2)) + '\\\\\n')

    Precision.write('\multirow{4}{*}{' + str(window) + 'd}' + ' & ' + str('Ori') + ' & ' + str(
        np.round(np.mean(np.mean(NB_Precision, axis=0), axis=0)[0], 2)) + ' $\pm$ ' + str(
        np.round(np.std(np.mean(NB_Precision, axis=0), axis=0)[0], 2)) + ' & ' + str(
        np.round(np.mean(np.mean(SVM_Precision, axis=0), axis=0)[0], 2)) + ' $\pm$ ' + str(
        np.round(np.std(np.mean(SVM_Precision, axis=0), axis=0)[0], 2)) + ' & ' + str(
        np.round(np.mean(np.mean(RF_Precision, axis=0), axis=0)[0], 2)) + ' $\pm$ ' + str(
        np.round(np.std(np.mean(RF_Precision, axis=0), axis=0)[0], 2)) + '\\\\\n')
    Precision.write(' & ' + str('FS') + ' & ' + str(
        np.round(np.mean(np.mean(NB_Precision, axis=0), axis=0)[1], 2)) + ' $\pm$ ' + str(
        np.round(np.std(np.mean(NB_Precision, axis=0), axis=0)[1], 2)) + ' & ' + str(
        np.round(np.mean(np.mean(SVM_Precision, axis=0), axis=0)[1], 2)) + ' $\pm$ ' + str(
        np.round(np.std(np.mean(SVM_Precision, axis=0), axis=0)[1], 2)) + ' & ' + str(
        np.round(np.mean(np.mean(RF_Precision, axis=0), axis=0)[1], 2)) + ' $\pm$ ' + str(
        np.round(np.std(np.mean(RF_Precision, axis=0), axis=0)[1], 2)) + '\\\\\n')
    Precision.write(' & ' + str('MVI') + ' & ' + str(
        np.round(np.mean(np.mean(NB_Precision, axis=0), axis=0)[2], 2)) + ' $\pm$ ' + str(
        np.round(np.std(np.mean(NB_Precision, axis=0), axis=0)[2], 2)) + ' & ' + str(
        np.round(np.mean(np.mean(SVM_Precision, axis=0), axis=0)[2], 2)) + ' $\pm$ ' + str(
        np.round(np.std(np.mean(SVM_Precision, axis=0), axis=0)[2], 2)) + ' & ' + str(
        np.round(np.mean(np.mean(RF_Precision, axis=0), axis=0)[2], 2)) + ' $\pm$ ' + str(
        np.round(np.std(np.mean(RF_Precision, axis=0), axis=0)[2], 2)) + '\\\\\n')

    Recall.write('\multirow{4}{*}{' + str(window) + 'd}' + ' & ' + str('Ori') + ' & ' + str(
        np.round(np.mean(np.mean(NB_Recall, axis=0), axis=0)[0], 2)) + ' $\pm$ ' + str(
        np.round(np.std(np.mean(NB_Recall, axis=0), axis=0)[0], 2)) + ' & ' + str(
        np.round(np.mean(np.mean(SVM_Recall, axis=0), axis=0)[0], 2)) + ' $\pm$ ' + str(
        np.round(np.std(np.mean(SVM_Recall, axis=0), axis=0)[0], 2)) + ' & ' + str(
        np.round(np.mean(np.mean(RF_Recall, axis=0), axis=0)[0], 2)) + ' $\pm$ ' + str(
        np.round(np.std(np.mean(RF_Recall, axis=0), axis=0)[0], 2)) + '\\\\\n')
    Recall.write(
        ' & ' + str('FS') + ' & ' + str(np.round(np.mean(np.mean(NB_Recall, axis=0), axis=0)[1], 2)) + ' $\pm$ ' + str(
            np.round(np.std(np.mean(NB_Recall, axis=0), axis=0)[1], 2)) + ' & ' + str(
            np.round(np.mean(np.mean(SVM_Recall, axis=0), axis=0)[1], 2)) + ' $\pm$ ' + str(
            np.round(np.std(np.mean(SVM_Recall, axis=0), axis=0)[1], 2)) + ' & ' + str(
            np.round(np.mean(np.mean(RF_Recall, axis=0), axis=0)[1], 2)) + ' $\pm$ ' + str(
            np.round(np.std(np.mean(RF_Recall, axis=0), axis=0)[1], 2)) + '\\\\\n')
    Recall.write(
        ' & ' + str('MVI') + ' & ' + str(np.round(np.mean(np.mean(NB_Recall, axis=0), axis=0)[2], 2)) + ' $\pm$ ' + str(
            np.round(np.std(np.mean(NB_Recall, axis=0), axis=0)[2], 2)) + ' & ' + str(
            np.round(np.mean(np.mean(SVM_Recall, axis=0), axis=0)[2], 2)) + ' $\pm$ ' + str(
            np.round(np.std(np.mean(SVM_Recall, axis=0), axis=0)[2], 2)) + ' & ' + str(
            np.round(np.mean(np.mean(RF_Recall, axis=0), axis=0)[2], 2)) + ' $\pm$ ' + str(
            np.round(np.std(np.mean(RF_Recall, axis=0), axis=0)[2], 2)) + '\\\\\n')
jvm.stop()
