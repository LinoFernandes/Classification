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
Perf = open(directory + 'AUC_Test_Prog.txt', 'a')
Recall = open(directory + 'Recall_Test_Prog.txt', 'a')
Precision = open(directory + 'Precision_Test_Prog.txt', 'a')
Scores = open(directory + 'ScoresSnapProg', 'a')

Perf.write('k,DS,NB,RF_20\n')
Recall.write('k,DS,NB,RF_20\n')
Precision.write('k,DS,NB,RF_20\n')
Scores.write('k,DS,AUC\n')


for window in Window:
    for dataset in ['Slow','Neutral', 'Fast']:


        NB_AUC = []
        RF_AUC = []
        SVM_AUC = []

        NB_Recall = []
        RF_Recall = []
        SVM_Recall = []

        NB_Precision = []
        RF_Precision = []
        SVM_Precision = []

        if sys.platform == "darwin":
            path = '/Users/Lino/PycharmProjects/Preprocessing/SnapProg/' + str(window) + 'd_Train' + '_' +str(dataset)+ '.csv'
            testpath = '/Users/Lino/PycharmProjects/Preprocessing/SnapProg/' +str(window) + 'd_Test' + '_' +str(dataset)+ '.csv'
            #pathMV = '/Users/Lino/PycharmProjects/Preprocessing/PreProcessedFoldsFinal/' + str(
            #window) + 'd_FOLDS' + '/S' + str(seed) + '/' + str(window) + 'd_FOLDS_train_' + str(
            #fold) + '.csv'
            #testpathMV = '/Users/Lino/PycharmProjects/Preprocessing/PreProcessedFoldsFinal/' + str(
            #window) + 'd_FOLDS' + '/S' + str(seed) + '/' + str(window) + 'd_FOLDS_test_' + str(
            #fold) + '.csv'

        directory = '/Users/Lino/PycharmProjects/Classification/FS/'
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

        ClassIndex=dataTrain.attribute(dataTrain.class_index).values
        yIndex = ClassIndex.index('Y')


        # dataTrainMV = loader.load_file(pathMV)
        # dataTestMV = loader.load_file(testpathMV)
        # dataTrainMV.class_is_last()
        # dataTestMV.class_is_last()

        from weka.filters import Filter

        toBeRemoved = []
        for attribute in range(0, dataTrain.attributes().data.class_index):
            if dataTrain.attribute_stats(attribute).missing_count == dataTrain.attributes().data.num_instances and dataTest.attribute_stats(attribute).missing_count == dataTest.attributes().data.num_instances:
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

        #Remove.inputformat(dataTrainMV)
        #dataTrainMV = Remove.filter(dataTrainMV)
        #dataTestMV = Remove.filter(dataTestMV)

        import weka.core.classes as wcc


        if dataset == 'MVI':
            ReplaceMV = Filter(classname="weka.filters.unsupervised.attribute.ReplaceMissingValues")
            ReplaceMV.inputformat(dataTrain)
            dataTrain = ReplaceMV.filter(dataTrain)
            ReplaceMV.inputformat(dataTest)
            dataTest = ReplaceMV.filter(dataTest)

        # ReplaceMV.inputformat(dataTest)
        # dataTest=ReplaceMV.filter(dataTest)


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



        Perf.write(str(window)+ '&' + dataset + '&' + str(np.round(evaluationNB.area_under_roc(1) * 100,2))+ '&' + str(np.round(evaluationNB.precision(yIndex) * 100,2))+'&' + str(np.round(evaluationNB.recall(yIndex) * 100,2))+'\n')
        Scores.write(str(window)+ ',' + dataset + ',' + str(np.round(evaluationNB.area_under_roc(1) * 100,2))+'\n')
        #Precision.write(str(window)+ '&' + dataset + '&' + str(np.round(evaluationNB.precision(1) * 100,2))+ '&' + str(np.round(evaluationRF.precision(1) * 100,2))+'\n')


        #Recall.write(str(window)+ '&' + dataset + '&' + str(np.round(evaluationNB.recall(1) * 100,2))+ '&' + str(np.round(evaluationRF.recall(1) * 100,2))+'\n')
jvm.stop()
