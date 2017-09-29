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
Perf = open(directory + 'AUC_Prog_Final.txt', 'a')
Recall = open(directory + 'Recall_Prog_Final.txt', 'a')
Precision = open(directory + 'Precision_Prog_Final.txt', 'a')

ScoresSlow = open(directory + 'SlowBounds_Snap_Final.csv', 'a')
ScoresNeutral = open(directory + 'NeutralBounds_Snap_Final.csv', 'a')
ScoresFast = open(directory + 'FastBounds_Snap_Final.csv', 'a')

for window in Window:

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
        for smote in [50]:
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
                        pathSlow = '/Users/Lino/PycharmProjects/Preprocessing/SnapProg/FOLDS_Prog/' + str(window) + 'd_FOLDS/Slow' +'/S' + str(seed) + '/' + str(window) + 'd_FOLDS_train_' + str(fold) +'.csv'
                        testpathSlow = '/Users/Lino/PycharmProjects/Preprocessing/SnapProg/FOLDS_Prog/' + str(window) + 'd_FOLDS/Slow' +'/S' + str(seed) + '/' + str(window) + 'd_FOLDS_test_' + str(fold) +'.csv'
                        pathNeutral = '/Users/Lino/PycharmProjects/Preprocessing/SnapProg/FOLDS_Prog/' + str(window) + 'd_FOLDS/Neutral' +'/S' + str(seed) + '/' + str(window) + 'd_FOLDS_train_' + str(fold) +'.csv'
                        testpathNeutral = '/Users/Lino/PycharmProjects/Preprocessing/SnapProg/FOLDS_Prog/' + str(window) + 'd_FOLDS/Neutral' +'/S' + str(seed) + '/' + str(window) + 'd_FOLDS_test_' + str(fold) +'.csv'
                        pathFast = '/Users/Lino/PycharmProjects/Preprocessing/SnapProg/FOLDS_Prog/' + str(
                            window) + 'd_FOLDS/Fast' + '/S' + str(seed) + '/' + str(window) + 'd_FOLDS_train_' + str(
                            fold) + '.csv'
                        testpathFast = '/Users/Lino/PycharmProjects/Preprocessing/SnapProg/FOLDS_Prog/' + str(
                            window) + 'd_FOLDS/Fast' + '/S' + str(seed) + '/' + str(window) + 'd_FOLDS_test_' + str(
                            fold) + '.csv'

                        directory = '/Users/Lino/PycharmProjects/Classification/NTPClass/'
                    #elif sys.platform == "win32":
                        #path = 'C:\\Users\\Lino\\PycharmProjects\\Preprocessing\\NTP\\' + str(ntp) + 'TP'
                    #sys.path.append(path)
                    # try:
                    from weka.core.converters import Loader

                    loader = Loader(classname="weka.core.converters.CSVLoader")
                    dataTrainSlow = loader.load_file(pathSlow)
                    dataTestSlow = loader.load_file(testpathSlow)

                    dataTrainFast = loader.load_file(pathFast)
                    dataTestFast = loader.load_file(testpathFast)

                    dataTrainNeutral = loader.load_file(pathNeutral)
                    dataTestNeutral = loader.load_file(testpathNeutral)

                    dataTrainSlow.class_is_last()
                    dataTestSlow.class_is_last()
                    dataTrainNeutral.class_is_last()
                    dataTestNeutral.class_is_last()
                    dataTrainFast.class_is_last()
                    dataTestFast.class_is_last()



                    ClassIndex = dataTrainSlow.attribute(dataTrainSlow.class_index).values
                    yIndexSlow = ClassIndex.index('Y')



                    ClassIndex = dataTrainSlow.attribute(dataTrainNeutral.class_index).values
                    yIndexNeutral = ClassIndex.index('Y')


                    ClassIndex = dataTrainFast.attribute(dataTrainSlow.class_index).values
                    yIndexFast = ClassIndex.index('Y')

                    from weka.filters import Filter

                    toBeRemoved = []
                    for attribute in range(0, dataTrainSlow.attributes().data.class_index):
                        if dataTrainSlow.attribute_stats(
                                attribute).missing_count == dataTrainSlow.attributes().data.num_instances and dataTestSlow.attribute_stats(attribute).missing_count == dataTestSlow.attributes().data.num_instances:
                            sys.exit("Fold has full missing column")
                        if (dataTrainSlow.attribute_stats(
                                attribute).missing_count / dataTrainSlow.attributes().data.num_instances) > 0.5 and (
                                    dataTestSlow.attribute_stats(
                                        attribute).missing_count / dataTestSlow.attributes().data.num_instances) > 0.5:
                            toBeRemoved.append(str(attribute))

                    Remove = Filter(classname="weka.filters.unsupervised.attribute.Remove",
                                    options=['-R', ','.join(toBeRemoved)])
                    Remove.inputformat(dataTrainSlow)
                    dataTrainSlow = Remove.filter(dataTrainSlow)
                    Remove.inputformat(dataTestSlow)
                    dataTestSlow = Remove.filter(dataTestSlow)

                    toBeRemoved = []
                    for attribute in range(0, dataTrainNeutral.attributes().data.class_index):
                        if dataTrainNeutral.attribute_stats(
                                attribute).missing_count == dataTrainNeutral.attributes().data.num_instances and dataTestNeutral.attribute_stats(
                            attribute).missing_count == dataTestNeutral.attributes().data.num_instances:
                            sys.exit("Fold has full missing column")
                        if (dataTrainNeutral.attribute_stats(
                                attribute).missing_count / dataTrainNeutral.attributes().data.num_instances) > 0.5 and (
                                    dataTestNeutral.attribute_stats(
                                        attribute).missing_count / dataTestNeutral.attributes().data.num_instances) > 0.5:
                            toBeRemoved.append(str(attribute))

                    Remove = Filter(classname="weka.filters.unsupervised.attribute.Remove",
                                    options=['-R', ','.join(toBeRemoved)])
                    Remove.inputformat(dataTrainNeutral)
                    dataTrainNeutral = Remove.filter(dataTrainNeutral)
                    Remove.inputformat(dataTestNeutral)
                    dataTestNeutral = Remove.filter(dataTestNeutral)

                    toBeRemoved = []
                    for attribute in range(0, dataTrainFast.attributes().data.class_index):
                        if dataTrainFast.attribute_stats(
                                attribute).missing_count == dataTrainFast.attributes().data.num_instances and dataTestFast.attribute_stats(
                            attribute).missing_count == dataTestFast.attributes().data.num_instances:
                            sys.exit("Fold has full missing column")
                        if (dataTrainFast.attribute_stats(
                                attribute).missing_count / dataTrainFast.attributes().data.num_instances) > 0.5 and (
                                    dataTestFast.attribute_stats(
                                        attribute).missing_count / dataTestFast.attributes().data.num_instances) > 0.5:
                            toBeRemoved.append(str(attribute))

                    Remove = Filter(classname="weka.filters.unsupervised.attribute.Remove",
                                    options=['-R', ','.join(toBeRemoved)])
                    Remove.inputformat(dataTrainFast)
                    dataTrainFast = Remove.filter(dataTrainFast)
                    Remove.inputformat(dataTestFast)
                    dataTestFast = Remove.filter(dataTestFast)


                    import weka.core.classes as wcc

                    FS = Filter(classname="weka.filters.supervised.attribute.AttributeSelection",
                                options=['-E', 'weka.attributeSelection.CfsSubsetEval -P 1 -E 1', '-S',
                                         "weka.attributeSelection.GreedyStepwise -T -1.7976931348623157E308 -N -1 -num-slots 1"])
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

                    # ReplaceMV = Filter(classname="weka.filters.unsupervised.attribute.ReplaceMissingValues")
                    #
                    # ReplaceMV.inputformat(dataTrainSlow)
                    # dataTrainSlow = ReplaceMV.filter(dataTrainSlow)
                    # ReplaceMV.inputformat(dataTestSlow)
                    # dataTestSlow = ReplaceMV.filter(dataTestSlow)
                    #
                    # ReplaceMV.inputformat(dataTrainNeutral)
                    # dataTrainNeutral = ReplaceMV.filter(dataTrainNeutral)
                    # ReplaceMV.inputformat(dataTestNeutral)
                    # dataTestNeutral = ReplaceMV.filter(dataTestNeutral)
                    #
                    # ReplaceMV.inputformat(dataTrainFast)
                    # dataTrainFast = ReplaceMV.filter(dataTrainFast)
                    # ReplaceMV.inputformat(dataTestFast)
                    # dataTestFast = ReplaceMV.filter(dataTestFast)

                    from weka.classifiers import Evaluation
                    from weka.core.classes import Random
                    from weka.classifiers import Classifier

                    if classifier == 0:
                        for kernel in range(0, 1):
                            if kernel == 0:
                                mapper = Classifier(classname="weka.classifiers.misc.InputMappedClassifier",
                                                    options=["-M", "-W", "weka.classifiers.bayes.NaiveBayes"])
                                Class = 'NaiveBayes'
                                mapper.build_classifier(dataTrainSlow)
                                evaluation = Evaluation(dataTrainSlow)
                                evaluation.test_model(mapper, dataTestSlow)

                                NB_AUC[seed-1,fold-1,0] = (evaluation.area_under_roc(1) * 100)
                                NB_Recall[seed-1,fold-1,0] =(evaluation.recall(yIndexSlow) * 100)
                                NB_Precision[seed-1,fold-1,0] =(evaluation.precision(yIndexSlow) * 100)

                                if window == 365:
                                    mapper = Classifier(classname="weka.classifiers.misc.InputMappedClassifier",
                                                        options=["-M", "-W", "weka.classifiers.bayes.NaiveBayes",'--','-K'])
                                else:
                                    mapper = Classifier(classname="weka.classifiers.misc.InputMappedClassifier",
                                                        options=["-M", "-W", "weka.classifiers.bayes.NaiveBayes"])

                                mapper.build_classifier(dataTrainNeutral)
                                evaluation = Evaluation(dataTrainNeutral)
                                evaluation.test_model(mapper, dataTestNeutral)

                                NB_AUC[seed - 1, fold - 1, 1] = (evaluation.area_under_roc(1) * 100)
                                NB_Recall[seed - 1, fold - 1, 1] = (evaluation.recall(yIndexNeutral) * 100)
                                NB_Precision[seed - 1, fold - 1, 1] = (evaluation.precision(yIndexNeutral) * 100)

                                mapper = Classifier(classname="weka.classifiers.misc.InputMappedClassifier",
                                                    options=["-M", "-W", "weka.classifiers.bayes.NaiveBayes"])

                                mapper.build_classifier(dataTrainFast)
                                evaluation = Evaluation(dataTrainFast)
                                evaluation.test_model(mapper, dataTestFast)

                                NB_AUC[seed - 1, fold - 1, 2] = (evaluation.area_under_roc(1) * 100)
                                NB_Recall[seed - 1, fold - 1, 2] = (evaluation.recall(yIndexFast) * 100)
                                NB_Precision[seed - 1, fold - 1, 2] = (evaluation.precision(yIndexFast) * 100)
                    elif classifier == 1:
                        for degree in [1]:
                            if window == 180:
                                SMOTE = Filter(classname="weka.filters.supervised.instance.SMOTE",
                                               options=['-P 0'])
                                SMOTE.inputformat(dataTrainSlow)
                                dataTrainSlow = SMOTE.filter(dataTrainSlow)
                                SMOTE.inputformat(dataTrainNeutral)
                                dataTrainNeutral = SMOTE.filter(dataTrainNeutral)
                                SMOTE.inputformat(dataTrainFast)
                                dataTrainFast = SMOTE.filter(dataTrainFast)
                            else:
                                SMOTE = Filter(classname="weka.filters.supervised.instance.SMOTE",
                                               options=['-P 0'])
                                SMOTE.inputformat(dataTrainSlow)
                                dataTrainSlow = SMOTE.filter(dataTrainSlow)
                                SMOTE.inputformat(dataTrainNeutral)
                                dataTrainNeutral = SMOTE.filter(dataTrainNeutral)
                                SMOTE.inputformat(dataTrainFast)
                                dataTrainFast = SMOTE.filter(dataTrainFast)


                            mapper = Classifier(classname="weka.classifiers.misc.InputMappedClassifier",
                                                options=["-M", "-W", "weka.classifiers.functions.SMO", "--", "-K",
                                                         "weka.classifiers.functions.supportVector.PolyKernel -E 3"])
                            Class = 'SVM'
                            mapper.build_classifier(dataTrainSlow)
                            evaluation = Evaluation(dataTrainSlow)
                            evaluation.test_model(mapper, dataTestSlow)

                            SVM_AUC[seed-1,fold-1,0] = (evaluation.area_under_roc(1) * 100)
                            SVM_Recall[seed - 1, fold - 1, 0] = (evaluation.recall(yIndexSlow) * 100)
                            SVM_Precision[seed - 1, fold - 1, 0] = (evaluation.precision(yIndexSlow) * 100)

                            mapper = Classifier(classname="weka.classifiers.misc.InputMappedClassifier",
                                                options=["-M", "-W", "weka.classifiers.functions.SMO", "--", "-K",
                                                         "weka.classifiers.functions.supportVector.PolyKernel -E 1"])

                            mapper.build_classifier(dataTrainNeutral)
                            evaluation = Evaluation(dataTrainNeutral)
                            evaluation.test_model(mapper, dataTestNeutral)

                            SVM_AUC[seed - 1, fold - 1, 1] = (evaluation.area_under_roc(1) * 100)
                            SVM_Recall[seed - 1, fold - 1, 1] = (evaluation.recall(yIndexNeutral) * 100)
                            SVM_Precision[seed - 1, fold - 1, 1] = (evaluation.precision(yIndexNeutral) * 100)

                            if window == 90:
                                mapper = Classifier(classname="weka.classifiers.misc.InputMappedClassifier",
                                                options=["-M", "-W", "weka.classifiers.functions.SMO", "--", "-K",
                                                         "weka.classifiers.functions.supportVector.PolyKernel -E 2"])
                            else:
                                mapper = Classifier(classname="weka.classifiers.misc.InputMappedClassifier",
                                                    options=["-M", "-W", "weka.classifiers.functions.SMO", "--", "-K",
                                                             "weka.classifiers.functions.supportVector.PolyKernel -E 1"])

                            mapper.build_classifier(dataTrainFast)
                            evaluation = Evaluation(dataTrainFast)
                            evaluation.test_model(mapper, dataTestFast)

                            SVM_AUC[seed - 1, fold - 1, 2] = (evaluation.area_under_roc(1) * 100)
                            SVM_Recall[seed - 1, fold - 1, 2] = (evaluation.recall(yIndexFast) * 100)
                            SVM_Precision[seed - 1, fold - 1, 2] = (evaluation.precision(yIndexFast) * 100)

                    else:
                        for numTrees in [20]:
                            SMOTE = Filter(classname="weka.filters.supervised.instance.SMOTE",
                                           options=['-P 0'])
                            SMOTE.inputformat(dataTrainSlow)
                            dataTrainSlow = SMOTE.filter(dataTrainSlow)
                            SMOTE.inputformat(dataTrainNeutral)
                            dataTrainNeutral = SMOTE.filter(dataTrainNeutral)
                            SMOTE.inputformat(dataTrainFast)
                            dataTrainFast = SMOTE.filter(dataTrainFast)

                            mapper = Classifier(classname="weka.classifiers.misc.InputMappedClassifier",
                                                options=["-M", "-W", "weka.classifiers.trees.RandomForest", "--", "-I",
                                                         str(numTrees)])
                            Class = 'RF'

                            mapper.build_classifier(dataTrainSlow)
                            evaluation = Evaluation(dataTrainSlow)
                            evaluation.test_model(mapper, dataTestSlow)

                            RF_AUC[seed - 1, fold - 1, 0] = (evaluation.area_under_roc(1) * 100)
                            RF_Recall[seed - 1, fold - 1, 0] = (evaluation.recall(yIndexSlow) * 100)
                            RF_Precision[seed - 1, fold - 1, 0] = (evaluation.precision(yIndexSlow) * 100)

                            mapper.build_classifier(dataTrainNeutral)
                            evaluation = Evaluation(dataTrainNeutral)
                            evaluation.test_model(mapper, dataTestNeutral)

                            RF_AUC[seed - 1, fold - 1, 1] = (evaluation.area_under_roc(1) * 100)
                            RF_Recall[seed - 1, fold - 1, 1] = (evaluation.recall(yIndexNeutral) * 100)
                            RF_Precision[seed - 1, fold - 1, 1] = (evaluation.precision(yIndexNeutral) * 100)

                            mapper.build_classifier(dataTrainFast)
                            evaluation = Evaluation(dataTrainFast)
                            evaluation.test_model(mapper, dataTestFast)

                            RF_AUC[seed - 1, fold - 1, 2] = (evaluation.area_under_roc(1) * 100)
                            RF_Recall[seed - 1, fold - 1, 2] = (evaluation.recall(yIndexFast) * 100)
                            RF_Precision[seed - 1, fold - 1, 2] = (evaluation.precision(yIndexFast) * 100)

    ScoresSlow.write('Ori,' + str(
        round(np.mean(np.mean(np.mean(NB_AUC, axis=0), axis=0)[0]) + np.std(np.mean(NB_AUC, axis=0), axis=0)[0],
              2)) + ',' + str(
        round(np.mean(np.mean(np.mean(NB_AUC, axis=0), axis=0)[0]) - np.std(np.mean(NB_AUC, axis=0), axis=0)[0],
              2)) + ',' + str(
        round(np.mean(np.mean(np.mean(RF_AUC, axis=0), axis=0)[0]) + np.std(np.mean(RF_AUC, axis=0), axis=0)[0],
              2)) + ',' + str(
        round(np.mean(np.mean(np.mean(RF_AUC, axis=0), axis=0)[0]) - np.std(np.mean(RF_AUC, axis=0), axis=0)[0],
              2)) + '\n')
    ScoresFast.write('MVI,' + str(
        round(np.mean(np.mean(np.mean(NB_AUC, axis=0), axis=0)[2]) + np.std(np.mean(NB_AUC, axis=0), axis=0)[2],
              2)) + ',' + str(
        round(np.mean(np.mean(np.mean(NB_AUC, axis=0), axis=0)[2]) - np.std(np.mean(NB_AUC, axis=0), axis=0)[2],
              2)) + ',' + str(
        round(np.mean(np.mean(np.mean(RF_AUC, axis=0), axis=0)[2]) + np.std(np.mean(RF_AUC, axis=0), axis=0)[2],
              2)) + ',' + str(
        round(np.mean(np.mean(np.mean(RF_AUC, axis=0), axis=0)[2]) - np.std(np.mean(RF_AUC, axis=0), axis=0)[2],
              2)) + '\n')
    ScoresNeutral.write('MVI,' + str(
        round(np.mean(np.mean(np.mean(NB_AUC, axis=0), axis=0)[1]) + np.std(np.mean(NB_AUC, axis=0), axis=0)[1],
              2)) + ',' + str(
        round(np.mean(np.mean(np.mean(NB_AUC, axis=0), axis=0)[1]) - np.std(np.mean(NB_AUC, axis=0), axis=0)[1],
              2)) + ',' + str(
        round(np.mean(np.mean(np.mean(RF_AUC, axis=0), axis=0)[1]) + np.std(np.mean(RF_AUC, axis=0), axis=0)[1],
              2)) + ',' + str(
        round(np.mean(np.mean(np.mean(RF_AUC, axis=0), axis=0)[1]) - np.std(np.mean(RF_AUC, axis=0), axis=0)[1],
              2)) + '\n')


    Perf.write('\multirow{4}{*}{' + str(window) + 'd}' + ' & ' + str('Slow') + ' & ' + str(
        np.round(np.mean(np.mean(NB_AUC, axis=0), axis=0)[0], 2)) + ' $\pm$ ' + str(
        np.round(np.std(np.mean(NB_AUC, axis=0), axis=0)[0], 2)) + ' & ' + str(
        np.round(np.mean(np.mean(SVM_AUC, axis=0), axis=0)[0], 2)) + ' $\pm$ ' + str(
        np.round(np.std(np.mean(SVM_AUC, axis=0), axis=0)[0], 2)) + ' & ' + str(
        np.round(np.mean(np.mean(RF_AUC, axis=0), axis=0)[0], 2)) + ' $\pm$ ' + str(
        np.round(np.std(np.mean(RF_AUC, axis=0), axis=0)[0], 2)) + '\\\\\n')
    Perf.write(
        ' & ' + str('Neutral') + ' & ' + str(np.round(np.mean(np.mean(NB_AUC, axis=0), axis=0)[1], 2)) + ' $\pm$ ' + str(
            np.round(np.std(np.mean(NB_AUC, axis=0), axis=0)[1], 2)) + ' & ' + str(
            np.round(np.mean(np.mean(SVM_AUC, axis=0), axis=0)[1], 2)) + ' $\pm$ ' + str(
            np.round(np.std(np.mean(SVM_AUC, axis=0), axis=0)[1], 2)) + ' & ' + str(
            np.round(np.mean(np.mean(RF_AUC, axis=0), axis=0)[1], 2)) + ' $\pm$ ' + str(
            np.round(np.std(np.mean(RF_AUC, axis=0), axis=0)[1], 2)) + '\\\\\n')
    Perf.write(
        ' & ' + str('Fast') + ' & ' + str(np.round(np.mean(np.mean(NB_AUC, axis=0), axis=0)[2], 2)) + ' $\pm$ ' + str(
            np.round(np.std(np.mean(NB_AUC, axis=0), axis=0)[2], 2)) + ' & ' + str(
            np.round(np.mean(np.mean(SVM_AUC, axis=0), axis=0)[2], 2)) + ' $\pm$ ' + str(
            np.round(np.std(np.mean(SVM_AUC, axis=0), axis=0)[2], 2)) + ' & ' + str(
            np.round(np.mean(np.mean(RF_AUC, axis=0), axis=0)[2], 2)) + ' $\pm$ ' + str(
            np.round(np.std(np.mean(RF_AUC, axis=0), axis=0)[2], 2)) + '\\\\\n')

    Precision.write('\multirow{4}{*}{' + str(window) + 'd}' + ' & ' + str('Slow') + ' & ' + str(
        np.round(np.mean(np.mean(NB_Precision, axis=0), axis=0)[0], 2)) + ' $\pm$ ' + str(
        np.round(np.std(np.mean(NB_Precision, axis=0), axis=0)[0], 2)) + ' & ' + str(
        np.round(np.mean(np.mean(SVM_Precision, axis=0), axis=0)[0], 2)) + ' $\pm$ ' + str(
        np.round(np.std(np.mean(SVM_Precision, axis=0), axis=0)[0], 2)) + ' & ' + str(
        np.round(np.mean(np.mean(RF_Precision, axis=0), axis=0)[0], 2)) + ' $\pm$ ' + str(
        np.round(np.std(np.mean(RF_Precision, axis=0), axis=0)[0], 2)) + '\\\\\n')
    Precision.write(' & ' + str('Neutral') + ' & ' + str(
        np.round(np.mean(np.mean(NB_Precision, axis=0), axis=0)[1], 2)) + ' $\pm$ ' + str(
        np.round(np.std(np.mean(NB_Precision, axis=0), axis=0)[1], 2)) + ' & ' + str(
        np.round(np.mean(np.mean(SVM_Precision, axis=0), axis=0)[1], 2)) + ' $\pm$ ' + str(
        np.round(np.std(np.mean(SVM_Precision, axis=0), axis=0)[1], 2)) + ' & ' + str(
        np.round(np.mean(np.mean(RF_Precision, axis=0), axis=0)[1], 2)) + ' $\pm$ ' + str(
        np.round(np.std(np.mean(RF_Precision, axis=0), axis=0)[1], 2)) + '\\\\\n')
    Precision.write(' & ' + str('Fast') + ' & ' + str(
        np.round(np.mean(np.mean(NB_Precision, axis=0), axis=0)[2], 2)) + ' $\pm$ ' + str(
        np.round(np.std(np.mean(NB_Precision, axis=0), axis=0)[2], 2)) + ' & ' + str(
        np.round(np.mean(np.mean(SVM_Precision, axis=0), axis=0)[2], 2)) + ' $\pm$ ' + str(
        np.round(np.std(np.mean(SVM_Precision, axis=0), axis=0)[2], 2)) + ' & ' + str(
        np.round(np.mean(np.mean(RF_Precision, axis=0), axis=0)[2], 2)) + ' $\pm$ ' + str(
        np.round(np.std(np.mean(RF_Precision, axis=0), axis=0)[2], 2)) + '\\\\\n')

    Recall.write('\multirow{4}{*}{' + str(window) + 'd}' + ' & ' + str('Slow') + ' & ' + str(
        np.round(np.mean(np.mean(NB_Recall, axis=0), axis=0)[0], 2)) + ' $\pm$ ' + str(
        np.round(np.std(np.mean(NB_Recall, axis=0), axis=0)[0], 2)) + ' & ' + str(
        np.round(np.mean(np.mean(SVM_Recall, axis=0), axis=0)[0], 2)) + ' $\pm$ ' + str(
        np.round(np.std(np.mean(SVM_Recall, axis=0), axis=0)[0], 2)) + ' & ' + str(
        np.round(np.mean(np.mean(RF_Recall, axis=0), axis=0)[0], 2)) + ' $\pm$ ' + str(
        np.round(np.std(np.mean(RF_Recall, axis=0), axis=0)[0], 2)) + '\\\\\n')
    Recall.write(
        ' & ' + str('Neutral') + ' & ' + str(np.round(np.mean(np.mean(NB_Recall, axis=0), axis=0)[1], 2)) + ' $\pm$ ' + str(
            np.round(np.std(np.mean(NB_Recall, axis=0), axis=0)[1], 2)) + ' & ' + str(
            np.round(np.mean(np.mean(SVM_Recall, axis=0), axis=0)[1], 2)) + ' $\pm$ ' + str(
            np.round(np.std(np.mean(SVM_Recall, axis=0), axis=0)[1], 2)) + ' & ' + str(
            np.round(np.mean(np.mean(RF_Recall, axis=0), axis=0)[1], 2)) + ' $\pm$ ' + str(
            np.round(np.std(np.mean(RF_Recall, axis=0), axis=0)[1], 2)) + '\\\\\n')
    Recall.write(
        ' & ' + str('Fast') + ' & ' + str(np.round(np.mean(np.mean(NB_Recall, axis=0), axis=0)[2], 2)) + ' $\pm$ ' + str(
            np.round(np.std(np.mean(NB_Recall, axis=0), axis=0)[2], 2)) + ' & ' + str(
            np.round(np.mean(np.mean(SVM_Recall, axis=0), axis=0)[2], 2)) + ' $\pm$ ' + str(
            np.round(np.std(np.mean(SVM_Recall, axis=0), axis=0)[2], 2)) + ' & ' + str(
            np.round(np.mean(np.mean(RF_Recall, axis=0), axis=0)[2], 2)) + ' $\pm$ ' + str(
            np.round(np.std(np.mean(RF_Recall, axis=0), axis=0)[2], 2)) + '\\\\\n')
jvm.stop()
