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
# Perf = open(directory + 'NTPClassSMOTE_FS_AUC.txt', 'a')
Recall = open(directory + 'NTPClassSMOTE_FS_Recall.txt', 'a')
Precision = open(directory + 'NTPClassSMOTE_FS_Precision.txt', 'a')
Scores = open(directory + 'NTPClassSMOTE_FS_AUC_Scores.txt', 'a')

for window in Window:
    for ntp in range(2, 6):
        # Perf.write('k,#TP,%SMOTE,NB,NB_K,SVM_1,SVM_2,SVM_3,RF_5,RF_10,RF_15,RF_20\n')
        # Recall.write('k,#TP,%SMOTE,NB,NB_K,SVM_1,SVM_2,SVM_3,RF_5,RF_10,RF_15,RF_20\n')
        # Precision.write('k,#TP,%SMOTE,NB,NB_K,SVM_1,SVM_2,SVM_3,RF_5,RF_10,RF_15,RF_20\n')
        for smote in [0]:  # , 50, 100, 150, 200]:
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
            for fold in range(1, 11):
                if window == 90 and fold == 10:
                    continue
                if sys.platform == "darwin":
                    path = '/Users/Lino/PycharmProjects/Preprocessing/NTP/New2/' + str(ntp) + 'TP'
                    lastpath = '/Users/Lino/PycharmProjects/Preprocessing/NTPtoLast/New2/' + str(ntp) + 'TP'
                    directory = '/Users/Lino/PycharmProjects/Classification/NTPClass/'
                    # elif sys.platform == "win32":
                    # path = 'C:\\Users\\Lino\\PycharmProjects\\Preprocessing\\NTP\\' + str(ntp) + 'TP'
                sys.path.append(path)
                # try:
                from weka.core.converters import Loader

                loader = Loader(classname="weka.core.converters.CSVLoader")
                data = loader.load_file(path + '/' + str(window) + 'd_FOLDS_train_' + str(fold) + '.csv')
                dataLast = loader.load_file(lastpath + '/' + str(window) + 'd_' + str(ntp-1) + 'to' + str(ntp-1) +'.csv')
                data.class_is_last()
                data.class_is_last()
                dataLast.class_is_last()
                from weka.filters import Filter

                toBeRemoved = []
                for attribute in range(0, data.attributes().data.class_index):
                    if data.attribute_stats(attribute).missing_count == data.attributes().data.num_instances:
                        sys.exit("Fold has full missing column")
                    if (data.attribute_stats(attribute).missing_count / data.attributes().data.num_instances) > 0.7:
                        toBeRemoved.append(str(attribute))

                Remove = Filter(classname="weka.filters.unsupervised.attribute.Remove",
                                options=['-R', ','.join(toBeRemoved)])
                Remove.inputformat(data)
                data = Remove.filter(data)

                toBeRemoved = []
                for attribute in range(0, dataLast.attributes().data.class_index):
                    if dataLast.attribute_stats(
                            attribute).missing_count == dataLast.attributes().data.num_instances:
                        sys.exit("Fold has full missing column")
                    if (dataLast.attribute_stats(attribute).missing_count / dataLast.attributes().data.num_instances) > 0.7:
                        toBeRemoved.append(str(attribute))

                Remove = Filter(classname="weka.filters.unsupervised.attribute.Remove",
                                options=['-R', ','.join(toBeRemoved)])
                # Remove.inputformat(dataLast)
                # dataLast = Remove.filter(dataLast)

                import weka.core.classes as wcc

                FS = Filter(classname="weka.filters.supervised.attribute.AttributeSelection",
                            options=['-E', 'weka.attributeSelection.CfsSubsetEval -P 1 -E 1', '-S',
                                     'weka.attributeSelection.RerankingSearch -method 2 -blockSize 20 -rankingMeasure 0 -search "weka.attributeSelection.GreedyStepwise -T -1.7976931348623157E308 -N 20 -num-slots 1"'])
                # FS.inputformat(data)
                # data = FS.filter(data)

                # FS.inputformat(dataLast)
                # dataLast = FS.filter(dataLast)

                # ReplaceMV = Filter(classname="weka.filters.unsupervised.attribute.ReplaceMissingValues")
                # ReplaceMV.inputformat(data)
                # data = ReplaceMV.filter(data)

                # ReplaceMV.inputformat(dataLast)
                # dataLast = ReplaceMV.filter(dataLast)

                data.class_is_last()
                # dataLast.class_is_last()
                from weka.core.converters import Saver

                saver = Saver(classname="weka.core.converters.ArffSaver")
                saver.save_file(data, '/Users/Lino/PycharmProjects/Classification/Snapshots/ARFF_PROG/Fast/Data_' + str(
                    window) + 'd_S' + str(seed) + '_' + str(fold) + 'FOLD' + '.arff')

jvm.stop()
print(MV)