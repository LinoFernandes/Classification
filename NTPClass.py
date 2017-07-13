import pandas as pd
import numpy as np
import sys
import weka.core.jvm as jvm
import os
import matplotlib.pyplot as plt

from weka.core.classes import from_commandline

os.environ["JAVA_HOME"] = "/Library/Java/JavaVirtualMachines/jdk1.8.0_131.jdk"
jvm.start()

Window = np.array([90, 180, 365])
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
directory = '/Users/Lino/PycharmProjects/Classification/NTPClass/'
if not os.path.exists(directory):
    os.makedirs(directory)
Perf = open(directory + 'NTPClass.txt', 'a')
Perf.write('k,#TP,NB,SVM,RF\n')
for window in Window:
    for ntp in range(1, 7):
        for classifier in range(0, 3):
            if sys.platform == "darwin":
                path = '/Users/Lino/PycharmProjects/Preprocessing/NTP/' + str(ntp) + 'TP'
                directory = '/Users/Lino/PycharmProjects/Classification/NTPClass/'
            elif sys.platform == "win32":
                path = 'C:\\Users\\Lino\\PycharmProjects\\Preprocessing\\NTP\\' + str(ntp) + 'TP'
            sys.path.append(path)
            try:
                from weka.core.converters import Loader

                loader = Loader(classname="weka.core.converters.CSVLoader")
                data = loader.load_file(path + '/' + str(window) + 'd_' + str(ntp) + '.csv')
                data.class_is_last()


                from weka.filters import Filter

                NominalToBinary = Filter(classname="weka.filters.unsupervised.attribute.NominalToBinary",
                                         options=["-R", "5,7,8"])
                NumericToNominal = Filter(classname="weka.filters.unsupervised.attribute.NumericToNominal")
                ReplaceMV = Filter(classname="weka.filters.unsupervised.attribute.ReplaceMissingValues")
                ReplaceMV.inputformat(data)
                data = ReplaceMV.filter(data)
                # ReplaceMV.inputformat(dataTest)
                # dataTest=ReplaceMV.filter(dataTest)
                from weka.classifiers import Evaluation
                from weka.core.classes import Random
                from weka.classifiers import Classifier
                if classifier == 0:
                    mapper = Classifier(classname="weka.classifiers.misc.InputMappedClassifier", options=["-W", "weka.classifiers.bayes.NaiveBayes", "--", "-K"])
                    Class = 'NaiveBayes'
                    evaluation = Evaluation(data)
                    evaluation.crossvalidate_model(mapper, data, 10, Random(42))
                    roc_NB = round(evaluation.area_under_roc(1)*100,2)
                elif classifier == 1:
                    mapper = Classifier(classname="weka.classifiers.misc.InputMappedClassifier", options=["-W", "weka.classifiers.functions.SMO", "--", "-K","weka.classifiers.functions.supportVector.PolyKernel -E 2.0"])
                    Class = 'SVM'
                    evaluation = Evaluation(data)
                    evaluation.crossvalidate_model(mapper, data, 10, Random(42))
                    roc_SVM = round(evaluation.area_under_roc(1)*100, 2)
                else:
                    mapper = Classifier(classname="weka.classifiers.misc.InputMappedClassifier", options=["-W", "weka.classifiers.trees.RandomForest", "--", "-I", "20"])
                    Class = 'RF'
                    evaluation = Evaluation(data)
                    evaluation.crossvalidate_model(mapper, data, 10, Random(42))
                    roc_RF = round(evaluation.area_under_roc(1)*100, 2)
                # mapper.build_classifier(data)
                # options = ["-K"]
                # cls = Classifier(classname="weka.classifiers.bayes.NaiveBayes")
                # cls.options=options
                # #print(cls.options)
                # cls.build_classifier(dataTrain)





                #print('Window' + str(window) + '_NTP' + str(ntp) + ': Performance')
                #print('AUC: ' + str(evaluation.area_under_roc(1)))
                #print('Sens: ' + str(evaluation.true_positive_rate(1)))
                #print('Spec:' + str(evaluation.true_negative_rate(1)))

            except:
                continue
        Perf.write(str(window) + ',' + str(ntp) + ',' + str(roc_NB) + '%,' + str(roc_SVM) + '%,' + str(roc_RF) + '%\n')

                # if window == 90:
                #     roc_90.append(100 * evaluation.area_under_roc(1))
                #     sens_90.append(100 * evaluation.true_positive_rate(1))
                #     spec_90.append(100 * evaluation.true_negative_rate(1))
                # elif window == 180:
                #     roc_180.append(100 * evaluation.area_under_roc(1))
                #     sens_180.append(100 * evaluation.true_positive_rate(1))
                #     spec_180.append(100 * evaluation.true_negative_rate(1))
                # else:
                #     roc_365.append(100 * evaluation.area_under_roc(1))
                #     sens_365.append(100 * evaluation.true_positive_rate(1))
                #     spec_365.append(100 * evaluation.true_negative_rate(1))



# # plot
#
# NTP = np.array(range(1, 9))
#
# plt.plot(NTP, roc_90, label=Class+'_90d')
# plt.plot(NTP, roc_180, label=Class+'_180')
# plt.plot(NTP, roc_365, label=Class+'_365')
# plt.xticks(NTP)
# plt.xlabel('#TP')
# plt.ylabel(Class+'-AUC')
# plt.grid()
# plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
#            ncol=2, mode="expand", borderaxespad=0., fontsize='11')
# plt.savefig(directory+Class+'_AUC.png')
# plt.close()

jvm.stop()