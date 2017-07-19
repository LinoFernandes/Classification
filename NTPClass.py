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
directory = '/Users/Lino/PycharmProjects/Classification/NTPClass/'
if not os.path.exists(directory):
    os.makedirs(directory)
Perf = open(directory + 'NTPClassSMOTE_50.txt', 'a')
Perf.write('k,#TP,NB,NB_K,SVM_1,SVM_2,SVM_3,RF_5,RF_10,RF_15,RF_20\n')
for window in Window:
    for ntp in range(1, 7):
        roc_NB = []
        roc_NB_K = []
        roc_SVM_1 = []
        roc_SVM_2 = []
        roc_SVM_3 = []
        roc_RF_5 = []
        roc_RF_10 = []
        roc_RF_15 = []
        roc_RF_20 = []
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
                for fold in range(1,11):

                    from weka.filters import Filter
                    StratifiedCV = Filter(classname="weka.filters.supervised.instance.StratifiedRemoveFolds", options=['-S', '42', '-N' ,'10', '-F', str(fold)])
                    StratifiedCV.inputformat(data)
                    dataTest = StratifiedCV.filter(data)

                    StratifiedCV = Filter(classname="weka.filters.supervised.instance.StratifiedRemoveFolds",
                                          options=['-S', '42', '-V' ,'-N', '10', '-F', str(fold)])
                    StratifiedCV.inputformat(data)
                    dataTrain = StratifiedCV.filter(data)


                    # for attr in range(0,len(dataTrain.attributes())):
                    #     MV.append((dataTrain.attribute_stats(attr).missing_count/dataTrain.attribute_stats(attr).total_count)*100)


                    ReplaceMV = Filter(classname="weka.filters.unsupervised.attribute.ReplaceMissingValues")
                    ReplaceMV.inputformat(dataTrain)
                    dataTrain = ReplaceMV.filter(dataTrain)
                    ReplaceMV.inputformat(dataTest)
                    dataTest = ReplaceMV.filter(dataTest)
                    # ReplaceMV.inputformat(dataTest)
                    # dataTest=ReplaceMV.filter(dataTest)

                    SMOTE = Filter(classname="weka.filters.supervised.instance.SMOTE", options=['-P','50.0'])
                    SMOTE.inputformat(dataTrain)
                    dataTrain = SMOTE.filter(dataTrain)

                    dataTrain.class_is_last()
                    dataTest.class_is_last()

                    from weka.classifiers import Evaluation
                    from weka.core.classes import Random
                    from weka.classifiers import Classifier
                    if classifier == 0:
                        for kernel in range(0,2):
                            if kernel == 0:
                                mapper = Classifier(classname="weka.classifiers.misc.InputMappedClassifier", options=["-W", "weka.classifiers.bayes.NaiveBayes"])
                                Class = 'NaiveBayes'
                                mapper.build_classifier(dataTrain)
                                evaluation = Evaluation(dataTrain)
                                evaluation.test_model(mapper,dataTest)
                                roc_NB.append(evaluation.area_under_roc(1)*100)
                            else:
                                mapper = Classifier(classname="weka.classifiers.misc.InputMappedClassifier",
                                                    options=["-W", "weka.classifiers.bayes.NaiveBayes", "--", "-K"])
                                Class = 'NaiveBayes'
                                mapper.build_classifier(dataTrain)
                                evaluation = Evaluation(dataTrain)
                                evaluation.test_model(mapper, dataTest)
                                roc_NB_K.append(evaluation.area_under_roc(1) * 100)
                    elif classifier == 1:
                        for degree in range(1,4):
                            mapper = Classifier(classname="weka.classifiers.misc.InputMappedClassifier", options=["-W", "weka.classifiers.functions.SMO", "--", "-K","weka.classifiers.functions.supportVector.PolyKernel -E " + str(degree)])
                            Class = 'SVM'
                            mapper.build_classifier(dataTrain)
                            evaluation = Evaluation(dataTrain)
                            evaluation.test_model(mapper, dataTest)
                            if degree == 1:
                                roc_SVM_1.append(evaluation.area_under_roc(1)*100)
                            elif degree == 2:
                                roc_SVM_2.append(evaluation.area_under_roc(1)*100)
                            else:
                                roc_SVM_3.append(evaluation.area_under_roc(1)*100)
                    else:
                        for numTrees in np.arange(5,25,5):
                            mapper = Classifier(classname="weka.classifiers.misc.InputMappedClassifier", options=["-W", "weka.classifiers.trees.RandomForest", "--", "-I", str(numTrees)])
                            Class = 'RF'
                            mapper.build_classifier(dataTrain)
                            evaluation = Evaluation(dataTrain)
                            evaluation.test_model(mapper, dataTest)
                            if numTrees == 5:
                                roc_RF_5.append(evaluation.area_under_roc(1)*100)
                            elif numTrees == 10:
                                roc_RF_10.append(evaluation.area_under_roc(1) * 100)
                            elif numTrees == 15:
                                roc_RF_15.append(evaluation.area_under_roc(1) * 100)
                            else:
                                roc_RF_20.append(evaluation.area_under_roc(1) * 100)
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
        Perf.write(str(window) + ',' + str(ntp) + ',' + str(round(np.mean(roc_NB),2)) + '%,' + str(round(np.mean(roc_NB_K),2)) + '%,' + str(round(np.mean(roc_SVM_1),2)) + '%,' + str(round(np.mean(roc_SVM_2),2)) + '%,' + str(round(np.mean(roc_SVM_3),2)) + '%,' + str(round(np.mean(roc_RF_5),2)) + '%,' + str(round(np.mean(roc_RF_10),2)) + '%,' + str(round(np.mean(roc_RF_15),2)) + '%,' + str(round(np.mean(roc_RF_20),2)) + '%\n')

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
print(MV)