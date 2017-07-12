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
roc_90 = []
spec_90 = []
sens_90 = []

roc_180 = []
spec_180 = []
sens_180 = []

roc_365 = []
spec_365 = []
sens_365 = []

if sys.platform == "darwin":
    completeName = os.path.join('/Users/Lino/Desktop', 'PerformanceNB.txt')
elif sys.platform == "win32":
    completeName = os.path.join('C:\\Users\\Lino\\Desktop', 'PerformanceNB.txt')
Perf = open(completeName, 'w')
for ntp in range(1, 9):
    if sys.platform == "darwin":
        path = '/Users/Lino/PycharmProjects/Preprocessing/NTP/' + str(ntp) + 'TP'
    elif sys.platform == "win32":
        path = 'C:\\Users\\Lino\\PycharmProjects\\Preprocessing\\NTP\\' + str(ntp) + 'TP'
    sys.path.append(path)
    for window in Window:
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

            from weka.classifiers import Classifier

            # mapper = Classifier(classname="weka.classifiers.misc.InputMappedClassifier", options=["-W", "weka.classifiers.bayes.NaiveBayes", "--", "-K"])
            # mapper = Classifier(classname="weka.classifiers.misc.InputMappedClassifier", options=["-W", "weka.classifiers.functions.SMO", "--", "-K","weka.classifiers.functions.supportVector.PolyKernel -E 2.0"])
            mapper = Classifier(classname="weka.classifiers.misc.InputMappedClassifier",
                                options=["-W", "weka.classifiers.trees.RandomForest", "--", "-I", "20"])
            # mapper.build_classifier(data)
            # options = ["-K"]
            # cls = Classifier(classname="weka.classifiers.bayes.NaiveBayes")
            # cls.options=options
            # #print(cls.options)
            # cls.build_classifier(dataTrain)



            from weka.classifiers import Evaluation
            from weka.core.classes import Random

            # print("Evaluating NB classifier")
            evaluation = Evaluation(data)
            evaluation.crossvalidate_model(mapper, data, 10, Random(42))
            print('Window' + str(window) + '_NTP' + str(ntp) + ': Performance')
            print('AUC: ' + str(evaluation.area_under_roc(1)))
            print('Sens: ' + str(evaluation.true_positive_rate(1)))
            print('Spec:' + str(evaluation.true_negative_rate(1)))

            Perf.write('Window' + str(window) + '_NTP' + str(ntp) + ': Performance\n\n')
            Perf.write('AUC: ' + str(evaluation.area_under_roc(1)) + '\n')
            Perf.write('Sens: ' + str(evaluation.true_positive_rate(1)) + '\n')
            Perf.write('Spec:' + str(evaluation.true_negative_rate(1)) + '\n')

            if window == 90:
                roc_90.append(100 * evaluation.area_under_roc(1))
                sens_90.append(100 * evaluation.true_positive_rate(1))
                spec_90.append(100 * evaluation.true_negative_rate(1))
            elif window == 180:
                roc_180.append(100 * evaluation.area_under_roc(1))
                sens_180.append(100 * evaluation.true_positive_rate(1))
                spec_180.append(100 * evaluation.true_negative_rate(1))
            else:
                roc_365.append(100 * evaluation.area_under_roc(1))
                sens_365.append(100 * evaluation.true_positive_rate(1))
                spec_365.append(100 * evaluation.true_negative_rate(1))


        except:
            continue
jvm.stop()

# plot

NTP = np.array(range(1, 9))

plt.plot(NTP, roc_90, label='RF_90d')
plt.plot(NTP, roc_180, label='RF_180')
plt.plot(NTP, roc_365, label='RF_365')
plt.xticks(NTP)
plt.xlabel('NTPs')
plt.ylabel('RandomForest-AUC')
plt.grid()
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0., fontsize='11')
plt.savefig('RandomForestAUC.png')