import pandas as pd
import numpy as np
import sys
import weka.core.jvm as jvm
import os
from weka.core.classes import from_commandline

os.environ["JAVA_HOME"] = "C:\\Program Files\\Java\\jdk1.8.0_131"
jvm.start()


Window = np.array([90, 180, 365])
roc = []
spec = []
sens = []
prec = []

completeName = os.path.join('C:\\Users\\Lino','Performance.txt')
Perf = open(completeName,'w')
for window in range(0,3):
    roc = []
    for seed in range(1,6):
        if sys.platform == "darwin":
            path = '/Users/Lino/PycharmProjects/Preprocessing/PreProcessedFolds/' + str(Window[window]) + 'd_FOLDS/S' + str(
                seed)
        elif sys.platform == "win32":
            path = 'C:\\Users\\Lino\\PycharmProjects\\Preprocessing\\PreProcessedFolds\\' + str(
                Window[window]) + 'd_FOLDS\\S' + str(seed)
        sys.path.append(path)
        for fold in range(1,11):
            try:

                from weka.core.converters import Loader

                loader = Loader(classname="weka.core.converters.CSVLoader")
                dataTrain = loader.load_file(path + '/' + str(Window[window]) + 'd_FOLDS_train_' + str(fold) + '.csv')
                dataTest = loader.load_file(path + '/' + str(Window[window]) + 'd_FOLDS_test_' + str(fold) + '.csv')
                dataTrain.class_is_last()
                dataTest.class_is_last()

                from weka.attribute_selection import AttributeSelection, ASEvaluation,ASSearch
                search = ASSearch(classname="weka.attributeSelection.RerankingSearch")#,options=["-method", "2"])
                evaluator = ASEvaluation(classname='weka.attributeSelection.ClassifierAttributeEval', options=['-B','weka.classifiers.bayes.NaiveBayes'])

                Eval = AttributeSelection(classname='weka.attributeSelection.ClassifierAttributeEval', options=['-B','weka.classifiers.bayes.NaiveBayes','--',"-S 'weka.attributeSelection.RerankingSearch -method 2'"])


                from weka.filters import Filter

                NominalToBinary = Filter(classname="weka.filters.unsupervised.attribute.NominalToBinary", options=["-R", "5,7,8"])
                NumericToNominal = Filter(classname="weka.filters.unsupervised.attribute.NumericToNominal")
                ReplaceMV = Filter(classname="weka.filters.unsupervised.attribute.ReplaceMissingValues")
                ReplaceMV.inputformat(dataTrain)
                dataTrain = ReplaceMV.filter(dataTrain)
                ReplaceMV.inputformat(dataTest)
                dataTest=ReplaceMV.filter(dataTest)
                from weka.classifiers import Classifier
                #mapper = Classifier(classname="weka.classifiers.misc.InputMappedClassifier", options=["-W", "weka.classifiers.functions.SMO", "--", "-K","weka.classifiers.functions.supportVector.PolyKernel -E 2.0"])
                mapper = Classifier(classname="weka.classifiers.misc.InputMappedClassifier", options=["-W", "weka.classifiers.trees.RandomForest", "--", "-I","20"])
                mapper.build_classifier(dataTrain)

                # options = ["-K"]
                # cls = Classifier(classname="weka.classifiers.bayes.NaiveBayes")
                # cls.options=options
                # #print(cls.options)
                # cls.build_classifier(dataTrain)



                from weka.classifiers import Evaluation
                #print("Evaluating NB classifier")
                evaluation = Evaluation(dataTrain)
                evl = evaluation.test_model(mapper, dataTest)
                print('Window' + str(Window[window]) + '_S' + str(seed) +'_Fold' +str(fold)+': Performance')
                #print(evaluation.summary())
                #print(evaluation.class_details())
                #print(evaluation.matrix())
                #print(evaluation.summary())
                #print(evaluation.class_details())
                #print(evaluation.matrix())
                roc.append(evaluation.area_under_roc(1))
                sens.append(evaluation.true_positive_rate(1))
                spec.append(evaluation.true_negative_rate(1))
                if fold == 10 and seed == 5:
                    print('Window' + str(Window[window]) + '_S' + str(seed) + '_Fold' + str(fold) + ': Performance')
                    print('AUC: ' + str(np.mean(roc)))
                    print('Sens: ' + str(np.mean(sens)))
                    print('Spec:' + str(np.mean(spec)))
                    Perf.write('Window' + str(Window[window]) + ': Performance\n\n')
                    Perf.write('AUC: ' + str(np.mean(roc)) + '\n')
                    Perf.write('Sens: ' + str(np.mean(sens)) + '\n')
                    Perf.write('Spec:' + str(np.mean(spec)) + '\n')

            except:
                continue