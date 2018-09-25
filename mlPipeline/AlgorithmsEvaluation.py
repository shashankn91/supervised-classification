import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.learning_curve import validation_curve
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC


class AlgoEval:

    def plot_validation_curve(self,estimator, title, X, y, param_name,param_range=None, cv=None,
                            n_jobs=1):

        train_scores, test_scores = validation_curve(
            estimator, X, y, param_name=param_name, param_range=param_range,
            cv=cv, scoring="accuracy", n_jobs=n_jobs)

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.figure()
        plt.title(title)
        plt.xlabel(param_name)
        plt.ylabel("accuracy")
        param_range = [str(i) for i in param_range]

        plt.fill_between(param_range, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(param_range, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(param_range, train_scores_mean, 'o-', color="r",
                 label="Training score")
        plt.plot(param_range, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")

        plt.legend(loc="best")





    def plot_learning_curve(self,estimator, title, X, y, ylim=None, cv=None,
                            n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")

        plt.legend(loc="best")
        return plt



    def evalFinal(self,X_train,Y_train,X_test,Y_test):
        #ANN
        ann = MLPClassifier(activation='relu' ,solver='lbfgs', alpha=1e-5,
                       random_state=1,hidden_layer_sizes = (5,2))
        ann = ann.fit(X_train,Y_train)
        Y_ann_pred = ann.predict(X_test)
        print("ANN Metrics = " , precision_recall_fscore_support(Y_test,Y_ann_pred,average='weighted'),accuracy_score(Y_test,Y_ann_pred, normalize=True))



        linSVM = LinearSVC(intercept_scaling = 10000,tol=1e-6 , dual=False , C = 0.1)
        linSVM = linSVM.fit(X_train,Y_train)
        Y_linSVM_pred = linSVM.predict(X_test)
        print("linSVM Metrics = " , precision_recall_fscore_support(Y_test,Y_linSVM_pred,average='weighted'),accuracy_score(Y_test,Y_linSVM_pred, normalize=True))

        #kNN
        neigh = KNeighborsClassifier(n_neighbors = 12)
        neigh = neigh.fit(X_train,Y_train)

        Y_neigh_pred = neigh.predict(X_test)
        print("neigh Metrics = " , precision_recall_fscore_support(Y_test,Y_neigh_pred,average='weighted'),accuracy_score(Y_test,Y_neigh_pred, normalize=True))

        #AdaBoost
        adaClf = AdaBoostClassifier(n_estimators = 13,
                                    learning_rate=1,
                                    random_state=0)
        adaClf = adaClf.fit(X_train,Y_train)

        Y_adaClf_pred = adaClf.predict(X_test)
        print("adaClf Metrics = " , precision_recall_fscore_support(Y_test,Y_adaClf_pred,average='weighted'),accuracy_score(Y_test,Y_adaClf_pred, normalize=True))

        # Decision Tree
        dtClf = tree.DecisionTreeClassifier(max_depth =1)
        dtClf = dtClf.fit(X_train,Y_train)

        Y_dtClf_pred = dtClf.predict(X_test)
        print("dtClf Metrics = " , precision_recall_fscore_support(Y_test,Y_dtClf_pred,average='weighted'),accuracy_score(Y_test,Y_dtClf_pred, normalize=True))


    def evalLearningCurve(self,X_train,Y_train,folderName):
        ann = MLPClassifier(activation='relu' ,solver='lbfgs', alpha=1e-5,
                       random_state=1,hidden_layer_sizes = (5,2))

        self.plot_learning_curve(estimator=ann,title="ANN Learning Curve",X=X_train,y=Y_train,ylim=None,cv=4,n_jobs=1)
        plt.savefig(folderName + '/ANNLearningCurve.png')
        print("ANN Done")

        #SVM
        linSVM = LinearSVC(intercept_scaling = 10000,tol=1e-6 , dual=False , C = 0.1)
        self.plot_learning_curve(estimator=linSVM,title="SVM Learning Curve",X=X_train,y=Y_train,ylim=None,cv=4,n_jobs=1)
        plt.savefig(folderName + '/SVMLearningCurve.png')
        print("SVM Done")

        #kNN
        neigh = KNeighborsClassifier(n_neighbors = 12)
        self.plot_learning_curve(estimator=neigh,title="KNN Learning Curve",X=X_train,y=Y_train,ylim=None,cv=4,n_jobs=10)
        plt.savefig(folderName + '/KNNLearningCurve.png')

        #AdaBoost
        adaClf = AdaBoostClassifier(n_estimators = 13,
                                    learning_rate=1,
                                    random_state=0)
        self.plot_learning_curve(estimator=adaClf,title="AdaBoost Learning Curve",X=X_train,y=Y_train,ylim=None,cv=4,n_jobs=10)
        plt.savefig(folderName + '/AdaBoostLearningCurve.png')

        # Decision Tree
        dtClf = tree.DecisionTreeClassifier(max_depth =4)
        self.plot_learning_curve(estimator=dtClf,title="Decision Tree Learning Curve",X=X_train,y=Y_train,ylim=None,cv=4,n_jobs=10)
        plt.savefig(folderName + '/DTLearningCurve.png')


    def evalAllValidationCurves(self,X_train,Y_train,folderName):

        ann = MLPClassifier(activation='relu' ,solver='lbfgs', alpha=1e-5,
                       random_state=1)
        self.plot_validation_curve(ann,"ANN Validation Curve",X_train,Y_train,"hidden_layer_sizes",[ (3, 2),(4, 2),(5, 2),(6, 2),(7, 2),(8, 2),(16,2)],cv =4,n_jobs=1)
        plt.savefig(folderName + '/ANNValidationCurve.png')
        print("ANN Done")

        #SVM
        linSVM = LinearSVC(intercept_scaling = 10000,tol=1e-6 , dual=False)
        self.plot_validation_curve(linSVM,"SVM Validation Curve",X_train,Y_train,"C",[0.0001,0.001,0.01,0.1,1,2,4,16,32,64,256],cv =4,n_jobs=1)
        plt.savefig(folderName + '/SVMValidationCurve.png')
        print("SVM Done")

        #kNN
        neigh = KNeighborsClassifier()
        self.plot_validation_curve(neigh,"KNN Validation Curve",X_train,Y_train,"n_neighbors",[1,2,4,6,8,12,14,16,64],cv =4,n_jobs=10)
        plt.savefig(folderName + '/KNNValidationCurve.png')

        #AdaBoost
        adaClf = AdaBoostClassifier(
                                    learning_rate=1,
                                    random_state=0)

        self.plot_validation_curve(adaClf,"AdaBoost Validation Curve",X_train,Y_train,"n_estimators",range(1,30,1),cv =4,n_jobs=50)
        plt.savefig(folderName + '/AdaBoostValidationCurve.png')

        # Decision Tree
        dtClf = tree.DecisionTreeClassifier()
        self.plot_validation_curve(dtClf,"Decision Tree Validation Curve",X_train,Y_train,"max_depth",[1,2,4,6,8,12,14,16,32],cv =4,n_jobs=10)
        plt.savefig(folderName + '/DTValidationCurve.png')

    def evalSVMNonLinear(self,X_train,Y_train,X_test,Y_test,folderName):
        svm = SVC( tol=1e-6 )
        self.plot_learning_curve(estimator=svm,title="SVM Learning Curve",X=X_train,y=Y_train,ylim=None,cv=4,n_jobs=1)
        plt.savefig(folderName + '/NonLinearSVMLearningCurve.png')
        print("Non linear SVM Done")

        svm = svm.fit(X_train,Y_train)
        Y_svm_pred = svm.predict(X_test)
        print("SVM Metrics = " , precision_recall_fscore_support(Y_test,Y_svm_pred,average='weighted'),accuracy_score(Y_test,Y_svm_pred, normalize=True))







