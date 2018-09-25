
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from mlPipeline.AlgorithmsEvaluation import AlgoEval


def filename_to_path(filename, base_dir=None):
    """Return CSV file path given ticker symbol."""
    if base_dir is None:
        base_dir = os.environ.get("DATA", '../data/')
    return os.path.join(base_dir, "{}.csv".format(str(filename)))



df = pd.read_csv(filename_to_path('winequality-red'),parse_dates=True)
qualityMap = {3:1,4:1,5:1,6:2,7:2,8:2}
df['quality']  = df['quality'].replace(qualityMap)

#Heat Map
f, ax = plt.subplots(figsize=(10, 6))
corr = df.corr()
hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm",fmt='.2f',
                 linewidths=.05)

f.subplots_adjust(top=0.93)
t= f.suptitle('Wine Attributes Correlation Heatmap', fontsize=14)
plt.savefig("small/heatMap.png")

X = df.drop(['quality'], axis = 1)
y = df['quality']

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X = sc.fit_transform(X)


Y = pd.DataFrame(y)
algoEval = AlgoEval()
algoEval.evalAllValidationCurves(X_train=X_train,Y_train=y_train,folderName="small")
algoEval.evalLearningCurve(X_train=X_train,Y_train=y_train,folderName="small")
algoEval.evalFinal(X_train=X_train,Y_train=y_train,X_test=X_test,Y_test=y_test)
algoEval.evalSVMNonLinear(X_train=X_train,Y_train=y_train,X_test=X_test,Y_test=y_test,folderName="small")