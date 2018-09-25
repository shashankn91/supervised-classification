import pandas as pd
import os
from pandas import DatetimeIndex
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

from mlPipeline.LargeAlgoEval import LargeAlgoEval


def filename_to_path(filename, base_dir=None):
    """Return CSV file path given ticker symbol."""
    if base_dir is None:
        base_dir = os.environ.get("DATA", '../data/')
    return os.path.join(base_dir, "{}.csv".format(str(filename)))



df_temp = pd.read_csv(filename_to_path('KaggleV2-May-2016'),parse_dates=True)
week_key = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

df_temp['ScheduledDay'] = pd.to_datetime(df_temp['ScheduledDay']).dt.date.astype('datetime64[ns]')
df_temp['AppointmentDay'] = pd.to_datetime(df_temp['AppointmentDay']).dt.date.astype('datetime64[ns]')
df_temp['Waiting_Time_days'] = df_temp['AppointmentDay'] - df_temp['ScheduledDay']
df_temp['Waiting_Time_days'] = df_temp['Waiting_Time_days'].dt.days
df_temp['ScheduledDay_DOW'] = df_temp['ScheduledDay'].dt.weekday_name
df_temp['AppointmentDay_DOW'] = df_temp['AppointmentDay'].dt.weekday_name
df_temp['ScheduledDay_DOW'] = df_temp['ScheduledDay_DOW'].replace(week_key,range(7))
df_temp['AppointmentDay_DOW'] = df_temp['AppointmentDay_DOW'].replace(week_key,range(7))

df_temp['ScheduledDayMonth'] = DatetimeIndex(df_temp['ScheduledDay']).month
df_temp['ScheduledDayDate'] = DatetimeIndex(df_temp['ScheduledDay']).day

df_temp['AppointmentDayMonth'] = DatetimeIndex(df_temp['AppointmentDay']).month
df_temp['AppointmentDayDate'] = DatetimeIndex(df_temp['AppointmentDay']).day

#df_temp['AppointmentDay'] = np.where((df_temp['AppointmentDay'] - df_temp['ScheduledDay']).dt.days < 0, df_temp['ScheduledDay'], df_temp['AppointmentDay'])

# Get the Waiting Time in Days of the Patients.
df_temp = df_temp.rename(index =str , columns = {"No-show" : "No_show"})
df_temp = df_temp.drop(['PatientId','AppointmentID'],axis=1)


df_temp['Gender'] = df_temp['Gender'].replace({'M':1,'F':0})
uniqueNiebhouhoods = df_temp.Neighbourhood.unique()
uniqueNiebhouhoodsMap =  { uniqueNiebhouhoods.tolist()[i]:i for i in range(len(uniqueNiebhouhoods.tolist())) }
df_temp["Neighbourhood"] = df_temp["Neighbourhood"].replace(uniqueNiebhouhoodsMap)
df_temp['No_show'] = df_temp['No_show'].replace({'Yes' : 1 ,"No" : 0})
df =df_temp
# df = pd.concat([df_temp,pd.get_dummies(df_temp["Neighbourhood"])],axis=1)
# df = pd.concat([df,pd.get_dummies(df_temp["Gender"])],axis=1)


f, ax = plt.subplots(figsize=(10, 6))
corr = df_temp.corr()
hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm",fmt='.2f',
                 linewidths=.05)

f.subplots_adjust(top=0.93)
t= f.suptitle('Appointment Correlation Heatmap', fontsize=14)
plt.savefig("big/heatMap.png")

Y = df['No_show']
Y = pd.DataFrame(Y)

X =df.drop(['AppointmentDay','ScheduledDay'],axis=1)
X = X.drop(['No_show'],axis =1)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=None)




algoEval = LargeAlgoEval()
algoEval.evalAllValidationCurves(X_train=X_train,Y_train=Y_train.values.ravel(),folderName="big")
algoEval.evalLearningCurve(X_train=X_train,Y_train=Y_train.values.ravel(),folderName="big")
algoEval.evalFinal(X_train=X_train,Y_train=Y_train.values.ravel(),X_test=X_test,Y_test=Y_test.values.ravel())
algoEval.evalSVMNonLinear(X_train=X_train,Y_train=Y_train.values.ravel(),X_test=X_test,Y_test=Y_test.values.ravel(),folderName="big")








