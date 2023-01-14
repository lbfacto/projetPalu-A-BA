# -*- coding: utf-8 -*-
"""palusvm.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1GNwK7pv0SMinAw8JsGE1QzzSan3EkV1z
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.model_selection import train_test_split

df=pd.read_excel("/content/drive/MyDrive/palu/palucleaner.xlsx")

df=df.drop(['ID','Sexe', 'Type_Hb', 'Gpe_ABO', 'Rhesus'], axis=1)

df

conditionlist=[(df['ratio']>2),
              (df['ratio']<=2)]
diagnostic = ['0', '1']
df['Resultat'] = np.select(conditionlist, diagnostic)

features = ['Age',	'ratio',	'G6PD',	'EP_6M_AVT',	'AcPf_6M_AVT',	'EP_1AN_AVT'	'AcPf_1AN_AVT'	'EP_6M_APR'	'AcPf_6M_APR'	'EP_1AN_APR'	'AcPf_1AN_APR']
target = ['Resultat']

for attr in ['mean', 'ste', 'largest']:
  for feature in features:
    target.append(feature + "_" + attr)

df['Resultat'] = df['Resultat'].astype(str).astype(int)

df.info()

df.shape

df.describe()



df['Resultat'].value_counts()



"""**0** -----> **Repond a l'antigene Positif au palu**

1 -----> **Ne repond pas a l'antigene Negatif au palu**
"""

X = df.drop(columns='Resultat', axis=1)
Y = df['Resultat']

print(X)

print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

classifier = svm.SVC(kernel='linear')

classifier.fit(X_train, Y_train)

# accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy score of the training data : ', training_data_accuracy)

# accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy score of the test data : ', test_data_accuracy)

input_data = (23,1,8.5,0,0,4,1,3,0,4,0)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = classifier.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('Repond a l_antigen paludisme positif')
else:
  print('Non pas de reponse paludisme negatif')



import pickle

filename = 'trained_model.pkl'
pickle.dump(classifier, open(filename, 'wb'))

input_data = (23,1,8.5,0,0,4,1,3,0,4,0)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = classifier.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('Repond a l_antigen paludisme positif')
else:
  print('Non pas de reponse antigene  negatif au paludisme ')