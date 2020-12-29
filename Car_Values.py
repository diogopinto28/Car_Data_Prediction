import pandas as pd
import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv('car.data.')



le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data['buying']))
maint = le.fit_transform(list(data['maint']))
door = le.fit_transform(list(data['door']))
persons = le.fit_transform(list(data['persons']))
lug_boot = le.fit_transform(list(data['lug_boot']))
safety = le.fit_transform(list(data['safety']))
cls = le.fit_transform(list(data['class']))

predict = 'class'

x = list(zip(buying,maint, door, persons, lug_boot, safety))
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)


# for loop to determine wich n_neighbor gets the best accuracy for this model
best = 0
for i in range(1,20):
    model = KNeighborsClassifier(n_neighbors=i)

    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)
    if acc > best:
        best = acc
        best_acc = i


model = KNeighborsClassifier(n_neighbors=best_acc)
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print('The accuracy was', acc)
print('The best accuracy was achieved with', best_acc, 'n_neighbors')

predicted = model.predict(x_test)

names = ['unacc', 'acc', 'good', 'vgood']

for x in range (len(x_test)):
    print('Predicted:', names[predicted[x]], 'vs Real value:', names[y_test[x]], 'Data: ', x_test[x])
