import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

path = os.path.dirname(os.getcwd())

# read data file
data_dict = pickle.load(open(path + '\\data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# split dataset
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# train model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# test model
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly !'.format(score * 100))

# create model file
file = open(path + '\\model.p', 'wb')
pickle.dump({'model': model}, file)
file.close()