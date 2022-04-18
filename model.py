# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

data = pd.read_csv('sbidatastores.csv')

df =pd.DataFrame(data,columns=['Open','Close'])

X = df.iloc[:, :-1].values
y = df.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)


from sklearn.svm import SVR
svrregressor = SVR(kernel = 'rbf')
svrregressor.fit(X_train, y_train)
svr=svrregressor.predict(X_test)

# Saving model to disk
#pickle.dump(svrregressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[509]]))
