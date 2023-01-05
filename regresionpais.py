

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('salarios.csv')
dataset.head(4)

dataset.shape

x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1:2].values
z = dataset.iloc[:,2:3].values

x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, z, test_size = 0.2, random_state = 0)
x_train

regression = LinearRegression()
regression.fit( y_train, x_train)

vision_train = plt
vision_train.scatter(z_train, y_train, color = 'blue')
vision_train.plot(z_train, regression.predict(y_train), color = 'black')
vision_train.title('Salario VS Pais (Entrenamiento)')
vision_train.xlabel('Pais')
vision_train.ylabel('Salario')
vision_train.show()

regression = LinearRegression()
regression.fit( y_test, x_test)

vision_train = plt
vision_train.scatter(z_train, y_train, color = 'blue')
vision_train.plot(z_train, regression.predict(y_train), color = 'black')
vision_train.title('salario de un progrmador (entrenamiento)')
vision_train.xlabel('Pais')
vision_train.ylabel('Salario')
vision_train.show()