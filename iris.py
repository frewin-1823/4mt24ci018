from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression

iris = load_iris()

X = iris.data[:, 0].reshape(-1, 1)
y = iris.data[:, 2]
model = LinearRegression()


model.fit(X, y)

value = float(input("Enter sepal length: "))

prediction = model.predict([[value]])

print("Predicted petal length:", prediction[0])