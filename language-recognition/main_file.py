import sklearn.datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

class Perceptron:
	def __init__ (self):
		self.w = None
		self.th = None

	def check(self, x):
		self.w = np.array(list(map(float,self.w)))
		x = np.array(list(map(float,x)))
		return 1 if (np.dot(self.w, x) >= self.th) else 0

	def predict(self, X):
		Y = []
		for x in X:
			result = self.check(x)
			Y.append(result)
		return np.array(Y)

	def learn(self, X, Y, iter = 1, rate = 1):
		self.w = np.ones(X.shape[1])
		self.th = 0
		accuracy = {}
		max_accuracy = 0.0
		for i in range(iter):
			
			for x, y in zip(X, Y):
				y_pred = self.check(x)
				if y == 1 and y_pred == 0:
					self.w = self.w + rate * x
					self.th = self.th - rate * 1
				elif y == 0 and y_pred == 1:
					self.w = self.w - rate * x
					self.th = self.th + rate * 1
				accuracy[i] = accuracy_score(self.predict(X), Y)

			accuracy[i] = accuracy_score(self.predict(X), Y)
			if (accuracy[i] > max_accuracy):
				max_accuracy = accuracy[i]
				chkptw = self.w
				chkptb = self.th
			if (accuracy[i] > 0.9):
				break
		self.w = chkptw
		self.th = chkptb

		return self.w

def fileToVector(path):
	letterCounter = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	with open(path) as f:
		lines = f.read().splitlines()
	letters = list(list(lines)[0])
	for el in letters:
		val = list(map(lambda x: ord(x) - ord('a') if (ord(x) > 96 and ord(x) < 123) else -1, el))[0]
		if val > -1:
			letterCounter[val] += 1
	letterCounter2 = list(map(lambda x: x/sum(letterCounter), letterCounter))
	return letterCounter2

perceptron1 = Perceptron()
perceptron2 = Perceptron()
perceptron3 = Perceptron()


train_files = []
train_Y_1 = []
train_Y_2 = []
train_Y_3 = []

def trejning():
	train_files.append(fileToVector('train/ang/1.txt'))
	train_Y_1.append(1)
	train_Y_2.append(0)
	train_Y_3.append(0)
	train_files.append(fileToVector('train/ang/2.txt'))
	train_Y_1.append(1)
	train_Y_2.append(0)
	train_Y_3.append(0)
	train_files.append(fileToVector('train/ang/3.txt'))
	train_Y_1.append(1)
	train_Y_2.append(0)
	train_Y_3.append(0)
	train_files.append(fileToVector('train/nor/1.txt'))
	train_Y_1.append(0)
	train_Y_2.append(1)
	train_Y_3.append(0)
	train_files.append(fileToVector('train/nor/2.txt'))
	train_Y_1.append(0)
	train_Y_2.append(1)
	train_Y_3.append(0)
	train_files.append(fileToVector('train/nor/3.txt'))
	train_Y_1.append(0)
	train_Y_2.append(1)
	train_Y_3.append(0)
	train_files.append(fileToVector('train/pl/1.txt'))
	train_Y_1.append(0)
	train_Y_2.append(0)
	train_Y_3.append(1)
	train_files.append(fileToVector('train/pl/2.txt'))
	train_Y_1.append(0)
	train_Y_2.append(0)
	train_Y_3.append(1)
	train_files.append(fileToVector('train/pl/3.txt'))
	train_Y_1.append(0)
	train_Y_2.append(0)
	train_Y_3.append(1)

trejning()

train_files = np.asarray(train_files)

perceptron1.learn(train_files, train_Y_1, 100, 0.2)
perceptron2.learn(train_files, train_Y_2, 100, 0.2)
perceptron3.learn(train_files, train_Y_3, 100, 0.2)


def test(path):
	test_vector = []
	test_vector.append(fileToVector(path))
	if perceptron1.predict(test_vector)[0] == 1:
		return "ang"
	if perceptron2.predict(test_vector)[0] == 1:
		return "nor"
	if perceptron3.predict(test_vector)[0] == 1:
		return "pl"
	return "?"


print(test('test/ang/1.txt'))
print(test('test/nor/1.txt'))
print(test('test/pl/1.txt'))
