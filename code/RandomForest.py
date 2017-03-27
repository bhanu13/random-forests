#!/usr/local/bin/python

from modelmetrics import * 
from DecisionTree import *
from dataset import *
import random


class RFTree(DecisionTree):
	feature_subset = 0.65

	def __init__(self):
		super(RFTree,self).__init__()
		self.maxDepth = 8
		self.remove_attr = False

	def subsetAttrs(self, attr_list):
		new_attr_list = list(attr_list)
		sample_size = int(len(attr_list)*self.feature_subset)
		rand_smpl = [new_attr_list[i] for i in sorted(random.sample(xrange(len(attr_list)), sample_size))]
		# print rand_smpl
		return rand_smpl

class RandomForests(object):
	forest_size = 0
	subset_sample = 0.
	forest = list()
	labels = list()

	def __init__(self):
		self.forest_size = 10
		self.subset_sample = 0.6
		self.forest = list()

	def partitionData(self, data):
		sample_size = int(len(data)*self.subset_sample)
		rand_smpl = [data[i] for i in sorted(random.sample(xrange(len(data)), sample_size))]
		return rand_smpl

	def trainRF(self, trainData):
		for i in range(0, self.forest_size):
			partitionedData = self.partitionData(trainData)
			tree = RFTree()
			tree.trainTree(partitionedData)
			self.forest.append(tree)
		self.labels = self.forest[0].labels
		return

	def getLabel(self, feature):
		predicted_labels = []
		for i in range(0, len(self.forest)):
			predicted_labels.append(self.forest[i].getLabel(self.forest[i].root, feature))
		return max(set(predicted_labels), key=predicted_labels.count)

	def predict(self, testData):
		accuracy = 0.0
		correct = 0
		for vec in testData:
			if self.getLabel(vec[0]) == vec[1]:
				correct += 1
		accuracy = float(correct)/float(len(testData))
		return accuracy

	def getConfusionMatrix(self, testData):
		matrix = {label:{label: 0 for label in self.labels} for label in self.labels}

		for vec in testData:
			predicted_label = self.getLabel(vec[0])
			label = vec[1]
			matrix[label][predicted_label] += 1
		return matrix

	def printConfustionMatrix(self, mat):
		for label in mat:
			for predicted_label in mat:
				print mat[label][predicted_label],
			print ""

def main():
	trainData, testData = getDataSet()
	rf = RandomForests()
	rf.trainRF(trainData)
	# acc = rf.predict(trainData)
	# print "Train Accuracy: {}".format(acc)
	acc = rf.predict(testData)
	# print "Test Accuracy: {}".format(acc)
	mat = rf.getConfusionMatrix(testData)
	rf.printConfustionMatrix(mat)
	metrics = ModelMetrics(mat)
	# metrics.getMetrics()

if __name__ == '__main__':
	main()