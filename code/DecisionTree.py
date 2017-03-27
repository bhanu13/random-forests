#!/usr/local/bin/python

from modelmetrics import * 
from dataset import *

class TreeNode:
	left = None
	right = None
	split_attr = None
	split_val = None
	label = None

	def __init__(self):
		self.left = None
		self.right = None
		self.split_attr = None
		self.split_val = None
		self.label = None

	def __str__(self):
		if self.split_attr != None and self.split_val != None:
			return "Split attr: {}, val: {}".format(self.split_attr, self.split_val)
		elif self.label:
			return "Label: {}".format(self.label)
		else:
			return "Uninitialized"

class DecisionTree(object):
	root = None
	labels = list()
	attrs = list()
	
	# Parameters
	ginithreshold = 0.005
	maxDepth = 100
	remove_attr = True

	def __init__(self):
		self.root = TreeNode()

	def getLabels(self, dataPartition):
		labels = {}
		for i in range(0, len(dataPartition)):
			label = dataPartition[i][1]
			if label not in labels:
				labels[label] = 1
			else:
				labels[label] += 1
		return labels

	def gini(self, data):
		labelFreq = {}
		for vec in data:
			label = vec[1]
			if label in labelFreq:
				labelFreq[label] += 1
			else:
				labelFreq[label] = 1
		gini = 1.0
		for label, freq in labelFreq.iteritems():
			gini -= ((freq/float(len(data))) ** 2)

		return gini

	def getGiniIndex(self, split_attr, split_val, dataPartition):
		leftData, rightData = self.partitionData(split_attr, split_val, dataPartition)
		g1 = self.gini(leftData)
		g2 = self.gini(rightData)
		return (float(len(leftData))/float(len(dataPartition)))*g1 + (float(len(rightData))/float(len(dataPartition)))*g2

	def getBestSplit(self, dataPartition, attr_list):
		minGini = 1.0
		minAttr, minVal = 0, 0
		for attr in attr_list:
			for val in self.attrs[attr]:
				gini = self.getGiniIndex(attr, val, dataPartition)
				if gini < minGini:
					minGini = gini
					minAttr, minVal = attr, val
		# print "gini: {}, minAttr: {}, minVal, {}".format(minGini, minAttr, minVal)
		assert not(minAttr == 0 and minVal == 0)

		g = self.gini(dataPartition)
		# Min. split threshold
		if (g - minGini) < self.ginithreshold:
			return None, None
		return minAttr, minVal

	def partitionData(self, split_attr, split_val, data):
		leftData, rightData = [], []
		for vec in data:
			if vec[0][split_attr] == split_val:
				leftData.append(vec)
			else:
				rightData.append(vec)

		return leftData, rightData

	def trainTree(self, dataPartition):
		self.setupTree(dataPartition)
		attr_list = set([i for i in range(0, len(self.attrs))])
		return self.buildTree(self.root, dataPartition, attr_list, 0)

	def subsetAttrs(self, attr_list):
		return attr_list

	def buildTree(self, node, dataPartition, attr_list, depth):
		labels = self.getLabels(dataPartition)
		# print "Length of the data Partition: {}".format(len(dataPartition))
		if len(labels) == 1:
			node.label = labels.keys()[0]
			return

		if not attr_list:
			node.label = max(labels, key=labels.get)
			return

		node.label = max(labels, key=labels.get)
		if depth > self.maxDepth:
			return

		subset_attr_list = self.subsetAttrs(attr_list)
		if not subset_attr_list:
			return
		# Get best split
		split_attr, split_val = self.getBestSplit(dataPartition, subset_attr_list)
		if split_attr == None and split_val == None:
			return
		node.split_attr, node.split_val = split_attr, split_val
		if self.remove_attr:
			attr_list.remove(split_attr)

		# Partition the data
		leftData, rightData = self.partitionData(split_attr, split_val, dataPartition)
		
		# Continue building after splitting
		if leftData:
			node.left = TreeNode()
			self.buildTree(node.left, leftData, set(attr_list), depth + 1)

		if rightData:
			node.right = TreeNode()
			self.buildTree(node.right, rightData, set(attr_list), depth + 1)

		assert (node.split_val != None and node.split_attr != None) or node.label != None
		return

	def setupTree(self, trainData):
		self.labels = list(set(map(lambda vec: vec[1], trainData)))
		features = map(lambda vec: vec[0], trainData)
		self.attrs = list()
		for i in range(0, len(features[0])):
			self.attrs.append(list(set([row[i] for row in features])))
		return

	# Binary Tree
	def getLabel(self, node, feature):
		if not node.left and not node.right:
			return node.label
		elif not node.right:
			return self.getLabel(node.left, feature)
		elif not node.left:
			return self.getLabel(node.right, feature)
		else:
			if feature[node.split_attr] == node.split_val:
				return self.getLabel(node.left, feature)
			else:
				return self.getLabel(node.right, feature)

	def predict(self, testData):
		accuracy = 0.0
		correct = 0
		for vec in testData:
			if self.getLabel(self.root, vec[0]) == vec[1]:
				correct += 1
		accuracy = float(correct)/float(len(testData))
		return accuracy

	def getConfusionMatrix(self, testData):
		matrix = {label:{label: 0 for label in self.labels} for label in self.labels}

		for vec in testData:
			predicted_label = self.getLabel(self.root, vec[0])
			label = vec[1]
			matrix[label][predicted_label] += 1
		return matrix

	def printConfustionMatrix(self, mat):
		for label in mat:
			for predicted_label in mat:
				print mat[label][predicted_label],
			print ""


	def __str__(self):
		nodes = []
		return "\n".join(self.preOrder(self.root, nodes))

	def preOrder(self, root, nodes):
		if root:
			nodes.append(str(root))
			self.preOrder(root.left, nodes)
			self.preOrder(root.right, nodes)
			return nodes


def main():
	trainData, testData = getDataSet()
	dTree = DecisionTree()
	dTree.trainTree(trainData)
	# acc = dTree.predict(trainData)
	# print "Train Accuracy: {}".format(acc)
	acc = dTree.predict(testData)
	# print "Test Accuracy: {}".format(acc)
	mat = dTree.getConfusionMatrix(testData)
	dTree.printConfustionMatrix(mat)
	metrics = ModelMetrics(mat)
	# metrics.getMetrics()
	# print "Possible Labels", dTree.labels
	# print dTree

if __name__ == '__main__':
	main()