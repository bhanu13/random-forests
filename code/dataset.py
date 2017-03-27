#!/usr/local/bin/python

import sys
import os

def convertToFeature(data):
	featurized_data = []
	try:
		for line in data:
			line = line.split(" ")
			features = map(lambda elem: elem.split(":"), line[1:])
			features = [elem[1] for elem in features]
			features = map(int, features)
			label = int(line[0])
			featurized_data.append([features, label])
		return featurized_data

	except Exception, e:
		print "Error in data format: ", e
		raise Exception

def read_file(fname):
	f = open(fname, 'r')
	data = f.readlines()
	return map(lambda line: line[:-1], data)

def getDataSet():
	files = sys.argv[1:]
	testData, trainData = None, None
	for i in range(0, len(files)):
		if os.path.exists(files[i]):
			if i == 0:
				trainData = read_file(files[i])
			if i == 1:
				testData = read_file(files[i])
		else:
			print "Invalid File Path : {}".format(files[i])

	trainData = filter(lambda x: x, trainData)
	testData = filter(lambda x: x, testData)

	trainData = convertToFeature(trainData)
	testData = convertToFeature(testData)
	return trainData, testData
