#!/usr/local/bin/python

class ModelMetrics(object):
	"""docstring for ModelMetrics"""
	matrix = None
	def __init__(self, matrix):
		self.matrix = matrix
		
	def getTP(self, cls):
		TP = 0
		for pred_cls in self.matrix[cls]:
			if pred_cls == cls:
				TP = self.matrix[cls][pred_cls]
		return TP

	def getFN(self, cls):
		FN = 0
		for pred_cls in self.matrix[cls]:
			if pred_cls != cls:
				FN += self.matrix[cls][pred_cls]
		return FN

	def getFP(self, cls):
		FP = 0
		for true_cls in self.matrix:
			if cls != true_cls:
				FP += self.matrix[true_cls][cls]
		return FP


	def getTN(self, cls):
		TN = 0
		for true_cls in self.matrix:
			if cls != true_cls:
				for pred_cls in self.matrix[true_cls]:
					if pred_cls != cls:
						TN += self.matrix[true_cls][pred_cls]
		return TN

	def accuracy(self):
		correct = 0
		total_size = 0
		for cls in self.matrix:
			for pred_cls in self.matrix[cls]:
				if cls == pred_cls:
					correct += self.matrix[cls][pred_cls]
				total_size += self.matrix[cls][pred_cls]
		accuracy = float(correct)/float(total_size)
		return accuracy

	def sensitivity(self, TP, FN):
		return TP/(TP + FN)

	def specificity(self, TN, FP):
		return  TN/(TN + FP)

	def precision(self, TP, FP):
		return TP/(TP + FP)

	def recall(self, TP, FN):
		return TP/(TP + FN)

	def f1score(self, TP, FP, FN):
		return 2*TP/(2*TP + FP + FN)

	def fbetascore(self, P, R, beta):
		return ((1 + (beta ** 2))*P*R)/((beta ** 2)*P + R)

	def getMetrics(self):
		accuracy = self.accuracy()
		print "Overall accuracy = {0:.3f}".format(accuracy)
		
		for cls in self.matrix:
			TP, FN, FP, TN = self.getTP(cls), self.getFN(cls), self.getFP(cls), self.getTN(cls)
			TP, FN, FP, TN = float(TP), float(FN), float(FP), float(TN)
			sensitivity, specificity, precision, recall = 0.0, 0.0, 0.0, 0.0

			if not (TP == 0 and FN == 0):
				sensitivity = self.sensitivity(TP, FN)
			
			if not (TN == 0 and FP == 0):
				specificity = self.specificity(TN, FP)
			
			if not (TP == 0 and FP == 0):
				precision = self.precision(TP, FP)

			if not (TP == 0 and FN == 0):
				recall = self.recall(TP, FN)
	
			f1score, fbetascore_0_5, fbetascore_2 = 0.0, 0.0, 0.0
			if not (precision == 0.0 and recall == 0.0):
				f1score = self.f1score(TP, FP, FN)
				fbetascore_0_5 = self.fbetascore(precision, recall, 0.5)
				fbetascore_2 = self.fbetascore(precision, recall, 2.0)

			print "For class {0}: sensitivity = {1:.3f}, specificity = {2:.3f}, precision = {3:.3f}, recall = {4:.3f}, f1score = {5:.3f}, fbetascore_0_5 = {6:.3f}, fbetascore_2 = {7:.3f}" \
			.format(cls, sensitivity, specificity, precision, recall, f1score, fbetascore_0_5, fbetascore_2)

		return


def main():
	metrics = ModelMetrics(None)
	metrics.getMetrics()

if __name__ == '__main__':
	main()
