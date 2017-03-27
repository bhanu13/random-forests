echo "For Data Set 1"
echo "DecisionTree"
python DecisionTree.py ../datasets/set1/balance-scale.train ../datasets/set1/balance-scale.test
echo "RandomForest"
python RandomForest.py ../datasets/set1/balance-scale.train ../datasets/set1/balance-scale.test
echo "---"

echo "For Data Set 2"
echo "DecisionTree"
python DecisionTree.py ../datasets/set2/nursery.data.train ../datasets/set2/nursery.data.test
echo "RandomForest"
python RandomForest.py ../datasets/set2/nursery.data.train ../datasets/set2/nursery.data.test
echo "---"

echo "For Data Set 3"
echo "DecisionTree"
python DecisionTree.py ../datasets/set3/led.train.new ../datasets/set3/led.test.new
echo "RandomForest"
python RandomForest.py ../datasets/set3/led.train.new ../datasets/set3/led.test.new
echo "---"

echo "For Data Set 4"
echo "DecisionTree"
python DecisionTree.py ../datasets/set4/poker.train ../datasets/set4/poker.test
echo "RandomForest"
python RandomForest.py ../datasets/set4/poker.train ../datasets/set4/poker.test
echo "---"