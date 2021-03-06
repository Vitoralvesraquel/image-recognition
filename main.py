from imageai.Classification import ImageClassification
import os
import time

t1 = time.time()

execution_path = os.getcwd()

prediction = ImageClassification()
prediction.setModelTypeAsDenseNet121()
prediction.setModelPath(os.path.join(execution_path, 'DenseNet-BC-121-32.h5'))
prediction.loadModel()
predictions, probabilities = prediction.classifyImage(os.path.join(execution_path, 'marci.jpg'), result_count=5)

for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction, " : ", eachProbability)

t2 = time.time()
t = round(t2 - t1, 2)
print(f'\nseconds elapsed:{t}')

# results:
#
# kelpie  :  37.8971666097641
# schipperke  :  34.560626745224
# Chihuahua  :  11.118903011083603
# Staffordshire_bullterrier  :  6.137801706790924
# Labrador_retriever  :  3.2416295260190964
# 
# seconds elapsed:38.56
