from numpy import *
import numpy as np
import operator
import os
import json

    # Labels is matching each data from database, k is set to get range about nearest beighbor
def classify0(inX, dataSet, labels, k): # inX is values you want to match, dataSet is learning database
	# get data first dimenson -> 4
	# in this database, it has 4 rows and 2 cols ->(4L,2L)-> result of shape[0] is the row values: 4, and the shape[1] is 2
	dataSetSize = dataSet.shape[0]

	# tile func is repetite the source data by your setting.
	# 1 is just repetiting one time in one-dimensional. if it is (2,1), it will create two-one-dimensional-array
	# regulation: you must input a data you want to predict, which has a same cols as learning data
	# and you need not to care the row, we can learn all from learning data except cols
	# the purpose of this suptraction -> (x-x)
	diffMat = tile(inX, (dataSetSize,1)) - dataSet

	sqDiffMat = diffMat**2 # (x-x)**2, remember it is still a array
	sqDistances = sqDiffMat.sum(axis=1) # (x-x)**2 + (x-x)**2, remember it is still a array
	# (axis=1) meaning adding each value in each row. the result of sum is still a array, BUT it is not (4L,2L). it become (1L,4L)
	# the result format is (1L,4L)
	distances = sqDistances**0.5 # ((x-x)**2 + (x-x)**2)**0.5, remember it is still a array
	
	sortedDistIndicies = distances.argsort() # rerurn the array's index by reverse = False
	classCount = {} # define a dictionary
	for i in range(k):
		voteIlabel = labels[sortedDistIndicies[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
	sortedClassCount = sorted(classCount.items(),
	key = operator.itemgetter(1), reverse = True)
	return sortedClassCount[0][0]#,sortedClassCount[0][1],sortedClassCount[1][0],sortedClassCount[1][1]

def img2vector(filename):
	returnVect = zeros((1,900))
	fr = open(filename)
	#print(fr)
	c = 0
	for i in range(30):
		lineStr = fr.readline()
		lineStr = lineStr.split(' ')
		#print(len(lineStr))
		#print(lineStr)
		#c += 1
		#print(c)
		for j in range(30):
			#print(type(lineStr[j]))
			#print(lineStr[j])
			returnVect[0, 30*i+j] = float(lineStr[j])
			
	return returnVect

#temp = img2vector('2.txt')
#print(temp)
#print(os.getcwd()+'\\train')
#print(os.listdir(os.getcwd()+'\\train'))
#"""
def handwritingClassTest():
	#dict_json = {}

	#hwLabels = []

	"""
	trainingFileList = os.listdir(os.getcwd()+'\\train')
	m = len(trainingFileList)
	trainingMat = zeros((m, 900))
	for i in range(m):
		fileNameStr = trainingFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr)
		hwLabels.append(classNumStr)
		trainingMat[i, :] = img2vector(os.getcwd()+'\\train\\%s'%fileNameStr)
		#dict_json[fileStr] = [str(cza) for cza in trainingMat[i, :]]
	
	#with open("cza.json",'w',encoding='utf-8') as json_file:
	#	json.dump(dict_json,json_file,ensure_ascii=False)
	np.savetxt('cza_values.txt',trainingMat)
	with open('cza_keys.txt','w') as f_w:
		hwLabels_s = [str(i) for i in hwLabels]
		#print(hwLabels_s)
		#print(''.join(hwLabels_s))
		#hwLabels_s = [str for i in hwLabels]
		f_w.write(''.join(hwLabels_s))
	"""
	trainingMat = np.loadtxt('cza_values.txt')
	with open('cza_keys.txt','r') as f_r:
		data = f_r.readline()
		hwLabels = [int(i) for i in data]

	testFileList = os.listdir(os.getcwd()+'\\test')
	errorCount = 0.0
	mTest = len(testFileList)
	for i in range(mTest):
		fileNameStr = testFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr)
		vectorUnderTest = img2vector(os.getcwd()+'\\test\\%s'%fileNameStr)

		#with open("cza.json",'r',encoding='utf-8') as json_file:
		#	dict_json=json.load(json_file)
			#print(dict_json)
		#	_trainingMat = [dict_json.values()]
		#	_hwLabels = [int(cza) for cza in dict_json.keys()]

		classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
		print('trainingMat_resylt_is ', classifierResult,'real result is ',classNumStr)
		if classifierResult != classNumStr:
			errorCount += 1
	print(errorCount)

#"""
handwritingClassTest()