import math,numpy,operator

def loadDataSet():
	listVec = [['my','maybe','has','flea','problems','help','ple'],
	          ['dog','licks','take','ple','him','to','dog','park'],
	          ['stop','posting','is','so','cute','love','dalmation'],
	          ['dog','licks','my','ste','how','to','xixi'],
	          ['mr','aa','my','ste','dd','to','xixi','fr'],
	          ['dog','licks','dd','ste','ww','to','xixi','fr']]
	classVec = [0,1,0,1,0,1]
	return listVec,classVec

def createUniqueList(dataSet):
	uniqueList = set([])
	for eachContent in dataSet:
		uniqueList = uniqueList | set(eachContent)
	return list(uniqueList)

def setListToNum(listVec,inputSet): # transform what you input into the standard format, if it exist in input, add 1.
	result = [0] * len(listVec)
	for eachWorkd in inputSet:
		if eachWorkd in listVec:
			result[listVec.index(eachWorkd)] = 1
	return result

def trainNB(trainMatrix,trainCategory): # the trainMatrix is matrix that has been transform into 0 and 1
	numTrainMatrix = len(trainMatrix) # so the length of this matrix[0] is all length of unique work in train source 
	numWords = len(trainMatrix[0])
	Rate = sum(trainCategory)/float(numTrainMatrix) # sum(trainCategory) is all 1 to add
	p0Num,p1Num = numpy.zeros(numWords),numpy.zeros(numWords)
	p0Denom,p1Denom = 0.0,0.0
	#print(trainCategory)
	#print(numTrainMatrix)
	for i in range(numTrainMatrix):
		#print(i,trainCategory[i])
		if trainCategory[i] == 1:
			p1Num += trainMatrix[i]
			p1Denom += sum(trainMatrix[i])
		else:
			p0Num += trainMatrix[i]
			p0Denom += sum(trainMatrix[i])
	p1Vec = p1Num/p1Denom
	p0Vec = p0Num/p0Denom
	return p0Vec,p1Vec,Rate

def classifyNB(vec,p0Vec,p1Vec,Rate):
	p1 = sum(vec * p1Vec) + numpy.log(Rate)
	p0 = sum(vec * p0Vec) + numpy.log(1.0 - Rate)
	if p1 > p0:
		return 1
	else:
		return 0

def testing():
	a1,b1 = loadDataSet()
	a2 = createUniqueList(a1)
	trainMatrix = []
	for i in a1:
		trainMatrix.append(setListToNum(a2,i))
	a3,b3,c3 = trainNB(numpy.array(trainMatrix),numpy.array(b1)) # this is training result, we can calculate probability value
	testEntry = ['love','my','dal']
	thisDoc = numpy.array(setListToNum(a2,testEntry))
	print('this is %s'%(classifyNB(thisDoc,a3,b3,c3)))
	testEntry = ['stupid','dog','gar']
	thisDoc = numpy.array(setListToNum(a2,testEntry))
	print('this is %s'%(classifyNB(thisDoc,a3,b3,c3)))

def test():
	a,b = loadDataSet()
	a1 = createUniqueList(a)
	trainMatrix = []
	for pos in a:
		trainMatrix.append(setListToNum(a1,pos))
	#print(a)
	#print(setListToNum(a,['aasda','aasdaa','aaasdaa','mr','licks','dd','ste','ww','to','xixi','fr','eqewq']))
	a3,b3,c3 = trainNB(trainMatrix,b)
	#print(a3,b3,c3)
	testing()

if __name__ == '__main__':
	test()