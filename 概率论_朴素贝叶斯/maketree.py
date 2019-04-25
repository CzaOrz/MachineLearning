def makeTree(dataSet,labels):
	colNum = [example[-1] for example in dataSet]
	if colNum.count(colNum[0]) == len(colNum):
		return colNum[0]
	if len(dataSet[0]) == 1:
		return majorCol[colNum]
	bestIndex = choose()
	bestlabels = labels[bestIndex]
	mytree = {bestlabels:{}}
	del(labels[bestIndex])
	bestValue = [example[bestIndex] for example in dataSet]
	unique = set(bestValue)
	for eachValue in unique:
		dataSetSplit = split(dataSet,bestIndex,eachValue)
		labels = labels[:]
		mytree[bestlabels][eachValue] = makeTree(dataSetSplit,labels)
	return mytree
