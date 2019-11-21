import json
import jieba


def cutTitle(title):
    return [i for i in jieba.lcut(title) if len(i.strip()) > 1]


def classifyV1(title=""):
    wordTitle = title
    with open('wordsTemplate.json', 'r') as f:
        allUniqueLabel = json.loads(f.read())
    index_dict = {}
    for word in cutTitle(wordTitle):
        if word in allUniqueLabel:
            index = allUniqueLabel.index(word)
            index_dict[index] = index_dict.get(index, 0) + 1
    wordLabel = None
    wordValue = 0
    with open('train.txt', 'r') as f:
        while True:
            content = f.readline()
            if not content:
                break
            json_data, label = content.strip().split('|')
            trainSet = json.loads(json_data)  # 得到训练数据
            wordValueTemp = 0
            for key, value in index_dict.items():
                trainKey = trainSet.get(str(key), None)
                if trainKey:
                    wordValueTemp += value * trainKey
            if wordValueTemp > wordValue:
                wordValue = wordValueTemp
                wordLabel = label
    return wordLabel
