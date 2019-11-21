import json
import jieba

from minitools.db.mongodb import get_mongodb_client


def cutTitle(title):
    return [i for i in jieba.lcut(title) if len(i.strip()) > 1]


def get_all_key_counter():
    cza = get_mongodb_client()['news']['train']
    sdict = {}
    for new in cza.find({}, {'_id': 0, 'label': 1, '标题': 1}):
        sdict[new['label']] = sdict.get(new['label'], 0) + 1
    with open('keyCounter.json', 'w') as f:
        f.write(json.dumps(sdict, ensure_ascii=False))


def clear_key_counter():
    with open('keyCounter.json', 'r') as f:
        json_data = json.loads(f.read())
    new_data = {}
    for key, value in json_data.items():
        if value >= 50:
            new_data[key] = value
    new_data.pop('Home')
    with open('keyCounter.json', 'w') as f:
        f.write(json.dumps(new_data, ensure_ascii=False))


def get_vector_wordsTemplate():
    cza = get_mongodb_client()['news']['train']
    titles = set()
    for new in cza.find({}, {'_id': 0, 'label': 1, '标题': 1}):
        titles |= set(cutTitle(new['标题']))
    with open('wordsTemplate.json', 'w') as f:
        f.write(json.dumps(list(titles), ensure_ascii=False))


def trainDataSet():
    mongoClient = get_mongodb_client()['news']['train']
    with open('keyCounter.json', 'r') as f:
        keyCounter = list(json.loads(f.read()).keys())
    with open('wordsTemplate.json', 'r') as f:
        wordsTemplate = json.loads(f.read())
    with open('train.txt', 'w+') as f:
        for key in keyCounter:
            index_dict = {}
            for new in mongoClient.find({'label': key}, {'_id': 0, 'label': 1, '标题': 1}):
                for word in cutTitle(new['标题']):
                    if word in wordsTemplate:
                        index = wordsTemplate.index(word)
                        index_dict[index] = index_dict.get(index, 0) + 1
            sumValue = sum(index_dict.values())
            index_dict = {key: value / sumValue for key, value in index_dict.items()}
            f.write(json.dumps(index_dict, ensure_ascii=False) + '|' + key + '\n')
            del index_dict
            print(key, 'done')


if __name__ == '__main__':
    # get_all_key_counter()
    # clear_key_counter()
    # get_vector_wordsTemplate()
    trainDataSet()
