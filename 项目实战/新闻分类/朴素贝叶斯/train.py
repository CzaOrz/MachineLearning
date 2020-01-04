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
    # mongoClient = get_mongodb_client()['news']['train']
    with open('keyCounter.json', 'r') as f:
        keyCounter = list(json.loads(f.read()).keys())
    with open('wordsTemplate.json', 'r') as f:
        json_data = json.loads(f.read())
    # with open('train.txt', 'w+') as f:
    #     for key in keyCounter:
    #         index_dict = {}
    #         for new in mongoClient.find({'label': key}, {'_id': 0, 'label': 1, '标题': 1}):
    #             for word in cutTitle(new['标题']):
    #                 if word in wordsTemplate:
    #                     index = wordsTemplate.index(word)
    #                     index_dict[index] = index_dict.get(index, 0) + 1
    #         sumValue = sum(index_dict.values())
    #         index_dict = {key: value / sumValue for key, value in index_dict.items()}
    #         f.write(json.dumps(index_dict, ensure_ascii=False) + '|' + key + '\n')
    #         del index_dict
    #         print(key, 'done')
    cza = get_mongodb_client()['news']['train']
    with open('train.txt', 'w+') as f:
        for key in keyCounter:
            index_dict = {}
            jss = 0  # 训练数据是不是需要稍微限制一下。还是需要限制一下，应该本身自己分得就不是很对
            for new in cza.find({'label': key}, {'_id': 0, 'label': 1, '标题': 1}):
                if jss == 5001:  # 限制5k把
                    break
                jss += 1
                for word in cutTitle(new['标题']):
                    if word in json_data:
                        index = json_data.index(word)
                        index_dict[index] = index_dict.get(index, 0) + 1
            dict_values = index_dict.values()
            dict_values_max = max(dict_values)
            dict_values_min = min(dict_values)
            if dict_values_max == dict_values_min:
                if dict_values_min == 1:
                    continue
                else:
                    dict_values_min = dict_values_min / 2
            denominator = dict_values_max - dict_values_min
            new_index_dict = {}
            for index_dict_key, index_dict_value in index_dict.items():
                v = (index_dict_value - dict_values_min) / denominator
                if v < 0.8:
                    v = min(max(v, 0.03), 0.06)
                elif v < 0.5:
                    pass
                elif v < 0.7:
                    v = v / 1.3
                else:
                    v = v / 5
                new_index_dict[index_dict_key] = float(v)
            f.write(json.dumps(new_index_dict) + '|' + key + '\n')
            del index_dict, new_index_dict
            print(key, 'done')


if __name__ == '__main__':
    # get_all_key_counter()
    # clear_key_counter()
    # get_vector_wordsTemplate()
    trainDataSet()
