import re
from minitools.db.mongodb import get_mongodb_client
from collections import Counter
from pprint import pprint
import json
import jieba
from minitools.ml.naivebayes import NaiveBayes
import numpy as np


def cutTitle(title):
    return [i for i in jieba.lcut(title) if len(i.strip()) > 1]


def test1():
    text = """
    军事：
    金融：补贴信息
    财经：建设进展
    时政：时政|国务院|两会|工作部门|工作动态|政务公开|政策解读|依法行政|组织机构|贯彻落实
    行政：行政权力
    视频：视频
    图片：图片报道
    民生：老年人|脱贫攻坚|便民|就业信息|住房保障
    交通：路况信息
    社会：专题专栏|新时代|热点聚集|长治市红十字会|聚焦民企
    人文：人文
    人事：专辑-|部门信息|工作交流|领导活动|领导信息
    数据：数据分析|公共数据
    文件：文件
    年报：年度报告|年报
    要闻：-要闻
    公告：采购公告|公示公告|公开报告
    最新动态：最新动态
    业务：业务信息|总结计划
    应急管理：应急管理
    行政处罚：行政处罚
    法律法规：公开规定
    监督检查：监督检查
    行政许可：行政许可
    """
    RER = re.compile('[：:](.*)').search
    filter_in = []
    for tex in text.strip().split('\n'):
        te = RER(tex).group(1)
        filter_in.extend([i for i in te.split('|') if i])
    cza = get_mongodb_client()['news']['train']
    count = 0
    for new in cza.find({}, {'_id': 0, '金融分类': 1}):
        if count == 300:
            break
        ccc = new['金融分类']
        flag = 0
        for i in filter_in:
            if i in ccc:
                flag = 1
                break
        if flag:
            continue
        print(ccc)
        count += 1


def test2():
    cza = get_mongodb_client()['news']['train']
    sett = set()
    count = 0
    NO_D = re.compile('\d+').search
    SSUB = re.compile('\s').sub
    for new in cza.find({}, {'金融分类': 1}):
        ccc = new['金融分类'].split('-')
        label = [i.strip() for i in ccc[-2:]]
        if len(label[0]) != 4 and len(label[1]) != 4 and \
                len(SSUB('', label[0])) != 4 and len(SSUB('', label[1])) != 4:
            # cza.delete_one({'_id': new['_id']})  # 删除是不是有点久啊
            continue
        label = label[0] if len(label[0]) == 4 else label[1]
        if NO_D(label):
            # cza.delete_one({'_id': new['_id']})  # 删除是不是有点久啊
            continue
        # cza.update_one({'_id': new['_id']}, {'$set': {'label': label}})
        count += 1
        sett.add(label)


def test3():
    cza = get_mongodb_client()['news']['train']
    sdict = {}
    for new in cza.find({}, {'_id': 0, 'label': 1, '标题': 1}):
        sdict[new['label']] = sdict.get(new['label'], 0) + 1
    pprint(sdict)
    with open('keyCounter.json', 'w') as f:
        f.write(json.dumps(sdict, ensure_ascii=False))


def test4():
    with open('keyCounter.json', 'r') as f:
        json_data = json.loads(f.read())
    # pprint(json_data)
    # print(len(json_data))  # 2343
    new_data = {}
    for key, value in json_data.items():
        if value >= 50:
            new_data[key] = value
    new_data.pop('Home')
    # pprint(new_data)
    # print(len(new_data))
    with open('newKeyCounter.json', 'w') as f:
        f.write(json.dumps(new_data, ensure_ascii=False))


def test5():
    with open('newKeyCounter.json', 'r') as f:  # 所有的分类以及各个分类的数据量
        json_data = json.loads(f.read())
    pprint(json_data)
    values = json_data.values()
    print(sum(values))  # 总计 483184 条


def test6():
    cza = get_mongodb_client()['news']['train']
    titles = set()
    for new in cza.find({}, {'_id': 0, 'label': 1, '标题': 1}):
        titles |= set(cutTitle(new['标题']))  # 改为非全切割，成功率会提高。但是任然存在模型致命缺陷。模型的前提就是词与词之间无关联，相互独立！
    # pprint(titles)
    print(len(titles))  # 501878 => 106506 最新少了好多
    with open('allUniqueLabel.json', 'w') as f:
        f.write(json.dumps(list(titles), ensure_ascii=False))


def test7():
    with open('allUniqueLabel.json', 'r') as f:
        json_data = json.loads(f.read())
    # NO_D = re.compile('\d+').search
    # print(len(json_data))
    # json_data = [i for i in json_data if not NO_D(i)]
    # print(len(json_data))
    # with open('allUniqueLabel.json', 'w') as f:
    #     f.write(json.dumps(json_data, ensure_ascii=False))


def test8():
    with open('newKeyCounter.json', 'r') as f:
        keyCounter = list(json.loads(f.read()).keys())
    with open('allUniqueLabel.json', 'r') as f:
        json_data = json.loads(f.read())
    # vector = np.array([1 for _ in range(len(json_data))])
    # count = len(json_data)
    # vector = vector / count
    # print(len(keyCounter))  # 832
    # -------------------------------------------------------------
    cza = get_mongodb_client()['news']['train']
    # count = []
    # ttt = 0
    # for key in keyCounter:
    #     ttt += 1
    #     count.append(cza.count({'label': key}))
    # print(ttt)
    # print(sum(count))

    # ---------------- 训练数据的具体代码 ------------------------------
    with open('train.txt', 'w+') as f:
        for key in keyCounter:
            index_dict = {}
            for new in cza.find({'label': key}, {'_id': 0, 'label': 1, '标题': 1}):
                # todo 可以只保存下标，获取得到字典，然后训练字典与目标字典进行合并，已朴素贝叶斯求解
                for word in cutTitle(new['标题']):
                    if word in json_data:
                        index_dict[json_data.index(word)] = index_dict.get(json_data.index(word), 0) + 1
            sumValue = sum(index_dict.values())
            index_dict = {key: value / sumValue for key, value in index_dict.items()}
            f.write(json.dumps(index_dict, ensure_ascii=False) + '|' + key + '\n')
            del index_dict
            print(key, 'done')


def test9(title="", fromSource=False):
    wordTitle = title
    with open('allUniqueLabel.json', 'r') as f:
        json_data = json.loads(f.read())
    index_dict = {}
    for word in cutTitle(wordTitle):
        if word in json_data:
            # print(word)
            index_dict[json_data.index(word)] = index_dict.get(json_data.index(word), 0) + 1
    # print(cutTitle('每日水质结果公示9月15日'))
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
            if label == '每日水质' or label == '人事任免':
                print(label, wordValueTemp, '##################')
            else:
                print(label, wordValueTemp)
            if wordValueTemp > wordValue:
                wordValue = wordValueTemp
                wordLabel = label
        print(f"原标题：{wordTitle}")
        print(f"原：{get_mongodb_client()['news']['train'].find_one({'标题': wordTitle}, {'label': 1, '_id': 0})}") \
            if fromSource else None
        print(f"预测分类为：{wordLabel}\n")


def test10(limit=-1):
    with open('allUniqueLabel.json', 'r') as f:
        json_data = json.loads(f.read())
    right_num = 0
    all_num = 0
    for doc in get_mongodb_client()['news']['train'].find({}, {'_id': 0}):
        if all_num == limit:
            break
        all_num += 1
        wordTitle = doc['标题']
        index_dict = {}
        for word in cutTitle(wordTitle):
            if word in json_data:
                index_dict[json_data.index(word)] = index_dict.get(json_data.index(word), 0) + 1
        wordLabel = None
        wordValue = 0
        with open('train.txt', 'r') as f:
            while True:
                content = f.readline()
                if not content:
                    break
                train_json_data, label = content.strip().split('|')
                trainSet = json.loads(train_json_data)  # 得到训练数据
                wordValueTemp = 0
                for key, value in index_dict.items():
                    trainKey = trainSet.get(str(key), None)
                    if trainKey:
                        wordValueTemp += value * trainKey
                if wordValueTemp > wordValue:
                    wordValue = wordValueTemp
                    wordLabel = label
            print(f"原标题：{wordTitle}")
            print(f"金融分类：{doc['金融分类']}")
            print(f"原label：{doc['label']}")
            print(f"预测分类为：{wordLabel}")
            if doc['label'] == wordLabel:
                print("预测正确\n")
                right_num += 1
            else:
                print("预测错误\n")
    print(f"正确率为：{100*right_num/all_num}%")
def test11():
    with open('allUniqueLabel.json', 'r') as f:
        json_data = json.loads(f.read())
    client = get_mongodb_client()['news']['test']
    for doc in client.find({}):
        wordTitle = doc['标题']
        index_dict = {}
        for word in cutTitle(wordTitle):
            if word in json_data:
                index_dict[json_data.index(word)] = index_dict.get(json_data.index(word), 0) + 1
        wordLabel = None
        wordValue = 0
        with open('train.txt', 'r') as f:
            while True:
                content = f.readline()
                if not content:
                    break
                train_json_data, label = content.strip().split('|')
                trainSet = json.loads(train_json_data)  # 得到训练数据
                wordValueTemp = 0
                for key, value in index_dict.items():
                    trainKey = trainSet.get(str(key), None)
                    if trainKey:
                        wordValueTemp += value * trainKey
                if wordValueTemp > wordValue:
                    wordValue = wordValueTemp
                    wordLabel = label
        client.update_one({'_id': doc['_id']}, {'$set': {'v1-label': wordLabel}})
if __name__ == '__main__':
    # test1()
    # test2()
    # test3()
    # test4()
    # test5()
    # test6()
    # test7()
    test8()

    # for title in [
    #     # '临夏河州大道—北滨河大道中压天然气管道 工程竣工并正式通气',
    #     # '关于陈光荣等同志任职的通知',
    #     # '民政部社会救助司调研指导临夏州脱贫攻坚',
    #     # '姚逊在武乡县调研时强调抓好基层党建 助推脱贫攻坚',
    #     # '州人民政府第32次常务会议',
    #     # '州人民政府第30次常务会议',
    #     # '长治市脱贫攻坚简报总第46期',
    #     #
    #     # '大九湖镇：狠抓一线落实 强化责任担当 全力以赴做好迎检工作',
    #     # '宋洛乡：党风廉政建设 “三治融合”显实效',
    #     # '中共神农架林区委员会组织部干部任前公示公告（2019年第009号)',
    #     '组织部干部任前公示公告',
    # ]:
    #     test9(title, fromSource=True)

    # test10(100)
    # test11()

    # with open('keyCounter.json', 'r') as f:
    #     json_data = json.loads(f.read())
    #     print(len(json_data))  # 2343

    # pprint(jieba.lcut('中共神农架林区委员会组织部干部任前公示公告'))
