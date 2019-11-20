import re
from minitools.db.mongodb import get_mongodb_client
from collections import Counter
from pprint import pprint
import json
import jieba
from minitools.ml.naivebayes import NaiveBayes
import numpy as np
def cutTitle(title):
    return [i for i in list(jieba.cut(title)) if len(i.strip()) > 1]
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
    for new in cza.find({}, {'_id': 0,'label': 1, '标题': 1}):
        sdict[new['label']] = sdict.get(new['label'], 0) + 1
    pprint(sdict)
    # with open('keyCounter.json', 'w') as f:
    #     f.write(json.dumps(sdict, ensure_ascii=False))
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
    with open('newKeyCounter.json', 'r') as f:
        json_data = json.loads(f.read())
    pprint(json_data)
    values = json_data.values()
    print(sum(values))
def test6():
    cza = get_mongodb_client()['news']['train']
    titles = set()
    for new in cza.find({}, {'_id': 0, 'label': 1, '标题': 1}):
        titles |= set(cutTitle(new['标题']))
    pprint(titles)
    # print(len(titles))  # 501878
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
    vector = np.array([1 for _ in range(len(json_data))])
    count = len(json_data)
    vector = vector / count
    cza = get_mongodb_client()['news']['train']
    aaaa = 0
    for key in keyCounter:
        if aaaa == 2:
            break
        if cza.count({'label': key}) > 100:
            continue
        aaaa += 1
        # vectorSelf = np.array([0 for _ in range(len(json_data))])
        index_dict = {}
        for new in cza.find({'label': key}, {'_id': 0, 'label': 1, '标题': 1}):
            # arr = np.array(NaiveBayes.set2vector(cutTitle(new['标题']), json_data))
            # vector = [0 for _ in range(len(json_data))]
            for word in cutTitle(new['标题']):
                if word in json_data:
                    index_dict[json_data.index(word)] = index_dict.get(json_data.index(word), 0) + 1
        print(index_dict)



        #     arr = 0
        #     vectorSelf += arr
        #     count += np.sum(arr)
        # res = list(np.log(vectorSelf/count + vector))
        # with open('train.txt', 'a+') as f:
        #     f.write(json.dumps(res, ensure_ascii=False)+','+key)
        # aaaa += 1
        # print(aaaa, key)


if __name__ == '__main__':
    # test1()
    # test2()
    # test3()
    # test4()
    # test5()
    # test6.()
    # test7()
    test8()


    # a = np.array([1,2,3,4,5])
    # b = np.array([1, 2, 3, 4, 5])
    # print(a + b)
    # print(np.mat(a) + np.mat(b))