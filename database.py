import re
from minitools.db.mongodb import get_mongodb_client
from pprint import pprint
import json
import jieba


def cutTitle(title):
    return [i for i in jieba.lcut(title) if len(i.strip()) > 1]


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
    new_data = {}
    for key, value in json_data.items():
        jump = False
        for fil in ['伊春', '文昌', '长治', '咸阳', '嘉兴', '贵州', '外埠', '阳泉', '盐城', '昌平',
                    '孙大军', '安顺', '宝坻', '山西', '巴中', '怀柔', '襄阳', '驻芜', '驻柳', '首页其他', '馆务', '馆藏',
                    '陵水', '南宁', '安阳', '乌海', '黎苗', '巴州', '盘锦', '凉都', '视频', '临夏', '柳州', '下载', '相关新闻',
                    '公开文件', '白沙', '顺义', '芜湖', '龙湾', '兴安', '兴谷', '衡水', '专题集锦', '河南', '走进', '图片',
                    '图表', '六个如何', '创模', '精彩', '他山之石', '好人', '焦作', '智库', '留言选登', '黔府', '两会',
                    '区人防办', '各地创建', '南开', '两会', '国有企业', '三变', '杨勤荣说', '项目介绍', '注销撤销', '网站首页',
                    '豫风楚韵', '澄迈', '桂林', '宜居生活', '百姓问政', '价格信息', '文件', '煤化', 'Home', '投资平台', '延庆',
                    '佳作欣赏', '代表建议', '信阳', '特色节目', '赛事动态', '主动发声', '创卫', '宿迁', '创建要闻', '创卫要闻']:
            if fil in key:
                jump = True
                break
        if jump:
            continue
        if value >= 50:
            new_data[key] = value
    print(f"{len(json_data)} => {len(new_data)}")
    with open('newKeyCounter.json', 'w') as f:
        f.write(json.dumps(new_data, ensure_ascii=False))


def test6():
    NO_D = re.compile('\d+').search
    cza = get_mongodb_client()['news']['train']
    titles = set()
    all_label = set()
    for new in cza.find({}, {'_id': 0, 'label': 1}):
        all_label.add(new['label'])
    for label in all_label:
        jump = False
        for fil in ['伊春', '文昌', '长治', '咸阳', '嘉兴', '贵州', '外埠', '阳泉', '盐城', '昌平',
                    '孙大军', '安顺', '宝坻', '山西', '巴中', '怀柔', '襄阳', '驻芜', '驻柳', '首页其他', '馆务', '馆藏',
                    '陵水', '南宁', '安阳', '乌海', '黎苗', '巴州', '盘锦', '凉都', '视频', '临夏', '柳州', '下载', '相关新闻',
                    '公开文件', '白沙', '顺义', '芜湖', '龙湾', '兴安', '兴谷', '衡水', '专题集锦', '河南', '走进', '图片',
                    '图表', '六个如何', '创模', '精彩', '他山之石', '好人', '焦作', '智库', '留言选登', '黔府', '两会',
                    '区人防办', '各地创建', '南开', '两会', '国有企业', '三变', '杨勤荣说', '项目介绍', '注销撤销', '网站首页',
                    '豫风楚韵', '澄迈', '桂林', '宜居生活', '百姓问政', '价格信息', '文件', '煤化', 'Home', '投资平台', '延庆',
                    '佳作欣赏', '代表建议', '信阳', '特色节目', '赛事动态', '主动发声', '创卫', '宿迁', '创建要闻', '创卫要闻']:
            if fil in label:
                jump = True
                break
        if jump:
            continue
        if cza.count({'label': label}) < 50:
            continue
        print(f'获取到: {label}')
        for new in cza.find({'label': label}, {'_id': 0, '标题': 1}):
            titles |= set([i for i in cutTitle(new['标题']) if not NO_D(i)])

    print(len(titles))  # 501878 => 106506 => 88991
    with open('allUniqueLabel.json', 'w') as f:
        f.write(json.dumps(list(titles), ensure_ascii=False))


def test8():
    with open('newKeyCounter.json', 'r') as f:
        keyCounter = list(json.loads(f.read()).keys())
    with open('allUniqueLabel.json', 'r') as f:
        json_data = json.loads(f.read())
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


def test9(title=""):
    wordTitle = title
    with open('allUniqueLabel.json', 'r') as f:
        json_data = json.loads(f.read())
    index_dict = {}
    for word in cutTitle(wordTitle):
        if word in json_data:
            index_dict[json_data.index(word)] = index_dict.get(json_data.index(word), 0) + 1
    wordLabel = None
    wordValue = 0
    with open('train.txt', 'r') as f:
        for content in f:
            json_data, label = content.strip().split('|')
            trainSet = json.loads(json_data)
            wordValueTemp = 0
            for key, value in index_dict.items():
                trainKey = trainSet.get(str(key), None)
                if trainKey:
                    wordValueTemp += value * trainKey
            if wordValueTemp > wordValue:
                wordValue = wordValueTemp
                wordLabel = '公示公告' if '公示' in label else label
                wordLabel = '其他信息' if '住房资讯' in label else label
                wordLabel = '其他信息' if '畜牧兽医' in label else label
        print(f"原标题：{wordTitle}")
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
                index = json_data.index(word)
                index_dict[index] = index_dict.get(index, 0) + 1
        wordLabel = None
        wordValue = 0
        with open('train.txt', 'r') as f:
            for content in f:
                train_json_data, label = content.strip().split('|')
                trainSet = json.loads(train_json_data)  # 得到训练数据
                wordValueTemp = 0
                for key, value in index_dict.items():
                    trainKey = trainSet.get(str(key), None)
                    if trainKey:
                        wordValueTemp += value * trainKey
                if wordValueTemp > wordValue:
                    wordValue = wordValueTemp
                    wordLabel = '公示公告' if '公示' in label else label
                    wordLabel = '其他信息' if '住房资讯' in label else label
                    wordLabel = '其他信息' if '畜牧兽医' in label else label
            print(f"原标题：{wordTitle}")
            print(f"金融分类：{doc['金融分类']}")
            print(f"原label：{doc['label']}")
            print(f"预测分类为：{wordLabel}")
            if doc['label'] == wordLabel:
                print("预测正确\n")
                right_num += 1
            else:
                print("预测错误\n")
    print(f"正确率为：{100 * right_num / all_num}%")


def test11():
    with open('allUniqueLabel.json', 'r') as f:
        json_data = json.loads(f.read())
    client = get_mongodb_client()['news']['test']
    for doc in client.find({}):
        wordTitle = doc['标题']
        index_dict = {}
        for word in cutTitle(wordTitle):
            if word in json_data:
                index = json_data.index(word)
                index_dict[index] = index_dict.get(index, 0) + 1
        wordLabel = None
        wordValue = 0
        with open('train.txt', 'r') as f:
            for content in f:
                train_json_data, label = content.strip().split('|')
                trainSet = json.loads(train_json_data)  # 得到训练数据
                wordValueTemp = 0
                for key, value in index_dict.items():
                    trainKey = trainSet.get(str(key), None)
                    if trainKey:
                        wordValueTemp += value * trainKey
                if wordValueTemp > wordValue:
                    wordValue = wordValueTemp
                    wordLabel = '公示公告' if '公示' in label else label
                    wordLabel = '其他信息' if '住房资讯' in label else label
                    wordLabel = '其他信息' if '畜牧兽医' in label else label
        print(f"v1:{doc['v1-label']}-v2:{doc['v2-label']}-v3:{wordLabel}")
        client.update_one({'_id': doc['_id']}, {'$set': {'v3-label': wordLabel}})
    client.close()


if __name__ == '__main__':
    # test2()
    # test3()
    # test4()
    # test6()
    # test8()

    # for title in [
    #     '临夏河州大道—北滨河大道中压天然气管道 工程竣工并正式通气',
    #     '关于陈光荣等同志任职的通知',
    #     '民政部社会救助司调研指导临夏州脱贫攻坚',
    #     '姚逊在武乡县调研时强调抓好基层党建 助推脱贫攻坚',
    #     '州人民政府第32次常务会议',
    #     '州人民政府第30次常务会议',
    #     '长治市脱贫攻坚简报总第46期',
    #
    #     '大九湖镇：狠抓一线落实 强化责任担当 全力以赴做好迎检工作',
    #     '宋洛乡：党风廉政建设 “三治融合”显实效',
    #     '中共神农架林区委员会组织部干部任前公示公告（2019年第009号)',
    #     '组织部干部任前公示公告',
    # ]:
    #     test9(title)

    # test10(100)
    test11()
