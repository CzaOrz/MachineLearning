import re
from minitools.db.mongodb import get_mongodb_client


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
        cza.update_one({'_id': new['_id']}, {'$set': {'label': label}})
        count += 1
        sett.add(label)
    # print(len(sett), count, '????')  # 2343 501878 ????
    # for i in sett:
    #     print(i)


if __name__ == '__main__':
    # test1()
    test2()