# 清洗 去重 去停用词 分词
import json
import pandas as pd
import jieba

# 清洗未填写评论的数据
def clean_blank(comment_list):
    after_clean_list = []
    result = 0
    for i in range(len(comment_list)):
        comment = comment_list[i]
        if comment['comment']['content'] != '此用户没有填写评价内容' and comment['comment']['content'] != '买家没有填写评价内容！':
            good_id = comment['good_id']+'-'+comment['shop_id']
            content = comment['comment']['content'].replace('\n', '')
            score = comment['comment']['qualityStar']
            out = good_id+"||"+content.strip()+"||"+str(score)
            after_clean_list.append(out)
            result += 1
    print(str(result)+" items have been cleaned.")
    return after_clean_list

# 去重
def remove_duplicated(comment_list):
    # columns = ['good_id', 'content', 'score']
    good_id_list = []
    content_list = []
    score_list = []
    after_remove_dup_list = []
    for comment in comment_list:
        items = comment.split('||')
        good_id_list.append(items[0])
        content_list.append(items[1])
        score_list.append(items[2])
    dic = {'good_id': good_id_list, 'content': content_list, 'score': score_list}
    frame = pd.DataFrame(dic)
    new_frame = frame.drop_duplicates(['good_id', 'content'])
    i = 0
    for indexs in new_frame.index:
        item = new_frame.loc[indexs].values
        out = item[0]+"||"+item[2]+"||"+item[1]
        i = i+1
        after_remove_dup_list.append(out)
    print(str(i)+" removed duplicated.")
    return after_remove_dup_list


# 创建停用词列表
def stopwordslist():
    stopwords = []
    stopwords_file = open('stopwords.txt', encoding='utf-8')
    for line in stopwords_file.readlines():
        stopwords.append(line.strip())
    stopwords_file.close()
    return stopwords

# 分词
def fenci(stopwords, comment_list):
    result_list = []
    i = 0
    for line in comment_list:
        content = line.split('||')[0]
        seg_list = jieba.cut(content, cut_all=False)
        remove_stop_list = []
        for word in seg_list:
            if word not in stopwords:
                remove_stop_list.append(word)
        i = i + 1
        result = " ".join(remove_stop_list)
        result_list.append(result+"\n")
    print(str(i)+" items have been finished cut.")
    return result_list


comment_list = []
directory = 'D:\\Programming\\python\\lstm\\commentData\\'
filename = directory + 'sn_middle_2.jsonlines'
file = open(filename,'rb')
for line in file.readlines():
    c = line[:-1]
    comment_json = json.loads(c)
    comment_list.append(comment_json)
file.close()
after_clean = clean_blank(comment_list)
after_remove_dup = remove_duplicated(after_clean)
stopwords = stopwordslist()
out_list = fenci(stopwords, after_remove_dup)

out_filename = 'preProcessFinished/bad.txt'
out_file = open(out_filename, 'a', encoding='utf-8')
for line in out_list:
    out_file.write(line)
out_file.close()
