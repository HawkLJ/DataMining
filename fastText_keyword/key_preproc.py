import pkuseg

stopwords = [line.strip() for line in open('mystopwords.txt', encoding="utf-8").readlines()]        #加载停用词

f = open("./tnews_public/train.json", encoding='utf-8')
lines = f.readlines()
seg = pkuseg.pkuseg()
new_data = []
i = 1
for line in lines:
    dic = eval(line)
    label = '__label__' + dic['label_desc'].split('_')[1]
    sentence = [i for i in seg.cut(dic['sentence']) if i not in stopwords]
    new_line = label + ' ' + ' '.join(sentence) + ' ' + dic['keywords'].replace(',', ' ')
    new_data.append(new_line)
    # i = 1 + i
    # if i == 10:
    #     for k in range(len(new_data)):
    #         print(new_data[k])
    #     break
f.close()

f = open('./key_data/train.txt', 'w', encoding='utf-8')
for item in new_data:
    f.write(item)
    f.write('\n')
f.close()
