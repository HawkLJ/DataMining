import fasttext
from sklearn.metrics import classification_report

classifier = fasttext.load_model("model_key_news.bin")      # 加载模型
y_true = []        # 真实label
y_pred = []     # 预测label
with open("./key_data/test.txt", encoding='utf-8') as fr:
    for line in fr:
        cur_true_label = line.split(" ")[0]
        y_true.append(cur_true_label)
        sentence = line.replace(cur_true_label+' ', '').rstrip()
        y_pred.append(classifier.predict(sentence)[0][0])

target_names=['news_agriculture','news_car','news_culture','news_edu','news_entertainment','news_finance','news_game','news_house','news_military','news_sports','news_stock','news_story','news_tech','news_travel','news_world']
print(classification_report(y_true,y_pred, target_names=target_names))