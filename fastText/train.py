import fasttext

# 52.26
# ngram 2 5318
# ngram 3 5264

model = fasttext.train_supervised(input="./data/train.txt", epoch=9, wordNgrams=2)
model.save_model("model_news.bin")
# ans = model.predict("哈登 和 保罗 就 像 一见钟情")
# print(ans)
ans = model.test("./data/test.txt")
print(ans)

