import fasttext

# 52.26
# ngram 2 5318
# ngram 3 5264

# 60.84
# for w in range(1, 6):
#     # model = fasttext.train_supervised(input="./key_data/train.txt", epoch=epoch, wordNgrams=2)
#     print("==>" + str(w))
#     model = fasttext.train_supervised(input="./key_data/train.txt", epoch=7, lr=0.1, wordNgrams=1)
#     model.save_model("model_key_news.bin")
#     # ans = model.predict("哈登 和 保罗 就 像 一见钟情")
#     # print(ans)
#     ans = model.test("./key_data/dev.txt")
#     print(ans)


model = fasttext.train_supervised(input="./key_data/train.txt", epoch=10, wordNgrams=2)
model.save_model("model_key_news.bin")
# ans = model.predict("哈登 和 保罗 就 像 一见钟情")
# print(ans)
ans = model.test("./key_data/test.txt")
print(ans)