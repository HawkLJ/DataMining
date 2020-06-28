import torch
from config import parse_config
from data_loader import DataBatchIterator
from sklearn.metrics import classification_report
def main():
    # 读配置文件
    config = parse_config()
    # 载入测试集
    test_data = DataBatchIterator(
        config=config,
        is_train=False,
        dataset="test",
        batch_size=config.batch_size)
    test_data.load()
    # 加载模型
    model = torch.load(config.save_model+".pt",
                         map_location = config.device)
    model.eval()
    test_data_iter = iter(test_data)
    y_pred=[]# 预测值
    y_true=[]# 真实标签
    for idx, batch in enumerate(test_data_iter):
        outputs = model(batch.sent)
        pred_each=torch.max(outputs,1)[1].numpy().tolist()
        true_each=batch.label.numpy().tolist()
        y_pred=y_pred + pred_each
        y_true=y_true + true_each
    target_names=['news_edu','news_finance','news_house','news_travel','news_tech','news_sports','news_game','news_culture','news_car','news_story','news_entertainment','news_tech','news_agriculture','news_world','news_stock']
    print(classification_report(y_true,y_pred,target_names=target_names))

if __name__ == "__main__":
    main()