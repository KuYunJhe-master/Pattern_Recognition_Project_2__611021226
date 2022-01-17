#--------------【設定】 貝氏分類器 回傳 判定結果集 和 正確答案集---------------------------------------------------------------
from sklearn.naive_bayes import GaussianNB

def DO_RECOG(Data_train , Data_test , Target_train , Target_test):

    model = GaussianNB()#建立高斯貝氏分類器模型
    model.fit(Data_train , Target_train)#訓練
    recog_anwser = model.predict(Data_test)#測試
    # print("recog_anwser" , recog_anwser)#個別判定結果

    # recog_proba_anwser = model.predict_proba(Data_test)#test集個別判定機率
    train_score = model.score(Data_train , Target_train)#訓練集正確率
    test_score = model.score(Data_test , Target_test)#測試集正確率

    print("訓練集正確率 = ",  train_score)#輸出訓練集正確率
    print("測試集正確率 = ",  test_score) #輸出測試集正確率

    return recog_anwser , Target_test #回傳 判定結果集 和 正確答案集
