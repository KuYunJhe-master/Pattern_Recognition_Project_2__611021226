#------------ 做 LDA 轉換資料的貝斯分類器---------------------------------------------------------------
import _Classifier

def DO_LDA_DATA_CLASSIFY(Data_train_LDA , Data_test_LDA, Label_train , Label_test):
    print("========================= LDA 分類辨識 STAET ==========================")

    print("------------------------------------  Dimention ""X (LDA) ---------------")
    #丟入貝氏分類器做分類判別
    LDA_recog_ans , LDA_correct_ans = _Classifier.DO_RECOG(Data_train_LDA , Data_test_LDA, Label_train , Label_test)
    
    print("============================  FINISH  =================================\n\n")
    return LDA_recog_ans , LDA_correct_ans