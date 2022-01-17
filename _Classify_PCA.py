#------------- 做 PCA 5個降維資料的貝氏分類器---------------------------------------------------------------
import _Classifier

def DO_PCA_DATA_CLASSIFY(Data_train_PCA_all_dim , Data_test_PCA_all_dim , Label_train , Label_test):
    print("====================== PCA 分類辨識 STAET =======================")
    PCA_recog_ans = [] #存放 【各個經PCA降維辨識結果】
    PCA_correct_ans = [] #存放 【各個經PCA降維正確答案】
    for cur_dimeniton in range(5):
        print("------------------------------------  Dimention "+ str(cur_dimeniton+1) + "0 --------------")

        #丟入貝氏分類器做分類判別
        cur_dim_recog_anwser , cur_dim_correct_answer = _Classifier.DO_RECOG(Data_train_PCA_all_dim[cur_dimeniton], Data_test_PCA_all_dim[cur_dimeniton], Label_train , Label_test)
        
        PCA_recog_ans.append(cur_dim_recog_anwser) #收集 【各個經PCA降維辨識結果】
        PCA_correct_ans.append(cur_dim_correct_answer) #收集 【各個經PCA降維正確答案】
    
    print("===========================  FINISH  ============================\n\n")
    return PCA_recog_ans , PCA_correct_ans