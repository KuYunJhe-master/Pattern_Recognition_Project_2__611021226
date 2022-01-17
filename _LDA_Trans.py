#------------- 做LDA轉換 ---------------------------------------------------------------
import _LDA

def LDA_TRANSFORM(Data_train , Data_test , Label_train):
    print("====================================== LDA 轉換  STAET =======================================")

    Data_train_LDA_all_dim = []  #存放降維後各維度的 【數據訓練集LDA降維結果】
    Data_test_LDA_all_dim = [] #存放降維後各維度的 【數據測試集LDA降維結果】

    set_LDA_dimeniton = 0

    #丟進LDA做轉換，得到 【數據訓練集LDA轉換結果】和【數據測試集LDA轉換結果】
    Data_train_LDA , Data_test_LDA = _LDA.DO_LDA(Data_train , Data_test , Label_train , set_LDA_dimeniton*10)
    print("轉換後 資料維度 ---   Data_train =", Data_train_LDA.shape , " || Data_test =", Data_test_LDA.shape , "========= 轉換 SUCCESS")



    # for set_LDA_dimeniton in range(1,6): #做 多種維度的降維
    #     #丟進LDA做轉換，得到該維度的 【數據訓練集LDA轉換結果】和【數據測試集LDA轉換結果】
    #     Data_train_LDA , Data_test_LDA = DO_LDA(Data_train , Data_test , Label_train , set_LDA_dimeniton*10)
    #     print("Data_train資料維度 ---", Data_train_LDA.shape , " ||| Data_test資料維度 ---", Data_test_LDA.shape)
    #     Data_train_LDA_all_dim.append(Data_train_LDA) #收集各維度的 【數據訓練集LDA轉換結果】
    #     Data_test_LDA_all_dim.append(Data_test_LDA)  #收集各維度的 【數據測試集LDA轉換結果】

    print("========================================  FINISH  ============================================\n\n")

    return Data_train_LDA , Data_test_LDA , set_LDA_dimeniton