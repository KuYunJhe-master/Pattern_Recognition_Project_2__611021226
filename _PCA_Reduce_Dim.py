#-------------- 用PCA把2個數據集降成5種維度---------------------------------------------------------------
import _PCA

def PCA_REDUCE_DIMENSION(Data_train , Data_test):

    print("======================================= PCA 降維  STAET ========================================")
    Data_train_PCA_all_dim = [] #存放降維後各維度的 【數據訓練集PCA降維結果】
    Data_test_PCA_all_dim = [] #存放降維後各維度的 【數據測試集PCA降維結果】

    for set_PCA_dimeniton in range(1,6): #做 10、20、30、40、50 維度的降維

        #丟進PCA做降維，得到該維度的 【數據訓練集PCA降維結果】和【數據測試集PCA降維結果】
        Data_train_PCA , Data_test_PCA = _PCA.DO_PCA(Data_train , Data_test , set_PCA_dimeniton*10) 

        # print(data_face_PCA.shape)
        # print(data_unfac_PCAe.shape)
        print("降維後 資料維度 ---   Data_train =", Data_train_PCA.shape , " || Data_test =", Data_test_PCA.shape , "========= 降維 SUCCESS")

        Data_train_PCA_all_dim.append(Data_train_PCA) #收集各維度的 【數據訓練集PCA降維結果】
        Data_test_PCA_all_dim.append(Data_test_PCA) #收集各維度的 【數據測試集PCA降維結果】
    print("==========================================  FINISH  ============================================\n\n")
    return Data_train_PCA_all_dim , Data_test_PCA_all_dim , set_PCA_dimeniton