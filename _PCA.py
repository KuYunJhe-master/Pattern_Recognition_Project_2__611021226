#--------------【設定】 PCA  回傳 【轉換後訓練資料集】 &　【轉換後測試資料集】---------------------------------------------------------------
from sklearn.decomposition import PCA

def DO_PCA(Data_train , Data_test , set_dim):
    pca = PCA(n_components = set_dim)#設定PCA降維維度
    pca.fit(Data_train)#訓練PCA降維train資料集
    # pca.fit(Data_test)#訓練PCA降維test資料集
    transf_Data_train = pca.transform(Data_train)#得到PCA的train資料集降維投影轉換
    transf_Data_test = pca.transform(Data_test)#得到PCA的test資料集降維投影轉換

    return transf_Data_train , transf_Data_test  #回傳 轉換後訓練資料集 &　轉換後測試資料集