#--------------【設定】 LDA  回傳 【轉換後訓練資料集】 &　【轉換後測試資料集】---------------------------------------------------------------
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


def DO_LDA(Data_train , Data_test , Label_train , set_dim):
    
    if set_dim == 0: #如果不降維 只轉換
        lda = LDA()
        lda.fit(Data_train , Label_train)#訓練LDA train資料集
        transf_Data_train = lda.transform(Data_train)#得到LDA的train資料集轉換投影
        transf_Data_test = lda.transform(Data_test)#得到LDA的test資料集轉換投影
    
    else: #如果降維 且轉換
        lda = LDA(n_components = set_dim)#設定LDA降維維度
        lda.fit(Data_train , Label_train)#訓練LDA train資料集
        transf_Data_train = lda.transform(Data_train)#得到LDA的train資料集轉換投影
        transf_Data_test = lda.transform(Data_test)#得到LDA的test資料集轉換投影
    
    return transf_Data_train,transf_Data_test #回傳 轉換後訓練資料集 &　轉換後測試資料集
