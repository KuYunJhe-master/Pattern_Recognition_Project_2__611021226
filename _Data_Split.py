#-------------- 圖片資料與標籤集 分組 ---------------------------------------------------------------
from sklearn.model_selection import train_test_split
import numpy as np

def DATA_AND_LABEL_SPLIT(img_face_10304x200 , target_face ,  img_unface_10304x200 , target_unface):
    print("================ 資料分組  STAET ================")

    Data =  np.append(img_face_10304x200 , img_unface_10304x200,axis= 0) #合併人臉與非人臉的資料
    Label = np.append(target_face , target_unface) #合併人臉與非人臉的標籤

    #分割訓練用的資料集 --- 數據訓練集、數據測試集、標籤訓練集、標籤測試集
    Data_train , Data_test , Label_train , Label_test = train_test_split(Data,Label,test_size = 0.3,stratify = Label) 

    print("數據訓練集 SIZE ----- Data_train  = " , Data_train.shape)
    print("數據測試集 SIZE ----- Data_test   = " , Data_test.shape)
    print("標籤訓練集 SIZE ----- Label_train = " , Label_train.shape)
    print("標籤訓練集 SIZE ----- Label_test  = " , Label_test.shape)

    print("====================  FINISH  ===================\n\n")

    return Data_train , Data_test , Label_train , Label_test
