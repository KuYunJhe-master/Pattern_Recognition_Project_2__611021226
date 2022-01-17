import _Read_FACE_image
import _Read_NOT_FACE_image
import _Data_Split
import _PCA_Reduce_Dim
import _Classify_PCA
import _LDA_Trans
import _Classify_LDA
import _DO_Confusion_Matrix


face_image_amount = 200 #--------------設定讀取來訓練的人臉照片的數量，需為40的倍數，最少80張，最多400張

not_face_image_amount = 200 #--------------設定讀取來訓練的非人臉照片的數量，需為20的倍數，最少20張，最多兩百張

face_img_data , face_img_label , img_face_SHAPE = _Read_FACE_image.READ_FACE_IMG(face_image_amount)#--------------處理人臉圖片輸入 轉換成個別圖片一維的樣本集

not_face_img_data , not_face_img_label = _Read_NOT_FACE_image.READ_NOT_FACE_IMG(img_face_SHAPE , not_face_image_amount)#--------------處理非人臉圖片輸入 轉換成個別圖片一維的樣本集

Data_train , Data_test , Label_train , Label_test = _Data_Split.DATA_AND_LABEL_SPLIT(face_img_data , face_img_label ,  not_face_img_data , not_face_img_label)#-------------- 圖片資料與標籤集分組

Data_train_PCA_all_dim , Data_test_PCA_all_dim , Current_PCA_dime = _PCA_Reduce_Dim.PCA_REDUCE_DIMENSION(Data_train , Data_test)#-------------- 用PCA把2個數據集降成5種維度

PCA_recog_ans , PCA_correct_ans = _Classify_PCA.DO_PCA_DATA_CLASSIFY(Data_train_PCA_all_dim , Data_test_PCA_all_dim , Label_train , Label_test)#------------- 做 PCA 5個降維資料的貝氏分類器

Data_train_LDA , Data_test_LDA , Current_LDA_dime = _LDA_Trans.LDA_TRANSFORM(Data_train , Data_test , Label_train)#------------- 做LDA轉換

LDA_recog_ans , LDA_correct_ans = _Classify_LDA.DO_LDA_DATA_CLASSIFY(Data_train_LDA , Data_test_LDA, Label_train , Label_test)#------------ 做 LDA 轉換資料的貝斯分類器

_DO_Confusion_Matrix.DO_CONFUSI_MATRIX(PCA_correct_ans , PCA_recog_ans , Current_PCA_dime , LDA_correct_ans , LDA_recog_ans , Current_LDA_dime)#------------做混淆矩陣

