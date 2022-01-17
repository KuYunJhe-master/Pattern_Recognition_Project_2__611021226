---
title: 圖型識別作業二
tags: 圖型識別
disqus: None
---

圖型識別
===


## 作業二   人臉辨識
學號：611021226
姓名：古昀哲

---
### 題目概述

1. 設計程式一個讀取資料庫pgm圖片。
2. 圖片轉灰階轉成10304x1的向量成為一個樣本。
3. 利用PCA計算樣本的轉換矩陣，降維至50、40、30、20、10維。
4. 用降維後的樣本來做分類器辨識。
5. 計算不同維度辨識結果的混淆矩陣。
6. 利用FLD(LDA)計算另一轉換矩陣，不降維。
7. 對此樣本矩陣重複第4和第5步驟。
---
### 使用套件
* Scikit-learn
* OpenCV
* Numpy
---
### 方法簡述

主要使用的方法主要可分成以下步驟，並區分不同功能或步驟，放到各別的python檔中執行

:::info
1. 讀取【是人臉】圖片(總圖片數400)，回傳圖片轉換成樣本的數據集
2. 讀取【非人臉】圖片(總圖片數200)，回傳圖片轉換成樣本的數據集
3. 分割資料成 -> 數據訓練集、數據測試集合、標籤訓練集、標籤測試集，以利PCA、LDA和分類器使用
4. 用迴圈設定PCA降維維度，傳入PCA降維，從單個圖片樣本的向量10301 降至 50、40、30、20、10維
5. 將五種維度的PCA轉換矩陣傳入貝氏分類器做分類辨識
6. 設定LDA轉換，傳入訓練的樣本資料集
7. 將LDA回傳的樣本轉換矩陣傳入貝氏分類器做分類辨識
8. 將貝氏分類器的回傳結果傳入副程式做混淆矩陣
:::

以上步驟的副程式透過主程式逐一呼叫並運行，以下則將重複使用的部分整合到同一個副程式
:::warning
* 呼叫使用PCA (fit、transform、降維)
* 呼叫使用LDA (fit、transform、轉換or降維+轉換)
* 呼叫使用貝氏分類器 (建model、fit、訓練、輸出正確率)
* 呼叫使用混淆矩陣(可單次處理一個樣本集或是一次處理多個樣本集)
:::

下面再各別解釋程式碼的部分功能

---
#### 【_main】的主程式中

:::info
主要用來設應最重要的三個變數
face_image_amount ：用來設定人臉圖片數量
not_face_image_amount ：用來設定非人臉圖片數量
test_set_scale：用來設定訓練集與測試集的比例
其餘部分為依序呼叫個副程式
:::spoiler 點這裡看程式
```python=
face_image_amount = 200 #--------------設定讀取來訓練的人臉照片的數量，需為40的倍數，最少80張，最多400張

not_face_image_amount = 200 #--------------設定讀取來訓練的非人臉照片的數量，需為20的倍數，最少20張，最多兩百張

test_set_scale = 0.3 #--------------設定訓練集與測試集的比例，以0~1之間限，數字以測試集為標準

face_img_data , face_img_label , img_face_SHAPE = _Read_FACE_image.READ_FACE_IMG(face_image_amount)#--------------處理人臉圖片輸入 轉換成個別圖片一維的樣本集

not_face_img_data , not_face_img_label = _Read_NOT_FACE_image.READ_NOT_FACE_IMG(img_face_SHAPE , not_face_image_amount)#--------------處理非人臉圖片輸入 轉換成個別圖片一維的樣本集

Data_train , Data_test , Label_train , Label_test = _Data_Split.DATA_AND_LABEL_SPLIT(face_img_data , face_img_label ,  not_face_img_data , not_face_img_label , test_set_scale)#-------------- 圖片資料與標籤集分組

Data_train_PCA_all_dim , Data_test_PCA_all_dim , Current_PCA_dime = _PCA_Reduce_Dim.PCA_REDUCE_DIMENSION(Data_train , Data_test)#-------------- 用PCA把2個數據集降成5種維度

PCA_recog_ans , PCA_correct_ans = _Classify_PCA.DO_PCA_DATA_CLASSIFY(Data_train_PCA_all_dim , Data_test_PCA_all_dim , Label_train , Label_test)#------------- 做 PCA 5個降維資料的貝氏分類器

Data_train_LDA , Data_test_LDA , Current_LDA_dime = _LDA_Trans.LDA_TRANSFORM(Data_train , Data_test , Label_train)#------------- 做LDA轉換

LDA_recog_ans , LDA_correct_ans = _Classify_LDA.DO_LDA_DATA_CLASSIFY(Data_train_LDA , Data_test_LDA, Label_train , Label_test)#------------ 做 LDA 轉換資料的貝斯分類器

_DO_Confusion_Matrix.DO_CONFUSI_MATRIX(PCA_correct_ans , PCA_recog_ans , Current_PCA_dime , LDA_correct_ans , LDA_recog_ans , Current_LDA_dime)#------------做混淆矩陣

```
:::




#### 【_Read_FACE_image】的副程式中

:::success
在主程式中可以指定要用來訓練的【人臉】和【非人臉】的圖片數量，此部分就是用來判斷數量是否是40的倍數或是有沒有超過資料庫的圖片總數，如果有數量設定錯誤，則先停止程式並輸出錯誤訊息。
數量上限是400，下限80，須以40為倍數來設定。
:::spoiler 點這裡看程式
```python=
if ((face_image_amount % 40) != 0) and (face_image_amount <= 400): #判斷指定的臉部圖片是否數量正確
        print("ERROR: Face image amount should be multiples of 40 ! 臉部圖片數量必須是40的倍數!\n") #不正確就回傳錯誤訊息然後結束程式
        exit()
    else:
        max_img_face_idx = int(face_image_amount / 40) +1 #正確就計算要開啟的圖片最大位置
```
:::

:::success
直接把設定數量的人臉圖片的label寫好
:::spoiler 點這裡看程式
```python=
target_face = np.ones(shape = (img_face_10304x200.shape[0]),dtype=int)#把標【人臉】的標籤存放到標籤的矩陣
```
:::

#### 【_Read_NOT_FACE_image】的副程式中

:::success
也是一樣為了要檢查指定的圖片數量是否正確。
數量上限是200，下限20，須以20為倍數來設定。
:::spoiler 點這裡看程式
```python=
if ((not_face_image_amount % 20) != 0) and (not_face_image_amount <= 200): #判斷指定的臉部圖片是否數量正確
    print("ERROR: Face image amount should be multiples of 20 ! 臉部圖片數量必須是40的倍數!\n") #不正確就回傳錯誤訊息然後結束程式
    exit()
else:
    max_img_not_face_idx = not_face_image_amount #正確就計算要開啟的圖片最大位置
```
:::

:::success
因為人臉辨識的負樣本是另外找到的，所以這部分的程式是用來將非人臉圖片確定灰階化，轉為單頻道，並置中裁切出與人臉正樣本相同大小的樣本圖片。
:::spoiler 點這裡看程式
```python=
img_unface = cv2.cvtColor(img_unface, cv2.COLOR_BGR2GRAY)#確認將圖片轉為灰階並僅留一個色彩(灰階)頻道

if (img_unface.shape[0] >= img_face_SHAPE[0]) and (img_unface.shape[1] >= img_face_SHAPE[1]): #可裁切的大小要大於要裁切的大小
    ori_pit_Y_of_cutted_unface = ((img_unface.shape[0]//2) - (img_face_SHAPE[0]//2)) #設定裁切位置對齊圖片中央
    ori_pit_X_of_cutted_unface = ((img_unface.shape[1]//2) - (img_face_SHAPE[1]//2)) #設定裁切位置對齊圖片中央
    
#裁切圖片中央的 92*112 大小
img_unface = img_unface[ori_pit_Y_of_cutted_unface:ori_pit_Y_of_cutted_unface+img_face_SHAPE[1] , ori_pit_X_of_cutted_unface : ori_pit_X_of_cutted_unface+img_face_SHAPE[0]]
```
:::spoiler 補充
:::danger
非人臉圖片並非來自相同的資料庫，主要是取自[Segmentation evaluation database](https://www.wisdom.weizmann.ac.il/~vision/Seg_Evaluation_DB/dl.html)的非人臉圖片資料庫，裡面共有兩百張非人臉的各式各樣物品、場景的照片，取其作為訓練的負樣本。
:::
:::danger

除了上述裁切以外，已另外透過另外寫的程式去取出檔案和更改檔名，，所以在這個程式裡無法直接使用他的資料庫，只能使用我整理好的附屬的圖片資料庫。
:::

:::success
直接把設定數量的非人臉圖片的label寫好
:::spoiler 點這裡看程式
```python=
target_unface = np.zeros(shape = (img_unface_10304x200.shape[0]),dtype=int)#把標【不是人臉】的標籤存放到標籤的矩陣
```
:::

:::info
上面兩讀取圖片的副程式回傳的資料，都是已經被攤平的圖片，成為樣本並集合起來
:::

#### 【_Data_Split】的副程式中

:::success
在主程式中可以指定要設定的訓練集與測試集的比例
這裡用來檢查比例參數是否正確
:::spoiler 點這裡看程式
```python=
if (test_set_scale >= 1) or (test_set_scale  <= 0): #判斷指定的測試集與訓練集比例是否正確
    print("test_set_scale must be 0~1 ! 測試集與訓練集比例須為0~1之間 !\n") #不正確就回傳錯誤訊息然後結束程式
    exit()
```
:::

:::success
把人臉的正負樣本合併成同一個樣本集
正負樣本標籤也合併成一個標籤集
:::spoiler 點這裡看程式
```python=
Data =  np.append(img_face_10304x200 , img_unface_10304x200,axis= 0) #合併人臉與非人臉的資料
Label = np.append(target_face , target_unface) #合併人臉與非人臉的標籤
```
:::

:::success
把數據Data和標籤Label分割成數據訓練集、數據測試集合、標籤訓練集、標籤測試集
以70%設定為訓練用，30%作為測試用
依照標籤集內的正負樣本數量比例來區分
:::spoiler 點這裡看程式
```python=
Data_train , Data_test , Label_train , Label_test = train_test_split(Data,Label,test_size = 0.3,stratify = Label) 
```
:::



#### 【_PCA_Reduce_Dim】的副程式中

:::success
用迴圈一次處理多個降維維度的PCA
傳入設定好的副程式去執行PCA
把各個維度的回傳的轉換矩陣存到 Data_train_PCA_all_dim 跟 Data_test_PCA_all_dim
:::spoiler 點這裡看程式
```python=
for set_PCA_dimeniton in range(1,6): #做 10、20、30、40、50 維度的降維
    #丟進PCA做降維，得到該維度的 【數據訓練集PCA降維結果】和【數據測試集PCA降維結果】
    Data_train_PCA , Data_test_PCA = _PCA.DO_PCA(Data_train , Data_test , set_PCA_dimeniton*10) 

    Data_train_PCA_all_dim.append(Data_train_PCA) #收集各維度的 【數據訓練集PCA降維結果】
    Data_test_PCA_all_dim.append(Data_test_PCA) #收集各維度的 【數據測試集PCA降維結果】
```
:::

#### 【_PCA】的副程式中

:::warning
使用 sklearn.decomposition 的 PCA
來設定降維維度、訓練、和轉換，得到轉換矩陣
:::spoiler 點這裡看程式
```python=
pca = PCA(n_components = set_dim)#設定PCA降維維度
pca.fit(Data_train)#訓練PCA降維train資料集
transf_Data_train = pca.transform(Data_train)#得到PCA的train資料集降維投影轉換
transf_Data_test = pca.transform(Data_test)#得到PCA的test資料集降維投影轉換
```
:::

#### 【_Classify_PCA】的副程式中

:::success
用迴圈一次處理PCA轉換矩陣的分類器辨識
訓練集和測試集的轉換矩陣，傳入設定好的貝氏分類器
把各個維度的回傳的辨識結果存到 PCA_recog_ans 
相對應的答案回傳到 PCA_correct_ans
:::spoiler 點這裡看程式
```python=
for cur_dimeniton in range(5):
    #丟入貝氏分類器做分類判別
    cur_dim_recog_anwser , cur_dim_correct_answer = _Classifier.DO_RECOG(Data_train_PCA_all_dim[cur_dimeniton], Data_test_PCA_all_dim[cur_dimeniton], Label_train , Label_test)
    PCA_recog_ans.append(cur_dim_recog_anwser) #收集 【各個經PCA降維辨識結果】
    PCA_correct_ans.append(cur_dim_correct_answer) #收集 【各個經PCA降維正確答案】
```
:::

#### 【_LDA_Trans】的副程式中

:::success
LDA不須降維，只要轉換即可
但比較特別的是他是監督式的算法，需要訓練
所以必須也把標籤訓練集也一起丟進去才能運作模組訓練
:::spoiler 點這裡看程式
```python=
LDA_recog_ans , LDA_correct_ans = _Classifier.DO_RECOG(Data_train_LDA , Data_test_LDA, Label_train , Label_test)
```
:::

#### 【_LDA】的副程式中

:::warning
一樣使用 sklearn.discriminant_analysis 的 LinearDiscriminantAnalysis
來設定訓練和轉換，得到轉換矩陣
為了讓如果有要降維的話也可以使用，所以寫了判斷式，只要有指定的維度傳入就會啟動
但外部不需要呼叫這部分，所以是沒有運行的狀態
:::spoiler 點這裡看程式
```python=
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
```
:::

#### 【_Classify_LDA】的副程式中

:::success
將LDA轉換好的矩陣丟到設定好的貝氏分類器，只有一種維度，就不需要迴圈處理了
回傳的辨識結果存到 LDA_recog_ans
相對應的答案回傳到 LDA_correct_ans
:::spoiler 點這裡看程式
```python=
LDA_recog_ans , LDA_correct_ans = _Classifier.DO_RECOG(Data_train_LDA , Data_test_LDA, Label_train , Label_test)
```
:::

#### 【_Classifier】的副程式中

:::warning
使用 sklearn.naive_bayes 的 GaussianNB
來設定訓練高斯貝氏分類器
這裡所有的數據訓練集、數據測試集、標籤訓練集、標籤測試集都會使用到了
回傳辨識的結果和對應的答案
:::spoiler 點這裡看程式
```python=
model = GaussianNB()#建立高斯貝氏分類器模型
model.fit(Data_train , Target_train)#訓練
recog_anwser = model.predict(Data_test)#測試
train_score = model.score(Data_train , Target_train)#訓練集正確率
test_score = model.score(Data_test , Target_test)#測試集正確率
```
:::

#### 【_DO_Confusion_Matrix】的副程式中

:::success
把各維度PCA和LDA轉換矩陣從貝氏分類器出來的辨識結果和對應的答案
丟到設定的混淆矩陣副程式裡面
來檢視辨識的成效
:::spoiler 點這裡看程式
```python=
_Confusion_Matrix.DO_Confusi_Matrix(PCA_correct_ans , PCA_recog_ans , set_PCA_dimeniton)  
_Confusion_Matrix.DO_Confusi_Matrix(LDA_correct_ans , LDA_recog_ans , set_LDA_dimeniton)
```
:::


#### 【_Confusion_Matrix】的副程式中

:::warning
為了要應對PCA一次有多種維度要計算
而LDA只有一個轉換矩陣
所以在混淆矩陣這關寫了判斷
這裡混淆矩陣使用 sklearn.metrics 的 confusion_matrix
:::spoiler 點這裡看程式
```python=

if count_dimeniton == 0:  #如果要處裡單個辨識結果的混淆矩陣
    cur_dim_confus_mix = confusion_matrix(correct_answer, recog_anwser)  
else:
    for cur_dimeniton in range(5):    #如果要一次處裡多個不同維度辨識結果的混淆矩陣
        cur_dim_confus_mix = confusion_matrix(correct_answer[cur_dimeniton], recog_anwser[cur_dimeniton])
```
:::



---

### 辨識結果


:::info
**人臉圖片數量：200
非人臉圖片數量：200
訓練集測試集比： 7:3**


**PCA x 高斯貝氏分類器** : 五個經PCA降維的矩陣辨識結果
|  Dimention   |    Dimention 10    | Dimention 20 |    Dimention 30    |    Dimention 40    |    Dimention 50    |
|:------------:|:------------------:|:------------:|:------------------:|:------------------:|:------------------:|
| 訓練集正確率 | 0.9964285714285714 |     0.95     | 0.9321428571428572 | 0.9214285714285714 | 0.9142857142857143 |
| 測試集正確率 | 0.9833333333333333 |    0.975     | 0.9416666666666667 | 0.9416666666666667 | 0.9166666666666666 |


**LDA x 高斯貝氏分類器** : 經LDA的轉換的辨識結果


|            |       LDA       |
|:----------:|:------------------:|
| 訓練集正確率 | 0.9964285714285714 |
| 測試集正確率 |       0.875        |


---

 **PCA  混淆矩陣**

 **Dimention 10**
 
| P\A  | 非臉 | 是臉 |
|:----:|:----:|:----:|
| 非臉 |  60  |  0   |
| 是臉 |  2   |  58  |


 **Dimention 20**

| P\A  | 非臉 | 是臉 |
|:----:|:----:|:----:|
| 非臉 |  57  |  3   |
| 是臉 |  0   |  60  |

 **Dimention 30**

| P\A  | 非臉 | 是臉 |
|:----:|:----:|:----:|
| 非臉 |  53  |  7   |
| 是臉 |  0   |  60  |

 **Dimention 40**

| P\A  | 非臉 | 是臉 |
|:----:|:----:|:----:|
| 非臉 |  53  |  7   |
| 是臉 |  0   |  60  |


 **Dimention 50**
| P\A  | 非臉 | 是臉 |
|:----:|:----:|:----:|
| 非臉 |  50  |  10   |
| 是臉 |  0   |  60  |



**LCD 混淆矩陣**

| P\A  | 非臉 | 是臉 |
|:----:|:----:|:----:|
| 非臉 |  45  |  15   |
| 是臉 |  0   |  60  |

:::

:::warning
**人臉圖片數量：200
非人臉圖片數量：200
訓練集測試集比： 7:3**

:::spoiler 完整輸出結果
```
============== READ 【with Face】 IMG START =============
Image with Face AMOUNT  ------ 200
Image with Face Flatten DATA SIZE  ------ (200, 10304)
Image with Face Lable Size  ------------- (200,)
============= READ 【with Face】 IMG SUCCESS ============


============ READ 【without Face】 IMG START ============
Image without Face AMOUNT  ------ 200
Image without Face Flatten DATA SIZE  ------ (200, 10304)
Image without Face Lable Size  ------------- (200,)
=========== READ 【without Face】 IMG SUCCESS ===========


================ 資料分組  STAET ================
數據訓練集 SIZE ----- Data_train  =  (280, 10304)
數據測試集 SIZE ----- Data_test   =  (120, 10304)
標籤訓練集 SIZE ----- Label_train =  (280,)
標籤訓練集 SIZE ----- Label_test  =  (120,)
====================  FINISH  ===================


======================================= PCA 降維  STAET ========================================
降維後 資料維度 ---   Data_train = (280, 10)  || Data_test = (120, 10) ========= 降維 SUCCESS
降維後 資料維度 ---   Data_train = (280, 20)  || Data_test = (120, 20) ========= 降維 SUCCESS
降維後 資料維度 ---   Data_train = (280, 30)  || Data_test = (120, 30) ========= 降維 SUCCESS
降維後 資料維度 ---   Data_train = (280, 40)  || Data_test = (120, 40) ========= 降維 SUCCESS
降維後 資料維度 ---   Data_train = (280, 50)  || Data_test = (120, 50) ========= 降維 SUCCESS
==========================================  FINISH  ============================================


====================== PCA 分類辨識 STAET =======================
------------------------------------  Dimention 10 --------------
訓練集正確率 =  0.9964285714285714
測試集正確率 =  0.9833333333333333
------------------------------------  Dimention 20 --------------
訓練集正確率 =  0.95
測試集正確率 =  0.975
------------------------------------  Dimention 30 --------------
訓練集正確率 =  0.9321428571428572
測試集正確率 =  0.9416666666666667
------------------------------------  Dimention 40 --------------
訓練集正確率 =  0.9214285714285714
測試集正確率 =  0.9416666666666667
------------------------------------  Dimention 50 --------------
訓練集正確率 =  0.9142857142857143
測試集正確率 =  0.9166666666666666
===========================  FINISH  ============================


====================================== LDA 轉換  STAET =======================================
轉換後 資料維度 ---   Data_train = (280, 1)  || Data_test = (120, 1) ========= 轉換 SUCCESS
========================================  FINISH  ============================================


========================= LDA 分類辨識 STAET ==========================
------------------------------------  Dimention X (LDA) ---------------
訓練集正確率 =  0.9964285714285714
測試集正確率 =  0.875
============================  FINISH  =================================


========= PCA 混淆矩陣  =========
----- Dimeniton : 1   -----
 [[60  0]
 [ 2 58]]
----- Dimeniton : 2   -----
 [[57  3]
 [ 0 60]]
----- Dimeniton : 3   -----
 [[53  7]
 [ 0 60]]
----- Dimeniton : 4   -----
 [[53  7]
 [ 0 60]]
----- Dimeniton : 5   -----
 [[50 10]
 [ 0 60]]
===== PCA 混淆矩陣 FINISH  ======


========= LDA 混淆矩陣  =========
----- Dimeniton : X   -----
 [[45 15]
 [ 0 60]]
===== LDA 混淆矩陣 FINISH  ======
```

:::
:::warning
**人臉圖片數量：400
非人臉圖片數量：200
訓練集測試集比： 7:3**

:::spoiler 完整輸出結果
```
============== READ 【with Face】 IMG START =============
Image with Face AMOUNT  ------ 400
Image with Face Flatten DATA SIZE  ------ (400, 10304)
Image with Face Lable Size  ------------- (400,)
============= READ 【with Face】 IMG SUCCESS ============


============ READ 【without Face】 IMG START ============
Image without Face AMOUNT  ------ 200
Image without Face Flatten DATA SIZE  ------ (200, 10304)
Image without Face Lable Size  ------------- (200,)
=========== READ 【without Face】 IMG SUCCESS ===========


================ 資料分組  STAET ================
數據訓練集 SIZE ----- Data_train  =  (420, 10304)
數據測試集 SIZE ----- Data_test   =  (180, 10304)
標籤訓練集 SIZE ----- Label_train =  (420,)
標籤訓練集 SIZE ----- Label_test  =  (180,)
====================  FINISH  ===================


======================================= PCA 降維  STAET ========================================
降維後 資料維度 ---   Data_train = (420, 10)  || Data_test = (180, 10) ========= 降維 SUCCESS
降維後 資料維度 ---   Data_train = (420, 20)  || Data_test = (180, 20) ========= 降維 SUCCESS
降維後 資料維度 ---   Data_train = (420, 30)  || Data_test = (180, 30) ========= 降維 SUCCESS
降維後 資料維度 ---   Data_train = (420, 40)  || Data_test = (180, 40) ========= 降維 SUCCESS
降維後 資料維度 ---   Data_train = (420, 50)  || Data_test = (180, 50) ========= 降維 SUCCESS
==========================================  FINISH  ============================================


====================== PCA 分類辨識 STAET =======================
------------------------------------  Dimention 10 --------------
訓練集正確率 =  0.9904761904761905
測試集正確率 =  0.9777777777777777
------------------------------------  Dimention 20 --------------
訓練集正確率 =  0.9738095238095238
測試集正確率 =  0.9444444444444444
------------------------------------  Dimention 30 --------------
訓練集正確率 =  0.969047619047619
測試集正確率 =  0.9166666666666666
------------------------------------  Dimention 40 --------------
訓練集正確率 =  0.9666666666666667
測試集正確率 =  0.9111111111111111
------------------------------------  Dimention 50 --------------
訓練集正確率 =  0.9619047619047619
測試集正確率 =  0.9055555555555556
===========================  FINISH  ============================


====================================== LDA 轉換  STAET =======================================
轉換後 資料維度 ---   Data_train = (420, 1)  || Data_test = (180, 1) ========= 轉換 SUCCESS
========================================  FINISH  ============================================


========================= LDA 分類辨識 STAET ==========================
------------------------------------  Dimention X (LDA) ---------------
訓練集正確率 =  0.9976190476190476
測試集正確率 =  0.9
============================  FINISH  =================================


========= PCA 混淆矩陣  =========
----- Dimeniton : 1   -----
 [[ 56   4]
 [  0 120]]
----- Dimeniton : 2   -----
 [[ 50  10]
 [  0 120]]
----- Dimeniton : 3   -----
 [[ 45  15]
 [  0 120]]
----- Dimeniton : 4   -----
 [[ 44  16]
 [  0 120]]
----- Dimeniton : 5   -----
 [[ 43  17]
 [  0 120]]
===== PCA 混淆矩陣 FINISH  ======


========= LDA 混淆矩陣  =========
----- Dimeniton : X   -----
 [[ 42  18]
 [  0 120]]
===== LDA 混淆矩陣 FINISH  ======
```
:::


:::warning
**人臉圖片數量：200
非人臉圖片數量：100
訓練集測試集比： 7:3**

:::spoiler 完整輸出結果
```
============== READ 【with Face】 IMG START =============
Image with Face AMOUNT  ------ 200
Image with Face Flatten DATA SIZE  ------ (200, 10304)
Image with Face Lable Size  ------------- (200,)
============= READ 【with Face】 IMG SUCCESS ============


============ READ 【without Face】 IMG START ============
Image without Face AMOUNT  ------ 100
Image without Face Flatten DATA SIZE  ------ (100, 10304)
Image without Face Lable Size  ------------- (100,)
=========== READ 【without Face】 IMG SUCCESS ===========


================ 資料分組  STAET ================
數據訓練集 SIZE ----- Data_train  =  (210, 10304)
數據測試集 SIZE ----- Data_test   =  (90, 10304)
標籤訓練集 SIZE ----- Label_train =  (210,)
標籤訓練集 SIZE ----- Label_test  =  (90,)
====================  FINISH  ===================


======================================= PCA 降維  STAET ========================================
降維後 資料維度 ---   Data_train = (210, 10)  || Data_test = (90, 10) ========= 降維 SUCCESS
降維後 資料維度 ---   Data_train = (210, 20)  || Data_test = (90, 20) ========= 降維 SUCCESS
降維後 資料維度 ---   Data_train = (210, 30)  || Data_test = (90, 30) ========= 降維 SUCCESS
降維後 資料維度 ---   Data_train = (210, 40)  || Data_test = (90, 40) ========= 降維 SUCCESS
降維後 資料維度 ---   Data_train = (210, 50)  || Data_test = (90, 50) ========= 降維 SUCCESS
==========================================  FINISH  ============================================


====================== PCA 分類辨識 STAET =======================
------------------------------------  Dimention 10 --------------
訓練集正確率 =  0.9761904761904762
測試集正確率 =  0.9888888888888889
------------------------------------  Dimention 20 --------------
訓練集正確率 =  0.9619047619047619
測試集正確率 =  0.9444444444444444
------------------------------------  Dimention 30 --------------
訓練集正確率 =  0.9571428571428572
測試集正確率 =  0.9333333333333333
------------------------------------  Dimention 40 --------------
訓練集正確率 =  0.9333333333333333
測試集正確率 =  0.9333333333333333
------------------------------------  Dimention 50 --------------
訓練集正確率 =  0.9428571428571428
測試集正確率 =  0.9222222222222223
===========================  FINISH  ============================


====================================== LDA 轉換  STAET =======================================
轉換後 資料維度 ---   Data_train = (210, 1)  || Data_test = (90, 1) ========= 轉換 SUCCESS
========================================  FINISH  ============================================


========================= LDA 分類辨識 STAET ==========================
------------------------------------  Dimention X (LDA) ---------------
訓練集正確率 =  0.9952380952380953
測試集正確率 =  0.9666666666666667
============================  FINISH  =================================


========= PCA 混淆矩陣  =========
----- Dimeniton : 1   -----
 [[29  1]
 [ 0 60]]
----- Dimeniton : 2   -----
 [[25  5]
 [ 0 60]]
----- Dimeniton : 3   -----
 [[24  6]
 [ 0 60]]
----- Dimeniton : 4   -----
 [[24  6]
 [ 0 60]]
----- Dimeniton : 5   -----
 [[23  7]
 [ 0 60]]
===== PCA 混淆矩陣 FINISH  ======


========= LDA 混淆矩陣  =========
----- Dimeniton : X   -----
 [[27  3]
 [ 0 60]]
===== LDA 混淆矩陣 FINISH  ======
```
:::

:::warning
**人臉圖片數量：200
非人臉圖片數量：200
訓練集測試集比： 5:5**

:::spoiler 完整輸出結果
```
============== READ 【with Face】 IMG START =============
Image with Face AMOUNT  ------ 200
Image with Face Flatten DATA SIZE  ------ (200, 10304)
Image with Face Lable Size  ------------- (200,)
============= READ 【with Face】 IMG SUCCESS ============


============ READ 【without Face】 IMG START ============
Image without Face AMOUNT  ------ 200
Image without Face Flatten DATA SIZE  ------ (200, 10304)
Image without Face Lable Size  ------------- (200,)
=========== READ 【without Face】 IMG SUCCESS ===========


================ 資料分組  STAET ================
數據訓練集 SIZE ----- Data_train  =  (200, 10304)
數據測試集 SIZE ----- Data_test   =  (200, 10304)
標籤訓練集 SIZE ----- Label_train =  (200,)
標籤訓練集 SIZE ----- Label_test  =  (200,)
====================  FINISH  ===================


======================================= PCA 降維  STAET ========================================
降維後 資料維度 ---   Data_train = (200, 10)  || Data_test = (200, 10) ========= 降維 SUCCESS
降維後 資料維度 ---   Data_train = (200, 20)  || Data_test = (200, 20) ========= 降維 SUCCESS
降維後 資料維度 ---   Data_train = (200, 30)  || Data_test = (200, 30) ========= 降維 SUCCESS
降維後 資料維度 ---   Data_train = (200, 40)  || Data_test = (200, 40) ========= 降維 SUCCESS
降維後 資料維度 ---   Data_train = (200, 50)  || Data_test = (200, 50) ========= 降維 SUCCESS
==========================================  FINISH  ============================================


====================== PCA 分類辨識 STAET =======================
------------------------------------  Dimention 10 --------------
訓練集正確率 =  0.965
測試集正確率 =  0.985
------------------------------------  Dimention 20 --------------
訓練集正確率 =  0.95
測試集正確率 =  0.95
------------------------------------  Dimention 30 --------------
訓練集正確率 =  0.94
測試集正確率 =  0.93
------------------------------------  Dimention 40 --------------
訓練集正確率 =  0.93
測試集正確率 =  0.92
------------------------------------  Dimention 50 --------------
訓練集正確率 =  0.92
測試集正確率 =  0.91
===========================  FINISH  ============================


====================================== LDA 轉換  STAET =======================================
轉換後 資料維度 ---   Data_train = (200, 1)  || Data_test = (200, 1) ========= 轉換 SUCCESS
========================================  FINISH  ============================================


========================= LDA 分類辨識 STAET ==========================
------------------------------------  Dimention X (LDA) ---------------
訓練集正確率 =  0.995
測試集正確率 =  0.83
============================  FINISH  =================================


========= PCA 混淆矩陣  =========
----- Dimeniton : 1   -----
 [[98  2]
 [ 1 99]]
----- Dimeniton : 2   -----
 [[ 90  10]
 [  0 100]]
----- Dimeniton : 3   -----
 [[ 86  14]
 [  0 100]]
----- Dimeniton : 4   -----
 [[ 84  16]
 [  0 100]]
----- Dimeniton : 5   -----
 [[ 82  18]
 [  0 100]]
===== PCA 混淆矩陣 FINISH  ======


========= LDA 混淆矩陣  =========
----- Dimeniton : X   -----
 [[67 33]
 [ 1 99]]
===== LDA 混淆矩陣 FINISH  ======
```
:::

:::warning
**人臉圖片數量：400
非人臉圖片數量：200
訓練集測試集比： 5:5**

:::spoiler 完整輸出結果
```
============== READ 【with Face】 IMG START =============
Image with Face AMOUNT  ------ 400
Image with Face Flatten DATA SIZE  ------ (400, 10304)
Image with Face Lable Size  ------------- (400,)
============= READ 【with Face】 IMG SUCCESS ============


============ READ 【without Face】 IMG START ============
Image without Face AMOUNT  ------ 200
Image without Face Flatten DATA SIZE  ------ (200, 10304)
Image without Face Lable Size  ------------- (200,)
=========== READ 【without Face】 IMG SUCCESS ===========


================ 資料分組  STAET ================
數據訓練集 SIZE ----- Data_train  =  (300, 10304)
數據測試集 SIZE ----- Data_test   =  (300, 10304)
標籤訓練集 SIZE ----- Label_train =  (300,)
標籤訓練集 SIZE ----- Label_test  =  (300,)
====================  FINISH  ===================


======================================= PCA 降維  STAET ========================================
降維後 資料維度 ---   Data_train = (300, 10)  || Data_test = (300, 10) ========= 降維 SUCCESS
降維後 資料維度 ---   Data_train = (300, 20)  || Data_test = (300, 20) ========= 降維 SUCCESS
降維後 資料維度 ---   Data_train = (300, 30)  || Data_test = (300, 30) ========= 降維 SUCCESS
降維後 資料維度 ---   Data_train = (300, 40)  || Data_test = (300, 40) ========= 降維 SUCCESS
降維後 資料維度 ---   Data_train = (300, 50)  || Data_test = (300, 50) ========= 降維 SUCCESS
==========================================  FINISH  ============================================


====================== PCA 分類辨識 STAET =======================
------------------------------------  Dimention 10 --------------
訓練集正確率 =  0.97
測試集正確率 =  0.98
------------------------------------  Dimention 20 --------------
訓練集正確率 =  0.9533333333333334
測試集正確率 =  0.9466666666666667
------------------------------------  Dimention 30 --------------
訓練集正確率 =  0.9566666666666667
測試集正確率 =  0.94
------------------------------------  Dimention 40 --------------
訓練集正確率 =  0.9533333333333334
測試集正確率 =  0.9266666666666666
------------------------------------  Dimention 50 --------------
訓練集正確率 =  0.9433333333333334
測試集正確率 =  0.9166666666666666
===========================  FINISH  ============================


====================================== LDA 轉換  STAET =======================================
轉換後 資料維度 ---   Data_train = (300, 1)  || Data_test = (300, 1) ========= 轉換 SUCCESS
========================================  FINISH  ============================================


========================= LDA 分類辨識 STAET ==========================
------------------------------------  Dimention X (LDA) ---------------
訓練集正確率 =  0.9966666666666667
測試集正確率 =  0.9866666666666667
============================  FINISH  =================================


========= PCA 混淆矩陣  =========
----- Dimeniton : 1   -----
 [[ 99   1]
 [  5 195]]
----- Dimeniton : 2   -----
 [[ 87  13]
 [  3 197]]
----- Dimeniton : 3   -----
 [[ 83  17]
 [  1 199]]
----- Dimeniton : 4   -----
 [[ 78  22]
 [  0 200]]
----- Dimeniton : 5   -----
 [[ 75  25]
 [  0 200]]
===== PCA 混淆矩陣 FINISH  ======


========= LDA 混淆矩陣  =========
----- Dimeniton : X   -----
 [[ 97   3]
 [  1 199]]
===== LDA 混淆矩陣 FINISH  ======
```
:::



:::warning
**人臉圖片數量：200
非人臉圖片數量：100
訓練集測試集比： 5:5**

:::spoiler 完整輸出結果
```
============== READ 【with Face】 IMG START =============
Image with Face AMOUNT  ------ 200
Image with Face Flatten DATA SIZE  ------ (200, 10304)
Image with Face Lable Size  ------------- (200,)
============= READ 【with Face】 IMG SUCCESS ============


============ READ 【without Face】 IMG START ============
Image without Face AMOUNT  ------ 100
Image without Face Flatten DATA SIZE  ------ (100, 10304)
Image without Face Lable Size  ------------- (100,)
=========== READ 【without Face】 IMG SUCCESS ===========


================ 資料分組  STAET ================
數據訓練集 SIZE ----- Data_train  =  (150, 10304)
數據測試集 SIZE ----- Data_test   =  (150, 10304)
標籤訓練集 SIZE ----- Label_train =  (150,)
標籤訓練集 SIZE ----- Label_test  =  (150,)
====================  FINISH  ===================


======================================= PCA 降維  STAET ========================================
降維後 資料維度 ---   Data_train = (150, 10)  || Data_test = (150, 10) ========= 降維 SUCCESS
降維後 資料維度 ---   Data_train = (150, 20)  || Data_test = (150, 20) ========= 降維 SUCCESS
降維後 資料維度 ---   Data_train = (150, 30)  || Data_test = (150, 30) ========= 降維 SUCCESS
降維後 資料維度 ---   Data_train = (150, 40)  || Data_test = (150, 40) ========= 降維 SUCCESS
降維後 資料維度 ---   Data_train = (150, 50)  || Data_test = (150, 50) ========= 降維 SUCCESS
==========================================  FINISH  ============================================


====================== PCA 分類辨識 STAET =======================
------------------------------------  Dimention 10 --------------
訓練集正確率 =  0.9866666666666667
測試集正確率 =  0.9866666666666667
------------------------------------  Dimention 20 --------------
訓練集正確率 =  0.96
測試集正確率 =  0.94
------------------------------------  Dimention 30 --------------
訓練集正確率 =  0.9666666666666667
測試集正確率 =  0.9333333333333333
------------------------------------  Dimention 40 --------------
訓練集正確率 =  0.96
測試集正確率 =  0.9133333333333333
------------------------------------  Dimention 50 --------------
訓練集正確率 =  0.9666666666666667
測試集正確率 =  0.9066666666666666
===========================  FINISH  ============================


====================================== LDA 轉換  STAET =======================================
轉換後 資料維度 ---   Data_train = (150, 1)  || Data_test = (150, 1) ========= 轉換 SUCCESS
========================================  FINISH  ============================================


========================= LDA 分類辨識 STAET ==========================
------------------------------------  Dimention X (LDA) ---------------
訓練集正確率 =  0.9933333333333333
測試集正確率 =  0.9666666666666667
============================  FINISH  =================================


========= PCA 混淆矩陣  =========
----- Dimeniton : 1   -----
 [[ 48   2]
 [  0 100]]
----- Dimeniton : 2   -----
 [[ 41   9]
 [  0 100]]
----- Dimeniton : 3   -----
 [[ 40  10]
 [  0 100]]
----- Dimeniton : 4   -----
 [[ 37  13]
 [  0 100]]
----- Dimeniton : 5   -----
 [[ 36  14]
 [  0 100]]
===== PCA 混淆矩陣 FINISH  ======


========= LDA 混淆矩陣  =========
----- Dimeniton : X   -----
 [[ 45   5]
 [  0 100]]
===== LDA 混淆矩陣 FINISH  ======
```
:::







---
### 結果討論

很意外的是對大部分的時候，無論人臉圖片跟非人臉圖片的張數改變，或是訓練集跟測試集的比例改變，整體來說PCA的辨識效果更勝LDA。

再來是圖片數量的改變，除了200/100(人臉/非人臉)這組的辨識率，有稍微低一點點以外，其實對PCA的辨識結果來說還算蠻穩定蠻好的。

另外本來沒有要調整測試集與訓練集的分配比例，但後來覺得也是一個可以觀察的項目之一，就把這項加上去，看結果雖然會發現它會因此更動最後轉換矩陣的各種類的數量，這點可以從混淆矩陣看出來，但整體來說辨識率的變化沒有因此改變太大。

最後回到PCA跟LDA的比較，認為LDA辨識率較低的原因可能在於，他是一種監督式的分析，可以一邊降維一邊訓練，雖然這裡設定的是不降維，但它轉換出來的最大維度就是我們目標要分成的種類數-1，也就是我們只分成【是人】or【不是人】的情況，它的樣本特徵就會被壓成一維，如果這時候直接去判斷辨識結果也許會比較好，但它又還要丟進分類器的話，對分類器來說只有一個特徵的樣本，可能還是太過困難。它原本的特徵已經被處理得很明顯，再多做一次分類辨識可能會多造成一點點偏差。

---
### 總結
雖然分類器、PCA、LDA都不是自己手刻出來的(可能還是有點太難)，但發現scikit-learn這個套件真的很好用，方法很簡單，網路上又有蠻多教學可以用的，而且效果很好，光是最基本的高斯貝是分類器就簡單好用。

剩下最難的工作大概就剩要釐清資料在每個環節的型態了，而且因為一開始要自己去處理非人臉圖片這塊負樣本，個別又有資料又有標籤，還要分成訓練跟測試，整個兜在一起之後要分流的資料種類太多，一度搞混，搞得變數名稱取得很長才能標示清楚。

這次覺得最好玩的部分應該是把程式切成很多區塊，彼此呼叫使用，以前的程式可能不需要這麼多行數，但這從圖片讀取開始，打一打不小心就很多行，最後邊打邊整理，決定還是用以前很少用的分檔案進行。


