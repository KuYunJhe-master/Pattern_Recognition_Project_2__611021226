#--------------處理人臉圖片輸入 轉換成個別圖片一維的樣本集------------------------------
import numpy as np
import cv2

def READ_FACE_IMG(face_image_amount):
    print("\n\n============== READ 【with Face】 IMG START =============")

    if ((face_image_amount % 40) != 0) and (face_image_amount <= 400): #判斷指定的臉部圖片是否數量正確
        print("ERROR: Face image amount should be multiples of 40 ! 臉部圖片數量必須是40的倍數!\n") #不正確就回傳錯誤訊息然後結束程式
        exit()
    else:
        max_img_face_idx = int(face_image_amount / 40) +1 #正確就計算要開啟的圖片最大位置

    img_face_10304x200 = np.zeros(shape = (1,10304),dtype=int)#預先建立要放入PCA的資料陣列

    for folder_face_idx in range(1,41):#開啟人臉圖片資料夾1~40
        for img_face_idx in range(1,max_img_face_idx):#選取開啟各資料夾人臉圖片1~10
            img_face = cv2.imread("att_faces/s" + str(folder_face_idx) + "/" + str(img_face_idx) + '.pgm')#讀取圖片
            # print(img_face.shape)

            # cv2.imshow('My Image', img_face)#顯示圖片(檢查使用)
            # # 按下任意鍵則關閉所有視窗
            # cv2.waitKey(1)
            # cv2.destroyAllWindows()

            img_face = cv2.cvtColor(img_face, cv2.COLOR_BGR2GRAY)#確認將圖片轉為灰階並僅留一個色彩(灰階)頻道
            img_face_1D = img_face.flatten()#將二維圖片攤平成一維
            img_face_1D = np.array([img_face_1D])#轉成二維，才能用np.append放到二維array(img_10304Dx200)裡面
            img_face_10304x1 = img_face_1D
            img_face_10304x200 =  np.append(img_face_10304x200, img_face_10304x1,axis= 0)#用np.append把img_10304x1 逐一放到 img_10304Dx200 裡面
            

    img_face_10304x200 = img_face_10304x200[1:]#砍掉最先建立的0列

    target_face = np.ones(shape = (img_face_10304x200.shape[0]),dtype=int)#把標【是人臉】的標籤存放到標籤的矩陣

    print("Image with Face Flatten DATA SIZE  ------" , img_face_10304x200.shape)
    print("Image with Face Lable Size  -------------" , target_face.shape)
    print("============= READ 【with Face】 IMG SUCCESS ============\n\n")

    return img_face_10304x200 , target_face , img_face.shape
