#--------------處理非人臉圖片輸入 轉換成個別圖片一維的樣本集----------------------------
import cv2
import numpy as np

def READ_NOT_FACE_IMG(img_face_SHAPE , not_face_image_amount):
    print("============ READ 【without Face】 IMG START ============")

    if ((not_face_image_amount % 20) != 0) and (not_face_image_amount <= 200): #判斷指定的臉部圖片是否數量正確
        print("ERROR: Face image amount should be multiples of 20 ! 臉部圖片數量必須是40的倍數!\n") #不正確就回傳錯誤訊息然後結束程式
        exit()
    else:
        max_img_not_face_idx = not_face_image_amount #正確就計算要開啟的圖片最大位置


    img_unface_10304x200 = np.zeros(shape = (1,10304),dtype=int)#預先建立要放入PCA的資料陣列

    for img_unface_idx in range(max_img_not_face_idx):#選取開啟各資料夾人臉圖片1~10
        img_unface = cv2.imread("data_not_face/" + str(img_unface_idx) + '.jpg')#讀取圖片
        img_unface = cv2.cvtColor(img_unface, cv2.COLOR_BGR2GRAY)#確認將圖片轉為灰階並僅留一個色彩(灰階)頻道

        if (img_unface.shape[0] >= img_face_SHAPE[0]) and (img_unface.shape[1] >= img_face_SHAPE[1]): #可裁切的大小要大於要裁切的大小
            ori_pit_Y_of_cutted_unface = ((img_unface.shape[0]//2) - (img_face_SHAPE[0]//2)) #設定裁切位置對齊圖片中央
            ori_pit_X_of_cutted_unface = ((img_unface.shape[1]//2) - (img_face_SHAPE[1]//2)) #設定裁切位置對齊圖片中央

        #裁切圖片中央的 92*112 大小
        img_unface = img_unface[ori_pit_Y_of_cutted_unface:ori_pit_Y_of_cutted_unface+img_face_SHAPE[1] , ori_pit_X_of_cutted_unface : ori_pit_X_of_cutted_unface+img_face_SHAPE[0]]
        # cv2.imshow('My Image', img_unface)
        # cv2.waitKey(500)
        # cv2.destroyAllWindows()
        
        img_unface_1D = img_unface.flatten()#將二維圖片攤平成一維
        img_unface_1D = np.array([img_unface_1D])#轉成二維，才能用np.append放到二維array(img_10304Dx200)裡面
        img_unface_10304x1 = img_unface_1D
        img_unface_10304x200 =  np.append( img_unface_10304x200, img_unface_10304x1,axis= 0)#用np.append把img_10304x1 逐一放到 img_10304Dx200 裡面

        
    img_unface_10304x200 = img_unface_10304x200[1:]#砍掉最先建立的0列

    target_unface = np.zeros(shape = (img_unface_10304x200.shape[0]),dtype=int)#把標【不是人臉】的標籤存放到標籤的矩陣


    print("Image without Face Flatten DATA SIZE  ------" , img_unface_10304x200.shape)
    print("Image without Face Lable Size  -------------" , target_unface.shape)
    print("=========== READ 【without Face】 IMG SUCCESS ===========\n\n")

    return img_unface_10304x200 , target_unface
