#--------------【設定】 混淆矩陣---------------------------------------------------------------

from sklearn.metrics import confusion_matrix

def DO_Confusi_Matrix(correct_answer , recog_anwser , count_dimeniton):
    
    if count_dimeniton == 0:  #如果要處裡單個辨識結果的混淆矩陣
        cur_dim_confus_mix = confusion_matrix(correct_answer, recog_anwser)
        print("----- Dimeniton : X   -----\n",cur_dim_confus_mix)
    
    else:
        for cur_dimeniton in range(5):    #如果要一次處裡多個不同維度辨識結果的混淆矩陣
            cur_dim_confus_mix = confusion_matrix(correct_answer[cur_dimeniton], recog_anwser[cur_dimeniton])
            print("----- Dimeniton : "+str(cur_dimeniton+1) +"   -----\n", cur_dim_confus_mix)
