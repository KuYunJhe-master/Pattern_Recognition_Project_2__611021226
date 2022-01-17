#------------做混淆矩陣---------------------------------------------------------------
import _Confusion_Matrix

def DO_CONFUSI_MATRIX(PCA_correct_ans , PCA_recog_ans , set_PCA_dimeniton , LDA_correct_ans , LDA_recog_ans , set_LDA_dimeniton):

    print("========= PCA 混淆矩陣  =========")
    _Confusion_Matrix.DO_Confusi_Matrix(PCA_correct_ans , PCA_recog_ans , set_PCA_dimeniton)  
    print("===== PCA 混淆矩陣 FINISH  ======\n\n")

    print("========= LDA 混淆矩陣  =========")
    _Confusion_Matrix.DO_Confusi_Matrix(LDA_correct_ans , LDA_recog_ans , set_LDA_dimeniton)
    print("===== LDA 混淆矩陣 FINISH  ======\n\n")