

import numpy as np
import pandas as pd
import tkinter.filedialog
import matplotlib.pyplot as plt
# clearfrom sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.models import Sequential
# from keras.layers import Dense
# from sklearn import preprocessing
import sys


# Ler os dados
filename = tkinter.filedialog.askopenfilename(
    title='Escolha o arquivo',
    filetypes=[('CSV files', '*.csv')]
)

if filename:
    df = pd.read_csv(filename)
    dados = np.array(df.values)
else:
    print("Nenhum arquivo foi selecionado.")
    sys.exit(0)

# Escolher os dados
par_col = [10, 2, 0, 1, 6]
print(par_col)

par = dados
print(par)
# tag = dados [: ,12].astype(int)
# #tag = tag. reshape(-1,1)
# lb = preprocessing. LabelBinarizer ()
# lb.fit([3,4,5,6,7,8,91)
# tag = lb. transform (tag)
# filename = tkinter.filedialog.askopenfilename(title = "Escolha o Arquivo", filetypes = [("Arquivos CSV","*.csv
# # Normalizar z3
# for i in range(5) :
# med = np.mean (par[:,i])
# st = np.std(par[:,1])
# bash +
# =============================================================================
