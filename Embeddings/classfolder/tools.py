import numpy as np
import scipy as sp
import pandas as pd



class MarkTools():
    grade = {'primero': 1, 'segundo': 2, 'tercero': 3, 'cuarto': 4,
        'quinto': 5, 'sexto': 6, 'septimo': 7, 'octavo': 8}
    level = {'basico': 0,'medio': 8}
    def __init__(self,X=None,y=None):
        self.X = X
        self.y = y
    def num_turn(self,course,grd=grade,lv=level):
        """
        Transformación: e.g.tercero basico' = 3; 'tercero medio'= 11
        """
        course = [word.strip() for word in course.split()]
        new_n = grd[course[0]] + lv[course[1]]
        return new_n
    def bool_mask(self, hl, a_column):
        """
        Genera boolean array marcando True las casillas que tienen tal valor hl.
        df: DataFrame a procesar.
        hl: highlight. valor destacado
        a_column: columna a analizar
        """
        bool_mark = np.array([hl in value for value in self.X[a_column]])
        return bool_mark
    def hasnulls(self, n=True):
        """Revisa si DataFrame tiene valores np.nan y retorna o el numero de
        estos, o un boolean. En los dos casos, genera display de la informacion
        n: True si es n#, False si es boolean
        """
        if n:
            n = self.X.isnull().sum().sum()
            print(f"The DataFrame has {n} Nan values.".format(i=n))
            return n
        else:
            has = self.X.isnull().values.any()
            i = "has" if has else "doesn't has"
            print(f"The DataFrame {i} Nan values.".format(i=i))
            return has
    def hasstr(x):
        return 1 if isinstance(x, str) else 0
    def numstr(sprs_matrix):
        whatisit = 0
        for i in range(sprs_matrix.getnnz()):
            #print(data_prepared.data()[i])
            n += hasstr(sprs_matrix.data[i])
        return n

def GrayEncoder(vector):#de una lista de notas, genera matriz de notas en código gray
    g_bits_list = GrayCode(bits_length)
    bits_list = generate_gray()
    marks_matrix = np.zeros((bits_length,len(vector)))
    count = 0
    for mark in vector:
        bits = bits_list[mark+1]
    b_count = 0
    for bit in bits:
        if bit == 1:
            marks_matrix[count][b_count] = 1
            b_count += 1
            count += 1
    return marks_matrix
"""
class PCA():
    def __init__(self, X, transpose=False):
        if isinstance(X, pd.DataFrame):
            self.X = X.to_numpy()
        self.X = np.matrix(X)
        self.X_mean = self.X.mean(axis=0)
        self.B = self.X - self.X_mean
        normalizer = 1/(len(X[:,0]) - 1)
        self.C = normalizer*np.matmul(B.T, B)
        self.u, self.s, self.v = np.linalg.svd(self.C, full_matrices=False)
"""
