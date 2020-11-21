###############################################################################
#main.py regresiones lineales y demases, beirifcación de cross_val_error      #
###############################################################################
"""Generando LinearRegression con los datos seleccionados de entrenamiento
"""
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
lin_reg = LinearRegression()

"""Calculando 'mean_squared_error' a predicción formada por 'm_test'
y comparada con 'm_te_label'
"""
from sklearn.metrics import mean_squared_error
#predictions = forest_reg.predict(fullpipe.transform(m_test))
#lin_mse = mean_squared_error(num_pipe.transform(m_te_label), predictions)
#lin_rmse = np.sqrt(lin_mse)

from sklearn.model_selection import cross_val_score

#scores = cross_val_score(forest_reg, tr_dummy, tr_label,
#scoring="neg_mean_squared_error", cv=10)
#tree_rmse_scores = np.sqrt(-scores)
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
#display_scores(scores)
"""segundo termino cuadratico de FactorizationMachine. Mediante broadcasting"""
factor_b = tf.reduce_sum(tf.reduce_sum(self.V**2, axis=2, keepdims=True)
* tf.transpose(inputs**2), axis=1)

"""Fallida funcion para separar dataset de forma expedita"""
def split_data(DataFrame, split=split, prop_column='curso'):
    for train_index, test_index in split.split(DataFrame, DataFrame[prop_column]):
        train_frame = DataFrame.iloc[train_index].copy()
        test_frame = DataFrame.iloc[test_index].copy()
        yield train_frame, test_frame
"""
for train_index, test_index in split.split(df_18, df_18['curso']):
    strat_train_set = df_18.iloc[train_index].copy()
    strat_test_set = df_18.iloc[test_index].copy()
#print(strat_test_set['curso'].value_counts() / len(strat_test_set))
m_test_label = strat_test_set.loc[:,'nota'].to_frame()
m_test = strat_test_set.drop(['nota'],axis=1)

for train_index, validation_index in split.split(strat_train_set, strat_train_set['curso']):
    m_train_set = strat_train_set.iloc[train_index].copy()
    m_validation_set = strat_train_set.iloc[validation_index].copy()

m_train_label = m_train_set.loc[:,'nota'].to_frame()
m_train = m_train_set.drop(['nota'], axis=1)

m_validation_label = m_validation_set.loc[:,'nota'].to_frame()
m_validation = m_validation_set.drop(['nota'], axis=1)
"""
