"""
raw_df.hist(bins=70,figsize=(20,15),grid=True)
ax = media.hist(bins=70,figsize=(20,15), grid=True)
basica.hist(bins=70,figsize=(20,15), ax=ax)
raw_df.hist(bins=70,figsize=(19,14),density=True)
plt.show()
print(raw_df[raw_df.isna().any(axis=1)])

raw_df.loc[raw_df['fecha'] > '2018-01-01'].plot(kind='scatter',
    x='curso_cat', y='nota', alpha=0.2, c='fecha',
    cmap=plt.get_cmap("jet"), figsize=(10,7))

h = raw_df['asignatura'].value_counts()
n_subj = 19
column = pd.Index(['notas totales'])
per_subj = pd.DataFrame(raw_df['nota'].values, columns = column)
h_index = h.index.to_numpy()
for i in range(n_subj):
    """
    Generando columnas en nuevo DataFrame per_subj, cada una es una asignatura
    con las notas correspondientes como values
    """
    per_subj[h_index[i]] = raw_df['nota'].loc[raw_df['asignatura'] == h_index[i]]

per_subj.hist(bins=60, figsize=(22,22),grid=True, sharex=True)
pyl.suptitle('Notas totales por asignatura')
#plt.show()

b = tool.bool_mark('medio','curso')
per_subj_media = per_subj.loc[b]
per_subj_basica = per_subj.loc[~b]

from matplotlib.patches import Rectangle
handles = [Rectangle((0,0),1,1,color=c) for c in ['g','r']]
labels= ['Media','Básica']

ax = per_subj_media.hist(bins=60, figsize=(20,15),
        alpha=0.8,sharex=True,color='g',layout=(4,5))
per_subj_basica.hist(bins=60,alpha=0.5,color='r',ax=ax)
pyl.suptitle('Histograma: Comparación entre Media y Básica',
    weight='bold',size=18)
ax[0][0].legend(handles, labels)
ax[-1][-1].set_xlabel('Notas',labelpad=20, weight='bold',size=14)
ax[0][0].set_ylabel('Conteo de Nota',labelpad=50,weight='bold',size=14)

"""
