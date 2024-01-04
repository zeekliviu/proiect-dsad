import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sb

data = pd.read_csv('dataIN/number-of-deaths-by-risk-factor.csv', index_col=0)
PATH_FOR_FILES = Path('dataOUT/ACP/Files')
PATH_FOR_PLOTS = Path('dataOUT/ACP/Plots')
PATH_FOR_PLOTS.mkdir(parents=True, exist_ok=True)
PATH_FOR_FILES.mkdir(parents=True, exist_ok=True)

for entry in data:
    data[entry] = np.where(np.isnan(data[entry]), np.nanmean(data[entry]), data[entry])

scaled_data = StandardScaler().fit_transform(data)
var_ratios = []
for i in range(1, len(data.columns)):
    pca = PCA(n_components=i)
    pca.fit(scaled_data)
    var_ratios.append(sum(pca.explained_variance_ratio_))
plt.plot(range(1, len(data.columns)), var_ratios)
plt.xticks(range(1, len(data.columns)))
plt.tick_params(axis='x', labelsize=8)
plt.grid()
plt.xlabel('Numărul de componente')
plt.ylabel('Rata de variație explicată')
plt.title('Rata de variație explicată în funcție de numărul de componente')
plt.savefig(PATH_FOR_PLOTS / 'varianța_explicată_de_nr_componentelor.png')

pca = PCA(n_components=len(data.columns))
pca.fit_transform(scaled_data)

# realizare grafic varianta explicata de componentele principale
plt.figure()
plt.ylim(-0.5, 4)
plt.tick_params(axis='x', labelsize=6)
plt.plot([f'C{i+1}' for i in range(len(pca.explained_variance_))], pca.explained_variance_, 'bo-')
plt.grid()
plt.xlabel('Componente principale')
plt.ylabel('Valori proprii')
plt.title('Varianța explicată de componentele principale')
plt.axhline(y=1, color='r', linestyle='-')
plt.savefig(PATH_FOR_PLOTS / 'varianța_explicată_de_componentele_principale.png')

# salvarea factorilor de sarcină într-un fișier CSV
PC_df = pd.DataFrame(pca.components_, columns=[f'C{i+1}' for i in range(len(pca.components_))], index=data.columns)
PC_df.to_csv(PATH_FOR_FILES / 'factori_de_sarcină.csv', index_label='Factor de risc')

# salvarea corelogramei factorilor de sarcină într-un fișier CSV
plt.figure(figsize=(29,29))
plt.title('Corelograma factorilor de sarcină', fontsize=40, color='k', verticalalignment='bottom')
sb.heatmap(data=PC_df, cmap='bwr', vmin=-1, vmax=1, annot=True)
plt.savefig(PATH_FOR_PLOTS / 'corelograma_factorilor_de_sarcină.png')

# salvarea scorurilor într-un fișier CSV
scores = pca.transform(scaled_data)
scores_df = pd.DataFrame(scores, columns=[f'C{i+1}' for i in range(len(scores[0]))], index=data.index)
scores_df.to_csv(PATH_FOR_FILES / 'scoruri_în_noul_spațiu.csv', index_label='Țară/Regiuine')

# salvarea calității reprezentării observațiilor într-un fișier CSV
quality_df = pd.DataFrame(np.square(scores), columns=[f'C{i+1}' for i in range(len(scores[0]))], index=data.index)
quality_df = quality_df.div(quality_df.sum(axis=1), axis=0)
quality_df.to_csv(PATH_FOR_FILES / 'calitatea_reprezentării_observațiilor.csv', index_label='Țară/Regiuine')

# salvarea contribuțiilor observațiilor într-un fișier CSV
contrib_df = pd.DataFrame(np.square(scores), columns=[f'C{i+1}' for i in range(len(scores[0]))], index=data.index)
contrib_df = contrib_df.div(contrib_df.sum(axis=0), axis=1)
contrib_df.to_csv(PATH_FOR_FILES / 'contribuțiile_observațiilor.csv', index_label='Țară/Regiuine')

# salvarea comunălităților într-un fișier CSV
comm_df = pd.DataFrame(np.cumsum(np.square(pca.components_), axis=1), columns=[f'C{i+1}' for i in range(len(scores[0]))], index=data.columns)
comm_df.to_csv(PATH_FOR_FILES / 'comunalități.csv', index_label='Factor de risc')

# salvarea corelogramei scorurilor într-un fișier CSV
plt.figure(figsize=(50, 50))
plt.title('Corelograma scorurilor', fontsize=40, color='k', verticalalignment='bottom')
plt.ylabel('Țară/Regiune', fontsize=40, color='b', verticalalignment='bottom')
sb.heatmap(data=scores_df, cmap='bwr', vmin=-1, vmax=1, annot=True)
plt.savefig(PATH_FOR_PLOTS / 'corelograma_scorurilor.png')