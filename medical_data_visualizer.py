import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. carregar o dataset
df = pd.read_csv('medical_examination.csv')

# 2. criar coluna de "overweight" usando a formula de BMI, a gente divide a altura por 100 pra ficar em metros, depois a gente converte tudo pra binario (0/1)
bmi = df['weight'] / (df['height'] / 100) ** 2
df['overweight'] = (bmi > 25).astype(int)

# 3. normalizar o "cholesterol" e "gluc", os valores 1 viram 0 (normal), > 1 viram 1 (anormal)
df['cholesterol'] = df['cholesterol'].gt(1).astype(int)
df['gluc'] = df['gluc'].gt(1).astype(int)

# 4
def draw_cat_plot():
    # 5. usamos o pd.melt para fazer o reshape dos dados de formato 'longo' para 'largo', deixando "cardio" como identificador
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 6. agrupar por "cardio", "variable" e "value" pra obter as contages
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    # 7. criar graficos de barra um do lado do outro mostrando as distribuicoes das caracteristicas para pacientes com/sem doenca vascular
    grid = sns.catplot(data=df_cat, x='variable', y='total', hue='value', col='cardio', kind='bar')

    # 8. gera a visualizacao das comparacoes
    fig = grid.fig
    fig.tight_layout()

    # 9. salva os graficos como uma imagem
    fig.savefig('catplot.png', bbox_inches='tight')
    return fig

# 10
def draw_heat_map():
    # 11.  limpar os dados tirando os outliers, tirando os casos onde a pressao diastolica > pressao sistolica, filtrar a altura/peso para porcentagens (2.5% ate 97.5%)
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12. calcular a matriz de correlacao entre todas as caracteristicas numericas
    corr = df_heat.corr()

    # 13. criar a mascara do triangulo superior pra mostrar apenas as correlacoes unicas (pq q eles tem uma funcao so pra isso)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14.
    fig, ax = plt.subplots(figsize=(12, 9))

    # 15. gerar o heatmap com os coeficientes de correlacao
    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', center=0, square=True, linewidths=0.5, cbar_kws={'shrink': 0.5}, ax=ax)
    # 16. salvar o heatmap em uma imagem
    fig.tight_layout()
    fig.savefig('heatmap.png', bbox_inches='tight')
    return fig

#Coisas q podem ser melhoradas:
#
# seaborn não foi utilizado