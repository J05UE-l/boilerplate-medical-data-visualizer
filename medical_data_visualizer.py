import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Carregar o conjunto de dados
df = pd.read_csv('medical_examination.csv')

# 2. Calcular o índice de massa corporal (IMC) e criar a coluna 'overweight'
#    Indica sobrepeso quando IMC > 25. A altura é convertida de centímetros para metros.
bmi = df['weight'] / (df['height'] / 100) ** 2
df['overweight'] = (bmi > 25).astype(int)

# 3. Normalizar as colunas 'cholesterol' e 'gluc'
#    Valores iguais a 1 são mapeados para 0 (normal) e valores maiores que 1 são mapeados para 1 (alterado)
df['cholesterol'] = df['cholesterol'].gt(1).astype(int)
df['gluc'] = df['gluc'].gt(1).astype(int)

# 4. Função para desenhar gráfico categórico
def draw_cat_plot():
    # 5. Utilizar pd.melt para reestruturar os dados em formato longo, mantendo 'cardio' como identificador
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 6. Agrupar por 'cardio', 'variable' e 'value' e contabilizar ocorrências
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    # 7. Gerar gráficos de barras comparativos para cada categoria de 'cardio'
    grid = sns.catplot(data=df_cat, x='variable', y='total', hue='value', col='cardio', kind='bar')

    # 8. Ajustar o layout da figura
    fig = grid.fig
    fig.tight_layout()

    # 9. Salvar a figura resultante em arquivo
    fig.savefig('catplot.png', bbox_inches='tight')
    return fig

# 10. Função para desenhar o mapa de calor de correlação
def draw_heat_map():
    # 11. Filtrar registros inválidos e remover outliers:
    #     - Excluir casos em que a pressão diastólica seja superior à pressão sistólica
    #     - Conservar registros de altura e peso entre os percentis 2.5% e 97.5%
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12. Calcular a matriz de correlação entre variáveis numéricas
    corr = df_heat.corr()

    # 13. Criar máscara para o triângulo superior da matriz de correlação (exibir apenas valores únicos)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14. Preparar figura e eixos para o heatmap
    fig, ax = plt.subplots(figsize=(12, 9))

    # 15. Plotar o heatmap com anotações dos coeficientes de correlação
    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', center=0, square=True, linewidths=0.5, cbar_kws={'shrink': 0.5}, ax=ax)

    # 16. Ajustar o layout e salvar a figura do heatmap
    fig.tight_layout()
    fig.savefig('heatmap.png', bbox_inches='tight')
    return fig
