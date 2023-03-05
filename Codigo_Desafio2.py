# Pacotes utilizados

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregando a base de dados

df_origin = pd.read_csv('C:/Users/Charles/OneDrive/Documents/UNINOVE/DATASET.csv',sep=';',date_parser=True)

# Criando um copia do conjunto de dados original 

    # É uma norma muito em comum na area de dados criarmos copias dos conjunto de dados da quais estamos trabalhando,
    # afim de evitar qualquer tipo de um futuro problema.    

df = df_origin

# Visualizando as dez primeiras linhas do cojunto de dados

df.head(10)

# Visualizando a estrutura do conjunto de dados
    # O conjunto de dados apresenta um total de 612 linhas e 7 colunas (ou variaveis).

df.shape

# Verificando informações sobre as variaveis

df.info()

#  Caracterer " - " do conjunto de dados

# Quando um conjunto de dados apresenta caracteres especias como ('#,@,& etc'). podemos lidar com as seguintes estapas.
    # 1° Se possivel validar com o reposvalve do conjunto de dados se aquelas informações presentes são de fato veritidas ou não.
        # 2° Remover esses caracteres especiais do conjunto de dados, 
        #dessa forma deixando comoo valores com null, 
        #afim de evitar qualquer tipo de problema na implementação de algum algoritimo#
            # 3° Remover a linha do conjunto de dados da qual apresenta esse caracter, 
            #esse metodo e utilizando quando aquele individuo (ou linha) apresenta um deficiencia em sua informações,
            #porem vale ressaltar que essa tecnica pode não ser adequado para conjunto de dados com baixa quantidade dados como é o caso.



            
            
# Removendo os caracteres especiais

    # Atraves da função abaixo podemos visualizar a quantidade do caracter (-) aparece no conjunto de dados
df[df == "-"].count()

# Transformando os caracteres especiais em valores nulos, afim de tratar a base.

df = df[df != "-"]

# Segunda verificação do conunto de dados sobre dados ausentes

df.isna().sum() 


# Transformando as variaveis.
    #  Restruturando os tipos das variaveis presentes no conjunto de dados,
    #  afim de conseguir realizar analises e tambem para implementação de alguma tecnica,
    #  Por exemplo se as varaiveis do conjunto de dados apresentam ser numericas porem se seu 
    #  type não quando formos por exemplo utilizar algum tecnica por exemplo: matriz de correlação,
    #  a linguagem python não entendera, dificultando a anlise dos dados. 

df["Temperature (K)"] = pd.to_numeric(df["Temperature (K)"])
df["Luminosity(L/Lo)"] = pd.to_numeric(df["Luminosity(L/Lo)"])
df["Radius(R/Ro)"] = pd.to_numeric(df["Radius(R/Ro)"])
df["Absolute magnitude(Mv)"] = pd.to_numeric(df["Absolute magnitude(Mv)"])

# Criando uma função para retornar todos os nomes do conjunto de dados
    # Normamlente utilizo essa pratica tanto no (Python / R), pois quando lidamos com muitos dataframes fica 
    # difcil lembrar o nome e tambem o formato escrito

for col in df:
    print(col)


# Verificando se o conjunto de dados apresenta valores ausentes
    # Antes de qualquer procedimento e muito ideal verificar se o conjunto de dados apresenta valores ausentes


df.isna().sum() 



# Verificando as diferentes tipos de estrelas e classes espectrais
    # Devido ao fato de estarmos trabalhando com algumas variaveis categoricas,
    # Conseguimos visualizar a quantidade de classes de estrelas.

df['Star type'].value_counts() , 
df['Spectral Class'].value_counts()

#  Grafico da Quantidade de estrelas

star_classes = pd.DataFrame(df['Spectral Class'].value_counts().sort_values(ascending=False))

plt.figure(figsize=(10, 6))
ax = sns.barplot(x = star_classes.index, y = 'Spectral Class' , data = star_classes, palette='magma')
plt.title("Classes das estrelas", color = "m", fontsize = 18)
plt.ylabel('Quantidade', color = 'b', fontsize = 15)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
ax = ax.set_xticklabels(ax.get_xticklabels(), rotation = 45)

## Verificando a quantidade de tipos de cor da estrela

print("Quantidade de cores de estrelas", df['Star color'].value_counts())
plt.figure(figsize = (13, 6))

## Grafico de barra mostrando a quantidade de cores
    # Atraves do codigo abaico conseguimos visualizar atraves de grafico de barra,
    # conseguimos visualizar quais são as cores que mais aparece no conjunto de dados

plt.figure(figsize = (13, 6))
color = pd.DataFrame(df['Star color'].value_counts().sort_values(ascending=False))
ax = sns.barplot(x = color.index, y = 'Star color' , data = color, palette='magma')
plt.title("Cores das Estrelas", color = "m", fontsize = 18)
plt.ylabel('Quantidade', color = 'b', fontsize = 15)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
ax = ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)


# Criando histogramas das variaveis quantitativas
    # Normalmente e importante criar historgramas das variaveis quantitativas (ou numericas),
    # para conseguirmos ver a representação gráfica em colunas ou em barras de um conjunto de variaveis
    # do conjunto de dados

df['Temperature (K)'].hist()
df['Luminosity(L/Lo)'].hist()
df['Radius(R/Ro)'].hist()
df['Absolute magnitude(Mv)'].hist()
df['Temperature (K)'].hist()
    

# Matriz de correalção
    # Criando uma matriz de correlaçao entre as variaveis numericas.
        # A matriz de correlação exibe o grau de correlação
        # entre várias interseções de medidas como uma matriz de células retangulares. 

df_cor = df[['Temperature (K)','Luminosity(L/Lo)','Radius(R/Ro)','Absolute magnitude(Mv)']]


df_cor.corr();print(df_cor.corr())

# Plotagem de um matriz de correlaçao

sns.heatmap(data = df.corr(), annot = True)


# Verificando o balaceamento das variaveis categoricas
    # Podemos visualizar com o codigo abaixo que a variavel (Star Type) apresenta um desbaleamento 
    # na classe (3), ou seja em modelos de classificação podemos obter um vies, indidico utilizar tecnicas
    # de balaceamento como oversampling and undersampling para verificar se o modelo de classificão 
    # obtem uma melhor performace.
    
        
sns.countplot(df['Star type'], palette = "viridis")


## ############# ############## ############# Modelando ######### ###################### #############

# Pacotes ou library Necessarias

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# Modelando - Part 1 (Normalizando e Analise de componentes principais (PCA) )

# Criando um novo conjunto de dados
    # Devido ao fato do conjunto de dados original apresentar valores null (ou ausentes),
    # Decide criar um copia do dataframe sem os dados ausentes, isso se deve pois para implementação
    # alguns modelos de machine learning e necessario remover esses conjuntos (null) ou utilizar alguma
    # tecnicas como Forward and Backward Propagation para substituir esses valores.

df2 = df.dropna(how='any',axis=0) 


# Utilizando a tecnica LabelEnconder
    # Codificamos as variaveis categoricas em valores numerico, sendo assim
    # a função labelEncoder codifica os rotulos com um valor entre 0 e numero de
    # rotulos

labelencoder = LabelEncoder()

df2['Star_color'] = labelencoder.fit_transform(df2['Star color'])
df2['Spectral_Class'] = labelencoder.fit_transform(df2['Spectral Class'])


# Criando as variaveis features(Caracteristicas) e labels (rotulos)

features = df2.drop(['Star type','Star color','Spectral Class'], axis = 1)
labels = df2['Star type']

# Escalando nosso modelo de treinamento
    # Atraves da função Standarcaler vamos criar um escala de valores 
    # para que nosso modelo possa empenhar melhor

scaler = StandardScaler()
scaled_train_features = scaler.fit_transform(features)


# Criando um variavel PCA (Ou Analise de componentes pricipais)
    # Como o propio no diz o intuito dessa tecanica e verificar quais
    # são as principais variaveis (ou componetes) que apresenta um certo nivel de
    # significancia no conjunto de dados

pca = PCA()
pca.fit(scaled_train_features)
exp_variance = pca.explained_variance_ratio_


# Grafico dos componetes principais

fig, ax = plt.subplots()
ax.bar(range(pca.n_components_), exp_variance)
ax.set_xlabel('N° de componentes principais')

# grafico de cutuvelo
    # atraves do grafico abaixo , conseguimos verificar
    # que Nº de componentes é no valor 2,
    # pois atraves da linha pontilihada azul , conseguimos
    # verificar que 85% pode ser explicada no componte 2

cum_exp_variance = np.cumsum(exp_variance)

fig, ax = plt.subplots()
ax.plot(cum_exp_variance)
ax.axhline(y=0.85, linestyle=':')


# Escolhendo o numero de compontes manualmente

n_component = 2

pca = PCA(n_component, random_state=10)
pca.fit(scaled_train_features)
pca_projection = pca.transform(scaled_train_features)


# Modelando - Part 2 (Arvore de deicosa e Regressao Logistica)

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# Divindo o conjunto de dados


train_features, test_features, train_labels, test_labels = train_test_split(pca_projection, labels, random_state=10)


# Arvore de decisão

dt = DecisionTreeClassifier(random_state=10)
dt.fit(train_features, train_labels)
pred_labels_tree = dt.predict(test_features)

# Modelo de regressao logistica

logreg = LogisticRegression(random_state=10)
logreg.fit(train_features, train_labels)
pred_labels_logit = logreg.predict(test_features)

# Validação

kf = KFold(n_splits=10)

tree = DecisionTreeClassifier()
logreg = LogisticRegression()


tree_score = cross_val_score(tree, pca_projection, labels, cv=kf)
logit_score = cross_val_score(logreg, pca_projection, labels, cv=kf)


# Verificando a média de cada modelo

print("Árvore de Decisão:", np.mean(tree_score),"regressao logistica:", np.mean(logit_score))
