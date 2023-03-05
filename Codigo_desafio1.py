# Pacotes utilizados

import pandas as pd
from selenium import webdriver
from bs4 import BeautifulSoup

# Criando um variavel com a url do site

url = 'https://www.basketball-database.com/csgc/league/nbb/3526#tab4'

# Criando uma sessão no Chrome para o site de Basktet. Vale ress

driver = webdriver.Chrome(executable_path='path_to_web_driver/')

# Carregando a página da web
driver.get(url)
# Aguardando a página carregar totalmente
driver.implicitly_wait(5)

soup = BeautifulSoup(driver.page_source, 'lxml')

tables = soup.find_all('table')

dfs = pd.read_html(str(tables))

# Visualizando a tabela dos players

print(dfs[3])

# Visualizando o total de tabelas existentes na pagina

print(f'Total tables: {len(dfs)}')

# Apos diversas tentativas 

# Não consegue encontrar uma forma de coletar os dados das outras 12 paginas, tentei utilizar o Beatifulsoup, selenium e etc.

# Ficaria muito feliz se o senhor pude-se emcaminhar um script com a forma correta de se carregar os dados das 12 paginas para intuito de aprendizado

# Atenciosamente Charles



