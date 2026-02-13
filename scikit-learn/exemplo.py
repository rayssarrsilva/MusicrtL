from sklearn.linear_model import LinearRegression
import pandas as pd

# Dados fictícios
dados = pd.DataFrame({"tamanho":[50,80,120], "preco":[150000, 250000, 400000]})

X = dados[["tamanho"]]
y = dados["preco"]

modelo = LinearRegression()
modelo.fit(X, y)

print("Preço previsto para casa de 100m²:", modelo.predict([[100]]))


