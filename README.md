# log_inf.
Arquivos Informatica

## Primeira Atividade: Apresentação Pessoal Slider usando  Canva 
https://www.canva.com/design/DAGfsex8VIQ/PrrSD1qal05fzlKFIe2B2w/view?utm_content=DAGfsex8VIQ&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=h46c816f8a7

![image](https://github.com/user-attachments/assets/e5dd2271-81dd-4d2c-a07e-93d50b9f7337)
# TERCEIRA ATIVIDADE DE INFORMATICA
1- Qual Bandeira Teve o Maior Valor de vendas ?
2- Qual Estado teve o MaioR Valor de Vendas 
3-Qual municipio teve o Maior valor em Vendas ?
![Captura de tela 2025-04-24 212803](https://github.com/user-attachments/assets/6129ef53-4022-42a9-814c-8d361e05e087)
https://fatecspgov-my.sharepoint.com/:u:/r/personal/joao_santos540_fatec_sp_gov_br/Documents/info-%20jo%C3%A3o%20vitor.pbix?csf=1&web=1&e=ZkhUjP
import kagglehub

# Download latest version
path = kagglehub.dataset_download("greenwing1985/housepricing")

print("Path to dataset files:", path)

# prompt: ler csv em dataframe os dados

import pandas as pd
import os

# Assuming the dataset contains a CSV file, find its path
# This part might need adjustment based on the actual structure of the downloaded dataset
csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]

if len(csv_files) > 0:
  csv_file_path = os.path.join(path, csv_files[0]) # Assuming the first CSV file is the one we want
  df = pd.read_csv(csv_file_path)
  print(df.head()) # Print the first few rows of the dataframe
else:
  print("No CSV files found in the downloaded dataset.")


# prompt: gerar mapa de correlação

import seaborn as sns
import matplotlib.pyplot as plt

# Calculate the correlation matrix
correlation_matrix = df.corr()

# Plot the correlation matrix as a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Map')
plt.show()

# prompt: gerar df sema coluna Prices

df_no_prices = df.drop('Prices', axis=1)
print(df_no_prices.head())

# prompt: gerar modelo de regressão linear  para prices

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Assuming 'df' is your pandas DataFrame loaded from the CSV

# Define features (X) and target (y)
# Here, we use all columns except 'Prices' as features
X = df.drop('Prices', axis=1)
y = df['Prices']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Print the coefficients and intercept
print(f'Coefficients: {model.coef_}')
print(f'Intercept: {model.intercept_}')

# prompt: you can print the model coefficients and intercep

# Print the coefficients and intercept
print(f'Coefficients: {model.coef_}')
print(f'Intercept: {model.intercept_}')
