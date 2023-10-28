import pandas as pd
import numpy as np
df_customer = pd.read_csv('C:/Users/hp/Downloads/PortoBuilder/Kalbe Internship/Final Project/Case Study Data Scientist/Case Study - Customer.csv', delimiter = ';') 
df_store = pd.read_csv('C:/Users/hp/Downloads/PortoBuilder/Kalbe Internship/Final Project/Case Study Data Scientist/Case Study - Store.csv', delimiter = ';')
df_product = pd.read_csv('C:/Users/hp/Downloads/PortoBuilder/Kalbe Internship/Final Project/Case Study Data Scientist/Case Study - Product.csv', delimiter = ';') 
df_transaction = pd.read_csv('C:/Users/hp/Downloads/PortoBuilder/Kalbe Internship/Final Project/Case Study Data Scientist/Case Study - Transaction.csv', delimiter = ';')
df_customer.shape, df_store.shape, df_product.shape, df_transaction.shape
df_customer.head()
df_customer.info()
df_store.head()
df_store.info()
df_product.head()
df_product.info()
df_transaction.head()
df_transaction.info()
# Print rows where 'Marital Status' column is null
null_marital_status = df_customer[df_customer['Marital Status'].isnull()]
print("Rows with null 'Marital Status' column:")
print(null_marital_status)
# Fill null values in 'Marital Status' column with the mode
mode_marital_status = df_customer['Marital Status'].mode()[0]
df_customer['Marital Status'].fillna(mode_marital_status, inplace=True)
df_customer.info()
# Remove non-numeric characters and whitespace from 'Latitude' and 'Longitude' columns
df_store['Latitude'] = df_store['Latitude'].str.replace('[^\d.]+', '', regex=True).astype(float)
df_store['Longitude'] = df_store['Longitude'].str.replace('[^\d.]+', '', regex=True).astype(float)

# Verify the changes
print(df_store.info())

# Check if 'TransactionID' column has duplicates
has_duplicate_transaction_ids = df_transaction['TransactionID'].duplicated().any()

if has_duplicate_transaction_ids:
    print("The 'TransactionID' column has duplicates.")
else:
    print("The 'TransactionID' column is unique.")
# Check for duplicate rows based on 'TransactionID' column
duplicate_transaction_ids = df_transaction[df_transaction.duplicated(subset='TransactionID', keep=False)]

# Print the rows with duplicate 'TransactionID'
print("Rows with duplicate 'TransactionID':")
print(duplicate_transaction_ids)

#Proof
df_transaction[df_transaction.TransactionID == 'TR91651']
# Convert the "Date" column to datetime data type with the correct format (assuming "dd/mm/yyyy")
df_transaction['Date'] = pd.to_datetime(df_transaction['Date'], format='%d/%m/%Y')

# Sort the DataFrame by 'TransactionID' and 'Date' in descending order
df_transaction.sort_values(by=['TransactionID', 'Date'], ascending=[True, False], inplace=True)

# Remove duplicates, keeping only the rows with the latest 'Date' for each 'TransactionID'
df_transaction.drop_duplicates(subset='TransactionID', keep='first', inplace=True)

# Reset the index if necessary
df_transaction.reset_index(drop=True, inplace=True)

# Print the updated DataFrame
print(df_transaction)

df_merged = pd.merge (df_transaction, df_product, on= 'ProductID', how='left')
df_merged = pd.merge (df_merged, df_store, on= 'StoreID', how='inner')
df_merged = pd.merge (df_merged, df_customer, on= 'CustomerID', how='inner')
df_merged.head()
df_merged.info()
import nbformat

def ipynb_to_py(input_notebook, output_script):
    with open(input_notebook, 'r', encoding='utf-8') as nb_file:
        notebook = nbformat.read(nb_file, as_version=4)
    
    code_cells = []
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            code_cells.append(cell.source)
    
    with open(output_script, 'w', encoding='utf-8') as py_file:
        py_file.write('\n'.join(code_cells))

# Replace 'input_notebook.ipynb' with the name of your input Jupyter Notebook file
# Replace 'output_script.py' with the desired name for the output Python script file
input_notebook = 'input_notebook.ipynb'
output_script = 'output_script.py'

ipynb_to_py(input_notebook, output_script)

df_merged.to_csv('C:/Users/hp/Downloads/PortoBuilder/Kalbe Internship/Final Project/Case Study Data Scientist/merged_data.csv', index=False)