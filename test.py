import pandas as pd

# Sample DataFrame
df = pd.DataFrame({
    'ID': [1, 2, 3],  # Sample IDs
    'Name': ['Alice', 'Bob', 'Charlie']  # Sample names
})

# Desired ID value
desired_id = 10

# Set the desired ID value at a specific row index (e.g., row index 1 in this example)
row_index_to_update = 1
df.at[row_index_to_update, 'ID'] = desired_id

# Display the updated DataFrame
print("Updated DataFrame:")
print(df)