import pandas as pd

# Load the CSV file
df = pd.read_csv('3param_sobol_results.csv')

# Sort by the first column (assumed to be 'index' or unnamed)
first_col = df.columns[0]
df_sorted = df.sort_values(by=first_col).reset_index(drop=True)

# Check that each index is 1+ of the one before it, and report where it breaks
indices = df_sorted[first_col].values
is_sequential = True
for i in range(1, len(indices)):
    if indices[i] != indices[i-1] + 1:
        print(f"Sequentiality breaks at line {i}: {indices[i-1]} -> {indices[i]}")
        is_sequential = False

print(f"Indices are sequential: {is_sequential}")

# Optionally, save the sorted DataFrame
df_sorted.to_csv('3param_sobol_results_sorted.csv', index=False)