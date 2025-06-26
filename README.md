# Nominal Data Encoding using One-Hot Encoding (OHE)

## Overview

This project demonstrates how to encode nominal (categorical) data using One-Hot Encoding (OHE) in Python with the help of pandas and scikit-learn. One-Hot Encoding is a common preprocessing step in machine learning pipelines, as most algorithms require numerical input.

## What is One-Hot Encoding?

**One-Hot Encoding** is a technique to convert categorical variables (with no intrinsic ordering) into a format that can be provided to ML algorithms. For each unique category, a new binary column is created. Each row will have a `1` in the column corresponding to its category and `0` elsewhere.

For example, for a `color` column with values `['red', 'green', 'blue']`, OHE will create three columns: `color_red`, `color_green`, and `color_blue`.

## Code Explanation

Below is the complete code with detailed explanations:

```python
# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Step 1: Create a sample DataFrame
df = pd.DataFrame({'color': ['red', 'green', 'blue', 'blue', 'red', 'green']})

# Display the original DataFrame
print("Original DataFrame:")
print(df)

# Step 2: Initialize the OneHotEncoder
# We use sparse=False to get a dense array as output (easier to convert to DataFrame)
encoder = OneHotEncoder(sparse=False)

# Step 3: Fit and transform the data
# The encoder expects a 2D array, so we use double brackets [[ ]] to select the column
encoded_array = encoder.fit_transform(df[['color']])

# Step 4: Create a DataFrame with the encoded data
# Get the new column names from the encoder
encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(['color']))

# Step 5: Concatenate the original and encoded DataFrames (optional)
final_df = pd.concat([df, encoded_df], axis=1)

# Display the final DataFrame
print("\nDataFrame after One-Hot Encoding:")
print(final_df)
```

## Output

### Original DataFrame

|   | color |
|---|-------|
| 0 | red   |
| 1 | green |
| 2 | blue  |
| 3 | blue  |
| 4 | red   |
| 5 | green |

### DataFrame after One-Hot Encoding

|   | color | color_blue | color_green | color_red |
|---|-------|------------|-------------|-----------|
| 0 | red   | 0.0        | 0.0         | 1.0       |
| 1 | green | 0.0        | 1.0         | 0.0       |
| 2 | blue  | 1.0        | 0.0         | 0.0       |
| 3 | blue  | 1.0        | 0.0         | 0.0       |
| 4 | red   | 0.0        | 0.0         | 1.0       |
| 5 | green | 0.0        | 1.0         | 0.0       |

## Key Points

- **One-Hot Encoding** is essential for converting categorical (nominal) data into a machine-readable format.
- Each unique category gets its own column.
- Use `OneHotEncoder` from `sklearn.preprocessing` for efficient encoding.
- Always check the resulting DataFrame to ensure correct encoding.

## Requirements

- Python 3.x
- pandas
- scikit-learn
