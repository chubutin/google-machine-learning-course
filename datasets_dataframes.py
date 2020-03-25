import numpy as np
import pandas as pd

from utils import color

# Create an 3x4 (3 rows x 4 columns) pandas DataFrame in which the columns are named Eleanor, Chidi, Tahani, and Jason.
# Populate each of the 12 cells in the DataFrame with a random integer between 0 and 100, inclusive.

# create arrays
data = np.array([np.random.randint(low=0, high=101, size=4),
                 np.random.randint(low=0, high=101, size=4),
                 np.random.randint(low=0, high=101, size=4),
                 np.random.randint(low=0, high=101, size=4)])

my_columns = ["Eleanor", "Chidi", "Tahani", "Jason"]

my_dataframe = pd.DataFrame(data=data, columns=my_columns)

# Output the following:
#
# the entire DataFrame
# the value in the cell of row #1 of the Eleanor column

print("\nThe entire DataFrame")
print(my_dataframe)
print("\nThe value in the cell of row #1 of the Eleanor column")
print(my_dataframe["Eleanor"][1], "\n")

# Create a fifth column named Janet, which is populated with the row-by-row sums of Tahani and Jason.
my_dataframe["Janet"] = my_dataframe["Tahani"] + my_dataframe["Jason"]
print(color.BOLD + "\nCreate a fifth column named Janet, which is populated with the row-by-row sums of Tahani and Jason" + color.END)
print(my_dataframe)