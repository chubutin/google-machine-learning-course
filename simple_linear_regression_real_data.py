import pandas as pd
from matplotlib import pyplot as plt

import os, requests

import model

if len(os.listdir('./csv_files/')) == 0:
    train_df_content = requests.get("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")
    test_df_content = requests.get("https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv")
    open('./csv_files/california_housing_train.csv', 'wb').write(train_df_content.content)
    open('./csv_files/california_housing_test.csv', 'wb').write(test_df_content.content)

train_df = pd.read_csv(open('./csv_files/california_housing_train.csv', 'r'))
test_df = pd.read_csv(open('./csv_files/california_housing_test.csv', 'r'))



# The following variables are the hyperparameters.
learning_rate = 0.06
epochs = 30
batch_size = 30

# Scale the label.
train_df["median_house_value"] /= 1000.0
train_df["rooms_per_person"] = train_df["total_rooms"] / train_df["population"]

pd.options.display.max_columns = None
pd.options.display.expand_frame_repr = False

print(train_df.head())

# Get statistics on the dataset.
print(train_df.describe())

print(train_df.corr())

# Specify the feature and the label.
my_feature = "median_income"  # the total number of rooms on a specific city block.
my_label = "median_house_value"  # the median value of a house on a specific city block.

my_model = model.build_model(learning_rate)
trained_weight, trained_bias, epochs, rmse = model.train_model(my_model, train_df, [my_feature],
                                                               [my_label], epochs,
                                                               batch_size)

print("\nThe learned weight for your model is %.4f" % trained_weight)
print("The learned bias for your model is %.4f\n" % trained_bias)

# model.plot_the_model(trained_weight, trained_bias, my_feature, my_label, train_df)
model.plot_the_loss_curve(epochs, rmse)


def predict_house_values(n, feature, label):
    """Predict house values based on a feature."""
    batch = train_df[feature][10000:10000 + n]
    predicted_values = my_model.predict_on_batch(x=batch)

    print("feature   label          predicted")
    print("  value   value          value")
    print("          in thousand$   in thousand$")
    print("--------------------------------------")
    for i in range(n):
        print("%5.0f %6.0f %15.0f" % (train_df[feature][i],
                                      train_df[label][i],
                                      predicted_values[i][0]))


predict_house_values(10, my_feature, my_label)
plt.show()
