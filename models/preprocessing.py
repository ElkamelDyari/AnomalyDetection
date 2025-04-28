import pandas as pd


def remove_low_variance(path, threshold):
    data = pd.read_csv(path)
    # Get the templates that appear only once
    single_occurrence_templates = data['template'].value_counts()[data['template'].value_counts() > threshold].index

    # Filter the original DataFrame using the 'isin' method
    filtered_data = data[data['template'].isin(single_occurrence_templates)]

    # Now you can work with the filtered_data DataFrame
    return filtered_data




path = "data/drain/bgl_templates_from_content.csv"
threshold = 2
filtered_data = remove_low_variance(path, threshold)
filtered_data = filtered_data.drop(["log", "params"], axis=1)
filtered_data.to_csv("data/preprocessed/filtered_data.csv", index=False)
print("Filtered data saved to data/preprocessed/filtered_data.csv")