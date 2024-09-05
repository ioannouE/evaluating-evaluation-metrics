import csv
import json

# Define the path to the CSV file containing the dataset
csv_path = "dataset_cleaned_3.csv"

# Define the number of classes (i.e., number of unique artist labels)
num_classes = 3355

# Initialize an empty dictionary to hold the label mappings
label_map = {}

# Open the CSV file and read the dataset
with open(csv_path, 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        # Extract the artist label from the current row
        artist_label = row['artist'].strip()
        
        # If the artist label is not in the label map, add it with a new integer key
        if artist_label not in label_map and len(label_map) < num_classes:
            label_map[artist_label] = len(label_map)
        
        # If we've reached the maximum number of classes, break the loop
        if len(label_map) == num_classes:
            break

# # Save the label map to a file
# with open("label_map.py", "w") as file:
#     file.write("label_map = " + str(label_map))

with open('labels_dicts/artists_3k_dict.json', 'w', encoding='utf8') as handle:
    json.dump(label_map, handle, ensure_ascii=False)