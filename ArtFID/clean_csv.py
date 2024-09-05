import csv
import PIL
from PIL import Image

# Open the CSV file and create a reader object
with open('output_dataset.csv', 'r') as infile:
    reader = csv.DictReader(infile)
    
    # Create a list of rows that meet the criteria
    idx = 0
    rows_to_keep = []
    for row in reader:
        idx += 1
        if row['artist'] is not None and row['style'] is not None and  row['status'] is not None and not row['status'].startswith('error') and row['status'] != 'failed':
            if (row['artist'] != 'None' and row['artist'] != '' and  row['style'] != 'None' and row['style'] != ''):
                rows_to_keep.append(row)
                img_path = row['filename']
                try:
                    image = Image.open(img_path).convert('RGB')
                    
                except (FileNotFoundError, PIL.UnidentifiedImageError):
                    # Skip over the image if it is corrupted or cannot be found
                    print("Skipping image at index", idx, "due to PIL.UnidentifiedImageError")
                    rows_to_keep.remove(row)
                    continue
           
        
    
# Open the CSV file for writing and create a writer object
with open('dataset_cleaned_3.csv', 'w', newline='') as outfile:
    # Write the header row
    writer = csv.DictWriter(outfile, fieldnames=['artist', 'style', 'url', 'filename', 'status'])
    writer.writeheader()
    
    # Write the rows that meet the criteria
    for row in rows_to_keep:
        writer.writerow(row)

