import os
import requests
import csv
from tqdm import tqdm

# create a directory to store the images
if not os.path.exists('dataset'):
    os.makedirs('dataset')

# Open the CSV file and create a new CSV file to write the output
with open('artfid_dataset.csv', 'r') as f_input, open('output_dataset.csv', 'w', newline='') as f_output:
    csv_input = csv.reader(f_input)
    csv_output = csv.writer(f_output)

    # Add a new header row for the status column
    header = next(csv_input)
    header.append('filename')
    # csv_output.writerow(header)

    # Add a new header row for the status column
    # header = next(csv_output)
    header.append('status')
    csv_output.writerow(header)

    # Count the total number of rows in the input CSV file
    num_rows = sum(1 for _ in f_input)
    f_input.seek(0)  # reset the file pointer to the beginning

    idx = 1
    # Iterate over the rows in the input CSV file, with a progress bar
    for row in tqdm(csv_input, total=num_rows):

        if row[2] == 'url':
            continue
        # Get the URL for the image from the row
        url = row[2].strip()
        artist = row[0].strip().replace(' ', '')
        artist = artist.replace(' ','').replace('/','').replace('.', '').replace(',','')
        style = row[1].strip().replace(' ', '')
        filesavename = str(idx) + '_' + artist + '_' + style + '.jpg'

        # Attempt to download the image using the URL
        try:
            response = requests.get(url)
            if response.status_code == 200:
                # Save the image to a file in the 'images' directory
                filename = os.path.join('dataset', filesavename) # url.split('/')[-1][:12]+'.jpg')# style + '_' + artist + '.jpg'  # url.split('/')[-1])
                row.append(filename)
                with open(filename, 'wb') as f:
                    f.write(response.content)
                # Add a new row to the output CSV file with the status 'downloaded'
                row.append('downloaded')
            else:
                # Add a new row to the output CSV file with the appropriate error status code
                row.append('error ' + str(response.status_code))
        except requests.exceptions.RequestException as e:
            # Add a new row to the output CSV file with the status 'failed'
            row.append('failed')

        # Write the new row to the output CSV file
        csv_output.writerow(row)

        # increment
        idx += 1
