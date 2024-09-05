import h5py
import pandas as pd
from PIL import Image
from torchvision import transforms
import os

transform = transforms.Compose([
    # transforms.Resize((299, 299)),
    transforms.ToTensor(),
   #  transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

csv_file = 'output_dataset.csv'
img_dir = '' # full path provided in csv file
output_file = 'art_dataset.h5'

# Open the output HDF5 file for writing
with h5py.File(output_file, 'w') as f:
    # Create a group for the images
    images_group = f.create_group('images')
    
    # Create a group for the artist labels
    artist_group = f.create_group('artist_labels')
    
    # Create a group for the style labels
    style_group = f.create_group('style_labels')
    
    # Read in the CSV file and iterate over the rows
    data = pd.read_csv(csv_file)
    for i in range(len(data)):
        # Load the image from disk
        # img_style = str(data.iloc[i, 1]).strip()
       
        # img_artist = str(data.iloc[i, 0]).strip().replace(' ','').replace('/','').replace('.', '').replace(',','') +'.jpg'
        # img_name = img_style  +'_'+str(i) + '_' + img_artist # data.iloc[i, 2]
        img_name = data.iloc[i,3]
        img_path = os.path.join(img_dir, img_name)

        # check if image has been downloaded
        if (data.iloc[i,4] == 'downloaded'):
            image = Image.open(img_path).convert('RGB')
            
            # Apply the image transformations
            image = transform(image)
            
            # Get the artist and style labels
            artist_label = data.iloc[i, 0]
            style_label = data.iloc[i, 1]
            
            # Create a dataset for the image, artist label, and style label
            images_group.create_dataset(str(i), data=image.numpy())
            artist_group.create_dataset(str(i), data=artist_label)
            style_group.create_dataset(str(i), data=style_label)
