import os
import pandas as pd
from shutil import copy2

csv_path = 'data/tabular/pokedex.csv'  # path to the pokedex.csv file
images_dir = 'data/images'  # path to the images directory
output_dir = 'data/images_by_type'  # path to save sorted images

# Create the output directory
os.makedirs(output_dir, exist_ok=True)

# Read the CSV file and iterate over the rows of the pokedex dataframe
pokedex = pd.read_csv(csv_path)

for _, row in pokedex.iterrows():
    pokemon_name = row['Name']
    pokemon_type = row['Type 1']
    image_name = f"{pokemon_name}.png"

    # Define the source and destination paths for the image
    type_dir = os.path.join(output_dir, pokemon_type)
    os.makedirs(type_dir, exist_ok=True)
    source_path = os.path.join(images_dir, image_name)
    dest_path = os.path.join(type_dir, image_name)

    # Copy the image to the appropriate folder
    if os.path.exists(source_path):
        copy2(source_path, dest_path)
    else:
        print(f"Image not found for Pokémon {pokemon_name}: {image_name}")

print("Images have been sorted into folders by type!")