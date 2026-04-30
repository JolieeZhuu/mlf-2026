import pycolmap
from pathlib import Path

database_path = Path("database.db")
image_path = Path("images/")

pycolmap.extract_features(database_path, image_path)

pycolmap.match_sequential(database_path)

output_path = Path("sparse_model/")
output_path.mkdir(exist_ok=True)

reconstructions = pycolmap.incremental_mapping(database_path, image_path, output_path)

print(reconstructions)