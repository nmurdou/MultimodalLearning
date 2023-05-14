import logging
import warnings
from pathlib import Path

import pandas as pd

from utils import DedupModel

# Disable warnings and set logging level
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.WARNING)

# Request user for directory of new image and possible duplicate title
unseen_img_path = input("Enter the path of image: ")
unseen_title = input("Enter the title: ")

# Instantiate the model
dummy_data = pd.DataFrame({"title": [unseen_title],
                           "path": [unseen_img_path],
                           "id": [5],
                           "label": [0]})
dummy_path = Path.cwd()

hyperparams = {
    # Required hyperparams
    "train_data": dummy_data,
    "val_data": dummy_data,
    "test_data": dummy_data,
    "img_dir": dummy_path,
    "embedding_dim": 512,
    "lang_feature_dim": 500,
    "vis_feature_dim": 600,
    "fusion_output_size": 256,
    "out_path": "model_outputs",
}

# Load from last checkpoint

checkpoints = list(Path("model_outputs").glob("*.ckpt"))

dedup_model = DedupModel.load_from_checkpoint(
    checkpoints[0],
    hyperparams=hyperparams
)

# Predict 1 if we have a duplicate pair, 0 otherwise
prediction = dedup_model.inference_sample(unseen_img_path, unseen_title)

print("Text and image represent the same advertisement") if prediction else print("Text and image are from different advertisements")