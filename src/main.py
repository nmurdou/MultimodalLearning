import logging
import warnings
import pandas as pd
from pathlib import Path
from utils import DedupModel

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.WARNING)


data_dir = Path.cwd()
img_path = data_dir / "images"

train_data = pd.read_csv("train_data.csv").dropna()
val_data = pd.read_csv("val_data.csv").dropna()
test_data = pd.read_csv("test_data.csv").dropna()


hyperparams = {

    "train_data": train_data,
    "val_data": val_data,
    "test_data": test_data,
    "img_dir": img_path,


    "embedding_dim": 512,
    "lang_feature_dim": 500,
    "vis_feature_dim": 600,
    "fusion_output_size": 256,
    "out_path": "model_outputs",
    "dev_limit": None,
    "lr": 0.0005,
    "max_epochs": 12,
    "n_gpu": 0,
    "batch_size": 64,
    "accumulate_grad_batches": 16,
    "early_stop_patience": 3,
}


if __name__ == '__main__':

    dedup_model = DedupModel(hyperparams=hyperparams)
    dedup_model.fit()

    checkpoints = list(Path("model_outputs").glob("*.ckpt"))

    dedup_model = DedupModel.load_from_checkpoint(
        checkpoints[0],
        hyperparams=hyperparams
    )
    test_roc, f1 = dedup_model.test_metrics()
    print(f"test ROC AUC score: {test_roc} and test F1 score is: {f1}")

