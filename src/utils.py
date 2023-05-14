import numpy as np
import pandas as pd
import torch
import torchvision
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score, f1_score
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from PIL import Image
from pathlib import Path
import random

# Instantiate clip-ViT-B-32 and freeze it. In addition, fix its tokenizer, which will be used
# to truncate titles that have more than 77 token when encoded. 77 is the maximum length for a textual input.
CLIP = SentenceTransformer('clip-ViT-B-32')
for param in CLIP.parameters():
    param.requires_grad = False
tokenizer = CLIP._first_module().processor.tokenizer


def truncate_title(title, tokenizer):
    """
    Truncate sentences that exceed the CLIP max token limit (77 tokens including the
    starting and ending tokens).

    Args:
        title(string): The sentence to truncate.
        tokenizer(CLIPTokenizer): Pretrained CLIP tokenizer.
    Returns:
        string: truncated sentence
    """

    cur_title = title
    tokens = tokenizer.encode(cur_title)

    if len(tokens) > 77:
        # Omit the first token, hence return 75 tokens
        truncated_tokens = tokens[1:76]
        cur_title = tokenizer.decode(truncated_tokens)

        # Recursive call, as the encode(decode()) could yield different result
        return truncate_title(cur_title, tokenizer)

    else:
        return cur_title


class DedupDataset(torch.utils.data.Dataset):
    """Preprocesses and serves
    dict of multimodal tensors as model input.
    """

    def __init__(
            self,
            data,
            image_directory,
            visual_transform,
            textual_transform,
            limit_dev_set=None,
            random_state=42,
    ):
        """
        Initialize a new DedupDataset.

        Args:
            data (pd.DataFrame): Dataframe containing image paths and labels.
            image_directory (str): Path to directory containing images.
            visual_transform (callable): A function/transform that takes in an PIL image
                and returns a transformed version of the image.
            textual_transform (callable): A function/transform that takes in a string of text
                and returns a transformed version of the text.
            limit_dev_set (int, optional): If specified, limits the size of the dataset. Use during development.
                Default is None.
            random_state (int, optional): Seed for random operations.
                Default is 42.
        """

        self.data_frame = data
        self.limit_dev_set = limit_dev_set
        if self.limit_dev_set:
            if self.data_frame.shape[0] > self.limit_dev_set:
                self.data_frame = self.data_frame.sample(
                    limit_dev_set, random_state=random_state
                )
        self.data_frame = self.data_frame.reset_index(
            drop=True
        )

        self.visual_transform = visual_transform
        self.textual_transform = textual_transform

    def __len__(self):
        """
           Returns the number of samples in the dataset.
           """

        return len(self.data_frame)

    def __getitem__(self, idx):
        """
           Returns the sample at the given index.

           Args:
               idx (int): Index of the sample to return.

           Returns:
               dict: A dictionary containing image, text, and label tensors.
        """

        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_id = self.data_frame.loc[idx, "id"]

        image = Image.open(
            self.data_frame.loc[idx, "path"]
        ).convert("RGB")
        image = torch.Tensor(self.visual_transform.encode(image))
        title = truncate_title(self.data_frame.loc[idx, "title"], tokenizer)

        text = torch.Tensor(
            self.textual_transform.encode(
                title
            )
        )

        if "label" in self.data_frame.columns:
            label = torch.Tensor(
                [self.data_frame.loc[idx, "label"]]
            ).long().squeeze()
            sample = {
                "id": image_id,
                "image": image,
                "text": text,
                "label": label
            }
        else:
            sample = {
                "id": image_id,
                "image": image,
                "text": text
            }

        return sample


class ConcatenateLanguageAndVision(torch.nn.Module):
    def __init__(
            self,
            num_cl,
            loss_function,
            lang_module,
            vis_module,
            lang_feature_dim,
            vis_feature_dim,
            fusion_out_dim,
            dropout_probability,

    ):
        """
        Initializes a module that concatenates language and vision features, fuses them, and outputs the predictions.

        Args:
            num_cl (int): The number of output classes.
            loss_function (function): The loss function to use for training.
            lang_module (torch.nn.Module): The module for processing language inputs.
            vis_module (torch.nn.Module): The module for processing visual inputs.
            lang_feature_dim (int): The number of output features from the language module.
            vis_feature_dim (int): The number of output features from the vision module.
            fusion_out_dim (int): The number of output features after fusing the language and vision features.
            dropout_probability (float): The probability of dropping out a neuron during training.

        Returns:
            None
        """
        super(ConcatenateLanguageAndVision, self).__init__()
        self.language_module = lang_module
        self.vision_module = vis_module
        self.fusion = torch.nn.Linear(
            in_features=(lang_feature_dim + vis_feature_dim),
            out_features=fusion_out_dim
        )
        self.fc = torch.nn.Linear(
            in_features=fusion_out_dim,
            out_features=num_cl
        )
        self.loss_function = loss_function
        self.dropout = torch.nn.Dropout(dropout_probability)

    def forward(self, text_inputs, image_inputs, labels=None):
        """
        Concatenates language and vision features, fuses them, and returns the predictions.

        Args:
            text_inputs (torch.Tensor): The input text.
            image_inputs (torch.Tensor): The input images.
            labels (torch.Tensor): The target labels.

        Returns:
            tuple: A tuple containing the predictions and the loss (if labels are provided).
        """
        language_features = torch.nn.functional.relu(
            self.language_module(text_inputs)
        )
        vision_features = torch.nn.functional.relu(
            self.vision_module(image_inputs)
        )
        combined_features = torch.cat(
            [language_features, vision_features], dim=1
        )
        fused_features = self.dropout(
            torch.nn.functional.relu(
                self.fusion(combined_features)
            )
        )
        logits = self.fc(fused_features)
        predictions = torch.nn.functional.softmax(logits)
        loss = (
            self.loss_function(predictions, labels)
            if labels is not None else labels
        )
        return (predictions, loss)


class DedupModel(pl.LightningModule):
    def __init__(self, hyperparams):
        """
        Initializes the DedupModel.

        Args:
            hyperparams (Namespace): Namespace containing the hyperparameters.
        """

        super(DedupModel, self).__init__()
        self.params = hyperparams

        self.embedding_dim = self.params.get("embedding_dim", 512)
        self.lang_feature_dim = self.params.get(
            "lang_feature_dim", 300
        )
        self.vis_feature_dim = self.params.get(
            # balance language and vision features by default
            "vis_feature_dim", self.lang_feature_dim
        )
        self.out_path = Path(
            self.params.get("out_path", "model_outputs")
        )

        self.textual_transform = self._create_textual_transform()
        self.visual_transform = self._create_visual_transform()
        self.train_dataset = self._create_dataset("train_data")
        self.val_dataset = self._create_dataset("val_data")

        # set up model and training
        self.model = self._create_model()
        self.trainer_params = self._set_trainer_params()
        self.validation_step_outputs = []

    # Required Lightning Methods

    def forward(self, text, image, label=None):
        return self.model(text, image, label)

    def training_step(self, batch, batch_nb):
        preds, loss = self.forward(
            text=batch["text"],
            image=batch["image"],
            label=batch["label"]
        )
        return {"loss": loss}

    def validation_step(self, batch, batch_nb):
        preds, loss = self.eval().forward(
            text=batch["text"],
            image=batch["image"],
            label=batch["label"]
        )
        self.validation_step_outputs.append(loss)
        self.log("avg_val_loss", loss, prog_bar=True)
        return {"batch_val_loss": loss}

    def on_validation_epoch_end(self):

        avg_loss = torch.stack(self.validation_step_outputs).mean()
        self.validation_step_outputs.clear()
        return {
            "val_loss": avg_loss,
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.params.get("lr", 0.001)
        )

        return {"optimizer": optimizer,
                "lr_scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
                "monitor": "avg_val_loss"}

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.params.get("batch_size", 4),
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.params.get("batch_size", 4),
        )

    def fit(self):

        self._set_seed(self.params.get("random_state", 42))
        self.trainer = pl.Trainer(**self.trainer_params)
        self.trainer.fit(self)

    def _set_seed(self, seed):
        """
        Sets the random seed for reproducibility.

        Args:
            seed (int): The random seed value.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _create_textual_transform(self):
        """
        Creates and returns the textual transform for the dataset.

        Returns:
            (SentenceTransformer): The textual transform.
        """

        language_transform = CLIP
        return language_transform

    def _create_visual_transform(self):
        """
        Creates and returns the visual transform for the dataset.

        Returns:
            (torchvision.transforms.Compose): The visual transform.
        """
        visual_transform = CLIP
        return visual_transform

    def _create_dataset(self, dataset_key):
        """
        Creates and returns the dataset for the given dataset key.

        Args:
            dataset_key (str): The key of the dataset.

        Returns:
            (DedupDataset): The dataset.
        """
        return DedupDataset(
            data=self.params[dataset_key],
            image_directory=self.params.get("img_dir"),
            visual_transform=self.visual_transform,
            textual_transform=self.textual_transform,
            # limit training samples only
            limit_dev_set=(
                self.params.get("dev_limit", None)
                if "train" in str(dataset_key) else None
            )
        )

    def _create_model(self):
        """
        Creates the model.

        Returns:
            (ConcatenateLanguageAndVision): Model.
        """

        language_module = torch.nn.Linear(
            in_features=self.embedding_dim,
            out_features=self.lang_feature_dim
        )

        vision_module = torch.nn.Linear(
            in_features=512,
            out_features=self.vis_feature_dim
        )

        return ConcatenateLanguageAndVision(
            num_cl=self.params.get("num_classes", 2),
            loss_function=torch.nn.CrossEntropyLoss(),
            lang_module=language_module,
            vis_module=vision_module,
            lang_feature_dim=self.lang_feature_dim,
            vis_feature_dim=self.vis_feature_dim,
            fusion_out_dim=self.params.get(
                "fusion_output_size", 512
            ),
            dropout_probability=self.params.get("dropout_p", 0.1),
        )

    def _set_trainer_params(self):
        """
        Sets the parameters for the model trainer.

        Returns:
        Dict: Dictionary containing the parameters for the model trainer.
        """
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=self.out_path,
            monitor=self.params.get(
                "checkpoint_monitor", "avg_val_loss"
            ),
            mode=self.params.get(
                "checkpoint_monitor_mode", "min"
            ),
            verbose=self.params.get("verbose", True),
            save_top_k=1

        )

        early_stop_callback = pl.callbacks.EarlyStopping(
            monitor=self.params.get(
                "early_stop_monitor", "avg_val_loss"
            ),
            min_delta=self.params.get(
                "early_stop_min_delta", 0.001
            ),
            patience=self.params.get(
                "early_stop_patience", 3
            ),
            verbose=self.params.get("verbose", True),
        )

        trainer_params = {
            "callbacks": [early_stop_callback, checkpoint_callback],
            "default_root_dir": self.out_path,
            "accumulate_grad_batches": self.params.get(
                "accumulate_grad_batches", 1
            ),
            "num_nodes": self.params.get("n_gpu", 1),
            "max_epochs": self.params.get("max_epochs", 100),
            "gradient_clip_val": self.params.get(
                "gradient_clip_value", 1
            ),
        }
        return trainer_params

    @torch.no_grad()
    def test_metrics(self):
        """
       Computes the Area Under the Curve (AUC) for the Receiver Operator Curve.
       Computes the F1 score for the classifier.

       Returns:
       float: The AUC on the test set.
       float: The F1 score on the test set.
        """
        test_dataset = self._create_dataset("test_data")
        test_predictions = pd.DataFrame(
            index=test_dataset.data_frame.id,
            columns=["proba", "label"]
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=self.params.get("batch_size", 4))
        for batch in tqdm(test_dataloader, total=len(test_dataloader)):
            preds, _ = self.model.eval()(
                batch["text"], batch["image"]
            )
            test_predictions.loc[batch["id"], "proba"] = preds[:, 1]
            test_predictions.loc[batch["id"], "label"] = preds.argmax(dim=1)
        test_predictions.proba = test_predictions.proba.astype(float)
        test_predictions.label = test_predictions.label.astype(int)
        return roc_auc_score(test_dataset.data_frame.label, test_predictions.proba), \
            f1_score(test_dataset.data_frame.label, test_predictions.label)

    def inference_sample(self, unseen_img_path, unseen_title):
        """
        This function takes a pre-trained CLIP model and uses it to make predictions on an unseen image and title, after
        preprocessing the title using a tokenizer and truncating it to a maximum length.

        Args:
            model: A pre-trained DedupModel model.
            unseen_img_path (str): The path to an unseen image.
            unseen_title (str): The title or description of the image.

        Returns:
            prediction (int): The predicted label index for the image.

        """
        title_embedding = torch.unsqueeze(torch.Tensor(CLIP.encode(truncate_title(unseen_title, tokenizer))), 0)
        img_embedding = torch.unsqueeze(torch.Tensor(CLIP.encode(Image.open(unseen_img_path).convert("RGB"))), 0)
        preds, _ = self.model.eval()(title_embedding, img_embedding)
        prediction = int(preds.argmax(dim=1).item())
        return prediction