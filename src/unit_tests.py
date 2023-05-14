import torch
import numpy as np
import pandas as pd
import unittest
from utils import DedupDataset
from sentence_transformers import SentenceTransformer
from pathlib import Path

CLIP = SentenceTransformer("clip-ViT-B-32")
for param in CLIP.parameters():
    param.requires_grad = False
tokenizer = CLIP._first_module().processor.tokenizer

data_dir = Path.cwd()
img_path = data_dir / "images"
sample_paths = [
    "1_0_-4416016469388839943.jpg",
    "1_1_-3673414655321898241.jpg",
    "1_2_1822361422379226510.jpg",
    "1_3_1822361422379226510.jpg",
    "1_4_1822361422379226510.jpg",
    "1_5_7120530738946192667.jpg",
    "1_6_-3673414655321898241.jpg",
    "1_7_7120530738946192667.jpg",
    "1_8_-4335113643265852997.jpg",
    "1_9_-3673414655321898241.jpg",
]
sample_images = [img_path / img for img in sample_paths]


class UnitTestDedupDataset(unittest.TestCase):
    def setUp(self):
        # Create a toy dataset with 10 samples
        self.sample_data = {
            "id": np.arange(10),
            "path": sample_images,
            "title": ["example title {}".format(i) for i in range(10)],
            "label": np.random.randint(0, 2, 10),
        }
        self.dataset = DedupDataset(
            data=pd.DataFrame(self.sample_data),
            image_directory=img_path,
            visual_transform=CLIP,
            textual_transform=CLIP,
        )

    def test_len(self):
        # Test that the length of the dataset is right
        self.assertEqual(len(self.dataset), 10)

    def test_getitem(self):
        # Test that the __getitem__ method outputs a sample dictionary with the expected keys
        sample = self.dataset[0]
        self.assertTrue(isinstance(sample, dict))
        self.assertTrue("id" in sample)
        self.assertTrue("image" in sample)
        self.assertTrue("text" in sample)
        self.assertTrue("label" in sample)

    def test_visual_transform(self):
        # Test that the visual transform is applied correctly to the image
        sample = self.dataset[0]
        image = sample["image"]
        self.assertTrue(isinstance(image, torch.Tensor))
        self.assertEqual(image.size(), torch.Size([512]))

    def test_textual_transform(self):
        # Test that the textual transform is applied correctly to the text
        sample = self.dataset[0]
        text = sample["text"]
        self.assertTrue(isinstance(text, torch.Tensor))
        self.assertEqual(text.size(), torch.Size([512]))


    def test_limit_dev_set(self):
        # Test that the dataset is limited to a specified size when limit_dev_set is set to a non-None value
        limited_dataset = DedupDataset(
            data=pd.DataFrame(self.sample_data),
            image_directory=img_path,
            visual_transform=CLIP,
            textual_transform=CLIP,
            limit_dev_set=5,
        )
        self.assertEqual(len(limited_dataset), 5)


if __name__ == "__main__":
    unittest.main()
