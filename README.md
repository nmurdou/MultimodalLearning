# Multimodal Learning 

This is a multimodal (vision, language) model for binary classification. The data used cannot be provided due to privacy issues.

The repo contains the following modules in the source directory:

* utils.py
* main.py
* inference.py
* unit_tests.py

We define all the necessary classes for the model in `utils.py`. Training and test performance estimation is done in the `main.py` module.

We save the checkpoint from the best model in the `./model_ouputs` folder. This is later used for inference and for the demo.
 
The `inference.py` module accepts as inputs from the command line the path to an unseen image and the title and yields the prediction. For this the `./model_ouputs` folder has to be in the same directory with the module.

Please go through the `demo.ipynp` notebook, where all the above is explained and demonstrated. A sample unit test is included in the `unit_test.py`.


