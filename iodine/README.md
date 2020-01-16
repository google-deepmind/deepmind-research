# IODINE
Reference implementation for the paper ["Multi-Object Representation Learning with Iterative Variational Inference"](https://arxiv.org/abs/1903.00450).
This repository contains:

* An IODINE implementation in Tensorflow v1.
* Configurations used in the paper (checkpoints available in Cloud Storage) for:
  * CLEVR
  * Multi-dSprites
  * Tetrominoes
* A notebook for running and inspecting the model and plotting the results


## Installation
1. Clone the DeepMind research repository:

    ``` bash
    git clone https://github.com/deepmind/deepmind-research.git
    cd deepmind-research
    ```

2. Download the checkpoints from GCP. A shell script is provided:

   ```bash
   ./iodine/download_checkpoints.sh
   ```

   On platforms without wget, the files can be downloaded from [this webpage](https://console.cloud.google.com/storage/browser/deepmind-research-iodine?pli=1)
   and the unzipped `checkpoints/` folder should be placed in
   `deepmind-research/iodine/checkpoints`.


3. Prepare a Python 3 environment - virtualenv is recommended.

   ```bash
   python3 -m venv iodine_venv
   source iodine_venv/bin/activate
   ```

4. Install dependencies:

   ```bash
   pip3 install -r iodine/requirements.txt
   ```

5. The `multi_object_datasets` package installed via requirements.txt provides python code to open the data files, but not the data files themselves.
   Download the desired datasets either manually from the [Google Cloud Storage](https://console.cloud.google.com/storage/browser/multi-object-datasets) or using the commands below:

    ```bash
    pushd iodine/multi_object_datasets
    # CLEVR
    wget https://storage.googleapis.com/multi-object-datasets/clevr_with_masks/clevr_with_masks_train.tfrecords
    # Multi-dSprites
    wget https://storage.googleapis.com/multi-object-datasets/multi_dsprites/multi_dsprites_colored_on_grayscale.tfrecords
    # Tetrominoes
    wget https://storage.googleapis.com/multi-object-datasets/tetrominoes/tetrominoes_train.tfrecords
    # Get back to location containing 'iodine' directory
    popd
    ```

    See [multi_object_datasets repository](https://github.com/deepmind/multi_object_datasets)
    for further details.
6. Make sure that you have CUDA 10 and CuDNN 7 installed


## Interact with a Model
Use the jupyter notebook `Eval.ipynb` to load and run one of the checkpoints.
It also contains code to plot the outputs and latent traversals.


## Train a Model
To train your own model use the [Sacred](https://github.com/IDSIA/sacred) experiment defined in `main.py`.
The configurations used in the paper for the different datasets are available as [named configs](https://sacred.readthedocs.io/en/latest/configuration.html#named-configurations) inside of `configuration.py`.
### Train a new model
 * CLEVR6

    ```bash
    python3 -m iodine.main -f with clevr6
    ```

 * Multi-dSprites

    ```bash
    python3 -m iodine.main -f with multi_dsprites
    ```

 * Tetrominoes

    ```bash
    python3 -m iodine.main -f with tetrominoes
    ```

It is recommended to add an observer to your run to let Sacred record the details of run.
To add a [FileStorageObserver](https://sacred.readthedocs.io/en/latest/command_line.html#filestorage-observer) add `-F my_storage_dir`, and add `-m my_db_name` for a [MongoObserver](https://sacred.readthedocs.io/en/latest/command_line.html#mongodb-observer).

### Adjusting Config Values
The experiment has a configuration that can be printed and adjusted from the commandline. E.g.:

``` bash
# print configuration
python3 -m iodine.main -f print_config with clevr6
# run experiment after adjusting batch_size and the size of the shuffle buffer
python3 -m iodine.main -f with clevr6 batch_size=2 data.shuffle_buffer=100
```

### Tensorboard
Each run stores checkpoints and summaries in the directory specified by `checkpoint_dir`, to which a suffix based on the run_id is appended.
If an observer is added the `run_id` is set automatically. Otherwise it should be set manually using e.g. `run_id=5`.

Summaries can be viewed using tensorboard. E.g. like this for clevr6 (assuming `run_id=1`):

```bash
tensorboard --log-dir iodine/checkpoints/clevr6_1
```

### Continue Previous Run
To continue a previous run pass `continue_run=True` and the path of the checkpoints:

```bash
python3 -m iodine.main -f with clevr6 checkpoint_dir=iodine/checkpoints/clevr6_1
```

## Code Structure
The main experiment defined in `main.py` uses `sacred` and the configurations for the different datasets are added as named configs and can be found in `configuration.py`.
The model implementation can be found in the `modules` directory and is based on `tensorflow` and `sonnet`:

 * `iodine.py` The main IODINE module that assembles the decoder, refinement network, distributions and factor regressor.
 * `decoder.py` The ComponentDecoder which is a wrapper around networks that takes care of splitting the output channels into means and masks.
 * `refinement.py` The refinement components assembles the encoder network, LSTM and refinement head.
 * `networks.py` Different standard networks such as CNN, BroadcastCNN, and LSTM.
 * `distribution.py` Definition of the latent and pixel distributions.
 * `factor_eval.py` Contains the factor regressor which predicts the true factors from the inferred object latents.
 * `data.py` Dataset wrappers around `multi_object_datasets` that take care of shuffling, batching and preprocessing.
 * `plotting.py` Helper functions for plotting results.
 * `utils.py` General helper functions.


---
**DISCLAIMER**

This is not an officially supported Google product.

---
