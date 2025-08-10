# PolyGen: An Autoregressive Generative Model of 3D Meshes

![](media/example_samples.png)

This package provides an implementation of PolyGen as described in:

> **PolyGen: An Autoregressive Generative Model of 3D Meshes**, *Charlie Nash, Yaroslav Ganin, S. M. Ali Eslami, Peter W. Battaglia*, ICML, 2020. ([abs](https://arxiv.org/abs/2002.10880))

PolyGen is a generative model of 3D meshes that sequentially outputs mesh
vertices and faces. PolyGen consists of two parts: A vertex model, that
unconditionally models mesh vertices, and a face model, that models the mesh
faces conditioned on input vertices. The vertex model uses a masked Transformer
decoder to express a distribution over the vertex sequences. For the face model
we combine Transformers with [pointer networks](https://arxiv.org/abs/1506.03134)
to express a distribution over variable length vertex sequences.

In this repository we provide model code in `modules.py`, as well as data
processing utilities in `data_utils.py`. We also provide Colabs that demo
training PolyGen from scratch on a toy dataset, as well as sampling from a
pre-trained model.

There are some minor differences between this implementation and the paper:
* We add global information (e.g. class label embeddings) as an additional input in the first sequence position rather than project it at each layer. This reduces parameters, but does not significantly impact performance.
* We use [ReZero](https://arxiv.org/abs/2003.04887) which improves training speed.
* We train with only shifting augmentations, which we find to be as effective as the combination of augmentations described in the paper. This helps to simplify the data pre-processing pipeline.

## Training Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deepmind/deepmind-research/blob/master/polygen/training.ipynb)

To train a PolyGen model from scratch on a collection of simple meshes use this
colab. This demonstrates the data pre-processing required to create inputs for
the vertex and face models.

## Sampling pre-trained model Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deepmind/deepmind-research/blob/master/polygen/sample-pretrained.ipynb)

To sample a model pre-trained on [ShapeNet](https://www.shapenet.org/)
use this colab. The model is class-conditional, and is trained on longer
sequence lengths than those described in the paper. This colab uses the
following checkpoints: ([Google Cloud Storage
bucket](https://console.cloud.google.com/storage/browser/deepmind-research-polygen)).

### Pre-trained Model Files

The pre-trained models are available in Google Cloud Storage:

- **vertex_model.tar.gz**: Contains the vertex model checkpoint (~400MB)
- **face_model.tar.gz**: Contains the face model checkpoint (~300MB)

**Download Options:**

1. **Using the download script (recommended):**
   ```bash
   python download_polygen_models.py --output_dir ./models
   ```

2. **Manual download with gsutil:**
   ```bash
   mkdir -p /tmp/vertex_model /tmp/face_model
   gsutil cp gs://deepmind-research-polygen/vertex_model.tar.gz /tmp/vertex_model/
   gsutil cp gs://deepmind-research-polygen/face_model.tar.gz /tmp/face_model/
   tar xzf /tmp/vertex_model/vertex_model.tar.gz -C /tmp/vertex_model/
   tar xzf /tmp/face_model/face_model.tar.gz -C /tmp/face_model/
   ```

3. **Direct HTTP download:**
   ```bash
   # Vertex model
   wget https://storage.googleapis.com/deepmind-research-polygen/vertex_model.tar.gz
   
   # Face model  
   wget https://storage.googleapis.com/deepmind-research-polygen/face_model.tar.gz
   ```

**Note:** Each model contains TensorFlow checkpoint files (`.data`, `.index`, `.meta`, and `checkpoint` files). Make sure to extract the tar.gz files before using them in your code.

### Troubleshooting Model Downloads

**Issue #588: "Where is the face_model.tar and vertices_model.tar?"**

The correct file names are:
- `face_model.tar.gz` (not `.tar`)
- `vertex_model.tar.gz` (not `vertices_model.tar`)

If you're having trouble downloading:

1. **Check internet connection** and ability to access Google Cloud Storage
2. **Use the download script** for automatic handling: `python download_polygen_models.py`
3. **Verify gsutil installation** if using manual gsutil method
4. **Check available disk space** (models are ~700MB total)
5. **Try alternative download methods** listed above

**Common Errors:**
- `gsutil: command not found` → Install Google Cloud SDK or use the Python download script
- `Permission denied` → Check write permissions in target directory
- `File not found` → Ensure you're using the correct file names with `.tar.gz` extension

## Installation

To install the package locally run:
```bash
git clone https://github.com/deepmind/deepmind-research.git .
cd deepmind-research/polygen
pip install -e .
```

### Downloading Pre-trained Models

If you want to use the pre-trained models, install the download dependencies and run the download script:

```bash
# Install download dependencies
pip install -r requirements-download.txt

# Download models to default location (/tmp)
python download_polygen_models.py

# Or download to custom directory
python download_polygen_models.py --output_dir ./models

# Verify existing models
python download_polygen_models.py --verify_only --output_dir ./models
```

The script will download and extract both `vertex_model.tar.gz` and `face_model.tar.gz` files automatically.

## Giving Credit

If you use this code in your work, we ask you to cite this paper:

```
@article{nash2020polygen,
  author={Charlie Nash and Yaroslav Ganin and S. M. Ali Eslami and Peter W. Battaglia},
  title={PolyGen: An Autoregressive Generative Model of 3D Meshes},
  journal={ICML},
  year={2020}
}
```

## Disclaimer

This is not an official Google product.
