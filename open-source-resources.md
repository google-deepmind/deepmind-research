# DeepMind Open Source Resources

This document provides a comprehensive index of all datasets, code repositories, and resources associated with DeepMind Research projects. This centralizes access to improve reproducibility and research collaboration.

## 📊 Datasets

### Computer Vision & Image Processing
- **[Perceiver IO](perceiver/)** - General architecture for structured inputs & outputs
  - Paper: [Perceiver IO: A General Architecture for Structured Inputs & Outputs](https://arxiv.org/abs/2107.14795)
  - Code: `perceiver/`
  - Datasets: Multi-modal examples included

### Reinforcement Learning
- **[RL Unplugged](rl_unplugged/)** - Benchmarks for offline reinforcement learning
  - Paper: [RL Unplugged: A Suite of Benchmarks for Offline Reinforcement Learning](https://arxiv.org/abs/2006.13888)
  - Code: `rl_unplugged/`
  - Datasets: Multiple domains (Atari, Control Suite, etc.)

### Physics & Simulation
- **[MeshGraphNets](meshgraphnets/)** - Learning mesh-based simulation with graph networks
  - Paper: [Learning Mesh-Based Simulation with Graph Networks](https://arxiv.org/abs/2010.03409)
  - Code: `meshgraphnets/`
  - Datasets: CFD (cylinder_flow), cloth simulation (flag_simple)
  - Download: `bash meshgraphnets/download_dataset.sh <dataset_name> <output_dir>`

- **[Learning to Simulate](learning_to_simulate/)** - Graph network-based physics simulators
  - Paper: [Learning to Simulate Complex Physics with Graph Networks](https://arxiv.org/abs/2002.09405)
  - Code: `learning_to_simulate/`
  - Datasets: Various physics simulations (Water, Sand, etc.)

### Natural Language Processing
- **[WikiGraphs](wikigraphs/)** - Wikipedia-Knowledge Graph paired dataset
  - Paper: [WikiGraphs: A Wikipedia Text - Knowledge Graph Paired Dataset](https://aclanthology.org/2021.textgraphs-1.7/)
  - Code: `wikigraphs/`
  - Datasets: Wikipedia articles with knowledge graphs

### Self-Supervised Learning
- **[BYOL](byol/)** - Bootstrap Your Own Latent
  - Paper: [Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning](https://arxiv.org/abs/2006.07733)
  - Code: `byol/`
  - Datasets: ImageNet pre-training pipeline

### Graph Neural Networks
- **[Graph Matching Networks](graph_matching_networks/)** - Learning similarity of graph structured objects
  - Paper: [Graph Matching Networks for Learning the Similarity of Graph Structured Objects](https://arxiv.org/abs/1904.12787)
  - Code: `graph_matching_networks/`
  - Datasets: Various graph matching tasks

## 🧬 Computational Biology
- **[AlphaFold CASP13](alphafold_casp13/)** - Protein structure prediction
  - Paper: [Improved protein structure prediction using potentials from deep learning](https://www.nature.com/articles/s41586-019-1923-7)
  - Code: `alphafold_casp13/`
  - Datasets: CASP13 protein structures

## 🎮 Game AI & Multi-Agent
- **[OpenSpiel Integration](https://github.com/deepmind/open_spiel)** - Collection of environments for research in general reinforcement learning
  - External Repository: [github.com/deepmind/open_spiel](https://github.com/deepmind/open_spiel)

## 🌡️ Climate & Weather
- **[Nowcasting](nowcasting/)** - Precipitation nowcasting using deep generative models
  - Paper: [Skilful precipitation nowcasting using deep generative models of radar](https://www.nature.com/articles/s41586-021-03854-z)
  - Code: `nowcasting/`
  - Datasets: UK precipitation radar data

## 📚 How to Use This Index

### For Researchers:
1. **Find your domain** of interest above
2. **Follow the paper link** to understand the methodology
3. **Navigate to the code directory** for implementation details
4. **Download datasets** using provided scripts or links

### For Dataset Access:
Most datasets are downloaded programmatically:
```bash
# Example: MeshGraphNets CFD data
bash meshgraphnets/download_dataset.sh cylinder_flow ./data/

# Example: RL Unplugged datasets
python -m rl_unplugged.load_dataset --task=cartpole_swingup
```

### For Code Usage:
Each project directory contains:
- `README.md` - Project-specific documentation
- `requirements.txt` - Python dependencies
- `run.sh` or similar - Execution scripts
- Example notebooks where applicable

## 🔗 External Resources

### Official DeepMind Repositories:
- **[DeepMind Lab](https://github.com/deepmind/lab)** - 3D learning environments
- **[OpenSpiel](https://github.com/deepmind/open_spiel)** - Multi-agent reinforcement learning
- **[dm-haiku](https://github.com/deepmind/dm-haiku)** - JAX neural network library
- **[Acme](https://github.com/deepmind/acme)** - Reinforcement learning research framework

### DeepMind Official Links:
- **[Research Publications](https://deepmind.com/research/publications/)**
- **[Open Source Datasets](https://deepmind.com/research/open-source/open-source-datasets/)**
- **[Open Source Code](https://deepmind.com/research/open-source/open-source-code/)**
- **[Open Source Environments](https://deepmind.com/research/open-source/open-source-environments/)**

## 🐛 Reporting Issues

If you find broken links or missing resources:
1. Check if the resource has moved to a dedicated repository
2. Open an issue in this repository with the "documentation" label
3. Include the specific resource and expected location

## 📄 License

All code and datasets are subject to their individual licenses as specified in each project directory. Most DeepMind research code is released under the Apache 2.0 License.

---

*Last updated: August 2025*
*This document is maintained by the community. Contributions welcome!*
