# Path-Specific Counterfactual Fairness (AAAI 2019)

Paper: [Path-Specific Counterfactual Fairness](https://ojs.aaai.org/index.php/AAAI/article/view/4777/4655)

If you use the code here please cite this paper:

    @inproceedings{chiappa2019path,
      title={Path-specific counterfactual fairness},
      author={Chiappa, Silvia},
      booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
      volume={33},
      number={01},
      pages={7801--7808},
      year={2019}
    }

### Overview

This release contains the path-specific counterfactual fairness method used
in the paper, as well as utility functions for loading the *Adult* dataset.

The following gives a brief overview of the contents, more detailed
documentation is available within each file:

*   __causal_network.py__: Defines the `Node` class, instances of which can be
    combined into a directed graph. Associated with each node is a
    `distribution_module`, a haiku module which builds a tensorflow
    `Distribution` instance as a function of the node's parents.
*   __util.py__: Miscellaneous utility functions.
*   __variational.py__: Class for performing variational
    inference. The `Variational` haiku module is a general-purpose approximate
        posterior, using an MLP to map from arbitrary inputs to the parameters
        of a Gaussian distribution.
*   __adult.py__: Utility functions for the *Adult* dataset.
*   __adult_pscf.py__: Training and 'fair' prediction process (using
    path-specific counterfactual fairness) on the *Adult* dataset.
*   __adult_pscf_config.py__: Configuration file with default training
        parameters.

### Dataset

The *Adult* dataset can be downloaded from
[https://archive.ics.uci.edu/ml/datasets/adult](https://archive.ics.uci.edu/ml/datasets/adult). You may use the following
command to download both necessary files to run the training script:

`sh download_dataset.sh ${OUTPUT_DIR}`

### Experiments

To download the dataset and run the main experiment reported in the paper,
you may run:

`sh run_adult_pscf.sh`

### Acknowledgements

Credits to Thomas P. S. Gillam for the original TF1 implementation.
