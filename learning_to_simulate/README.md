# Learning to Simulate Complex Physics with Graph Networks

This is a model implementation for the ICML 2020 submission (also available in
arXiv [arxiv.org/abs/2002.09405](https://arxiv.org/abs/2002.09405). If you use
the code here please cite this paper:

    @article{sanchezgonzalez2020learning,
        title={Learning to Simulate Complex Physics with Graph Networks},
        author={Alvaro Sanchez-Gonzalez and
                Jonathan Godwin and
                Tobias Pfaff and
                Rex Ying and
                Jure Leskovec and
                Peter W. Battaglia},
        url={https://arxiv.org/abs/2002.09405},
        year={2020},
        eprint={2002.09405},
        archivePrefix={arXiv},
        primaryClass={cs.LG}
    }

## Contents

*   `model.py`: implementation of the graph network use as the learnable part of
    the model.
*   `model_demo.py`: example connecting the model to input dummy data.

## Running demo

(From one directory above)

    pip install -r learning_to_simulate/requirements.txt
    python -m learning_to_simulate.model_demo
