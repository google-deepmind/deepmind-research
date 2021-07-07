# PyTorch evaluation

We provide PyTorch evaluation code for convenience. If you developed a version
of our training pipeline for PyTorch, please let us know as we will link it from
here.

Here are known PyTorch implementations of our training pipeline:

* https://github.com/imrahulr/adversarial_robustness_pytorch (by Rahul Rade)

Here are few consideration when reproducing our training pipeline in PyTorch.
As opposed to the [RST](https://github.com/yaircarmon/semisup-adv) code
(provided by Carmon et al.):

* We set the batch normalization decay to 0.99 (instead of 0.9).
* We do not apply weight decay (l2 regularization) to the batch normalization
  scale and offset
* We use Haiku's default initialization for all layers (except the last, which
  is initialized with zeros).
* The PGD attack used during training uniformly initializes the initial solution
  over the l-p norm ball.
* We run the attack over the local batch statistics (rather than the evaluation
  statistics).
* We update batch normalization statistics from adversarial examples only (
  rather than both clean and adversarial examples).
* We use 10 epochs warm-up to our learning schedule.


