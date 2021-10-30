# Generic Diff Pruning Implementation for Transformers from Huggingface

Implmentation of [Parameter-Efficient Transfer Learning with Diff Pruning](https://arxiv.org/abs/2012.07463) using [torchreparam](https://github.com/SsnL/PyTorch-Reparam-Module) to allow autograd to handle the entire process. The original implementation from the paper can be found [here](), but uses manual gradient copying for backprop. This implementation will hopefully make the technique easier to use. This version currently doesn't implement the structured L0 norm that gets the best results in their paper, but I'll be working on adding it!

# Citing Diff Pruning
```
@misc{guo2020parameterefficient,
      title={Parameter-Efficient Transfer Learning with Diff Pruning}, 
      author={Demi Guo and Alexander M. Rush and Yoon Kim},
      year={2020},
      eprint={2012.07463},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
If you use this particular implementation, I'd appreciate if you mentioned it in a footnote :)
