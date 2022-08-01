# Ambiguity-Induced Contrastive Learning for Instance-Dependent Partial Label Learning

Code for "Ambiguity-Induced Contrastive Learning for Instance-Dependent Partial Label Learning" in IJCAI-ECAI 2022.


## Getting started

- Create directory `./data` (if `./data` does not exist)
- Create directory `./pmodel` (if `./pmodel` does not exist)
- Change directory to `./pmodel`
- Download [pmodel](https://drive.google.com/drive/folders/1MdlcuBaX2UL-dV0RL41tHY-b_CwWHAM9?usp=sharing)


## Running

```
python -u main.py --dataset cifar10 --data-dir './data' --workers 0 --num-class 10 --pmodel_path './pmodel/cifar10.pt' --arch resnet18 --temperature 0.1 --loss_weight 1.0 --cuda_VISIBLE_DEVICES '0' --epochs 500 --batch-size 64 --lr 0.01 --wd 1e-3 --cosine --seed 123

python -u main.py --dataset mnist --data-dir './data' --workers 0 --num-class 10 --pmodel_path './pmodel/mnist.pt' --arch resnet18 --temperature 0.1 --loss_weight 1.0 --cuda_VISIBLE_DEVICES '0' --epochs 500 --batch-size 64 --lr 0.01 --wd 1e-3 --cosine --seed 123

python -u main.py --dataset fmnist --data-dir './data' --workers 0 --num-class 10 --pmodel_path './pmodel/fmnist.pt' --arch resnet18 --temperature 0.1 --loss_weight 1.0 --cuda_VISIBLE_DEVICES '0' --epochs 500 --batch-size 64 --lr 0.01 --wd 1e-3 --cosine --seed 123

python -u main.py --dataset kmnist --data-dir './data' --workers 0 --num-class 10 --pmodel_path './pmodel/kmnist.pt' --arch resnet18 --temperature 0.1 --loss_weight 1.0 --cuda_VISIBLE_DEVICES '0' --epochs 500 --batch-size 64 --lr 0.01 --wd 1e-3 --cosine --seed 123
```


## Acknowledgment

J. Lv, M. Xu, L. Feng, G. Niu, X. Geng, and M. Sugiyama. Progressive identification of true labels for partial-label learning. In International Conference on Machine Learning, pages 6500–6510, Virtual Event, July 2020. ACM.

Ning Xu, Congyu Qiao, Xin Geng, and Min-Ling Zhang. Instance-dependent partial label learning. In Advances in Neural Information Processing Systems, Virtual Event, December 2021. MIT Press.

Haobo Wang, Ruixuan Xiao, Yixuan Li, Lei Feng, Gang Niu, Gang Chen, and Junbo Zhao. PiCO: Contrastive label disambiguation for partial label learning. In International Conference on Learning Representations, 2022.
