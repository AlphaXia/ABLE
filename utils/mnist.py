import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from collections import OrderedDict

from utils.utils_mlp import mlp_partialize
from .utils_algo import generate_instancedependent_candidate_labels


def load_mnist(args):
    
    test_transform = transforms.Compose([
        transforms.Grayscale(3), 
        transforms.ToTensor(),
        transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))
    ])
    
    original_train = dsets.MNIST(root=args.data_dir, train=True, download=True, transform=transforms.ToTensor())
    ori_data, ori_labels = original_train.data, original_train.targets.long()
    
    test_dataset = dsets.MNIST(root=args.data_dir, train=False, download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=args.batch_size*4, \
        shuffle=False, num_workers=args.workers, pin_memory=False
    )
       
    num_features = 28 * 28
    ori_data = ori_data.view((ori_data.shape[0], -1)).float()
    partialize_net = mlp_partialize(n_inputs=num_features, n_outputs=args.num_class)
    partialize_net.load_state_dict(torch.load('./pmodel/mnist.pt'))
    
    partialY_matrix = generate_instancedependent_candidate_labels(partialize_net, ori_data, ori_labels)
    
    ori_data = original_train.data
    temp = torch.zeros(partialY_matrix.shape)
    temp[torch.arange(partialY_matrix.shape[0]), ori_labels] = 1
    
    if torch.sum(partialY_matrix * temp) == partialY_matrix.shape[0]:
        print('data loading done !')

    partial_training_dataset = MNIST_Partialize(ori_data, partialY_matrix.float(), ori_labels.float())
    
    partial_training_dataloader = torch.utils.data.DataLoader(
        dataset=partial_training_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True
    )
    
    return partial_training_dataloader, partialY_matrix, test_loader


class MNIST_Partialize(Dataset):
    def __init__(self, images, given_partial_label_matrix, true_labels):
        
        self.ori_images = images
        self.given_partial_label_matrix = given_partial_label_matrix
        self.true_labels = true_labels
        
        self.weak_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=28, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))
        ])
        
        self.strong_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=28, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))
        ])

    def __len__(self):
        return len(self.true_labels)
        
    def __getitem__(self, index):
        
        each_image_w = self.weak_transform(self.ori_images[index])
        each_image_s = self.strong_transform(self.ori_images[index])
        each_label = self.given_partial_label_matrix[index]
        each_true_label = self.true_labels[index]
        
        return each_image_w, each_image_s, each_label, each_true_label, index

