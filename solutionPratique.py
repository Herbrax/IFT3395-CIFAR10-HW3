import random
import numpy as np
import torch
from typing import Tuple, List, NamedTuple
from tqdm import tqdm
import torchvision
from torchvision import transforms
from collections import OrderedDict

# from torchvision import models
# from torchsummary import summary


# Seed all random number generators
np.random.seed(197331)
torch.manual_seed(197331)
random.seed(197331)


class NetworkConfiguration(NamedTuple):
    n_channels: Tuple[int, ...] = (16, 32, 48)
    kernel_sizes: Tuple[int, ...] = (3, 3, 3)
    strides: Tuple[int, ...] = (1, 1, 1)
    dense_hiddens: Tuple[int, ...] = (256, 256)


class Trainer:
    def __init__(self,
                 network_type: str = "mlp",
                 net_config: NetworkConfiguration = NetworkConfiguration(),
                 lr: float = 0.001,
                 batch_size: int = 128,
                 activation_name: str = "relu"):

        self.lr = lr
        self.batch_size = batch_size
        self.train, self.test = self.load_dataset(self)
        dataiter = iter(self.train)
        images, labels = next(dataiter)
        input_dim = images.shape[1:]
        self.network_type = network_type
        activation_function = self.create_activation_function(activation_name)
        if network_type == "mlp":
            self.network = self.create_mlp(input_dim[0]*input_dim[1]*input_dim[2], 
                                           net_config,
                                           activation_function)
        elif network_type == "cnn":
            self.network = self.create_cnn(input_dim[0], 
                                           net_config, 
                                           activation_function)
        else:
            raise ValueError("Network type not supported")
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)

        self.train_logs = {'train_loss': [], 'test_loss': [],
                           'train_mae': [], 'test_mae': []}

    @staticmethod
    def load_dataset(self):
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size,
                                                shuffle=True)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size,
                                                shuffle=False)

        return trainloader, testloader

    @staticmethod
    def create_mlp(input_dim: int, net_config: NetworkConfiguration,
                   activation: torch.nn.Module) -> torch.nn.Module:
        """
        Create a multi-layer perceptron (MLP) network.

        :param net_config: a NetworkConfiguration named tuple. Only the field 'dense_hiddens' will be used.
        :param activation: The activation function to use.
        :return: A PyTorch model implementing the MLP.
        """
        dense_hiddens = net_config.dense_hiddens
        

        List_Operations = [('flatten1',torch.nn.Flatten()),
                           ('Dense1', torch.nn.Linear(input_dim,dense_hiddens[0])),
                           ('Activation1', activation)]
        
        for i in range(len(dense_hiddens)-1):
            List_Operations += [('Dense' + str(i+2), torch.nn.Linear(dense_hiddens[i],dense_hiddens[i+1])),
                                ('Activation' + str(i+2), activation)]

        List_Operations += [('Densefinal', torch.nn.Linear(dense_hiddens[-1], 1))]
        
        Dict_Operations = OrderedDict(List_Operations)
        model = torch.nn.Sequential(Dict_Operations)
        return model

    @staticmethod
    def create_cnn(in_channels: int, net_config: NetworkConfiguration,
                   activation: torch.nn.Module) -> torch.nn.Module:
        """
        Create a convolutional network.

        :param in_channels: The number of channels in the input image.
        :param net_config: a NetworkConfiguration specifying the architecture of the CNN.
        :param activation: The activation function to use.
        :return: A PyTorch model implementing the CNN.
        """
        n_channels = net_config.n_channels
        kernel_sizes = net_config.kernel_sizes
        strides = net_config.strides
        dense_hiddens = net_config.dense_hiddens
        
        nb_convol = len(n_channels)
        nb_dense = len(dense_hiddens)
         
        if nb_convol != 0:
            #Première convolution
            List_Operations = [('convol1',torch.nn.Conv2d(in_channels, n_channels[0],kernel_size = kernel_sizes[0], stride=strides[0]))]
            
            #Activation -> MaxPool -> Convol, pour les autres couches
            for i in range(nb_convol-1):
                List_Operations += [ ('ActivationConv' + str(i+1), activation),
                                        ('MaxPool' + str(i+1),torch.nn.MaxPool2d(kernel_size=2)),
                                        ('convol'+ str(i+2),torch.nn.Conv2d(n_channels[i], n_channels[i+1],kernel_size = kernel_sizes[i+1], stride=strides[i+1]))]
            
            # Dernier Max
            List_Operations += [('activationConvFinal', activation)]
            
        List_Operations += [('MaxPoolConvFinal', torch.nn.AdaptiveMaxPool2d((4, 4))),
                                ('flatten1', torch.nn.Flatten())]
        
        # Couches complétement connectés
        if nb_dense != 0:
            if nb_convol != 0 :
                List_Operations+= [('Dense1',torch.nn.Linear( 16 * n_channels[-1], dense_hiddens[0])),
                                         ('ActivationDense1', activation)]
            else: 
                List_Operations+=[('Dense1',torch.nn.Linear( 16 * in_channels, dense_hiddens[0])),
                                        ('ActivationDense1', activation)]
    
            for i in range(nb_dense-1):
                List_Operations+=[('Dense' + str(i+2),torch.nn.Linear( dense_hiddens[i], dense_hiddens[i+1]) ),
                                        ('ActivationDense'+ str(i+2), activation)]
                                       
            List_Operations+=[('LastDense', torch.nn.Linear(dense_hiddens[-1], 1))]
            
        else : 
            List_Operations+=[('LastDense', torch.nn.Linear(16*in_channels, 1))]
                
        print(List_Operations)
        Dict_Operations = OrderedDict(List_Operations)
        model = torch.nn.Sequential(Dict_Operations)
        return model

    @staticmethod
    def create_activation_function(activation_str: str) -> torch.nn.Module:
        if activation_str == "relu": return torch.nn.ReLU()
        elif activation_str == "tanh": return torch.nn.Tanh()
        elif activation_str == "sigmoid": return torch.nn.Sigmoid()
        else: 
            print('Chosen function is not relu, tanh or sigmoid')
            return None

    def compute_loss_and_mae(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        predictions= self.network(X)
        lossL2 = torch.nn.MSELoss()
        output_lossL2 = lossL2(predictions, y.float())
        
        lossL1= torch.nn.L1Loss()
        output_lossL1 = lossL1(predictions, y.float())
        
        #print(output_lossL2.dtype, output_lossL1.dtype,predictions.dtype, y.dtype)
        return (output_lossL2 , output_lossL1 )

    def training_step(self, X_batch: torch.Tensor, y_batch: torch.Tensor):
        self.optimizer.zero_grad()
        loss, mae = self.compute_loss_and_mae(X_batch,y_batch)
        loss.backward()
        self.optimizer.step()
        return (loss.item(), mae.item())

    def train_loop(self, n_epochs: int) -> dict:
        N = len(self.train)
        for epoch in tqdm(range(n_epochs)):
            train_loss = 0.0
            train_mae = 0.0
            for i, data in enumerate(self.train):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                loss, mae = self.training_step(inputs, labels)
                train_loss += loss
                train_mae += mae

            # Log data every epoch
            self.train_logs['train_mae'].append(train_mae / N)
            self.train_logs['train_loss'].append(train_loss / N)
            self.evaluation_loop()
    
        return self.train_logs

    def evaluation_loop(self) -> None:
        self.network.eval()
        N = len(self.test)
        with torch.inference_mode():
            test_loss = 0.0
            test_mae = 0.0
            for data in self.test:
                inputs, labels = data
                loss, mae = self.compute_loss_and_mae(inputs, labels)
                test_loss += loss.item()
                test_mae += mae.item()

        self.train_logs['test_mae'].append(test_mae / N)
        self.train_logs['test_loss'].append(test_loss / N)


    def evaluate(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        loss, mae = self.compute_loss_and_mae(X,y)
        return (loss,mae)


# test = Trainer(network_type= 'cnn')
# test.train_loop(2)