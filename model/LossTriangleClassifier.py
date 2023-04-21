"""
This file contains the LossTriangleClassifier class, which is a neural
network that can be used to classify loss triangles as either incremental
or cumulative.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Any

class LossTriangleClassifier(nn.Module):
    """
    This class is a neural network that can be used to classify loss triangles
    as either incremental or cumulative.

    Architecture:
    -------------
        - Convolutional layers
            - Convolutional layers are used to extract features from the loss
              triangles, by convolving a kernel over the input data. The kernel
              is a window that is used to convolve over the input data. The size
              of the kernel is specified by the `kernel_size` parameter, and the
              stride is specified by the `stride` parameter. The number of nodes
              in each convolutional layer is specified by the `base_conv_nodes`
              parameter, and the number of convolutional layers is specified by
              the `num_conv_layers` parameter.
        - Batch normalization layers
            - Batch normalization layers are used to normalize the output of the
              convolutional layers. This helps to prevent the "internal covariate
              shift" problem, where the distribution of the output of a layer
              changes during training, which can cause the network to train
              slower.
        - Pooling layers
            - Pooling layers are used to reduce the size of the output of the
              convolutional layers. This helps to reduce the number of parameters
              in the network, which helps to prevent overfitting.
        - Linear layers
            - Linear layers are used to combine the features extracted by the
              convolutional layers into a single vector. The number of nodes in
              each linear layer is specified by the `linear_nodes` parameter,
              and the number of linear layers is equal to the length of the
              `linear_nodes` parameter.
              These are also known as fully connected layers.
        - Dropout layers
            - Dropout layers are used to randomly drop out nodes during training.
              This helps to prevent overfitting by randomly dropping out nodes  
              so that the network cannot rely on any one node.

    Parameters:
    -----------
        input_shape: tuple
            The shape of the input data. This should be the shape of the
            loss triangle data, not including the batch size.
        num_classes: int
            The number of classes to classify the loss triangles as (since we are 
            only classifying as incremental or cumulative, this should be 2, and
            this is the default value).
        num_conv_layers: int
            The number of convolutional layers to use in the network. The number
            of pooling layers will be equal to the number of convolutional
            Default: 4
        base_conv_nodes: int
            The number of nodes in the first convolutional layer. The number of
            nodes in each subsequent convolutional layer will be double the
            number of nodes in the previous layer (i.e. the number of nodes in
            the second convolutional layer will be 2 * base_conv_nodes, the
            number of nodes in the third convolutional layer will be 4 *
            base_conv_nodes, etc.). This is a common practice in convolutional
            neural networks.
            Default: 64
        kernel_size: tuple
            The size of the kernel to use in the convolutional layers. The
            kernel is the window that is used to convolve over the input data.
            Larger kernels will be able to detect features over a larger area,
            but will also be more computationally expensive.
            
            `kernel_size` should be a tuple of two integers, where the first
            integer is the height of the kernel and the second integer is the
            width of the kernel. The default value is (2, 2), which is a 2x2
            kernel.
        stride: tuple
            The stride to use in the convolutional layers. The stride is the
            number of pixels to move the kernel over in each direction. A
            larger stride will be able to detect features over a larger area,
            but will also be more computationally expensive.
            
            This should be a tuple of two integers, where the first integer is
            the height of the stride and the second integer is the width of the
            stride. The default value is (1, 1), which is a 1x1 stride.
        padding: tuple
            The padding to use in the convolutional layers. The padding is the
            number of pixels to pad the input data with on each side. Padding
            the input data helps to prevent the loss of information at the
            edges of the input data.
        linear_nodes: list
            The number of nodes to use in each linear (fully connected) layer.
            The number of linear layers will be equal to the length of this
            list. 
            
            The first linear layer will have an input size equal to the
            number of nodes in the last convolutional layer, and the output
            size will be equal to the number of nodes in the first element of
            this list.
            
            The second linear layer will have an input size equal to the
            number of nodes in the first element of this list, and the output
            size will be equal to the number of nodes in the second element of
            this list.

            Continuing in this manner, the last linear layer will have an input
            size equal to the number of nodes in the second to last element of
            this list, and the output size will be equal to the number of
            classes.

            The default value is [512, 256], which will create two linear
            layers, with the first linear layer having 512 nodes and the second
            linear layer having 256 nodes.

            Note that the number of nodes in the last linear layer must be
            equal to the number of classes.
        linear_dropout: list
            The dropout to use in each linear layer. The number of dropout
            layers will be equal to the length of this list. The dropout is the
            probability that a node will be dropped out during training. This
            helps to prevent overfitting by randomly dropping out nodes during
            training.

            Default: [0.4, 0.2]
        relu_neg_slope: float
            This network uses leaky ReLU activation functions. Leaky ReLU is
            similar to ReLU, but instead of setting the output to 0 when the
            input is negative, it sets the output to a negative slope times
            the input. This helps to prevent the "dying ReLU" problem, where
            the output of a ReLU layer is always 0, which prevents the network
            from learning anything.

            The default value is 0.1, which is the default value used in the
            PyTorch implementation of leaky ReLU.

            Note that this parameter is only used if `activation` is not
            specified.
        activation: function
            The activation function to use in the network. This should be a
            function that takes a tensor as input and returns a tensor as
            output. The default value is None, which will use leaky ReLU
            activation functions with a negative slope of `relu_neg_slope`.
        output_activation: function
            The activation function to use in the output layer. This should be
            a function that takes a tensor as input and returns a tensor as
            output.
            
            The default value is a lambda function that applies softmax to the
            output of the network. This is used to convert the output of the
            network into a probability distribution over the classes.
    """
    def __init__(self,
                 input_shape : tuple
                 , num_classes : int = 2
                 , num_conv_layers : int = 4
                 , base_conv_nodes : int = 64
                 , kernel_size : tuple = (2, 2)
                 , stride : tuple = (1, 1)
                 , padding : tuple = (1, 1)
                 , linear_nodes : list = [512, 256]
                 , linear_dropout : list = [0.4, 0.2]
                 , relu_neg_slope : float = 0.1
                 , activation : Any = None
                 , output_activation : Any = lambda x: F.softmax(x, dim=1)
                 ):
        # Call the parent class's constructor
        super(LossTriangleClassifier, self).__init__()

        # set the activation function depending on the input parameters
        if activation is None:
            self.activation = lambda x: F.leaky_relu(x, relu_neg_slope)
        else:
            self.activation = activation

        # set the output activation function
        self.output_activation = output_activation

        # initialize the module lists that will hold the layers
        self.convolution_layers = nn.ModuleList()
        self.batch_normalization_layers = nn.ModuleList()
        self.pooling_layers = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        self.linear_dropout_layers = nn.ModuleList()

        # build the convolutional layers - loop through the number of layers
        # passed in as a parameter
        for i in range(num_conv_layers):

            # for each layer, nodes are doubled and the kernel size is halved
            node = base_conv_nodes * (2 ** i)

            # add the convolutional layer to the list of layers
            # the first layer has a different input shape than the rest, because
            # it takes in the input data
            self.convolution_layers.append(
                nn.Conv2d(1, node, kernel_size=kernel_size, stride=stride, padding=padding) if i == 0 else nn.Conv2d(base_conv_nodes * (2 ** (i - 1)), node, kernel_size=kernel_size, stride=stride, padding=padding)
            )

            # add the batch normalization layer to the list of layers
            self.batch_normalization_layers.append(nn.BatchNorm2d(node))
            
            # add the pooling layer to the list of layers
            self.pooling_layers.append(nn.MaxPool2d(kernel_size[0], kernel_size[1]))

        # once the convolutional layers are built, we need to flatten the
        # output of the last convolutional layer so that it can be used as
        # input to the linear layers
        self.flatten = nn.Flatten()

        # build the linear layers - loop through the number of layers passed
        for i, l in enumerate(linear_nodes):

            # the first layer takes in the flattened output of the last 
            # convolutional layer, and the last layer has a number of nodes
            # equal to the number of classes
            if i == 0:
                # first layer
                self.linear_layers.append(nn.Linear(self._get_flattened_size(input_shape), l))
            elif i == (len(linear_nodes) - 1):
                # last layer
                self.linear_layers.append(nn.Linear(linear_nodes[i - 1], num_classes))
            else:
                # middle layers
                self.linear_layers.append(nn.Linear(linear_nodes[i - 1], l))

            # after each linear layer, add a dropout layer
            self.linear_dropout_layers.append(nn.Dropout(linear_dropout[i]))

    def forward(self
                , x : torch.Tensor
                ) -> torch.Tensor:
        """
        Forward pass through the network.
        """
        # loop through the convolutional layers, batch normalization layers,
        # and pooling layers, and apply them to the input data
        for c, b, p in zip(self.convolution_layers
                           , self.batch_normalization_layers
                           , self.pooling_layers):
            # convolution
            x = c(x)

            # batch normalization
            x = b(x)

            # activation
            x = self.activation(x)
            
            # pooling
            x = p(x)

        # flatten the output of the last convolutional layer
        x = self.flatten(x)

        # loop through the linear layers and dropout layers, and apply them
        # to the input data
        for l, d in zip(self.linear_layers
                        , self.linear_dropout_layers):
            # linear layer
            x = l(x)

            # activation
            x = self.activation(x)

            # dropout
            x = d(x)

        # apply the output activation function to the output of the network
        return self.output_activation(x)

    def _get_flattened_size(self
                            , input_shape : tuple
                            ) -> int:
        """
        Helper function to get the size of the flattened output of the last
        convolutional layer.
        """
        # create a dummy tensor with the input shape
        dummy_output = torch.zeros(1, *input_shape)

        # loop through the convolutional layers, batch normalization layers,
        # and pooling layers, and apply them to the dummy tensor
        for c, b, p in zip(self.convolution_layers
                           , self.batch_normalization_layers
                           , self.pooling_layers):
            dummy_output = c(dummy_output)
            dummy_output = b(dummy_output)
            dummy_output = self.activation(dummy_output)
            dummy_output = p(dummy_output)

        # return the number of elements (`numel`) in the flattened output
        return dummy_output.numel()