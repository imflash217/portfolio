# Popular Deep Learning Architectures using EINOPS

In this section we will be rewriting the building blocks of deep learning
in both the traditional `PyTorch` way as well as using `einops` library.

## Imports

Firstly, we will import the necessary libraries to be used.

```python hl_lines="10-11"
## importing necessary libraries

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, reduce, repeat, asnumpy, parse_shape
from einops.layers.torch import Rearrange, Reduce
```

## Simple ConvNet

???+ danger "Using only PyTorch"
    Here is an implementation of a simple **ConvNet** using only **`PyTorch`**
    without `einops`.

    ```python hl_lines="29"
    class ConvNet(nn.Module):
        def __init__(self):
            super(ConvNet, self).__init__()
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.conv2_drop = nn.Dropout2d()
            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(20, 10)
            
        def forward(self, x):
            x = self.conv1(x)
            x = F.max_pool2d(x, 2)
            x = F.relu(x)
            
            x = self.conv2(x)
            x = self.conv2_drop(x)
            x = F.max_pool2d(x, 2)
            x = F.relu(x)

            x = x.view(-1, 320)
            x = self.fc1(x)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)
    
    ## Instantiating the ConvNet class
       
    conv_net_old = ConvNet()
    ```

???+ done "Using EINOPS + PyTorch"
    Implementing the same above ConvNet using **`einops`** & `PyTorch`
    
    ```python hl_lines="9"
    conv_net_new = nn.Sequential(
        nn.Conv2d(1, 10, kernel_size=5),
        nn.MaxPool2d(kernel_size=2),
        nn.ReLU(),
        nn.Conv2d(10, 20, kernel_size=5),
        nn.MaxPool2d(kernel_size=2),
        nn.ReLU(),
        nn.Dropout2d(),
        Rearrange("b c h w -> b (c h w)"),
        nn.Linear(320, 50),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(50, 10),
        nn.LogSoftmax(dim=1)
    )
    ```

???+ quote "Why prefer EINOPS implementation?"
    Following are the reasons to prefer the new implementation:

    - [x] In the original code, if the input is changed and the **`batch_size`** 
            is divisible by 16 (which usually is), we will get something senseless after reshaping.
        - [ ] :rotating_light: The new code using **`einops`** explicitly raise ERROR in the above scenario. Hence better!!
    - [x] We won't forget to use the flag **`self.training`** with the new implementation.
    - [x] Code is straightforward to read and analyze.
    - [x] **`nn.Sequential`** makes printing/saving/passing trivial. 
            And there is no need in your code to **load** the model (which also has lots of benefits).
    - [x] Don't need **`logsoftmax`**? Now, you can use **`conv_net_new[-1]`**. 
            Another reason to prefer **`nn.Sequential`**
    - [x] ... And we culd also add **inplace `ReLU`**


## Super-resolution

???+ danger "Only PyTorch"
    ```python hl_lines="9-10 16"
    class SuperResolutionNetOLD(nn.Module):
        def __init__(self, upscale_factor):
            super(SuperResolutionNetOLD, self).__init__()
            
            self.relu = nn.ReLU()
            self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
            self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
            self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
            self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
            self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        
        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.relu(self.conv3(x))
            x = self.pixel_shuffle(self.conv4(x))
            return x
    ```

???+ done "Using EINOPS"
    ```python hl_lines="10"
    def SuperResolutionNetNEW(upscale_factor):
        return nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, upscale_factor ** 2, kernel_size=3, padding=1),
            Rearrange("b (h2 w2) h w -> b (h h2) (w w2)", h2=upscale_factor, w2=upscale_factor)
        )
    ```

???+ quote "Improvements over the old implementation"
    - [x] No need in special instruction **`pixel_shuffle`** (& the result is transferrable b/w the frameworks)
    - [x] Output does not contain a fake axis (& we could do the same for the input)
    - [x] inplace **`ReLU`** used now. For high resolution images this becomes critical
            and saves a lot of memory.
    - [x] and all the benefits of **`nn.Sequential`**

## Gram Matrix / Style Transfer

Restyling Graam Matrix for style transfer.

???+ danger "Original Code using ONLY PyTorch"
    The original code is already very good. First line shows what kind of input is expected.
    
    ```python
    def gram_matrix_old(y):
        (b, c, h, w) = y.size()
        features = y.view(b, c, h * w)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram
    ```

???+ done "Using EINSUM"
    ```python hl_lines="3"
    def gram_matrix_new(y):
        b, c, h, w = y.shape
        return torch.einsum("bchw, bdhw -> bcd", [y, y]) / (h * w)
    ```

???+ quote "Improvements"
    **`einsum`** operations should be read like:

    - [x] For each batch & each pair of channels we sum over **`h`** and **`w`**.
    - [x] The normalization is also changed, because that's how **Gram Matrix** is defined.
            Else we should call it **Normalized Gram Matrix** or alike.


## Recurrent Models (RNNs)

???+ danger "ONLY PyTorch"
    ```python hl_lines="14-15"
    class RNNModelOLD(nn.Module):
        """Container module with an ENCODER, a RECURRENT module & a DECODER module"""
        def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5):
            super(RNNModelOLD, self).__init__()
            self.drop = nn.Dropout(dropout)
            self.encoder = nn.Embedding(ntoken, ninp)
            self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
            self.decoder = nn.Linear(nhid, ntoken)
        
        def forward(self, input, hidden):
            emb = self.drop(self.encoder(input))
            output, hidden = self.rnn(emb, hidden)
            output = self.drop(output)
            decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
            return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    ```


???+ done "Using EINOPS"
    ```python hl_lines="11 14-15"
    def RNNModelNEW(nn.Module):
        """Container module with an ENCODER, RNN & a DECODER modules."""
        def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5):
            super(RNNModelNEW, self).__init__()
            self.drop = nn.Dropout(dropout)
            self.encoder = nn.Embedding(ntoken, ninp)
            self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
            self.decoder = nn.Linear(nhid, ntoken)
        
        def forward(self, input, hidden):
            t, b = input.shape[:2]
            emb = self.drop(self.encoder(input))
            output, hidden = self.rnn(emb, hidden)
            output = rearrange(self.drop(output), "t b nhid -> (t b) nhid")
            decoded = rearrange(self.decoder(output), "(t b) token -> t b token", t=t, b=b)
            return decoded, hidden
    ```


## Improving RNN

???+ danger "Only PyTorch"
    ```python
    class RNNold(nn.Module):
        def __init__(
            self,
            vocab_size,
            embedding_dim,
            hidden_dim,
            output_dim,
            n_layers,
            bidirectional,
            dropout
        ):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.rnn = nn.LSTM(embedding_dim,
                                hidden_dim,
                                num_layers=n_layers,
                                bidirectional=bidirectional,
                                dropout=dropout)
            self.fc = nn.Linear(hidden_dim * 2, output_dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            ## x = [sent_len, batch_size]
            embedded = self.dropout(self.embedding(x))      ## size = [sent_len, batch_size, emb_dim]
            output, (hidden, cell) = self.rnn(embedded)
            
            ## output.shape = [sent_len, batch_size, hid_dim * num_directions]
            ## hidden.shape = [num_layers * num_directions, batch_size, hid_dim]
            ## cell.shape = [num_layers * num_directions, batch_size, hid_dim]
            
            ## concat the final dropout (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
            ## and apply dropout
            ## hidden.size = [batch_size, hid_dim * num_directions]
            hidden = self.dropout(torch.cat([hidden[-2,:,:], hidden[-1,:,:]], dim=1))

            return self.fc(hidden.squeeze(0))
    ```

???+ done "Using EINOPS"
    ```python
    class RNNnew(nn.Module):
        def __init__(
            self,
            vocab_size,
            embedding_dim,
            hidden_dim,
            output_dim,
            n_layers,
            bidirectional,
            dropout
        ):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.rnn = nn.LSTM(embedding_dim,
                                hidden_dim,
                                num_layers=n_layers,
                                bidirectional=bidirectional,
                                dropout=dropout)
            self.dropout = nn.Dropout(dropout)
            self.directions = 2 if bidirectional else 1
            self.fc = nn.Linear(hidden_dim * self.directions, output_dim)
        
        def forward(self, x):
            embedded = self.dropout(self.embedding(x))
            output, (hidden, cell) = self.rnn(embedded)
            hidden = rearrange(hidden, "(layer dir) b c -> layer b (dir c)", dir=self.directions)
            
            ## take the fina layer's hidden
            return self.fc(self.dropout(hidden[-1]))
    ```


## Channel Shuffle (from ShuffleNet)

???+ danger "ONLY PyTorch"
    ```python
    def channel_shuffle_old(x, groups):
        b, c, h, w = x.data.size()
        channels_per_group = c // groups
        
        ## reshape
        x = x.view(b, groups, channels_per_group, h, w)
        
        ## transpose
        ## - contiguous() is required if transpose() is used before view()
        ##   See https://github.com/pytorch/pytorch/issues/764
        x = x.transpose(1, 2).contiguous()
        
        ## flatten
        x = x.view(b, -1, h, w)
        return x
    ```

???+ done "Using EINOPS"
    ```python
    def channel_shuffle_new(x, groups):
        return rearrange(x, "b (c1 c2) h w -> b (c2 c1) h w", c1=groups)
    ```

## ShuffleNet

???+ danger "ONLY PyTorch"
    ```python
    from collections import OrderedDict
    
    def channel_shuffle(x, groups):
        b, c, h, w = x.data.size()
        channels_per_group = c // groups
        
        ## reshape
        x = x.view(b, groups, channels_per_group, h, w)

        ## transpose
        ## - contiguous() is required if transpose() is used before view()
        x = x.transpose(1, 2).contiguous()
        
        x = x.view(b, -1, h, w)
        return x

    class ShuffleUnitOLD(nn.Module):
        def __init__(self, 
                     in_channels, 
                     out_channels,
                     groups=3,
                     grouped_conv=True,
                     combine="add",
        ):
            super(ShuffleUnitOLD, self).__init__()
            
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.grouped_conv = grouped_conv
            self.combine = combine
            self.groups = groups
            self.bottleneck_channels = self.out_channels // 4
            
            ## define the type of ShuffleUnit
            if self.combine == "add":
                ## shuffleUnit fig-2b
                self.depthwise_stride = 1
                self._combine_func = self._add
            elif self.combine == "concat":
                ## ShuffleUnit fig-2c
                self.depthwise_stride = 2
                self._combine_func = self._concat
                
                ## ensure output of the concat has the same channels
                ## as the original input channels
                self.out_channels -= self.in_channels
            else:
                raise ValueError(f"Cannot combine tensors with {self.combine}.\n"
                                 f"Only 'add' & 'concat' supported.")
            
            ## Use a 1x1 grouped or non-grouped convolution to reduce input channels
            ## to bottleneck channels, as in ResNet bottleneck module.
            ## NOTE: do not use group convolution for the first conv1x1 in stage-2
            self.first_1x1_groups = self.groups if grouped_conv else 1
            
            self.g_conv_1x1_compress = self._make_grouped_conv1x1(
                self.in_channels,
                self.bottleneck_channels,
                self.first_1x1_groups,
                batch_norm=True,
                relu=True,
            )
            
            ## 3x3 depthwise convolution followed by batch normalization
            self.depthwise_conv3x3 = conv3x3(
                self.bottleneck_channels,
                self.bottleneck_channels,
                stride=self.depthwise_stride,
                groups=self.bottleneck_channels
            )
            self.bn_after_depthwise = nn.BatchNordm2d(self.bottleneck_channels)

            ## use 1x1 grouped convolution to expand from bottleneck_channels to out_channels
            self.g_conv_conv_1x1_expand = self._make_grouped_conv1x1(
                self.bottleneck_channels,
                self.out_channels,
                self.groups,
                batch_norm=True,
                relu=False
            )

        
        @staticmethod
        def _add(x, out):
            ## residual connection
            return x + out

        @staticmethod
        def _concat(x, out):
            ## concat along channel dim
            return torch.cat((x, out), 1)

        def _make_grouped_conv1x1(
            self,
            in_channels,
            out_channels,
            groups,
            batch_norm=True,
            relu=False
        ):
            modules = OrderedDict()
            conv = conv1x1(in_channels, out_channels, groups=groups)
            modules['conv1x1'] = conv

            if batch_norm:
                modules['batch_norm'] = nn.BatchNorm2d(out_channels)
            if relu:
                modules['relu'] = nn.ReLU()
            if len(modules) > 1:
                return nn.Sequential(modules)
            else:
                return conv

        def forward(self, x):
            ## save for combining later with output
            residual = x
            if self.combine == "concat":
                residual = F.avg_pool2d(residual, kernel_size=3, stride=2, padding=1)
            
            out = self.g_con_1x1_compress(x)
            out = channel_shuffle(out, self.groups)
            out = self.depthwise_conv3x3(out)
            out = self.bn_after_depthwise(out)
            out = self.g_conv_1x1_expand(out)
            
            out = self._combine_func(residual, out)
            return F.relu(out)
            
    ```

???+ done "Using EINOPS"
    ```python
    class ShuffleUnitNEW(nn.Module):
        def __init__(
            self,
            in_channels,
            out_channels,
            groups=3,
            grouped_conv=True,
            combine="add"
        ):
            super().__init__()
            first_1x1_groups = groups if grouped_conv else 1
            bottleneck_channels = out_channels // 4
            self.combine = combine
            if combine == "add":
                ## ShuffleUnit fig-2b
                self.left = Rearrange("...->...")   ## identity
                depthwise_stride = 1
            else:
                ## ShuffleUnit fig-2c
                self.left = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
                depthwise_stride = 2
                ## ensure output of concat has the same channels as the original output channels
                out_channels -= in_channels
                assert out_channels > 0
            
            self.right = nn.Sequential(
                ## use a 1x1grouped or non-grouped convolution to reduce
                ## input channels to bottleneck channels as in ResNet bottleneck module.
                conv1x1(in_channels, bottleneck_channels, groups=first_1x1_groups),
                nn.BatchNorm2d(bottleneck_channels),
                nn.ReLU(inplace=True),
                ## channel shuffle
                Rearrange("b (c1 c2) h w -> b (c2 c1) h w", c1=groups),
                ## 3x3 depthwise convolution followed by BatchNorm
                conv3x3(bottleneck_channels, 
                        bottleneck_channels, 
                        stride=depthwise_stride,
                        groups=bottleneck_channels),
                nn.BatchNorm2d(bottleneck_channels),
                ## Use 1x1 grouped convolution to expand from bottleneck_channels to output_channels
                conv1x1(bottleneck_channels, out_channels, groups=groups),
                nn.BatchNorm2d(out_channels),
            )

        def forward(self, x):
            if self.combine == "add":
                combined = self.left(x) + self.right(x)
            else:
                combined = torch.cat([self.left(x), self.right(x)], dim=1)
            return F.relu(combined, inplace=True)
            
    ```

???+ quote "Improvements"
    Rewriting the code helped to identify the following:
    
    - [x] There is no sense in doing reshuffling and not using groups in the first convolution
            (indeed in the paper it is not so). **However , the result is an equivalent model**.
    - [x] It is strage that the first convolution may not be grouped,
            while the last convolution is always grouped. (**and th's different from the paper**)

    Also,
    
    - [x] There is an identity layer for pyTorch introduced here.
    - [x] The last thing to do is to get rid of **`conv1x1`** and **`conv3x3`** 
            (those are not better than the standard implementation)


## ResNet

???+ danger "ONLY PyTorch"
    ```python
    class ResNetOLD(nn.Module):
        def __init__(self, block, layers, num_classes=1000):
            self.inplanes = 64
            super(ResNetOLD, self).__init__()
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaaxPool2d(kernel_size=3, stride=2, paadding=1)
            self.layer1 = self._make_layer(block, 64, layers[0])
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
            self.avgpool = nn.AvgPool2d(7, stride=1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)
            
            for m in self.modules:
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2./n))
                elif isinstance(m, nn.BatchNord2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
        
        def _make_layer(self, block, planes, blocks, stride=1):
            downsample = None
            if stride != 1 or self.inplanes != planes*block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            
            layers = []
            layers.append(block(self.inplanes, planes, stride, downsample))
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))
            
            return nn.Sequential(*layers)
            
        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

            return x

    ```

???+ done "Using EINOPS"
    ```python
    def make_layer(inplanes, planes, block, n_blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            ## output-size won't match input-size; so adjust the residual
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        return nn.Sequential(
            block(inplanes, planes, stride, downsample),
            *[block(planes * block.expansion, planes) for _ in range(1, n_blocks)]
        )

    def ResNetNEW(block, layers, num_classes=1000):
        e = block.expansion
        
        resnet = nn.Sequential(
            Rearrange("b c h w -> b c h w", c=3, h=224, w=224),
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            make_layer(64,      64,  block, layers[0], stride=1),
            make_layer(64 * e,  128, block, layers[1], stride=2),
            make_layer(128 * e, 256, block, layers[2], stride=2),
            make_layer(256 * e, 512, block, layers[3], stride=2),
            ## Combined AvgPool & view in one single operation
            Reduce("b c h w -> b c", "mean"),
            n.Linear(512 * e, num_classes),
        )

        ## initialization
        for m in resnet.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.zero_()
        
        return resnet

    ```


## FastText

???+ danger "ONLY PyTorch"
    ```python
    class FastTextOLD(nn.Module):
        def __init__(self, vocab_size, embedding_dim, output_dim):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.fc = nn.Linear(embedding_dim, output_dim)

        def forward(self, x):
            embedded = self.embedding(x)
            embedded = embedded.permute(1, 0, 2)
            pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1)
            return self.fc(pooled)
    ```

???+ done "Using EINOPS"
    ```python hl_lines="3 7"
    def FastTextNEW(vocab_size, embedding_dim, output_dim):
        return nn.Sequential(
            Rearrange("t b -> t b"),
            nn.Embedding(vocab_size, embedding_dim),
            Reduce("t b c -> b c", "mean"),
            nn.Linear(embedding_dim, output_dim),
            Rearrange("b c -> b c"),
        )
    ```
    
    - [x] Here, the first and last operations (highlighted) do nothing and can be removed.
            But, were added to explicitly added to show expected input and output shape
    - [x] This also gives us the flexibility of changing interface by editing a single line.
            Should you need to accept inputs of shape **`(b, t)`** we just need to change 
            the line to **`Rearrange("b t -> t b")`**


## CNNs for text classification

???+ danger "ONLY PyTorch"
    ```python
    class CNNold(nn.Module):
        def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.conv_0 = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(filter_sizes[0], embedding_dim))
            self.conv_1 = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(filter_sizes[1], embedding_dim))
            self.conv_2 = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(filter_sizes[2], embedding_dim))
            self.fc = nn.Linear(len(filter_sizes)*n_filters, output_dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            x = x.permute(1, 0)
            embedded = self.embedding(x)
            embedded = embedded.unsqueeze(dim=1)

            conved_0 = F.relu(self.conv_0(embedded).squeeze(dim=3))
            conved_1 = F.relu(self.conv_1(embedded).squeeze(dim=3))
            conved_2 = F.relu(self.conv_2(embedded).squeeze(dim=3))

            pooled_0 = F.max_pool1d(conved_0, conved_0.shape[2]).squeeze(dim=2)
            pooled_1 = F.max_pool1d(conved_1, conved_1.shape[2]).squeeze(dim=2)
            pooled_2 = F.max_pool1d(conved_2, conved_2.shape[2]).squeeze(dim=2)
            
            cat = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2), dim=1))
            return self.fc(cat)

    ```

???+ done "Using EINOPS"
    ```python hl_lines="5-6 11"
    class CNNnew(nn.Module):
        def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.convs = nn.ModuleList([nn.Conv1d(embedding_dim, n_filters, kernel_size=size) 
                                        for size in filter_sizes])
            self.fc = nn.Linear(len(filter_sizes)*n_filters, output_dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            x = rearrange(x, "t b -> t b")
            emb = rearrange(self.embedding(x), "t b c -> b c t")
            pooled = [reduce(conv(emb), "b c t -> b c") for conv in self.convs]
            concatenated = rearrange(pooled, "filter b c -> b (filter c)")
            return self.fc(self.dropout(F.relu(concatenated)))
    ```

???+ quote "Discussion"
    
    - [x] Original code misuses **`nn.Conv2d`** while **`nn.Conv1d`** is the right choice.
    - [x] New code can work with any number of **`filter_sizes`** and won't fail.
    - [x] First line in the new code does nothing, but was just added for simplicity & clarity of shapes.


## Highway Convolutions

???+ danger "ONLY PyTorch"
    ```python hl_lines="4"
    class HighwayConv1dOLD(nn.Conv1d):
        def forward(self, inputs):
            L = super(HIghwayCon1dOLD, self).forward(inputs)
            H1, H2 = torch.chunk(L, 2, dim=1)   ## chunk at the feature dimension
            torch.sigmoid_(H1)
            return H1 * H2 + (1.0 - H1) * inputs
    ```

???+ done "Using EINOPS"
    ```python hl_lines="4"
    class HighwayConv1dNEW(nn.Conv1d):
        def forward(self, inputs):
            L = super().forward(inputs)
            H1, H2 = rearrange(L, "b (split c) t -> split b c t", split=2)
            torch.sigmoid_(H1)
            return H1 * H2 + (1.0 - H1) * inputs
    ```


## Simple Attention

???+ danger "ONLY PyTorch"
    ```python hl_lines="6 8"
    class Attention(nn.Module):
        def __init__(self):
            super(Attention, self).__init__()

        def forward(self, K, Q, V):
            A = torch.bmm(K.transpose((1, 2), Q) / np.sqrt(Q.shape[1])
            A = F.softmax(A, dim=1)
            R = torch.bmm(V, A)
            return torch.cat((R, Q), dim=1)
    ```

???+ done "Using EINOPS"
    ```python hl_lines="3 5"
    def attention(K, Q, V):
        _, n_channels, _ = K.shape
        A = torch.einsum("bct,bcl->btl", [K, Q])
        A = F.softmax(A * n_channels ** (-0.5), dim=1)
        R = torch.einsum("bct,btl->bcl", [V, A])
        return torch.cat((R, Q), dim=1)
    ```




## Multi-head Attention

???+ danger "ONLY PyTorch"
    ```python
    class ScaledDotProductAttention(nn.Module):
        """Scaled Dot Product Attention"""
        def __init__(self, temperature, attn_dropout=0.1):
            super().__init__()
            self.temperature = temperature
            self.dropout = nn.Dropout(attn_dropout)
            self.softmax = nn.Softmax(dim=2)

        def forward(self, q, k, v, mask=None):
            attn = torch.bmm(q, k.transpose(1, 2))
            attn /= self.temperature
            if mask is not None:
                attn = attn.masked_fill(mask, -np.inf)
            attn = self.softmax(attn)
            attn = self.dropout(attn)
            output = torch.bmm(attn, v)
            return output, attn

    class MultiHeadAttentionOLD(nn.Module):
        """Multi Head Attention Module"""
        def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
            super().__init__()
            self.d_k = d_k
            self.d_v = d_v
            self.n_head = n_head
            
            self.w_qs = nn.Linear(d_model, n_head * d_k)
            self.w_ks = nn.Linear(d_model, n_head * d_k)
            self.w_vs = nn.Linear(d_model, n_head * d_v)
            nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
            nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
            nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
            
            self.attention = ScaledDotProductAttention(temperature=d_k**0.5)
            self.layer_norm = nn.LayerNorm(d_model)
            self.fc = nn.Linear(n_head * d_v, d_model)
            nn.init.xavier_normal_(self.fc.weight)
            self.dropout = nn.Dropout(dropout)

        def forward(self, q, k, v, mask=None):
            ...
    
    ```









## References

[^1]: http://einops.rocks/pytorch-examples.html
[^2]: https://github.com/arogozhnikov/einops
