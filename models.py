import torch.nn as nn
import torch
import math


class DeepNN(nn.Module):
    sigma = 1
    beta = 1

    def __init__(self, d_in, d_out, L, C_L, activation='relu', init='custom'):
        """
        :param d_in: (int) input size
        :param d_out: (int) output size
        :param L: (int) depth
        :param C_L: (int) scaling factor
        :param activation: (str) activation function, default='relu'
        :param init: (str) type of parameter initialization, default='custom'
        """

        if init is not None:
            assert init in ['custom']
        assert L > 1

        super(DeepNN, self).__init__()

        self.d_in = d_in
        self.d_out = d_out
        self.L = L
        self.C_L = C_L
        self.init = init

        self.head = nn.Linear(d_in, d_in + d_out + 1)
        self.body = [nn.Linear(d_in + d_out + 1, d_in + d_out + 1) for _ in range(L - 1)]
        self.body = nn.ModuleList(self.body)
        self.tail = nn.Linear(d_in + d_out + 1, d_out)

        if activation == 'relu':
            self.activation = torch.nn.ReLU(inplace=True)

        if init == 'custom':
            self.custom_init()

    def custom_init(self):
        with torch.no_grad():
            self.head.weight[0:self.d_in, 0:self.d_in] = self.C_L * torch.eye(self.d_in)
            self.head.weight[self.d_in] = torch.normal(torch.zeros((self.d_in,)), self.sigma / math.sqrt(self.d_in))
            self.head.weight[self.d_in+1:, :] = torch.zeros((self.d_out, self.d_in))

            self.head.bias[0:self.d_in] = torch.zeros((self.d_in,))
            self.head.bias[self.d_in] = torch.normal(0, self.C_L * self.beta, size=(1,))
            self.head.bias[self.d_in + 1:] = self.C_L * torch.ones((self.d_out,))

            for i in range(len(self.body)):
                nn.init.eye_(self.body[i].weight)
                self.body[i].weight[self.d_in][0:self.d_in] = torch.normal(torch.zeros((self.d_in,)),
                                                                           self.sigma / math.sqrt(self.d_in))
                self.body[i].weight[self.d_in][self.d_in] = 0

                nn.init.zeros_(self.body[i].bias)
                self.body[i].bias[self.d_in] = torch.normal(0, self.C_L * self.beta, size=(1,))

            nn.init.zeros_(self.tail.weight)
            self.tail.weight[:, self.d_in+1:] = torch.eye(self.d_out)

            nn.init.constant_(self.tail.bias, -self.C_L)

    def forward(self, x):
        out = self.head(x)
        out = self.activation(out)

        for i in range(len(self.body)):
            out = self.body[i](out)
            out = self.activation(out)

        out = self.tail(out)
        return out


# parallelized version of DeepNN
class ParallelDeepNN(DeepNN):
    def __init__(self, d_in, d_out, L, C_L, activation='relu', init='custom', dev_list=None):
        """
        :param dev_list: (list of str) list of device names
        """
        assert len(dev_list) > 1

        super(ParallelDeepNN, self).__init__(d_in, d_out, L, C_L, activation, init)

        self.dev_list = dev_list

        self.block_partition = [i * (len(self.body) // len(self.dev_list)) for i in range(len(self.dev_list))]

        self.head = self.head.to(self.dev_list[0])
        for i in range(len(self.block_partition) - 1):
            for j in range(self.block_partition[i], self.block_partition[i + 1]):
                self.body[j] = self.body[j].to(self.dev_list[i])

        for i in range(self.block_partition[-1], len(self.body)):
            self.body[i] = self.body[i].to(self.dev_list[-1])

        self.tail = self.tail.to(self.dev_list[-1])

    def forward(self, x):
        out = self.head(x)

        out = self.activation(out)

        for i in range(len(self.block_partition) - 1):
            out = out.to(self.dev_list[i])
            for j in range(self.block_partition[i], self.block_partition[i + 1]):
                out = self.body[j](out)
                out = self.activation(out)

        out = out.to(self.dev_list[-1])
        for i in range(self.block_partition[-1], len(self.body)):
            out = self.body[i](out)
            out = self.activation(out)

        out = self.tail(out)

        return out


class DeepCNN(nn.Module):
    sigma = 1
    beta = 1
    pool_size = 4

    def __init__(self, input_size, c_in, c_out, L, C_L, kernel_size=3, activation='relu', init='custom'):
        """
        :param input_size: (int) size of input along one dimension: e.g. input shpae = 28x28 -> input_size = 28
        :param c_in: (int) number of input channels
        :param c_out: (int) number of output channels
        :param L: (int) depth
        :param C_L: (int) scaling factor
        :param kernel_size: (int) size of kernel along one dimension: e.g. 3x3 kernel -> kernel_size = 3, default = 3
        :param activation: (str) activation function, default='relu'
        :param init: (str) type of parameter initialization, default='custom'
        """


        assert L > 1
        super(DeepCNN, self).__init__()
        self.input_size = input_size
        self.c_in = c_in
        self.c_out = c_out
        self.L = L
        self.C_L = C_L
        self.kernel_size = kernel_size

        self.avgpool = nn.AvgPool2d(self.pool_size)

        self.head = nn.Conv2d(in_channels=c_in, out_channels=c_out + 2, kernel_size=self.kernel_size, stride=1,
                              padding=1, bias=True)

        self.body = [nn.Conv2d(in_channels=c_out + 2, out_channels=c_out + 2, kernel_size=self.kernel_size, stride=1,
                               padding=1, bias=True)
                     for _ in range(self.L - 1)]
        self.body = nn.ModuleList(self.body)

        self.tail = nn.Conv2d(in_channels=c_out + 2, out_channels=c_out, kernel_size=self.kernel_size, stride=1,
                              padding=1, bias=True)

        self.avgpool_2 = nn.AvgPool2d(self.input_size // self.pool_size)

        if init == 'custom':
            self.custom_init()

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)

    def custom_init(self):
        mid_idx = self.kernel_size // 2
        with torch.no_grad():
            nn.init.zeros_(self.head.weight[0][0])
            self.head.weight[0][0][mid_idx][mid_idx] = self.C_L
            nn.init.normal_(self.head.weight[1][0], mean=0, std=self.sigma / (math.sqrt(self.c_in) * self.kernel_size))
            nn.init.zeros_(self.head.weight[2:, :, :, :])
            nn.init.zeros_(self.head.bias[0])
            nn.init.normal_(self.head.bias[1], mean=0, std=self.beta * self.C_L)
            nn.init.constant_(self.head.bias[2:], self.C_L)

            for i in range(len(self.body)):
                nn.init.zeros_(self.body[i].weight[0])
                self.body[i].weight[0][0][mid_idx][mid_idx] = 1

                nn.init.normal_(self.body[i].weight[1][0], mean=0,
                                std=self.sigma / (math.sqrt(self.c_in) * self.kernel_size))
                nn.init.zeros_(self.body[i].weight[1, 1:, :, :])

                nn.init.zeros_(self.body[i].weight[2:, :, :, :])
                for j in range(self.c_out):
                    self.body[i].weight[2 + j][2 + j][mid_idx][mid_idx] = 1

                nn.init.zeros_(self.body[i].bias)
                nn.init.normal_(self.body[i].bias[1], mean=0, std=self.beta * self.C_L)

            for i in range(self.c_out):
                nn.init.zeros_(self.tail.weight[i])
                self.tail.weight[i][i + 2][mid_idx][mid_idx] = 1

            nn.init.constant_(self.tail.bias, -self.C_L)

    def forward(self, x):
        out = self.avgpool(x)

        out = self.head(out)
        out = self.activation(out)

        for i in range(len(self.body)):
            out = self.body[i](out)
            out = self.activation(out)

        out = self.tail(out)

        out = self.avgpool_2(out)

        return out


# parallelized version of DeepCNN
class ParallelDeepCNN(DeepCNN):
    def __init__(self, input_size, c_in, c_out, L, C_L, kernel_size=3, activation='relu', init='custom', dev_list=None):
        """
        :param dev_list: (list of str) list of device names
        """
        assert len(dev_list) > 1

        super(ParallelDeepCNN, self).__init__(input_size=input_size, c_in=c_in, c_out=c_out, L=L, C_L=C_L,
                                              kernel_size=kernel_size, activation=activation, init=init)

        self.dev_list = dev_list

        self.block_partition = [i * (len(self.body) // len(self.dev_list)) for i in range(len(self.dev_list))]

        self.avgpool = self.avgpool.to(self.dev_list[0])

        self.head = self.head.to(self.dev_list[0])
        for i in range(len(self.block_partition) - 1):
            for j in range(self.block_partition[i], self.block_partition[i + 1]):
                self.body[j] = self.body[j].to(self.dev_list[i])

        for i in range(self.block_partition[-1], len(self.body)):
            self.body[i] = self.body[i].to(self.dev_list[-1])

        self.tail = self.tail.to(self.dev_list[-1])

        self.avgpool_2 = self.avgpool_2.to(self.dev_list[-1])

    def forward(self, x):
        out = self.avgpool(x)

        out = self.head(out)

        out = self.activation(out)

        for i in range(len(self.block_partition) - 1):
            out = out.to(self.dev_list[i])
            for j in range(self.block_partition[i], self.block_partition[i + 1]):
                out = self.body[j](out)
                out = self.activation(out)

        out = out.to(self.dev_list[-1])
        for i in range(self.block_partition[-1], len(self.body)):
            out = self.body[i](out)
            out = self.activation(out)

        out = self.tail(out)

        out = self.avgpool_2(out)

        return out






