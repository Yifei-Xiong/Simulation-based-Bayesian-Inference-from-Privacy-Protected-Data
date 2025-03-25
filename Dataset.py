# Dataset for simulation-based models

import torch
import numpy as np
import math
import pandas as pd
import os
import time
import ctypes
import torch.distributions as dist
from scipy.stats import binom
from scipy.stats.qmc import Sobol
from sklearn.cluster import KMeans
from torch.multiprocessing import Pool

torch.multiprocessing.set_sharing_strategy('file_system')

class CustomDirichlet:
    def __init__(self, I, K, J, device):
        self.I = I
        self.K = K
        self.J = J
        self.device = device
        self.group_dist = dist.Dirichlet((torch.zeros(J) + 2).to(self.device))
        self.last_dist = dist.Dirichlet((torch.zeros(I) + 2).to(self.device))
        self.total_components = I * K + 1
        self.total_dim = I * K * J + I

    def sample(self, sample_shape=torch.Size()):
        sample_first = self.group_dist.sample(sample_shape + (self.I * self.K,)).reshape(sample_shape + (-1,))  # sample_shape * (I*K*J)
        sample_last = self.last_dist.sample(sample_shape)
        return torch.cat((sample_first, sample_last), dim=-1)

    def log_prob(self, value):
        assert value.shape[-1] == self.total_dim
        sample_shape = value.shape[:-1]
        value_first = value[..., 0:(self.I * self.K * self.J)].reshape(sample_shape + (self.I * self.K, self.J))
        value_last = value[..., (self.I * self.K * self.J):]
        log_prob_first = self.group_dist.log_prob(value_first)
        log_prob_last = self.last_dist.log_prob(value_last)
        return torch.sum(log_prob_first, dim=-1) + log_prob_last


class Simulator:
    def __init__(self, dataset_name, device, dtype=torch.float32, normalize=False, eps=1., flag=None):
        if os.sep == "/":
            self.FileSavePath = ""
        else:
            self.FileSavePath = ""
        self.dataset_name = dataset_name
        self.dim_theta = 0
        self.dim_x = 0
        self.device = device
        self.dtype = dtype
        self.true_theta = None
        self.normalize = normalize
        self.eps = eps  # eps-differentially private value (>0)
        assert dataset_name in ['sir', 'linear-reg', 'log-linear']
        if dataset_name == 'sir':
            self.clamp_upper = 1000.
            self.clamp_lower = 0.
            self.dim_theta = 2
            self.dim_theta_full = self.dim_theta
            self.dim_x = 10
            self.prior = torch.distributions.MultivariateNormal(torch.log(torch.tensor([0.4, 0.125])).to(self.device),
                torch.tensor([[0.5, 0], [0, 0.2]]).to(self.device), validate_args=False)
            self.dim_s = 10
            self._N = 1000  # 1000
            self._K = 1000000  # 1000000
            self._M = (self._N * self.dim_s) / self.eps  # 1000
            self.true_theta = torch.tensor([[-0.5, -3]], device=torch.device('cpu'))
            self.columns = ['$\\log \\beta$', '$\\log \\gamma$']
            self.scale = None
            if self.eps == 10.0:
                self.sdp_obs = torch.tensor(
                    [[0.0010, 0.0310, 0.6140, 0.2630, 0.1230, 0.0470, 0.0180, 0.0090, 0.0050, 0.0030]], device=device)
                self.reference_theta = torch.tensor(
                    pd.read_csv(self.FileSavePath + "reference_theta" + os.sep + "Da+0_eps10.csv").iloc[:, 1:(self.dim_theta + 1)].values,
                    device=self.device, dtype=self.dtype)
                print("using eps10 csv.")
            elif self.eps == 1.0:
                self.sdp_obs = torch.tensor(
                    [[0.0120, 0.0150, 0.7020, 0.3130, 0.1380, 0.0530, 0.0370, 0.0120, 0.0160, 0.0130]], device=device)
                self.reference_theta = torch.tensor(
                    pd.read_csv(self.FileSavePath + "reference_theta" + os.sep + "Da+0_eps1.csv").iloc[:, 1:(self.dim_theta + 1)].values,
                    device=self.device, dtype=self.dtype)
                print("using eps1 csv.")
            elif self.eps == 0.1:
                self.sdp_obs = torch.tensor(
                    [[0.0750, 0.0850, 0.6820, 0.3120, 0.2240, 0.1130, 0.1120, 0.0930, 0.0810, 0.0860]], device=device)
                self.reference_theta = torch.tensor(
                    pd.read_csv(self.FileSavePath + "reference_theta" + os.sep + "Da+0_eps.1.csv").iloc[:, 1:(self.dim_theta + 1)].values,
                    device=self.device, dtype=self.dtype)
                print("using eps.1 csv.")
            else:
                raise NotImplementedError
            if flag in [1, 2, 3, 4, 5]:  # 1: flu, 2-4: ebola, 5: covid
                self.reference_theta = None
                self._K = 100000
                self._M = 100
                self._N = 100
                self.prior = torch.distributions.MultivariateNormal(torch.tensor([-2., -3.]).to(self.device),
                                                                    torch.tensor([[1., 0], [0, 1.]]).to(self.device), validate_args=False)
                if flag == 1:
                    self.sdp_obs = torch.tensor([[0.0010, 0.0039, 0.0105, 0.0367, 0.0996, 0.2910, 0.3840, 0.3368, 0.3106, 0.2516]], device=device)
                    self.true_theta = torch.tensor([[-2.30, -3.66]], device=torch.device('cpu'))
                elif flag == 2:
                    self.sdp_obs = torch.tensor([[0.0010, 0.01105, 0.008450, 0.01290, 0.05095, 0.05200, 0.02240, 0.02115, 0.008350, 0.002250]], device=device)
                    self.true_theta = torch.tensor([[-1.46, -1.81]], device=torch.device('cpu'))
                elif flag == 3:
                    self.sdp_obs = torch.tensor([[0.0010, 0.0006500, 0.001850, 0.06635, 0.2579, 0.07415, 0.06100, 0.05420, 0.01720, 0.0003000]], device=device)
                    self.true_theta = torch.tensor([[-1.67, -2.46]], device=torch.device('cpu'))
                elif flag == 4:
                    self.sdp_obs = torch.tensor([[0.0010, 0.0079, 0.0434, 0.2156, 0.2054, 0.09275, 0.05485, 0.03655, 0.0237, 0.0232]], device=device)
                    self.true_theta = torch.tensor([[-1.58, -2.52]], device=torch.device('cpu'))
                elif flag == 5:
                    self.sdp_obs = torch.tensor([[0.0010, 0.02812, 0.05655, 0.09776, 0.2108, 0.2443, 0.2574, 0.08640, 0.04337, 0.02629]], device=device)
                    self.true_theta = torch.tensor([[-1.94, -2.95]], device=torch.device('cpu'))
                else:
                    raise NotImplementedError
            else:
                pass
                # raise NotImplementedError
        elif dataset_name == 'linear-reg':
            self.clamp_upper = 10.
            self.clamp_lower = -10.
            self._n = 100  # number of records
            self._p = 2  # dimension of X
            self.prior = torch.distributions.MultivariateNormal(torch.zeros(self._p + 1).to(device), torch.diag(torch.ones(self._p + 1, ) * 1.).to(device))
            self.prior.set_default_validate_args(False)
            self.dim_theta = self._p + 1  # 3; model parameters is \beta.
            self.dim_theta_full = self.dim_theta
            self.dim_x = (self._p + 1) * self._n  # 300
            self.dim_s = int(0.5*(self._p + 1)*(self._p + 2)) + (self._p + 1)  # 9
            self.delta = ((self._p + 3) * self._p + 3) / self._n
            self._x0_dist = torch.distributions.MultivariateNormal(torch.tensor([0.9, -1.17], device=device),
                                                                   torch.diag(torch.ones(self._p, ) * 1.).to(device))
            self._y_dist = torch.distributions.MultivariateNormal(torch.tensor([0.], device=device),
                                                                   torch.diag(torch.ones(1, ) * 2.).to(device))
            self._laplace_dist = torch.distributions.Laplace(torch.tensor([0.], device=device), torch.tensor(self.delta / self.eps, device=device))
            self.s_obs = torch.tensor([[-37.4211, -6.2943, 2.9950, 24.9949, 9.3791, -12.6974, 1.7999, -0.9352, 2.8001]], device=device) / self._n  # from Ju, 2022
            self.true_theta = torch.tensor([[-1.79, -2.89, -0.66]], device=torch.device('cpu'))
            self.columns = ['$\\beta_0$', '$\\beta_1$', '$\\beta_2$']
            self.scale = None
            if self.eps == 10.0:
                self.sdp_obs = torch.tensor(
                    [[-38.2440, -6.6783, 3.1997, 27.1985, 9.8835, -13.8503, 2.1882, -2.2884, 3.4092]], device=device) / self._n  # from Ju, 2022
                self.reference_theta = torch.tensor(
                    pd.read_csv(self.FileSavePath + "reference_theta" + os.sep + "Da+1_eps10.csv").iloc[:, 1:(self.dim_theta + 1)].values,
                    device=self.device, dtype=self.dtype)
            elif self.eps == 1.0:
                self.sdp_obs = torch.tensor(
                    [[-55.1400, -6.0000, 15.0400, 23.7800, 41.3000, -16.2100, -9.4600, -11.3300, 35.9500]], device=device) / self._n
                self.reference_theta = torch.tensor(
                    pd.read_csv(self.FileSavePath + "reference_theta" + os.sep + "Da+1_eps1.csv").iloc[:, 1:(self.dim_theta + 1)].values,
                    device=self.device, dtype=self.dtype)
            elif self.eps == 0.1:
                self.sdp_obs = torch.tensor(
                    [[-333.8400, -106.5100, -1.3100, 399.9300, 229.9800, 19.6700, 84.5900, -55.4100, 88.9000]], device=device) / self._n
                self.reference_theta = torch.tensor(
                    pd.read_csv(self.FileSavePath + "reference_theta" + os.sep + "Da+1_eps.1.csv").iloc[:, 1:(self.dim_theta + 1)].values,
                    device=self.device, dtype=self.dtype)
            else:
                raise NotImplementedError
        elif dataset_name == 'log-linear':
            self._n = 100  # number of records
            self._i = 2  # number of classes
            self._k = 2  # number of features
            self._j = 2  # possible values for each feature
            self.dim_theta = (self._j - 1) * self._k * self._i + self._i - 1  # self._i * (self._k * self._j + 1)
            self.dim_theta_full = self._i * (self._k * self._j + 1)
            self.dim_x = self._i * (self._k * self._j + 1)
            self.dim_s = self._i * self._k * self._j
            self.delta = (2 * self._k) / self._n
            self.prior = CustomDirichlet(self._i, self._k, self._j, device=self.device)
            self._laplace_dist = torch.distributions.Laplace(torch.tensor([0.], device=device), torch.tensor(2 * self._k / self.eps, device=device))
            self.sdp_obs = torch.tensor([[32.7540, 45.2008, 58.6172, 18.2683, 12.9260,  8.5805, 12.8789,  9.5372]], device=device) / self._n  # mean
            self.true_theta = torch.tensor([[0.3887, 0.6113, 0.7537, 0.2463, 0.6534, 0.3466, 0.5834, 0.4166, 0.8489, 0.1511]], device=torch.device('cpu'))
            self.columns = ['$p_{11}^1$', '$p_{12}^1$', '$p_{11}^2$', '$p_{12}^2$',
                            '$p_{21}^1$', '$p_{22}^1$', '$p_{21}^2$', '$p_{22}^2$', '$p_{1}$', '$p_{2}$']
            self._forward_transform = lambda theta_org: torch.log(theta_org[:, 0:(theta_org.shape[1] - 1)]) + (
                        torch.log((1. / theta_org[:, -1]) - 1) - torch.log(torch.sum(theta_org[:, 0:(theta_org.shape[1] - 1)], dim=1))).reshape(-1, 1)
            self._forward_logdet = lambda theta_org: -torch.sum(torch.log(theta_org), dim=1)
            self._backward_transform = lambda theta_tar: torch.cat((torch.exp(theta_tar), torch.ones(theta_tar.shape[0], 1, device=self.device)), dim=1) / torch.sum(
                torch.cat((torch.exp(theta_tar), torch.ones(theta_tar.shape[0], 1, device=self.device)), dim=1), dim=1).reshape(-1, 1)
            self._backward_logdet = lambda theta_tar: torch.sum(theta_tar, dim=1) - (theta_tar.shape[1] + 1) * torch.logsumexp(
                torch.cat((theta_tar, torch.zeros(theta_tar.shape[0], 1, device=self.device)), dim=1), dim=1)
            self.scale = None
            self.reference_theta = torch.tensor(
                pd.read_csv(self.FileSavePath + "reference_theta" + os.sep + "Da+2.csv").iloc[:, 1:(self.dim_theta_full + 1)].values,
                device=self.device, dtype=self.dtype)
        else:
            raise NotImplementedError

    def gen_data(self, para, param=None):
        # input param: theta
        # input shape: batch * dim_theta
        # output param: x from p(x|theta)
        # output shape: batch * dim_x
        batch = para.shape[0]
        if self.dataset_name == 'sir':
            cpudv = torch.device('cpu')
            para_c = torch.exp(para.cpu().type(torch.float64))
            rand_int = torch.randint(65536, size=(batch,)).cpu().reshape(-1, 1)
            K_repeat = torch.zeros(batch, 1, device=cpudv) + self._K
            para_c = torch.cat((para_c, rand_int, K_repeat), dim=1)
            if True:
                if os.sep == "\\":
                    Cfun = ctypes.WinDLL('libsir_c_msvc.dll', winmode=0)
                else:
                    Cfun = ctypes.CDLL('libsir_c.so')
                n = 10  # length for each task
                s = 4 if os.sep == "\\" else 4  # number of threads
                k = batch  # number of tasks
                input_value = torch.cat((para_c, torch.zeros(batch, 6).cpu()), dim=1)
                output_value = torch.zeros(input_value.shape[0], input_value.shape[1], dtype=self.dtype, device=cpudv)
                num_parts = (batch + k - 1) // k
                for i in range(num_parts):
                    # print(i)
                    start_idx = i * k
                    end_idx = min((i + 1) * k, batch)
                    input_list = [float(s), float(k)] + input_value[start_idx:end_idx].reshape(-1).tolist()
                    c_values = (ctypes.c_double * len(input_list))(*input_list)
                    Cfun.sir_multi_thread(c_values)
                    output_value[start_idx:end_idx] = torch.tensor([c_values[j+2] for j in range(len(c_values)-2)], device=cpudv).reshape(-1, n)
                return output_value.to(self.device)
        elif self.dataset_name == 'linear-reg':
            x0_sample = self._x0_dist.sample((batch, self._n,))
            x_sample = torch.concat((torch.ones(batch, self._n).unsqueeze(2).to(self.device), x0_sample), dim=2)
            y_sample = self._y_dist.sample((batch, self._n,)) + torch.bmm(x_sample, para.unsqueeze(2))
            # truncation & normalization
            x_sample_norm = 2. * (torch.clamp(x_sample, min=self.clamp_lower, max=self.clamp_upper) - self.clamp_lower) / (
                        self.clamp_upper - self.clamp_lower) - 1
            y_sample_norm = 2. * (torch.clamp(y_sample, min=self.clamp_lower, max=self.clamp_upper) - self.clamp_lower) / (
                        self.clamp_upper - self.clamp_lower) - 1
            x_sample_norm[:, :, 0] = 1.
            return x_sample_norm, y_sample_norm
        elif self.dataset_name == 'log-linear':
            pijk = para[:, 0:(self._i * self._k * self._j)].reshape(batch, self._i, self._k, self._j)
            pi = para[:, (self._i * self._k * self._j):]
            ni = torch.distributions.Multinomial(self._n, pi).sample()
            nijk = torch.zeros(batch, self._i, self._k, self._j).to(self.device)
            for b_ in range(batch):
                for i_ in range(self._i):
                    ni_int = int(ni[b_, i_])
                    if ni_int != 0:
                        nijk[b_, i_, :, :] = torch.distributions.Multinomial(ni_int, pijk[b_, i_, :, :]).sample()
            return torch.cat((nijk.reshape(batch, self._i * self._k * self._j), ni), dim=-1)
        else:
            raise NotImplementedError

    def gen_s(self, data, param=None):
        # input param: data
        # output param: S(data)
        # output shape: batch * dim_s
        if self.dataset_name == 'sir':
            # already get summary statistics in C++ extension
            return data
        if self.dataset_name == 'linear-reg':
            x_sample_norm, y_sample_norm = data  # unpack X, Y
            batch = x_sample_norm.shape[0]
            xty = torch.einsum('bnp,bna->bp', x_sample_norm, y_sample_norm) / self._n  # batch * (p+1)
            yty = torch.einsum('bna,bna->ba', y_sample_norm, y_sample_norm) / self._n  # batch * 1
            xtx = torch.einsum('bnp,bnq->bpq', x_sample_norm, x_sample_norm) / self._n  # batch * (p+1) * (p+1)
            return torch.concat((xty, yty, xtx.reshape(batch, -1)[:, [1, 2, 4, 5, 8]]), dim=1)  # batch * dim_s
        elif self.dataset_name == 'log-linear':
            return data[:, 0:(self._i * self._k * self._j)] / self._n
        else:
            raise NotImplementedError

    def gen_sdp(self, s, gen_num=1, qmc_flag=False, param=None):
        # input param: summary stat S, private value epsilon, generate number per sample, QMC Flag
        # input shape (of s): batch * dim_s
        # output param: privatized Sdp
        # output shape: batch * gen_num * dim_s
        batch = s.shape[0]
        dim_s = s.shape[1]
        if qmc_flag:
            m = int(np.ceil(np.log2(gen_num)))
            assert gen_num == 2 ** m
            unif_result = torch.zeros(batch, gen_num, dim_s, dtype=torch.float64)
            for idx in range(batch):
                generator = Sobol(d=dim_s)
                unif_result[idx] = torch.from_numpy(generator.random_base2(m=m))
        else:
            unif_result = torch.rand(batch, gen_num, dim_s, dtype=torch.float64)
        if self.dataset_name in ['linear-reg', 'log-linear']:
            # inverse cumulative
            noise = - (self.delta / self.eps) * torch.sgn(unif_result - 0.5) * torch.log(1 - 2 * (torch.abs(unif_result - 0.5)) + 1e-8)
            assert torch.all(torch.isfinite(noise))
            return noise.type(torch.float32).to(self.device) + s.reshape(batch, 1, -1).expand(batch, gen_num, -1)
        elif self.dataset_name == 'sir':
            binom_params = (s + self._M) / (self._K + self._M * 2)  # batch * dim_s
            # unif_result.shape = batch * gen_num * dim_s, the input p-value
            quantile = torch.tensor(binom.ppf(unif_result.cpu().numpy(), self._N, binom_params.reshape(batch, 1, dim_s).cpu().numpy()), dtype=torch.float32)
            return (quantile / self._N).to(self.device)
        else:
            raise NotImplementedError

    def forward_transform(self, theta_org):
        # transforming bounded prior regions to unbounded regions
        if self.dataset_name == 'log-linear':
            batch = theta_org.shape[0]
            value_first = self._forward_transform(theta_org[:, 0:(self._i * self._k * self._j)].reshape(-1, self._j)).reshape(batch, -1)
            value_last = self._forward_transform(theta_org[:, (self._i * self._k * self._j):])
            return torch.cat((value_first, value_last), dim=1)
        else:
            return theta_org

    def forward_logdet(self, theta_org):
        # return log transformed jacobian, theta should in prior support
        if self.dataset_name == 'log-linear':
            batch = theta_org.shape[0]
            value_first = torch.sum(self._forward_logdet(theta_org[:, 0:(self._i * self._k * self._j)].reshape(-1, self._j)).reshape(batch, -1), dim=1)
            value_last = self._forward_logdet(theta_org[:, (self._i * self._k * self._j):])
            return value_first + value_last
        else:
            return torch.tensor([0.]).to(self.device)

    def backward_transform(self, theta_tar):
        # transforming unbounded regions to bounded prior regions
        if self.dataset_name == 'log-linear':
            batch = theta_tar.shape[0]
            split_idx = (self._j - 1) * self._k * self._i
            value_first = self._backward_transform(theta_tar[:, 0:split_idx].reshape(-1, self._j - 1)).reshape(batch, -1)
            value_last = self._backward_transform(theta_tar[:, split_idx:])
            return torch.cat((value_first, value_last), dim=1)
        else:
            return theta_tar

    def backward_logdet(self, theta_tar):
        # return log transformed jacobian, theta range has no limited
        if self.dataset_name == 'log-linear':
            batch = theta_tar.shape[0]
            split_idx = (self._j - 1) * self._k * self._i
            value_first = torch.sum(self._backward_logdet(theta_tar[:, 0:split_idx].reshape(-1, self._j - 1)).reshape(batch, -1), dim=1)
            value_last = self._backward_logdet(theta_tar[:, split_idx:])
            return value_first + value_last
        else:
            return torch.tensor([0.]).to(self.device)


def process_batch(idx, dim_s, m, unif_result):
    generator = Sobol(d=dim_s)
    unif_result[idx] = torch.from_numpy(generator.random_base2(m=m))

