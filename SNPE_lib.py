# This library is an implementation of (conditional) neural density structures
# including: mixture of Gaussian (MoG), masked autoregressive flow (MAF), neural spline flow (NSF)
# Requires pytorch library, recommended python version 3.7+.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time
from torch.autograd import Variable
from sbibm.utils.nflows import get_flow


# Conditional Gaussian Mixture: q(\theta | x) = \sum_{i=1}^k \alpha_i N(\theta | \mu_i, \Sigma_i)
class Cond_Mix_Gauss(nn.Module):
    def __init__(self, dim_x, dim_theta, k, n_hidden, keep_weight=False):
        """
        :param dim_x: dimension of x (input dim)
        :param dim_theta: dimension of theta (output dim)
        :param k: gaussian mixture number, default is 8
        :param n_hidden: list of hidden units, default is [50, 50]
        :param keep_weight: True: change alpha; False: set alpha = 1/k without change
        """

        super(Cond_Mix_Gauss, self).__init__()
        self.dim_x = dim_x
        self.dim_theta = dim_theta
        self.k = k
        self.n_hidden = n_hidden
        self.act_func = nn.ReLU()
        self.keep_weight = keep_weight

        self.fc_1 = nn.Linear(dim_x, n_hidden[0])
        self.fc_2 = nn.Linear(n_hidden[0], n_hidden[1])
        self.fc_alpha = nn.Linear(n_hidden[1], k)
        self.fc_mu = nn.Linear(n_hidden[1], k * dim_theta)
        self.fc_sigma = nn.Linear(n_hidden[1], k * dim_theta * dim_theta)

        triu_mask = torch.triu(torch.ones(dim_theta, dim_theta), diagonal=1)
        diag_mask = torch.diag(torch.ones(dim_theta))
        self.register_buffer('triu_mask', Variable(triu_mask))
        self.triu_mask.requires_grad = False
        self.register_buffer('diag_mask', Variable(diag_mask))
        self.diag_mask.requires_grad = False
        invariant_alpha = torch.ones(k) / k
        self.register_buffer('invariant_alpha', Variable(invariant_alpha))
        self.invariant_alpha.requires_grad = False

    def forward(self, x):
        x = self.act_func(self.fc_1(x))
        x = self.act_func(self.fc_2(x))
        if self.keep_weight:
            alpha = self.invariant_alpha.reshape(1, -1).repeat(x.shape[0], 1)
        else:
            alpha = F.softmax(self.fc_alpha(x), dim=1)  # batch * k
        mu = self.fc_mu(x).reshape(-1, self.k, self.dim_theta)  # batch * k * dim
        # note that U here is an upper triangular matrix such that U'*U is the precision
        sigma = self.fc_sigma(x).reshape(-1, self.k, self.dim_theta, self.dim_theta)  # batch * k * dim_t * dim_t
        sigma = torch.einsum('abcd,cd->abcd', sigma, self.triu_mask) + \
                torch.einsum('abcd,cd->abcd', torch.exp(sigma.clamp_(-5, 3)), self.diag_mask)
        sigma_cache = torch.einsum('abcd,abde->abce', sigma.permute(0, 1, 3, 2), sigma)
        return alpha, mu, sigma_cache + (self.diag_mask[None, None, :, :] * 5e-3)
        # return alpha, mu, sigma

    def density(self, theta, alpha, mu, sigma):
        # theta: batch * dim_t
        # alpha: batch * k
        # mu: batch * k * dim_t
        # sigma: batch * k * dim_t * dim_t
        # return: batch * 1
        k = alpha.shape[1]
        dim_t = theta.shape[1]
        arg_theta = theta.unsqueeze(1).repeat(1, k, 1)  # batch * k * dim_t
        multivariate_normal_dist = torch.distributions.MultivariateNormal(mu.reshape(-1, dim_t),
                                                                          sigma.reshape(-1, dim_t, dim_t))
        density_value = torch.exp(multivariate_normal_dist.log_prob(arg_theta.reshape(-1, dim_t)) + 1e-7)
        return torch.sum(density_value.reshape(-1, k) * alpha, dim=1)

    def density2(self, theta, density_family_parameter):
        # return log_prob of gmm
        # alpha shape: batch * k; mu shape: batch * k * dim_t; sigma shape: batch * k * dim_t * dim_t
        alpha, mu, sigma = density_family_parameter
        mix = torch.distributions.Categorical(alpha)
        comp = torch.distributions.MultivariateNormal(mu, sigma)
        gmm = torch.distributions.MixtureSameFamily(mix, comp)
        return gmm.log_prob(theta)

    def log_density_value_at_data(self, data_sample, theta_sample):
        return self.density2(theta_sample, self.forward(data_sample))

    def density_evaluate(self, theta, evaluate_para):
        # theta: batch * dim_t
        # alpha: 1 * k
        # mu: 1 * k * dim_t
        # sigma: 1 * k * dim_t * dim_t
        # return: batch * 1
        alpha, mu, sigma = evaluate_para
        mix = torch.distributions.Categorical(alpha.squeeze(0))
        comp = torch.distributions.MultivariateNormal(mu.squeeze(0), sigma.squeeze(0))
        gmm = torch.distributions.MixtureSameFamily(mix, comp)
        return torch.exp(gmm.log_prob(theta))

    def multivariate_normal_density(self, x, mu, sigma):
        return (1.0 / (math.sqrt((2 * math.pi) ** 2 * torch.det(sigma))) * torch.exp(
            -torch.einsum('bd,bde,be->bde', x - mu, torch.inverse(sigma), x - mu) / 2))

    def gen_dist(self, density_family_parameter):
        alpha, mu, sigma = density_family_parameter
        mix = torch.distributions.Categorical(alpha.squeeze(0))
        comp = torch.distributions.MultivariateNormal(mu.squeeze(0), sigma.squeeze(0))
        return torch.distributions.MixtureSameFamily(mix, comp)

    def gen_sample(self, sample_size, x_0, qmc_flag=False):
        param = self.forward(x_0)
        dist = self.gen_dist(param)
        if qmc_flag:
            data = None
        else:
            data = dist.sample((sample_size,))
        return data


class Cond_MADE(nn.Module):
    def __init__(self, dim_in, dim_out, n_hidden, device, random_order=False, random_degree=False, residual=False):
        """
        :param dim_in: dimension of (conditional) inputs
        :param dim_out: dimension of outputs
        :param n_hidden: list of hidden units, default is [50, 50]
        :param device: -
        :param random_order: Whether to use random input order, default is False.
        :param random_degree: Whether to use random degree, default is False.
        :param residual: Whether to enable residual structure.
        """

        super(Cond_MADE, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out  # for gaussian mu and log-sigma output
        self.dim_condition = dim_in - dim_out
        self.n_hidden = n_hidden
        # self.act_func = nn.ReLU()
        self.act_func = nn.Tanh()
        self.random_order = random_order
        self.random_degree = random_degree
        self.device = device
        self.residual = residual

        self.degrees = self.create_degrees(dim_out, n_hidden, random_order, random_degree)  # only connect x to first hidden layer!!
        # add degrees for conditional param
        self.degrees[0] = np.concatenate(([0] * (dim_in - dim_out), self.degrees[0])).astype('int32')
        # self.mask_matrix = self.create_mask(self.degrees)

        dim_list = [self.dim_in, *n_hidden, self.dim_out * 2]
        if not residual:
            self.layers = []
            for i in range(len(dim_list) - 2):
                self.layers.append(MaskedLinear(dim_list[i], dim_list[i + 1]), )
                self.layers.append(self.act_func)
            self.layers.append(MaskedLinear(dim_list[-2], dim_list[-1]))
            self.model = nn.Sequential(*self.layers)
            mask_matrix = self.create_mask(self.degrees)
            mask_iter = iter(mask_matrix)
            for module in self.model.modules():
                if isinstance(module, MaskedLinear):
                    module.initialise_mask(torch.tensor(next(mask_iter).transpose(), device=self.device))
            # self.set_masked_linear()
        else:
            assert len(n_hidden) == 2 and len(dim_list) == 4  # TBA
            self.linear1 = MaskedLinear(dim_list[0], dim_list[1])
            self.linear2 = MaskedLinear(dim_list[1], dim_list[2])
            self.linear3 = MaskedLinear(dim_list[2], dim_list[3])
            self.linear_res = MaskedLinear(dim_list[0], dim_list[2])
            mask_matrix = self.create_mask(self.degrees)
            self.linear1.initialise_mask(torch.tensor(mask_matrix[0].transpose(), device=self.device))
            self.linear2.initialise_mask(torch.tensor(mask_matrix[1].transpose(), device=self.device))
            self.linear3.initialise_mask(torch.tensor(mask_matrix[2].transpose(), device=self.device))
            mask_matrix_res = self.create_mask([self.degrees[0], self.degrees[2]])
            self.linear_res.initialise_mask(torch.tensor(mask_matrix_res[0].transpose(), device=self.device))

    def create_degrees(self, dim_in, n_hidden, random_order, random_degree):
        # for p(theta|x), only connect x to first hidden layer
        degrees = []
        # create degrees for inputs
        if isinstance(random_order, bool):
            if random_order:
                degrees_0 = np.arange(1, dim_in + 1)
                np.random.shuffle(degrees_0[self.dim_condition:])
                print("MADE using random input order.")
            else:
                # print("MADE using sequential input order.")
                degrees_0 = np.arange(1, dim_in + 1)

        else:
            input_order = np.array(random_order)
            assert np.all(np.sort(input_order) == np.arange(1, dim_in + 1)), 'invalid input order'
            degrees_0 = input_order
        degrees.append(degrees_0)
        # create degrees for hiddens
        if random_degree:
            for N in n_hidden:
                min_prev_degree = min(np.min(degrees[-1]), dim_in - 1)
                degrees_l = np.random.randint(min_prev_degree, dim_in, N)
                degrees.append(degrees_l)
        else:
            for N in n_hidden:
                degrees_l = np.arange(N) % max(1, dim_in - 1) + min(1, dim_in - 1)
                degrees.append(degrees_l)
        if random_degree:
            pass
        return degrees

    def create_mask(self, degrees):
        Ms = []
        for l, (d0, d1) in enumerate(zip(degrees[:-1], degrees[1:])):
            Ms.append(d0[:, np.newaxis] <= d1)
        # last_mat = degrees[-1][:, np.newaxis] < degrees[0][-int(self.dim_out/2):]
        last_mat = (degrees[-1][:, np.newaxis] < degrees[0])[:, self.dim_condition:]
        Ms.append(np.concatenate((last_mat, last_mat), axis=1))
        return Ms

    def set_masked_linear(self):
        mask_iter = iter(self.mask_matrix)
        for module in self.model.modules():
            if isinstance(module, MaskedLinear):
                module.initialise_mask(torch.tensor(next(mask_iter).transpose(), device=self.device))

    def forward(self, x):
        if self.residual:
            iden = self.linear_res(x)
            out = self.linear1(x)
            out = self.act_func(out)
            out = self.linear2(out)
            out += iden
            out = self.act_func(out)
            out = self.linear3(out)
            return out
        else:
            return self.model(x)

    def log_density_value_at_data(self, data_sample, theta_sample):
        param = self.forward(torch.cat((data_sample, theta_sample), dim=1))
        # mu, logs = torch.split(param, int(param.shape[1] / 2), dim=1)
        mu, logs = torch.chunk(param, 2, dim=1)
        log_density = - logs - torch.log(2 * torch.tensor(math.pi)) / 2 - ((theta_sample - mu) / torch.exp(logs + 1e-7)) ** 2 / 2
        return torch.sum(log_density, dim=1)


class MaskedLinear(nn.Linear):
    def __init__(self, n_in: int, n_out: int, bias: bool = True) -> None:
        super().__init__(n_in, n_out, bias)
        self.mask = None

    def initialise_mask(self, mask):
        # mask shape: (out_features, in_features)
        self.mask = mask

    def forward(self, x):
        # overrride return F.linear(input, self.weight, self.bias)
        return F.linear(x, self.mask * self.weight, self.bias)


class BatchNormLayer(nn.Module):
    def __init__(self, dim_in, dim_out, eps=1e-5):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(1, dim_out))
        self.beta = nn.Parameter(torch.zeros(1, dim_out))
        self.batch_mean = None
        self.batch_var = None

    def forward(self, x):
        x_part_1 = x[:, :(self.dim_in - self.dim_out)]
        x_part_2 = x[:, (self.dim_in - self.dim_out):]
        x_hat, log_det = self._forward(x_part_2)
        return torch.cat((x_part_1, x_hat), dim=1), log_det

    def _forward(self, x):
        # x[(self.dim_in - self.dim_out):]
        if self.training:
            m = x.mean(dim=0)
            v = x.var(dim=0) + self.eps  # torch.mean((x - m) ** 2, axis=0) + self.eps
            # v = torch.mean((x - m) ** 2, dim=0) + self.eps
            self.batch_mean = None
        else:
            if self.batch_mean is None:
                self.set_batch_stats_func(x)
            m = self.batch_mean.clone()
            v = self.batch_var.clone()

        x_hat = (x - m) / torch.sqrt(v)
        x_hat = x_hat * torch.exp(self.gamma) + self.beta
        log_det = torch.sum(self.gamma - 0.5 * torch.log(v))
        return x_hat, log_det

    def backward(self, x):
        x_part_1 = x[:, :(self.dim_in - self.dim_out)]
        x_part_2 = x[:, (self.dim_in - self.dim_out):]
        x_hat, log_det = self._backward(x_part_2)
        return torch.cat((x_part_1, x_hat), dim=1), log_det

    def _backward(self, x):
        if self.training:
            m = x.mean(dim=0)
            v = x.var(dim=0) + self.eps
            self.batch_mean = None
        else:
            if self.batch_mean is None:
                self.set_batch_stats_func(x)
            m = self.batch_mean
            v = self.batch_var

        x_hat = (x - self.beta) * torch.exp(-self.gamma) * torch.sqrt(v) + m
        log_det = torch.sum(-self.gamma + 0.5 * torch.log(v))
        return x_hat, log_det

    def set_batch_stats_func(self, x):
        # print("setting batch stats for validation")
        self.batch_mean = x.mean(dim=0)
        self.batch_var = x.var(dim=0) + self.eps


class Cond_MAF_Layer(nn.Module):
    def __init__(self, dim_in, dim_out, n_hidden, device, reverse=True, random_order=False, random_degree=False, residual=False):
        """
        :param dim_in: dimension of (conditional) inputs
        :param dim_out: dimension of outputs
        :param n_hidden: list of hidden units, default is [50, 50]
        :param device: -
        :param reverse: Whether to reverse input in each MADE.
        :param random_order: Whether to use random input order, default is False.
        :param random_degree: Whether to use random degree, default is False.
        :param residual: Whether to enable residual structure.
        """

        super(Cond_MAF_Layer, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.n_hidden = n_hidden
        self.device = device
        self.reverse = reverse
        self.random_order = random_order
        self.random_degree = random_degree
        self.residual = residual
        self.made = Cond_MADE(dim_in, dim_out, n_hidden, device, random_order, random_degree, residual)
        self.param_trun = False

    def forward(self, x):
        mu, logs = torch.chunk(self.made(x), 2, dim=1)
        if self.param_trun:
            mu[mu > 100.] = 100.
            mu[mu < -100.] = -100.
            logs[logs > 20.] = 20.
            logs[logs < -20.] = -20.
            # mu.clamp_(-100., 100.)
            # logs.clamp_(-20., 20.)
        u = (x[:, (self.dim_in - self.dim_out):] - mu) * torch.exp(-logs + 1e-7)
        if self.reverse:
            x = torch.cat((x[:, 0:(self.dim_in - self.dim_out)].flip(dims=(1,)), u.flip(dims=(1,))), dim=1)
        else:
            x = torch.cat((x[:, 0:(self.dim_in - self.dim_out)], u), dim=1)
        return x, - torch.sum(logs, dim=1)

    def backward(self, u):
        if self.reverse:
            u = torch.cat((u[:, 0:(self.dim_in - self.dim_out)].flip(dims=(1,)),
                           u[:, (self.dim_in - self.dim_out):].flip(dims=(1,))), dim=1)
        x = torch.zeros_like(u)
        # print('backward fun called')
        x[:, 0:(self.dim_in - self.dim_out)] = u[:, 0:(self.dim_in - self.dim_out)]
        for dim in range(self.dim_out):
            mu, logs = torch.chunk(self.made(x), 2, dim=1)
            logs = torch.clamp(logs, max=10)
            x[:, (dim + self.dim_in - self.dim_out)] = mu[:, dim] + u[:, (dim + self.dim_in - self.dim_out)] * torch.exp(logs[:, dim])
        log_det = torch.sum(logs, dim=1)
        return x, log_det


class Cond_MAF(nn.Module):
    def __init__(self, dim_in, dim_out, n_layer, n_hidden, device, batch_norm=False,
                 reverse=True, random_order=False, random_degree=False, residual=False):
        """
        :param dim_in: dimension of (conditional) inputs
        :param dim_out: dimension of outputs
        :param n_layer: layer size of MADE
        :param n_hidden: list of hidden units, default is [50, 50]
        :param device: -
        :param batch_norm: Whether to enable batch normalization
        :param reverse: Whether to reverse input in each MADE.
        :param random_order: Whether to use random input order, default is False.
        :param random_degree: Whether to use random degree, default is False.
        :param residual: Whether to enable residual structure.
        """

        super(Cond_MAF, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.n_layer = n_layer
        self.n_hidden = n_hidden
        self.device = device
        self.reverse = reverse
        self.random_order = random_order
        self.random_degree = random_degree
        self.layers = nn.ModuleList()
        self.batch_norm = batch_norm
        self.residual = residual
        for lay in range(n_layer):
            self.layers.append(Cond_MAF_Layer(dim_in, dim_out, n_hidden, device, reverse, random_order, random_degree, residual))
            # print(lay)
            if self.batch_norm and lay != (n_layer - 1):
                # if self.batch_norm:
                self.layers.append(BatchNormLayer(dim_in, dim_out))
                # self.layers.append(nn.BatchNorm1d(dim_out))

    def forward(self, x):
        log_det_sum = torch.zeros(x.shape[0], device=self.device)  # x.shape[0] is batch_size
        # layer_is_bn = False
        for layer in self.layers:
            x, log_det = layer(x)
            log_det_sum += log_det
            '''
            if layer_is_bn:
                # x[:, (self.dim_in - self.dim_out):] = layer(x[:, (self.dim_in - self.dim_out):])
                bnout, log_det = layer(x[:, (self.dim_in - self.dim_out):])
                x = torch.cat((x[:, 0:(self.dim_in - self.dim_out)], bnout), dim=1)
                log_det_sum += log_det
            else:
                x, log_det = layer(x)
                log_det_sum += log_det
            layer_is_bn = not layer_is_bn
            '''
        return x, log_det_sum

    def backward(self, x):
        log_det_sum = torch.zeros(x.shape[0], device=self.device)
        for layer in reversed(self.layers):
            x, log_det = layer.backward(x)
            log_det_sum += log_det

        return x, log_det_sum

    def log_density_value_at_data(self, data_sample, theta_sample):
        x, log_det_sum = self.forward(torch.cat((data_sample, theta_sample), dim=1))
        u = x[:, (self.dim_in - self.dim_out):]
        log_density = - self.dim_out * torch.log(2 * torch.tensor(math.pi)) / 2 - (u ** 2).sum(dim=1) / 2 + log_det_sum
        return log_density

    def gen_sample(self, sample_size, x_0, qmc_flag=False):
        if qmc_flag:
            normal_data = None
        else:
            dist = torch.distributions.MultivariateNormal(torch.zeros(self.dim_out), torch.diag(torch.ones(self.dim_out, )))
            normal_data = dist.sample((sample_size,)).to(self.device)

        input_data = torch.cat((x_0.repeat([normal_data.shape[0], 1]), normal_data), dim=1)
        out, _ = self.backward(input_data)
        # print('sample gene success')
        return out[:, (self.dim_in - self.dim_out):]


class Cond_NSF(nn.Module):
    def __init__(self, dim_in, dim_out, n_layer, n_hidden, device):
        """
        :param dim_in: dimension of (conditional) inputs
        :param dim_out: dimension of outputs
        :param n_layer: layer size of Block
        :param n_hidden: list of hidden units, default is [50, 50]
        :param device: -
        """

        super(Cond_NSF, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.n_layer = n_layer
        self.n_hidden = n_hidden
        self.nsf_from_sbibm = get_flow(model="nsf", dim_distribution=dim_out, dim_context=dim_in - dim_out,
                                       hidden_features=n_hidden[0], flow_num_transforms=n_layer).to(device)

    def log_density_value_at_data(self, data_sample, theta_sample):
        return self.nsf_from_sbibm.log_prob(theta_sample, data_sample)

    def gen_sample(self, sample_size, x_0, qmc_flag=False):
        return self.nsf_from_sbibm.sample(int(sample_size), x_0).squeeze(0)


def MCMC_MH(log_density, init_value, sample_size, generate_size, cut_size, proposal_std=1., seq_sample=True, disp=True, batch=10):
    # Metropolis-Hastings MCMC
    # drop first 'cut_size' samples
    # assert generate_size >= (sample_size + cut_size)
    sample = init_value
    device = init_value.device
    dim = init_value.shape[1]
    total_sample = torch.zeros(generate_size, dim, device=device)
    j_dist = torch.distributions.MultivariateNormal(torch.zeros(dim).to(device), torch.diag(torch.ones(dim, )).to(device))
    uniform_dist_sample = torch.distributions.Uniform(0, 1).sample((generate_size,)).to(device)
    j_sample = j_dist.sample((generate_size,)).to(device) * proposal_std  # normal(0, I * proposal_std ^ 2)
    change_times = 0
    last_round_log_density = log_density(sample).squeeze()
    index = 0
    while True:
        if (index + batch) > generate_size:
            batch = generate_size - index
        if batch == 0:
            break
        sample_new = j_sample[index:(index + batch)] + sample  # batch * dim
        log_density_new = log_density(sample_new).reshape(-1)  # batch
        compare = uniform_dist_sample[index:(index + batch)] > torch.exp(log_density_new - last_round_log_density)  # batch
        if torch.all(compare):
            total_sample[index:(index + batch)] = sample.repeat(batch, 1)
            index += batch
        else:
            trans_idx = torch.where(torch.logical_not(compare))[0][0].item()
            total_sample[index:(index + trans_idx)] = sample.repeat(trans_idx, 1)
            total_sample[index + trans_idx] = sample_new[trans_idx]
            sample = sample_new[trans_idx]
            last_round_log_density = log_density_new[trans_idx]
            change_times += 1
            index += (trans_idx + 1)
    if disp:
        print("mcmc change rate: %.3f, subsamp %d from %d" % (change_times / generate_size, sample_size, generate_size - cut_size))
    if seq_sample:  # Isometric samples
        idx = torch.linspace(cut_size, generate_size - 1, sample_size).type(torch.int64)
    else:
        perm = torch.randperm(generate_size - cut_size)
        idx = perm[:sample_size] + cut_size
    return total_sample[idx]
