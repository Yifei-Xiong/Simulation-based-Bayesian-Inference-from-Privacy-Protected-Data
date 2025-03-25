import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
import shutil
import seaborn as sns
import Dataset
import datetime


def calc_dist(stats_1, stats_2):
    """Calculates the distance between two observations. Here the euclidean distance is used."""
    diff = stats_1 - stats_2
    dist = np.sqrt(np.dot(diff, diff))
    return dist


def calc_dist_batch(stats_1, stats_2, scale=None):
    """Calculates the distance between two observations. Here the euclidean distance is used."""
    # stats_1: shape: batch * dim
    # state_2: shape: batch * dim
    if scale is None:
        return torch.sqrt(torch.sum((stats_1 - stats_2) ** 2, dim=1))
    else:
        return torch.sqrt(torch.sum(((stats_1 - stats_2) / scale) ** 2, dim=1))


def add_vline_in_plot(x, label, color):
    value = x.item()
    plt.axvline(value, color='red')


def clear_cache(c_output_density, c_output_loss, c_FileSavePath):
    if c_output_loss and c_output_density:
        dir_list = ['output_density', 'output_loss', 'output_theta', 'output_mmd', 'output_log']
        for name in dir_list:
            for root, dirs, files in os.walk(c_FileSavePath + os.sep + name):
                for f in files:
                    os.unlink(os.path.join(root, f))
                for d in dirs:
                    shutil.rmtree(os.path.join(root, d))
        print("delete cache success.")


def run_smc_abc(simulator, n_particles, param):
    """Runs sequential monte carlo abc and saves results."""
    # set parameters
    batch = 100000
    n_particles = 2000  # n_particles given by function parameters
    eps_init = 1.0
    eps_last = eps_init / 5.0
    eps_decay = 0.7
    print("start eps: %s" % eps_init)
    print("exit eps: %s" % eps_last)
    round_exc = int(np.log(eps_last / eps_init) / np.log(eps_decay)) + 1
    print("exit round: %.2f" % round_exc)
    ess_min = 0.5
    ModelInfo, FileSavePath = param
    # load observed data
    obs_stats = simulator.sdp_obs
    n_dim = simulator.dim_theta
    all_ps = []
    all_logweights = []
    all_eps = []
    all_nsims = []
    # sample initial population
    ps = torch.zeros(n_particles, n_dim)
    weights = torch.ones(n_particles) / n_particles
    logweights = torch.log(weights)
    eps = eps_init
    iter = 0
    nsims = 0
    prior = simulator.prior
    batch_obs_states = obs_stats.repeat(batch, 1)
    generated_samp = 0
    while True:
        batch_sample = simulator.forward_transform(prior.sample((batch,)))  # batch * dim_theta
        batch_data = simulator.gen_sdp(simulator.gen_s(simulator.gen_data(
            simulator.backward_transform(batch_sample)))).reshape(batch, simulator.dim_s)  # batch * dim_data
        dist = calc_dist_batch(batch_data, batch_obs_states, simulator.scale)  # batch
        new_sample = batch_sample[dist < eps]
        if new_sample.shape[0] == 0:
            nsims += batch
            continue
        else:
            if generated_samp + new_sample.shape[0] >= n_particles:
                sample_part_size = n_particles - generated_samp
                ps[generated_samp:] = new_sample[:sample_part_size]
                nsims += (((dist < eps) == True).nonzero()[sample_part_size - 1]).item()
                break
            else:
                ps[generated_samp:(generated_samp + new_sample.shape[0])] = new_sample
                generated_samp += new_sample.shape[0]
                nsims += batch
                continue
    all_ps.append(ps.clone())
    all_logweights.append(logweights.clone())
    all_eps.append(eps)
    all_nsims.append(nsims)
    print('iteration = {0}, eps = {1:.4}, ess = {2:.4%}, sim_num = {3:}'.format(iter, eps, 1.0, nsims))
    while True:
        # save csv and plot
        plot_df = pd.DataFrame(simulator.backward_transform(ps).cpu())
        plot_df.to_csv(FileSavePath + 'output_abc' + os.sep + 'theta_' + ModelInfo + '_' + str(iter) + '.csv')
        plot_df.columns = simulator.columns if (simulator.columns is not None) else plot_df.columns
        g = sns.pairplot(plot_df, plot_kws=dict(marker="+", s=0.2, linewidth=1))
        if simulator.true_theta is not None:
            true_theta = pd.DataFrame(simulator.true_theta.detach().numpy())
            true_theta.columns = plot_df.columns
            g.data = true_theta
            g.map_offdiag(sns.scatterplot, s=120, marker=".", edgecolor="black")
            g.map_diag(add_vline_in_plot)
        plt.savefig(FileSavePath + 'output_abc' + os.sep + 'theta_' + ModelInfo + '_' + str(iter) + '.jpg', dpi=400)
        plt.close()
        if eps <= eps_last:
            break
        # calculate next round
        iter += 1
        eps *= eps_decay
        # calculate population covariance
        mean = torch.mean(ps, dim=0)
        cov = 1.0 * (ps.t() @ ps / n_particles - torch.outer(mean, mean))
        # print(torch.det(cov))
        # print('sim numbers: %d' % nsims)
        std = torch.linalg.cholesky(cov)
        # perturb particles
        new_ps = torch.zeros_like(ps)
        new_logweights = torch.zeros_like(logweights)
        discrete_sampler = torch.distributions.Categorical(weights)
        normal_sampler = torch.distributions.Normal(0., 1.)
        generated_samp = 0
        while True:
            batch_idx = discrete_sampler.sample((batch,))  # batch
            normal_sample = normal_sampler.sample((n_dim, batch))
            batch_new_ps = ps[batch_idx] + (std @ normal_sample).t()  # batch * dim_theta
            batch_data = simulator.gen_sdp(simulator.gen_s(simulator.gen_data(
                simulator.backward_transform(batch_new_ps)))).reshape(batch, simulator.dim_s)  # batch * dim_data
            dist = calc_dist_batch(batch_data, batch_obs_states, simulator.scale)  # batch
            prop_idx = torch.logical_and(dist < eps, dist < eps)
            new_sample = batch_new_ps[prop_idx]
            if new_sample.shape[0] == 0:
                nsims += batch
                continue
            else:
                if generated_samp + new_sample.shape[0] >= n_particles:
                    sample_part_size = n_particles - generated_samp
                    new_ps[generated_samp:] = new_sample[:sample_part_size]
                    nsims += (prop_idx.nonzero()[sample_part_size - 1]).item()
                    break
                else:
                    new_ps[generated_samp:(generated_samp + new_sample.shape[0])] = new_sample
                    generated_samp += new_sample.shape[0]
                    nsims += batch
                    print("nsims at %d/%d round: %d, generated sample: %d" % (iter, round_exc, nsims, generated_samp))
                    continue
        for i in range(n_particles):
            logkernel = -0.5 * torch.sum(torch.linalg.solve(std, (new_ps[i] - ps).t()) ** 2, dim=0)
            new_logweights[i] = prior.log_prob(simulator.backward_transform(new_ps[i])) - torch.logsumexp(logweights + logkernel, dim=0)
            # new_logweights[i] = float('-inf') if prior.log_prob(simulator.backward_transform(new_ps[i])).item() == float('-inf') else -torch.logsumexp(logweights + logkernel, dim=0)
        ps = new_ps
        logweights = new_logweights - torch.logsumexp(new_logweights, dim=0)
        weights = torch.exp(logweights)
        # calculate effective sample size
        ess = 1.0 / (torch.sum(weights ** 2) * n_particles)
        print('iteration = {0}, eps = {1:.4}, ess = {2:.2%}, sim_num = {3:}'.format(iter, eps, ess, nsims))
        if ess < ess_min:
            # resample particles
            discrete_sampler = torch.distributions.Categorical(weights)
            idx = discrete_sampler.sample((n_particles,))
            ps = ps[idx]
            weights = torch.ones(n_particles) / n_particles
            logweights = torch.log(weights)
        all_ps.append(ps.clone())
        all_logweights.append(logweights.clone())
        all_eps.append(eps)
        all_nsims.append(nsims)
    return all_ps, all_logweights, all_eps, all_nsims


if __name__ == '__main__':
    time_start = time.time()
    torch.set_default_dtype(torch.float32)
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=1, help='gpu_available')  # 0: cpu; 1: cuda:0, 2: cuda:1, ...
    parser.add_argument('--eps', type=float, default=1.0, help='epsilon')  # eps-differentially private value (>0)
    parser.add_argument('--mkstr', type=str, default="smcabc", help='mark_str')
    parser.add_argument('--data', type=int, default=1, help='dataset')  # 0: SIR; 1: Linear Regression; 2: Log-linear Model
    parser.add_argument('--me', type=int, default=1, help='method')  # 0: reject-ABC; 1:SMC-ABC
    parser.add_argument('--dbg1', type=int, default=0, help='debug_flag_1')
    parser.add_argument('--dbg2', type=int, default=50, help='debug_flag_2')
    args = parser.parse_args()
    if args.gpu == 0:
        print('using cpu')
        device = torch.device('cpu')
    else:
        print('using gpu: %d (in 1-4)' % args.gpu)
        device = torch.device("cuda:" + str(args.gpu - 1))
    torch.set_default_device(device)
    mark_str = args.mkstr  # extra str for parallel running
    print("mark str: " + mark_str)
    dataset_arg = ['log-linear', 'linear-reg', 'gandk', 'sir', 'gaussian-mix', 'slcp']
    print("using " + dataset_arg[args.data] + " dataset.")
    simulator = Dataset.Simulator(dataset_arg[args.data], device, torch.get_default_dtype(), normalize=False, eps=args.eps, flag=args.dbg1)
    print("dbg1: %d, dbg2: %d" % (args.dbg1, args.dbg2))
    plt.switch_backend("Agg")
    if os.sep == "/":
        FileSavePath = ""
    else:
        FileSavePath = ""
    print("File Save Path: " + FileSavePath)
    clear_output_density = False
    clear_output_loss = clear_output_density
    debug_flag = False
    print_mean_std = True
    tolerance = 2.0
    n_sample = 50000
    n_particles = 2000
    clear_cache(clear_output_density, clear_output_loss, FileSavePath)
    dim_x = simulator.dim_x
    dim_theta = simulator.dim_theta
    ModelInfo = "Mk+ABC_" + mark_str + "_Da+" + str(args.data) + "_Me+" + str(args.me)
    if args.me == 0:
        print('using Reject-ABC algorithm.')
    elif args.me == 1:
        print('using SMC-ABC algorithm.')
        all_ps, all_logweights, all_eps, all_nsims = run_smc_abc(simulator, n_particles, (ModelInfo, FileSavePath))
        torch.save((all_ps, all_logweights, all_eps, all_nsims), FileSavePath + 'output_abc' + os.sep + 'abc' + ModelInfo + '.pt')
        true_ps = simulator.true_theta.squeeze() if simulator.true_theta is not None else torch.zeros_like(all_ps[0][0])
        for ps, logweights, eps, idx in zip(all_ps, all_logweights, all_eps, range(len(all_ps))):
            if idx != (len(all_ps) - 1):
                continue
            weights = torch.exp(logweights)
            # print estimates with error bars
            means = weights @ ps
            stds = torch.sqrt(weights @ (ps ** 2) - means ** 2)
            print('eps = {0:.2}'.format(eps))
            if print_mean_std:
                for i in range(simulator.dim_theta):
                    print('w{0}: true = {1:.2} \t estimate = {2:.2} +/- {3:.2}'.format(i + 1, true_ps[i].item(), means[i], 2.0 * stds[i]))
            plot_df = pd.DataFrame(simulator.backward_transform(ps).cpu())
            # plot_df.to_csv(FileSavePath + 'output_theta' + os.sep + 'theta_' + ModelInfo + '_' + str(idx) + '.csv')
            plot_df.to_csv(FileSavePath + 'output_abc' + os.sep + 'theta_' + ModelInfo + '.csv')
            plot_df.columns = simulator.columns if (simulator.columns is not None) else plot_df.columns
            g = sns.pairplot(plot_df, plot_kws=dict(marker="+", s=0.2, linewidth=1))
            if simulator.true_theta is not None:
                true_theta = pd.DataFrame(simulator.true_theta.detach().numpy())
                true_theta.columns = plot_df.columns
                g.data = true_theta
                g.map_offdiag(sns.scatterplot, s=120, marker=".", edgecolor="black")
                g.map_diag(add_vline_in_plot)
            plt.savefig(FileSavePath + 'output_abc' + os.sep + 'theta_' + ModelInfo + '.jpg', dpi=400)
            plt.close()
    else:
        raise NotImplementedError

    time_end = time.time()
    print('time cost', time_end - time_start, 's')
