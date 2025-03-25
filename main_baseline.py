# Code for "Simulation-based Bayesian Inference from Privacy Protected Data"

import torch
import argparse
import os
import shutil
import time
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import qmcpy as qp
import SNPE_lib
import Dataset
from tqdm import tqdm
from sbibm import metrics
from sklearn.neighbors import KernelDensity

def loss_ppe_baseline(batch_data_sample, batch_theta_sample, batch_prior_logprob, qmc_sampling):
    current_batch_size = batch_data_sample.shape[0]
    loss_val = -torch.mean(density_family.log_density_value_at_data(batch_data_sample, batch_theta_sample))
    if r_idx == 0:
        return loss_val
    else:
        inner_theta_sample_size = current_batch_size * apt_atoms
        inner_theta_idx = torch.multinomial(torch.ones(theta_sample.shape[0]), inner_theta_sample_size, replacement=True)
        inner_theta_all = theta_sample[inner_theta_idx]
        inner_theta_prior = prior_log_prob[inner_theta_idx]
        replace_idx = torch.linspace(0, (current_batch_size - 1) * apt_atoms, current_batch_size).type(torch.int32)
        inner_theta_all[replace_idx] = batch_theta_sample
        inner_theta_prior[replace_idx] = batch_prior_logprob
        data_firs_expand = batch_data_sample.view(current_batch_size, 1, -1).expand(current_batch_size, apt_atoms, -1).reshape(current_batch_size * apt_atoms, -1)
        loss_val += torch.mean(torch.logsumexp(
            density_family.log_density_value_at_data(data_firs_expand, inner_theta_all).view(current_batch_size, apt_atoms) -
            inner_theta_prior.view(current_batch_size, apt_atoms), dim=1))
        return loss_val

def loss_ple_baseline(batch_data_sample, batch_theta_sample, batch_prior_logprob, qmc_sampling):
    loss_val = -torch.mean(density_family.log_density_value_at_data(batch_theta_sample, batch_data_sample))
    return loss_val

def clear_cache(clear_flag, c_FileSavePath):
    if clear_flag:
        dir_list = ['output_loss', 'output_theta', 'output_log']
        for name in dir_list:
            for root, dirs, files in os.walk(c_FileSavePath + os.sep + name):
                for f in files:
                    os.unlink(os.path.join(root, f))
                for d in dirs:
                    shutil.rmtree(os.path.join(root, d))
        print("delete cache success.")


def add_vline_in_plot(x, label, color):
    value = x.item()
    plt.axvline(value, color='red')


if __name__ == '__main__':
    # parse parameter
    default_dtype = torch.float32
    torch.set_default_dtype(default_dtype)
    # torch.set_default_dtype(torch.float32)
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=1, help='gpu_available')  # 0: cpu; 1: cuda:0, 2: cuda:1, ...
    parser.add_argument('--qmc', type=int, default=1, help='qmc_available')  # 0: without qmc; 1: use qmc
    parser.add_argument('--eps', type=float, default=1.0, help='epsilon')  # eps-differentially private value (>0)
    parser.add_argument('--mkstr', type=str, default="dbg", help='mark_str')  # log file prefix
    parser.add_argument('--data', type=int, default=1, help='dataset')  # 0: SIR model; 1: Linear Regression; 2: Log-linear model
    parser.add_argument('--ear', type=int, default=20, help='early_stop')  # 0: disable early stop; N: early stop torlarance = N;
    parser.add_argument('--fl', type=int, default=2, help='flow_type')  # 0: mix of gaussian(MOG), 1: masked autoregressive flow(MAF), 2: NSF
    parser.add_argument('--me', type=int, default=0, help='method')  # 0: SPPE; 1: SPLE
    parser.add_argument('--clip', type=float, default=5.0, help='gradient_cut')  # cut gradient value
    parser.add_argument('--iss', type=int, default=32, help='inner_sample_size')  # pseudo sample size
    parser.add_argument('--seed', type=int, default=10000, help='random_seed')  # seed
    parser.add_argument('--dbg1', type=int, default=0, help='debug_flag_1')
    parser.add_argument('--dbg2', type=int, default=50, help='debug_flag_2')
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    print("seed: ", args.seed)
    if args.gpu == 0:
        print('using cpu')
        device = torch.device('cpu')
    else:
        print('using gpu: %d (in 1-4)' % args.gpu)
        device = torch.device("cuda:" + str(args.gpu - 1))
    if args.qmc == 0:
        print("qmc sampling is disabled.")
        qmc_sampling = False
    else:
        print("qmc sampling is enabled.")
        qmc_sampling = True
    privacy_eps = args.eps
    print("privacy epsilon: %.2f" % privacy_eps)
    mark_str = args.mkstr  # extra str for parallel running
    print("mark str: " + mark_str)
    if args.ear == 0:
        early_stop = False
        early_stop_tolarance = 0
        print("disable early stop.")
    else:
        early_stop = True
        early_stop_tolarance = args.ear
        print("enable early stop. torlarance: %d" % early_stop_tolarance)
    if args.fl == 0:
        density_family_type = "MOG"
        print("conditional density type is mix of gaussian (MoG).")
    elif args.fl == 1:
        density_family_type = "MAF"
        print("conditional density type is masked autoregressive flow (MAF).")
    elif args.fl == 2:
        density_family_type = "NSF"
        print("conditional density type is neural spline flow (NSF).")
    else:
        raise NotImplementedError
    if args.me == 0:
        method_type = "PPE"
        loss_func = loss_ppe_baseline
        print("using private posterior estimation (PPE) method.")
    elif args.me == 1:
        method_type = "PLE"
        snl_mcmc_std = [0.10, 0.5, 0.4][args.data]
        snl_mcmc_thin = [4, 4, 10][args.data]
        print("using private likelihood estimation (PLE) method.")
        loss_func = loss_ple_baseline
    else:
        raise NotImplementedError
    if args.clip > 1e-3:
        grad_clip = True
        grad_clip_val = args.clip
        print("using gradient clip at %.2f" % args.clip)
    else:
        grad_clip = False
    inner_sample_size = args.iss
    print("pseudo sample size: %d" % inner_sample_size)
    print("dbg1: %d, dbg2: %d" % (args.dbg1, args.dbg2))
    plt.switch_backend("Agg")  # plt.switch_backend("tkagg")
    if os.sep == "/":
        FileSavePath = ""
    else:
        FileSavePath = ""
    dataset_arg = ['sir', 'linear-reg', 'log-linear']
    print("using " + dataset_arg[args.data] + " dataset.")

    # load dataset
    simulator = Dataset.Simulator(dataset_arg[args.data], device, torch.get_default_dtype(), normalize=False, eps=privacy_eps, flag=args.dbg1)
    dim_theta = simulator.dim_theta
    dim_x = simulator.dim_x
    dim_s = simulator.dim_s
    # figure, model, result save
    load_trained_model = False
    proposal_update = True
    plot_loss_figure_save = False  # plot loss
    plot_theta_figure_save = False  # plot theta density
    model_save = True  # save model after training
    output_log = True  # save log file
    model_compile = True  # compile flow model
    detected_log_file = False
    clear_output_folder = False  # clear history data
    clear_cuda_cache = False
    save_theta_csv = True
    clear_cache(clear_output_folder, FileSavePath)
    # training param
    steps_max = 10000  # max iter steps per round
    print_state = 1000
    figure_dpi = 600
    R = 10  # round numbers
    N = 1000  # samples simulate per round
    eval_theta_sample_size = 2000  # evaluate sample size
    apt_atoms = 10
    valid_rate = 0.05
    N_valid = int(valid_rate * N)  # validation data size
    # constructing Flow
    n_layer = 8  # layer numbers
    n_hidden = np.array([50, 50])  # hidden node numbers
    batch_size = 100
    ModelInfo = "Mk+" + mark_str + "_Da+" + str(args.data) + "_Fl+" + density_family_type + "_Me+" + str(args.me) +\
                "_Qmc+" + str(args.qmc) + "_R+" + str(R) + "_N+" + str(N) + "_Ba+" + str(batch_size) + "_St+" + str(steps_max)
    if detected_log_file:
        assert not os.path.exists(FileSavePath + 'output_log' + os.sep + 'log_' + ModelInfo + '.csv')
    if method_type == "PPE":
        if density_family_type == "MOG":
            density_family_org = SNPE_lib.Cond_Mix_Gauss(dim_x, dim_theta, 8, n_hidden, keep_weight=True)
        elif density_family_type == "MAF":
            density_family_org = SNPE_lib.Cond_MAF(dim_s + dim_theta, dim_theta, n_layer, n_hidden, device, reverse=True,
                                                   batch_norm=False, random_order=False, random_degree=False, residual=False)
        elif density_family_type == "NSF":
            density_family_org = SNPE_lib.Cond_NSF(dim_s + dim_theta, dim_theta, n_layer, n_hidden, device)
    elif method_type == "PLE":
        if density_family_type == "MOG":
            density_family_org = SNPE_lib.Cond_Mix_Gauss(dim_theta, dim_x, 8, n_hidden, keep_weight=True)
        elif density_family_type == "MAF":
            density_family_org = SNPE_lib.Cond_MAF(dim_s + dim_theta, dim_s, n_layer, n_hidden, device, reverse=True,
                                                   batch_norm=False, random_order=False, random_degree=False, residual=False)
        elif density_family_type == "NSF":
            density_family_org = SNPE_lib.Cond_NSF(dim_s + dim_theta, dim_s, n_layer, n_hidden, device)
    if model_compile and os.sep == '/':
        density_family = torch.compile(density_family_org, mode="max-autotune")
        print("using compiled model.")
        enable_model_compile = True
    else:
        density_family = density_family_org
        print("using uncompiled model.")
        enable_model_compile = False
    if device == torch.device('cuda:0') or device == torch.device('cuda:1') or device == torch.device('cuda:2') or device == torch.device('cuda:3'):
        density_family = density_family.to(device)
        torch.cuda.empty_cache()

    # training params
    optimizer = torch.optim.Adam(density_family.parameters(), lr=5e-4, eps=1e-8, weight_decay=1e-4)
    LossInfo = []
    LossInfo_valid = []
    LossInfo_x = 0
    LossInfo_x_list = []
    if output_log:
        output_log_idx = 0
        output_log_variables = {'round': int(), 'mmd': float(), 'nlog': float(), 'medd': float(),
                                'c2st': float(), 'iter': int(), 'mkstr': ''}
        output_log_df = pd.DataFrame(output_log_variables, index=[])
    # get observed Sdp
    sdp_obs = simulator.sdp_obs
    # get parameters prior distribution
    prior = simulator.prior
    proposal = prior
    # store data for sample reuse
    full_theta = torch.tensor([], device=device)
    full_data = torch.tensor([], device=device)
    full_state_dict = []
    mcmc_init_value = simulator.forward_transform(prior.sample((1,)))

    # training
    for r_idx in range(0, R):

        # theta & data sampling
        print("start theta sampling, round = " + str(r_idx))
        with torch.no_grad():
            proposal_sample_size = N
            if r_idx == 0 or (not proposal_update):
                # sampling theta from invariant proposal
                print("sampling theta from invariant proposal.")
                proposal_sample = simulator.forward_transform(proposal.sample((proposal_sample_size,)))
            else:
                if method_type == "PPE":
                    # sampling theta from variant proposal
                    print("sampling theta from posterior estimation from last round.")
                    proposal_sample = density_family.gen_sample(proposal_sample_size, sdp_obs, qmc_flag=False)
                    # resample if theta out of support
                    if not torch.all(prior.log_prob(simulator.backward_transform(proposal_sample)) != float('-inf')):
                        proposal_sample_in_support = proposal_sample[prior.log_prob(simulator.backward_transform(proposal_sample)) != float('-inf')]
                        proposal_out_num = int(proposal_sample_size - proposal_sample_in_support.shape[0])
                        resample_times = 0
                        while True:
                            proposal_sample_extra = density_family.gen_sample(proposal_out_num * 3, sdp_obs, qmc_flag=False)
                            proposal_sample_extra_in_support = proposal_sample_extra[prior.log_prob(simulator.backward_transform(proposal_sample_extra)) != float('-inf')]
                            proposal_sample_in_support = torch.cat((proposal_sample_in_support, proposal_sample_extra_in_support), dim=0)
                            proposal_out_num = int(proposal_sample_size - proposal_sample_in_support.shape[0])
                            resample_times += 1
                            if proposal_out_num <= 0:
                                proposal_sample = proposal_sample_in_support[:proposal_sample_size]
                                break
                            if resample_times == 100:
                                print('proposal sampling error!')
                                break
                        print('resample times: %d, out num: %d' % (resample_times, proposal_out_num))
                        assert torch.all(prior.log_prob(simulator.backward_transform(proposal_sample)) != float('-inf'))
                elif method_type == "PLE":
                    time_start = time.perf_counter()
                    if r_idx != 0:
                        mcmc_init_value = mcmc_snl_cache
                    else:
                        mcmc_log_density = lambda theta: density_family.log_density_value_at_data(
                            theta, sdp_obs.repeat([theta.shape[0], 1])) + prior.log_prob(simulator.backward_transform(theta)) + simulator.backward_logdet(theta)
                        proposal_sample = SNPE_lib.MCMC_MH(mcmc_log_density, mcmc_init_value,
                                                        sample_size=proposal_sample_size, generate_size=int(proposal_sample_size * snl_mcmc_thin),
                                                        cut_size=proposal_sample_size, proposal_std=snl_mcmc_std, seq_sample=True, batch=20)
                    time_end = time.perf_counter()
                    print("proposal_std = %.3f, mcmc generate time = %.3f s" % (snl_mcmc_std, time_end - time_start))
                    mcmc_init_value = proposal_sample[-1].reshape(1, -1)
            theta_sample = proposal_sample
            # shuffle
            theta_sample = theta_sample[torch.randperm(theta_sample.shape[0])]

            # data sampling
            time_start = time.perf_counter()
            data_sample = simulator.gen_sdp(simulator.gen_s(simulator.gen_data(simulator.backward_transform(theta_sample)))).squeeze(1)
            time_end = time.perf_counter()
            print("%d data sampling time cost: %.2fs" % (theta_sample.shape[0], time_end-time_start))

            # sample reuse
            full_theta = torch.cat((full_theta, theta_sample), dim=0)
            full_data = torch.cat((full_data, data_sample), dim=0)
            perm = torch.randperm(full_theta.shape[0])
            theta_sample = full_theta[perm]
            data_sample = full_data[perm]

            # calculate log prior
            prior_log_prob = prior.log_prob(simulator.backward_transform(theta_sample)) + simulator.backward_logdet(theta_sample)
            if plot_theta_figure_save:
                plot_df = pd.DataFrame(simulator.backward_transform(theta_sample).cpu())
                plot_df.columns = simulator.columns if (simulator.columns is not None) else plot_df.columns
                g = sns.pairplot(plot_df, plot_kws=dict(marker="+", s=0.2, linewidth=1))
                if simulator.true_theta is not None:
                    true_theta = pd.DataFrame(simulator.true_theta.detach().numpy())
                    true_theta.columns = plot_df.columns
                    g.data = true_theta
                    g.map_offdiag(sns.scatterplot, s=120, marker=".", edgecolor="black")
                    g.map_diag(add_vline_in_plot)
                plt.savefig(FileSavePath + 'output_theta' + os.sep + 'theta_train_' + ModelInfo + '_' +
                            str(r_idx) + '.jpg', dpi=400)
                plt.close()
                if method_type == "PPE":
                    plot_df = pd.DataFrame(simulator.backward_transform(density_family.gen_sample(N, sdp_obs)).cpu())
                elif method_type == "PLE":
                    if r_idx == 0:
                        plot_df = pd.DataFrame(simulator.backward_transform(theta_sample).cpu())
                    else:
                        plot_df = pd.DataFrame(eval_sample.cpu())
                plot_df.columns = simulator.columns if (simulator.columns is not None) else plot_df.columns
                g = sns.pairplot(plot_df, plot_kws=dict(marker="+", s=0.2, linewidth=1))
                if simulator.true_theta is not None:
                    true_theta = pd.DataFrame(simulator.true_theta.detach().numpy())
                    true_theta.columns = plot_df.columns
                    g.data = true_theta
                    g.map_offdiag(sns.scatterplot, s=120, marker=".", edgecolor="black")
                    g.map_diag(add_vline_in_plot)
                plt.savefig(FileSavePath + 'output_theta' + os.sep + 'theta_valid_' + ModelInfo + '_' +
                            str(r_idx) + '.jpg', dpi=400)
                plt.close()

        # network training
        valid_idx = data_sample.shape[0] - N_valid
        valid_data_sample = data_sample[valid_idx:]
        valid_theta_sample = theta_sample[valid_idx:]
        valid_prior_log_prob = prior_log_prob[valid_idx:]
        valid_loss_best = float('inf')
        valid_loss_best_idx = 0
        training_set = torch.utils.data.TensorDataset(data_sample[:valid_idx], theta_sample[:valid_idx], prior_log_prob[:valid_idx])
        dataset_generator = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=0)
        density_family.train()
        range_generator = tqdm(range(steps_max)) if True else range(steps_max)

        # training loop
        for i in range_generator:
            # training
            for batch_data_sample, batch_theta_sample, batch_prior_logprob in dataset_generator:
                loss = loss_func(batch_data_sample, batch_theta_sample, batch_prior_logprob, qmc_sampling)
                LossInfo.append(loss.detach().cpu().numpy())
                LossInfo_x += (1 / len(dataset_generator))
                LossInfo_x_list.append(LossInfo_x)
                optimizer.zero_grad()  # init gradient
                loss.backward()  # calculate gradient
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(density_family.parameters(), grad_clip_val)
                optimizer.step()  # update model parameters

            # validation
            with torch.no_grad():
                valid_loss = loss_func(valid_data_sample, valid_theta_sample, valid_prior_log_prob, qmc_sampling).detach().cpu()
            if i % 20 == 0 and False:
                print("i: %d, valid loss: %4f" % (i, valid_loss))
                now = datetime.datetime.now()
                print(now.strftime("%Y-%m-%d %H:%M:%S"))
            LossInfo_valid.append(valid_loss)
            if valid_loss < valid_loss_best:
                valid_loss_best = valid_loss
                valid_loss_best_idx = i
            else:
                if (i > (valid_loss_best_idx + early_stop_tolarance)) and early_stop:
                    round_total_steps = i + 1
                    print('round: %d, early stop condition satisfied.' % r_idx)
                    break
            if r_idx != 0 and (i+1) % 10 == 0 and False:
                # clear cache
                if device == torch.device('cuda:0') or device == torch.device('cuda:1') or device == torch.device('cuda:2') or device == torch.device(
                        'cuda:3'):
                    with torch.cuda.device(device):
                        if clear_cuda_cache:
                            torch.cuda.empty_cache()
            if (i+1) % print_state == 0:
                # print info
                print('----------')
                now = datetime.datetime.now()
                print(now.strftime("%Y-%m-%d %H:%M:%S"))
                print('Newest Loss: %.4f' % (LossInfo[-1]))
                print('i: %d / %d, round: %d / %d, mkstr: %s' % ((i+1), steps_max, r_idx, R - 1, mark_str))

        # Evaluate model
        # plot loss
        density_family.eval()
        if plot_loss_figure_save:
            plt.plot(LossInfo_x_list, LossInfo, '.', markersize=2)
            plt.plot([loss_iter for loss_iter in range(len(LossInfo_valid))], LossInfo_valid, '.', markersize=2)
            plt.xlabel("Number of iterations")
            plt.ylabel("Loss")
            plt.legend(['train loss', 'valid loss'])
            plt.tight_layout()
            plt.savefig(FileSavePath + 'output_loss' + os.sep + 'loss_' + ModelInfo + '_' +
                        str(r_idx) + '_' + str(i + 1) + '.jpg', dpi=figure_dpi)
            plt.close()

        # generate evaluate theta sample
        if method_type == "PPE":
            with torch.no_grad():
                eval_sample = density_family.gen_sample(eval_theta_sample_size, sdp_obs)
        elif method_type == "PLE":
            with torch.no_grad():
                mcmc_log_density = lambda theta: density_family.log_density_value_at_data(
                    theta, sdp_obs.repeat([theta.shape[0], 1])) + prior.log_prob(
                    simulator.backward_transform(theta)) + simulator.backward_logdet(theta)
                time_start = time.perf_counter()
                eval_sample = SNPE_lib.MCMC_MH(mcmc_log_density, mcmc_init_value,
                                               sample_size=eval_theta_sample_size,
                                               generate_size=int(eval_theta_sample_size * snl_mcmc_thin),
                                               cut_size=eval_theta_sample_size, proposal_std=snl_mcmc_std,
                                               seq_sample=True, batch=20)
                time_end = time.perf_counter()
                if r_idx == 0:
                    mcmc_init_value = eval_sample[-1].reshape(1, -1)
                # mcmc_snl_cache: subsample N samples from MCMC samples
                mcmc_snl_cache = eval_sample[torch.randint(0, eval_sample.shape[0], (N,))]
            print("(eval) proposal_std = %.3f, mcmc generate time = %.3f s" % (snl_mcmc_std, time_end - time_start))
        eval_sample = simulator.backward_transform(eval_sample)

        # calculate negative log density of true theta
        nlog = torch.tensor([0.])
        if simulator.true_theta is not None:
            kde = KernelDensity(bandwidth="scott", kernel='gaussian').fit(eval_sample.cpu())
            nlog = -kde.score_samples(simulator.true_theta.cpu())

        # calculate median distance
        medd = torch.tensor([0.])
        time_start = time.perf_counter()
        if True:
            medd_data_samp = simulator.gen_sdp(simulator.gen_s(simulator.gen_data(eval_sample))).reshape(
                eval_sample.shape[0], -1)
            medd = torch.nanmedian(torch.norm((medd_data_samp - sdp_obs), dim=1)).cpu()
        time_end = time.perf_counter()
        time_medd = time_end - time_start
        time_start = time.perf_counter()
        c2st = torch.tensor([0.])
        if simulator.reference_theta is not None:
            c2st = metrics.c2st(simulator.reference_theta.cpu(), eval_sample.cpu())
        time_end = time.perf_counter()
        time_c2st = time_end - time_start

        # calculate mmd
        mmd = torch.tensor([0.])
        time_start = time.perf_counter()
        if simulator.reference_theta is not None:
            mmd = metrics.mmd(simulator.reference_theta, eval_sample)
        time_end = time.perf_counter()
        time_mmd = time_end - time_start
        print('medd: %.4f, time: %.2fs, c2st: %.4f, time: %.2fs, mmd: %.4f, time: %.2fs, nlog: %.4f, mkstr: %s' %
              (medd.item(), time_medd, c2st.item(), time_c2st, mmd.item(), time_mmd, nlog.item(), mark_str))

        # save qF theta sample as csv file
        # if save_theta_csv and r_idx == (R-1):
        if save_theta_csv:
            # using eval_sample
            pd.DataFrame(eval_sample.cpu()).to_csv(FileSavePath + 'output_theta' + os.sep + ModelInfo + '_' +
                                                   str(r_idx) + '.csv')

        # clear cache
        if device == torch.device('cuda:0') or device == torch.device('cuda:1') or device == torch.device(
                'cuda:2') or device == torch.device('cuda:3'):
            with torch.cuda.device(device):
                if clear_cuda_cache:
                    torch.cuda.empty_cache()
        density_family.train()

        # proposal update
        if output_log:
            output_log_df.loc[len(output_log_df.index)] = [r_idx + 1, mmd.item(), nlog.item(),
                                                           medd.item(), c2st.item(), round_total_steps, mark_str]
    # save output
    density_family.eval()
    with torch.no_grad():
        if output_log:
            output_log_df.to_csv(FileSavePath + 'output_log' + os.sep + 'log_' + ModelInfo + '.csv')
        if model_save:
            pd.DataFrame(LossInfo).to_csv(FileSavePath + 'output_loss' + os.sep + 'loss_' + ModelInfo + '.csv')
            torch.save(density_family.state_dict(), FileSavePath + 'output_model' + os.sep + ModelInfo + ".pt")

