#include "library.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <random>
#include <ctime>
#include <thread>
#include <chrono>

using namespace std;

// SIR simulator, generate one observation
void sir_main(const vector<double> &para, int seed, double *result) {
    const double dt = 1.0;
    const double total_time = 160.0; // 160.0
    const int sim_step = static_cast<int>(total_time / dt) + 1; // 161
    const int N = 1000000;
    const int iter_max = 2000000;
    int state[2] = {N - 1, 1};
    double current_time = 0.0;
    double event_time = 0.0;
    int iter_num = 0;
    srand(time(NULL));
    std::random_device rd;
    std::mt19937 gen(rand() + seed + rd());
    std::uniform_real_distribution<> sampler(0.0, 1.0);
    for (int j = 0; j < sim_step; j++) {
        while (current_time > event_time) {
            double rate[] = {state[0] * para[0] / N * state[1], state[1] * para[1]};
            double total_rate = rate[0] + rate[1];
            if (total_rate == 0 || iter_num >= iter_max) {
                event_time = numeric_limits<double>::infinity();
                break;
            }
            event_time += -log(sampler(gen)) / total_rate;
            double event_idx = sampler(gen);
            iter_num++;
            if (event_idx < rate[0] / total_rate) {
                state[0] -= 1;
                state[1] += 1;
            } else {
                state[1] -= 1;
            }
        }
        current_time += dt;
        if (j % 17 == 0) {
            result[j/17] = state[1];
        }
    }
}

// SIR simulator, generate K observation means
void sir_main_repeat(const vector<double> &para, int seed, double *result) {
    const int repeat_times = 100;
    const double dt = 1.0;
    const double total_time = 160.0; // 160.0
    const int sim_step = static_cast<int>(total_time / dt) + 1; // 161
    const int N = 10000;
    const int iter_max = 2000000;
    int state[2];
    int iter_num;
    int total_state[sim_step][2];
    int all_result[repeat_times][10];
    srand(time(NULL));
    std::random_device rd;
    std::mt19937 gen(rand() + seed + rd());
    std::uniform_real_distribution<> sampler(0.0, 1.0);
    for (int repeat_idx = 0; repeat_idx < repeat_times; repeat_idx++) {
        double current_time = 0.0;
        double event_time = 0.0;
        state[0] = N - 1;
        state[1] = 1;
        iter_num = 0;
        for (int j = 0; j < sim_step; j++) {
            while (current_time > event_time) {
                double rate[] = {state[0] * para[0] / N * state[1], state[1] * para[1]};
                double total_rate = rate[0] + rate[1];
                if (total_rate == 0 || iter_num >= iter_max) {
                    event_time = numeric_limits<double>::infinity();
                    break;
                }
                event_time += -log(sampler(gen)) / total_rate;
                double event_idx = sampler(gen);
                iter_num++;
                if (event_idx < rate[0] / total_rate) {
                    state[0] -= 1;
                    state[1] += 1;
                } else {
                    state[1] -= 1;
                }
            }
            total_state[j][0] = state[0];
            total_state[j][1] = state[1];
            current_time += dt;
        }
        for (int idx = 0; idx < 10; idx++) {
            std::binomial_distribution<int> binomial(1000, double(total_state[idx * 17][1]) / N);
            all_result[repeat_idx][idx] = binomial(gen);
        }
    }
    // calculate mean
    for (int i = 0; i < 10; i++) {
        result[i] = 0;
        for (int j = 0; j < repeat_times; j++) {
            result[i] += all_result[j][i];
        }
    }
}

// subtask for multi thread, generate one observation
void sir_sub_task(std::vector<double> &arr) {
    vector<double> para = {arr[0], arr[1]};
    int seed = int(arr[2]);
    double ret[10];
    sir_main(para, seed, ret);
    for (int i = 0; i < 10; ++i) {
        arr[i] = ret[i];
    }
}

// subtask for multi thread, generate K observation means
void sir_sub_task_repeat(std::vector<double> &arr) {
    vector<double> para = {arr[0], arr[1]};
    int seed = int(arr[2]);
    double ret[10];
    sir_main_repeat(para, seed, ret);
    for (int i = 0; i < 10; ++i) {
        arr[i] = ret[i];
    }
}


#ifdef WIN32
#define EXTERN extern "C" __declspec(dllexport)
#elif __GNUC__
#define EXTERN extern "C"
#endif


EXTERN void sir_multi_thread(double x[]) {
    const int n = 10; // length for each task
    const int s = int(x[0]); // number of threads
    const int k = int(x[1]); // number of tasks
    std::vector<double> input(n * k); // storage all input
    std::vector<std::vector<double>> subArrays(k); // storage all input for each task
    for (int i = 0; i < n * k; ++i) {
        input[i] = x[i + 2];
    }
    for (int i = 0; i < k; ++i) {
        subArrays[i] = std::vector<double>(input.begin() + i * n, input.begin() + (i + 1) * n);
    }
    int subArraysPerThread = k / s;
    int remainingSubArrays = k % s;
    std::vector<std::thread> threads;
    int startIdx = 0;
    for (int i = 0; i < s; ++i) {
        int endIdx = startIdx + subArraysPerThread + (i < remainingSubArrays ? 1 : 0);
        threads.emplace_back([startIdx, endIdx, &subArrays]() {
            for (int j = startIdx; j < endIdx; ++j) {
                sir_sub_task(subArrays[j]);
            }
        });
        startIdx = endIdx;
    }
    for (auto &thread: threads) {
        thread.join();
    }
    int x_idx = 2;
    for (const auto &subArray: subArrays) {
        for (const auto &value: subArray) {
            x[x_idx] = value;
            x_idx++;
        }
    }
}
