// C++ raw ORT benchmark — pure engine latency (random input, no preprocess/postprocess).
// Ceiling reference: shows the maximum speed achievable via ORT C++ API directly.
#include <chrono>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <vector>
#include <algorithm>
#include <string>
#include "onnxruntime_cxx_api.h"

int main(int argc, char* argv[]) {
    const char* model_path = argc > 1 ? argv[1] : "models/rf-detr/rf-detr-base.onnx";
    int n_warmup = 5, n_runs = 30;
    bool use_cuda = argc > 2 && std::string(argv[2]) == "cuda";

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "bench");
    Ort::SessionOptions opts;
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    if (use_cuda) {
        OrtCUDAProviderOptions cuda_opts{};
        cuda_opts.device_id = 0;
        opts.AppendExecutionProvider_CUDA(cuda_opts);
    }

    // cold start includes session creation
    auto t_cold = std::chrono::high_resolution_clock::now();
    Ort::Session session(env, model_path, opts);
    // allocate input: [1,3,560,560] float32, all zeros
    std::vector<float> input_data(1*3*560*560, 0.0f);
    std::vector<int64_t> input_shape = {1, 3, 560, 560};
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(mem, input_data.data(), input_data.size(), input_shape.data(), input_shape.size());
    // first run (cold)
    const char* in_names[]  = {"input"};
    const char* out_names[] = {"logits", "boxes"};
    auto outputs = session.Run(Ort::RunOptions{}, in_names, &input_tensor, 1, out_names, 2);
    double cold_ms = std::chrono::duration<double,std::milli>(std::chrono::high_resolution_clock::now() - t_cold).count();

    // warmup
    for (int i = 0; i < n_warmup; i++) {
        Ort::Value in2 = Ort::Value::CreateTensor<float>(mem, input_data.data(), input_data.size(), input_shape.data(), input_shape.size());
        session.Run(Ort::RunOptions{}, in_names, &in2, 1, out_names, 2);
    }

    // timed runs
    std::vector<double> times;
    times.reserve(n_runs);
    for (int i = 0; i < n_runs; i++) {
        Ort::Value in2 = Ort::Value::CreateTensor<float>(mem, input_data.data(), input_data.size(), input_shape.data(), input_shape.size());
        auto t0 = std::chrono::high_resolution_clock::now();
        session.Run(Ort::RunOptions{}, in_names, &in2, 1, out_names, 2);
        times.push_back(std::chrono::duration<double,std::milli>(std::chrono::high_resolution_clock::now() - t0).count());
    }

    std::sort(times.begin(), times.end());
    double mean = std::accumulate(times.begin(), times.end(), 0.0) / n_runs;
    double p50  = times[n_runs * 50 / 100];
    double p95  = times[n_runs * 95 / 100];
    double p99  = times[std::min((int)(n_runs * 99 / 100), n_runs-1)];

    printf("{\"p50_ms\":%.2f,\"p95_ms\":%.2f,\"p99_ms\":%.2f,\"mean_ms\":%.2f,\"cold_ms\":%.1f,\"throughput_rps\":%.2f,\"n_runs\":%d}\n",
           p50, p95, p99, mean, cold_ms, 1000.0/mean, n_runs);
    return 0;
}
