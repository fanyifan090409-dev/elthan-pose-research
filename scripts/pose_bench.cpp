#include "net.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// 真机延迟基准：640x640 恒定输入，预热后计时多轮，输出 avg/min/max
// 用法: pose_bench <param> <bin> [inname] [out1 out2 out3] [threads] [gpu]
// 默认 inname=images 输出 out0/out1/out2；gpu 开启 Vulkan（Android 上自动加载系统 libvulkan）
static std::string find_blob(ncnn::Net& net, const char* suffix) {
    const std::vector<ncnn::Blob>& blobs = net.blobs();
    for (size_t i = 0; i < blobs.size(); i++) {
        const std::string& name = blobs[i].name;
        size_t pos = name.rfind(suffix);
        if (pos != std::string::npos && pos + strlen(suffix) == name.size())
            return name;
    }
    return "";
}

static std::vector<double> run_bench(ncnn::Net& net, const char* inname,
                                     const std::vector<std::string>& outs,
                                     int runs) {
    std::vector<double> ms;
    ms.reserve(runs);
    ncnn::Mat in(640, 640, 3);
    in.fill(0.5f);
    for (int i = 0; i < runs; i++) {
        ncnn::Extractor ex = net.create_extractor();
        ex.input(inname, in);
        ncnn::Mat out;
        auto t0 = std::chrono::steady_clock::now();
        for (size_t k = 0; k < outs.size(); k++) ex.extract(outs[k].c_str(), out);
        auto t1 = std::chrono::steady_clock::now();
        ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    return ms;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: pose_bench <param> <bin> [inname] [out1 out2 out3] [threads] [gpu]\n");
        return 1;
    }
    const char* param = argv[1];
    const char* bin = argv[2];
    const char* inname = argc > 3 ? argv[3] : "images";
    std::vector<std::string> outs;
    if (argc > 6) {
        for (int k = 0; k < 3; k++) outs.push_back(argv[4 + k]);
    } else {
        outs = {"out0", "out1", "out2"};
    }
    int threads = argc > 7 ? atoi(argv[7]) : 8;
    bool gpu = false;
    for (int i = 1; i < argc; i++)
        if (strcmp(argv[i], "gpu") == 0) gpu = true;

    ncnn::Net net;
    net.opt.use_vulkan_compute = gpu;
    net.opt.use_fp16_packed = gpu;  // Vulkan 下用 fp16 打包提速
    net.opt.use_fp16_storage = false;
    net.opt.num_threads = threads;

    if (net.load_param(param) != 0 || net.load_model(bin) != 0) {
        fprintf(stderr, "load model failed\n");
        return 1;
    }

    // 输出名按 blob 后缀兜底匹配（pnnx 模型的 out0/out1/out2 就是确切名）
    for (size_t k = 0; k < outs.size(); k++) {
        std::string found = find_blob(net, outs[k].c_str());
        if (!found.empty()) outs[k] = found;
    }

    // 预热：构建并执行一次计算图（Vulkan 管线编译开销不计入计时）
    ncnn::Mat in(640, 640, 3);
    in.fill(0.5f);
    {
        ncnn::Extractor ex = net.create_extractor();
        ex.input(inname, in);
        ncnn::Mat out;
        for (size_t k = 0; k < outs.size(); k++) ex.extract(outs[k].c_str(), out);
        printf("warmup done, out0 %dx%dx%d\n", out.w, out.h, out.c);
    }

    std::vector<double> ms = run_bench(net, inname, outs, 50);
    double sum = 0, mn = 1e9, mx = 0;
    for (double v : ms) { sum += v; if (v < mn) mn = v; if (v > mx) mx = v; }
    printf("%s %s: threads=%d gpu=%d  avg=%.2fms min=%.2fms max=%.2fms\n",
           param, gpu ? "vulkan" : "cpu", threads, gpu, sum / ms.size(), mn, mx);
    return 0;
}
