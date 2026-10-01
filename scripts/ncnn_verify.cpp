#include "net.h"
#include <cstdio>
#include <fstream>
#include <vector>

static void dump(const ncnn::Mat& m, const char* path) {
    FILE* f = fopen(path, "wb");
    const float* p = m.channel(0);
    size_t total = (size_t)m.w * m.h * m.c;
    fwrite(p, sizeof(float), total, f);
    fclose(f);
}

int main(int argc, char** argv) {
    const char* param = argv[1];
    const char* bin = argv[2];
    int idx = atoi(argv[3]);
    const char* inname = argc > 4 ? argv[4] : "images";
    const char* outname = argc > 5 ? argv[5] : "output0";
    char inpath[256], outpath[256];
    snprintf(inpath, sizeof(inpath), "/home/ubuntu/isef/deploy/ncnn/pose-round3/verify/in_%d.bin", idx);
    snprintf(outpath, sizeof(outpath), "/home/ubuntu/isef/deploy/ncnn/pose-round3/verify/out_%d.bin", idx);

    ncnn::Net net;
    net.opt.use_vulkan_compute = false;
    net.load_param(param);
    net.load_model(bin);

    std::ifstream f(inpath, std::ios::binary);
    std::vector<float> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    ncnn::Mat in(640, 640, 3, buf.data());
    ncnn::Extractor ex = net.create_extractor();
    ex.input(inname, in);
    ncnn::Mat out;
    ex.extract(outname, out);
    printf("%s idx=%d out %dx%dx%d\n", param, idx, out.w, out.h, out.c);
    dump(out, outpath);
    return 0;
}
