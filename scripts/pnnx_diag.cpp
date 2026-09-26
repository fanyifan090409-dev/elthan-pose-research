#include "net.h"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static void dump(const ncnn::Mat& m, const char* path) {
    FILE* f = fopen(path, "wb");
    size_t total = (size_t)m.w * m.h * m.c;
    fwrite(m.channel(0), sizeof(float), total, f);
    fclose(f);
}

int main(int argc, char** argv) {
    const char* param = argv[1];
    const char* bin = argv[2];
    int idx = atoi(argv[3]);

    ncnn::Net net;
    net.opt.use_vulkan_compute = false;
    net.load_param(param);
    net.load_model(bin);

    char inpath[256];
    snprintf(inpath, sizeof(inpath), "/home/ubuntu/isef/deploy/ncnn/pose-round3/verify/in_%d.bin", idx);
    std::ifstream f(inpath, std::ios::binary);
    std::vector<float> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    ncnn::Mat in(640, 640, 3, buf.data());
    ncnn::Extractor ex = net.create_extractor();
    ex.input("in0", in);

    const char* blobs[] = {"1", "2", "3", "4", "6", "8"};
    for (int k = 0; k < 6; k++) {
        ncnn::Mat out;
        int ret = ex.extract(blobs[k], out);
        printf("k=%d blob=%s ret=%d out %dx%dx%d\n", k, blobs[k], ret, out.w, out.h, out.c);
        char outpath[256];
        snprintf(outpath, sizeof(outpath), "/home/ubuntu/isef/deploy/ncnn/pose-round3/verify/pnnx_diag_%d_%d.bin", idx, k);
        dump(out, outpath);
    }
    return 0;
}
