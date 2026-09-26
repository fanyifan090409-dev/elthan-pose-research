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
    ex.input("images", in);

    const char* checks[] = {
        "/model.0/conv/Conv_output_0",
        "/model.0/act/Mul_output_0",
        "/model.1/conv/Conv_output_0",
        "/model.1/act/Mul_output_0",
        "/model.2/cv1/act/Mul_output_0",
        "/model.2/Split_output_1",
        "/model.2/m.0/cv2/act/Mul_output_0",
        "/model.2/m.0/Add_output_0",
        "/model.2/Concat_output_0",
    };
    for (int k = 0; k < 9; k++) {
        std::string blob = find_blob(net, checks[k]);
        if (blob.empty()) { fprintf(stderr, "blob %s not found\n", checks[k]); continue; }
        ncnn::Mat out;
        int ret = ex.extract(blob.c_str(), out);
        printf("k=%d blob=%s ret=%d out %dx%dx%d\n", k, blob.c_str(), ret, out.w, out.h, out.c);
        char outpath[256];
        snprintf(outpath, sizeof(outpath), "/home/ubuntu/isef/deploy/ncnn/pose-round3/verify/ncnn_diag_%d_%d.bin", idx, k);
        dump(out, outpath);
    }
    return 0;
}
