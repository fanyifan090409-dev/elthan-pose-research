#include "net.h"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

static void dump(const ncnn::Mat& m, const char* path) {
    ncnn::Mat f32;
    if (m.elemsize == 2u) {
        ncnn::cast_float16_to_float32(m, f32);
    } else {
        f32 = m;
    }
    FILE* f = fopen(path, "wb");
    size_t total = (size_t)f32.w * f32.h * f32.c;
    fwrite(f32.channel(0), sizeof(float), total, f);
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
    const char* inname = argc > 4 ? argv[4] : "images";
    const char* outnames[3];
    if (argc > 7) {
        for (int i = 0; i < 3; i++) outnames[i] = argv[5 + i];
    } else {
        outnames[0] = "/model.23/Concat_output_0";
        outnames[1] = "/model.23/Concat_1_output_0";
        outnames[2] = "/model.23/Concat_2_output_0";
    }
    char inpath[256];
    snprintf(inpath, sizeof(inpath), "/home/ubuntu/isef/deploy/ncnn/pose-round3/verify/in_%d.bin", idx);

    ncnn::Net net;
    net.opt.use_vulkan_compute = false;
    net.load_param(param);
    net.load_model(bin);

    std::ifstream f(inpath, std::ios::binary | std::ios::ate);
    size_t fsize = (size_t)f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<float> buf(fsize / 4);
    f.read((char*)buf.data(), (std::streamsize)fsize);

    ncnn::Mat in(640, 640, 3, buf.data());
    ncnn::Extractor ex = net.create_extractor();
    ex.input(inname, in);

    for (int k = 0; k < 3; k++) {
        std::string blob = find_blob(net, outnames[k]);
        if (blob.empty()) {
            fprintf(stderr, "blob %s not found\n", outnames[k]);
            return 1;
        }
        ncnn::Mat out;
        int ret = ex.extract(blob.c_str(), out);
        printf("%s idx=%d k=%d blob=%s ret=%d out %dx%dx%d elemsize=%zu\n", param, idx, k, blob.c_str(), ret, out.w, out.h, out.c, out.elemsize);
        char outpath[256];
        snprintf(outpath, sizeof(outpath), "/home/ubuntu/isef/deploy/ncnn/pose-round3/verify/out_%d_%d.bin", idx, k);
        dump(out, outpath);
    }
    return 0;
}
