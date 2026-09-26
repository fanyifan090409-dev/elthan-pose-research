#include "layer.h"
#include "layer/swish.h"
#include <cstdio>

int main() {
    float data[3] = {5.0f, -5.0f, 0.0f};
    ncnn::Mat in(3, 1, 1, data);

    ncnn::ParamDict pd;
    ncnn::Swish sw;
    sw.load_param(pd);
    sw.create_pipeline(ncnn::Option());

    ncnn::Mat out;
    int rc = sw.forward(in, out, ncnn::Option());
    printf("forward rc=%d\n", rc);
    const float* p = out.channel(0);
    printf("swish(5)=%f (expect 4.966513)  swish(-5)=%f (expect -0.033487)  swish(0)=%f\n", p[0], p[1], p[2]);
    return 0;
}
