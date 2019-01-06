/*!
  \file
  \brief  OpenCL Kernel
  \author Ilya Shoshin (Galarius)
  \copyright (c) 2016, Research Institute of Instrument Engineering
*/

#define PM_RED(rgb)     (( (rgb) >> 16) & 0xffu)
#define PM_GREEN(rgb)   (( (rgb) >> 8 ) & 0xffu)
#define PM_BLUE(rgb)    (  (rgb)        & 0xffu)
#define PM_RGB(r, g, b) (0xffu << 24) | (( (r) & 0xffu) << 16) | (( (g) & 0xffu) << 8) | ( (b) & 0xffu)

int getChannel(uint rgb, int channel)
{
    switch(channel) {
        case 0:
            return PM_RED(rgb);
        case 1:
            return PM_GREEN(rgb);
        case 2:
            return PM_BLUE(rgb);
    }
    return 0;
}

float quadric(int norm, float thresh)
{
    return 1.0f / (1.0f + norm * norm / (thresh * thresh));
}

float exponential(int norm, float thresh)
{
    return exp(- norm * norm / (thresh * thresh));
}

__kernel void pm(__global uint *bits,
                 __private float thresh,
                 __private float eval_func,
                 __private float lambda,
                 __private int w,
                 __private int h,
                 __private int offsetX,
                 __private int offsetY)
{
    const int x = offsetX + get_global_id(0);
    const int y = offsetY + get_global_id(1);

    if(x < w && y < h) {
        uint rgb[3];

        for(int ch = 0; ch < 3; ++ch) {
            int p = getChannel(bits[x + y * w], ch);
            int deltaW = getChannel(bits[x + (y-1) * w], ch) - p;
            int deltaE = getChannel(bits[x + (y+1) * w], ch) - p;
            int deltaS = getChannel(bits[x+1 + y * w], ch) - p;
            int deltaN = getChannel(bits[x-1 + y * w], ch) - p;
            float cN = eval_func ? exponential(abs(deltaN), thresh)
                       : quadric(abs(deltaN), thresh);
            float cS = eval_func ? exponential(abs(deltaS), thresh)
                       : quadric(abs(deltaS), thresh);
            float cE = eval_func ? exponential(abs(deltaE), thresh)
                       : quadric(abs(deltaE), thresh);
            float cW = eval_func ? exponential(abs(deltaW), thresh)
                       : quadric(abs(deltaW), thresh);
            rgb[ch] = (uint)(p + lambda * (cN * deltaN + cS * deltaS +
                                           cE * deltaE + cW * deltaW));
        }

        bits[x + y * w] = PM_RGB(rgb[0], rgb[1], rgb[2]);
    }
}