/*!
  \file OpenCL Kernel
  \author Илья Шошин (ГосНИИП, АПР), 2016
*/

int get_channel(uint rgba, int channel)
{
	switch(channel) {
		case 0:
			return ((rgba >> 16) & 0xff);   // red

		case 1:
			return ((rgba >> 8)  & 0xff);   // green

		case 2:
			return (rgba & 0xff);           // blue

		default:
			return rgba >> 24;             // alpha
	}
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
		uint rgba[4];

		for(int ch = 0; ch < 3; ++ch) {
			int p = get_channel(bits[y + x * h], ch);
			int deltaW = get_channel(bits[y + (x-1) * h], ch) - p;
			int deltaE = get_channel(bits[y + (x+1) * h], ch) - p;
			int deltaS = get_channel(bits[y+1 + x * h], ch) - p;
			int deltaN = get_channel(bits[y-1 + x * h], ch) - p;
			float cN = eval_func ? exponential(abs(deltaN), thresh)
			           : quadric(abs(deltaN), thresh);
			float cS = eval_func ? exponential(abs(deltaS), thresh)
			           : quadric(abs(deltaS), thresh);
			float cE = eval_func ? exponential(abs(deltaE), thresh)
			           : quadric(abs(deltaE), thresh);
			float cW = eval_func ? exponential(abs(deltaW), thresh)
			           : quadric(abs(deltaW), thresh);
			rgba[ch] = (uint)(p + lambda * (cN * deltaN + cS * deltaS +
			                                cE * deltaE + cW * deltaW));
		}

		rgba[3] = get_channel(bits[y + x * h], 3);
		bits[y+x*h] = ((rgba[3] & 0xff) << 24) |
		              ((rgba[0] & 0xff) << 16) |
		              ((rgba[1] & 0xff) << 8)  |
		              (rgba[2] & 0xff);
	}
}