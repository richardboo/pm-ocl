/*!
  \file PPM image
  \author Илья Шошин (ГосНИИП, АПР), 2016
*/

#ifndef __APR_PPM_IMAGE__
#define __APR_PPM_IMAGE__

#include <vector>
#include <string>

namespace apr
{
    class PPMImage
    {
        public:
            PPMImage();
            ~PPMImage();
            PPMImage(int w, int h);
            PPMImage(std::vector<char> data, int w, int h);
            PPMImage(const PPMImage& other);
            PPMImage& operator=(const PPMImage& other); 
        public:
            /*!
                \exception std::invalid_argument
            */
            static PPMImage load(std::string path);
            static void save(const PPMImage& input, std::string path);
            static PPMImage toRGBA (const PPMImage& input);
            static PPMImage toRGB (const PPMImage& input);
            int packData(unsigned int** packed);
            void unpackData(unsigned int* packed, int size);
            void clear();
        public:
            std::vector<char> pixel;
	        int width, height;
    };
}

#endif /* __APR_PPM_IMAGE__ */