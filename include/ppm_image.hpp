/*!
  \file 
  \brief PPM image io
  \author Илья Шошин, 2016
*/

#ifndef __PPM_IMAGE__
#define __PPM_IMAGE__

#include <vector>
#include <string>

class PPMImage
{
public:
	PPMImage();
	~PPMImage();
	PPMImage(int w, int h);
	PPMImage(std::vector<char> data, int w, int h);
	PPMImage(const PPMImage &other);
	PPMImage &operator=(const PPMImage &other);
public:
	/*!
		\exception std::invalid_argument
	*/
	static PPMImage load(const std::string& path);
	static void save(const PPMImage &input, std::string path);
	static PPMImage to_rgba(const PPMImage &input);
	static PPMImage to_rgb(const PPMImage &input);
	int pack_data(unsigned int **packed);
	void unpack_data(unsigned int *packed, int size);
	void clear();
public:
	std::vector<char> pixel;
	int width, height;
};

#endif // __PPM_IMAGE__