/*!
  \file ppm_image.cpp
  \brief PPM image io
  \author Ilya Shoshin (Galarius), 2016-2017
  		  State Research Institute of Instrument Engineering
*/

#include "ppm_image.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

PPMImage::PPMImage() { }
PPMImage::~PPMImage() {}

PPMImage::PPMImage(int w, int h)
    : width(w)
    , height(h)
{ }

PPMImage::PPMImage(std::vector<char> data, int w, int h)
    : pixel(data)
    , width(w)
    , height(h)
{ }

PPMImage::PPMImage(const PPMImage &other) :
    width(other.width)
    , height(other.height)
    , pixel(other.pixel)
{ }

PPMImage &PPMImage::operator=(const PPMImage &other)
{
    if(this != &other) {
        width = other.width;
        height = other.height;
        pixel = other.pixel;
    }

    return *this;
}

PPMImage PPMImage::load(const std::string &path)
{
    std::string header;
    int width, height, maxColor;
    std::ifstream in (path, std::ios::binary);
    in >> header;

    if(header != "P6") {
        throw std::invalid_argument("wrong format");
    }

    /* Пропустить комментарии */
    while(true) {
        getline(in, header);

        if(header.empty()) {
            continue;
        }

        if(header [0] != '#') {
            break;
        }
    }

    std::stringstream prpps(header);
    prpps >> width >> height;
    in >> maxColor;

    if(maxColor != 255) {
        throw std::invalid_argument("wrong format");
    }

    // Пропустить пока не конец строки
    std::string tmp;
    getline(in, tmp);
    std::vector<char> data(width * height * 3);
    in.read(reinterpret_cast<char *>(data.data()), data.size());
    in.close();
    PPMImage img(data, width, height);
    return img;
}

void PPMImage::save(const PPMImage &input, std::string path)
{
    std::ofstream out(path, std::ios::binary);
    out << "P6\n";
    out << input.width << " " << input.height << "\n";
    out << "255\n";
    out.write(input.pixel.data(), input.pixel.size());
    out.close();
}

PPMImage PPMImage::toRGBA(const PPMImage &input)
{
    PPMImage result(input.width, input.height);

    for(std::size_t i = 0; i < input.pixel.size(); i += 3) {
        result.pixel.push_back(input.pixel [i + 0]);
        result.pixel.push_back(input.pixel [i + 1]);
        result.pixel.push_back(input.pixel [i + 2]);
        result.pixel.push_back(0);
    }

    return result;
}

PPMImage PPMImage::toRGB(const PPMImage &input)
{
    PPMImage result(input.width, input.height);

    for(std::size_t i = 0; i < input.pixel.size(); i += 4) {
        result.pixel.push_back(input.pixel [i + 0]);
        result.pixel.push_back(input.pixel [i + 1]);
        result.pixel.push_back(input.pixel [i + 2]);
    }

    return result;
}

int PPMImage::packData(unsigned int **packed)
{
    *packed = new unsigned int[pixel.size() / 4];

    for(int i = 0, j = 0; i < pixel.size(); i+=4, ++j) {
        int r = (int)pixel[i+0];
        int g = (int)pixel[i+1];
        int b = (int)pixel[i+2];
        int a = (int)pixel[i+3];
        (*packed)[j] = ((a & 0xff) << 24) | ((r & 0xff) << 16) | ((g & 0xff) << 8) | (b & 0xff);
    }

    return pixel.size() / 4;
}

void PPMImage::unpackData(unsigned int *packed, int size)
{
    pixel.reserve(size * 4);

    for(int i = 0; i < size; ++i) {
        int rgba = packed[i];
        int r = ((rgba >> 16) & 0xff);   // red
        int g = ((rgba >> 8)  & 0xff);   // green
        int b = (rgba & 0xff);           // blue
        int a = rgba >> 24;              // alpha
        pixel.push_back((char)r);
        pixel.push_back((char)g);
        pixel.push_back((char)b);
        pixel.push_back((char)a);
    }
}

void PPMImage::clear()
{
    pixel.clear();
}