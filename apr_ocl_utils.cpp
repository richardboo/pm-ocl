/*!
  \file OpenCL helper utils
  \author Илья Шошин (ГосНИИП, АПР), 2016
*/

#include "apr_ocl_utils.hpp"

#include <fstream>
#include <iostream>
#include <sstream>

namespace apr
{
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

        PPMImage::PPMImage(const PPMImage& other) :
            width(other.width)
            ,height(other.height)
            ,pixel(other.pixel)
        { }

        PPMImage& PPMImage::operator=(const PPMImage& other)
        {
            if(this != &other)
            {
                width = other.width;
                height = other.height;
                pixel = other.pixel;
            }
            return *this;
        }

        PPMImage PPMImage::load (std::string path)
        {
            std::string header;
            int width, height, maxColor;

            std::ifstream in (path, std::ios::binary);
            in >> header;
            if (header != "P6") {
                exit (1);
            }
            /* Пропустить комментарии */
            while (true) {
                getline (in, header);
                if (header.empty()) {
                    continue;
                }
                if (header [0] != '#') {
                    break;
                }
            }

            std::stringstream prpps(header);
            prpps >> width >> height;
            in >> maxColor;

            if (maxColor != 255) {
                exit (1);
            }
            // Пропустить пока не конец строки
            std::string tmp;
            getline(in, tmp);

            std::vector<char> data(width * height * 3);
            in.read(reinterpret_cast<char*>(data.data ()), data.size ());
            in.close();

            PPMImage img(data, width, height);
            return img;
        }

        void PPMImage::save (const PPMImage& input, std::string path)
        {
            std::ofstream out (path, std::ios::binary);
            out << "P6\n";
            out << input.width << " " << input.height << "\n";
            out << "255\n";
            out.write (input.pixel.data (), input.pixel.size ());
        }

        PPMImage PPMImage::toRGBA (const PPMImage& input)
        {
            PPMImage result(input.width, input.height);
            for (std::size_t i = 0; i < input.pixel.size (); i += 3) {
                result.pixel.push_back (input.pixel [i + 0]);
                result.pixel.push_back (input.pixel [i + 1]);
                result.pixel.push_back (input.pixel [i + 2]);
                result.pixel.push_back (0);
            }

            return result;
        }

        PPMImage PPMImage::toRGB (const PPMImage& input)
        {
            PPMImage result(input.width, input.height);
            for (std::size_t i = 0; i < input.pixel.size (); i += 4) {
                result.pixel.push_back (input.pixel [i + 0]);
                result.pixel.push_back (input.pixel [i + 1]);
                result.pixel.push_back (input.pixel [i + 2]);
            }

            return result;
        }

        int PPMImage::packData(unsigned int** packed)
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

        void PPMImage::unpackData(unsigned int* packed, int size)
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
}

namespace apr
{
    std::vector<cl_platform_id> OCLHelper::available_platforms(int* recommendedId)
    {
        cl_uint platformIdCount = 0;
        clGetPlatformIDs (0, nullptr, &platformIdCount);
        if (platformIdCount == 0) {
            std::cerr << "No OpenCL platform found" << std::endl;
            exit(EXIT_FAILURE);
        } else {
            std::cout << "Found " << platformIdCount << " platform(s)" << std::endl;
        }
        std::vector<cl_platform_id> platformIds (platformIdCount);
        clGetPlatformIDs (platformIdCount, platformIds.data(), nullptr);
        for (cl_uint i = 0; i < platformIdCount; ++i) {
            std::cout << "\t (" << (i) << ") : " << apr::OCLHelper::platformName (platformIds [i]) << std::endl;
        }
        if(recommendedId) *recommendedId = 0;
        return platformIds;
    }

    std::vector<cl_device_id> OCLHelper::available_devices(cl_platform_id platformId, cl_uint* deviceIdCount, int* recommendedId)
    {
        cl_uint devIdCount = 0;
        clGetDeviceIDs (platformId, CL_DEVICE_TYPE_ALL, 0, nullptr,
            &devIdCount);
        if (devIdCount == 0) {
            std::cerr << "No OpenCL devices found" << std::endl;
            exit(EXIT_FAILURE);
        } else {
            std::cout << "Found " << devIdCount << " device(s)" << std::endl;
        }
        std::vector<cl_device_id> deviceIds (devIdCount);
        clGetDeviceIDs (platformId, CL_DEVICE_TYPE_ALL, devIdCount,
            deviceIds.data (), nullptr);
        for (cl_uint i = 0; i < devIdCount; ++i) {
            std::cout << "\t (" << (i) << ") : " << apr::OCLHelper::deviceName (deviceIds [i]) << std::endl;
        }
        if(deviceIdCount) *deviceIdCount = devIdCount;
        if(recommendedId) *recommendedId = devIdCount > 1 ? 1 : 0; 
        return deviceIds;
    }

    std::string OCLHelper::platformName (cl_platform_id id)
    {
        size_t size = 0;
        clGetPlatformInfo (id, CL_PLATFORM_NAME, 0, nullptr, &size);

        std::string result;
        result.resize (size);
        clGetPlatformInfo (id, CL_PLATFORM_NAME, size,
            const_cast<char*> (result.data ()), nullptr);

        return result;
    }

    std::string OCLHelper::deviceName (cl_device_id id)
    {
        size_t size = 0;
        clGetDeviceInfo (id, CL_DEVICE_NAME, 0, nullptr, &size);

        std::string result;
        result.resize (size);
        clGetDeviceInfo (id, CL_DEVICE_NAME, size,
            const_cast<char*> (result.data ()), nullptr);

        return result;
    }

    std::string OCLHelper::loadKernel (std::string name)
    {
        std::ifstream in (name);
        std::string result (
            (std::istreambuf_iterator<char> (in)),
            std::istreambuf_iterator<char> ());
        return result;
    }

    cl_program OCLHelper::createProgram (const std::string& source, cl_context context)
    {
        size_t lengths [1] = { source.size () };
        const char* sources [1] = { source.data () };

        cl_int error = 0;
        cl_program program = clCreateProgramWithSource (context, 1, sources, lengths, &error);
        CheckError (error);

        return program;
    }

    double OCLHelper::mesuareTime(cl_event& event)
    {
        /* работы ядра завершена */
        clWaitForEvents(1 , &event);
        /* получить данные профилирования по времени */
        cl_ulong time_start, time_end;
        double total_time;
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, nullptr);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, nullptr);
        total_time = time_end - time_start;
        return total_time;
    }
}