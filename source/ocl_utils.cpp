/*!
  \file ocl_utils.cpp
  \brief Вспомогательные функции OpenCL
  \author Ilya Shoshin (Galarius), 2016-2017
  		  State Research Institute of Instrument Engineering
*/

#include "ocl_utils.hpp"

#include <fstream>
#include <iostream>
#include <sstream>

namespace OCLUtils
{
    std::vector<cl_platform_id> availablePlatforms(cl_uint *recommended_id)
    {
        cl_uint platform_id_count = 0;
        clGetPlatformIDs(0, nullptr, &platform_id_count);

        if(!platform_id_count) {
            std::cerr << "No OpenCL platform found!" << std::endl;
            exit(EXIT_FAILURE);
        } else {
            std::cout << "Found " << platform_id_count
                      << " platform(s)" << std::endl;
        }

        std::vector<cl_platform_id> platform_ids(platform_id_count);
        clGetPlatformIDs(platform_id_count, platform_ids.data(), nullptr);

        for(cl_uint i = 0; i < platform_id_count; ++i) {
            std::cout << "\t (" << (i) << ") : "
                      << platformName(platform_ids[i]) << std::endl;
        }

        if(recommended_id)
            *recommended_id = 0;

        return platform_ids;
    }

    std::vector<cl_device_id> availableDevices(cl_platform_id platform_id,
            cl_uint *device_id_count,
            cl_uint *recommended_id)
    {
        cl_uint dev_id_count = 0;
        clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 0, nullptr, &dev_id_count);

        if(!dev_id_count) {
            std::cerr << "No OpenCL devices found" << std::endl;
            exit(EXIT_FAILURE);
        } else {
            std::cout << "Found " << dev_id_count << " device(s)" << std::endl;
        }

        std::vector<cl_device_id> device_list(dev_id_count);
        clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, dev_id_count,
                       device_list.data(), nullptr);

        for(cl_uint i = 0; i < dev_id_count; ++i) {
            std::cout << "\t (" << (i) << ") : " << deviceName(device_list [i]) << std::endl;
        }

        if(device_id_count)
            *device_id_count = dev_id_count;

        if(recommended_id)
            *recommended_id = dev_id_count > 1 ? 1 : 0;

        return device_list;
    }

    std::string platformName(cl_platform_id id)
    {
        std::string result;
        size_t size = 0;
        clGetPlatformInfo(id, CL_PLATFORM_NAME, 0, nullptr, &size);
        result.resize(size);
        clGetPlatformInfo(id, CL_PLATFORM_NAME, size,
                          const_cast<char *>(result.data()), nullptr);
        return result;
    }

    std::string deviceName(cl_device_id id)
    {
        std::string result;
        size_t size = 0;
        clGetDeviceInfo(id, CL_DEVICE_NAME, 0, nullptr, &size);
        result.resize(size);
        clGetDeviceInfo(id, CL_DEVICE_NAME, size,
                        const_cast<char *>(result.data()), nullptr);
        return result;
    }

    std::string loadKernel(const std::string &name)
    {
        std::ifstream in(name);
        std::string result(
            (std::istreambuf_iterator<char> (in)),
            std::istreambuf_iterator<char> ());
        return result;
    }

    cl_program createProgram(const std::string &source, cl_context context)
    {
        size_t lengths [1] = { source.size() };
        const char *sources [1] = { source.data() };
        cl_int error = 0;
        cl_program program = clCreateProgramWithSource(context, 1, sources, lengths, &error);
        CheckOCLError(error);
        return program;
    }

    cl_program createProgramFromBitcode(const std::string &bitcode_path, cl_context context, cl_uint device_id_count, const cl_device_id *device_list) 
    {
        cl_int error = 0;
        
        const unsigned char* binaries[1] = { (unsigned char *)bitcode_path.c_str() };
        const size_t len[1] = { bitcode_path.length() };
        cl_program program = clCreateProgramWithBinary(context, 
            device_id_count, device_list, len, binaries, NULL, &error);
        CheckOCLError(error);

        return program;
    }

    bool buildProgram(cl_program program,
                      cl_uint device_id_count,
                      const cl_device_id *device_list)
    {
        cl_int err = clBuildProgram(program, device_id_count, device_list, nullptr, nullptr, nullptr);

        /* вывод ошибок при наличии */
        if(err == CL_BUILD_PROGRAM_FAILURE) {
            for(cl_uint i = 0; i < device_id_count; ++i) {
                std::cerr << deviceName(device_list[i]) << ":" << std::endl;
                /* размер лога */
                size_t log_size;
                clGetProgramBuildInfo(program, device_list[i], CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
                /* выделить память под лог */
                char *log = (char *) malloc(log_size);
                /* получить лог */
                clGetProgramBuildInfo(program, device_list[i], CL_PROGRAM_BUILD_LOG, log_size, log, nullptr);
                /* распечатать лог */
                std::cerr << log << std::endl;
                free(log);
                return false;
            }
        }

        return true;
    }

    double mesuareTimeSec(cl_event &event)
    {
        /* работы ядра завершена */
        clWaitForEvents(1, &event);
        /* получить данные профилирования по времени */
        cl_ulong time_start, time_end;
        double total_time;
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, nullptr);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, nullptr);
        total_time = time_end - time_start;
        return total_time;
    }
}