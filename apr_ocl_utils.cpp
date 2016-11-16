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

    bool OCLHelper::buildProgram(cl_program program, 
                                cl_uint deviceIdCount, 
                                const cl_device_id* deviceIds,
                                int deviceId)
    {
        cl_int err = clBuildProgram (program, deviceIdCount, deviceIds, nullptr, nullptr, nullptr);
        /* вывод ошибок при наличии */
        if (err == CL_BUILD_PROGRAM_FAILURE) {
            /* размер лога */
            size_t log_size;
            clGetProgramBuildInfo(program, deviceIds[deviceId], CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            /* выделить память под лог */
            char *log = (char *) malloc(log_size);
            /* получить лог */
            clGetProgramBuildInfo(program, deviceIds[deviceId], CL_PROGRAM_BUILD_LOG, log_size, log, nullptr);
            /* распечатать лог */
            std::cerr << log << std::endl;
            return false;
        }
        return true;
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