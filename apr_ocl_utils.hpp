/*!
  \file OpenCL helper utils
  \author Илья Шошин (ГосНИИП, АПР), 2016
*/

#ifndef __APR_OCL_UTILS__
#define __APR_OCL_UTILS__

#ifdef __APPLE__
	#include <OpenCL/opencl.h>
#else
	#include <CL/cl.h>
#endif

#include <string>
#include <vector>
#include <cstdlib>
#include <iostream>

#define CheckError(error)               \
    do                                  \
    {                                   \
        if ((error) != CL_SUCCESS) {    \
            std::cerr << "OpenCL call failed with error " << error << std::endl; \
            std::exit (1);              \
        }                               \
    }while(false)                       \

namespace apr
{
    class OCLHelper
    {
        public:
            OCLHelper();
            ~OCLHelper();

        public:

            static std::vector<cl_platform_id> available_platforms(int* recommendedId);

            static std::vector<cl_device_id> available_devices(cl_platform_id platformId, cl_uint* deviceIdCount, int* recommendedId);

            static std::string platformName (cl_platform_id id);

            static std::string deviceName (cl_device_id id);

            static std::string loadKernel (std::string name);

            static cl_program createProgram (const std::string& source, cl_context context);

            static bool buildProgram(cl_program program, 
                                     cl_uint deviceIdCount, 
                                     const cl_device_id* deviceIds,
                                     int deviceId);

            static double mesuareTime(cl_event& event);
    };
}

#endif