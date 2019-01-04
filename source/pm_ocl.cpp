/*!
  \file pm_ocl.cpp
  \brief Параллельная реализация фильтра Перона-Малика
  \author Ilya Shoshin (Galarius), 2016-2017
          State Research Institute of Instrument Engineering
*/

#include "pm_ocl.hpp"

#include "ocl_utils.hpp" /* OCLUtils  */

#include <iostream>
#include <iomanip>  /* setprecision */
#include <cmath>    /* ceil */

#define KernelArgumentBits 0
#define KernelArgumentThresh 1
#define KernelArgumentEvalFunc 2
#define KernelArgumentLambda 3
#define KernelArgumentWidth 4
#define KernelArgumentHeight 5
#define KernelArgumentOffsetX 6
#define KernelArgumentOffsetY 7

/*!
* Запуск параллельной фильтрации с помощью OpenCL
*/
void pm_parallel(img_data *idata, proc_data *pdata, int platformId, int deviceId, std::string kernel_file)
{
    cl_uint recommendedPlatformId;
    cl_uint recommendedDeviceId;
    cl_uint deviceIdCount;
    std::vector<cl_platform_id> platformIds;
    std::vector<cl_device_id> deviceIds;
    cl_int error = CL_SUCCESS;
    cl_context context;
    cl_program program;
    cl_kernel kernel;
    cl_command_queue queue;
    /* получить доступные платформы */
    platformIds = OCLUtils::availablePlatforms(&recommendedPlatformId);

    if(platformId < 0 || platformId >= platformIds.size()) {
        platformId = recommendedPlatformId;
    }

    /* получить доступные устройства */
    deviceIds = OCLUtils::availableDevices(
                    platformIds[platformId], &deviceIdCount, &recommendedDeviceId);

    if(deviceId < 0 || deviceId >= deviceIds.size()) {
        deviceId = recommendedDeviceId;
    }

    std::cout << "selected platform: " << platformId << std::endl;
    std::cout << "selected device: " << deviceId << std::endl;
    /* создать контекст */
    const cl_context_properties contextProperties[] = {
        CL_CONTEXT_PLATFORM,
        reinterpret_cast<cl_context_properties>(platformIds[platformId]),
        0, 0
    };
    context = clCreateContext(contextProperties, deviceIdCount, deviceIds.data(), nullptr, nullptr, &error);
    CheckOCLError(error);
    std::cout << "context created" << std::endl;
    /* создать бинарник из кода программы */
    std::cout << "kernel file: " << kernel_file << std::endl;
    program = OCLUtils::createProgram(
                  OCLUtils::loadKernel(kernel_file), context);

    /* скомпилировать программу */
    if(!OCLUtils::buildProgram(program, deviceIdCount, deviceIds.data())) {
        exit(EXIT_FAILURE);
    }

    /* создать ядро */
    kernel = clCreateKernel(program, "pm", &error);
    CheckOCLError(error);
    /* размер глобальной памяти */
    cl_ulong global_size;
    CheckOCLError(clGetDeviceInfo(deviceIds[deviceId], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &global_size, nullptr));

    if(global_size < idata->size * sizeof(uint)) {
        std::cerr << "image size is too large, max available memory size for device " << deviceId << " is " << global_size << std::endl;
        exit(EXIT_FAILURE);
    }

    /* создать хранилище данных изображения (вход-выход) */
    cl_mem bits = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, idata->size * sizeof(uint), idata->bits, &error);
    CheckOCLError(error);
    /* создаем команду */
#ifdef ENABLE_PROFILER
    /* с профилированием */
    queue = clCreateCommandQueue(context, deviceIds[deviceId], CL_QUEUE_PROFILING_ENABLE, &error);
#else
    queue = clCreateCommandQueue(context, deviceIds[deviceId], {0}, &error);
#endif // ENABLE_PROFILER
    CheckOCLError(error);
    /* установить параметры */
    clSetKernelArg(kernel, KernelArgumentBits, sizeof(cl_mem), (void *)&bits);
    clSetKernelArg(kernel, KernelArgumentThresh, sizeof(float), (void *)&pdata->thresh);
    clSetKernelArg(kernel, KernelArgumentEvalFunc, sizeof(float), (void *)&pdata->conduction_func);
    clSetKernelArg(kernel, KernelArgumentLambda, sizeof(float), (void *)&pdata->lambda);
    clSetKernelArg(kernel, KernelArgumentWidth, sizeof(int), (void *)&idata->w);
    clSetKernelArg(kernel, KernelArgumentHeight, sizeof(int), (void *)&idata->h);
    cl_event event;
    /* максимальный размер рабочей группы */
    size_t max_work_group_size;
    CheckOCLError(clGetDeviceInfo(deviceIds[deviceId], CL_DEVICE_MAX_WORK_GROUP_SIZE,
                                  sizeof(size_t), &max_work_group_size, nullptr));
    size_t work_group_x = std::min((size_t)idata->w, max_work_group_size);
    size_t work_group_y = std::min((size_t)idata->h, max_work_group_size);
    std::size_t work_size [3] = { work_group_x, work_group_y, 1 };
    std::cout << "work group size: " << work_size[0] << ", " << work_size[1] << std::endl;
    std::cout << "image size: " << idata->w << ", " << idata->h << std::endl;
    double total_time = 0.0;

    if(idata->w <= max_work_group_size &&
            idata->h <= max_work_group_size) {
        const int zero = 0;
        clSetKernelArg(kernel, KernelArgumentOffsetX, sizeof(int), (void *)&zero);
        clSetKernelArg(kernel, KernelArgumentOffsetY, sizeof(int), (void *)&zero);

        for(int it = 0; it < pdata->iterations; ++it) {
            /* все очередные операции завершены */
            clFinish(queue);
            /* выполнить ядро */
            CheckOCLError(clEnqueueNDRangeKernel(queue, kernel, 2,
                                                 nullptr, work_size,
                                                 nullptr, 0,
                                                 nullptr, &event));
#ifdef ENABLE_PROFILER
            /* получить данные профилирования по времени */
            total_time = OCLUtils::mesuareTimeSec(event);
#endif // ENABLE_PROFILER
        }
    } else {
        int parts_x = ceil(idata->w / (float)max_work_group_size);
        int parts_y = ceil(idata->h / (float)max_work_group_size);
        int offset_x = 0, offset_y = 0;

        for(int it = 0; it < pdata->iterations; ++it) {
            for(int py = 0; py < parts_y; ++py) {
                offset_y = py*work_size[1];
                clSetKernelArg(kernel, KernelArgumentOffsetY, sizeof(int), (void *)&offset_y);

                for(int px = 0; px < parts_x; ++px) {
                    offset_x = px*work_size[0];
                    clSetKernelArg(kernel, KernelArgumentOffsetX, sizeof(int), (void *)&offset_x);
                    clFinish(queue);
                    CheckOCLError(clEnqueueNDRangeKernel(queue, kernel, 2,
                                                         nullptr, work_size, nullptr, 0, nullptr, &event));
#ifdef ENABLE_PROFILER
                    /* получить данные профилирования по времени */
                    total_time += OCLUtils::mesuareTimeSec(event);
#endif // ENABLE_PROFILER
                }
            }
        }
    }

#ifdef ENABLE_PROFILER
    std::cout << "parallel execution time in milliseconds = " << std::fixed
              << std::setprecision(3) << (total_time / 1000000.0) << " ms" << std::endl;
#endif // ENABLE_PROFILER
    /* считать результат */
    CheckOCLError(clEnqueueReadBuffer(queue, bits, CL_TRUE, 0, idata->size * sizeof(uint), idata->bits, 0, nullptr, nullptr));
    /* очистка */
    clReleaseMemObject(bits);
    clReleaseCommandQueue(queue);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseContext(context);
}