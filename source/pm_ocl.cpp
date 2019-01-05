/*!
  \file pm_ocl.cpp
  \brief Параллельная реализация фильтра Перона-Малика
  \author Ilya Shoshin (Galarius), 2016-2017
          State Research Institute of Instrument Engineering
*/

#include "pm_ocl.hpp"

#include <iostream>
#include <fstream>
#include <iomanip>  /* setprecision */
#include <cmath>    /* ceil */

//#define __CL_ENABLE_EXCEPTIONS 
#if defined(__APPLE__) || defined(__MACOSX)
    #include "cl.hpp"
#else
    #include <CL/cl.hpp>
#endif

#define KernelArgumentBits 0
#define KernelArgumentThresh 1
#define KernelArgumentEvalFunc 2
#define KernelArgumentLambda 3
#define KernelArgumentWidth 4
#define KernelArgumentHeight 5
#define KernelArgumentOffsetX 6
#define KernelArgumentOffsetY 7


/*!
    \def HandleCL
    \brief Проверка на наличие ошибки после выполнения OpenCL функции
    \param ret код, возвращённый OpenCL функцией
*/
#define HandleCL(ret)                   \
    do                                  \
    {                                   \
        if ((ret) != CL_SUCCESS) {      \
            std::cerr << "OpenCL call failed with code " << (ret) << std::endl; \
            std::exit (ret);            \
        }                               \
    }while(false)                       \

/*!
* Запуск параллельной фильтрации с помощью OpenCL
*/
static void pm_parallel(img_data *idata, proc_data *pdata, int platformId, int deviceId, const std::string& path, bool bitcode)
{
    cl_int err = CL_SUCCESS;

    /* получить доступные платформы */
    std::vector<cl::Platform> platforms;
    HandleCL(cl::Platform::get(&platforms));
    if(!platforms.size()) {
        std::cerr << "No OpenCL platforms were found!" << std::endl;
        exit(EXIT_FAILURE);
    }
    
    cl::Platform platform;
    if(platformId >= 0 && platformId < platforms.size()) {
        platform = platforms[platformId];
    } else {
        platform = platforms.front();
    }

    /* получить доступные устройства */
    std::vector<cl::Device> devices;
    HandleCL(platform.getDevices(CL_DEVICE_TYPE_ALL, &devices));
    if(!devices.size()) {
        std::cerr << "No OpenCL devices were found!" << std::endl;
        exit(EXIT_FAILURE);
    }

    cl::Device device;
    if(deviceId >= 0 && deviceId < devices.size()) {
        device = devices[deviceId];
    } else {
        device = devices.front();
    }
    std::vector<cl::Device> ds { device };

    std::string pname, dname;
    HandleCL(platform.getInfo(CL_PLATFORM_NAME, &pname));
    HandleCL(device.getInfo(CL_DEVICE_NAME, &dname));
    std::cout << "Selected platform: " << pname << std::endl;
    std::cout << "Selected device: "   << dname << std::endl;

    /* создать контекст */
    cl::Context context(ds, NULL, NULL, NULL, &err);
    HandleCL(err);

    /* создание программы */
    cl::Program program;
    if(bitcode) {
        /* создать бинарник из бит кода */
        std::cout << "bitcode file: " << path << std::endl;
        auto binaries = cl::Program::Binaries {std::make_pair<const void*, ::size_t>(path.c_str(), path.length())};
        std::vector<cl_int> binStatus;
        program = cl::Program(context, ds, binaries, &binStatus, &err);
        for(auto bs : binStatus) {
            HandleCL(bs);
        }
        HandleCL(err);
    } else {
        /* создать бинарник из кода программы */
        /* загрузить  исходный код */
        std::ifstream in(path);
        if(in.fail()) {
            std::cerr << "Failed to load kernel source: " << path << std::endl;
            exit(EXIT_FAILURE);
        }
        std::string source(
            (std::istreambuf_iterator<char> (in)),
             std::istreambuf_iterator<char> ());
        in.close();
        program = cl::Program(context, source, false, &err);
        HandleCL(err);
    }

    /* скомпилировать и слинковать программу */
    cl_int status = program.build(ds, NULL, NULL, NULL);
    if(status == CL_BUILD_PROGRAM_FAILURE) {
        std::string buildLog;
        HandleCL(program.getBuildInfo(device, CL_PROGRAM_BUILD_LOG, &buildLog));
        std::cerr << buildLog << std::endl;
    }
    HandleCL(status);
    
    /* размер глобальной памяти */
    cl_ulong global_size;
    device.getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &global_size);

    if(global_size < idata->size * sizeof(uint)) {
        std::cerr << "image size is too large, max available memory size for device " << deviceId << " is " << global_size << std::endl;
        exit(EXIT_FAILURE);
    }

    /* создать хранилище данных изображения (вход-выход) */
    cl::Buffer bits(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, idata->size * sizeof(uint), idata->bits, &err);
    HandleCL(err);

    /* создаем команду */
    cl::CommandQueue queue;

#ifdef ENABLE_PROFILER
    /* с профилированием */
    queue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);    
#else
    queue = cl::CommandQueue(context, device, 0, &err);
#endif // ENABLE_PROFILER

    HandleCL(err);

    /* создать ядро */
    cl::Kernel kernel(program, "pm", &err);
    auto pmKernel = cl::make_kernel<cl::Buffer&, float, float, float, int, int, int, int>(kernel);
    HandleCL(err);

    /* максимальный размер рабочей группы */
    size_t max_work_group_size;
    HandleCL(device.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &max_work_group_size));
    size_t work_group_x = std::min((size_t)idata->w, max_work_group_size);
    size_t work_group_y = std::min((size_t)idata->h, max_work_group_size);
    std::cout << "work group size: " << work_group_x << ", " << work_group_y << std::endl;
    std::cout << "image size: " << idata->w << ", " << idata->h << std::endl;

    cl::Event event;
    cl::EnqueueArgs enqueueArgs(queue, cl::NDRange(work_group_x, work_group_y));

    double total_time = 0.0;
    if(idata->w <= max_work_group_size && idata->h <= max_work_group_size) {
        
        const int zero = 0;
        HandleCL(kernel.setArg(KernelArgumentOffsetX, sizeof(int), (void *)&zero));
        HandleCL(kernel.setArg(KernelArgumentOffsetY, sizeof(int), (void *)&zero));
        for(int it = 0; it < pdata->iterations; ++it) {
            /* все очередные операции завершены */
            queue.finish();

            #ifdef ENABLE_PROFILER
                HandleCL(kernel.setArg(KernelArgumentBits, sizeof(cl_mem), (void *)&bits));
                HandleCL(kernel.setArg(KernelArgumentThresh, sizeof(float), (void *)&pdata->thresh));
                HandleCL(kernel.setArg(KernelArgumentEvalFunc, sizeof(float), (void *)&pdata->conduction_func));
                HandleCL(kernel.setArg(KernelArgumentLambda, sizeof(float), (void *)&pdata->lambda));
                HandleCL(kernel.setArg(KernelArgumentWidth, sizeof(int), (void *)&idata->w));
                HandleCL(kernel.setArg(KernelArgumentHeight, sizeof(int), (void *)&idata->h));
                HandleCL(kernel.setArg(KernelArgumentWidth, sizeof(int), (void *)&idata->w));
                HandleCL(kernel.setArg(KernelArgumentHeight, sizeof(int), (void *)&idata->h));

                HandleCL(queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(work_group_x, work_group_y), cl::NullRange, NULL, &event));
                /* получить данные профилирования по времени */
                event.wait();
                /* получить данные профилирования по времени */
                cl_ulong time_start, time_end;
                HandleCL(event.getProfilingInfo(CL_PROFILING_COMMAND_START, &time_start));
                HandleCL(event.getProfilingInfo(CL_PROFILING_COMMAND_END, &time_end));
                total_time += (time_end - time_start);
            #else
                /* выполнить ядро */
                pmKernel(enqueueArgs, bits, pdata->thresh, pdata->conduction_func, pdata->lambda, idata->w, idata->h, zero, zero);
            #endif // ENABLE_PROFILER
        }
    } else {
        int parts_x = ceil(idata->w / (float)max_work_group_size);
        int parts_y = ceil(idata->h / (float)max_work_group_size);
        int offset_x = 0, offset_y = 0;

        for(int it = 0; it < pdata->iterations; ++it) {
            for(int py = 0; py < parts_y; ++py) {
                offset_y = py*work_group_y;
                for(int px = 0; px < parts_x; ++px) {
                    offset_x = px*work_group_x;
                    /* все очередные операции завершены */
                    queue.finish();
                    
                    #ifdef ENABLE_PROFILER
                        HandleCL(kernel.setArg(KernelArgumentBits, sizeof(cl_mem), (void *)&bits));
                        HandleCL(kernel.setArg(KernelArgumentThresh, sizeof(float), (void *)&pdata->thresh));
                        HandleCL(kernel.setArg(KernelArgumentEvalFunc, sizeof(float), (void *)&pdata->conduction_func));
                        HandleCL(kernel.setArg(KernelArgumentLambda, sizeof(float), (void *)&pdata->lambda));
                        HandleCL(kernel.setArg(KernelArgumentWidth, sizeof(int), (void *)&idata->w));
                        HandleCL(kernel.setArg(KernelArgumentHeight, sizeof(int), (void *)&idata->h));
                        HandleCL(kernel.setArg(KernelArgumentWidth, sizeof(int), (void *)&idata->w));
                        HandleCL(kernel.setArg(KernelArgumentHeight, sizeof(int), (void *)&idata->h));
                        HandleCL(kernel.setArg(KernelArgumentOffsetX, sizeof(int), (void *)&offset_x));
                        HandleCL(kernel.setArg(KernelArgumentOffsetY, sizeof(int), (void *)&offset_y));

                        HandleCL(queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(work_group_x, work_group_y), cl::NullRange, NULL, &event));
                        /* получить данные профилирования по времени */
                        event.wait();
                        /* получить данные профилирования по времени */
                        cl_ulong time_start, time_end;
                        HandleCL(event.getProfilingInfo(CL_PROFILING_COMMAND_START, &time_start));
                        HandleCL(event.getProfilingInfo(CL_PROFILING_COMMAND_END, &time_end));
                        total_time += (time_end - time_start);
                    #else
                    /* выполнить ядро */
                        pmKernel(enqueueArgs, bits, pdata->thresh, pdata->conduction_func, pdata->lambda, idata->w, idata->h, offset_x, offset_y);
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
    HandleCL(queue.enqueueReadBuffer(bits, CL_TRUE, 0, idata->size * sizeof(uint), idata->bits, nullptr, nullptr));
}

void pm_parallel_kernel(img_data *idata, proc_data *pdata, int platformId, int deviceId, const std::string& kernel_file)
{
    pm_parallel(idata, pdata, platformId, deviceId, kernel_file, false);
}

void pm_parallel_bitcode(img_data *idata, proc_data *pdata, int platformId, int deviceId, const std::string& bitcodePath)
{
    pm_parallel(idata, pdata, platformId, deviceId, bitcodePath, true);
}