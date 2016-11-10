/*!
  \file Perona Malic Anisotropic Diffusion
  \author Илья Шошин (ГосНИИП, АПР), 2016

  \note reference: https://people.eecs.berkeley.edu/~malik/papers/MP-aniso.pdf
*/

#include <iostream> /* cout, setprecision, endl */
#include <iomanip>  /* fixed */
#include <cstdlib>  /* exit */
#include <cmath>    /* exp */
#include <ctime>    /* clock_t */

#include "apr_ocl_utils.hpp" /* PPMImage, OCLHelper */

//---------------------------------------------------------------
// Структуры, типы
//---------------------------------------------------------------
typedef unsigned int uint;

/*!
*   \brief Указатель на функцию для вычисления
*          коэффициента проводимости
*/
typedef float (*conduction)(int, float);     

typedef struct
{
    ///@{
    uint* bits;     ///< (упакованные rgba)
    size_t size;    ///< размер bits
    int w;          ///< ширина
    int h;          ///< высота
    ///@}
} img_data; ///< данные изображения

typedef struct
{
    int iterations;         ///< кол-во итераций
    /*!
    * \brief Тип функции для вычисления
    *        коэффициента проводимости
    * \note варианты: 0,1
    */
    int conduction_func;
    /*!
    * \brief Указатель на функцию для вычисления
    *        коэффициента проводимости
    */
    conduction conduction_ptr;
    /*!
    * \brief Пороговое значение для выделения 
    *        контуров в функции проводимости
    */
    float thresh;   
    float lambda;   ///< коэффициент Лапласиана (стабильный = 0.25f)
} proc_data; ///< параметры обработки
//---------------------------------------------------------------
// Прототипы
//---------------------------------------------------------------
char* getArgOption(char**, char**, const char*);
void  print_help();
void  run_parallel(img_data*, proc_data*, int, int, std::string);
int   get_channel(uint , int );
float quadric(int , float );
float exponential(int , float );
int   apply_channel(img_data* , proc_data* , int , int , int );
void  apply(img_data* , proc_data* );
//---------------------------------------------------------------
// Точка входа
//---------------------------------------------------------------
int main (int argc, char * argv[])
{
    int iterations = 16;
    int conduction_function = 1; /* [0, 1] */
    float thresh = 30.0f;
    float lambda = 0.25f;
    int run_mode = 2;
    std::string kernel_file = "kernel.cl"; 
    int platformId = -1; 
    int deviceId = -1;

    char* pinfo_str = getArgOption(argv, argv + argc, "-pi");
    if(pinfo_str) {
        /* получить доступные платформы */
	    apr::OCLHelper::available_platforms(nullptr);
        exit(0);
    }
    char* dinfo_str = getArgOption(argv, argv + argc, "-di");
    if(dinfo_str) {
        /* получить доступplatformIdsные платформы */
        std::vector<cl_platform_id> platformIds = apr::OCLHelper::available_platforms(nullptr);
        int idx = atoi(dinfo_str);
        if(idx >= 0 && idx < platformIds.size()) {
            /* получить доступные устройства */
            std::vector<cl_device_id> deviceIds = apr::OCLHelper::available_devices(
            platformIds[idx], nullptr, nullptr);
        }
        exit(0);
    }

    if(argc < 3) {
        print_help();
        exit(EXIT_FAILURE);
    }
    char* src  = argv[1];
    char* dest = argv[2];
    try {
        char* platform_str = getArgOption(argv, argv + argc, "-p");
        char* device_str = getArgOption(argv, argv + argc, "-d");
        char* kernel_file_str = getArgOption(argv, argv + argc, "-k");
        if(kernel_file_str) kernel_file = std::string(kernel_file_str); 
        if(platform_str) platformId = atoi(platform_str); 
        if(device_str)   deviceId = atoi(device_str);
        char* rmode_str = getArgOption(argv, argv + argc, "-r");
        if(rmode_str) run_mode = atoi(rmode_str);
        if(run_mode < 0 || run_mode > 2) run_mode = 2;
        char* iter_str = getArgOption(argv, argv + argc, "-i");
        if(iter_str) iterations = atoi(iter_str);
        char* conduction_function_str = getArgOption(argv, argv + argc, "-f");
        if(conduction_function_str) conduction_function = atoi(conduction_function_str);
        char* thresh_str = getArgOption(argv, argv + argc, "-t");
        if(thresh_str) thresh = atoi(thresh_str);
    } catch(...) {
        std::cerr << "failed to parse arguments, using defaults..." << std::endl;
    }
    std::cout << "number of iterations: " << iterations << std::endl;
    std::cout << "conduction function (0-quadric, 1-exponential): " 
              << conduction_function << std::endl;
    std::cout << "conduction function threshold for edge enhancement: " 
              << thresh << std::endl;
    std::cout << "run mode: " << run_mode << std::endl; 
    std::cout << "reading input image..." << std::endl;
    /* Загрузка изображения (.ppm) */
    /* .ppm хранит rgb изображения, но алгоритм адаптировван
       для работы с rgba, поэтому выполним конвертацию  */
    apr::PPMImage input_img = apr::PPMImage::toRGBA(apr::PPMImage::load(src));
    /* "Упаковка" rgba каналов в unsigned int */
    unsigned int* packed_data = nullptr;
    size_t packed_size = input_img.packData(&packed_data);
    input_img.clear();
    /* Данные изображения и параметры обработки */
    img_data idata = { packed_data, packed_size, 
                       input_img.width, input_img.height };
    /* Выбор функции для вычисления коэффициента проводимости */
    conduction conduction_ptr = conduction_function ? &exponential : &quadric;
    proc_data pdata = {iterations,conduction_function,conduction_ptr,thresh,lambda};
    /* Отфильтрованное изображение */
    apr::PPMImage ouput_img(idata.w, idata.h);
    //---------------------------------------------------------------------------------
    if(run_mode == 0 || run_mode == 2) {
        std::cout << "processing sequentially..." << std::endl;
        clock_t start = clock();
        apply(&idata, &pdata);  /* Запуск последовательной фильтрации */
        clock_t end = clock();
        double timeSpent = (end-start)/(double)CLOCKS_PER_SEC;
        std::cout << "secuential execution time in milliseconds = " << std::fixed 
                  << std::setprecision(3) << (timeSpent * 1000.0) << " ms" << std::endl;
        std::cout << "saving image..." << std::endl;
        ouput_img.unpackData(idata.bits, packed_size); 
        apr::PPMImage::save(apr::PPMImage::toRGB(ouput_img), "s_" + std::string(dest));
        delete[] packed_data;
        packed_data = nullptr;
    }
    //---------------------------------------------------------------------------------
    if(run_mode == 1 || run_mode == 2) {
        std::cout << "processing in parallel..." << std::endl;
        input_img = apr::PPMImage::toRGBA(apr::PPMImage::load(src));
        packed_size = input_img.packData(&packed_data);
        input_img.clear();
        /* Запуск параллельной фильтрации */
        run_parallel(&idata, &pdata, platformId, deviceId, kernel_file); 
        std::cout << "saving image..." << std::endl;
        ouput_img.unpackData(idata.bits, packed_size);
        apr::PPMImage::save(apr::PPMImage::toRGB(ouput_img), "p_" + std::string(dest));
        delete[] packed_data;
        packed_data = nullptr;
    }
    //---------------------------------------------------------------------------------
    std::cout << "done\n" << std::endl;
    return 0;
}
//---------------------------------------------------------------
// Вспомогательные функции
//---------------------------------------------------------------
/*!
* \brief Получить значение аргумента, следующего за флагом
*
* \code{.c++}
*   char* r_str = getArgOption(argv, argv + argc, "-r");
* \endcode
*/
char* getArgOption(char **begin, char **end, const char* option)
{
    const char* optr = 0;
    char* bptr = 0;
    for (; begin != end; ++begin) {
        optr = option;
        bptr = *begin;
        for (; *bptr == *optr; ++bptr, ++optr) {
            if(*bptr == '\0') {
                if (bptr != *end && ++bptr != *end) {
                    return bptr;
                }
            }
        }
    }
    return 0;
}
/*!
* \brief Краткое руководство к запуску программы
*/
void print_help() {
    std::cout << "./pm source_file_ppm destination_file_ppm [ -pi (shows platform list) \
    -di <platform index> (shows devices list) -r <run mode \
    -p <platform idx>   \
    -d <device idx>     \
    -k <kernel file (default:kernel.cl)> \
    (0-sequentional,1-parallel,2-both {default} )> -i <iterations> \
    -f <conduction function (0-quadric [wide regions over smaller ones], \
    1-exponential [high-contrast edges over low-contrast])> \
    -t <conduction function threshold> ]" << std::endl;
}
//---------------------------------------------------------------
// Параллельная фильтрация
//---------------------------------------------------------------
/*!
* Запуск параллельной фильтрации с помощью OpenCL
*/
void run_parallel(img_data* idata, proc_data* pdata, int platformId, int deviceId, std::string kernel_file)
{
    int recommendedPlatformId;
    int recommendedDeviceId;
    cl_uint deviceIdCount;
    /* получить доступные платформы */
	std::vector<cl_platform_id> platformIds = apr::OCLHelper::available_platforms(&recommendedPlatformId);
    if(platformId < 0 || platformId >= platformIds.size()) {
        platformId = recommendedPlatformId; 
    }
	/* получить доступные устройства */
	std::vector<cl_device_id> deviceIds = apr::OCLHelper::available_devices(
        platformIds[platformId], &deviceIdCount, &recommendedDeviceId);
    if(deviceId < 0 || deviceId >= deviceIds.size()) {
        deviceId = recommendedDeviceId; 
    }
    std::cout << "selected platform: " << platformId << std::endl;
    std::cout << "selected device: " << deviceId << std::endl;
	/* создать контекст */
	const cl_context_properties contextProperties[] =
	{
		CL_CONTEXT_PLATFORM, 
        reinterpret_cast<cl_context_properties>(platformIds[platformId]),
		0, 0
	};
	cl_int error = CL_SUCCESS;
	cl_context context = clCreateContext (contextProperties, deviceIdCount,
		deviceIds.data (), nullptr, nullptr, &error);
	CheckError (error);
	std::cout << "context created" << std::endl;
	/* создать бинарник из кода программы */
    std::cout << "kernel file: " << kernel_file << std::endl;
	cl_program program = apr::OCLHelper::createProgram(
        apr::OCLHelper::loadKernel(kernel_file), context);
    /* скомпилировать программу */
	cl_int err = clBuildProgram (program, deviceIdCount, deviceIds.data (), 
		nullptr, nullptr, nullptr);
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
        std::cout << log;
    }
    /* создать ядро */
	cl_kernel kernel = clCreateKernel (program, "pm", &error);
	CheckError (error);
    /* размер глобальной памяти */
    cl_ulong global_size;
    CheckError(clGetDeviceInfo(deviceIds[deviceId], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &global_size, nullptr));
    if(global_size < idata->size * sizeof(uint)) {
        std::cerr << "image size is too large, max available memory size for device " << deviceId << " is " << global_size << std::endl;
        exit(EXIT_FAILURE);
    }
	/* создать хранилище данных изображения (вход-выход) */
	cl_mem bits = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, idata->size * sizeof(uint), idata->bits, &error);
	CheckError (error);	
	/* создаем команду с профилированием */
	cl_command_queue queue = clCreateCommandQueue (context, deviceIds[deviceId],
		CL_QUEUE_PROFILING_ENABLE, &error);
	CheckError (error);
    /* установить параметры */
    clSetKernelArg (kernel, 0, sizeof (cl_mem), (void *)&bits);
    clSetKernelArg (kernel, 1, sizeof (float), (void *)&pdata->thresh);
    clSetKernelArg (kernel, 2, sizeof (float), (void *)&pdata->conduction_func);
    clSetKernelArg (kernel, 3, sizeof (float), (void *)&pdata->lambda);
    clSetKernelArg (kernel, 4, sizeof (int), (void *)&idata->w);
    clSetKernelArg (kernel, 5, sizeof (int), (void *)&idata->h);
    cl_event event;
    /* максимальный размер рабочей группы */
    size_t max_work_group_size;
    CheckError(clGetDeviceInfo(deviceIds[deviceId], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, nullptr));
    size_t work_group_x = std::min((size_t)idata->w, max_work_group_size);
    size_t work_group_y = std::min((size_t)idata->h, max_work_group_size);
    std::size_t work_size [3] = { work_group_x, work_group_y, 1 };
    std::cout << "work group size: " << work_size[0] << ", " << work_size[1] << std::endl;
    std::cout << "image size: " << idata->w << ", " << idata->h << std::endl;
    if(idata->w <= max_work_group_size &&
       idata->h <= max_work_group_size) {
        const int zero = 0;
        clSetKernelArg (kernel, 6, sizeof (int), (void *)&zero);
        clSetKernelArg (kernel, 7, sizeof (int), (void *)&zero); 
        for(int it = 0; it < pdata->iterations; ++it) {
            /* все очередные операции завершены */
            clFinish(queue);    
            /* выполнить ядро */
            CheckError(clEnqueueNDRangeKernel(queue, kernel, 2, 
                                              nullptr, work_size,
                                              nullptr, 0, 
                                              nullptr, &event));
        }
    } else {
        int partsX = ceil(idata->w / (float)max_work_group_size);
        int partsY = ceil(idata->h / (float)max_work_group_size);
        int offsetX = 0, offsetY = 0;
        for(int it = 0; it < pdata->iterations; ++it) {
            for(int px = 0; px < partsX; ++px) {
                offsetX = px*work_size[0];
                clSetKernelArg (kernel, 6, sizeof (int), (void *)&offsetX);
                for(int py = 0; py < partsY; ++py) {
                    offsetY = py*work_size[1];
                    clSetKernelArg (kernel, 7, sizeof (int), (void *)&offsetY);        
                    clFinish(queue);
                    CheckError(clEnqueueNDRangeKernel(queue, kernel, 2, 
                        nullptr, work_size, nullptr, 0, nullptr, &event));
                }
            }
        }
    }
    /* работы ядра завершена */
    clWaitForEvents(1 , &event);
    /* получить данные профилирования по времени */
    cl_ulong time_start, time_end;
    double total_time;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, nullptr);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, nullptr);
    total_time = time_end - time_start;
    std::cout << "parallel execution time in milliseconds = " << std::fixed 
                  << std::setprecision(3) << (total_time / 1000000.0) << " ms" << std::endl;
    /* считать результат */
    CheckError(clEnqueueReadBuffer (queue, bits, CL_TRUE,
        0, idata->size * sizeof(uint), idata->bits, 0, nullptr, nullptr));
    /* очистка */
    clReleaseMemObject (bits);
	clReleaseCommandQueue (queue);
	clReleaseKernel (kernel);
	clReleaseProgram (program);
	clReleaseContext (context);
}

//---------------------------------------------------------------
// Последовательная фильтрация
//---------------------------------------------------------------

int get_channel(uint rgba, int channel)
{
    switch(channel)
    {
    case 0: return ((rgba >> 16) & 0xff);   // red
    case 1: return ((rgba >> 8)  & 0xff);   // green
    case 2: return (rgba & 0xff);           // blue
    default: return rgba >> 24;             // alpha    
    }
}

float quadric(int norm, float thresh) {
    return 1.0f / (1.0f + norm * norm / (thresh * thresh));
}

float exponential(int norm, float thresh) {
    return exp( - norm * norm / (thresh * thresh));
}

int apply_channel(img_data* idata, proc_data* pdata, int x, int y, int ch)
{
    int p = get_channel(idata->bits[y + x * idata->h], ch);
    int deltaW = get_channel(idata->bits[y + (x-1) * idata->h], ch) - p;
    int deltaE = get_channel(idata->bits[y + (x+1) * idata->h], ch) - p;
    int deltaS = get_channel(idata->bits[y+1 + x * idata->h], ch) - p;
    int deltaN = get_channel(idata->bits[y-1 + x * idata->h], ch) - p;
    float cN = pdata->conduction_ptr(abs(deltaN), pdata->thresh);
    float cS = pdata->conduction_ptr(abs(deltaS), pdata->thresh);
    float cE = pdata->conduction_ptr(abs(deltaE), pdata->thresh);
    float cW = pdata->conduction_ptr(abs(deltaW), pdata->thresh);
    return p + pdata->lambda * (cN * deltaN + cS * deltaS + cE * deltaE + cW * deltaW);
}

void apply(img_data* idata, proc_data* pdata)
{
    for (int it = 0; it < pdata->iterations; ++it) {
        for (int x = 1; x < idata->w-1; ++x) {
            for (int y = 1; y < idata->h-1; ++y) {
                int r = apply_channel(idata,pdata,x,y,0);
                int g = apply_channel(idata,pdata,x,y,1);
                int b = apply_channel(idata,pdata,x,y,2);
                int a = get_channel(idata->bits[y+x*idata->h], 3);
                idata->bits[y+x*idata->h] = ((a & 0xff) << 24) | 
                                            ((r & 0xff) << 16) | 
                                            ((g & 0xff) << 8)  | 
                                            (b & 0xff);
            }
        }
    }
}