/*!
  \file	main.cpp
  \brief  GPU Powered Perona – Malik Anisotropic Filter
  \author Ilya Shoshin (Galarius), 2016-2017
            State Research Institute of Instrument Engineering
*/

#include <iostream> /* cout, endl */
#include <fstream>
#include <iomanip>  /* setprecision, fixed */
#include <cstdlib>  /* exit */
#include <cmath>    /* exp */
#include <ctime>    /* clock_t */

extern "C" {
#include "pm.h"			 /* pm(...)	  */
}
#include "pm_ocl.hpp"
#include "ppm_image.hpp" /* PPMImage  */
#include "ocl_utils.hpp" /* OCLUtils  */

/* Раскомментировать, чтобы включить режим профилирования
   или использовать -D ENABLE_PROFILER в настройке компилятора
*/
// #define ENABLE_PROFILER

//---------------------------------------------------------------
// Прототипы
//---------------------------------------------------------------
char *getArgOption(char **, char **, const char *);
bool isArgOption(char **, char **, const char *);
void printHelp();
//---------------------------------------------------------------
// Точка входа
//---------------------------------------------------------------

int main(int argc, char *argv[])
{
    /* значения по умолчанию */
    int iterations = 16;
    float thresh = 30.0f;
    int conduction_function = 1; /* [0, 1] */
    const float lambda = 0.25f;
    int platformId = -1;
    int deviceId = -1;
    int run_mode = 1;   /*[0,1,2]*/
    std::string kernel_file = "kernel.cl";
    std::string bitcode_file;

    /* считывание аргументов командной строки */
    if(isArgOption(argv, argv + argc, "-h")) {  /* справка */
        printHelp();
        exit(EXIT_SUCCESS); // -> EXIT_SUCCESS
    }

    if(isArgOption(argv, argv + argc, "-pi")) { /* список платформ */
        /* получить доступные платформы */
        OCLUtils::availablePlatforms(nullptr);
        exit(EXIT_SUCCESS); // -> EXIT_SUCCESS
    }

    char *dinfo_str = getArgOption(argv, argv + argc, "-di");   /* список устройств для платформы */

    if(dinfo_str) {
        /* получить доступные платформы */
        std::vector<cl_platform_id> platformIds = OCLUtils::availablePlatforms(nullptr);
        int idx = atoi(dinfo_str);  // индекс выбранной платформы

        if(idx >= 0 && idx < platformIds.size()) {
            /* получить доступные устройства */
            std::vector<cl_device_id> deviceIds = OCLUtils::availableDevices(
                    platformIds[idx], nullptr, nullptr);
        }

        exit(EXIT_SUCCESS); // -> EXIT_SUCCESS
    }

    /* проверить количество аргументов командной строки */
    if(argc < 3) {
        printHelp();
        exit(EXIT_FAILURE);
    }

    try {
        char *iter_str      = getArgOption(argv, argv + argc, "-i");        /* кол-во проходов фильтра */
        char *thresh_str    = getArgOption(argv, argv + argc, "-t");        /* коэффициент чувствительности к границам */
        char *conduction_function_str = getArgOption(argv, argv + argc, "-f");  /* функция для получения коэффициента сглаживания */
        char *platform_str  = getArgOption(argv, argv + argc, "-p");        /* индекс платформы */
        char *device_str    = getArgOption(argv, argv + argc, "-d");        /* индекс устройства */
        char *rmode_str     = getArgOption(argv, argv + argc, "-r");        /* режим запуска [0,1,2] */
        char *kernel_file_str = getArgOption(argv, argv + argc, "-k");      /* файл с ядром программы*/
        char *bitcode_file_str = getArgOption(argv, argv + argc, "-b");      /* файл с бит кодом*/

        if(iter_str) iterations = atoi(iter_str);

        if(thresh_str) thresh = atoi(thresh_str);

        if(conduction_function_str) conduction_function = atoi(conduction_function_str);

        if(platform_str) platformId = atoi(platform_str);

        if(device_str)   deviceId = atoi(device_str);

        if(rmode_str) run_mode = atoi(rmode_str);

        if(run_mode < 0 || run_mode > 2) run_mode = 2;

        if(kernel_file_str) { 
            kernel_file = std::string(kernel_file_str);
            if(kernel_file.empty()) {
                std::cerr << "Error: empty bitcode file.\n";
                exit(EXIT_FAILURE);
            }
            std::ifstream in(kernel_file);
            if(in.fail()) {
                std::cerr << "Error: file " << kernel_file << "does not exist.\n";
                exit(EXIT_FAILURE);
            }
        }
        if(bitcode_file_str) { 
            bitcode_file = std::string(bitcode_file_str);
            if(bitcode_file.empty()) {
                std::cerr << "Error: empty bitcode file.\n";
                exit(EXIT_FAILURE);
            }
            std::ifstream in(bitcode_file);
            if(in.fail()) {
                std::cerr << "Error: file " << bitcode_file << "does not exist.\n";
                exit(EXIT_FAILURE);
            }
        }
    } catch(...) {
        std::cerr << "failed to parse arguments, using defaults..." << std::endl;
    }

    char *src  = argv[argc - 2];   // входящее изображение
    char *dest = argv[argc - 1];   // результат обработки
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
    PPMImage input_img;

    try {
        input_img = PPMImage::toRGBA(PPMImage::load(src));
    } catch(std::invalid_argument e) {
        std::cerr << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    /* "Упаковка" rgba каналов в unsigned int */
    unsigned int *packed_data = nullptr;
    size_t packed_size = input_img.packData(&packed_data);
    input_img.clear();
    /* Данные изображения и параметры обработки */
    img_data idata = { packed_data, packed_size,
                       input_img.width, input_img.height
                     };
    /* Выбор функции для вычисления коэффициента проводимости */
    conduction conduction_ptr = conduction_function ? &pm_exponential : &pm_quadric;
    proc_data pdata = {iterations, conduction_function, conduction_ptr, thresh, lambda};
    /* Отфильтрованное изображение */
    PPMImage ouput_img(idata.w, idata.h);

    //---------------------------------------------------------------------------------
    if(run_mode == 0 || run_mode == 2) {    // Последовательная фильтрация
        std::cout << "processing sequentially..." << std::endl;
#ifdef ENABLE_PROFILER
        clock_t start = clock();
        pm(&idata, &pdata);  /* Запуск последовательной фильтрации */
        clock_t end = clock();
        double timeSpent = (end-start)/(double)CLOCKS_PER_SEC;
        std::cout << "sequential execution time in milliseconds = " << std::fixed
                  << std::setprecision(3) << (timeSpent * 1000.0) << " ms" << std::endl;
#else
        pm(&idata, &pdata);  /* Запуск последовательной фильтрации */
#endif // ENABLE_PROFILER
        std::cout << "saving image..." << std::endl;
        ouput_img.unpackData(idata.bits, packed_size);
        PPMImage::save(PPMImage::toRGB(ouput_img), std::string(dest));
        delete[] packed_data;
        packed_data = nullptr;
    }

    //---------------------------------------------------------------------------------
    if(run_mode == 1 || run_mode == 2) {    // Параллельная фильтрация
        std::cout << "processing in parallel..." << std::endl;
        input_img = PPMImage::toRGBA(PPMImage::load(src));
        packed_size = input_img.packData(&packed_data);
        input_img.clear();
        /* Запуск параллельной фильтрации */
        if(bitcode_file.empty()) {
            pm_parallel_kernel(&idata, &pdata, platformId, deviceId, kernel_file);
        } else {
            pm_parallel_bitcode(&idata, &pdata, platformId, deviceId, bitcode_file);

        }
        std::cout << "saving image..." << std::endl;
        ouput_img.unpackData(idata.bits, packed_size);

        try {
            PPMImage::save(PPMImage::toRGB(ouput_img), std::string(dest));
        } catch(std::invalid_argument e) {
            std::cerr << e.what() << std::endl;
        }
        
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
char *getArgOption(char **begin, char **end, const char *option)
{
    const char *optr = 0;
    char *bptr = 0;

    for(; begin != end; ++begin) {
        optr = option;
        bptr = *begin;

        for(; *bptr == *optr; ++bptr, ++optr) {
            if(*bptr == '\0') {
                if(bptr != *end && ++bptr != *end) {
                    return bptr;
                }
            }
        }
    }

    return 0;
}
/*!
* \brief Указан ли флаг
*
* \code{.c++}
    bool flag = isArgOption(argv, argv + argc, "-f");
* \endcode
*/
bool isArgOption(char **begin, char **end, const char *option)
{
    const char *optr = 0;
    char *bptr = 0;

    for(; begin != end; ++begin) {
        optr = option;
        bptr = *begin;

        for(; *bptr == *optr; ++bptr, ++optr) {
            if(*bptr == '\0') {
                return true;
            }
        }
    }

    return false;
}
/*!
* \brief Краткое руководство к запуску программы
*/
void printHelp()
{
    std::cout << "GPU Powered Perona – Malik Anisotropic Filter" << std::endl <<
              "Ilya Shoshin (Galarius), 2016-2017" << std::endl <<
              "State Research Institute of Instrument Engineering" << std::endl << std::endl <<
              "USAGE" << std::endl <<
              "-----" << std::endl << std::endl <<
              "./pm [-i -t -f -p -d -r -k -b] source_file.ppm destination_file.ppm" << std::endl <<
              "----------------------------------------------------------------" << std::endl <<
              "   -i <iterations>" << std::endl <<
              "   -t <conduction function threshold> ]" << std::endl <<
              "   -f <conduction function (0-quadric [wide regions over smaller ones]," <<
              "1-exponential [high-contrast edges over low-contrast])>"  << std::endl <<
              "   -p <platform idx>"  << std::endl <<
              "   -d <device idx>"  << std::endl <<
              "   -r <run mode (0-sequential, 1-parallel {default}, 2-both )>"  << std::endl <<
              "   -k <kernel file (default:kernel.cl)>" << std::endl <<
              "   -b <bitcode file>" << std::endl << std::endl <<
              "./pm [-pi -di -h]" << std::endl <<
              "-----------------" << std::endl <<
              "   -pi (shows platform list)"  << std::endl <<
              "   -di <platform index> (shows devices list)" << std::endl <<
              "   -h (help)" << std::endl << std::endl <<
              "Example" << std::endl <<
              "-------" << std::endl <<
              "   ./pm -i 16 -t 30 -f 1 images/in.ppm images/out.ppm"<< std::endl;
}