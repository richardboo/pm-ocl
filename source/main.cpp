/*!
  \file
  \brief  GPU Powered Perona – Malik Anisotropic Filter
  \author Ilya Shoshin (Galarius)
  \copyright (c) 2016, Research Institute of Instrument Engineering
*/

#define __CL_ENABLE_EXCEPTIONS 

#if defined(__APPLE__) || defined(__MACOSX)
    #include "cl.hpp"
#else
    #include <CL/cl.hpp>
#endif

#include <iostream> /* cout, endl */
#include <fstream>  /* fstream */
#include <iomanip>  /* setprecision, fixed */
#include <cstdlib>  /* exit */
#include <cmath>    /* exp */
#include <ctime>    /* clock_t */

extern "C" {
    #include "pm.h"      /* pm(...)	  */
}

#include "pm_ocl.hpp"    /* pm_parallel(...) */
#include "ppm_image.hpp" /* PPMImage  */

#define VERSION "1.0"

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
    
    bool profile = isArgOption(argv, argv + argc, "-g");
    bool verbose = isArgOption(argv, argv + argc, "-v");

    if(isArgOption(argv, argv + argc, "-h")) {  /* справка */
        printHelp();
        exit(EXIT_SUCCESS); // -> EXIT_SUCCESS
    }

    if(isArgOption(argv, argv + argc, "-pi")) { /* список платформ */
        /* получить доступные платформы */
        try
        {
            std::vector<cl::Platform> platforms;
            cl::Platform::get(&platforms);
            if(platforms.size()) {
                std::string pname;
                std::cout << "Platforms: " << pname << std::endl;
                int idx = 0;
                for(auto p : platforms) {
                    p.getInfo(CL_PLATFORM_NAME, &pname);
                    std::cout << "\t" << idx << ". " << pname << std::endl;
                    ++idx;
                }
            } else {
                std::cerr << "No OpenCL platforms were found!\n";
            }
        } catch (cl::Error err) {
            std::cerr << "ERROR: " << err.what() << "(" << err.err() << ")" << std::endl;
        }

        exit(EXIT_SUCCESS); // -> EXIT_SUCCESS
    }

    char *dinfo_str = getArgOption(argv, argv + argc, "-di");   /* список устройств для платформы */

    if(dinfo_str) {
        try
        {
            /* получить доступные платформы */
            std::vector<cl::Platform> platforms;
            cl::Platform::get(&platforms);
            if(platforms.size()) {
                int idx = atoi(dinfo_str);  // индекс выбранной платформы
                if(idx >= 0 && idx < platforms.size()) {
                    /* получить доступные устройства */
                    std::vector<cl::Device> devices;
                    platforms[idx].getDevices(CL_DEVICE_TYPE_ALL, &devices);

                    std::string pname, dname;
                    platforms[idx].getInfo(CL_PLATFORM_NAME, &pname);
                    std::cout << "Platform: " << pname << std::endl;
                    if(devices.size()) {
                        std::cout << "Devices: " << std::endl;
                        int idx = 0;
                        for(auto d : devices) {
                            d.getInfo(CL_DEVICE_NAME, &dname);
                            std::cout << "\t" << idx << ". " << dname << std::endl;
                            ++idx;
                        }
                    } else {
                        std::cerr << "No OpenCL devices were found!\n";
                    }
                }
            }
        } catch (cl::Error err) {
            std::cerr << "ERROR: " << err.what() << "(" << err.err() << ")" << std::endl;
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
        char *kernel_file_str = getArgOption(argv, argv + argc, "-k");      /* файл с ядром программы */
        char *bitcode_file_str = getArgOption(argv, argv + argc, "-b");     /* файл с бит кодом */

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

    if(verbose) {
        std::cout << "number of iterations: " << iterations << std::endl;
        std::cout << "conduction function (0-quadric, 1-exponential): "
                << conduction_function << std::endl;
        std::cout << "conduction function threshold for edge enhancement: "
                << thresh << std::endl;
        std::cout << "run mode: " << run_mode << std::endl;
        std::cout << "reading input image..." << std::endl;
    }

    /* загрузка изображения (.ppm) */
    PPMImage input_img;

    try {
        input_img = PPMImage::toRGB(PPMImage::load(src));
    } catch(std::invalid_argument e) {
        std::cerr << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    /* "упаковка" rgb каналов в unsigned int */
    unsigned int *packed_data = nullptr;
    size_t packed_size = input_img.packData(&packed_data);
    input_img.clear();
    /* данные изображения и параметры обработки */
    img_data idata = { packed_data, packed_size,
                       input_img.width, input_img.height
                     };
    /* выбор функции для вычисления коэффициента проводимости */
    conduction conduction_ptr = conduction_function ? &pm_exponential : &pm_quadric;
    proc_data pdata = {iterations, conduction_function, conduction_ptr, thresh, lambda};
    /* отфильтрованное изображение */
    PPMImage ouput_img(idata.w, idata.h);

    //---------------------------------------------------------------------------------
    // последовательная фильтрация
    //---------------------------------------------------------------------------------
    if(run_mode == 0 || run_mode == 2) {    
        if(verbose) {
           std::cout << "processing sequentially..." << std::endl;
        }
        
        if(profile) {
            clock_t start = clock();
            pm(&idata, &pdata);  /* Запуск последовательной фильтрации */
            clock_t end = clock();
            double timeSpent = (end-start)/(double)CLOCKS_PER_SEC;
            std::cout << "sequential execution time in milliseconds = " << std::fixed
                    << std::setprecision(3) << (timeSpent * 1000.0) << " ms" << std::endl;
        } else {
            pm(&idata, &pdata);  /* Запуск последовательной фильтрации */
        }

        if(verbose) {
            std::cout << "saving image..." << std::endl;
        }

        ouput_img.unpackData(idata.bits, packed_size);

        try
        {
            PPMImage::save(PPMImage::toRGB(ouput_img), std::string(dest));
        } catch(std::invalid_argument e) {
            std::cerr << e.what();
        }

        delete[] packed_data;
        packed_data = nullptr;
    }

    //---------------------------------------------------------------------------------
    // параллельная фильтрация
    //---------------------------------------------------------------------------------
    if(run_mode == 1 || run_mode == 2) {
        if(verbose) {
            std::cout << "processing in parallel..." << std::endl;
        }

        input_img = PPMImage::toRGB(PPMImage::load(src));
        packed_size = input_img.packData(&packed_data);
        input_img.clear();
        
        cl_data cdata = { platformId, deviceId, profile, kernel_file, false, verbose};
        if(!bitcode_file.empty()) {
            cdata.filename = bitcode_file;
            cdata.bitcode = true;
        }

        try
        {
            /* запуск параллельной фильтрации */
            pm_parallel(&idata, &pdata, &cdata);
            
            if(verbose) {
                std::cout << "saving image..." << std::endl;
            }
            
            ouput_img.unpackData(idata.bits, packed_size);
            PPMImage::save(PPMImage::toRGB(ouput_img), std::string(dest));
            
        } catch (cl::Error err) {  
            std::cerr << "ERROR: " << err.what() << "(" << err.err() << ")" << std::endl;
        } catch(std::invalid_argument e) {
            std::cerr << e.what();
        } catch(std::runtime_error e) {
            std::cerr << e.what();
        }
        
        delete[] packed_data;
        packed_data = nullptr;
    }

    if(verbose) {
        std::cout << "done\n" << std::endl;
    }
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
              "Version: " << VERSION << std::endl <<
              "Author: Ilya Shoshin (Galarius)" << std::endl <<
              "Copyright (c) 2016, Research Institute of Instrument Engineering" << std::endl << 
              std::endl <<
              "USAGE" << std::endl <<
              "-----" << std::endl << 
              std::endl <<
              "./pm [-i -t -f -p -d -r -k -b -g -v] source_file.ppm destination_file.ppm" << std::endl <<
              "----------------------------------------------------------------" << std::endl <<
              "   -i <iterations>" << std::endl <<
              "   -t <conduction function threshold> ]" << std::endl <<
              "   -f <conduction function (0-quadric [wide regions over smaller ones]," <<
              "1-exponential [high-contrast edges over low-contrast])>"  << std::endl <<
              "   -p <platform idx>"  << std::endl <<
              "   -d <device idx>"  << std::endl <<
              "   -r <run mode (0-sequential, 1-parallel {default}, 2-both )>"  << std::endl <<
              "   -k <kernel file (default:kernel.cl)>" << std::endl <<
              "   -b <bitcode file>" << std::endl <<
              "   -g - profile" << std::endl <<
              "   -v - verbose" << std::endl << std::endl <<
              "./pm [-pi -di -h]" << std::endl <<
              "-----------------" << std::endl <<
              "   -pi (shows platform list)"  << std::endl <<
              "   -di <platform index> (shows devices list)" << std::endl <<
              "   -h (help)" << std::endl << std::endl <<
              "Examples" << std::endl <<
              "-------" << std::endl <<
              "   ./pm -v -i 16 -t 30 -f 1 in.ppm out.ppm"<< std::endl <<
              "   ./pm -g in.ppm out.ppm"<< std::endl <<
              "   ./pm -k kernel/kernel.cl in.ppm out.ppm"<< std::endl <<
              "   ./pm -b kernel.gpu_64.bc in.ppm out.ppm"<< std::endl;
}