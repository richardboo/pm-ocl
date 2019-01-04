/*!
  \file ocl_utils.hpp
  \brief Вспомогательные функции OpenCL
  \author Ilya Shoshin (Galarius), 2016-2017
  		  State Research Institute of Instrument Engineering
*/

#ifndef __OCL_UTILS__
#define __OCL_UTILS__

#ifdef __APPLE__
    #include <OpenCL/opencl.h>
#else
    #include <CL/cl.h>
#endif

#include <string>
#include <vector>
#include <cstdlib>
#include <iostream>

/*!
    \def CheckOCLError
    \brief Проверка на наличие ошибки после выполнения OpenCL функции
    \param error код ошибки, возвращённый OpenCL функцией
*/
#define CheckOCLError(error)            \
    do                                  \
    {                                   \
        if ((error) != CL_SUCCESS) {    \
            std::cerr << "OpenCL call failed with error " << error << std::endl; \
            std::exit (1);              \
        }                               \
    }while(false)                       \

namespace OCLUtils
{
    /*!
        \brief Получить список доступных платформ
        \param[out] recommended_id Рекомендуемая платформа
        \return Список платформ
    */
    std::vector<cl_platform_id> availablePlatforms(cl_uint *recommended_id);
    /*!
        \brief Получить список доступных устройств
        \param[in]  platform_id Платформа (см. availablePlatforms)
        \param[out] device_id_count Количество устройств
        \param[out] recommended_id Рекомендуемое устройство
        \see availablePlatforms
        \return Список устройств
    */
    std::vector<cl_device_id> availableDevices(cl_platform_id platform_id, cl_uint *device_id_count, cl_uint *recommended_id);
    /*!
        \brief Получить имя платформы по её идентификатору
        \param id Идентификатор платформы
        \see availablePlatforms
    */
    std::string platformName(cl_platform_id id);
    /*!
        \brief Получить имя устройства по его идентификатору
        \param id Идентификатор устройства
        \see availableDevices
    */
    std::string deviceName(cl_device_id id);
    /*!
        \brief Загрузка kernel файлв
        \param name Имя kernel файла
        \return Текст программы
    */
    std::string loadKernel(const std::string &name);
    /*!
        \brief Создать программный объект для указанного OpenCL контекста
               и исходного кода kernel файла
        \param source Текст программы
        \param context Open CL контекст
        \return Программный объект
    */
    cl_program createProgram(const std::string &source, cl_context context);
    /*!
        \brief Компиляция и линковка
    	\param program Программный объект
    	\param device_id_count Кол-во устройств для которых будет выполнена компиляция и линковка кода
        \param device_list Указатель на список устройств для которых будет выполнена компиляция и линковка кода
        \return True, если успешное построение
    */
    bool buildProgram(cl_program program,
                      cl_uint device_id_count,
                      const cl_device_id *device_list);

    /*!
        \brief Оценка времени выполнения kernel в сек.
    */
    double mesuareTimeSec(cl_event &event);
}

#endif	// __OCL_UTILS__