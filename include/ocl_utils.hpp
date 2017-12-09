/*!
  \file
  \brief Вспомогательные функции OpenCL
  \author Илья Шошин, 2016
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
	std::vector<cl_platform_id> available_platforms(cl_uint *recommended_id);
	/*!
	    \brief Получить список доступных устройств
	    \param[in]  platform_id Платформа (см. available_platforms)
	    \param[out] device_id_count Количество устройств
	    \param[out] recommended_id Рекомендуемое устройство
	    \see available_platforms
	    \return Список устройств
	*/
	std::vector<cl_device_id> available_devices(cl_platform_id platform_id, cl_uint *device_id_count, cl_uint *recommended_id);
	/*!
	    \brief Получить имя платформы по её идентификатору
	    \param id Идентификатор платформы
	    \see available_platforms
	*/
	std::string platform_name(cl_platform_id id);
	/*!
	    \brief Получить имя устройства по его идентификатору
	    \param id Идентификатор устройства
	    \see available_devices
	*/
	std::string device_name(cl_device_id id);
	/*!
	    \brief Загрузка kernel файлв
	    \param name Имя kernel файла
	    \return Текст программы
	*/
	std::string load_kernel(const std::string &name);
	/*!
	    \brief Создать программный объект для указанного OpenCL контекста
	           и исходного кода kernel файла
	    \param source Текст программы
	    \param context Open CL контекст
	    \return Программный объект
	*/
	cl_program create_program(const std::string &source, cl_context context);
	/*!
	    \brief Компиляция и линковка
		\param program Программный объект
		\param device_id_count Кол-во устройств для которых будет выполнена компиляция и линковка кода
	    \param device_list Указатель на список устройств для которых будет выполнена компиляция и линковка кода
	    \return True, если успешное построение
	*/
	bool build_program(cl_program program,
	                   cl_uint device_id_count,
	                   const cl_device_id *device_list);

	/*!
	    \brief Оценка времени выполнения kernel в сек.
	*/
	double mesuare_time_sec(cl_event &event);
}

#endif	// __OCL_UTILS__