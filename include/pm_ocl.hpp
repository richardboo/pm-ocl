/*!
  \file pm_ocl.hpp
  \brief Параллельная реализация фильтра Перона-Малика
  \author Ilya Shoshin (Galarius), 2016-2017
          State Research Institute of Instrument Engineering
*/

#ifndef __pm_ocl_hpp__
#define __pm_ocl_hpp__

extern "C" {
  #include "pm.h" // img_data, proc_data
}

#include <string>

typedef struct {
	/*!\{*/
	  int platformId; ///< индекс платформы
    int deviceId;   ///< индекс устройства
    bool profile;   ///< включить профилирование?
    std::string filename; ///< имя файла (kernel или биткод)
    bool bitcode;   ///< filename указывает на биткод?
    bool verbose;   ///< подробный вывод
	/*!\}*/
} cl_data;  /*! параметры OpenCL */

/*!
 * Параллельное выполнение фильтра Перона-Малика
 *
 * \param idata - данные изображения
 * \param pdata - параметры фильтра
 * \param cdata - параметры opencl
 * \throws cl::Error
 * \throws std::runtime_error
 * \throws std::invalid_argument
 */
void pm_parallel(img_data *idata, proc_data *pdata, cl_data *cdata);

#endif  /* __pm_ocl_hpp__ */