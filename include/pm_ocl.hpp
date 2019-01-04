/*!
  \file pm_ocl.hpp
  \brief Параллельная реализация фильтра Перона-Малика
  \author Ilya Shoshin (Galarius), 2016-2017
          State Research Institute of Instrument Engineering
*/

#ifndef __pm_ocl_hpp__
#define __pm_ocl_hpp__

extern "C" {
#include "pm.h" /* img_data, proc_data */
}

#include <string>

/*!
* Запуск параллельной фильтрации с помощью OpenCL
*/
void pm_parallel(img_data *idata, proc_data *pdata, int platformId, int deviceId, std::string kernel_file);

#endif  /* __pm_ocl_hpp__ */