/*!
  \file pm.h
  \brief Последовательная реализация фильтра Перона-Малика
  \author Ilya Shoshin (Galarius), 2016-2017
          State Research Institute of Instrument Engineering
*/

#ifndef __pm_h__
#define __pm_h__

typedef unsigned int uint;
typedef unsigned long ulong;

/*!
*   \brief Указатель на функцию для вычисления
*          коэффициента проводимости
*/
typedef float (*conduction)(int, float);

typedef struct {
	/*!\{*/
	uint *bits; /*!< (упакованные rgba) */
	ulong size;  /*!< размер bits        */
	int w;      /*!< ширина             */
	int h;      /*!< высота             */
	/*!\}*/
} img_data; /*!< данные изображения     */

typedef struct {
	int iterations;         /*!< кол-во итераций */
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
	float lambda;   /*!< коэффициент Лапласиана (стабильный = 0.25f) */
} proc_data; /*!< параметры обработки */

/*!
 * \brief Последовательная реализация фильтра Перона-Малика
 * \note reference: https://people.eecs.berkeley.edu/~malik/papers/MP-aniso.pdf
 * \see img_data
 * \see proc_data
*/
void pm(img_data *idata, proc_data *pdata);

/*!
 * \brief Функции для вычисления коэффициента проводимости
 * \note reference: https://people.eecs.berkeley.edu/~malik/papers/MP-aniso.pdf
 * \{
 */
float pm_quadric(int norm, float thresh);
float pm_exponential(int norm, float thresh);
/*!\}*/


#endif  /* __pm_h__ */