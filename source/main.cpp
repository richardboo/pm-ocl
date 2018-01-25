/*!
  \file	main.cpp
  \brief  GPU Powered Perona – Malik Anisotropic Filter
  \author Ilya Shoshin (Galarius), 2016-2017
  		  State Research Institute of Instrument Engineering

  \note reference: https://people.eecs.berkeley.edu/~malik/papers/MP-aniso.pdf
*/

#include <iostream> /* cout, setprecision, endl */
#include <fstream>
#include <iomanip>  /* fixed */
#include <cstdlib>  /* exit */
#include <cmath>    /* exp */
#include <ctime>    /* clock_t */

#include "ppm_image.hpp" /* PPMImage  */
#include "ocl_utils.hpp" /* OCLUtils */

/* Раскомментировать, чтобы включить режим профилирования
   или использовать -D ENABLE_PROFILER в настройке компилятора
*/
// #define ENABLE_PROFILER

#define KernelArgumentBits 0
#define KernelArgumentThresh 1
#define KernelArgumentEvalFunc 2
#define KernelArgumentLambda 3
#define KernelArgumentWidth 4
#define KernelArgumentHeight 5
#define KernelArgumentOffsetX 6
#define KernelArgumentOffsetY 7

//---------------------------------------------------------------
// Структуры, типы
//---------------------------------------------------------------
typedef unsigned int uint;

/*!
*   \brief Указатель на функцию для вычисления
*          коэффициента проводимости
*/
typedef float (*conduction)(int, float);

typedef struct {
	///@{
	uint *bits;     ///< (упакованные rgba)
	size_t size;    ///< размер bits
	int w;          ///< ширина
	int h;          ///< высота
	///@}
} img_data; ///< данные изображения

typedef struct {
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
char *getArgOption(char **, char **, const char *);
bool isArgOption(char **, char **, const char *);
void  printHelp();
void  runParallel(img_data *, proc_data *, int, int, std::string);
int   getChannel(uint, int);
float quadric(int, float);
float exponential(int, float);
int   applyChannel(img_data *, proc_data *, int, int, int);
void  apply(img_data *, proc_data *);
void  report(cl_platform_id, cl_device_id,
             int, int, int, double);
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
	std::string kernel_file = "../kernel/kernel.cl";

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

		if(iter_str) iterations = atoi(iter_str);

		if(thresh_str) thresh = atoi(thresh_str);

		if(conduction_function_str) conduction_function = atoi(conduction_function_str);

		if(platform_str) platformId = atoi(platform_str);

		if(device_str)   deviceId = atoi(device_str);

		if(rmode_str) run_mode = atoi(rmode_str);

		if(run_mode < 0 || run_mode > 2) run_mode = 2;

		if(kernel_file_str) kernel_file = std::string(kernel_file_str);
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
		std::cerr << e.what();
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
	conduction conduction_ptr = conduction_function ? &exponential : &quadric;
	proc_data pdata = {iterations, conduction_function, conduction_ptr, thresh, lambda};
	/* Отфильтрованное изображение */
	PPMImage ouput_img(idata.w, idata.h);

	//---------------------------------------------------------------------------------
	if(run_mode == 0 || run_mode == 2) {    // Последовательная фильтрация
		std::cout << "processing sequentially..." << std::endl;
#ifdef ENABLE_PROFILER
		clock_t start = clock();
		apply(&idata, &pdata);  /* Запуск последовательной фильтрации */
		clock_t end = clock();
		double timeSpent = (end-start)/(double)CLOCKS_PER_SEC;
		std::cout << "sequential execution time in milliseconds = " << std::fixed
		          << std::setprecision(3) << (timeSpent * 1000.0) << " ms" << std::endl;
#else
		apply(&idata, &pdata);  /* Запуск последовательной фильтрации */
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
		runParallel(&idata, &pdata, platformId, deviceId, kernel_file);
		std::cout << "saving image..." << std::endl;
		ouput_img.unpackData(idata.bits, packed_size);
		PPMImage::save(PPMImage::toRGB(ouput_img), std::string(dest));
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
	          "./pm [-i -t -f -p -d -r -k] source_file.ppm destination_file.ppm" << std::endl <<
	          "----------------------------------------------------------------" << std::endl <<
	          "   -i <iterations>" << std::endl <<
	          "   -t <conduction function threshold> ]" << std::endl <<
	          "   -f <conduction function (0-quadric [wide regions over smaller ones]," <<
	          "1-exponential [high-contrast edges over low-contrast])>"  << std::endl <<
	          "   -p <platform idx>"  << std::endl <<
	          "   -d <device idx>"  << std::endl <<
	          "   -r <run mode (0-sequential, 1-parallel {default}, 2-both )>"  << std::endl <<
	          "   -k <kernel file (default:kernel.cl)>" << std::endl << std::endl <<
	          "./pm [-pi -di -h]" << std::endl <<
	          "-----------------" << std::endl <<
	          "   -pi (shows platform list)"  << std::endl <<
	          "   -di <platform index> (shows devices list)" << std::endl <<
	          "   -h (help)" << std::endl << std::endl <<
	          "Example" << std::endl <<
	          "-------" << std::endl <<
	          "   ./pm -i 16 -t 30 -f 1 images/in.ppm images/out.ppm"<< std::endl;
}
//---------------------------------------------------------------
// Параллельная фильтрация
//---------------------------------------------------------------
/*!
* Запуск параллельной фильтрации с помощью OpenCL
*/
void runParallel(img_data *idata, proc_data *pdata, int platformId, int deviceId, std::string kernel_file)
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
	report(platformIds[platformId], deviceIds[deviceId], pdata->iterations, idata->w, idata->h, total_time/1000000.0);
	/* считать результат */
	CheckOCLError(clEnqueueReadBuffer(queue, bits, CL_TRUE, 0, idata->size * sizeof(uint), idata->bits, 0, nullptr, nullptr));
	/* очистка */
	clReleaseMemObject(bits);
	clReleaseCommandQueue(queue);
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseContext(context);
}

void  report(cl_platform_id platformId, cl_device_id deviceId, int iterations, int width, int height, double time)
{
	std::ofstream out("report.md", std::ios::out | std::ios::app);
	out << "|platform & device | " << "iterations | " << "width x height, px | " << "time, ms |" << std::endl;
	out << "|------------------|------------|--------------------|----------|" << std::endl;
	out << " | " << OCLUtils::platformName(platformId) << " " <<
	    OCLUtils::deviceName(deviceId)     << " | " <<
	    iterations << " | " << width << " x " << height << " | " <<
	    std::fixed << std::setprecision(3) << time << " | ";
	out.close();
}

//---------------------------------------------------------------
// Последовательная фильтрация
//---------------------------------------------------------------

int getChannel(uint rgba, int channel)
{
	switch(channel) {
		case 0:
			return ((rgba >> 16) & 0xff);   // red

		case 1:
			return ((rgba >> 8)  & 0xff);   // green

		case 2:
			return (rgba & 0xff);           // blue

		default:
			return rgba >> 24;             // alpha
	}
}

float quadric(int norm, float thresh)
{
	return 1.0f / (1.0f + norm * norm / (thresh * thresh));
}

float exponential(int norm, float thresh)
{
	return exp(- norm * norm / (thresh * thresh));
}

int applyChannel(img_data *idata, proc_data *pdata, int x, int y, int ch)
{
	int p = getChannel(idata->bits[x + y * idata->w], ch);
	int deltaW = getChannel(idata->bits[x + (y-1) * idata->w], ch) - p;
	int deltaE = getChannel(idata->bits[x + (y+1) * idata->w], ch) - p;
	int deltaS = getChannel(idata->bits[x+1 + y * idata->w], ch) - p;
	int deltaN = getChannel(idata->bits[x-1 + y * idata->w], ch) - p;
	float cN = pdata->conduction_ptr(abs(deltaN), pdata->thresh);
	float cS = pdata->conduction_ptr(abs(deltaS), pdata->thresh);
	float cE = pdata->conduction_ptr(abs(deltaE), pdata->thresh);
	float cW = pdata->conduction_ptr(abs(deltaW), pdata->thresh);
	return p + pdata->lambda * (cN * deltaN + cS * deltaS + cE * deltaE + cW * deltaW);
}

void apply(img_data *idata, proc_data *pdata)
{
	std::cout << idata->w << " " << idata->h << "\n";

	for(int it = 0; it < pdata->iterations; ++it) {
		for(int y = 1; y < idata->h-1; ++y) {
			for(int x = 1; x < idata->w-1; ++x) {
				int r = applyChannel(idata, pdata, x, y, 0);
				int g = applyChannel(idata, pdata, x, y, 1);
				int b = applyChannel(idata, pdata, x, y, 2);
				int a = getChannel(idata->bits[x+y*idata->w], 3);
				idata->bits[x+y*idata->w] = ((a & 0xff) << 24) |
				                            ((r & 0xff) << 16) |
				                            ((g & 0xff) << 8)  |
				                            (b & 0xff);
			}
		}
	}
}