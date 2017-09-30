/*!
  \file Perona Malic Anisotropic Diffusion
  \author Илья Шошин (ГосНИИП, АПР), 2016

  \note reference: https://people.eecs.berkeley.edu/~malik/papers/MP-aniso.pdf
  \todo Сделать поддержку обработки изображений, где ширина != высоте.
  \todo Исправить считывание параметров на windows.
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
void  print_help();
void  run_parallel(img_data *, proc_data *, int, int, std::string);
int   get_channel(uint, int);
float quadric(int, float);
float exponential(int, float);
int   apply_channel(img_data *, proc_data *, int, int, int);
void  apply(img_data *, proc_data *);
void  report(cl_platform_id, cl_device_id,
             int, int, int, double);
//---------------------------------------------------------------
// Экспериментальные функции
//---------------------------------------------------------------
void  binarization(img_data *, proc_data *);
void  edges(img_data *, proc_data *);
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
		print_help();
		exit(EXIT_SUCCESS); // -> EXIT_SUCCESS
	}

	if(isArgOption(argv, argv + argc, "-pi")) { /* список платформ */
		/* получить доступные платформы */
		OCLUtils::available_platforms(nullptr);
		exit(EXIT_SUCCESS); // -> EXIT_SUCCESS
	}

	char *dinfo_str = getArgOption(argv, argv + argc, "-di");   /* список устройств для платформы */

	if(dinfo_str) {
		/* получить доступplatformIdsные платформы */
		std::vector<cl_platform_id> platformIds = OCLUtils::available_platforms(nullptr);
		int idx = atoi(dinfo_str);  // индекс выбранной платформы

		if(idx >= 0 && idx < platformIds.size()) {
			/* получить доступные устройства */
			std::vector<cl_device_id> deviceIds = OCLUtils::available_devices(
			        platformIds[idx], nullptr, nullptr);
		}

		exit(EXIT_SUCCESS); // -> EXIT_SUCCESS
	}

	/* проверить количество аргументов командной строки */
	if(argc < 3) {
		print_help();
		exit(EXIT_FAILURE);
	}

	char *src  = argv[1];   // входящее изображение
	char *dest = argv[2];   // результат обработки

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
		input_img = PPMImage::to_rgba(PPMImage::load(src));
	} catch(std::invalid_argument e) {
		std::cerr << e.what();
		exit(EXIT_FAILURE);
	}

	/* "Упаковка" rgba каналов в unsigned int */
	unsigned int *packed_data = nullptr;
	size_t packed_size = input_img.pack_data(&packed_data);
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
		//ouput_img.unpack_data(idata.bits, packed_size);
		//PPMImage::save(PPMImage::to_rgb(ouput_img), std::string(dest));
		//binarization(&idata, &pdata);  /* бинаризация */
		//edges(&idata, &pdata);
		clock_t end = clock();
		double timeSpent = (end-start)/(double)CLOCKS_PER_SEC;
		std::cout << "secuential execution time in milliseconds = " << std::fixed
		          << std::setprecision(3) << (timeSpent * 1000.0) << " ms" << std::endl;
#else
		apply(&idata, &pdata);  /* Запуск последовательной фильтрации */
#endif // ENABLE_PROFILER
		std::cout << "saving image..." << std::endl;
		//PPMImage edges_img(idata.w, idata.h);
		//edges_img.unpack_data(idata.bits, packed_size);
		ouput_img.unpack_data(idata.bits, packed_size);
		PPMImage::save(PPMImage::to_rgb(ouput_img), std::string(dest));
		//PPMImage::save(PPMImage::to_rgb(edges_img), std::string("images/edges.ppm"));
		delete[] packed_data;
		packed_data = nullptr;
	}

	//---------------------------------------------------------------------------------
	if(run_mode == 1 || run_mode == 2) {    // Параллельная фильтрация
		std::cout << "processing in parallel..." << std::endl;
		input_img = PPMImage::to_rgba(PPMImage::load(src));
		packed_size = input_img.pack_data(&packed_data);
		input_img.clear();
		/* Запуск параллельной фильтрации */
		run_parallel(&idata, &pdata, platformId, deviceId, kernel_file);
		std::cout << "saving image..." << std::endl;
		ouput_img.unpack_data(idata.bits, packed_size);
		PPMImage::save(PPMImage::to_rgb(ouput_img), std::string(dest));
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
void print_help()
{
	std::cout << "./pm source_file.ppm destination_file.ppm [-i -t -f -p -d -r -k]" << std::endl <<
	          "----------------------------------------------------------------" << std::endl <<
	          "   -i <iterations>" << std::endl <<
	          "   -t <conduction function threshold> ]" << std::endl <<
	          "   -f <conduction function (0-quadric [wide regions over smaller ones]," <<
	          "1-exponential [high-contrast edges over low-contrast])>"  << std::endl <<
	          "   -p <platform idx>"  << std::endl <<
	          "   -d <device idx>"  << std::endl <<
	          "   -r <run mode (0-sequentional,1-parallel {default},2-both )>"  << std::endl <<
	          "   -k <kernel file (default:kernel.cl)>" << std::endl <<
	          "./pm [-pi -di -h]" << std::endl <<
	          "-----------------" << std::endl <<
	          "   -pi (shows platform list)"  << std::endl <<
	          "   -di <platform index> (shows devices list)" << std::endl <<
	          "   -h (help)" << std::endl <<
	          "Example:" << std::endl <<
	          "   ./pm images/in.ppm images/out.ppm -i 16 -t 30 -f 1"<< std::endl;
}
//---------------------------------------------------------------
// Параллельная фильтрация
//---------------------------------------------------------------
/*!
* Запуск параллельной фильтрации с помощью OpenCL
*/
void run_parallel(img_data *idata, proc_data *pdata, int platformId, int deviceId, std::string kernel_file)
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
	platformIds = OCLUtils::available_platforms(&recommendedPlatformId);

	if(platformId < 0 || platformId >= platformIds.size()) {
		platformId = recommendedPlatformId;
	}

	/* получить доступные устройства */
	deviceIds = OCLUtils::available_devices(
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
	program = OCLUtils::create_program(
	              OCLUtils::load_kernel(kernel_file), context);

	/* скомпилировать программу */
	if(!OCLUtils::build_program(program, deviceIdCount, deviceIds.data())) {
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
	clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&bits);
	clSetKernelArg(kernel, 1, sizeof(float), (void *)&pdata->thresh);
	clSetKernelArg(kernel, 2, sizeof(float), (void *)&pdata->conduction_func);
	clSetKernelArg(kernel, 3, sizeof(float), (void *)&pdata->lambda);
	clSetKernelArg(kernel, 4, sizeof(int), (void *)&idata->w);
	clSetKernelArg(kernel, 5, sizeof(int), (void *)&idata->h);
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
		clSetKernelArg(kernel, 6, sizeof(int), (void *)&zero);
		clSetKernelArg(kernel, 7, sizeof(int), (void *)&zero);

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
			total_time = OCLUtils::mesuare_time_sec(event);
#endif // ENABLE_PROFILER
		}
	} else {
		int partsX = ceil(idata->w / (float)max_work_group_size);
		int partsY = ceil(idata->h / (float)max_work_group_size);
		int offsetX = 0, offsetY = 0;

		for(int it = 0; it < pdata->iterations; ++it) {
			for(int px = 0; px < partsX; ++px) {
				offsetX = px*work_size[0];
				clSetKernelArg(kernel, 6, sizeof(int), (void *)&offsetX);

				for(int py = 0; py < partsY; ++py) {
					offsetY = py*work_size[1];
					clSetKernelArg(kernel, 7, sizeof(int), (void *)&offsetY);
					clFinish(queue);
					CheckOCLError(clEnqueueNDRangeKernel(queue, kernel, 2,
					                                     nullptr, work_size, nullptr, 0, nullptr, &event));
#ifdef ENABLE_PROFILER
					/* получить данные профилирования по времени */
					total_time += OCLUtils::mesuare_time_sec(event);
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
	out << " | " << OCLUtils::platform_name(platformId) << " " <<
	    OCLUtils::device_name(deviceId)     << " | " <<
	    iterations << " | " << width << " x " << height << " | " <<
	    std::fixed << std::setprecision(3) << time << " | ";
	out.close();
}

//---------------------------------------------------------------
// Последовательная фильтрация
//---------------------------------------------------------------

int get_channel(uint rgba, int channel)
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

int apply_channel(img_data *idata, proc_data *pdata, int x, int y, int ch)
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

void apply(img_data *idata, proc_data *pdata)
{
	for(int it = 0; it < pdata->iterations; ++it) {
		for(int x = 1; x < idata->w-1; ++x) {
			for(int y = 1; y < idata->h-1; ++y) {
				int r = apply_channel(idata, pdata, x, y, 0);
				int g = apply_channel(idata, pdata, x, y, 1);
				int b = apply_channel(idata, pdata, x, y, 2);
				int a = get_channel(idata->bits[y+x*idata->h], 3);
				idata->bits[y+x*idata->h] = ((a & 0xff) << 24) |
				                            ((r & 0xff) << 16) |
				                            ((g & 0xff) << 8)  |
				                            (b & 0xff);
			}
		}
	}
}

//---------------------------------------------------------------
// Экспериментальные функции
//---------------------------------------------------------------

void binarization(img_data *idata, proc_data *pdata)
{
	int tresh = 127;

	for(int x = 0; x < idata->w; ++x) {
		for(int y = 0; y < idata->h; ++y) {
			int r = get_channel(idata->bits[y+x*idata->h], 0);
			int g = get_channel(idata->bits[y+x*idata->h], 1);
			int b = get_channel(idata->bits[y+x*idata->h], 2);
			int gray = sqrt((r*r+g*g+b*b)/3.0);   // to b&w
			int bin = gray >= tresh ? 255 : 0;    // binarization
			bin = !bin ? 255 : 0;                 // inverted
			int a = 1;
			idata->bits[y+x*idata->h] = ((a & 0xff) << 24) |
			                            ((bin & 0xff) << 16) |
			                            ((bin & 0xff) << 8)  |
			                            (bin & 0xff);
		}
	}
}

void bw(img_data *idata)
{
	for(int x = 0; x < idata->w; ++x) {
		for(int y = 0; y < idata->h; ++y) {
			int r = get_channel(idata->bits[y+x*idata->h], 0);
			int g = get_channel(idata->bits[y+x*idata->h], 1);
			int b = get_channel(idata->bits[y+x*idata->h], 2);
			int a = get_channel(idata->bits[y+x*idata->h], 3);
			int bw = sqrt((r*r+g*g+b*b)/3.0);
			idata->bits[y+x*idata->h] = ((a & 0xff) << 24)  |
			                            ((bw & 0xff) << 16) |
			                            ((bw & 0xff) << 8)  |
			                            (bw & 0xff);
		}
	}
}

void edges_laplacian(img_data *idata, proc_data *pdata)
{
	bw(idata);
	float laplacian[3][3] = {{0, 1, 0},
		{1, -4, 1},
		{0, 1, 0}
	};
	int p, a;
	float bw;

	for(int x = 1; x < idata->w-1; ++x) {
		for(int y = 1; y < idata->h-1; ++y) {
			a = get_channel(idata->bits[y+x*idata->h], 3);
			bw = 0.0f;

			for(int i = 0; i < 3; ++i) {
				for(int j = 0; j < 3; ++j) {
					p = get_channel(idata->bits[(y+j) + (x+i) * idata->h], 0);
					bw += p * (float)laplacian[i][j];
				}
			}

			idata->bits[y+x*idata->h] = ((a & 0xff) << 24)  |
			                            (((int)bw & 0xff) << 16) |
			                            (((int)bw & 0xff) << 8)  |
			                            ((int)bw & 0xff);
		}
	}
}

void edges(img_data *idata, proc_data *pdata)
{
	bw(idata);

	for(int x = 1; x < idata->w-1; ++x) {
		for(int y = 1; y < idata->h-1; ++y) {
			int p = get_channel(idata->bits[y + x * idata->h], 0);
			int deltaW = get_channel(idata->bits[y + (x-1) * idata->h], 0) - p;
			int deltaE = get_channel(idata->bits[y + (x+1) * idata->h], 0) - p;
			int deltaS = get_channel(idata->bits[y+1 + x * idata->h], 0) - p;
			int deltaN = get_channel(idata->bits[y-1 + x * idata->h], 0) - p;
			float cN = pdata->conduction_ptr(abs(deltaN), pdata->thresh);
			float cS = pdata->conduction_ptr(abs(deltaS), pdata->thresh);
			float cE = pdata->conduction_ptr(abs(deltaE), pdata->thresh);
			float cW = pdata->conduction_ptr(abs(deltaW), pdata->thresh);
			//std::cout << cN << " " << cS << " " << cE << " " << cW << std::endl;
			p = (cN + cS + cW + cE) / 4.0 >= 0.7f ? 0 : 255.0;

			if(p < 0) p = 0;

			if(p > 255) p = 255;

			idata->bits[y+x*idata->h] = ((1 & 0xff) << 24)  |
			                            ((p & 0xff) << 16) |
			                            ((p & 0xff) << 8)  |
			                            (p & 0xff);
		}
	}
}