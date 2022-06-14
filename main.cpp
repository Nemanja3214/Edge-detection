#include <iostream>
#include <stdlib.h>
#include "BitmapRawConverter.h"
#include <tbb/task_group.h>
#include <tbb/tick_count.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#define __ARG_NUM__				10
#define THRESHOLD				128
#define CUT_OFF					300

using namespace std;

// Prewitt operators
int filterHor3[3 * 3] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
int filterVer3[3 * 3] = {-1, -1, -1, 0, 0, 0, 1, 1, 1};

int filterHor5[5 * 5] = {9, 9, 9, 9, 9,
						9, 5, 5, 5, 9,
						-7, -3, 0, -3, -7,
						-7, -3, -3, -3, -7,
						-7, -7, -7, -7, -7, };
int filterVer5[5 * 5] = { 9, 9, -7, -7, -7,
						9, 5, -3, -3, -7,
						9, 5, 0, -3, -7,
						9, 5, -3, -3, -7,
						9, 9, -7, -7, -7
						};

int filterHor7[7 * 7] = {-3, -2, -1, 0, 1, 2, 3,
						-3, -2, -1, 0, 1, 2, 3,
						-3, -2, -1, 0, 1, 2, 3,
						-3, -2, -1, 0, 1, 2, 3,
						-3, -2, -1, 0, 1, 2, 3, };
int filterVer7[7 * 7] = { -3, -2, -1, 0, 1, 2, 3,
						-3, -2, -1, 0, 1, 2, 3,
						-3, -2, -1, 0, 1, 2, 3,
						-3, -2, -1, 0, 1, 2, 3,
						-3, -2, -1, 0, 1, 2, 3,
};

/**
* @brief Convolves submatrix and filters and returns G
* @param pixelRow current pixel row value
* @param pixelColumn current pixel column value
* @param inBuffer buffer of input image
* @param outBuffer buffer of output image
* @param width image width
* @param filterVer vertical component filter
* @param filterHor horizontal component filter
* @param filterSize size of the filter
*/
int prewitt(int pixelRow, int pixelColumn, int* inBuffer, int* outBuffer, int width, int* filterVer, int* filterHor,
	int filterSize) {
	int pixelRowStart = pixelRow - (filterSize / 2);
	int pixelColumnStart = pixelColumn - (filterSize / 2);
	int sumGy = 0, sumGx = 0;
	for (int i = 0; i < filterSize; ++i) {
		for (int j = 0; j < filterSize; ++j) {
			sumGy += inBuffer[(pixelRowStart + i) * width + (pixelColumnStart + j)] * filterVer[i * filterSize + j];
			sumGx += inBuffer[(pixelRowStart + i) * width + (pixelColumnStart + j)] * filterHor[i * filterSize + j];
		}
	}
	return std::abs(sumGy) + std::abs(sumGx);
}

/**
* @brief Searches surrounding area to see if the pixel is part of the edge
* @param pixelRow current pixel row value
* @param pixelColumn current pixel column value
* @param inBuffer buffer of input image
* @param outBuffer buffer of output image
* @param width image width
* @param lookupWidth size of neighbour lookup matrix
*/
int detectEdges(int pixelRowStart, int pixelColumnStart, int* inBuffer, int* outBuffer, int width, int lookupWidth) {
	int P = 0, O = 1;
	for (int i = 0; i < lookupWidth; ++i) {
		for (int j = 0; j < lookupWidth; ++j) {
			if (inBuffer[(pixelRowStart + i) * width + (pixelColumnStart + j)] >= 128)
				P = 1;
			if (inBuffer[(pixelRowStart + i) * width + (pixelColumnStart + j)] < 128)
				O = 0;
		}
	}
	return std::abs(P - O);
}


/**
* @brief Serial version of edge detection algorithm implementation using Prewitt operator
* @param inBuffer buffer of input image
* @param outBuffer buffer of output image
* @param width image width
* @param height image height
* @param filterVer vertical component filter
* @param filterHor horizontal component filter
* @param filterSize size of the filter
* @param rowStart from where does row processing start
* @param rowEnd where does row processing end
*/
void filter_serial_prewitt(int *inBuffer, int *outBuffer, int width, int height, int* filterVer, int* filterHor,
	int filterSize, int rowStart=0, int rowEnd=-1)
{
	int offset = filterSize / 2;
	if (rowEnd == -1)
		rowEnd = height - offset;
	
	for (int i = rowStart; i < rowEnd; ++i) {
		for (int j = offset; j < width - offset; ++j) {
			if (i < filterSize / 2 || i > height - filterSize / 2)
				continue;
			outBuffer[i * width + j] = prewitt(i, j, inBuffer, outBuffer, width, filterVer, filterHor, filterSize) >= 128 ? 255 : 0;
		}
	}
}


/**
* @brief Parallel version of edge detection algorithm implementation using Prewitt operator
* 
* @param inBuffer buffer of input image
* @param outBuffer buffer of output image
* @param width image width
* @param height image height
* @param filterVer vertical component filter
* @param filterHor horizontal component filter
* @param filterSize size of the filter
* @param rowStart from where does row processing start
* @param rowEnd where does row processing end
*/
void filter_parallel_prewitt(int *inBuffer, int *outBuffer, int width, int height, int* filterVer, int* filterHor, int filterSize,
	int rowStart=0, int rowEnd=-1)
{	
	if (rowEnd == -1)
		rowEnd = height - filterSize / 2;
	if ((rowEnd - rowStart) < CUT_OFF) {
		filter_serial_prewitt(inBuffer, outBuffer, width, height, filterVer, filterHor, filterSize, rowStart, rowEnd);
	}
	else {
		tbb::task_group tg;
		tg.run([=]() {filter_parallel_prewitt(inBuffer, outBuffer, width, height, filterVer, filterHor, filterSize, rowStart, (rowStart + rowEnd) / 2); });
		tg.run([=]() {filter_parallel_prewitt(inBuffer, outBuffer, width, height, filterVer, filterHor, filterSize, (rowStart + rowEnd) / 2, rowEnd); });
		tg.wait();
	}
}

/**
* @brief Serial version of edge detection algorithm
* @param inBuffer buffer of input image
* @param outBuffer buffer of output image
* @param width image width
* @param height image height
* @param lookupWidth size of neighbour lookup matrix
* @param rowStart from where does row processing start
* @param rowEnd where does row processing end
*/
void filter_serial_edge_detection(int *inBuffer, int *outBuffer, int width, int height, int lookupWidth, int rowStart=0, int rowEnd=-1)
{
	int offset = lookupWidth / 2;
	if (rowEnd == -1)
		rowEnd = height - offset;

	for (int i = rowStart; i < rowEnd; ++i) {
		for (int j = offset; j < width - offset; ++j) {
			if (i < lookupWidth / 2 || i > height - lookupWidth / 2)
				continue;
			outBuffer[i * width + j] = detectEdges(i - offset, j - offset, inBuffer, outBuffer, width, lookupWidth) ? 255 : 0;
		}
	}
}

/**
* @brief Parallel version of edge detection algorithm
* 
* @param inBuffer buffer of input image
* @param outBuffer buffer of output image
* @param width image width
* @param height image height
* @param lookupWidth size of neighbour lookup matrix
* @param filterVer vertical component filter
* @param filterHor horizontal component filter
* @param filterSize size of the filter
* @param rowStart from where does row processing start
* @param rowEnd where does row processing end
*/
void filter_parallel_edge_detection(int *inBuffer, int *outBuffer, int width, int height, int lookupWidth, int rowStart=0, int rowEnd=-1)
{
	if (rowEnd == -1)
		rowEnd = height - lookupWidth / 2;
	if ((rowEnd - rowStart) < CUT_OFF) {
		filter_serial_edge_detection(inBuffer, outBuffer, width, height, lookupWidth, rowStart, rowEnd);
	}
	else {
		tbb::task_group tg;
		tg.run([=]() {filter_parallel_edge_detection(inBuffer, outBuffer, width, height, lookupWidth, rowStart, (rowStart + rowEnd) / 2); });
		tg.run([=]() {filter_parallel_edge_detection(inBuffer, outBuffer, width, height, lookupWidth, (rowStart + rowEnd) / 2, rowEnd); });
		tg.wait();
	}
}

/**
* @brief Structure to be called for parallel for implementations for Prewwit edge detection
*
* @param inBuffer buffer of input image
* @param outBuffer buffer of output image
* @param width image width
* @param height image height
* @param filterVer vertical component filter
* @param filterHor horizontal component filter
* @param filterSize size of the filter
*/

struct ApplyPrewitt {
	int* inBuffer;
	int* outBuffer;
	int width;
	int height;
	int* filterVer;
	int* filterHor;
	int filterSize;
	ApplyPrewitt(int* inBuffer, int* outBuffer, int width, int height, int* filterVer, int* filterHor, int filterSize) : inBuffer(inBuffer),
		outBuffer(outBuffer), width(width), height(height), filterHor(filterHor), filterVer(filterVer), filterSize(filterSize) {};
	void operator()(const tbb::blocked_range<int> range) const{
		int offset = filterSize / 2;

		for (int i = range.begin(); i < range.end(); ++i) {
			for (int j = offset; j < width - offset; ++j) {
				if (i < filterSize / 2 || i > height - filterSize / 2)
					continue;
				outBuffer[i * width + j] = prewitt(i, j, inBuffer, outBuffer, width, filterVer, filterHor, filterSize) >= 128 ? 255 : 0;
			}
		}
	}
};

/**
* @brief Parallel for version of edge detection algorithm implementation using Prewitt operator
*
* @param inBuffer buffer of input image
* @param outBuffer buffer of output image
* @param width image width
* @param height image height
* @param filterVer vertical component filter
* @param filterHor horizontal component filter
* @param filterSize size of the filter
* @param affinity should it use affinity toward cache memory or no
*/
void filter_parallel_for_prewitt(int* inBuffer, int* outBuffer, int width, int height, int* filterVer, int* filterHor,
	int filterSize, bool affinity = false)
{
	int rowStart = 0, rowEnd = height - filterSize / 2;
	ApplyPrewitt ap(inBuffer, outBuffer, width, height, filterVer, filterHor, filterSize);
	if (affinity) {
		static tbb::affinity_partitioner affinityPartitioner;
		tbb::parallel_for(tbb::blocked_range<int>(rowStart, rowEnd), ap, affinityPartitioner);
	}
		
	else
		tbb::parallel_for(tbb::blocked_range<int>(rowStart, rowEnd), ap, tbb::auto_partitioner());
}

/**
* @brief Structure to be called for parallel for implementations for edge detection algorithm
*
* @param inBuffer buffer of input image
* @param outBuffer buffer of output image
* @param width image width
* @param height image height
* @param lookupWidth size of neighbour lookup matrix
*/

struct ApplyEdge {
	int* inBuffer;
	int* outBuffer;
	int width;
	int height;
	int lookupWidth;
	ApplyEdge(int* inBuffer, int* outBuffer, int width, int height, int lookupWidth) :
		inBuffer(inBuffer), outBuffer(outBuffer), width(width), height(height), lookupWidth(lookupWidth) {};
	void operator()(const tbb::blocked_range<int> range) const {
		int offset = lookupWidth / 2;
		for (int i = range.begin(); i < range.end(); ++i) {
			for (int j = offset; j < width - offset; ++j) {
				if (i < lookupWidth / 2 || i > height - lookupWidth / 2)
					continue;
				outBuffer[i * width + j] = detectEdges(i - offset, j - offset, inBuffer, outBuffer, width, lookupWidth) ? 255 : 0;
			}
		}
	}
};

/**
* @brief Parallel for version of edge detection algorithm implementation
*
* @param inBuffer buffer of input image
* @param outBuffer buffer of output image
* @param width image width
* @param height image height
* @param lookupWidth size of neighbour lookup matrix
* @param affinity should it use affinity toward cache memory or no
*/

void filter_parallel_for_edge_detection(int* inBuffer, int* outBuffer, int width, int height, int lookupWidth, bool affinity = false)
{
	int rowStart = 0, rowEnd = height - lookupWidth / 2;
	ApplyEdge ae(inBuffer, outBuffer, width, height, lookupWidth);
	if (affinity) {
		static tbb::affinity_partitioner affinityPartitioner;
		tbb::parallel_for(tbb::blocked_range<int>(rowStart, rowEnd), ae, affinityPartitioner);
	}
	else
		tbb::parallel_for(tbb::blocked_range<int>(rowStart, rowEnd), ae, tbb::auto_partitioner());
}




/**
* @brief Function for running test.
*
* @param testNr test identification, 1: for serial version, 2: for parallel version
* @param ioFile input/output file, firstly it's holding buffer from input image and than to hold filtered data
* @param outFileName output file name
* @param outBuffer buffer of output image
* @param width image width
* @param height image height
* @param filterVer vertical component filter
* @param filterHor horizontal component filter
* @param filterSize size of the filter
*/


void run_test_nr(int testNr, BitmapRawConverter* ioFile, char* outFileName, int* outBuffer, unsigned int width,
	unsigned int height, int lookupWidth, int* filterVer, int* filterHor, int filterSize )
{
	auto start = tbb::tick_count::now();

	switch (testNr)
	{
		case 1:
			cout << "Running serial version of edge detection using Prewitt operator" << endl;
			filter_serial_prewitt(ioFile->getBuffer(), outBuffer, width, height, filterVer, filterHor, filterSize);
			break;
		case 2:
			cout << "Running parallel version of edge detection using Prewitt operator" << endl;
			filter_parallel_prewitt(ioFile->getBuffer(), outBuffer, width, height, filterVer, filterHor, filterSize);
			break;
		case 5:
			cout << "Running parallel for version of edge detection using Prewitt operator" << endl;
			filter_parallel_for_prewitt(ioFile->getBuffer(), outBuffer, width, height, filterVer, filterHor, filterSize);
			break;
		case 7:
			cout << "Running parallel for affinity version of edge detection using Prewitt operator" << endl;
			filter_parallel_for_prewitt(ioFile->getBuffer(), outBuffer, width, height, filterVer, filterHor, filterSize, true);
			break;


		case 3:
			cout << "Running serial version of edge detection" << endl;
			filter_serial_edge_detection(ioFile->getBuffer(), outBuffer, width, height, lookupWidth);
			break;
		case 4:
			cout << "Running parallel version of edge detection" << endl;
			filter_parallel_edge_detection(ioFile->getBuffer(), outBuffer, width, height, lookupWidth);
			break;
		case 6:
			cout << "Running parallel for version of edge detection" << endl;
			filter_parallel_for_edge_detection(ioFile->getBuffer(), outBuffer, width, height, lookupWidth);
			break;
		case 8:
			cout << "Running parallel for affinity version of edge detection" << endl;
			filter_parallel_for_edge_detection(ioFile->getBuffer(), outBuffer, width, height, lookupWidth, true);
			break;
		default:
			cout << "ERROR: invalid test case, must be 1, 2, 3 or 4!";
			break;
	}
	auto end = tbb::tick_count::now();
	cout << "Lasted: " << (end - start).seconds() << endl;

	ioFile->setBuffer(outBuffer);
	ioFile->pixelsToBitmap(outFileName);
}

/**
* @brief Print program usage.
*/
void usage()
{
	cout << "\n\ERROR: call program like: " << endl << endl; 
	cout << "ProjekatPP.exe";
	cout << " input.bmp";
	cout << " outputSerialPrewitt.bmp";
	cout << " outputParallelPrewitt.bmp";
	cout << " outputSerialEdge.bmp";
	cout << " outputParallelEdge.bmp";

	cout << " outputParallelForPrewitt.bmp";
	cout << " outputParallelForEdge.bmp";
	cout << " outputParallelForAffinityPrewitt.bmp";
	cout << " outputParallelForAffinityEdge.bmp" << endl << endl;
}

int main(int argc, char * argv[])
{

	if(argc != __ARG_NUM__)
	{
		usage();
		return 0;
	}

	BitmapRawConverter inputFile(argv[1]);
	BitmapRawConverter outputFileSerialPrewitt(argv[1]);
	BitmapRawConverter outputFileParallelPrewitt(argv[1]);
	BitmapRawConverter outputFileSerialEdge(argv[1]);
	BitmapRawConverter outputFileParallelEdge(argv[1]);

	BitmapRawConverter outputFileParallelForPrewitt(argv[1]);
	BitmapRawConverter outputFileParallelForEdge(argv[1]);
	BitmapRawConverter outputFileParallelForAffinityPrewitt(argv[1]);
	BitmapRawConverter outputFileParallelForAffinityEdge(argv[1]);

	unsigned int width, height;

	int test;
	
	width = inputFile.getWidth();
	height = inputFile.getHeight();

	int* outBufferSerialPrewitt = new int[width * height];
	int* outBufferParallelPrewitt = new int[width * height];

	memset(outBufferSerialPrewitt, 0x0, width * height * sizeof(int));
	memset(outBufferParallelPrewitt, 0x0, width * height * sizeof(int));

	int* outBufferSerialEdge = new int[width * height];
	int* outBufferParallelEdge = new int[width * height];

	memset(outBufferSerialEdge, 0x0, width * height * sizeof(int));
	memset(outBufferParallelEdge, 0x0, width * height * sizeof(int));



	int* outBufferParallelForPrewitt = new int[width * height];
	int* outBufferParallelForEdge = new int[width * height];

	memset(outBufferParallelForPrewitt, 0x0, width * height * sizeof(int));
	memset(outBufferParallelForEdge, 0x0, width * height * sizeof(int));

	int* outBufferParallelForAffinityPrewitt = new int[width * height];
	int* outBufferParallelForAffinityEdge = new int[width * height];

	memset(outBufferParallelForAffinityPrewitt, 0x0, width * height * sizeof(int));
	memset(outBufferParallelForAffinityEdge, 0x0, width * height * sizeof(int));


	int lookupWidth;
	int* filterVer;
	int* filterHor;

	cout << "Choose lookup width for edge detection: " << endl;
	cin >> lookupWidth;
	if (lookupWidth < 3 || lookupWidth % 2 == 0) {
		cout << "Invalid lookup width, default 3 is set" << endl;
		lookupWidth = 3;
	}

	int filterSize;
	cout << "Choose filter size for prewitt matrix (valid options are 3, 5 and 7): " << endl;
	cin >> filterSize;
	switch (filterSize) {
	case 3:
		filterSize = 3;
		filterHor = filterHor3;
		filterVer = filterVer3;
		break;
	case 5:
		filterSize = 5;
		filterHor = filterHor5;
		filterVer = filterVer5;
		break;
	case 7:
		filterSize = 7;
		filterHor = filterHor7;
		filterVer = filterVer7;
		break;
	default:
		cout << "Invalid filter size is selected, default 3 is set" << endl;
		filterSize = 3;
		filterHor = filterHor3;
		filterVer = filterVer3;
	}

	if (filterSize < 3 || filterSize % 2 == 0) {
		cout << "Invalid filter size, default 3 is set" << endl;
		filterSize = 3;
		filterHor = filterHor3;
		filterVer = filterVer3;
	}

	// serial version Prewitt
	run_test_nr(1, &outputFileSerialPrewitt, argv[2], outBufferSerialPrewitt, width, height, lookupWidth, filterVer, filterHor, filterSize);

	// parallel version Prewitt
	run_test_nr(2, &outputFileParallelPrewitt, argv[3], outBufferParallelPrewitt, width, height, lookupWidth, filterVer, filterHor, filterSize);

	// parallel for version Prewitt
	run_test_nr(5, &outputFileParallelForPrewitt, argv[6], outBufferParallelForPrewitt, width, height, lookupWidth, filterVer, filterHor, filterSize);

	// parallel for version Prewitt
	run_test_nr(7, &outputFileParallelForAffinityPrewitt, argv[8], outBufferParallelForAffinityPrewitt, width, height, lookupWidth, filterVer, filterHor, filterSize);

	cout << endl << endl;

	// serial version special
	run_test_nr(3, &outputFileSerialEdge, argv[4], outBufferSerialEdge, width, height, lookupWidth, filterVer, filterHor, filterSize);

	// parallel version special
	run_test_nr(4, &outputFileParallelEdge, argv[5], outBufferParallelEdge, width, height, lookupWidth, filterVer, filterHor, filterSize);

	// parallel for version special
	run_test_nr(6, &outputFileParallelForEdge, argv[7], outBufferParallelForEdge, width, height, lookupWidth, filterVer, filterHor, filterSize);

	// parallel for version special
	run_test_nr(8, &outputFileParallelForAffinityEdge, argv[9], outBufferParallelForAffinityEdge, width, height, lookupWidth, filterVer, filterHor, filterSize);

	cout << endl << endl;

	// verification
	cout << "Verification: " << endl;
	// task parallel
	test = memcmp(outBufferSerialPrewitt, outBufferParallelPrewitt, width * height * sizeof(int));

	if(test != 0)
	{
		cout << "Prewitt task FAIL!" << endl;
	}
	else
	{
		cout << "Prewitt task PASS." << endl;
	}

	// parallel for
	test = memcmp(outBufferSerialPrewitt, outBufferParallelForPrewitt, width * height * sizeof(int));

	if (test != 0)
	{
		cout << "Prewitt for FAIL!" << endl;
	}
	else
	{
		cout << "Prewitt for PASS." << endl;
	}

	// parallel for affinity
	test = memcmp(outBufferSerialPrewitt, outBufferParallelForAffinityPrewitt, width * height * sizeof(int));

	if (test != 0)
	{
		cout << "Prewitt for affinity FAIL!" << endl;
	}
	else
	{
		cout << "Prewitt for affinity PASS." << endl;
	}




	// task parallel 
	test = memcmp(outBufferSerialEdge, outBufferParallelEdge, width * height * sizeof(int));

	if(test != 0)
	{
		cout << "Edge detection task FAIL!" << endl;
	}
	else
	{
		cout << "Edge detection task PASS." << endl;
	}

	// parallel for
	test = memcmp(outBufferSerialEdge, outBufferParallelForEdge, width * height * sizeof(int));

	if (test != 0)
	{
		cout << "Edge detection for FAIL!" << endl;
	}
	else
	{
		cout << "Edge detection for PASS." << endl;
	}

	// parallel for
	test = memcmp(outBufferSerialEdge, outBufferParallelForAffinityEdge, width * height * sizeof(int));

	if (test != 0)
	{
		cout << "Edge detection for affinity FAIL!" << endl;
	}
	else
	{
		cout << "Edge detection for affinity PASS." << endl;
	}

	// clean up
	delete outBufferSerialPrewitt;
	delete outBufferParallelPrewitt;

	delete outBufferSerialEdge;
	delete outBufferParallelEdge;

	delete outBufferParallelForPrewitt;
	delete outBufferParallelForEdge;
	delete outBufferParallelForAffinityPrewitt;
	delete outBufferParallelForAffinityEdge;

	return 0;
} 