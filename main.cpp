#include <iostream>
#include <stdlib.h>
#include "BitmapRawConverter.h"
#include <tbb/task_group.h>
#include <tbb/tick_count.h>

#define __ARG_NUM__				6
#define FILTER_SIZE				5
#define THRESHOLD				128
#define CUT_OFF					1000

using namespace std;

// Prewitt operators
//int filterHor[FILTER_SIZE * FILTER_SIZE] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
//int filterVer[FILTER_SIZE * FILTER_SIZE] = {-1, -1, -1, 0, 0, 0, 1, 1, 1};

int filterHor[FILTER_SIZE * FILTER_SIZE] = { 9, 9, 9, 9, 9,
											9, 5, 5, 5, 9,
											-7, -3, 0, -3, -7,
											-7, -3, -3, -3, -7,
											-7, -7, -7, -7, -7, };
int filterVer[FILTER_SIZE * FILTER_SIZE] = { 9, 9, -7, -7, -7,
											9, 5, -3, -3, -7,
											9, 5, 0, -3, -7,
											9, 5, -3, -3, -7,
											9, 9, -7, -7, -7
											};

/**
* @brief Convolves submatrix and filter
* @param pixelRow current pixel row value
* @param pixelColumn current pixel column value
* @param inBuffer buffer of input image
* @param outBuffer buffer of output image
* @param width image width
*/
int prewitt(int pixelRow, int pixelColumn, int* inBuffer, int* outBuffer, int width) {
	int pixelRowStart = pixelRow - (FILTER_SIZE / 2);
	int pixelColumnStart = pixelColumn - (FILTER_SIZE / 2);
	int sumGy = 0, sumGx = 0;
	for (int i = 0; i < FILTER_SIZE; ++i) {
		for (int j = 0; j < FILTER_SIZE; ++j) {
			sumGy += inBuffer[(pixelRowStart + i) * width + (pixelColumnStart + j)] * filterVer[i * FILTER_SIZE + j];
			sumGx += inBuffer[(pixelRowStart + i) * width + (pixelColumnStart + j)] * filterHor[i * FILTER_SIZE + j];
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
*/
int detectEdges(int pixelRowStart, int pixelColumnStart, int* inBuffer, int* outBuffer, int width) {
	int P = 0, O = 1;
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
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
*/
void filter_serial_prewitt(int *inBuffer, int *outBuffer, int width, int height, int rowStart=0, int rowEnd=-1)
{
	int offset = FILTER_SIZE / 2;
	if (rowEnd == -1)
		rowEnd = height - offset;
	
	for (int i = rowStart; i < rowEnd; ++i) {
		for (int j = offset; j < width - offset; ++j) {
			if (i < FILTER_SIZE / 2 || i > height - FILTER_SIZE / 2)
				continue;
			outBuffer[i * width + j] = prewitt(i, j, inBuffer, outBuffer, width) >= 128 ? 255 : 0;
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
*/
void filter_parallel_prewitt(int *inBuffer, int *outBuffer, int width, int height, int rowStart=0, int rowEnd=-1)
{	
	if (rowEnd == -1)
		rowEnd = height - FILTER_SIZE / 2;
	if ((rowEnd - rowStart) < CUT_OFF) {
		filter_serial_prewitt(inBuffer, outBuffer, width, height, rowStart, rowEnd);
	}
	else {
		tbb::task_group tg;
		tg.run([=]() {filter_parallel_prewitt(inBuffer, outBuffer, width, height, rowStart, (rowStart + rowEnd) / 2); });
		tg.run([=]() {filter_parallel_prewitt(inBuffer, outBuffer, width, height, (rowStart + rowEnd) / 2, rowEnd); });
		tg.wait();
	}
}

/**
* @brief Serial version of edge detection algorithm
* @param inBuffer buffer of input image
* @param outBuffer buffer of output image
* @param width image width
* @param height image height
*/
void filter_serial_edge_detection(int *inBuffer, int *outBuffer, int width, int height, int rowStart=0, int rowEnd=-1)
{
	int offset = FILTER_SIZE / 2;
	if (rowEnd == -1)
		rowEnd = height - offset;

	for (int i = rowStart; i < rowEnd; ++i) {
		for (int j = offset; j < width - offset; ++j) {
			if (i < FILTER_SIZE / 2 || i > height - FILTER_SIZE / 2)
				continue;
			outBuffer[i * width + j] = detectEdges(i - 1, j - 1, inBuffer, outBuffer, width) ? 255 : 0;
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
*/
void filter_parallel_edge_detection(int *inBuffer, int *outBuffer, int width, int height, int rowStart=0, int rowEnd=-1)
{
	if (rowEnd == -1)
		rowEnd = height - FILTER_SIZE / 2;
	if ((rowEnd - rowStart) < CUT_OFF) {
		filter_serial_edge_detection(inBuffer, outBuffer, width, height, rowStart, rowEnd);
	}
	else {
		tbb::task_group tg;
		tg.run([=]() {filter_parallel_edge_detection(inBuffer, outBuffer, width, height, rowStart, (rowStart + rowEnd) / 2); });
		tg.run([=]() {filter_parallel_edge_detection(inBuffer, outBuffer, width, height, (rowStart + rowEnd) / 2, rowEnd); });
		tg.wait();
	}
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
*/


void run_test_nr(int testNr, BitmapRawConverter* ioFile, char* outFileName, int* outBuffer, unsigned int width, unsigned int height)
{
	auto start = tbb::tick_count::now();

	switch (testNr)
	{
		case 1:
			cout << "Running serial version of edge detection using Prewitt operator" << endl;
			filter_serial_prewitt(ioFile->getBuffer(), outBuffer, width, height);
			break;
		case 2:
			cout << "Running parallel version of edge detection using Prewitt operator" << endl;
			filter_parallel_prewitt(ioFile->getBuffer(), outBuffer, width, height);
			break;
		case 3:
			cout << "Running serial version of edge detection" << endl;
			filter_serial_edge_detection(ioFile->getBuffer(), outBuffer, width, height);
			break;
		case 4:
			cout << "Running parallel version of edge detection" << endl;
			filter_parallel_edge_detection(ioFile->getBuffer(), outBuffer, width, height);
			break;
		default:
			cout << "ERROR: invalid test case, must be 1, 2, 3 or 4!";
			break;
	}
	auto end = tbb::tick_count::now();
	cout << "Lasted: " << (end - start).count() << endl;

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
	cout << " outputParallelEdge.bmp" << endl << endl;
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

	// serial version Prewitt
	run_test_nr(1, &outputFileSerialPrewitt, argv[2], outBufferSerialPrewitt, width, height);

	// parallel version Prewitt
	run_test_nr(2, &outputFileParallelPrewitt, argv[3], outBufferParallelPrewitt, width, height);

	// serial version special
	run_test_nr(3, &outputFileSerialEdge, argv[4], outBufferSerialEdge, width, height);

	// parallel version special
	run_test_nr(4, &outputFileParallelEdge, argv[5], outBufferParallelEdge, width, height);

	// verification
	cout << "Verification: ";
	test = memcmp(outBufferSerialPrewitt, outBufferParallelPrewitt, width * height * sizeof(int));

	if(test != 0)
	{
		cout << "Prewitt FAIL!" << endl;
	}
	else
	{
		cout << "Prewitt PASS." << endl;
	}

	test = memcmp(outBufferSerialEdge, outBufferParallelEdge, width * height * sizeof(int));

	if(test != 0)
	{
		cout << "Edge detection FAIL!" << endl;
	}
	else
	{
		cout << "Edge detection PASS." << endl;
	}

	// clean up
	delete outBufferSerialPrewitt;
	delete outBufferParallelPrewitt;

	delete outBufferSerialEdge;
	delete outBufferParallelEdge;

	return 0;
} 