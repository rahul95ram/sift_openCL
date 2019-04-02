#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "CL/cl.h"
#include <opencv.hpp>
#include <windows.h>
#include "read_source.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <iostream>
#include <sstream>
#include <cstring>
#include <string>
#include <windows.h>


using namespace std;

float GetMean(float * number, int count);
float GetStandardDeviation(float *number, float average, int count);
cl_image_desc img_description(bool is_img_array, int width, int height, int array_size = 0);

int main(int argc, char** argv)
{
	//add declarations for necessary OpenCl API variables
	cl_platform_id *platforms = NULL;
	cl_uint num_platforms = 0;
	cl_device_id *device_list = NULL;
	cl_uint num_devices = 0;
	char *name = NULL;
	char *device_name = NULL;
	size_t size;
	size_t device_size;
	size_t file_size;
	cl_int err;

	//image setting
	cl_mem InputImage;
	cl_mem DownSampleImage;
	cl_mem GaussianBlurImage;
	cl_mem Filter;
	cl_mem FSum;
	cl_mem Sigma;
	cl_mem DoGImage;
	cl_mem Extrema;
	cl_mem KeyPoint;
	cl_mem Index1;
	cl_mem Index2;
	int width, height, channels;

	float sigma[11] = {0.707,1.0,1.414,2,2.828,4,5.656, 8, 11.313, 16, 22.627};
	int num_sigma = 6;
	cl_int sizex = 1;//downsample size
	cl_int sizey = 1;
	cl_int filterWidth = 10/sizex;
	cl_int filterSize = filterWidth * filterWidth*6;
	int * idx1 = (int*)_aligned_malloc(sizeof(int) * 1, 4096); idx1[0] = 0;
	int * idx2 = (int*)_aligned_malloc(sizeof(int) * 1, 4096); idx2[0] = 0;
	int * extremapoints = (int*)_aligned_malloc(sizeof(int) * 800*3, 4096);
	int * key_points = (int*)_aligned_malloc(sizeof(int) * 800*3, 4096);
	float * gaussBlurFilter = (float*)_aligned_malloc(sizeof(float)*filterSize, 4096);
	float * filtsum = (float*)_aligned_malloc(sizeof(float)*filterSize, 4096);
	
	size_t filterworksize[] = { filterWidth*filterWidth,6,0 };
	size_t filterlocalworksize[] = { filterWidth*filterWidth,1,0 };
	LARGE_INTEGER frequency;
	LARGE_INTEGER start;
	LARGE_INTEGER end;
	string fileIdx[] = { "0","1","2","3","4","5" };
	cl_ulong start_time;
	cl_ulong end_time;
	int iteration = 1;
	float time_ave = 0;
	float time_standard_deviation = 0;
	float time_series[100];

	cl_context context;
	cl_program program;
	cl_command_queue command;
	cl_kernel kernel[6] = {NULL, NULL,NULL,NULL,NULL,NULL};

	int dimention = 2;
	char *KernelSource = read_source("device.cl", &file_size);

	unsigned char *indata = NULL;
	unsigned char *graydata = NULL;
	unsigned char *downsample = NULL;
	unsigned char *gaussianblur = NULL;
	unsigned char *dogimage = NULL;
	unsigned char *outdata = NULL;
	unsigned char *outimage = NULL;

	string inputfile = "C:/Final5/Final/images/512_512.png";
	string outputfile = "C:/Final5/Final/images/";

	cv::Mat image = cv::imread(inputfile);
	indata = stbi_load(inputfile.c_str(), &width, &height, &channels, 0);

	graydata = (unsigned char*)malloc(width*height * sizeof(unsigned char));

	downsample = (unsigned char *)malloc(width*height* sizeof(unsigned char)/sizex/sizey);

	gaussianblur = (unsigned char*)malloc(width*height * sizeof(unsigned char) / sizex / sizey * num_sigma);

	dogimage = (unsigned char*)malloc(width*height * sizeof(unsigned char) / sizex / sizey * (num_sigma-1));

	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			graydata[i*height + j] = (float)(indata[(i*height + j)*channels + 0])* 0.299f+ (float)(indata[(i*height + j)*channels + 1]) * 0.587f+(float)(indata[(i*height + j)*channels + 2]) * 0.114f;
		}
		
	}

	size_t globalworksize[] = { width,height,num_sigma };
	size_t globalworksize_DOG[]= { width,height,num_sigma-1 };
	size_t globalworksize_Extrema[] = { width,height,2 };
	size_t globalworksize_DownSample[] = { width / sizex,height / sizey,num_sigma };
	size_t localworksize[] = { 1,1,1};
	size_t localworksize2[] = { 1,0,0 };

	outdata = (unsigned char*)malloc((width / sizex )*(height / sizey) * sizeof(unsigned char)*num_sigma);
	outimage = (unsigned char*)malloc((width / sizex)*(height / sizey) * sizeof(unsigned char));

	//Get platform
	clGetPlatformIDs(0, NULL, &num_platforms);//first call,get the num_platforms
	platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id)*num_platforms);//memory allocation
	err = clGetPlatformIDs(num_platforms, platforms, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to create a platform group!\n");
		return EXIT_FAILURE;
	}

	//choose the first platform,i.e. intel(R)
	clGetPlatformInfo(platforms[0], CL_PLATFORM_NAME, NULL, NULL, &size);
	name = (char*)malloc(size);
	clGetPlatformInfo(platforms[0], CL_PLATFORM_NAME, size, name, NULL);
	printf(name);
	printf("\n");

	//get device information
	err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
	device_list = (cl_device_id*)malloc(sizeof(cl_device_id)*num_devices);
	err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, num_devices, device_list, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to create a device group!\n");
		return EXIT_FAILURE;
	}

	//choose the first device
	clGetDeviceInfo(device_list[0], CL_DEVICE_NAME, NULL, NULL, &device_size);
	device_name = (char*)malloc(device_size);
	clGetDeviceInfo(device_list[0], CL_DEVICE_NAME, device_size, device_name, NULL);
	printf(device_name);
	printf("\n");

	//creat context
	context = clCreateContext(0, 1, &device_list[0], NULL, NULL, &err);
	if (!context)
	{
		printf("Error:Failed to creat a context!\n");
		return EXIT_FAILURE;
	}

	//create program
	program = clCreateProgramWithSource(context, 1, (const char **)& KernelSource, NULL, &err);
	if (!program)
	{
		printf("Error:Failed to creat a program!\n");
		return EXIT_FAILURE;
	}

	//build program
	err = clBuildProgram(program, 1, &device_list[0], "-cl-std=CL2.0", NULL, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to build program!\n");
		return EXIT_FAILURE;
	}

	//create command
	command = clCreateCommandQueueWithProperties(context, device_list[0], 0, NULL);
	if (!command)
	{
		printf("Error:Failed to creat a command queue!\n");
		return EXIT_FAILURE;
	}

	//create kernel
	kernel[0] = clCreateKernel(program, "DownSample", &err);
	if (!kernel[0])
	{
		printf("Error:Failed to creat kernel0!\n");
		return EXIT_FAILURE;
	}
	
	kernel[1] = clCreateKernel(program, "GaussianFilter", &err);
	if (!kernel[1])
	{
		printf("Error:Failed to creat kernel1!\n");
		return EXIT_FAILURE;
	}

	kernel[2] = clCreateKernel(program, "GaussianBlur", &err);
	if (!kernel[2])
	{
		printf("Error:Failed to creat kernel2!\n");
		return EXIT_FAILURE;
	}
	
	kernel[3] = clCreateKernel(program, "DifferenceOfGaussian", &err);
	if (!kernel[3])
	{
		printf("Error:Failed to creat kernel3!\n");
		return EXIT_FAILURE;
	}
	
	kernel[4] = clCreateKernel(program, "Extrema", &err);
	if (!kernel[4])
	{
		printf("Error:Failed to creat kernel4!\n");
		return EXIT_FAILURE;
	}

	kernel[5] = clCreateKernel(program, "KeyPoints", &err);
	if (!kernel[5])
	{
		printf("Error:Failed to creat kernel5!\n");
		return EXIT_FAILURE;
	}
	
	//create input and output buffer
	InputImage = clCreateBuffer(context, CL_MEM_USE_HOST_PTR , sizeof(unsigned char)*width*height, graydata, &err);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to create inputimage buffer!\n");
		return EXIT_FAILURE;
	}

	DownSampleImage = clCreateBuffer(context, CL_MEM_USE_HOST_PTR, sizeof(unsigned char)*width*height/sizex/sizey, downsample, &err);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to create DownSampleImage!\n");
		return EXIT_FAILURE;
	}
	
	GaussianBlurImage = clCreateBuffer(context, CL_MEM_USE_HOST_PTR, sizeof(unsigned char)*width*height /sizex / sizey*num_sigma, gaussianblur, &err);

	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to create GaussianBlurImage!\n");
		return EXIT_FAILURE;
	}
	
	DoGImage = clCreateBuffer(context, CL_MEM_USE_HOST_PTR, sizeof(unsigned char)*width*height / sizex / sizey * (num_sigma-1), dogimage, &err);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to create DoGImage!\n");
		return EXIT_FAILURE;
	}
	
	Extrema = clCreateBuffer(context, CL_MEM_USE_HOST_PTR, sizeof(int) * 2400, extremapoints, &err);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to create buffer for Extrema!\n");
		return EXIT_FAILURE;
	}

	KeyPoint = clCreateBuffer(context, CL_MEM_USE_HOST_PTR, sizeof(int) * 2400, key_points, &err);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to create buffer for Key Points!\n");
		return EXIT_FAILURE;
	}
	
	Filter = clCreateBuffer(context, CL_MEM_USE_HOST_PTR, sizeof(float)*filterSize, gaussBlurFilter, &err);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to create buffer!\n");
		return EXIT_FAILURE;
	}

	FSum = clCreateBuffer(context, CL_MEM_USE_HOST_PTR, sizeof(float)*filterSize, filtsum, &err);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to create buffer!\n");
		return EXIT_FAILURE;
	}

	Index1 = clCreateBuffer(context, CL_MEM_USE_HOST_PTR, sizeof(int)*1, idx1, &err);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to create buffer!\n");
		return EXIT_FAILURE;
	}

	Index2 = clCreateBuffer(context, CL_MEM_USE_HOST_PTR, sizeof(int) * 1, idx2, &err);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to create buffer!\n");
		return EXIT_FAILURE;
	}

	Sigma = clCreateBuffer(context, CL_MEM_USE_HOST_PTR, sizeof(float)*num_sigma, sigma, &err);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to create buffer!\n");
		return EXIT_FAILURE;
	}
	
	//Set the kernel arguments

	// Downsample
	err = clSetKernelArg(kernel[0], 0, sizeof(cl_mem), &InputImage);
	err |= clSetKernelArg(kernel[0], 1, sizeof(cl_mem), &DownSampleImage);
	err |= clSetKernelArg(kernel[0], 2, sizeof(cl_mem), &sizex);
	err |= clSetKernelArg(kernel[0], 3, sizeof(cl_mem), &sizey);
	// Build Filter
	err |= clSetKernelArg(kernel[1], 0, sizeof(cl_mem), &filterWidth);
	err |= clSetKernelArg(kernel[1], 1, sizeof(cl_mem), &Sigma);
	err |= clSetKernelArg(kernel[1], 2, sizeof(cl_mem), &Filter);
	err |= clSetKernelArg(kernel[1], 3, sizeof(cl_mem), &FSum);
	// Gauss Blur
	err |= clSetKernelArg(kernel[2], 0, sizeof(cl_mem), &DownSampleImage);
	err |= clSetKernelArg(kernel[2], 1, sizeof(cl_mem), &GaussianBlurImage);
	err |= clSetKernelArg(kernel[2], 2, sizeof(cl_mem), &filterWidth);
	err |= clSetKernelArg(kernel[2], 3, sizeof(cl_mem), &Filter);
	// DOG
	err |= clSetKernelArg(kernel[3], 0, sizeof(cl_mem), &GaussianBlurImage);
	err |= clSetKernelArg(kernel[3], 1, sizeof(cl_mem), &DoGImage); 
	// Extrema
	err |= clSetKernelArg(kernel[4], 0, sizeof(cl_mem), &DoGImage);
	err |= clSetKernelArg(kernel[4], 1, sizeof(cl_mem), &Extrema);
	err |= clSetKernelArg(kernel[4], 2, sizeof(cl_mem), &Index1);
	// KeyPoint
	err |= clSetKernelArg(kernel[5], 0, sizeof(cl_mem), &DoGImage);
	err |= clSetKernelArg(kernel[5], 1, sizeof(cl_mem), &Extrema);
	err |= clSetKernelArg(kernel[5], 2, sizeof(cl_mem), &KeyPoint);
	err |= clSetKernelArg(kernel[5], 3, sizeof(cl_mem), &Index2);
	
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to set kernel argument!\n");
		return EXIT_FAILURE;
	}

	QueryPerformanceFrequency(&frequency);

	for (int i = 0; i < iteration; i++)
	{
		QueryPerformanceCounter(&start);

		//Execute the kernel

		//Downsample
		err = clEnqueueNDRangeKernel(command, kernel[0], dimention, NULL, globalworksize_DownSample, localworksize, 0, NULL, NULL);
		if (err != CL_SUCCESS)
		{
			printf("Error: Failed to enqueue NDRange Kernel - 0!\n");
			return EXIT_FAILURE;
		}

		err = clFinish(command);
		if (CL_SUCCESS != err)
		{
			printf("Error: clFinish Failed!\n");
			return EXIT_FAILURE;
		}
		

		//Building Gaussian Matrix
		err = clEnqueueNDRangeKernel(command, kernel[1], 2, NULL, filterworksize, filterlocalworksize, 0, NULL, NULL);
		if (err != CL_SUCCESS)
		{
			printf("Error: Failed to enqueue NDRange Kernel - 1!\n");
			return EXIT_FAILURE;
		}

		err = clFinish(command);
		if (CL_SUCCESS != err)
		{
			printf("Error: clFinish Failed!\n");
			return EXIT_FAILURE;
		}

		//Gaussian Blur
		err = clEnqueueNDRangeKernel(command, kernel[2], 3, NULL, globalworksize_DownSample, localworksize, 0, NULL, NULL);
		if (err != CL_SUCCESS)
		{
			printf("Error: Failed to enqueue NDRange Kernel - 2!\n");
			return EXIT_FAILURE;
		}

		err = clFinish(command);
		if (CL_SUCCESS != err)
		{
			printf("Error: clFinish Failed!\n");
			return EXIT_FAILURE;
		}
		
		//DoG Image
		err = clEnqueueNDRangeKernel(command, kernel[3], 3, NULL, globalworksize_DOG, localworksize, 0, NULL, NULL);
		if (err != CL_SUCCESS)
		{
			printf("Error: Failed to enqueue NDRange Kernel - 3!\n");
			return EXIT_FAILURE;
		}

		err = clFinish(command);
		if (CL_SUCCESS != err)
		{
			printf("Error: clFinish Failed!\n");
			return EXIT_FAILURE;
		}
		printf("DoG completed\n");
		
		
		//Extrema Points
		err = clEnqueueNDRangeKernel(command, kernel[4], 3, NULL, globalworksize_Extrema, localworksize, 0, NULL, NULL);
		if (err != CL_SUCCESS)
		{
			printf("Error: Failed to enqueue NDRange Kernel - 4!\n");
			return EXIT_FAILURE;
		}

		err = clFinish(command);
		if (CL_SUCCESS != err)
		{
			printf("Error: clFinish Failed!\n");
			return EXIT_FAILURE;
		}
		printf("Extrema completed");
		idx1 = (int *)clEnqueueMapBuffer(command, Index1, CL_TRUE, CL_MAP_READ, NULL, sizeof(int) * 1, 0, NULL, NULL, &err);
		extremapoints = (int *)clEnqueueMapBuffer(command,Extrema , CL_TRUE, CL_MAP_READ, NULL, sizeof(int) * (*idx1) * 3, 0, NULL, NULL, &err);

		//Key Points
		size_t globalworksize_KeyPoint[] = {*idx1 ,0,0 };
		err = clEnqueueNDRangeKernel(command, kernel[5], 1, NULL, globalworksize_KeyPoint, localworksize2, 0, NULL, NULL);
		if (err != CL_SUCCESS)
		{
			printf("Error: Failed to enqueue NDRange Kernel - 4!\n");
			return EXIT_FAILURE;
		}

		err = clFinish(command);
		if (CL_SUCCESS != err)
		{
			printf("Error: clFinish Failed!\n");
			return EXIT_FAILURE;
		}
		idx2 = (int *)clEnqueueMapBuffer(command, Index2, CL_TRUE, CL_MAP_READ, NULL, sizeof(int) * 1, 0, NULL, NULL, &err);
		key_points = (int *)clEnqueueMapBuffer(command,KeyPoint, CL_TRUE, CL_MAP_READ, NULL, sizeof(int) * (*idx2)*3, 0, NULL, NULL, &err);

		QueryPerformanceCounter(&end);

		time_series[i] = (float)(end.QuadPart - start.QuadPart) * 1000 / frequency.QuadPart;

	}

	printf("\n***** NDRange is finished ***** \n");

	time_ave = GetMean(time_series, iteration);
	time_standard_deviation = GetStandardDeviation(time_series, time_ave, iteration);
	printf("\tImageWidth: %d, ImageHeight: %d \n ", width, height);
	printf("\tAverage Execution Time: %f ms\n ", time_ave);
	printf("\tStandard Deviation of Time: %f\n ", time_standard_deviation);
	
	outdata = (unsigned char *)clEnqueueMapBuffer(command, GaussianBlurImage, CL_TRUE, CL_MAP_READ, NULL, sizeof(unsigned char)*width*height / sizex / sizey * (num_sigma-1), 0, NULL, NULL, &err);

	for (int i = 0; i < num_sigma; i++)
	{
		for (int j = 0; j < width*height / sizex / sizey;j++)
		{
			outimage[j] = outdata[i*width*height / sizex / sizey + j];
		}
		string fileName="";
		fileName += fileIdx[i];
		fileName += ".jpg";
		string output = outputfile;
		output.append(fileName);
		stbi_write_jpg(output.c_str(), width/sizex, height/sizey, 1, outimage, 100);
		/*
		string fileName2 = "dog_";
		fileName2 += fileIdx[i];
		fileName2 += ".jpg";
		output = outputfile;
		output.append(fileName2);
		stbi_write_jpg(output.c_str(), width/sizex, height/sizey, channels, outdata, 100);
		*/
	}
	outdata = (unsigned char *)clEnqueueMapBuffer(command, DoGImage, CL_TRUE, CL_MAP_READ, NULL, sizeof(unsigned char)*width*height / sizex / sizey * (num_sigma - 1), 0, NULL, NULL, &err);
	for (int i = 0; i < num_sigma - 1; i++)
	{
		for (int j = 0; j < width*height / sizex / sizey; j++)
		{
			outimage[j] = outdata[i*width*height / sizex / sizey + j];
		}
		
		string fileName2 = "dog_";
		fileName2 += fileIdx[i];
		fileName2 += ".jpg";
		string output = outputfile;
		output = outputfile;
		output.append(fileName2);
		stbi_write_jpg(output.c_str(), width/sizex, height/sizey, channels, outdata, 100);
		

	}
	for (int i = 0; i < *idx1;i++ )
	{
		cv::Point point;//keypoints
		point.x = extremapoints[i*3+0]*sizex;//x
		point.y = extremapoints[i*3+1]*sizey;//y
		//printf("(%d,%d)", point.x, point.y);
		cv::circle(image, point, 2, cv::Scalar(0, 255, 0));//green
	}

	for (int i = 0; i < *idx2; i++)
	{
		cv::Point point;//keypoints
		point.x = key_points[i * 3 + 0] * sizex;//x
		point.y = key_points[i * 3 + 1] * sizey;//y
		//printf("(%d,%d)", point.x, point.y);
		cv::circle( image, point, 4 , cv::Scalar(0,0,255));//red
	}
	
	cv::imshow("image",image);
	cv::waitKey(0);
	
	

	//Release Memory
	clReleaseMemObject(InputImage);
	clReleaseMemObject(DownSampleImage);
	clReleaseMemObject(GaussianBlurImage);
	clReleaseMemObject(DoGImage);

	clReleaseProgram(program);

	clReleaseKernel(kernel[0]);
	clReleaseKernel(kernel[1]);
	clReleaseKernel(kernel[2]);
	clReleaseKernel(kernel[3]);
	clReleaseKernel(kernel[4]);
	clReleaseKernel(kernel[5]);
	clReleaseCommandQueue(command);
	clReleaseContext(context);


	return 0;
}

float GetMean(float* number, int count)
{
	float sum = 0.0;
	for (int i = 0; i < count; i++)
	{
		sum += *number;
		number++;

	}
	return (float)(sum / count);
}

float GetStandardDeviation(float* number, float average, int count)
{
	float sum = 0.0;
	for (int i = 0; i < count; i++)
	{
		sum += (*number - average)*(*number - average);
		number++;
	}
	return (float)sqrt((sum / (count - 1)));
}


cl_image_desc img_description(bool is_img_array, int width, int height, int array_size)
{
	cl_image_desc desc;
	if (is_img_array) {
		desc.image_type = CL_MEM_OBJECT_IMAGE2D_ARRAY;
	}
	else
	{
		desc.image_type = CL_MEM_OBJECT_IMAGE2D;
	}
	desc.image_array_size = array_size;
	desc.image_width = width;
	desc.image_height = height;
	desc.image_depth = 0;

	desc.image_row_pitch = 0;
	desc.image_slice_pitch = 0;
	desc.num_mip_levels = 0;
	desc.num_samples = 0;
	desc.buffer = NULL;

	return desc;
}

