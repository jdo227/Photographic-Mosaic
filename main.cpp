// Project: Photographic Mosaic
#include <iostream>
#include <experimental/filesystem>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <string>
#include <tuple>
#include <fstream>
#include <stdio.h>
#include <vector>
#include <cstdio>
#include <math.h>
#include <typeinfo>
#include <iostream>
#include <vector>
#include <stdio.h>      
#include <opencv2/opencv.hpp>    
#include "opencv2/imgproc.hpp"
#include "opencv2/core/core_c.h"

using namespace std;
using namespace cv;
namespace fs = std::experimental::filesystem; 

void meanIntensity(Mat M, double& blue, double& green, double& red);
void linspace(double start, double end, int N, double vec[]);
int minDist(Mat bgr_list, double blue, double green, double red, vector<string> path_list);
void makeList(bool yn, string filename, string path);
void makeTestData(bool yn, char im_dir[], int rows, int cols, int ncolors);
Mat readList(char bgrfile[], vector<string>& path_list);
void resize(const Mat sub_img, Mat &img_crop, int rows, int cols) {
	img_crop = Mat::zeros(rows, cols, CV_8UC3);
	int scale = sub_img.rows / img_crop.rows;
	for (int i = 0; i < img_crop.rows; i++) {
		for (int j = 0; j < img_crop.cols; j++) {
			img_crop.at<Vec3b>(i, j) = sub_img.at<Vec3b>(i * scale, j * scale);
		}
	}
}
void similarity(Mat imgA, Mat imgB, float& emd);

// examples: make && ./Mosaic lib_img_crop/ main.jpg 10 lib_img/ lib_img_crop/  
int main(int argc, char* argv[]) {
	int rows = 50;//pixels in row of grid
	int cols = 50;//pixels in column of grid

	// Prepare the image data sets
	vector<int> compression_params;
	compression_params.push_back(IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(9);
	string imgsets = argv[4];
	char* imgcrops = argv[5];
	string filepath;
	string filename;
	double scale;
	Mat image, img_crop;

	//create resized image pool
	int count = 0;
	vector<string> img_crop_list;
	char buf[100];
	Mat sub_img;
	
	for (const auto& entry : fs::directory_iterator(imgsets)) {
		const auto filenameStr = entry.path().filename().string();
		filepath = imgsets + filenameStr;
		image = imread(filepath, IMREAD_COLOR);// read image file
		if (image.rows > image.cols) {
			transpose(image, image);
		}
		scale = min(floor(image.rows / rows), floor(image.cols / cols));
		sub_img = image.colRange(0, scale * cols).rowRange(0, scale * rows);
		resize(sub_img, img_crop, rows, cols);
		filename = imgcrops + filenameStr;
		imwrite(filename, img_crop, compression_params);
		img_crop_list.push_back(filename);
		printf("Cropping: %d\n", count);
		count++;
	}
	printf("Cropping finished.\n");
	//Read Original Image
	string target = argv[2];
	double blue = 0, green = 0, red = 0;
	image = imread(target, IMREAD_COLOR);// read image file
	int py_r, px_c,N_r,N_c;
	py_r = px_c = atoi(argv[3]);
	N_r = floor(image.rows / py_r)-1;
	N_c = floor(image.cols / px_c)-1;
	Mat canvas= Mat::zeros(N_r*rows, N_c*cols, CV_8UC3);

	//Process each block
	int idx = 0, Tp_x, Tp_y;
	float emd, min_emd = 1;//emd 0 is best matching
	Mat img_crops, sub_image, aux;
	for (int i = 0; i < N_r; i++) {
		for (int j = 0; j < N_c; j++) {
			min_emd = 1;
			Tp_x = j * cols;			
			Tp_y = i * rows;
			sub_image = image(Rect(j*px_c, i*py_r, px_c, py_r));
			for (int k = 0; k < count; k++) {
				filepath = img_crop_list[k];
				img_crops = imread(filepath, IMREAD_COLOR);// read image file
				similarity(sub_image, img_crops, emd);
				if (emd < min_emd) {
					min_emd = emd;
					idx = k;
				}
			}
			img_crops = imread(img_crop_list[idx], IMREAD_COLOR);
			aux = canvas.colRange(Tp_x, Tp_x + cols).rowRange(Tp_y, Tp_y + rows);
			img_crops.copyTo(aux);
			printf("Row:%d/%d, Col:%d/%d\n", i, N_r - 1, j, N_c - 1);
		}
	}
	// Write Mosaic
	imwrite("Mosaic.png", canvas, compression_params);
	return 0;
}

// Functions
void similarity(Mat imgA, Mat imgB, float& emd) {
	//variables preparing      
	int hbins = 30, sbins = 32;
	int channels[] = { 0,  1 };
	int histSize[] = { hbins, sbins };
	float hranges[] = { 0, 180 };
	float sranges[] = { 0, 255 };
	const float* ranges[] = { hranges, sranges };

	Mat patch_HSV;
	MatND HistA, HistB;

	//cal histogram & normalization   
	cvtColor(imgA, patch_HSV, COLOR_BGR2HSV);
	calcHist(&patch_HSV, 1, channels, Mat(), // do not use mask   
		HistA, 2, histSize, ranges,
		true, // the histogram is uniform   
		false);
	normalize(HistA, HistA, 0, 1, CV_MINMAX);

	cvtColor(imgB, patch_HSV, COLOR_BGR2HSV);
	calcHist(&patch_HSV, 1, channels, Mat(),// do not use mask   
		HistB, 2, histSize, ranges,
		true, // the histogram is uniform   
		false);
	normalize(HistB, HistB, 0, 1, CV_MINMAX);

	//compare histogram      
	int numrows = hbins * sbins;

	//make signature
	Mat sig1(numrows, 3, CV_32FC1);
	Mat sig2(numrows, 3, CV_32FC1);

	//fill value into signature
	float binval;
	for (int h = 0; h < hbins; h++)
	{
		for (int s = 0; s < sbins; s++)
		{
			binval = HistA.at< float>(h, s);
			sig1.at< float>(h * sbins + s, 0) = binval;
			sig1.at< float>(h * sbins + s, 1) = h;
			sig1.at< float>(h * sbins + s, 2) = s;

			binval = HistB.at< float>(h, s);
			sig2.at< float>(h * sbins + s, 0) = binval;
			sig2.at< float>(h * sbins + s, 1) = h;
			sig2.at< float>(h * sbins + s, 2) = s;
		}
	}

	//compare similarity of 2images using emd.
	emd = cv::EMD(sig1, sig2, DIST_L2); //emd 0 is best matching. 
}
Mat readList(char bgrfile[], vector<string>& path_list) {
	FILE* ptr = fopen(bgrfile, "r");
	char buf[100];
	double bl, gr, re;

	int lSize = 0;
	char ch;
	while ((ch = fgetc(ptr)) != EOF) {
		lSize++;
	}
	rewind(ptr);
	Mat bgr_list(lSize, 3, CV_64F);
	int i = 0;
	while ((ch = fgetc(ptr)) != EOF) {
		ungetc(ch, ptr);
		fscanf(ptr, "%s %lf %lf %lf\n", buf, &bl, &gr, &re);
		path_list.push_back(buf);
		bgr_list.at<double>(i, 0) = bl;
		bgr_list.at<double>(i, 1) = gr;
		bgr_list.at<double>(i, 2) = re;
		i++;
	}
	fclose(ptr);
	return bgr_list;
}
void makeTestData(bool yn, char im_dir[],int rows, int cols, int ncolors) {
	if (yn) {
		int nb, ng, nr; //number of color in a channel
		nb = ng = nr = ncolors;
		double v[nb];
		linspace(0, 255.0, nb, v);
		Mat im;
		vector<int> compression_params;
		compression_params.push_back(IMWRITE_PNG_COMPRESSION);
		compression_params.push_back(9);
		char buffer[50];
		for (int i = 0; i < nb; i++) {
			Mat ch_B = Mat::ones(rows, cols, CV_8U) * v[i];
			for (int j = 0; j < ng; j++) {
				Mat ch_G = Mat::ones(rows, cols, CV_8U) * v[j];
				for (int k = 0; k < nr; k++) {
					Mat ch_R = Mat::ones(rows, cols, CV_8U) * v[k];
					vector<Mat> channels{ ch_B,ch_G,ch_R };
					merge(channels, im);
					sprintf(buffer, "%s/img_%d_%d_%d.png", im_dir, i, j, k);
					imwrite(buffer, im, compression_params);
				}
			}
		}
	}
}
void makeList(bool yn,string filename,string path) {
	if (yn) {
		Mat image;
		string filepath;
		double blue = 0, green = 0, red = 0;
		ofstream txtout;
		txtout.open(filename, ios::out | ios::trunc);
		for (const auto& entry : fs::directory_iterator(path)) {
			const auto filenameStr = entry.path().filename().string();
			filepath = path + filenameStr;
			image = imread(filepath, IMREAD_COLOR);// read image file
			meanIntensity(image, blue, green, red);
			txtout << filepath << "\t" << blue << "\t" << green << "\t" << red << endl;
		}
		txtout.close();
	}
}
int minDist(Mat bgr_list, double blue, double green, double red, vector<string> path_list) {
	double mindist = 1e10;
	double dist = 0;
	int idx = 0;
	for (int i = 0; i < bgr_list.rows; i++) {
		dist = pow(bgr_list.at<double>(i, 0) - blue, 2) + pow(bgr_list.at<double>(i, 1) - green, 2) + pow(bgr_list.at<double>(i, 2) - red, 2);
		if (dist < mindist) {
			mindist = dist;
			idx = i;
		}
	}
	return idx;
}
void meanIntensity(Mat M, double& blue, double& green, double& red) {
	blue = green = red = 0;
	for (int i = 0; i < M.rows; i++)
	{
		for (int j = 0; j < M.cols; j++)
		{
			Vec3b intensity = M.at<Vec3b>(i, j);//imread gets uchar
			blue += intensity.val[0];
			green += intensity.val[1];
			red += intensity.val[2];
		}
	}
	int N = M.rows * M.cols;
	blue /= N;
	green /= N;
	red /= N;
}
void linspace(double start, double end, int N, double vec[]) {
	double step = (end - start) / (N - 1);
	for (int i = 0; i < N; i++) {
		vec[i] = step * i;
	}
}