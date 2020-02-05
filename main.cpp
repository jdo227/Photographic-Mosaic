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

using namespace std;
using namespace cv;
namespace fs = std::experimental::filesystem; 

void meanIntensity(Mat M, double& blue, double& green, double& red);
void linspace(double start, double end, int N, double vec[]);
int minDist(Mat bgr_list, double blue, double green, double red, vector<string> path_list);
void makeList(bool yn, string filename, string path);
void makeTestData(bool yn, char im_dir[]);
Mat readList(char bgrfile[], vector<string>& path_list);

// examples: ./Mosaic bar_set/ main.jpg 50
int main(int argc, char * argv[]){

	// Create mean BGR value list
	string path = argv[1];
	makeList(0, "BGR_mean.txt", path);

	//Create Test Data Sets
	char im_dir[] = "bgr_set";
	makeTestData(0, im_dir);

	// Read RGB data set value list
	char bgrfile[] = "BGR_mean.txt";
	vector<string> path_list;
	Mat bgr_list = readList(bgrfile,path_list);

	//Read Original Image
	string filepath = argv[2];
	double blue = 0, green = 0, red = 0;
	Mat image = imread(filepath, IMREAD_COLOR);// read image file
	int py_r, px_c,N_r,N_c;
	py_r = px_c = atoi(argv[3]);
	N_r = floor(image.rows / py_r);
	N_c = floor(image.cols / px_c);
	int res = 64;// lateral size of replacement sub_image
	Mat canvas(N_r*res, N_c*res, CV_8UC3);
	//Process each block
	int idx = 0;
	for (int i = 0; i < N_r; i++) {
		for (int j = 0; j < N_c; j++) {
			int Tp_x = j * res;			
			int Tp_y = i * res;
			Mat sub_image = image(Rect(j*px_c, i*py_r, px_c, py_r));
			meanIntensity(sub_image, blue, green, red);
			idx = minDist(bgr_list, blue, green, red, path_list);
			sub_image = imread(path_list[idx], IMREAD_COLOR);
			Mat aux = canvas.colRange(Tp_x, Tp_x + res).rowRange(Tp_y, Tp_y + res);
			sub_image.copyTo(aux);
		}
	}
	// Write Mosaic
	vector<int> compression_params;
	compression_params.push_back(IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(9);
	imwrite("mosaic.png", canvas, compression_params);
	return 0;
}

// Functions
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
void makeTestData(bool yn, char im_dir[]) {
	if (yn) {
		int a = 64;// lateral size
		int nb, ng, nr; //number of color in a channel
		nb = ng = nr = 20;
		double v[nb];
		linspace(0, 255.0, nb, v);
		Mat im;
		vector<int> compression_params;
		compression_params.push_back(IMWRITE_PNG_COMPRESSION);
		compression_params.push_back(9);
		char buffer[50];
		//char im_dir[] = *foldername;
		for (int i = 0; i < nb; i++) {
			Mat ch_B = Mat::ones(a, a, CV_8U) * v[i];
			for (int j = 0; j < ng; j++) {
				Mat ch_G = Mat::ones(a, a, CV_8U) * v[j];
				for (int k = 0; k < nr; k++) {
					Mat ch_R = Mat::ones(a, a, CV_8U) * v[k];
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
			string ext = entry.path().extension();
			if (ext == ".png") {
				filepath = path + filenameStr;
				image = imread(filepath, IMREAD_COLOR);// read image file
				meanIntensity(image, blue, green, red);
				txtout << filepath << "\t" << blue << "\t" << green << "\t" << red << endl;
			}
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