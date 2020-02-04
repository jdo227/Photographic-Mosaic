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

int main(int argc, char * argv[]){
		// Set images
		string path = argv[1];
		Mat image;
		string filepath;
		double blue=0, green=0, red=0;
		ofstream txtout;
		txtout.open("BGR_mean.txt",ios::out|ios::trunc);
		//txtout <<"FileName"<<" "<<"Blue" << " " << "Green" << " " << "Red" <<endl;
		for (const auto& entry : fs::directory_iterator(path)) {
			const auto filenameStr = entry.path().filename().string();
			string ext = entry.path().extension();
			if (ext == ".png") {
				filepath = argv[1] + filenameStr;
				image = imread(filepath, IMREAD_COLOR);// read image file
				if(image.empty())                      // Check for invalid input
					{
						cout <<  "Could not open or find the image" << std::endl ;
						return -1;
					}			
				meanIntensity(image, blue, green, red);
				txtout << filepath <<"\t" << blue << "\t" << green << "\t" << red << endl;
			}
		}
		txtout.close();

		//Create Test Data Sets
		bool create_data = 0;
		if (create_data) {
			int a = 64;
			int nb, ng, nr;
			nb = ng = nr = 20;
			double v[nb];
			linspace(0, 255.0, nb, v);
			Mat im;
			vector<int> compression_params;
			compression_params.push_back(IMWRITE_PNG_COMPRESSION);
			compression_params.push_back(9);
			char buffer[50];
			char im_dir[] = "bgr_set";
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

		// Read textfile
		FILE* ptr = fopen("BGR_mean.txt", "r");
		char buf[100];
		double bl, gr, re;
		rewind(ptr);
		vector<string> path_list;
		int lSize = 0;
		char ch;
		while ((ch = fgetc(ptr)) != EOF) {
			lSize++;
		}
		rewind(ptr);
		Mat bgr_list(lSize,3,CV_64F);
		int i = 0;
		while ((ch = fgetc(ptr)) != EOF) {
			ungetc(ch, ptr);
			fscanf(ptr, "%s %lf %lf %lf\n", buf, &bl, &gr, &re);
			path_list.push_back(buf);
			bgr_list.at<double>(i,0) = bl;
			bgr_list.at<double>(i,1) = gr;
			bgr_list.at<double>(i,2) = re;
			i++;
		}
		fclose(ptr);

		 //Read Original Image
		filepath = "main.jpg";
		image = imread(filepath, IMREAD_COLOR);// read image file
		int py_r, px_c,N_r,N_c;
		py_r = px_c = 5;
		N_r = floor(image.rows / py_r);
		N_c = floor(image.cols / px_c);
		//cout << N_r << "\t" << N_c << endl;
		int res = 64;
		Mat canvas(N_r*res, N_c*res, CV_8UC3);
		//Process each block
		int idx = 0;
		for (int i = 0; i < N_r; i++) {
			for (int j = 0; j < N_c; j++) {
				int Tp_x = j * res;			
				int Tp_y = i * res;
				//cout << Tp_x << "\t" << Tp_y << endl;
				Mat sub_image = image(Rect(j*px_c, i*py_r, px_c, py_r));
				meanIntensity(sub_image, blue, green, red);
				idx = minDist(bgr_list, blue, green, red, path_list);
				sub_image = imread(path_list[idx], IMREAD_COLOR);
				Mat aux = canvas.colRange(Tp_x, Tp_x + res).rowRange(Tp_y, Tp_y + res);
				sub_image.copyTo(aux);
			}
		}
		vector<int> compression_params;
		compression_params.push_back(IMWRITE_PNG_COMPRESSION);
		compression_params.push_back(9);
		imwrite("mosaic.png", canvas, compression_params);
		return 0;
}

int minDist(Mat bgr_list, double blue, double green, double red, vector<string> path_list) {
	Mat im_selected;
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
			Vec3b intensity = M.at<Vec3b>(i, j);
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