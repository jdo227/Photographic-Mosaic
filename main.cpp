// Project: Photographic Mosaic
#include <iostream>
#include <experimental/filesystem>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <string>
#include <tuple>

using namespace std;
using namespace cv;
namespace fs = std::experimental::filesystem; 

void meanIntensity(Mat M, double * blue, double * green, double * red){
	for(int i=0;i<M.rows;i++)
	{
		for (int j = 0; j < M.cols; j++)
		{
			Vec3b intensity = M.at<Vec3b>(i, j);
			*blue += intensity.val[0];
			*green += intensity.val[1];
			*red += intensity.val[2];
		}
	}
	int N = M.rows * M.cols;
	*blue /= N;
	*green /= N;
	*red /= N;
}

int main(int argc, char * argv[]){
		// Set images
		string path = argv[1];
		Mat image;
		string filepath;
		double blue=0, green=0, red=0;
		for (const auto& entry : fs::directory_iterator(path)) {
			const auto filenameStr = entry.path().filename().string();
			string ext = entry.path().extension();
			if (ext == ".JPG") {
				filepath = argv[1] + filenameStr;
				image = imread(filepath, IMREAD_COLOR);// read image file
				meanIntensity(image, &blue, &green, &red);
				cout << blue <<" "<< green <<" "<< red << endl;
				if(image.empty())                      // Check for invalid input
					{
						cout <<  "Could not open or find the image" << std::endl ;
						return -1;
					}				
			}
		}
		return 0;
}