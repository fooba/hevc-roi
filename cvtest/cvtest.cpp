#include "stdafx.h"
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
using namespace cv;
using namespace std;

int main(){

	Mat im = imread("C:/Users/Public/Pictures/Sample Pictures/Koala.jpg");
	if (im.empty()){

		cout << "Cannot load image!" << endl;
		waitKey(0);
		return -1;

	}

	imshow("Image", im);
	waitKey(0);
	return 0;
}