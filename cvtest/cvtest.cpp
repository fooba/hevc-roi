#include "stdafx.h"
#include "stdio.h"
#include <iostream>

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

int wait4Cam = 2;
String test;
cv::String face_cascade_name = "haarcascade_frontalface_default.xml";
CascadeClassifier face_cascade;
string window_name = "Face detection & ROI encoding using HEVC";

//RNG rng(12345);
char key;

/**
* @brief start an openCV session to the webcam and detect faces in stream
*        quit with escape-btn
*/
void startCV(void);


/**
* @brief detectFaces detektiert Gesichter im uebergebenen frame.
* @param frame Frame in dem nach gesichtern gesucht werden soll
* @return vektor aller gefundenen Gesichter
*/
vector<Rect> detectFaces(Mat frame);


int main(){

	//Mat im = imread("C:/Users/Public/Pictures/Sample Pictures/Koala.jpg");
	//if (im.empty()){

	//	cout << "Cannot load image!" << endl;
	//	waitKey(0);
	//	return -1;

	//}

	//imshow("Image", im);

	startCV();
	waitKey(0);
	return 0;
}


void startCV(void){
	cout << "clicked"<<endl;

	CvCapture* capture;
	Mat frame;

	//-- 1. Load the cascades
	if (!face_cascade.load(face_cascade_name)){ cout << "--(!)Error loading: " << face_cascade_name << "\n" << endl; return; };

	//-- 2. Read the video stream
	capture = cvCaptureFromCAM(CV_CAP_ANY); //Webcam

	while (1){
		if (capture){
			frame = cvQueryFrame(capture); //Create image frames from capture
			if (!frame.empty()){
				std::vector<Rect> faces = detectFaces(frame);
				for (size_t i = 0; i < faces.size(); i++) //"Male" Rechteck um jedes Gesicht
				{
					Point p1(faces[i].x, faces[i].y); //Oben links vom Gesicht
					Point p2(faces[i].x + faces[i].width, faces[i].y + faces[i].height); //Unten rechts vom Gesicht
					rectangle(frame, p1, p2, Scalar(255, 0, 255), 4, 8, 0); //Rechteck ums Gesicht
				}
				imshow(window_name, frame); //Zeige frame
			}
			else{
				wait4Cam--;
				if (wait4Cam <= 0)
					printf(" --(!) No captured frame -- :( !");
			}
		}
		else cout << "keine Webcam ";
		key = cvWaitKey(10);     //Capture Keyboard stroke
		if (char(key) == 27){
			break;      //If you hit ESC key loop will break.
		}
	}

	cvReleaseCapture(&capture); //Release capture.
}

vector<Rect> detectFaces(Mat frame){
	std::vector<Rect> faces;
	Mat frame_gray;

	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	//-- Detect faces
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

	return faces;
}