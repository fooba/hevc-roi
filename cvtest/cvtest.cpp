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
void startCam(void);

/**
* @brief start an openCV session to the webcam and detect faces in stream
*        quit with escape-btn
*/
int startVid(void);

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

	//TODO Aenderung bzgl funktionsaufruf
	int ret = 0;
	//ret = startCam();
	ret = startVid();
	if (! ret) 
		waitKey(0);
	cin.ignore();
	return ret;
}

void startCam(void){
	cout << "clicked" << endl;

	VideoCapture capture;
	Mat frame;

	//-- 1. Load the cascades
	if (!face_cascade.load(face_cascade_name)){ cout << "--(!)Error loading: " << face_cascade_name << "\n" << endl; return; };

	//-- 2. Read the video stream
	capture.open(0); 

	
	while (1){
		if (capture.isOpened()){
			
			capture.read(frame);
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

//	cvReleaseCapture(&capture); //Release capture.
}

int startVid(void){
	cout << "clicked" << endl;

	VideoCapture capture;
	Mat frame;

	//-- 1. Load the cascades
	if (!face_cascade.load(face_cascade_name)){ cout << "--(!)Error loading: " << face_cascade_name << "\n" << endl; return -1; };

	//-- 2. Read the video stream
	capture.open("C:/test3.mov");
	int framecounter = 0;

	if (!capture.isOpened()){
		cout << "Can't open Video..." << endl;
		return -1;
	}

	//Groesse des Input Vids
	Size S = Size((int)capture.get(CV_CAP_PROP_FRAME_WIDTH),    // Acquire input size
		(int)capture.get(CV_CAP_PROP_FRAME_HEIGHT));
	double fps = capture.get(CV_CAP_PROP_FPS); //fps

	//Video Output
	VideoWriter outputVideo;
	if (!outputVideo.open("C:/outshort.yuv", CV_FOURCC('X', '2', '6', '4'), fps, S, true)){ //out
	//if (! outputVideo.open("C:/outshort.yuv", CV_FOURCC('H', 'E', 'V', 'C'), fps, S, true)){ //out
	//if (! outputVideo.open("C:/outshort.yuv", CV_FOURCC('P', 'I', 'M', '1'), fps, S, true)){ //out //works but mov
	//if (! outputVideo.open("C:/outshort.yuv", CV_FOURCC('Y', 'V', '1', '2'), fps, S, true)){ //out
		cout << "Could not create Video writer....";
		return -1;
	}

	if (!outputVideo.isOpened())
	{
		cout << "Could not open the output video to write... " << endl;
		return -1;
	}

	while (capture.read(frame)){
			cout << "Frame " << framecounter++ << endl;
			
			if (!frame.empty()){
				std::vector<Rect> faces = detectFaces(frame);
				for (size_t i = 0; i < faces.size(); i++) //"Male" Rechteck um jedes Gesicht
				{
					Point p1(faces[i].x, faces[i].y); //Oben links vom Gesicht
					Point p2(faces[i].x + faces[i].width, faces[i].y + faces[i].height); //Unten rechts vom Gesicht
					rectangle(frame, p1, p2, Scalar(255, 0, 255), 4, 8, 0); //Rechteck ums Gesicht
				}
				//imshow(window_name, frame); //Zeige frame //TODO auskommentiert schneller??
				outputVideo << frame; //Schreibe frame in output video
			}
			else{
					//break;
			}
	}

	return 0;
	//	cvReleaseCapture(&capture); //Release capture.
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