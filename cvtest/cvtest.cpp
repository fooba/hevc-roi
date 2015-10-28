#include "stdafx.h"
#include "stdio.h"
#include <iostream>
#include <ctime>

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

//Da Kamera nich immer direkt verfügbar ist
int wait4Cam = 2;

typedef struct{
	int frameno;
	std::vector<Rect> facesinframe;
	std::vector<Mat>  faces;
}facesfromframe;

#define SHOW_DETECTED_FACES //comment for no output of the frames in face detection
#define WITH_FACE_RECTANGLE //comment for no Rectangle in ouput file where Face is detected

const std::string inputfilename = "C:\\testshort.mp4";
const std::string outputfilename = "C:\\out.mkv";
const std::string backgroundname   = "C:\\background.mkv";
const int fourcc_output_codec = -1;// CV_FOURCC('P', 'I', 'M', '1');
std::vector<facesfromframe> facesVideo;

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
* @brief fuegt die erkannten Faces und in den Hintergrund ein und enkodiert erneut
*/
int encodeToOne(void);

/**
* @brief detectFaces detektiert Gesichter im uebergebenen frame.
* @param frame Frame in dem nach gesichtern gesucht werden soll
* @return vektor aller gefundenen Gesichter
*/
vector<Rect> detectFaces(Mat frame);


int main(){
	//Zeitmessung
	clock_t start, afterFacedet; 
	float elapsed, elapsedafterfacedet;

	start = clock();

	int ret = 0;
	//ret = startCam();
	ret = startVid();
	if (ret != 0){
		cout << "Error " << ret << " ... closing now" << endl;
		waitKey(0);
	}
	else{
		elapsedafterfacedet = (float)(clock() - start) / CLOCKS_PER_SEC;
		start = clock(); //reset clock

		ret = encodeToOne();
		if (ret != 0){
			cout << "Error " << ret << " ... closing now" << endl;
			waitKey(0);
		}
	}
	elapsed = (float)(clock() - start) / CLOCKS_PER_SEC;
	cout << "Time for Facedetection: " << elapsedafterfacedet << " Time for Encoding together: " << elapsed << " Sum: " << elapsedafterfacedet + elapsed << endl;

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
	cout << "\t Load input, detect faces and encode Background:\n" << endl;

	VideoCapture capture;
	Mat frame;

	//-- 1. Load the cascades
	if (!face_cascade.load(face_cascade_name)){ cout << "--(!)Error loading: " << face_cascade_name << "\n" << endl; return -1; };

	//-- 2. Read the video stream
	//capture.open("D:/Users/Julian.T - mobile/..Studium/masys/projekt/samples/SpreedMovie-640x360.mkv");
	capture.open(inputfilename);
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
	if (!outputVideo.open(backgroundname, fourcc_output_codec, fps, S, true)){ //out
	//if (!outputVideo.open("C:/outshort.yuv", CV_FOURCC('X', '2', '6', '4'), fps, S, true)){ //out
	//if (! outputVideo.open("C:/outshort.yuv", CV_FOURCC('H', 'E', 'V', 'C'), fps, S, true)){ //out
	//if (! outputVideo.open("C:/outshort.yuv", CV_FOURCC('P', 'I', 'M', '1'), fps, S, true)){ //out //works but mov
	//if (! outputVideo.open("C:/outshort.yuv", CV_FOURCC('Y', 'V', '1', '2'), fps, S, true)){ //out
		cout << "Could not create Video writer...." << endl;
		return -1;
	}


	if (!outputVideo.isOpened())
	{
	cout << "Could not open the output video to write... " << endl;
	return -1;
	}
	
	Mat roi;
	while (capture.read(frame)){
		cout << "Frame " << framecounter++ << endl;

		//if (framecounter >= 50) break; //TODO ONLY READ first 10 FRAMES

		if (!frame.empty()){
			std::vector<Rect> faces = detectFaces(frame);
			for (size_t i = 0; i < faces.size(); i++) //"Male" Rechteck um jedes Gesicht
			{

#ifdef WITH_FACE_RECTANGLE
				Point p1(faces[i].x, faces[i].y); //Oben links vom Gesicht
				Point p2(faces[i].x + faces[i].width, faces[i].y + faces[i].height); //Unten rechts vom Gesicht
				rectangle(frame, p1, p2, Scalar(255, 0, 255), 4, 8, 0); //Rechteck ums Gesicht
#endif

				//Region of Interest = gesicht
				roi = frame(faces.at(i));
#ifdef SHOW_DETECTED_FACES
				imshow("roi", roi);
				waitKey(1);
#endif
				//fuer speicherung
				vector<Mat> dummyFaces;
				dummyFaces.push_back(roi);
				facesfromframe dummy;
				dummy.frameno = framecounter;
				dummy.facesinframe = faces;
				dummy.faces = dummyFaces;
				facesVideo.push_back(dummy);
			}
#ifdef SHOW_DETECTED_FACES			
			imshow(window_name, frame); //Zeige frame //TODO auskommentiert schneller??
			waitKey(1);
#endif
			outputVideo << frame; //Schreibe frame in output video
		}
		else{
			break;
		}
	}
	cout << "Facerecognition... ready" << endl;
	return 0;
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

int encodeToOne(void){
	cout << "\t Put faces in ecoded Background and encode both together " << endl; 
	
	//in
	VideoCapture inputFile;
	inputFile.open(backgroundname);
	if (! inputFile.isOpened() ){
		cout << "Can't open tempory Video File..." << endl;
		return -1;
	}
	Mat frame;

	//Define Params for output
	Size S = Size((int)inputFile.get(CV_CAP_PROP_FRAME_WIDTH),    // Acquire input size
		(int)inputFile.get(CV_CAP_PROP_FRAME_HEIGHT));
	double fps = inputFile.get(CV_CAP_PROP_FPS); //fps

	//out
	VideoWriter outputFile;
	if (! outputFile.open(outputfilename, fourcc_output_codec, fps, S, true) ){ //out
		cout << "Can' creat VideoWriter for output...";
	}
	if (! outputFile.isOpened() ){
		cout << "Can't open Video for writing out" << endl;
		return -1;
	}

	//Put together
	int inputframeno=0;
	while (inputFile.read(frame)){ //read
		cout << "frame " << inputframeno++ << endl;
		// put face in frame when needed
		facesfromframe* dummy = &facesVideo.at(0);
		if (dummy->frameno == inputframeno){ //Gesicht in frame muss hinzugefuegt werden
			for (int i = 0; i < dummy->faces.size(); i++){ //Fuer alle Faces im Frame
				Point ol(dummy->facesinframe.at(i).x, dummy->facesinframe.at(i).y); //Oben links vom Gesicht
				Point ur(dummy->facesinframe.at(i).x + dummy->facesinframe.at(i).width, dummy->facesinframe.at(i).y + dummy->facesinframe.at(i).height); //Unten rechts vom Gesicht

				//Go through frame and change to not encoded roi
				int indexx = 0, indexy = 0;
				for (int y = ol.y; y <= ur.y; y++){ //go through lines
					for (int x = ol.x; x <= ur.x; x++){ //go through elements
						frame.at<double>(Point(x,y)) = dummy->faces.at(i).at<double>(Point(indexx, indexy)); //TODO KACKT AB
						indexx++; indexy;
					}
				}
			}
		}
		imshow("out",frame);
		waitKey(1);
		outputFile << frame; //write out
	}


	cout << "Ready :D" << endl;
	return 0;
}