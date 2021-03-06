#include "stdafx.h"
#include <stdio.h>
#include <iostream>
#include <ctime>
#include <stdlib.h>
#include <assert.h>

//OpenCV
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv\cv.h> //older cv things

//Converts YUV -> CV
#include "yuv.h"

//FFMPEG (C-compiled)
extern "C" {
	//Library containing decoders and encoders for audio/video codecs.
#include <libavcodec/avcodec.h>
	//Library containing demuxers and muxers for multimedia container formats.
#include <libavformat/avformat.h>
	//Library performing highly optimized image scaling and color space/pixel format conversion operations.
#include <libswscale/swscale.h>
#include <libavutil\mem.h>
}


//x265
#include <x265.h>

using namespace cv;
using namespace std;

//Da Kamera nich immer direkt verf�gbar ist
int wait4Cam = 2;

typedef struct{
	int frameno;
	std::vector<Rect> facesinframe;
	std::vector<Mat>  faces;
}facesfromframe;

typedef struct{
	int frameid;
	std::vector<Rect> rect;
}faceInFrame;

//#define SHOW_DETECTED_FACES //comment for no output of the frames in face detection
//#define WITH_FACE_RECTANGLE //comment for no Rectangle in ouput file where Face is detected
#define GROP_FACE_SIZE 0.1 //Face must be greater than GROP_FACE_SIZE*InputVideo.Width

/**
  * Definitions for Bitrates
*/
#define BITRATE_ONE       "500k"
#define BITRATE_FACE	  "600k"
#define BITRATE_SURROUND  "400k"

/**
  * Definitionen der groessen des zu extrahierenden Gesichtsvideos
*/
#define FACES_VIDEO_WIDTH   1280	
#define FACES_VIDEO_HEIGHT  720
#define FACES_PRINT_MAX

/**
  * Definitionen damit Gesicht groesser ausgeschnitten wird
*/
#define FACE_BIG_TOP	0.3
#define FACE_BIG_BTM	0.2
#define FACE_BIG_LEFT	0.05
#define FACE_BIG_RIGHT	0.05


const std::string inputfilename    = "C:\\tagesschau_720_cut.mp4";
const std::string roistr		   = "C:\\faces.yuv";
const std::string videostr		   = "C:\\video.yuv";
const std::string backstr		   = "C:\\background.yuv";
const std::string outvideostr	   = "C:\\video.mkv";
const std::string faceoutstr	   = "C:\\faces.mkv";
const std::string backoutstr	   = "C:\\background.mkv";

const std::string yuv_encoded_one  = "C:\\yuv_one.yuv";
const std::string yuv_encoded_back = "C:\\yuv_back.yuv";
const std::string yuv_encoded_face = "C:\\yuv_face.yuv";

const std::string yuv_together	   = "C:\\together.yuv";
const std::string hevc_together    = "C:\\together.mkv";


const int fourcc_output_codec =  CV_FOURCC('I', '4', '2', '0');
std::vector<facesfromframe> facesVideo;
static int framecounter=0;
VideoWriter outputVideo;
VideoWriter faceVideo;
VideoWriter backVideo;
VideoWriter outtogether;
Size S;
double fps;

//Std Framerate und Groesse, aenderbar auf eingangsvideo oder aehnlichem
std::string fpsstr = "29";
std::string sizestr = "1280x720";

String test;
//cv::String face_cascade_name = "haarcascade_frontalface_default.xml";
cv::String face_cascade_name = "haarcascade_frontalface_alt.xml";
cv::String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
string window_name = "Face detection & ROI encoding using HEVC";
Mat cvFrameContext;
struct SwsContext *pVImgConvertCtx;

//fmpeg
int videoStream; //Variable to store the index of the Video Stream.
int frameFinished; //Variable to determine if the frame data was fetched properly
AVFormatContext *pVFormatCtx;
AVCodecContext *pVCodecCtx;
AVCodec *pVCodec;
AVFrame *pVFrame;
AVFrame *pVFrameBGR;
uint8_t *bufferBGR;
AVPacket pVPacket;
Mat inFrames;
VideoCapture capture;
std::vector<faceInFrame> facesinFrame;

//RNG rng(12345);
char key; 

/**
  * Zeigt mehrere Imgs in einem Fenster an
*/
extern void cvShowManyImages(char* title, int nArgs, ...);

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
* Open a Video with FFMPEG
*/
bool ffmpegopenVid(const char * filename);

/**
* @brief detectFaces detektiert Gesichter im uebergebenen frame.
* @param frame Frame in dem nach gesichtern gesucht werden soll
* @return vektor aller gefundenen Gesichter
*/
vector<Rect> detectFaces(Mat frame);

/**
* @brief  Verarbeitet ein neues Frame des Input Videos
* @return neuer Frame verfuegbar??
*/
bool querryFrame(void);

/**
 * @brief encodiert als HEVC mit Command-line Prompt
*/
int encodeCmd(std::string filename, std::string outfilename, std::string bitrate, std::string vsize);

/**
  * @brief decode inFile in outFile
*/
int decodeCmd(std::string inFilename, std::string outFilename);

/**
  * @brief encode an the inFilename to outFilename lossless
  */
int encodeLosslessHevc(std::string inFilename, std::string outFilename);

/**
  * @brief inits the Conversation from the YUV-File given by filenmae to OpenCV Mat
  */
void yuv2MatInit(std::string filename, FILE* fin, YUV_ReturnValue* ret, IplImage* bgr, YUV_Capture* cap);

/**
  * @brief Converts a YUV capture to OpenCV Mat bgr coded
  */
cv::Mat yuv2Mat(YUV_ReturnValue* ret, IplImage* bgr, YUV_Capture* cap);

int main(){
	//Zeitmessung
	clock_t start, afterFacedet; 
	float elapsed, elapsedafterfacedet;

	av_register_all();

	start = clock();
	
	int ret = 0;
	//ffmpegopenVid(inputfilename.c_str());


	int i = 0;
	//while (querryFrame()){
		//cout << "frame " << ++i << endl;
		ret = startVid();		
		//imshow(window_name, cvFrameContext); 
		cvDestroyAllWindows();
		cout << "facedet finished" << endl;
		waitKey(1);
	//}


	if (ret != 0){
		cout << "Error " << ret << " ... closing now" << endl;
		waitKey(0);
	}
	else{
		cout << "HEVC decoding video" << endl;
		encodeCmd(videostr, outvideostr,BITRATE_ONE, sizestr);
		cout << "HEVC decoding background " << endl;
		encodeCmd(backstr, backoutstr, BITRATE_SURROUND, sizestr);
		cout << "HEVC decoding faces" << endl;
		std::ostringstream fsize;
		fsize << FACES_VIDEO_WIDTH << "x" << FACES_VIDEO_HEIGHT;
		encodeCmd(roistr, faceoutstr, BITRATE_FACE, fsize.str());
		
	}
	elapsed = (float)(clock() - start) / CLOCKS_PER_SEC;
	cout << "elapsed encoding Time: " << elapsed << endl;

	///ENCODING FINISHED:
	///Compare different video files

	//Convert encoded Videos from HEVC to YUV
	decodeCmd(outvideostr, yuv_encoded_one); //Einzel Video
	decodeCmd(faceoutstr, yuv_encoded_face); //Faces Video
	decodeCmd(backoutstr, yuv_encoded_back); //Background Video

	//Init the new YUV-Videos to OpenCv
	//One Video
	FILE *f_one = NULL;
	struct YUV_Capture cap_one;
	enum YUV_ReturnValue ret_one;
	IplImage bgr_one;
	//Init:
	f_one = fopen(yuv_encoded_one.c_str(), "rb");
	if (!f_one)
		cout << "Couldn't open " << yuv_encoded_one << "to convert to CV" << endl;
	ret_one = YUV_init(f_one, S.width, S.height, &cap_one);
	if (ret != YUV_OK)
		cout << "YUV_Init failed by " << yuv_encoded_one << endl;
	bgr_one = *(cvCreateImage(cvSize(S.width, S.height), IPL_DEPTH_8U, 3));
	if (!&bgr_one)
		cout << "bgr_create image failed by" << yuv_encoded_one << endl;


	//Background Video
	FILE *f_back = NULL;
	struct YUV_Capture cap_back;
	enum YUV_ReturnValue ret_back;
	IplImage bgr_back;
	//Init:
	f_back = fopen(yuv_encoded_back.c_str(), "rb");
	if (!f_back)
		cout << "Couldn't open " << yuv_encoded_back << "to convert to CV" << endl;
	ret_back = YUV_init(f_back, S.width, S.height, &cap_back);
	if (ret != YUV_OK)
		cout << "YUV_Init failed by " << yuv_encoded_back << endl;
	bgr_back = *(cvCreateImage(cvSize(S.width, S.height), IPL_DEPTH_8U, 3));
	if (!&bgr_back)
		cout << "bgr_create image failed by" << yuv_encoded_back << endl;

	//Mat Files of the videos
	cv::Mat yOne, yBack;
	int aktFrame = 0;

	//Durchlaufe alle einglesenen Frames und kombiniere Background und face sowie Anzeige & Speichern
	if (!outtogether.open(yuv_together, fourcc_output_codec, fps, S, true)){
		cout << "Could not create Video writer...." << yuv_together << endl;
		return -1;
	}
	bool ready_one = false, ready_back = false;

	cout << "\nFaces in Frames";
	for (int q = 0; q < facesinFrame.size(); q++)
		cout << facesinFrame.at(q).frameid << ", ";
	cout << endl;

	while(!(ready_one && ready_back)){ //Durchlauf solange nicht YUV EOF der Videos
		cout << "YUV conv no. " << aktFrame+1 << endl;

		//One
		ret_one = YUV_read(&cap_one);
		if (ret_one == YUV_EOF){
			cout << "YUV EOF one" << endl;
			ready_one = true;
			break;
		}
		else if (ret_one == YUV_IO_ERROR){
			cout << "IO-Error yuv2cv reading one" << endl;
			break;
		}
		cvCvtColor(cap_one.ycrcb, &bgr_one, CV_YCrCb2BGR);
		yOne = cv::Mat(&bgr_one);

		//back
		ret_back = YUV_read(&cap_back);
		if (ret_back == YUV_EOF){
			cout << "YUV EOF back" << endl;
			ready_back = true;
			break;
		}
		else if (ret_back == YUV_IO_ERROR){
			cout << "IO-Error yuv2cv reading back" << endl;
			break;
		}
		cvCvtColor(cap_back.ycrcb, &bgr_back, CV_YCrCb2BGR);
		yBack = cv::Mat(&bgr_back);

		//Faces wurde in Frame entdeckt
		int verschiebung = 0;
		for (int d = 0; d < facesinFrame.size(); d++){ //Durchlaufe faces rects und
			if (aktFrame == facesinFrame.at(d).frameid){ //Suche Frame in Faces Vector
				for (int j = 0; j < facesinFrame.at(d).rect.size(); j++){
					if (j > 0){
						verschiebung++;
						cout << "Verschiebung " << j << endl; 
					}
					cout << "frameID: " << facesinFrame.at(d).frameid << endl;
					cout << "aktFrame: " << aktFrame << endl;
					cout << "face: " << j << endl;
					int width = facesinFrame.at(d).rect.at(j).width;
					int height = facesinFrame.at(d).rect.at(j).height;
					int x = facesinFrame.at(d).rect.at(j).x;
					int y = facesinFrame.at(d).rect.at(j).y;

					//Look for face in yuv-Vid
					//Face Video
					FILE *f_face = NULL;
					struct YUV_Capture cap_face;
					enum YUV_ReturnValue ret_face;
					IplImage bgr_face;
					//Init:
					f_face = fopen(yuv_encoded_face.c_str(), "rb");
					if (!f_face)
						cout << "Couldn't open " << yuv_encoded_face << "to convert to CV" << endl;
					ret_face = YUV_init(f_face, FACES_VIDEO_WIDTH, FACES_VIDEO_HEIGHT, &cap_face);
					if (ret != YUV_OK)
						cout << "YUV_Init failed by " << yuv_encoded_face << endl;
					bgr_face = *(cvCreateImage(cvSize(FACES_VIDEO_WIDTH, FACES_VIDEO_HEIGHT), IPL_DEPTH_8U, 3));
					if (!&bgr_face)
						cout << "bgr_create image failed by" << yuv_encoded_face << endl;
					//decode
					for (int p = 0; p <= (d + verschiebung); p++){ //Suche Frame in faces yuv plus verschiebung um frames mit mehren faces
						ret_face = YUV_read(&cap_face);
						if (ret_face == YUV_EOF){
							cout << "YUV EOF face" << endl;
							break;
						}
						else if (ret_face == YUV_IO_ERROR){
							cout << "IO-Error yuv2cv reading face" << endl;
							break;
						}
					}
					cvCvtColor(cap_face.ycrcb, &bgr_face, CV_YCrCb2BGR);
					cv::Mat yFace = cv::Mat(&bgr_face);
						

					cv::Mat croppedFace = yFace(Rect(0, 0, width, height)); //Cut faces frame
					cv::Rect pos(cv::Point(x, y), cv::Size(width, height)); //rect where Face should be
					cv::Mat destination(yBack, pos);
					imshow("destination", destination);
					cvWaitKey(2);
					croppedFace.copyTo(destination);
					imshow("destination2", destination);
					cvWaitKey(2);

					//Memory Leak
					fclose(f_face);
					croppedFace.release();
					destination.release();
				}
				break; //Abort searching for another frame because it won't be anymore left
			}
		}

		//show
		//namedWindow("one", WINDOW_NORMAL);
		//namedWindow("together", WINDOW_NORMAL);
		imshow("one", yOne);
		cvWaitKey(1);
		imshow("together", yBack);
		cvWaitKey(1);
		outtogether << yBack;
		cvWaitKey(1);
		//memory leak
		//cvShowManyImages("compare", 2, cvCloneImage(&(IplImage)yOne), cvCloneImage(&(IplImage)yBack)); //Show in one window
		cvWaitKey(1);

		aktFrame++;
	}

	//Lossless encoding to HEVC
	encodeLosslessHevc(yuv_together,hevc_together);

	elapsed = (float)(clock() - start) / CLOCKS_PER_SEC;
	cout << "complete elapsed Time: " << elapsed << endl;

	cout << " Press Enter to exit..." << endl;
	cin.ignore();
	return ret;
}

int startVid(void){
	
	cout << "\t Load input, detect faces and encode Background:\n" << endl;
	Mat frame;
	
	//-- 1. Load the cascades
	if (!face_cascade.load(face_cascade_name)){ cout << "--(!)Error loading: " << face_cascade_name << "\n" << endl; return -1; }; //face
	if (!eyes_cascade.load(eyes_cascade_name)){ cout << "--(!)Error loading: " << eyes_cascade_name << "\n" << endl; return -1; }; //eyes
	cout << "loading cascades.. Ok" << endl;
	
	//-- 2. Read the video stream
	capture.open(inputfilename);
	int framecounter = 0;

	if (!capture.isOpened()){
		cout << "Can't open Video... in: " << inputfilename << endl;
		return -1;
	}

	//Video Output
	/*Size*/ S = Size((int)capture.get(CV_CAP_PROP_FRAME_WIDTH),    // Acquire input size
		(int)capture.get(CV_CAP_PROP_FRAME_HEIGHT));
	fps = capture.get(CV_CAP_PROP_FPS); //fps
	if (!outputVideo.open(videostr, fourcc_output_codec, fps, S, true)){ //out
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

	//Video for Faces
	//if (!faceVideo.open(roistr, fourcc_output_codec, fps, S, true)){ //TODO
	if (!faceVideo.open(roistr, fourcc_output_codec, fps, cv::Size(FACES_VIDEO_WIDTH, FACES_VIDEO_HEIGHT), true)){
		cout << "Could not create facevideo writer...." << endl;
		return -1;
	}
	
	if (!faceVideo.isOpened())
	{
		cout << "Could not open the output face Stream to write... " << endl;
		return -1;
	}

	//Video for background
	if (!backVideo.open(backstr, fourcc_output_codec, fps, S, true)){
		cout << "Could not create background-video writer...." << endl;
		return -1;
	}

	if (!backVideo.isOpened()){
		cout << "Could not open the output background Stream to write... " << endl;
		return -1;
	}


	//Saving information for HEVC encoder
	std::ostringstream dummy, dummy2;
	dummy << fps;
	fpsstr = dummy.str();
	dummy2 << S.width << "x" << S.height;
	sizestr = dummy2.str();


	Mat roi;
	int f = 0;
	int anzahlFaces = 0;
	double maxFrameSizeX = 0.0, maxFrameSizeY = 0.0; //biggest Facesize in X and Y
	while (capture.read(frame)){
		cout << "  frame: " << ++f << endl;

		//TODO 
		//if (f >= 12) break; //TODO ONLY READ first 12 FRAMES

		if (!frame.empty()){
			std::vector<Rect> faces = detectFaces(frame);
			for (size_t i = 0; i < faces.size(); i++) //Durchlaufe alle detektierten Geischter und male Rechteck sowie Speicherung
			{
				anzahlFaces++;
				//Remove too small faces
				if (faces.at(i).width <= ((int)(GROP_FACE_SIZE*(capture.get(CV_CAP_PROP_FRAME_WIDTH))))){
					cout << "face witdth: " << faces.at(i).width << " frame_size: " << GROP_FACE_SIZE* capture.get(CV_CAP_PROP_FRAME_WIDTH) << endl;
					cout << "face smaller than " << GROP_FACE_SIZE << "x of video -> delete" << endl;
					faces.erase(faces.begin() + i);
					i--;
					//cin.ignore();
					anzahlFaces--;
				}else{

#ifdef WITH_FACE_RECTANGLE
					Point p21(faces[i].x, faces[i].y); //Oben links vom Gesicht
					Point p22(faces[i].x + faces[i].width, faces[i].y + faces[i].height); //Unten rechts vom Gesicht
					rectangle(frame, p21, p22, Scalar(0, 255, 0), 4, 8, 0); //Rechteck ums Gesicht
#endif

					//Erweiterung der H�he in Stirn und Kinn 
					//height
					((faces.at(i).y + faces.at(i).height * FACE_BIG_TOP) < S.height) ? faces.at(i).y = faces.at(i).y - faces.at(i).height * FACE_BIG_TOP : faces.at(i).y = S.height;
					faces.at(i).height += (FACE_BIG_TOP + FACE_BIG_BTM) * faces.at(i).height;
					//width
					((faces.at(i).x + faces.at(i).x * FACE_BIG_RIGHT) < S.width) ? faces.at(i).x = faces.at(i).x - faces.at(i).width * FACE_BIG_RIGHT : faces.at(i).x = S.width;
					faces.at(i).width += (FACE_BIG_LEFT + FACE_BIG_RIGHT) * faces.at(i).width;

#ifdef WITH_FACE_RECTANGLE
					Point p1(faces[i].x, faces[i].y); //Oben links vom Gesicht
					Point p2(faces[i].x + faces[i].width, faces[i].y + faces[i].height); //Unten rechts vom Gesicht
					rectangle(frame, p1, p2, Scalar(255, 0, 255), 4, 8, 0); //Rechteck ums Gesicht
#endif
					//detect biggest frame size
					(faces[i].width > maxFrameSizeX)  ? maxFrameSizeX = faces[i].width  : maxFrameSizeX ;
					(faces[i].height > maxFrameSizeY) ? maxFrameSizeY = faces[i].height : maxFrameSizeY ;

					//Region of Interest = gesicht
					roi = frame(faces.at(i));
					imshow("roi", roi);
					waitKey(1);
					cout << "writing face no. " << i + 1 << endl;
					Mat lroi = Mat::zeros(cv::Size(FACES_VIDEO_WIDTH, FACES_VIDEO_HEIGHT), roi.type());
					//Mat lroi = Mat::zeros(S, roi.type());
					roi.copyTo(lroi(Rect(0, 0, roi.cols, roi.rows)));
					imshow("lroi", lroi);
					waitKey(1);
					faceVideo << lroi;

					//Speichere die Gesichter in Vektor zur spaeteren Zusammenfuerung
					faceInFrame dummy;
					dummy.frameid = f - 1;
					dummy.rect = faces;
					facesinFrame.push_back(dummy);
					cout << "adding frame " << dummy.frameid << " faces " << dummy.rect.size() << endl;
				}
			}

			imshow("face det", frame); //showing
			waitKey(1);
			cout << "Writing..." << endl;
			outputVideo << frame; //Schreibe frame in output video
			backVideo   << frame; //Schreibe frame in background video
		}
		else cout << "ERROR: Null Frame..." << endl;
		framecounter++;
	}
	cout << " total " << anzahlFaces << " faces detected" << endl;
	cout << " faces array "<< facesinFrame.size() << endl;
#ifdef 	FACES_PRINT_MAX
	cout << " biggest Frame: w: " << maxFrameSizeX << " h: " << maxFrameSizeY << endl;
	cout << "Press Enter to continue..." << endl;
	cin.ignore();
#endif
	return 0;
}


vector<Rect> detectFaces(Mat frame){
	// Idea from:
	// http://docs.opencv.org/2.4/doc/tutorials/objdetect/cascade_classifier/cascade_classifier.html

	cout << "detect faces..." << endl;

	std::vector<Rect> faces;
	Mat frame_gray;
	bool del = false;

	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray); //Normalisiere Histogram

	//-- Detect faces
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));


	//Make better detection
	for (size_t i = 0; i < faces.size(); i++){


#ifdef WITH_EYES_DETECTION

		//Look if we detect eyes in face to make a better Facedetection
		Mat faceROI = frame_gray(faces[i]);
		std::vector<Rect> eyes;

		//-- In each face, detect eyes
		eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

		if (eyes.size() == 0 && !del){ //No eyes detected -> remove face from vector
			faces.erase(faces.begin() + i);
			cout << "No eye detected -> deleted Face No. " << i << endl;
			del = true;
		}
#endif
	}


	cout << "detecting faces finished..." << endl;
	return faces;
}

bool ffmpegopenVid(const char * filename){
	bool errorStatus = false;

	//openVideo File
	if (avformat_open_input(&pVFormatCtx, filename, NULL, NULL) != 0){
		cout << "ERROR:openVideoFile:Could not open the video file " <<filename<< endl;
		return errorStatus = true;
	}

	// Retrieve stream information. Populates pVFormatCtx->streams with the proper information.
	if (avformat_find_stream_info(pVFormatCtx, NULL) < 0){
		cout << "ERROR:openVideoFile:Could not open the stream" << endl;
	}

	av_dump_format(pVFormatCtx, 0, filename, 0);

	for (int i = 0; i < pVFormatCtx->nb_streams; i++){
		if (pVFormatCtx->streams[i]->codec->codec_type == AVMEDIA_TYPE_VIDEO){
			videoStream = i;
			break;
		}
	}

	if (videoStream < 0){
		cout << "ERROR: Could not find a valid codec..." << endl;
		return errorStatus = true;
	}

	//Get a pointer to the codec context for the video stream. The stream's information about the codec is in what we call the "codec context." This contains all the information about the codec that the stream is using, and now we have a pointer to it.
	pVCodecCtx = pVFormatCtx->streams[videoStream]->codec;

	// Find the actual codec and open it. Find the decoder for the video stream.
	pVCodec = avcodec_find_decoder(pVCodecCtx->codec_id);

	if (pVCodec == NULL){
		cout << "ERROR:openVideoFile:Unsupported codec or codec not found!" << endl;
		return errorStatus = true;
	}

	cout << "openVideoFile:Decoder: " << pVCodec->name << endl;

	if (avcodec_open2(pVCodecCtx, pVCodec, NULL) < 0){
		cout << "ERROR:openVideoFile : Could not open codec!" << endl;
		return errorStatus = true;
	}

	//allocate Memory Space
	pVFrame = av_frame_alloc();
	pVFrameBGR = av_frame_alloc();

	if (pVFrameBGR == NULL){
		cout << "ERROR:openVideoFile:Could Not Allocate the frame!" << endl;
		return errorStatus = true;
	}

	int numBytes = avpicture_get_size(AV_PIX_FMT_BGR24, pVCodecCtx->width, pVCodecCtx->height);
	bufferBGR = (uint8_t *)av_mallocz(numBytes*sizeof(uint8_t));
	avpicture_fill((AVPicture*)pVFrameBGR, bufferBGR, AV_PIX_FMT_BGR24, pVCodecCtx->width, pVCodecCtx->height);

	cvFrameContext.create(pVCodecCtx->height, pVCodecCtx->width, CV_8UC(3));
	return errorStatus;
}

bool querryFrame(void){
	if (av_read_frame(pVFormatCtx, &pVPacket) < 0){
		cout << "ERROR:queryFram:Could nor read frame!" << endl;
		return false;
	}

	if (pVPacket.stream_index == videoStream){
		cout << "is Stream" << pVPacket.stream_index << endl;
		if (avcodec_decode_video2(pVCodecCtx, pVFrame, &frameFinished, &pVPacket) < 0){
			cout << "Error:querryFrame:Could not decode Video!" << endl;
			return false;
		}

		if (frameFinished){
			cout << "frame finished" << endl;
			frameFinished = 0;
			if (pVImgConvertCtx == NULL){
				pVImgConvertCtx = sws_getContext(pVCodecCtx->width, pVCodecCtx->height, pVCodecCtx->pix_fmt, pVCodecCtx->width, pVCodecCtx->height, AV_PIX_FMT_BGR24, SWS_BICUBIC, NULL, NULL, NULL);
			}
			if (pVImgConvertCtx == NULL){
				cout << "ERROR:queryFrame:Cannot initialize the conversation context!" << endl;
				return false;
			}
			
			sws_scale(pVImgConvertCtx, pVFrame->data, pVFrame->linesize, 0, pVCodecCtx->height, pVFrameBGR->data, pVFrameBGR->linesize);

			//OpenCV 
			for (int y = 0; y < pVCodecCtx->height; y++){
				for (int x = 0; x < pVCodecCtx->width; x++){
					cvFrameContext.at<cv::Vec3b>(y, x)[0] = pVFrameBGR->data[0][y * pVFrameBGR->linesize[0] + x * 3 + 0];
					cvFrameContext.at<cv::Vec3b>(y, x)[1] = pVFrameBGR->data[0][y * pVFrameBGR->linesize[0] + x * 3 + 1];
					cvFrameContext.at<cv::Vec3b>(y, x)[2] = pVFrameBGR->data[0][y * pVFrameBGR->linesize[0] + x * 3 + 2];
				}
			}
		}
	}
	return true;
}


int encodeCmd(std::string filename, std::string outfilename, std::string bitrate, std::string vsize){
	int ret = 0;
	//std::string befehl = "ffmpeg -y -s:v " + vsize + " -r " + fpsstr + " -i " + filename + " -b:v " + bitrate + " -bufsize " + bitrate + " -c:v libx265 -loglevel quiet " + outfilename;
	std::string befehl = "ffmpeg -y -s:v " + vsize + " -r " + fpsstr + " -i " + filename + " -b:v " + bitrate + " -bufsize " + bitrate + " -c:v libx265 " + outfilename;
	cout << endl << "Encoding msg: " << endl;
	cout << befehl << endl;
	ret = system(befehl.c_str());	
	return ret;
}

int decodeCmd(std::string inFilename, std::string outFilename){
	int ret = 0;
	std::string befehl = "ffmpeg -y -i " + inFilename + " -loglevel quiet " + outFilename;
	cout << endl << "Decoding msg: " << endl;
	cout << befehl << endl;
	ret = system(befehl.c_str());
	return ret;
}

int encodeLosslessHevc(std::string inFilename, std::string outFilename){
	int ret = 0;
	std::string befehl = "ffmpeg -y -s:v " + sizestr + " -r " + fpsstr +" -i " + inFilename + " -c:v libx265 -x265-params lossless " + outFilename;
	cout << endl << "Decoding msg: " << endl;
	cout << befehl << endl;
	ret = system(befehl.c_str());
	return ret;
}

/*
void x265Encoding(){
	x265_param *param = x265_param_alloc();
	int x265_param_default_preset(param, const char *preset, const char *tune);
	int x265_param_apply_profile(param, const char *profile);

	for (faces){
		int x265_param_parse(param, "display-window", const char *value);
	}

	x265_encoder* enc = x265_encoder_open(param);

}
*/
//TODO unuse now maybe
static bool decodeHEVC(const char *filename){
	AVCodec * codec;
	AVCodecContext *c = NULL;
	int i, ret, x, y, got_output;
	FILE *f;
	AVFrame *picture;
	AVPacket pkt;
	uint8_t encode[] = { 0, 0, 1, 0xb7 };

	av_init_packet(&pkt);

	//Find HEVC encoder
	codec = avcodec_find_encoder(AV_CODEC_ID_HEVC);
	if (!codec){
		cout << "ERROR:output HEVC encoder not found" << endl;
		return false;
	}
	c = avcodec_alloc_context3(codec);
	if (!c){
		cout << "ERROR:Could not allocate video codec context\n" << endl;
		return false;
	}


	c->bit_rate = 400000;
	c->width  = pVFrame->width;
	c->height = pVFrame->height;
	//fps
	//c->time_base = (AVRational)(1, 25);
	c->time_base.den = 1;
	c->time_base.num = 25;
	c->gop_size = 10;
	c->max_b_frames = 10; //TODO
	c->pix_fmt = AV_PIX_FMT_YUV420P;
	
	//open
	if (avcodec_open2(c, codec, NULL) < 0){
		cout << "ERROR:Could not open HEVC Encoder" << endl;
		return false;
	}

	f = fopen(filename, "wb");
	if (!f){
		cout << "ERROR: Could not open Output file: " << filename << endl;
		return false;
	}

	/* alloc image and output buffer */
	uint8_t *outbuf, *picture_buf;
	int outbuf_size = 100000, size;
	outbuf = (uint8_t*) malloc(outbuf_size);
	size = c->width * c->height;
	picture_buf = (uint8_t*) malloc((size * 3) / 2); //size of YUV420

	picture->data[0] = picture_buf;
	picture->data[1] = picture->data[0] + size;
	picture->data[2] = picture->data[1] + size / 4;
	picture->linesize[0] = c->width;
	picture->linesize[1] = c->width / 2;
	picture->linesize[2] = c->width / 2;

	return true;
}


void yuv2MatInit(std::string filename, FILE* fin, YUV_ReturnValue* ret, IplImage* bgr, YUV_Capture* cap){
	fin = fopen(filename.c_str(),"rb");
	if (!fin){
		cout << "Couldn't open " << filename << "to convert to CV" << endl;
		return;
	}
	*ret = YUV_init(fin, S.width, S.height, cap);
	if (*ret != YUV_OK)
		cout << "YUV_INIT failed" << endl;

	bgr = cvCreateImage(cvSize(S.width, S.height), IPL_DEPTH_8U, 3);
	if (!bgr)
		cout << "bgr_Create Image failed" << endl;
}

cv::Mat yuv2Mat(YUV_ReturnValue* ret, IplImage* bgr, YUV_Capture* cap){
	*ret = YUV_read(cap);
	if (*ret == YUV_EOF){
		cout << "YUV EOF" << endl;
		return cv::Mat();
	}
	else if (*ret == YUV_IO_ERROR){
		cout << "IO-Error yuv2cv reading " << endl;
		return cv::Mat();
	}
	cout << "yuv2Mat vor cvtColor" << endl;
	::cvCvtColor(cap->ycrcb, bgr, CV_YCrCb2BGR);
	cout << "yuv2Mat nach cvtColor" << endl;
	return cv::Mat(bgr);
}