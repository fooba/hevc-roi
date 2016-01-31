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

//Da Kamera nich immer direkt verfügbar ist
int wait4Cam = 2;

typedef struct{
	int frameno;
	std::vector<Rect> facesinframe;
	std::vector<Mat>  faces;
}facesfromframe;

#define SHOW_DETECTED_FACES //comment for no output of the frames in face detection
#define WITH_FACE_RECTANGLE //comment for no Rectangle in ouput file where Face is detected
#define GROP_FACE_SIZE 0.1 //Face must be greater than GROP_FACE_SIZE*InputVideo.Width

/**
  * Definitions for Bitrates
*/
#define BITRATE_ONE       "10k"
#define BITRATE_FACE	  "15k"
#define BITRATE_SURROUND  "5k"

const std::string inputfilename    = "C:\\testshort.mp4";
const std::string roistr		   = "C:\\faces.yuv";
const std::string videostr		   = "C:\\video.yuv";
const std::string backstr		   = "C:\\background.yuv";
const std::string outvideostr	   = "C:\\video.mkv";
const std::string faceoutstr	   = "C:\\faces.mkv";
const std::string backoutstr	   = "C:\\background.mkv";

const std::string yuv_encoded_one  = "C:\\yuv_one.yuv";
const std::string yuv_encoded_back = "C:\\yuv_back.yuv";
const std::string yuv_encoded_face = "C:\\yuv_face.yuv";


const int fourcc_output_codec =  CV_FOURCC('I', '4', '2', '0');
std::vector<facesfromframe> facesVideo;
static int framecounter=0;
VideoWriter outputVideo;
VideoWriter faceVideo;
VideoWriter backVideo;
Size S;

//Std Framerate und Groesse, aenderbar auf eingangsvideo oder aehnlichem
std::string fpsstr = "25";
std::string sizestr = "1920x10080";

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
int encodeCmd(std::string filename, std::string outfilename, std::string bitrate);

/**
  * @brief decode inFile in outFile
*/
int decodeCmd(std::string inFilename, std::string outFilename);

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
		encodeCmd(videostr, outvideostr,BITRATE_ONE);
		cout << "HEVC decoding background " << endl;
		encodeCmd(backstr, backoutstr, BITRATE_SURROUND);
		cout << "HEVC decoding faces" << endl;
		encodeCmd(roistr, faceoutstr,BITRATE_FACE);
		
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
	yuv2MatInit(yuv_encoded_one, f_one, &ret_one, &bgr_one, &cap_one);

	//Background Video
	FILE *f_back = NULL;
	struct YUV_Capture cap_back;
	enum YUV_ReturnValue ret_back;
	IplImage bgr_back;
	yuv2MatInit(yuv_encoded_back, f_back, &ret_back, &bgr_back, &cap_back);

	//Face Video
	FILE *f_face = NULL;
	struct YUV_Capture cap_face;
	enum YUV_ReturnValue ret_face;
	IplImage bgr_face;
	yuv2MatInit(yuv_encoded_face, f_face, &ret_face, &bgr_face, &cap_face);

	//Mat Files of the videos
	cv::Mat yOne, yBack, yFace;
	int aktFrame = 0;

	//Durchlaufe alle einglesenen Frames und kombiniere Background und face sowie Anzeige
	do{
		cout << "YUV conv no. " << ++aktFrame << endl;
		yOne  = yuv2Mat(&ret_one,  &bgr_one,  &cap_one );
		yBack = yuv2Mat(&ret_back, &bgr_back, &cap_back);
		yFace = yuv2Mat(&ret_face, &bgr_face, &cap_face);

		imshow("One", yOne);
		imshow("back", yBack);
		imshow("face", yFace);
	} while (!(yOne.empty() || yBack.empty()));

	elapsed = (float)(clock() - start) / CLOCKS_PER_SEC;
	cout << "complete elapsed Time: " << elapsed << endl;

	cout << " Press Enter to exit..." << endl;
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
	double fps = capture.get(CV_CAP_PROP_FPS); //fps
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
	if (!faceVideo.open(roistr, fourcc_output_codec, fps, S, true)){
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

	if (!faceVideo.isOpened())
	{
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
	int i = 0;
	while (capture.read(frame)){
		cout << "  frame: " << ++i << endl;

		//TODO 
		if (i >= 2) break; //TODO ONLY READ first 2 FRAMES

		if (!frame.empty()){
			std::vector<Rect> faces = detectFaces(frame);
			for (size_t i = 0; i < faces.size(); i++) //"Male" Rechteck um jedes Gesicht
			{
				//Remove too small faces
				if (faces.at(i).width <= ((int)(GROP_FACE_SIZE*(capture.get(CV_CAP_PROP_FRAME_WIDTH))))){
					cout << "face witdth: " << faces.at(i).width << " frame_size: " << GROP_FACE_SIZE* capture.get(CV_CAP_PROP_FRAME_WIDTH) << endl;
					cout << "face smaller than " << GROP_FACE_SIZE << "x of video -> delete" << endl;
					cout << "faces before: " << faces.size() << endl;
					faces.erase(faces.begin() + i);
					cout << "faces after: " << faces.size() << endl;
					//cin.ignore();
					continue;
				}


#ifdef WITH_FACE_RECTANGLE
				Point p1(faces[i].x, faces[i].y); //Oben links vom Gesicht
				Point p2(faces[i].x + faces[i].width, faces[i].y + faces[i].height); //Unten rechts vom Gesicht
				rectangle(frame, p1, p2, Scalar(255, 0, 255), 4, 8, 0); //Rechteck ums Gesicht
				imshow("face det", frame);
				waitKey(1);
#endif

				//Region of Interest = gesicht
				roi = frame(faces.at(i));
				imshow("face", roi);
				waitKey(1);
				cout << "writing face no. " << i + 1 << endl;
				Mat lroi = Mat::zeros(S, roi.type());
				roi.copyTo(lroi(Rect(0, 0, roi.cols, roi.rows)));
				faceVideo << lroi;
				/*
				//fuer speicherung
				vector<Mat> dummyFaces;
				dummyFaces.push_back(roi);
				facesfromframe dummy;
				dummy.frameno = framecounter;
				dummy.facesinframe = faces;
				dummy.faces = dummyFaces;
				facesVideo.push_back(dummy);
				*/

			}

			cout << "Writing..." << endl;
			outputVideo << frame; //Schreibe frame in output video
			backVideo   << frame; //Schreibe frame in background video

		}
		else cout << "ERROR: Null Frame..." << endl;
	}
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


int encodeCmd(std::string filename, std::string outfilename, std::string bitrate){
	int ret = 0;
	std::string befehl = "ffmpeg -y -s:v " + sizestr + " -r " + fpsstr + " -i " + filename + " -b:v " + bitrate + " -bufsize " + bitrate + " -c:v libx265 -loglevel quiet " + outfilename;
	cout << endl << "Encodeing msg: " << endl;
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
	assert(*ret == YUV_OK);

	bgr = cvCreateImage(cvSize(S.width, S.height), IPL_DEPTH_8U, 3);
	assert(bgr);
}

cv::Mat yuv2Mat(YUV_ReturnValue* ret, IplImage* bgr, YUV_Capture* cap){
	*ret = YUV_read(cap);
	if (*ret == YUV_EOF){
		return cv::Mat();
	}
	else if (*ret == YUV_IO_ERROR){
		cout << "IO-Error yuv2cv reading " << endl;
		return cv::Mat();
	}
	cout << "yuv2Mat vor cvtColor" << endl;
	::cvCvtColor(cap->ycrcb, &bgr, CV_YCrCb2BGR);
	cout << "yuv2Mat nach cvtColor" << endl;
	return cv::Mat(bgr);
}