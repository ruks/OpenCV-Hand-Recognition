// opencv.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <vector>
#include "Serial.h"
#include "string.h"
#include <ctime>
#include <atltime.h>

using namespace cv;
using namespace std;

void see(){
	Mat frame;
	Mat back;
	Mat fore;
	VideoCapture cap(0);
	BackgroundSubtractorMOG2 bg;
	//bg.nmixtures = 3;
	//bg.bShadowDetection = false;

	std::vector<std::vector<cv::Point> > contours;

	cv::namedWindow("Frame");
	cv::namedWindow("Background");

	for(;;)
	{
		cap >> frame;
		bg.operator ()(frame,fore);
		bg.getBackgroundImage(back);
		cv::erode(fore,fore,cv::Mat());
		cv::dilate(fore,fore,cv::Mat());
		cv::findContours(fore,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
		cv::drawContours(frame,contours,-1,cv::Scalar(0,0,255),2);
		cv::imshow("Frame",frame);
		cv::imshow("Background",back);
		if(cv::waitKey(30) >= 0) break;
	}
}

void see1(){
	RNG rng(12345);
	VideoCapture cap(0); // open the default camera

	if(!cap.isOpened()){
		cout<<"Error"<<endl;
	}

	Mat edges;
	//namedWindow("edges",1);

	for(;;){
		Mat img,blur ,gray,thresh;
		vector<vector<Point>> contours;
		cap >> img; 	

		cvtColor(img, gray, CV_RGB2GRAY);
		GaussianBlur(gray, blur , Size(7,7), 1.5, 1.5);

		threshold(blur, thresh, 70, 100, CV_THRESH_BINARY_INV+THRESH_OTSU);

		//erode(thresh,thresh,Mat());
		//dilate(thresh,thresh,Mat());

		findContours(thresh,contours,RETR_TREE,CHAIN_APPROX_SIMPLE);
		//drawContours(thresh,contours,-1,Scalar(0,0,255),2);

		vector<Point> cnt,hul;
		int area,max_area=0,ci=-1;
		vector<vector<Point>> hull( contours.size() );
		//Mat hull;
		int s=contours.size();
		for (int i = 0; i < s; i++)
		{
			cnt=contours.at(i);
			area = contourArea(cnt);
			if(area > max_area){
				max_area=area;
				ci=i;
				cnt=contours.at(ci);
			}

		}

		if(ci<0){
			cout<<"Cnt error"<<endl;
			continue;
		}
		convexHull(Mat(contours[ci]),hull[ci],false);	

		Mat drawing = Mat::zeros( thresh.size(), CV_8UC3 );
		//cout << ci <<endl;

		for( int i = 0; i< 1; i++ ){
			Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
			drawContours( drawing, contours, ci, color, 1, 8, vector<Vec4i>(), 0, Point() );
			drawContours( drawing, hull, ci, color, 1, 8, vector<Vec4i>(), 0, Point() );
		}


		Moments moment = moments(contours[ci],false);
		area = moment.m00;//m00 gives the area
		int x = moment.m10/area;//gives the x coordinate
		int y = moment.m01/area;  //gives y coordiante
		cout << x <<" " << y <<endl;

		Scalar col = Scalar( 255,0,0 );
		circle(img, Point(x,y), 20, col, -1);

		imshow("edges",drawing);
		imshow("move",img);
		cvWaitKey (10);
	}
}

void detect_and_draw( IplImage* img )
{
	const char *cascade_name="hand.xml";
	// Create memory for calculations
	static CvMemStorage* storage = 0;

	// Create a new Haar classifier
	static CvHaarClassifierCascade* cascade = 0;

	// Sets the scale with which the rectangle is drawn with
	int scale = 1;

	// Create two points to represent the hand locations
	CvPoint pt1, pt2;

	// Looping variable
	int i; 

	// Load the HaarClassifierCascade
	cascade = (CvHaarClassifierCascade*)cvLoad( cascade_name, 0, 0, 0 );

	// Check whether the cascade has loaded successfully. Else report and error and quit
	if( !cascade )
	{
		fprintf( stderr, "ERROR: Could not load classifier cascade\n" );
		return;
	}

	// Allocate the memory storage
	storage = cvCreateMemStorage(0);

	// Create a new named window with title: result
	cvNamedWindow( "result", 1 );

	// Clear the memory storage which was used before
	cvClearMemStorage( storage );

	// Find whether the cascade is loaded, to find the hands. If yes, then:
	if( cascade )
	{

		// There can be more than one hand in an image. So create a growable sequence of hands.
		// Detect the objects and store them in the sequence
		CvSeq* hands = cvHaarDetectObjects( img, cascade, storage,
			1.1, 2, CV_HAAR_DO_CANNY_PRUNING,
			cvSize(40, 40) );

		// Loop the number of hands found.
		for( i = 0; i < (hands ? hands->total : 0); i++ )
		{
			// Create a new rectangle for drawing the hand
			CvRect* r = (CvRect*)cvGetSeqElem( hands, i );

			// Find the dimensions of the hand,and scale it if necessary
			pt1.x = r->x*scale;
			pt2.x = (r->x+r->width)*scale;
			pt1.y = r->y*scale;
			pt2.y = (r->y+r->height)*scale;

			// Draw the rectangle in the input image
			cvRectangle( img, pt1, pt2, CV_RGB(230,20,232), 3, 8, 0 );
		}
	}

	// Show the image in the window named "result"
	cvShowImage( "result", img );   
}

void see2(){
	CvCapture* cap= cvCaptureFromCAM(-1);

	if(!cap){
		cout<<"Error"<<endl;
	}

	namedWindow("edges",1);

	for(;;){
		IplImage* frame = cvQueryFrame( cap);

		detect_and_draw(frame);
		cvWaitKey (10);
	}
}

void diff()
{


	VideoCapture cap(0); // open the default camera

	if(!cap.isOpened()){
		cout<<"Error"<<endl;
	}

	Mat edges;
	namedWindow("edges",1);


	for(;;){
		Mat prev_frame,current_frame,next_frame;

		cap >> prev_frame; 					
		cvtColor(prev_frame, prev_frame, CV_RGB2GRAY);
		GaussianBlur(prev_frame, prev_frame, Size(7,7), 1.5, 1.5);
		Canny(prev_frame, prev_frame, 0, 30, 3);

		cap >> current_frame; 					
		cvtColor(current_frame, current_frame, CV_RGB2GRAY);
		GaussianBlur(current_frame, current_frame, Size(7,7), 1.5, 1.5);
		Canny(current_frame, current_frame, 0, 30, 3);

		cap >> next_frame; 					
		cvtColor(next_frame, next_frame, CV_RGB2GRAY);
		GaussianBlur(next_frame, next_frame, Size(7,7), 1.5, 1.5);
		Canny(next_frame, next_frame, 0, 30, 3);

		Mat d1, d2, motion;

		while (true){
			vector<vector<Point> > contours;
			prev_frame = current_frame;
			current_frame = next_frame;

			cap >> next_frame;			

			cvtColor(next_frame, next_frame, CV_RGB2GRAY);
			GaussianBlur(next_frame, next_frame, Size(7.5,7.5), 1.5, 1.5);
			Canny(next_frame, next_frame, 0, 30, 3);
			imshow("edges",next_frame);	

			absdiff(prev_frame, current_frame, d1);
			absdiff(next_frame, current_frame, d2);			

			bitwise_and(d1, d2, motion);
			threshold(motion, motion, 35, 255, CV_THRESH_BINARY);
			imshow("d1",d1);							
			imshow("d2",d2);

			imshow("motion",motion);							

			cvWaitKey (10);
		}

	}
	system("Pause");
}

int transfer ()
{
	/*FILE *comport;

	if ((comport = fopen("COM1", "wt")) == NULL)
	{
	printf("Failed to open the communication port COM2\n");
	printf("The port may be disabled or in use\n");
	//int wait=getch();
	return 1;
	}
	printf("COM2 opended successfully\n");

	int x=5;
	int y=10;
	string s=x+" "+y;
	//fputs(s.c_str, comport);
	fflush(comport);
	fclose(comport);*/
	return 0;
}

void transfer1(){

	tstring commPortName(TEXT("COM1"));
	Serial serial(commPortName, 9600);

	//Serial serial(commPortName);
	cout << "Port opened" << endl;

	cout << "writing something to the serial port" << endl;
	serial.flush();

	int x=5;
	int y=10;



	string str=to_string(x)+" "+to_string(y);
	float an=y/x;
	float len=sqrt(x*x+y*y);
	cout<<an<<" "<<len<<endl;
	cout<<to_string(an)<<" "<<to_string(len)<<endl;
	cout<<str<<endl;


	system("pause");
	return;
	int TempNumOne=str.size();
	char str_o[100];
	for (int a=0;a<=TempNumOne;a++)
	{
		str_o[a]=str[a];
	}
	//str_o[TempNumOne]='\n';

	int bytesWritten = serial.write(str_o);
	cout << bytesWritten << " bytes were written to the serial port" << endl;
	if(bytesWritten != sizeof(str_o) - 1)
	{
		cout << "Writing to the se rial port timed out" << endl;
	}


}

string currentDateTime() {
	SYSTEMTIME st;
	GetSystemTime(&st);


	string year=to_string(st.wYear);
	string month=to_string(st.wMonth);
	string day=to_string(st.wDay);
	string hour=to_string(st.wHour);
	string minute=to_string(st.wMinute);
	string second=to_string(st.wSecond);
	string mSecond=to_string(st.wMilliseconds);

	string time="file_"+year+"_"+month+"_"+day+"_"+hour+"-"+minute+"-"+second+"-"+mSecond+".jpg";	
	//cout<<time<<endl;
	return time;
}

void saveIplImage(IplImage *pSaveImg,string folder){
	string folderCreateCommand = "mkdir " + folder;
	system(folderCreateCommand.c_str());

	string str=currentDateTime();
	str=folder+"/"+str;
	char filename[100];
	for (int a=0;a<=str.size();a++)
	{
		char c=str[a];
		if(c==' '){
			filename[a]='_';
		}else if(c==':'){
			filename[a]='-';
		}else{
			filename[a]=c;
		}

	}

	int status=cvSaveImage(filename,pSaveImg);	

	if(status>0){
		cout<<filename<<" saved succesfully"<<endl;
	}else{
		cout<<filename<<" saved error"<<endl;
	}
}

void saveImgMat(Mat image,string folder){
	string folderCreateCommand = "mkdir " + folder;
	system(folderCreateCommand.c_str());

	string str=currentDateTime();
	str=folder+"/"+str;	

	boolean b=imwrite(str, image);

	if(b){
		cout<<str<<" saved succesfully"<<endl;
	}else{
		cout<<str<<" saved error"<<endl;
	}
}

int _tmain(int argc, _TCHAR* argv[]){	

	CvCapture *pCapturedImage = cvCreateCameraCapture(0);
	VideoCapture cap(0);
	Mat frame;
	while(true){
		
		IplImage *pSaveImg = cvQueryFrame(pCapturedImage);
		cap >>frame;

		saveIplImage(pSaveImg,"test");		
		//saveImgMat(frame,"test1");

		cvWaitKey(10);	
	}
	system("pause");
	return 0;
}

