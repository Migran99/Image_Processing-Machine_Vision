#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include<cmath>

using namespace cv; //Using namespaces is not recommended for big projects
using namespace std;

void gaussVignettingGRAY(const Mat& myImage, Mat& retImage, float sigma, bool printResults) {
	Mat testIm, grayIm; //Internal images are a gray-scaled copy of the input image
	cvtColor(myImage, testIm, COLOR_BGR2GRAY);
    grayIm = testIm.clone();

    //We create two Gaussia distributions, one vertical and one horizontal
	Mat coefC = getGaussianKernel(myImage.cols, sigma);
	Mat coefR = getGaussianKernel(myImage.rows, sigma);
    //The matrix distribution is the product of the both above
    Mat mask;
	mask = coefR * coefC.t();

    //Now we get the maximum value of the matrix and divide everything by it, in order to normalize it.
	double maxVal, minVal;
	Point minLoc, maxLoc;
	minMaxLoc(mask, &minVal, &maxVal, &minLoc, &maxLoc);
	mask = mask / maxVal;

    //Now we apply our distrubution to each pixel
	for (int j = 0; j < myImage.rows; j++) {
		for (int i = 0; i < myImage.cols; i++) {
			testIm.at<uchar>(j,i) = saturate_cast<uchar>(testIm.at<uchar>(j, i) * mask.at<double>(j, i));
            //An inverted effect can be obtained by dividing instead of multiplying. Sigma has to be much higher in this case
		}
	}
    // If we want t show the results or not
	if (printResults) {
		namedWindow("original", WINDOW_NORMAL);
		imshow("original", grayIm);
		namedWindow("edit", WINDOW_NORMAL);
		imshow("edit", testIm);
	}
    //We edit the return Image. It could be done from the beginning with retImage instead of testIm.
    retImage=testIm.clone();
}