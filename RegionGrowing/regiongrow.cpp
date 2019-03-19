#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <queue>

using namespace cv;
using namespace std;

const uchar setConst = (uchar)255;

int growRegionColMin(Mat& hsvimg, Mat& sobelimg, Point2i seed, Point2i neigh[], Vec3b maxShift,Mat& regionBuffer, Mat& allRegionBuffer, Mat& resultimg) {
    int imgwidth = hsvimg.cols;
    int imgheight = hsvimg.rows;

    int regionSize = 1;

    //printf("hsvimgtop X: %d Y: %d \n", seed.x, seed.y);
    Vec3d regionCol = (Vec3d)hsvimg.at<Vec3b>(seed.y, seed.x);

    queue<Point2i> pointsToGrow;
    pointsToGrow.push(seed);

    Point2i neighInd;
    Point2i currentPoint;

    Vec3b seedVal;
    Vec3b neighVal;
    uchar sobelVal;

    Vec3i colDistance;

    //Mat occupationImg = Mat::zeros(imgheight, imgwidth, CV_8U);

    while (pointsToGrow.size() > 0) {
        currentPoint = pointsToGrow.front();
        pointsToGrow.pop();

        //printf("size: %d", pointsToGrow.size());

        //cout << "occupationImg + hsvimg" << endl;

        //printf("occupationImg1 X: %d Y: %d \n", currentPoint.x, currentPoint.y);
        regionBuffer.at<uchar>(currentPoint.y, currentPoint.x) = setConst;

        //printf("hsvimg0 X: %d Y: %d \n", currentPoint.x, currentPoint.y);
        seedVal = hsvimg.at<Vec3b>(currentPoint.y, currentPoint.x);
        regionCol = ((Vec3d)regionCol * regionSize + (Vec3d)seedVal) / ((double)regionSize + 1);
        regionSize += 1;
        seedVal = regionCol;

        resultimg.at<Vec3b>(currentPoint.y, currentPoint.x) = (Vec3b)regionCol;

        for (int i = 0; i < sizeof(neigh); i++){
            neighInd = currentPoint + neigh[i];

            //printf("occupationImg2 X: %d Y: %d \n", neighInd.x, neighInd.y);
            if(0 <= neighInd.x && imgwidth > neighInd.x && 0 <= neighInd.y && imgheight > neighInd.y/* &&
                occupationImg.at<uchar>(neighInd.y, neighInd.x) != setConst*/){

                if (regionBuffer.at<uchar>(neighInd.y, neighInd.x) != setConst && allRegionBuffer.at<uchar>(neighInd.y, neighInd.x) != setConst) {

                    //printf("hsvimg1 X: %d Y: %d \n", neighInd.x, neighInd.y);
                    neighVal = hsvimg.at<Vec3b>(neighInd.y, neighInd.x);
                    //printf("sobelimg X: %d Y: %d \n", neighInd.x, neighInd.y);
                    sobelVal = sobelimg.at<uchar>(neighInd.y, neighInd.x);

                    if (sobelVal < 50) {
                        absdiff((Vec3f)neighVal, (Vec3f)seedVal, colDistance);
                        if (colDistance[0] < maxShift[0] && colDistance[1] < maxShift[1] && colDistance[2] < maxShift[2]) {
                            pointsToGrow.push(neighInd);

                            //convert result img color
                            //cout << "resultimg" << endl;
                            regionBuffer.at<uchar>(neighInd.y, neighInd.x) = setConst;
                            resultimg.at<Vec3b>(neighInd.y, neighInd.x) = (Vec3b)regionCol;
                            //resultimg.at<Vec3b>(neighInd.y, neighInd.x) = regionCol;
                        }
                    }
                }
            }
        }
    }

    /*vector<Mat> resultChannels(3);
    split(resultimg, resultChannels);
    occupationImg = (occupationImg / setConst);
    resultChannels[0] = (resultChannels[0] & ~occupationImg) + occupationImg * regionCol[0];
    resultChannels[1] = (resultChannels[1] & ~occupationImg) + occupationImg * regionCol[1];
    resultChannels[2] = (resultChannels[2] & ~occupationImg) + occupationImg * regionCol[2];
    merge(resultChannels, resultimg);*/
    //resultimg = (resultimg & ~occupationImg) * (Mat)regionCol;

    allRegionBuffer = (allRegionBuffer | regionBuffer);
    /*if (regionSize < 1000) {
        vector<Mat> resultChannels(3);
        split(resultimg, resultChannels);
        resultChannels[0] = (resultChannels[0] & ~regionBuffer);
        resultChannels[1] = (resultChannels[1] & ~regionBuffer);
        resultChannels[2] = (resultChannels[2] & ~regionBuffer);
        merge(resultChannels, resultimg);
    }*/

    /*if (regionSize > 2000) {
        char windowname[100];
        //blur(occupationImg, occupationImg, Size(5, 5));
        GaussianBlur(occupationImg, occupationImg, Size(5, 5), 0.6, 0.6);
        sprintf(windowname, "region res %d %d", seed.x, seed.y);
        namedWindow(windowname, WINDOW_AUTOSIZE);// Create a window for display.
        imshow(windowname, occupationImg);
    }*/

    //occupationImg.release();

    return regionSize;
}

void applySobel(Mat& orgimg,Mat& dstBuffer) {

    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;
    Sobel(orgimg, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
    /// Gradient Y
    Sobel(orgimg, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
    convertScaleAbs(grad_x, abs_grad_x);
    convertScaleAbs(grad_y, abs_grad_y);
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dstBuffer);

    grad_x.release();
    grad_y.release();
    abs_grad_x.release();
    abs_grad_y.release();
}

Mat growRegionGrid(Mat& orgimg, Point2i seeddensity, Vec3b maxShift) {
    int maxPix = 80000;
    int orgw = orgimg.cols;
    int orgh = orgimg.rows;

    vector<Mat> hsvChannels(3);
    Mat fullSobelImg;
    Mat scaledSobelImg;
    Mat grayImg;

    
    Mat fullHsvImage;
    cvtColor(orgimg, fullHsvImage, cv::COLOR_BGR2HSV);
    split(fullHsvImage, hsvChannels);
    applySobel(hsvChannels[0], fullSobelImg);
    
    //Resizing image
    Mat colImg;
    int factor = 1;
    if ((orgw * orgh) > maxPix) {
        factor = orgw * orgh / maxPix;
        Size scaledSize = Size(int(orgw / factor), int(orgh / factor));
        resize(orgimg, colImg, scaledSize);
        resize(fullSobelImg, scaledSobelImg, scaledSize);
    }
    else {
        colImg = orgimg;
        scaledSobelImg = fullSobelImg;
    }

    int colw = colImg.cols;
    int colh = colImg.rows;

    Mat3b regionImg(colh, colw);
    Mat hsvImg;
    cvtColor(colImg, hsvImg, cv::COLOR_BGR2HSV);
    Point2i neighborhood[8] = { Point2i(1,0),Point2i(-1,0),Point2i(0,1),Point2i(0,-1),
                                Point2i(-1,-1),Point2i(1,1),Point2i(-1,1),Point2i(1,-1) };

    if (seeddensity.x > colw) {
        seeddensity.x = colw;
    }

    if (seeddensity.y > colh) {
        seeddensity.y = colh;
    }

    double stepSizeX = (double)colw / seeddensity.x;
    double stepSizeY = (double)colh / seeddensity.y;

    int minRegionSize = 6;

    //Mat allRegionBuffer;
    //Mat allRegionBuffer(colh, colw, CV_8U);
    Mat zeroColBuffer = Mat::zeros(colh, colw, CV_8U);
    Mat allRegionBuffer = Mat::zeros(colh, colw, CV_8U);

    Mat regionBuffer = Mat::zeros(colh, colw, CV_8U);;
    Mat scaledRegionBuffer = Mat::zeros(orgh, orgw, CV_8U);

    Point2i selectedSeedPoint;
    int regionSize = 0;
    for (int x = 0; x < seeddensity.x; x++) {
        for (int y = 0; y < seeddensity.y; y++) {
            selectedSeedPoint.x = (int)(x*stepSizeX);
            selectedSeedPoint.y = (int)(y*stepSizeY);

            //regionBuffer = Mat(zeroColBuffer);
            zeroColBuffer.copyTo(regionBuffer);

            if ((int)allRegionBuffer.at<uchar>(selectedSeedPoint.y, selectedSeedPoint.x) == 0) {
                regionSize = growRegionColMin(hsvImg, scaledSobelImg, selectedSeedPoint, neighborhood, maxShift, regionBuffer, allRegionBuffer, regionImg);
            
                if (regionSize > maxPix/1000) {
                    resize(regionBuffer, scaledRegionBuffer, Size(orgw, orgh),0,0, INTER_NEAREST);
                    char windowname[100];
                    int blurSize = 8;
                    //blur(scaledRegionBuffer, scaledRegionBuffer, Size(factor * blurSize, factor * blurSize));
                    //medianBlur(scaledRegionBuffer, scaledRegionBuffer, 21);
                    //GaussianBlur(scaledRegionBuffer, scaledRegionBuffer, Size(factor * blurSize, factor * blurSize), 4, 4);
                    //threshold(scaledRegionBuffer, scaledRegionBuffer, (int)(100*5/ blurSize), 255, THRESH_BINARY);
                    //threshold(scaledRegionBuffer, scaledRegionBuffer, 120, 255, THRESH_BINARY);
                    //GaussianBlur(scaledRegionBuffer, scaledRegionBuffer, Size(5, 5), 0.6, 0.6);
                    sprintf(windowname, "region res %d %d", selectedSeedPoint.x, selectedSeedPoint.y);
                    namedWindow(windowname, WINDOW_AUTOSIZE);// Create a window for display.
                    imshow(windowname, scaledRegionBuffer);
                }
            }
        }
    }

    zeroColBuffer.release();
    regionBuffer.release();

    Mat result1, result2, result3;
    resize(regionImg, result1, Size(orgw,orgh),0,0,0);
    resize(hsvImg, result2, Size(orgw, orgh), 0, 0, 0);
    cvtColor(result1, result3, cv::COLOR_HSV2BGR);
    //blur(result3, result3, Size(3, 3));

    namedWindow("Displayxx window", WINDOW_AUTOSIZE);// Create a window for display.
    imshow("Displayxx window", result3);
    namedWindow("Display window", WINDOW_AUTOSIZE);// Create a window for display.
    imshow("Display window", result1);
    namedWindow("Displayhsv window", WINDOW_AUTOSIZE);// Create a window for display.
    imshow("Displayhsv window", result2);
    namedWindow("Displayxx window", WINDOW_AUTOSIZE);// Create a window for display.
    imshow("Displayxx window", result3);
    namedWindow("Display2 window", WINDOW_AUTOSIZE);// Create a window for display.
    imshow("Display2 window", fullSobelImg);
    namedWindow("Display3 window", WINDOW_AUTOSIZE);// Create a window for display.
    imshow("Display3 window", orgimg);
    namedWindow("Display4 window", WINDOW_AUTOSIZE);// Create a window for display.
    imshow("Display4 window", colImg);

    return regionImg;
}

int main(int argc, char** argv)
{

    cout << " TEst --------------------------" << endl;

    /*if (argc != 2)
    {
        cout << " Usage: display_image ImageToLoadAndDisplay" << endl;
        return -1;
    }*/

    Mat image;

    string filepath = "C:\\Users\\Max\\Pictures\\53146095_10155809091156428_4847299437430571008_n.jpg";
    image = imread(filepath, cv::IMREAD_COLOR);
    //image = imread(argv[1], CV_LOAD_IMAGE_COLOR);   // Read the file

    if (!image.data)                              // Check for invalid input
    {
        cout << "Could not open or find the image" << std::endl;
        return -1;
    }

    growRegionGrid(image, Point2i(1000, 1000), Vec3b(20,50,100));

    //namedWindow("Display window", WINDOW_AUTOSIZE);// Create a window for display.
    //imshow("Display window", image);                   // Show our image inside it.

    waitKey(0);                                          // Wait for a keystroke in the window
    return 0;
}