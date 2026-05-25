// CGRA352-Assignment4.cpp : This file contains the 'main' function. Program execution begins and ends there.

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

int main()
{
    cv::Mat test = cv::imread("frames/Frame000.jpg");
    cv::imshow("daksd", test);
    cv::waitKey(0);
    std::cout << "Hello World!\n";
}
