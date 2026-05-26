// CGRA352-Assignment4.cpp : This file contains the 'main' function. Program execution begins and ends there.

#include <iostream>
#include <random>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

std::vector<cv::Mat> load_images(std::string filepath) {
	// parse all images
	std::vector<cv::Mat> images;
	std::cout << "Loading images ..." << std::endl;
	std::vector<cv::String> lf_imgs;
	cv::glob(filepath, lf_imgs);
	for (cv::String cv_str : lf_imgs) {
		// get the filepath
		std::string filepath(cv_str);
		size_t pos = filepath.find_last_of("/\\");

		if (pos != std::string::npos) {
			std::string filename = filepath.substr(pos + 1);

			// parse for values
			std::istringstream ss(filename);

			if (ss.good()) {
				cv::Mat current = cv::imread(filepath);
				images.push_back(current);

			}
			else {
				// throw error otherwise
				std::cerr << "Filepath error with : " << filepath << std::endl;
				abort();
			}
		}

	}

	std::cout << "Finished loading light field" << std::endl;
	return images;
}

int main()
{

	std::vector<cv::Mat> frames = load_images("frames\\*.jpg");
	
	cv::Mat img_1 = frames[39];
	cv::Mat img_2 = frames[41];

	cv::Ptr<cv::SIFT> sift = sift->create();

	std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
	sift->detect(img_1, keypoints_1);
	sift->detect(img_2, keypoints_2);

	cv::Mat descriptors_1,descriptors_2;
	sift->compute(img_1, keypoints_1, descriptors_1);
	sift->compute(img_2, keypoints_2, descriptors_2);

	cv::BFMatcher matcher(cv::NORM_L2,true);
	std::vector<cv::DMatch> matches;

	matcher.match(descriptors_1, descriptors_2, matches);

	cv::Mat concat_img = img_1.clone();
	cv::vconcat(img_1, img_2, concat_img);
	for (cv::DMatch match : matches) {

		cv::Point top_point = keypoints_1[match.queryIdx].pt;
		cv::Point bottom_point = keypoints_2[match.trainIdx].pt;
		cv::Point true_bottom_point = cv::Point(bottom_point.x, bottom_point.y + img_1.rows);

		cv::line(concat_img, top_point, true_bottom_point, cv::Scalar(0, 255, 0));
	}

	cv::imshow("Core Part 1", concat_img);
	cv::waitKey(0);

	


	cv::warpPerspective(img_1, img_2, best_homography);
	

}


cv::Mat calculate_h(std::vector<cv::DMatch> matches, std::vector<cv::KeyPoint> keypoints_1, std::vector<cv::KeyPoint> keypoints_2) {
	cv::Mat best_homography;
	std::vector<int> best_inliers;
	float epsilon = 10.0;
	std::random_device rd;
	std::mt19937 rand(rd());

	//RANSAC
	std::uniform_int_distribution<> rand_int(0, matches.size());
	for (int i = 0; i < 100; i++) {

		//Random samples
		//select 4 random matches
		std::vector<int> random_matches_index;
		random_matches_index.push_back(rand_int(rand));
		random_matches_index.push_back(rand_int(rand));
		random_matches_index.push_back(rand_int(rand));
		random_matches_index.push_back(rand_int(rand));

		// Compute H from these 4 matches
		std::vector<cv::Point> r1, r2;
		for (int i : random_matches_index) {
			cv::DMatch match = matches[random_matches_index[i]];
			r1.push_back(keypoints_1[match.queryIdx].pt);
			r2.push_back(keypoints_2[match.trainIdx].pt);
		}

		cv::Mat h = cv::findHomography(r1, r2);

		std::vector<int> inliers;
		for (cv::DMatch match : matches) {

		}

	}
}
