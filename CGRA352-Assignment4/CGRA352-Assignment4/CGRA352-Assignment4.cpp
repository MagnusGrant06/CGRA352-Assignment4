// CGRA352-Assignment4.cpp : This file contains the 'main' function. Program execution begins and ends there.

#include <iostream>
#include <random>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

std::vector<cv::DMatch> best_inliers;

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

cv::Mat calculate_h(std::vector<cv::DMatch> matches, std::vector<cv::KeyPoint> keypoints_1, std::vector<cv::KeyPoint> keypoints_2) {
	cv::Mat best_homography;
	float epsilon = 50.0;
	std::random_device rd;
	std::mt19937 rand(rd());

	//RANSAC
	std::uniform_int_distribution<> rand_int(0, matches.size() - 1);
	for (int i = 0; i < 100; i++) {

		//Random samples
		//select 4 random matches
		std::vector<int> random_matches_index;
		random_matches_index.push_back(rand_int(rand));
		random_matches_index.push_back(rand_int(rand));
		random_matches_index.push_back(rand_int(rand));
		random_matches_index.push_back(rand_int(rand));

		// Compute H from these 4 matches
		std::vector<cv::Point2f> r1, r2;
		for (int idx : random_matches_index) {
			cv::DMatch match = matches[idx];
			r1.push_back(keypoints_1[match.queryIdx].pt);
			r2.push_back(keypoints_2[match.trainIdx].pt);
		}

		cv::Mat h = cv::findHomography(r1, r2);

		std::vector<cv::DMatch> inliers;
		for (cv::DMatch match : matches) {
			cv::Point2f top_point = keypoints_1[match.queryIdx].pt;
			cv::Point2f bottom_point = keypoints_2[match.trainIdx].pt;

			std::vector<cv::Point2f> src = { top_point }, dst;
			cv::perspectiveTransform(src, dst, h);

			float dx = dst[0].x - bottom_point.x;
			float dy = dst[0].y - bottom_point.y;

			float dist = sqrt(dx * dx - dy * dy);

			if (dist < epsilon) {
				inliers.push_back(match);
			}
		}

		if (inliers.size() > best_inliers.size()) {
			best_homography = h;
			best_inliers = inliers;
		}

	}
	return best_homography;
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
	cv::vconcat(img_1.clone(), img_2.clone(), concat_img);
	for (cv::DMatch match : matches) {

		cv::Point top_point = keypoints_1[match.queryIdx].pt;
		cv::Point bottom_point = keypoints_2[match.trainIdx].pt;
		cv::Point true_bottom_point = cv::Point(bottom_point.x, bottom_point.y + img_1.rows);

		cv::line(concat_img, top_point, true_bottom_point, cv::Scalar(0, 255, 0));
	}

	cv::imshow("Core Part 1", concat_img);
	cv::waitKey(0);
	
	
	//core part 2

	cv::Mat homogaphy_concat_img = img_1.clone();
	cv::vconcat(img_1.clone(), img_2.clone(), homogaphy_concat_img);

	cv::Mat h = calculate_h(matches, keypoints_1, keypoints_2);
	
	for (cv::DMatch match : matches) {
		cv::Point top_point = keypoints_1[match.queryIdx].pt;
		cv::Point bottom_point = keypoints_2[match.trainIdx].pt;
		cv::Point true_bottom_point = cv::Point(bottom_point.x, bottom_point.y + img_1.rows);

		auto found = std::find_if(best_inliers.begin(), best_inliers.end(), [&](const cv::DMatch& m) {
			return m.queryIdx == match.queryIdx;
			});

		if (found != best_inliers.end()) {
			cv::line(homogaphy_concat_img, top_point, true_bottom_point, cv::Scalar(0, 255, 0));
		}
		else {
			cv::line(homogaphy_concat_img, top_point, true_bottom_point, cv::Scalar(0, 0, 255));
		}
	}
	
	cv::imshow("best inliers", homogaphy_concat_img);
	cv::waitKey(0);
	cv::Mat warped;
	//cv::warpPerspective(img_1, warped, h, output_size);
	// create a big empty output
	cv::Mat output = cv::Mat::zeros(cv::Size(img_1.cols+50, img_1.rows+50), img_1.type());

	// copy img_2 into the top left
	img_2.copyTo(output(cv::Rect(25, 25, img_2.cols, img_2.rows)));

	// warp img_1 on top
	cv::warpPerspective(img_1, output, h, cv::Size(img_1.cols + 50, img_1.rows + 50),
		cv::INTER_LINEAR);
	cv::imshow("warped", output);
	cv::waitKey(0);

}



