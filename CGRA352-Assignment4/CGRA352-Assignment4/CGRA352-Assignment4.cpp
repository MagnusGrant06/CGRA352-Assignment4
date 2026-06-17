// CGRA352-Assignment4.cpp : This file contains the 'main' function. Program execution begins and ends there.

#include <iostream>
#include <random>
#include <algorithm>
#include <iomanip>
#include <sstream>

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

//output the images to the disk with the format Stable000.png
void save_images(std::vector<cv::Mat> images, const std::string& output_name) {

	int i = 0;
	for (const cv::Mat& img : images) {

		if (img.empty()) {
			std::cerr << " image file is empty, could not save" << std::endl;
		}

		std::ostringstream filename;
		filename << output_name << std::setw(3) << std::setfill('0') << i << ".png";

		cv::imwrite(filename.str(), img);
		i++;

	}
}

//calculates homographic transformation matrix from matches and keypoints
cv::Mat calculate_h(std::vector<cv::DMatch> matches, std::vector<cv::KeyPoint> keypoints_1, std::vector<cv::KeyPoint> keypoints_2) {
	std::cout << "matches size: " << matches.size() << std::endl;
	cv::Mat best_homography;
	float epsilon = 50.0;
	std::random_device rd;
	std::mt19937 rand(rd());
	best_inliers.clear();
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
		//loop through all matches and see if we have more inliers than previous iteration
		for (cv::DMatch match : matches) {
			cv::Point2f top_point = keypoints_1[match.queryIdx].pt;
			cv::Point2f bottom_point = keypoints_2[match.trainIdx].pt;

			std::vector<cv::Point2f> src = { top_point }, dst;
			cv::perspectiveTransform(src, dst, h); //distance from matching points

			float dx = dst[0].x - bottom_point.x;
			float dy = dst[0].y - bottom_point.y;

			float dist = sqrt(dx * dx + dy * dy);

			if (dist < epsilon) {
				inliers.push_back(match);
			}
		}

		if (inliers.size() > best_inliers.size()) {
			best_homography = h;
			best_inliers = inliers;
		}

	}

	std::vector<cv::Point2f> best_r1, best_r2;
	for (cv::DMatch match : best_inliers) {
		best_r1.push_back(keypoints_1[match.queryIdx].pt);
		best_r2.push_back(keypoints_2[match.trainIdx].pt);
	}

	return cv::findHomography(best_r1, best_r2);
}

void draw_matches(cv::Mat img_1, cv::Mat img_2, std::vector<cv::DMatch> matches, std::vector<cv::KeyPoint> keypoints_1, std::vector<cv::KeyPoint> keypoints_2) {
	cv::Mat concat_img = img_1.clone();
	cv::vconcat(img_1.clone(), img_2.clone(), concat_img);

	//draw lines between matches using their keypoints and cv::line
	for (cv::DMatch match : matches) {

		cv::Point top_point = keypoints_1[match.queryIdx].pt;
		cv::Point bottom_point = keypoints_2[match.trainIdx].pt;
		cv::Point true_bottom_point = cv::Point(bottom_point.x, bottom_point.y + img_1.rows);

		cv::line(concat_img, top_point, true_bottom_point, cv::Scalar(0, 255, 0));
	}

	cv::imshow("Core Part 1", concat_img);
	cv::waitKey(0);

}

void draw_inliers(cv::Mat img_1, cv::Mat img_2, std::vector<cv::DMatch> matches, std::vector<cv::KeyPoint> keypoints_1, std::vector<cv::KeyPoint> keypoints_2) {

	cv::Mat homogaphy_concat_img = img_1.clone();
	cv::vconcat(img_1.clone(), img_2.clone(), homogaphy_concat_img);

	//draw lines between matches, green if they are inliers red otherwise
	for (cv::DMatch match : matches) {
		cv::Point top_point = keypoints_1[match.queryIdx].pt;
		cv::Point bottom_point = keypoints_2[match.trainIdx].pt;
		cv::Point true_bottom_point = cv::Point(bottom_point.x, bottom_point.y + img_1.rows);

		auto found = std::find_if(best_inliers.begin(), best_inliers.end(), [&](const cv::DMatch& m) { //lambda to compare match indexes as DMatches dont overload ==
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

}

//primary method to calculate homographic translation, using SIFT and a brute force matcher
cv::Mat compute_homographic_transformation(cv::Mat img_1, cv::Mat img_2, bool draw_examples) {
	cv::Ptr<cv::SIFT> sift = sift->create();

	std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
	sift->detect(img_1, keypoints_1);
	sift->detect(img_2, keypoints_2);

	cv::Mat descriptors_1, descriptors_2;
	sift->compute(img_1, keypoints_1, descriptors_1);
	sift->compute(img_2, keypoints_2, descriptors_2);

	cv::BFMatcher matcher(cv::NORM_L2, true);
	std::vector<cv::DMatch> matches;

	matcher.match(descriptors_1, descriptors_2, matches);

	if (draw_examples) { //for assignment output
		draw_matches(img_1, img_2, matches, keypoints_1, keypoints_2);
	}

	cv::Mat h = calculate_h(matches, keypoints_1, keypoints_2);

	if (draw_examples) { //for assignment output
		draw_inliers(img_1, img_2, matches, keypoints_1, keypoints_2);
	}

	return h;
}

void find_best_cropping_windows(const std::vector<cv::Mat>& frames, const std::vector<cv::Mat>& u_transforms) {
	cv::Size frame_size = frames[0].size();
	//challenge

	cv::Mat full_mask = cv::Mat::ones(frame_size, CV_8U) * 255;
	//warpperspective to get mask out
	for (size_t i = { 0 }; i < u_transforms.size(); ++i) {
		cv::Mat white = cv::Mat::ones(frames[i].size(), CV_8U) * 255;
		cv::Mat mask;
		cv::warpPerspective(white, mask, u_transforms[i], frames[i].size());
		cv::bitwise_and(full_mask, mask, full_mask);
	}

	//resize to square
	cv::Mat square_mask;
	//use cv::resize to get full mask to square mask
	cv::resize(full_mask, square_mask, cv::Size(frame_size.width, frame_size.width));

	//dynamic programming
	//find biggest inscribed square using mask
	cv::Mat S = cv::Mat::zeros(square_mask.size(), CV_32S);
	int best_size = 0;
	int best_r = 0;
	int best_c = 0;
	for (int r = square_mask.rows - 1; r >= 0; r--) {
		for (int c = square_mask.rows - 1; c >= 0; c--) {
			if (square_mask.at<uchar>(r, c) == 255) {
				if (r == square_mask.rows - 1 || c == square_mask.cols - 1) {
					S.at<int>(r, c) = 1;

				}
				else {
					S.at<int>(r, c) = std::min({
						S.at<int>(r + 1, c),
						S.at<int>(r, c + 1),
						S.at<int>(r + 1, c + 1)
						}) + 1;
				}

				if (S.at<int>(r, c) > best_size) {
					best_size = S.at<int>(r, c);
					best_r = r;
					best_c = c;
				}
			}
		}
	}

	//use minMaxLoc
	std::vector<cv::Mat> output;
	
	//scale square mask back to rectangle
	float x_scale = (float)full_mask.cols / 800; // 800/800 = 1.0
	float y_scale = (float)full_mask.rows / 800; // 450/800 = 0.5625

	cv::Rect rectangle(
		best_c * x_scale,
		best_r * y_scale,
		best_size * x_scale,  //width scaled by x
		best_size * y_scale   //height scaled by y
	);

	for (cv::Mat frame : frames) {
		output.push_back(frame(rectangle));
	}

	save_images(output, std::string("Cropped"));

	cv::imshow("Cropped frame 031", output[31]);
	cv::imshow("Cropped frame 040", output[40]);
	cv::waitKey(0);
}

void create_stabilised_frames(std::vector<cv::Mat> frames) {

	//create homographic transformations for each frame pair
	std::vector<cv::Mat> h_transforms;
	h_transforms.push_back(cv::Mat::eye(3, 3, CV_64FC1));
	for (int i = 1; i < frames.size(); i++) {
		std::cout << i << std::endl;
		cv::Mat current_h = compute_homographic_transformation(frames[i], frames[i - 1], false);
		h_transforms.push_back(current_h.clone());
	}

	//calculate and create cumulative homographic transformations for each frame
	std::vector<cv::Mat> h_tilde_transforms;
	cv::Mat cumulative = cv::Mat::eye(3, 3, CV_64FC1);
	for (int i = 0; i < frames.size(); i++) {
		cumulative = cumulative * h_transforms[i];
		h_tilde_transforms.push_back(cumulative.clone());
	}

	//do a weighted average of the sorrounding translations to get a smoother transition
	std::vector<float> weights = { 0.1, 0.3, 0.5, 0.3, 0.1 }; //guassian weighting
	std::vector<cv::Mat> h_smooth_transitions;
	for (int i = 0; i < h_tilde_transforms.size(); i++) {

		cv::Mat smoothed = cv::Mat::zeros(3, 3, CV_64F);
		float weighted_sum = 0;

		for (int j = -2; j <= 2; j++) {
			int window_index = i + j;
			std::cout << int(frames.size() - 1) << " next window " << window_index << std::endl;
			window_index = window_index = std::min(std::max(0, window_index), int(h_tilde_transforms.size() - 1));

			smoothed += weights[j + 2] * h_tilde_transforms[window_index];
			weighted_sum += weights[j + 2];
		}

		h_smooth_transitions.push_back(smoothed / weighted_sum);
	}

	//turn smoothed transitions into actual translations to be used on images
	std::vector<cv::Mat> u_transforms;
	for (int i = 0; i < h_smooth_transitions.size(); i++) {
		std::cout << i << std::endl;
		cv::Mat U_i = h_smooth_transitions[i].inv() * h_tilde_transforms[i];
		u_transforms.push_back(U_i);
	}

	std::vector<cv::Mat> stabilised_frames;
	for (size_t i = { 0 }; i < u_transforms.size(); ++i) {
		cv::Mat output;
		cv::warpPerspective(frames[i], output, u_transforms[i], frames[i].size());
		stabilised_frames.push_back(output);
	}

	std::cout << stabilised_frames.size() << std::endl;
	save_images(stabilised_frames, std::string("Stable"));

	cv::imshow("Stabilised frame 031", stabilised_frames[31]);
	cv::imshow("Stabilised frame 040", stabilised_frames[40]);
	cv::waitKey(0);

	find_best_cropping_windows(stabilised_frames, u_transforms);

}


int main()
{

	std::vector<cv::Mat> frames = load_images("frames\\*.jpg");
	
	cv::Mat img_1 = frames[41];
	cv::Mat img_2 = frames[39];

	cv::Mat h = compute_homographic_transformation(img_1, img_2, true);


	int border = 100; //padding around the outside

	//offset matrix to push img_2 inward
	cv::Mat translation = (cv::Mat_<double>(3, 3) <<
		1, 0, border,
		0, 1, border,
		0, 0, 1);

	//make canvas big enough to include the border on all sides
	cv::Size output_size(img_1.cols + border * 2, std::max(img_1.rows, img_2.rows) + border * 2);
	cv::Mat output(output_size, img_1.type(), cv::Scalar(0, 0, 0)); // green background

	//copy img_2 offset by the border
	img_2.copyTo(output(cv::Rect(border, border, img_2.cols, img_2.rows)));

	// apply translation to homography
	cv::Mat shifted_h = translation * h;
	cv::warpPerspective(img_1, output, shifted_h, output_size,
		cv::INTER_LINEAR, cv::BORDER_TRANSPARENT);

	cv::imshow("stitched", output);
	cv::waitKey(0);

	create_stabilised_frames(frames);


}



