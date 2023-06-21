#include <iostream>
#include <cstring>
#include <string.h>
#include <math.h>
#include <cmath>
#include <vector>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <Eigen/Dense>

using namespace cv;
using namespace Eigen;
using namespace Eigen::internal;
using namespace Eigen::Architecture;
using namespace std;

void mean_and_std(Mat img, Mat mask, float m[], float s[]) {

	// Mean
	float b = 0.0, g = 0.0, r = 0.0;
	int count = 0;
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if ((mask.at<uchar>(i, j) != 0)) {
				b += img.at<Vec3b>(i, j)[0];
				g += img.at<Vec3b>(i, j)[1];
				r += img.at<Vec3b>(i, j)[2];
				count++;
			}
		}
	}
	m[0] = b / count;
	m[1] = g / count;
	m[2] = r / count;

	// STD
	b = 0.0, g = 0.0, r = 0.0;
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if ((mask.at<uchar>(i, j) != 0)) {
				b += (img.at<Vec3b>(i, j)[0] - m[0]) * (img.at<Vec3b>(i, j)[0] - m[0]);
				g += (img.at<Vec3b>(i, j)[1] - m[1]) * (img.at<Vec3b>(i, j)[1] - m[1]);
				r += (img.at<Vec3b>(i, j)[2] - m[2]) * (img.at<Vec3b>(i, j)[2] - m[2]);
			}
		}
	}
	s[0] = sqrt(b / count);
	s[1] = sqrt(g / count);
	s[2] = sqrt(r / count);
}

void color_trans(Mat img, Mat mask, float a, float b, int channel) {
	int row = img.rows;
	int col = img.cols;
	int count = 0;
	float temp;
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			if (mask.at<uchar>(i, j) != 0) {
				temp = (float)img.at<Vec3b>(i, j)[channel];
				temp = temp * a + b;
				img.at<Vec3b>(i, j)[channel] = saturate_cast<uchar>(temp);
			}
		}
	}
}

int cal_adw_len(int row_len, int col_len, double tar_p, double ref_p) {
	double min = tar_p < ref_p ? tar_p : ref_p;
	min = sqrt(row_len * col_len * min);
	min = (int)min % 2 == 0 ? (int)min - 1 : (int)min;
	return (int)min;
}

float cal_block(Mat img, int channel, int x, int y, int i, int j, int half, int num) {
	// ignore vertical & horizontal & center pixel
	float val = 0;
	x += i * half;
	y += j * half;
	switch (num) {
	case 1:
		for (int i = x - half; i < x; i++) {
			for (int j = y - half; j < y; j++) {
				val += img.at<Vec3b>(i, j)[channel];
			}
		}
		break;
	case 2:
		for (int i = x - half; i < x; i++) {
			for (int j = y + 1; j < y + half; j++) {
				val += img.at<Vec3b>(i, j)[channel];
			}
		}
		break;
	case 3:
		for (int i = x + 1; i < x + half; i++) {
			for (int j = y - half; j < y; j++) {
				val += img.at<Vec3b>(i, j)[channel];
			}
		}
		break;
	case 4:
		for (int i = x + 1; i < x + half; i++) {
			for (int j = y + 1; j < y + half; j++) {
				val += img.at<Vec3b>(i, j)[channel];
			}
		}
		break;
	}
	return val;
}

void gamma(Mat global_img, int channel, int x, int y, float coef) {
	float result = (float)(global_img.at<Vec3b>(x, y)[channel]) / 255.0;
	global_img.at<Vec3b>(x, y)[channel] = saturate_cast<uchar>(pow(result, coef) * 255.0f);
}

void get_center_pos(int* center_pos, int center_num, int row_count, int col_count, float* x, float* y) {
	int a, b;
	a = (center_num / col_count) % row_count;
	b = (center_num % col_count);
	*x = *(center_pos + a * col_count * 2 + b * 2);
	*y = *(center_pos + a * col_count * 2 + b * 2 + 1);
}

Mat img_bgr_to_g(Mat img) {
	Mat temp;
	cvtColor(img, temp, COLOR_BGR2GRAY);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			temp.at<uchar>(i, j) = temp.at<uchar>(i, j) != 0 ? 255 : 0;
		}
	}
	return temp;
}

void bilinear(Mat global_img, Mat tar_region, int channel, Mat tar_img, Mat ref_img, float* tar_adw_mean, float* ref_adw_mean, int* center_pos, int row_count, int col_count) {

	int p1, p2, p3, p4;	// center num
	float p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y;	// center pos
	float tar_bi, ref_bi, coef;
	
	for (int i = 0; i < row_count - 1; i++) {
		for (int j = 0; j < col_count - 1; j++) {
			// cal square cor
			// first	: get 4 center's number.			  
			p1 = i * col_count + j, p2 = p1 + 1, p3 = p1 + col_count, p4 = p3 + 1;
			//cout << channel << "[" << p1 << "," << p2 << "," << p3 << "," << p4 << "]\n";
			// second	: get 4 center & first pixel's pos (row = p1's row+1, col = p1's col+1)
			get_center_pos(center_pos, p1, row_count, col_count, &p1x, &p1y);
			get_center_pos(center_pos, p2, row_count, col_count, &p2x, &p2y);
			get_center_pos(center_pos, p3, row_count, col_count, &p3x, &p3y);
			get_center_pos(center_pos, p4, row_count, col_count, &p4x, &p4y);

			// third : pixel by pixel bilinear then gamma
			// middle region
			for (float x = p1x + 1.0; x < p3x; x++) {
				for (float y = p1y + 1.0; y < p2y; y++) {
					// tar's bilinear
					tar_bi = ((x - p1x) / (p3x - p1x)) * (((p4y - y) / (p4y - p3y) * *(tar_adw_mean + p3)) + ((y - p3y) / (p4y - p3y) * *(tar_adw_mean + p4))) +
						((p3x - x) / (p3x - p1x)) * (((p2y - y) / (p2y - p1y) * *(tar_adw_mean + p1)) + ((y - p1y) / (p4y - p3y) * *(tar_adw_mean + p2)));
					// ref's bilinear
					ref_bi = ((x - p1x) / (p3x - p1x)) * (((p4y - y) / (p4y - p3y) * *(ref_adw_mean + p3)) + ((y - p3y) / (p4y - p3y) * *(ref_adw_mean + p4))) +
						((p3x - x) / (p3x - p1x)) * (((p2y - y) / (p2y - p1y) * *(ref_adw_mean + p1)) + ((y - p1y) / (p4y - p3y) * *(ref_adw_mean + p2)));
					coef = tar_region.at<uchar>(x, y) == 255 ? log(tar_bi) / log(ref_bi) : log(ref_bi) / log(tar_bi);
					//gamma(global_img, channel, x, y, coef);
					gamma(tar_img, channel, x, y, log(tar_bi) / log(ref_bi));
					gamma(ref_img, channel, x, y, log(ref_bi) / log(tar_bi));
				}
			}
			// Right			
			for (float y = p1y + 1; y < p2y; y++) {
				tar_bi = ((p2y - y) / (p2y - p1y) * *(tar_adw_mean + p1)) + ((y - p1y) / (p2y - p1y) * *(tar_adw_mean + p2));
				ref_bi = ((p2y - y) / (p2y - p1y) * *(ref_adw_mean + p1)) + ((y - p1y) / (p2y - p1y) * *(ref_adw_mean + p2));
				coef = tar_region.at<uchar>(p1x, y) == 255 ? log(tar_bi) / log(ref_bi) : log(ref_bi) / log(tar_bi);
				//gamma(global_img, channel, p1x, y, coef);
				gamma(tar_img, channel, p1x, y, log(tar_bi) / log(ref_bi));
				gamma(ref_img, channel, p1x, y, log(ref_bi) / log(tar_bi));
			}
			// Down
			for (float x = p1x + 1; x < p3x; x++) {
				tar_bi = ((p3x - x) / (p3x - p1x) * *(tar_adw_mean + p1)) + ((x - p1x) / (p3x - p1x) * *(tar_adw_mean + p3));
				ref_bi = ((p3x - x) / (p3x - p1x) * *(ref_adw_mean + p1)) + ((x - p1x) / (p3x - p1x) * *(ref_adw_mean + p3));
				coef = tar_region.at<uchar>(x, p1y) == 255 ? log(tar_bi) / log(ref_bi) : log(ref_bi) / log(tar_bi);
				//gamma(global_img, channel, x, p1y, coef);
				gamma(tar_img, channel, x, p1y, log(tar_bi) / log(ref_bi));
				gamma(ref_img, channel, x, p1y, log(ref_bi) / log(tar_bi));
			}
			// next col's Down (edge case for last col)
			if (j == col_count - 2) {
				for (float x = p2x + 1; x < p4x; x++) {
					tar_bi = ((p4x - x) / (p4x - p2x) * *(tar_adw_mean + p2)) + ((x - p2x) / (p4x - p2x) * *(tar_adw_mean + p4));
					ref_bi = ((p4x - x) / (p4x - p2x) * *(ref_adw_mean + p2)) + ((x - p2x) / (p4x - p2x) * *(ref_adw_mean + p4));
					coef = tar_region.at<uchar>(x, p2y) == 255 ? log(tar_bi) / log(ref_bi) : log(ref_bi) / log(tar_bi);
					//gamma(global_img, channel, x, p2y, coef);
					gamma(tar_img, channel, x, p2y, log(tar_bi) / log(ref_bi));
					gamma(ref_img, channel, x, p2y, log(ref_bi) / log(tar_bi));
				}
			}
			// next row's Right (edge case for last row)
			if (i == row_count - 2) {
				for (float y = p3y + 1; y < p4y; y++) {
					tar_bi = ((p4y - y) / (p4y - p3y) * *(tar_adw_mean + p3)) + ((y - p3y) / (p4y - p3y) * *(tar_adw_mean + p4));
					ref_bi = ((p4y - y) / (p4y - p3y) * *(ref_adw_mean + p3)) + ((y - p3y) / (p4y - p3y) * *(ref_adw_mean + p4));
					coef = tar_region.at<uchar>(p3x, y) == 255 ? log(tar_bi) / log(ref_bi) : log(ref_bi) / log(tar_bi);
					//gamma(global_img, channel, p3x, y, coef);
					gamma(tar_img, channel, p3x, y, log(tar_bi) / log(ref_bi));
					gamma(ref_img, channel, p3x, y, log(ref_bi) / log(tar_bi));
				}
			}
			/// center (ignore first & last row & col)
			if ((i != row_count - 2) && (j != col_count - 2)) {
				p1 = p4 + 1, p4 = p4 + col_count;
				tar_bi = (*(tar_adw_mean + p1) + *(tar_adw_mean + p1) + *(tar_adw_mean + p1) + *(tar_adw_mean + p4)) / 4;
				ref_bi = (*(ref_adw_mean + p1) + *(ref_adw_mean + p1) + *(ref_adw_mean + p1) + *(ref_adw_mean + p4)) / 4;
				coef = tar_region.at<uchar>(p4x, p4y) == 255 ? log(tar_bi) / log(ref_bi) : log(ref_bi) / log(tar_bi);
				//gamma(global_img, channel, p4x, p4y, coef);
				gamma(tar_img, channel, p4x, p4y, log(tar_bi) / log(ref_bi));
				gamma(ref_img, channel, p4x, p4y, log(ref_bi) / log(tar_bi));
			}
		}
	}
	// change warp for both tar and ref
	/*
	float first[2], last[2];
	get_center_pos(center_pos, 0, row_count, col_count, &first[0], &first[1]);
	get_center_pos(center_pos, row_count * col_count - 1, row_count, col_count, &last[0], &last[1]);
	for (int i = first[0]; i < last[0] + 1; i++) {
		for(int j = first[1]; j < last[1] + 1; j++) {
			tar_img.at<Vec3b>(i, j)[channel] = global_img.at<Vec3b>(i, j)[channel];
			ref_img.at<Vec3b>(i, j)[channel] = global_img.at<Vec3b>(i, j)[channel];
		}
	}
	*/
}

void cal_grid(Mat global_img, Mat tar_region, Mat tar_img, Mat ref_img, int x, int y, int row_len, int col_len, int len, int channel) {
	int half = (len - 1) / 2;
	int row_count = (row_len / (len - 1)) + 1;
	int col_count = (col_len / (len - 1)) + 1;

	cout << "len : " << len << "\n";
	cout << "row_count : " << row_count << "\n";
	cout << "col_count : " << col_count << "\n\n";

	// cal each center cooridinate's correspound position 
	int* center_pos = new int[row_count * col_count * 2];

	for (int i = 0; i < row_count; i++) {
		for (int j = 0; j < col_count; j++) {
			*(center_pos + i * col_count * 2 + j * 2) = x + half * 2 * i;
			*(center_pos + i * col_count * 2 + j * 2 + 1) = y + half * 2 * j;
		}
	}

	// establish tar & ref mean array size decided by ADW's len
	float* tar_adw_mean = new float[row_count * col_count];
	float* ref_adw_mean = new float[row_count * col_count];
	int block = half * half;
	float temp;
	Mat img;


	for (int ref = 0; ref < 2; ref++) {	// 0 for tar, 1 for ref
		for (int i = 0; i < row_count; i++) {
			for (int j = 0; j < col_count; j++) {
				temp = 0;
				ref == 1 ? ref_img.copyTo(img) : tar_img.copyTo(img);
				if (i == 0) {							// first row ignore : 1 , 2 
					if (j == 0) {					// first col ignore : 1 , 3
						// block 4
						temp = cal_block(img, channel, x, y, i, j, half, 4);
						temp /= block;
					}
					else if (j == col_count - 1) {	// last col ignore : 2 , 4
						// block 3
						temp = cal_block(img, channel, x, y, i, j, half, 3);
						temp /= block;
					}
					else {							// middle col
						// block 3,4
						temp = cal_block(img, channel, x, y, i, j, half, 4);
						temp += temp = cal_block(img, channel, x, y, i, j, half, 3);
						temp /= (block * 2);
					}
				}
				else if (i == row_count - 1) {			// last row : ignore 3 , 4
					if (j == 0) {					// first col ignore : 1 , 3
						// block 2
						temp = cal_block(img, channel, x, y, i, j, half, 2);
						temp /= block;
					}
					else if (j == col_count - 1) {	// last col ignore : 2 , 4
						// block 1
						temp = cal_block(img, channel, x, y, i, j, half, 1);
						temp /= block;
					}
					else {							// middle col
						// block 1,2
						temp = cal_block(img, channel, x, y, i, j, half, 1);
						temp += cal_block(img, channel, x, y, i, j, half, 2);
						temp /= (block * 2);
					}
				}
				else {									// middle row
					if (j == 0) {					// first col ignore : 1 , 3
						// block 2,4
						temp = cal_block(img, channel, x, y, i, j, half, 2);
						temp += cal_block(img, channel, x, y, i, j, half, 4);
						temp /= (block * 2);
					}
					else if (j == col_count - 1) {	// last col ignore : 2 , 4
						// block 1,3
						temp = cal_block(img, channel, x, y, i, j, half, 1);
						temp += cal_block(img, channel, x, y, i, j, half, 3);
						temp /= (block * 2);
					}
					else {							// middle col
						// block 1,2,3,4
						temp = cal_block(img, channel, x, y, i, j, half, 1);
						temp += cal_block(img, channel, x, y, i, j, half, 2);
						temp += cal_block(img, channel, x, y, i, j, half, 3);
						temp += cal_block(img, channel, x, y, i, j, half, 4);
						temp /= (block * 4);
					}
				}
				if (ref) {
					// save ref mean
					*(ref_adw_mean + i * col_count + j) = temp;
				}
				else {
					//save tar mean
					*(tar_adw_mean + i * col_count + j) = temp;
				}
			}	// j = col
		}	// i = row
	}	// ref or tar img to be cal
	// do bilnearr-interpolation
	bilinear(global_img, tar_region, channel, tar_img, ref_img,tar_adw_mean, ref_adw_mean, center_pos, row_count, col_count);
}

void cal_glob_coef(float *img1_mean, float *img1_std, float *img2_mean, float *img2_std, float *img1_coef, float *img2_coef) {
	MatrixXf A(6, 4);
	MatrixXf x(4, 1);
	MatrixXf L(6, 1);

	// i for channel
	for (int i = 0; i < 3; i++) {
		A << img1_mean[i], 1, (-1)* img2_mean[i], (-1),
			img1_std[i], 0, (-1)* img2_std[i], 0,
			img1_mean[i], 1, 0, 0,
			img1_std[i], 0, 0, 0,
			0, 0, img2_mean[i], 1,
			0, 0, img2_std[i], 0;
		L << 0, 0, img1_mean[i], img1_std[i], img2_mean[i], img2_std[i];
		/*	Ax - L = 0 (no - sol)
		*	-> Ax = L (no - sol)
		*	-> A^t x = A^t L (least square sol)
		*/ 
		x = (A.transpose() * A).fullPivLu().solve(A.transpose() * L);

		img1_coef[i * 2] = x(0), img1_coef[i * 2 + 1] = x(1);
		img2_coef[i * 2] = x(2), img2_coef[i * 2 + 1] = x(3);
	}
}

int cal_weight(Mat overlap, Mat region) {
	int count = 0;
	
	for (int i = 0; i < overlap.rows; i++) {
		for (int j = 0; j < overlap.cols; j++) {
			if ((overlap.at<uchar>(i, j) != 0) && (region.at<uchar>(i, j) != 0))
				count++;
		}
	}

	return count;
}

void hole_fill(Mat img, Mat img_0, Mat img_1, Mat img_2) {
	Mat temp;
	cvtColor(img, temp, COLOR_RGB2GRAY);
	/*
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (temp.at<uchar>(i, j) == 0)
				temp.at<uchar>(i, j) = 255;
			else
				temp.at<uchar>(i, j) = 0;
		}
	}
	imwrite("hole.png", temp);
	*/
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (temp.at<uchar>(i, j) == 0) {
				img.at<Vec3b>(i, j)[0] = (float)(img_0.at<Vec3b>(i, j)[0] + img_1.at<Vec3b>(i, j)[0] + img_2.at<Vec3b>(i, j)[0]) / 3;
				img.at<Vec3b>(i, j)[1] = (float)(img_0.at<Vec3b>(i, j)[1] + img_1.at<Vec3b>(i, j)[1] + img_2.at<Vec3b>(i, j)[1]) / 3;
				img.at<Vec3b>(i, j)[2] = (float)(img_0.at<Vec3b>(i, j)[2] + img_1.at<Vec3b>(i, j)[2] + img_2.at<Vec3b>(i, j)[2]) / 3;
			}
		}
	}
}

Mat draw_local(Mat img, int x, int y, int row_len, int col_len, int len) {
	Mat mask;
	cvtColor(img, mask, COLOR_GRAY2BGR);
	int half = (len - 1) / 2;
	int count = 0;
	int flip = 1;	// change color
	int row_count = (row_len / (len - 1)) + 1;
	int col_count = (col_len / (len - 1)) + 1;
	int i, j;
	for (i = x; i < x + ((row_count - 1) * half * 2) + 1; i += 2 * half) {
		for (j = y; j < y + ((col_count - 1) * half * 2) + 1; j += 2 * half) {			
			// center point of each ADW (red)
			count++;
			mask.at<Vec3b>(i, j)[0] = 0;
			mask.at<Vec3b>(i, j)[1] = 0;
			mask.at<Vec3b>(i, j)[2] = 255;
			//cout << count - 1 << " c-pos : [" << i << "," << j << "] draw\n";
			// col-(1)half
			for (int jj = j + 1; (jj < j + half + 1) && (jj < y + ((col_count - 1) * 2 * half) + 1); jj++) {
				// horizontal
				mask.at<Vec3b>(i, jj)[0] = flip == 1 ? 255 : 0;
				mask.at<Vec3b>(i, jj)[1] = flip == 0 ? 255 : 0;
				mask.at<Vec3b>(i, jj)[2] = 0;
				for (int ii = i + 1; (ii < i + half + 1) && (ii < x + ((row_count - 1) * 2 * half) + 1); ii++) {
					// square
					mask.at<Vec3b>(ii, jj)[0] = flip == 1 ? 255 : 0;
					mask.at<Vec3b>(ii, jj)[1] = flip == 0 ? 255 : 0;
					mask.at<Vec3b>(ii, jj)[2] = 0;
				}
			}
			// col-(2)half
			for (int jj = j + half + 1; (jj < j + 2 * half + 1) && (jj < y + ((col_count - 1) * 2 * half) + 1); jj++) {
				// horizontal
				mask.at<Vec3b>(i, jj)[0] = flip == 1 ? 0 : 255;
				mask.at<Vec3b>(i, jj)[1] = flip == 0 ? 0 : 255;
				mask.at<Vec3b>(i, jj)[2] = 0;
				for (int ii = i + 1; (ii < i + half + 1) && (ii < x + ((row_count - 1) * 2 * half) + 1); ii++) {
					// square
					mask.at<Vec3b>(ii, jj)[0] = flip == 1 ? 0 : 255;
					mask.at<Vec3b>(ii, jj)[1] = flip == 0 ? 0 : 255;
					mask.at<Vec3b>(ii, jj)[2] = 0;
				}
			}
			// row-(1)half
			for (int ii = i + 1; (ii < i + half + 1) && (ii < x + ((row_count - 1) * 2 * half) + 1); ii++) {
				// vertical
				mask.at<Vec3b>(ii, j)[0] = flip == 1 ? 255 : 0;
				mask.at<Vec3b>(ii, j)[1] = flip == 0 ? 255 : 0;
				mask.at<Vec3b>(ii, j)[2] = 0;
			}
			// row-(2)half
			for (int ii = i + half + 1; (ii < i + 2 * half + 1) && (ii < x + ((row_count - 1) * 2 * half) + 1); ii++) {
				// vertical
				mask.at<Vec3b>(ii, j)[0] = flip == 1 ? 0 : 255;
				mask.at<Vec3b>(ii, j)[1] = flip == 0 ? 0 : 255;
				mask.at<Vec3b>(ii, j)[2] = 0;
				for (int jj = j + 1; (jj < j + half + 1) && (jj < y + ((col_count - 1) * 2 * half) + 1); jj++) {
					mask.at<Vec3b>(ii, jj)[0] = flip == 1 ? 0 : 255;
					mask.at<Vec3b>(ii, jj)[1] = flip == 0 ? 0 : 255;
					mask.at<Vec3b>(ii, jj)[2] = 0;
				}
				for (int jj = j + half + 1; (jj < j + 2 * half + 1) && (jj < y + ((col_count - 1) * 2 * half) + 1); jj++) {
					mask.at<Vec3b>(ii, jj)[0] = flip == 1 ? 255 : 0;
					mask.at<Vec3b>(ii, jj)[1] = flip == 0 ? 255 : 0;
					mask.at<Vec3b>(ii, jj)[2] = 0;
				}
			}
			flip = flip == 1 ? 0 : 1;
			
		}
		flip = flip == 1 ? 0 : 1;
	}
	//imwrite("local_mask.png", mask);
	cout << "\tcenter count : " << count << endl;
	return mask;
}

float dist(int* a, int* b) {
	int x = a[0] - b[0];
	int y = a[1] - b[1];
	return sqrt((x * x) + (y * y));
}

void find_in_square(Mat img, int* left_up, int* right_up, int* left_down, int* right_down) {
	int row = img.rows;
	int col = img.cols;
	int one[2] = { 0, 0 };
	int two[2] = { 0, col - 1 };
	int three[2] = { row - 1, 0 };
	int four[2] = { row - 1, col - 1 };
	int temp[2] = { 0, 0 };
	int t;
	float one_min = row * col, two_min = one_min, three_min = one_min, four_min = one_min;

	// Step 1 : find corner of img closest point of image corner
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			if (img.at<uchar>(i, j) != 0) {
				temp[0] = i;
				temp[1] = j;
				if (dist(one, temp) < one_min) { left_up[0] = i, left_up[1] = j, one_min = dist(one, temp); }
				if (dist(two, temp) < two_min) { right_up[0] = i, right_up[1] = j, two_min = dist(two, temp); }
				if (dist(three, temp) < three_min) { left_down[0] = i, left_down[1] = j, three_min = dist(three, temp); }
				if (dist(four, temp) < four_min) { right_down[0] = i, right_down[1] = j, four_min = dist(four, temp); }
			}
		}
	}

	// Step 2 : find inner square of img 
	int rm = left_up[0] > right_up[0] ? left_up[0] : right_up[0];
	int rM = left_down[0] < right_down[0] ? left_down[0] : right_down[0];
	int cm = left_up[1] > left_down[1] ? left_up[1] : left_down[1];
	int cM = right_up[1] < right_down[1] ? right_up[1] : right_down[1];

	left_up[0] = rm, left_up[1] = cm;
	right_up[0] = rm, right_up[1] = cM;
	left_down[0] = rM, left_down[1] = cm;
	right_down[0] = rM, right_down[1] = cM;

	if (left_up[0] > left_down[0]) {
		t = left_up[0];
		left_up[0] = left_down[0];
		left_down[0] = t;
	}
}

int cal_size(Mat overlap, int x, int y, int* cor) {
	int ax,ay,bx,by;
	// find the cor of row major rectangle's diagonal point
	for (int i = x; i < overlap.rows; i++) {
		if (overlap.at<uchar>(i, y) == 0) { 
			ax = i - 1; 
			i = overlap.rows;
			for (int j = y; j < overlap.cols; j++) {
				if (overlap.at<uchar>(ax, j) == 0) {
					ay = j - 1;
					j = overlap.cols;
				}
			}
		}		
	}

	// find the cor of col major rectangle's diagonal point
	for (int j = y; j < overlap.cols; j++) {
		if (overlap.at<uchar>(x, j) == 0) { 
			by = j - 1;
			j = overlap.cols;
			for (int i = x; i < overlap.rows; i++) {
				if (overlap.at<uchar>(i, by) == 0) {
					bx = i - 1;
					i = overlap.rows;
				}
			}
		}		
	}

	// comapre the size of square 
	if (((ax - x) * (ay - y)) > ((bx - x) * (by - y))) {
		cor[0] = ax;
		cor[1] = ay;
	}
	else {
		cor[0] = bx;
		cor[1] = by;
	}	
	return (cor[0] - x) * (cor [1] - y);
}

void in_square(Mat overlap, int* one, int* four) {
	int score;
	int max = 0;
	int lu[2] = { 0, 0 };
	int rd[2] = { 0, 0 };
	int temp[2] = { 0, 0 };
	for (int i = 0; i < overlap.rows; i++) {
		for (int j = 0; j < overlap.cols; j++) {
			if (overlap.at<uchar>(i, j) != 0) {
				score = cal_size(overlap, i, j, temp);
				if (score > max) {
					max = score;
					lu[0] = i, lu[1] = j;
					rd[0] = temp[0], rd[1] = temp[1];
					j = overlap.cols;
				}
			}
		}
	}
	one[0] = lu[0], one[1] = lu[1];
	four[0] = rd[0], four[1] = rd[1];
}

int main1() {
	Mat overlap = imread("overlap/4__5__overlap.png", 0);
	Mat result;
	//cvtColor(overlap, result, COLOR_GRAY2BGR);
	
	int row = overlap.rows;
	int col = overlap.cols;
	int score;
	int max = 0;
	int one[2] = { 0, 0 };
	int four[2] = { 0, 0 };
	int temp[2] = { 0, 0 };
	int adw_len = 11;

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {			
			if (overlap.at<uchar>(i, j) != 0) {
				score = cal_size(overlap,i,j,temp);
				if (max < score) {					
					max = score;
					cout << "\t[win] \t(" << i << " , " << j << ") = " << score << endl;	
					one[0] = i, one[1] = j;
					four[0] = temp[0], four[1] = temp[1];
					//j = col;
				}
				else
					cout << "[lose] \t(" << i << " , " << j << ") = " << score << endl;
				/*one[0] = i, one[1] = j;
				four[0] = temp[0], four[1] = temp[1];
				cvtColor(overlap, result, COLOR_GRAY2BGR);
				for (int ii = one[0]; ii < four[0]; ii++) {
					for (int jj = one[1]; jj < four[1]; jj++) {
						result.at<Vec3b>(ii, jj)[0] = 0;
						result.at<Vec3b>(ii, jj)[1] = 0;
						result.at<Vec3b>(ii, jj)[2] = 255;
					}
				}
				std::string str1 = std::to_string(i);
				std::string str2 = std::to_string(j);

				imwrite("test/" + str1 + "_" + str2 + "_result.png", result);*/
				j = col;
			}
		}
	}

	cvtColor(overlap, result, COLOR_GRAY2BGR);
	for (int i = one[0]; i < four[0]; i++) {
		for (int j = one[1]; j < four[1]; j++) {
			if (overlap.at<uchar>(i, j) != 0) {
				result.at<Vec3b>(i, j)[0] = 0;
				result.at<Vec3b>(i, j)[1] = 0;
				result.at<Vec3b>(i, j)[2] = 255;
			}
		}
	}
	
	imwrite("result.png", result);
}

int main() {
	Mat warp_0 = imread("warp/0__warped_img.png", 1);
	Mat warp_1 = imread("warp/1__warped_img.png", 1);
	Mat warp_2 = imread("warp/2__warped_img.png", 1);
	Mat warp_3 = imread("warp/3__warped_img.png", 1);
	Mat warp_4 = imread("warp/4__warped_img.png", 1);
	//Mat warp_5 = imread("warp/5__warped_img.png", 1);
	//Mat warp_6 = imread("warp/6_warped_img.png", 1);
	//Mat result = imread("ori_result_test.png", 1);
	
	Mat mask_0_1 = imread("mask/0__1__mask.png", 0);
	//Mat mask_0_2 = imread("mask/0__2__mask.png", 0);
	//Mat mask_0_3 = imread("mask/0__3__mask.png", 0);

	Mat mask_1_0 = imread("mask/1__0__mask.png", 0);
	Mat mask_1_2 = imread("mask/1__2__mask.png", 0);
	//Mat mask_1_3 = imread("mask/1__3__mask.png", 0);
	/*Mat mask_1_4 = imread("mask/1__4__mask.png", 0);
	Mat mask_1_5 = imread("mask/1__5__mask.png", 0);*/

	//Mat mask_2_0 = imread("mask/2__0__mask.png", 0);
	Mat mask_2_1 = imread("mask/2__1__mask.png", 0);
	Mat mask_2_3 = imread("mask/2__3__mask.png", 0);
	//Mat mask_2_4 = imread("mask/2__4__mask.png", 0);
	/*Mat mask_2_5 = imread("mask/2__5__mask.png", 0);*/

	//Mat mask_3_0 = imread("mask/3__0__mask.png", 0);
	//Mat mask_3_1 = imread("mask/3__1__mask.png", 0);
	Mat mask_3_2 = imread("mask/3__2__mask.png", 0);
	Mat mask_3_4 = imread("mask/3__4__mask.png", 0);
	//Mat mask_3_5 = imread("mask/3__5__mask.png", 0);
	/*Mat mask_3_6 = imread("mask/3__6__mask.png", 0);*/

	//Mat mask_4_1 = imread("mask/4__1__mask.png", 0);
	//Mat mask_4_2 = imread("mask/4__2__mask.png", 0);
	Mat mask_4_3 = imread("mask/4__3__mask.png", 0);
	//Mat mask_4_5 = imread("mask/4__5__mask.png", 0);
	//Mat mask_4_6 = imread("mask/4__6__mask.png", 0);

	//Mat mask_5_1 = imread("mask/5__1__mask.png", 0);
	//Mat mask_5_2 = imread("mask/5__2__mask.png", 0);
	//Mat mask_5_3 = imread("mask/5__3__mask.png", 0);
	//Mat mask_5_4 = imread("mask/5__4__mask.png", 0);
	/*Mat mask_5_6 = imread("mask/5__6__mask.png", 0);

	Mat mask_6_3 = imread("mask/6__3__mask.png", 0);
	Mat mask_6_4 = imread("mask/6__4__mask.png", 0);
	Mat mask_6_5 = imread("mask/6__5__mask.png", 0);*/

	Mat overlap_0_1 = imread("overlap/0__1__overlap.png", 0);
	//Mat overlap_0_2 = imread("overlap/0__2__overlap.png", 0);
	//Mat overlap_0_3 = imread("overlap/0__3__overlap.png", 0);

	Mat overlap_1_0 = imread("overlap/1__0__overlap.png", 0);
	Mat overlap_1_2 = imread("overlap/1__2__overlap.png", 0);
	//Mat overlap_1_3 = imread("overlap/1__3__overlap.png", 0);
	/*Mat overlap_1_4 = imread("overlap/1__4__overlap.png", 0);
	Mat overlap_1_5 = imread("overlap/1__5__overlap.png", 0);*/

	//Mat overlap_2_0 = imread("overlap/2__0__overlap.png", 0);
	Mat overlap_2_1 = imread("overlap/2__1__overlap.png", 0);
	Mat overlap_2_3 = imread("overlap/2__3__overlap.png", 0);
	//Mat overlap_2_4 = imread("overlap/2__4__overlap.png", 0);
	/*Mat overlap_2_5 = imread("overlap/2__5__overlap.png", 0);

	Mat overlap_3_0 = imread("overlap/3__0__overlap.png", 0);*/
	//Mat overlap_3_1 = imread("overlap/3__1__overlap.png", 0);
	Mat overlap_3_2 = imread("overlap/3__2__overlap.png", 0);
	Mat overlap_3_4 = imread("overlap/3__4__overlap.png", 0);
	//Mat overlap_3_5 = imread("overlap/3__5__overlap.png", 0);
	/*Mat overlap_3_6 = imread("overlap/3__6__overlap.png", 0);*/

	//Mat overlap_4_1 = imread("overlap/4__1__overlap.png", 0);
	//Mat overlap_4_2 = imread("overlap/4__2__overlap.png", 0);
	Mat overlap_4_3 = imread("overlap/4__3__overlap.png", 0);
	//Mat overlap_4_5 = imread("overlap/4__5__overlap.png", 0);
	/*Mat overlap_4_6 = imread("overlap/4__6__overlap.png", 0);

	Mat overlap_5_1 = imread("overlap/5__1__overlap.png", 0);
	Mat overlap_5_2 = imread("overlap/5__2__overlap.png", 0);*/
	//Mat overlap_5_3 = imread("overlap/5__3__overlap.png", 0);
	//Mat overlap_5_4 = imread("overlap/5__4__overlap.png", 0);
	/*Mat overlap_5_6 = imread("overlap/5__6__overlap.png", 0);

	Mat overlap_6_3 = imread("overlap/6__3__overlap.png", 0);
	Mat overlap_6_4 = imread("overlap/6__4__overlap.png", 0);
	Mat overlap_6_5 = imread("overlap/6__5__overlap.png", 0);*/

	/*Mat seam_0_1 = imread("seam/0__1__seam.png", 0);
	Mat seam_1_0 = imread("seam/1__0__seam.png", 0);
	Mat seam_1_2 = imread("seam/1__2__seam.png", 0);
	Mat seam_2_1 = imread("seam/2__1__seam.png", 0);
	Mat seam_2_3 = imread("seam/2__3__seam.png", 0);
	Mat seam_3_2 = imread("seam/3__2__seam.png", 0);
	Mat seam_3_4 = imread("seam/3__4__seam.png", 0);
	Mat seam_4_3 = imread("seam/4__3__seam.png", 0);
	Mat seam_4_5 = imread("seam/4__5__seam.png", 0);
	Mat seam_5_4 = imread("seam/5__4__seam.png", 0);*/

	int row = warp_0.rows;
	int col = warp_0.cols;
	Mat temp;

	/*
	result.copyTo(temp);	

	// draw stitch line
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			if (seam_0_1.at<uchar>(i, j) != 0) {
				temp.at<Vec3b>(i, j)[0] = 255;
				temp.at<Vec3b>(i, j)[1] = 0;
				temp.at<Vec3b>(i, j)[2] = 0;
			}
			if (seam_0_2.at<uchar>(i, j) != 0) {
				temp.at<Vec3b>(i, j)[0] = 0;
				temp.at<Vec3b>(i, j)[1] = 255;
				temp.at<Vec3b>(i, j)[2] = 0;
			}
			if (seam_1_2.at<uchar>(i, j) != 0) {
				temp.at<Vec3b>(i, j)[0] = 0;
				temp.at<Vec3b>(i, j)[1] = 0;
				temp.at<Vec3b>(i, j)[2] = 255;
			}
		}
	}
	imwrite("stitch.png",temp);

	*/

	/********************************* global *****************************************/
	// produce impact region mask for each img (oerlap two mask of each img)
	Mat region_0 = Mat::zeros(row, col, CV_8U);
	Mat region_1 = Mat::zeros(row, col, CV_8U);
	Mat region_2 = Mat::zeros(row, col, CV_8U);
	Mat region_3 = Mat::zeros(row, col, CV_8U);
	Mat region_4 = Mat::zeros(row, col, CV_8U);
	//Mat region_5 = Mat::zeros(row, col, CV_8U);
	/*Mat region_6 = Mat::zeros(row, col, CV_8U); */
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			if ((mask_0_1.at<uchar>(i, j) != 0) /*&& (mask_0_2.at<uchar>(i, j) != 0) && (mask_0_3.at<uchar>(i, j) != 0)*/)
				region_0.at<uchar>(i, j) = 255;
			if ((mask_1_0.at<uchar>(i, j) != 0) && (mask_1_2.at<uchar>(i, j) != 0) /*&& (mask_1_3.at<uchar>(i, j) != 0) && (mask_1_4.at<uchar>(i, j) != 0) && (mask_1_5.at<uchar>(i, j) != 0)*/)
				region_1.at<uchar>(i, j) = 255;
			if (/*(mask_2_0.at<uchar>(i, j) != 0) &&*/  (mask_2_1.at<uchar>(i, j) != 0) && (mask_2_3.at<uchar>(i, j) != 0) /* && (mask_2_4.at<uchar>(i, j) != 0) && (mask_2_5.at<uchar>(i, j) != 0)*/)
				region_2.at<uchar>(i, j) = 255;
			if (/*(mask_3_0.at<uchar>(i, j) != 0) && (mask_3_1.at<uchar>(i, j) != 0) &&*/ (mask_3_2.at<uchar>(i, j) != 0)  && (mask_3_4.at<uchar>(i, j) != 0) /*&& (mask_3_5.at<uchar>(i, j) != 0) && (mask_3_6.at<uchar>(i, j) != 0)*/)
				region_3.at<uchar>(i, j) = 255;
			if (/*(mask_4_1.at<uchar>(i, j) != 0) && (mask_4_2.at<uchar>(i, j) != 0) &&*/ (mask_4_3.at<uchar>(i, j) != 0)  /*&& (mask_4_5.at<uchar>(i, j) != 0) && (mask_4_6.at<uchar>(i, j) != 0)*/)
				region_4.at<uchar>(i, j) = 255;
			//if (/*(mask_5_1.at<uchar>(i, j) != 0) && (mask_5_2.at<uchar>(i, j) != 0) &&*/ (mask_5_3.at<uchar>(i, j) != 0) && (mask_5_4.at<uchar>(i, j) != 0) /*&& (mask_5_6.at<uchar>(i, j) != 0)*/)
				//region_5.at<uchar>(i, j) = 255;
			/*if ((mask_6_3.at<uchar>(i, j) != 0) && (mask_6_4.at<uchar>(i, j) != 0) && (mask_6_5.at<uchar>(i, j) != 0))
				region_6.at<uchar>(i, j) = 255;*/
		}
	}
	imwrite("region_0.png", region_0);
	imwrite("region_1.png", region_1);
	imwrite("region_2.png", region_2);
	imwrite("region_3.png", region_3);
	imwrite("region_4.png", region_4);
	//imwrite("region_5.png", region_5);
	//imwrite("region_6.png", region_6);
	

	
	// cal img's overlap region Mean and Standard deviation
	float	mean_0_1[3], //mean_0_2[3], //mean_0_3[3],
			mean_1_0[3], mean_1_2[3],/*mean_1_3[3], mean_1_4[3], mean_1_5[3],
			mean_2_0[3],*/ mean_2_1[3], mean_2_3[3],/* mean_2_4[3], mean_2_5[3],*/
			/*mean_3_0[3], mean_3_1[3],*/ mean_3_2[3], mean_3_4[3],/* mean_3_5[3], mean_3_6[3],
			mean_4_1[3], mean_4_2[3],*/ mean_4_3[3]/*, mean_4_5[3], mean_4_6[3],
			mean_5_1[3], mean_5_2[3], mean_5_3[3], mean_5_4[3], mean_5_6[3], 
			mean_6_3[3], mean_6_4[3], mean_6_5[3]*/;
	float	std_0_1[3], //std_0_2[3],/*std_0_3[3],*/
			std_1_0[3], std_1_2[3], /*std_1_3[3], std_1_4[3], std_1_5[3],
			std_2_0[3],*/ std_2_1[3], std_2_3[3],/* std_2_4[3], std_2_5[3],
			std_3_0[3], std_3_1[3],*/ std_3_2[3], std_3_4[3],/* std_3_5[3], std_3_6[3],
			std_4_1[3], std_4_2[3],*/ std_4_3[3]/*, std_4_5[3], std_4_6[3],
			std_5_1[3], std_5_2[3], std_5_3[3], std_5_4[3], std_5_6[3],
			std_6_3[3], std_6_4[3], std_6_5[3]*/;

	mean_and_std(warp_0, overlap_0_1, mean_0_1, std_0_1);
	//mean_and_std(warp_0, overlap_0_2, mean_0_2, std_0_2);
	//mean_and_std(warp_0, overlap_0_3, mean_0_3, std_0_3);

	mean_and_std(warp_1, overlap_1_0, mean_1_0, std_1_0);
	mean_and_std(warp_1, overlap_1_2, mean_1_2, std_1_2);
	//mean_and_std(warp_1, overlap_1_3, mean_1_3, std_1_3);
	/*mean_and_std(warp_1, overlap_1_4, mean_1_4, std_1_4);
	mean_and_std(warp_1, overlap_1_5, mean_1_5, std_1_5);*/

	//mean_and_std(warp_2, overlap_2_0, mean_2_0, std_2_0);
	mean_and_std(warp_2, overlap_2_1, mean_2_1, std_2_1);
	mean_and_std(warp_2, overlap_2_3, mean_2_3, std_2_3);
	//mean_and_std(warp_2, overlap_2_4, mean_2_4, std_2_4);
	//mean_and_std(warp_2, overlap_2_5, mean_2_5, std_2_5);
	 
	//mean_and_std(warp_3, overlap_3_0, mean_3_0, std_3_0);
	//mean_and_std(warp_3, overlap_3_1, mean_3_1, std_3_1);
	mean_and_std(warp_3, overlap_3_2, mean_3_2, std_3_2);
	mean_and_std(warp_3, overlap_3_4, mean_3_4, std_3_4);
	//mean_and_std(warp_3, overlap_3_5, mean_3_5, std_3_5);
	/*mean_and_std(warp_3, overlap_3_6, mean_3_6, std_3_6);
	 
	mean_and_std(warp_4, overlap_4_1, mean_4_1, std_4_1);*/
	//mean_and_std(warp_4, overlap_4_2, mean_4_2, std_4_2);
	mean_and_std(warp_4, overlap_4_3, mean_4_3, std_4_3);
	//mean_and_std(warp_4, overlap_4_5, mean_4_5, std_4_5);
	/*mean_and_std(warp_4, overlap_4_6, mean_4_6, std_4_6);

	mean_and_std(warp_5, overlap_5_1, mean_5_1, std_5_1);
	mean_and_std(warp_5, overlap_5_2, mean_5_2, std_5_2);*/
	//mean_and_std(warp_5, overlap_5_3, mean_5_3, std_5_3);
	//mean_and_std(warp_5, overlap_5_4, mean_5_4, std_5_4);
	/*mean_and_std(warp_5, overlap_5_6, mean_5_6, std_5_6);

	mean_and_std(warp_6, overlap_6_3, mean_6_3, std_6_3);
	mean_and_std(warp_6, overlap_6_4, mean_6_4, std_6_4);
	mean_and_std(warp_6, overlap_6_5, mean_6_5, std_6_5);*/

	/*
	cout << endl;
	cout << "0_1's Mean :" << "\n[" << mean_0_1[0] << "]\t[" << mean_0_1[1] << "]\t[" << mean_0_1[2] << "]\n";
	cout << "0_1's STD :" << "\n[" << std_0_1[0] << "]\t[" << std_0_1[1] << "]\t[" << std_0_1[2] << "]\n";
	cout << endl;
	cout << "0_2's Mean :" << "\n[" << mean_0_2[0] << "]\t[" << mean_0_2[1] << "]\t[" << mean_0_2[2] << "]\n";
	cout << "0_2's STD :" << "\n[" << std_0_2[0] << "]\t[" << std_0_2[1] << "]\t[" << std_0_2[2] << "]\n";
	cout << endl;
	cout << "1_0's Mean :" << "\n[" << mean_1_0[0] << "]\t[" << mean_1_0[1] << "]\t[" << mean_1_0[2] << "]\n";
	cout << "1_0's STD :" << "\n[" << std_1_0[0] << "]\t[" << std_1_0[1] << "]\t[" << std_1_0[2] << "]\n";
	cout << endl;
	cout << "1_2's Mean :" << "\n[" << mean_1_2[0] << "]\t[" << mean_1_2[1] << "]\t[" << mean_1_2[2] << "]\n";
	cout << "1_2's STD :" << "\n[" << std_1_2[0] << "]\t[" << std_1_2[1] << "]\t[" << std_1_2[2] << "]\n";
	cout << endl;
	cout << "2_0's Mean :" << "\n[" << mean_2_0[0] << "]\t[" << mean_2_0[1] << "]\t[" << mean_2_0[2] << "]\n";
	cout << "2_0's STD :" << "\n[" << std_2_0[0] << "]\t[" << std_2_0[1] << "]\t[" << std_2_0[2] << "]\n";
	cout << endl;
	cout << "2_1's Mean :" << "\n[" << mean_2_1[0] << "]\t[" << mean_2_1[1] << "]\t[" << mean_2_1[2] << "]\n";
	cout << "2_1's STD :" << "\n[" << std_2_1[0] << "]\t[" << std_2_1[1] << "]\t[" << std_2_1[2] << "]\n";
	*/

	// cal global coef
	float	coef_0_1[6], //coef_0_2[6], //coef_0_3[6],
			coef_1_0[6], coef_1_2[6],/* coef_1_3[6], coef_1_4[6], coef_1_5[6],
			coef_2_0[6],*/ coef_2_1[6], coef_2_3[6],/* coef_2_4[6], coef_2_5[6],
			coef_3_0[6], coef_3_1[6],*/ coef_3_2[6], coef_3_4[6],/* coef_3_5[6], coef_3_6[6],
			coef_4_1[6], coef_4_2[6],*/ coef_4_3[6]/*, coef_4_5[6], coef_4_6[6],
			coef_5_1[6], coef_5_2[6], coef_5_3[6], coef_5_4[6], coef_5_6[6],
			coef_6_3[6], coef_6_4[6], coef_6_5[6]*/;


	cal_glob_coef(mean_0_1, std_0_1, mean_1_0, std_1_0, coef_0_1, coef_1_0);
	//cal_glob_coef(mean_0_2, std_0_2, mean_2_0, std_2_0, coef_0_2, coef_2_0);
	//cal_glob_coef(mean_0_3, std_0_3, mean_3_0, std_3_0, coef_0_3, coef_3_0);

	cal_glob_coef(mean_1_2, std_1_2, mean_2_1, std_2_1, coef_1_2, coef_2_1);
	//cal_glob_coef(mean_1_3, std_1_3, mean_3_1, std_3_1, coef_1_3, coef_3_1);
	/*cal_glob_coef(mean_1_4, std_1_4, mean_4_1, std_4_1, coef_1_4, coef_4_1);
	cal_glob_coef(mean_1_5, std_1_5, mean_5_1, std_5_1, coef_1_5, coef_5_1);*/
		
	cal_glob_coef(mean_2_3, std_2_3, mean_3_2, std_3_2, coef_2_3, coef_3_2);
	//cal_glob_coef(mean_2_4, std_2_4, mean_4_2, std_4_2, coef_2_4, coef_4_2);
	/*cal_glob_coef(mean_2_5, std_2_5, mean_5_2, std_5_2, coef_2_5, coef_5_2);*/

	cal_glob_coef(mean_3_4, std_3_4, mean_4_3, std_4_3, coef_3_4, coef_4_3);
	//cal_glob_coef(mean_3_5, std_3_5, mean_5_3, std_5_3, coef_3_5, coef_5_3);
	/*cal_glob_coef(mean_3_6, std_3_6, mean_6_3, std_6_3, coef_3_6, coef_6_3);*/

	//cal_glob_coef(mean_4_5, std_4_5, mean_5_4, std_5_4, coef_4_5, coef_5_4);
	/*cal_glob_coef(mean_4_6, std_4_6, mean_6_4, std_6_4, coef_4_6, coef_6_4);

	cal_glob_coef(mean_5_6, std_5_6, mean_6_5, std_6_5, coef_5_6, coef_6_5);*/

	// revise the coef by overlap pixel size difference
	float coef_0[6],coef_1[6],coef_2[6],coef_3[6],coef_4[6],coef_5[6], coef_6[6];
	float sum, a, b, c, d, e, f;

	// img_0
	a = cal_weight(overlap_0_1, overlap_0_1);
	//b = cal_weight(overlap_0_2, overlap_0_2);
	//c = cal_weight(overlap_0_3, overlap_0_3);
	sum = a /*+ b*/;
	for (int i = 0; i < 6; i++) {
		coef_0[i] = (coef_0_1[i] * a / sum) /*+ (coef_0_2[i] * b / sum) + (coef_0_3[i] * c / sum)*/;
		cout << "coef_0[" << i << "] = " << coef_0[i] << "\n";
	}

	// img_1
	a = cal_weight(overlap_1_0, overlap_1_0);
	b = cal_weight(overlap_1_2, overlap_1_2);
	//c = cal_weight(overlap_1_3, overlap_1_3);
	/*d = cal_weight(overlap_1_4, overlap_1_4);
	e = cal_weight(overlap_1_5, overlap_1_5);*/
	sum = a + b /*+ c + d + e */ ;
	for (int i = 0; i < 6; i++) {
		coef_1[i] = (coef_1_0[i] * a / sum) + (coef_1_2[i] * b / sum) /*+ (coef_1_3[i] * c / sum)  + (coef_1_4[i] * d / sum) + (coef_1_5[i] * e / sum) */ ;
		cout << "coef_1[" << i << "] = " << coef_1[i] << "\n";
	}

	//img_2
	//a = cal_weight(overlap_2_0, overlap_2_0);
	b = cal_weight(overlap_2_1, overlap_2_1);
	c = cal_weight(overlap_2_3, overlap_2_3);
	//d = cal_weight(overlap_2_4, overlap_2_4);
	/*e = cal_weight(overlap_2_5, overlap_2_5);*/
	sum = a + b /*+ c + d + e */;
	for (int i = 0; i < 6; i++) {
		coef_2[i] = /*(coef_2_0[i] * a / sum) +*/ (coef_2_1[i] * b / sum) + (coef_2_3[i] * c / sum) /*+ (coef_2_4[i] * d / sum) + (coef_2_5[i] * e / sum)*/;
		cout << "coef_2[" << i << "] = " << coef_2[i] << "\n";
	}

	//img_3
	/*a = cal_weight(overlap_3_0, overlap_3_0);*/
	//b = cal_weight(overlap_3_1, overlap_3_1);
	c = cal_weight(overlap_3_2, overlap_3_2);
	d = cal_weight(overlap_3_4, overlap_3_4);
	//e = cal_weight(overlap_3_5, overlap_3_5);
	/*f = cal_weight(overlap_3_6, overlap_3_6);*/
	sum = /*a + b +*/ c + d /*+ e + f*/;
	for (int i = 0; i < 6; i++) {
		coef_3[i] = /*(coef_3_0[i] * a / sum) + (coef_3_1[i] * b / sum) +*/ (coef_3_2[i] * c / sum) + (coef_3_4[i] * d / sum) /*+ (coef_3_5[i] * e / sum) + (coef_3_6[i] * f / sum)*/;
		cout << "coef_3[" << i << "] = " << coef_3[i] << "\n";
	}

	//img_4
	/*a = cal_weight(overlap_4_1, overlap_4_1);*/
	//b = cal_weight(overlap_4_2, overlap_4_2);
	c = cal_weight(overlap_4_3, overlap_4_3);
	//d = cal_weight(overlap_4_5, overlap_4_5);
	/*e = cal_weight(overlap_4_6, overlap_4_6);*/
	sum = /*b +*/ c /*+ d + e*/;
	for (int i = 0; i < 6; i++) {
		coef_4[i] = /*(coef_4_1[i] * a / sum) + (coef_4_2[i] * b / sum) +*/ (coef_4_3[i] * c / sum) /*+ (coef_4_5[i] * d / sum) + (coef_4_6[i] * e / sum)*/;
		cout << "coef_4[" << i << "] = " << coef_4[i] << "\n";
	}

	//img_5
	/*a = cal_weight(overlap_5_1, overlap_5_1);
	b = cal_weight(overlap_5_2, overlap_5_2);*/
	//c = cal_weight(overlap_5_3, overlap_5_3);
	//d = cal_weight(overlap_5_4, overlap_5_4);
	///*e = cal_weight(overlap_5_6, overlap_5_6);*/
	//sum = /*a + b +*/ c + d /*+ e*/;
	//for (int i = 0; i < 6; i++) {
	//	coef_5[i] = /*(coef_5_1[i] * a / sum) + (coef_5_2[i] * b / sum) +*/ (coef_5_3[i] * c / sum) + (coef_5_4[i] * d / sum) /*+ (coef_5_6[i] * e / sum)*/;
	//	cout << "coef_5[" << i << "] = " << coef_5[i] << "\n";
	//}

	//img_6
	/*a = cal_weight(overlap_6_3, overlap_6_3);
	b = cal_weight(overlap_6_4, overlap_6_4);
	c = cal_weight(overlap_6_5, overlap_6_5);
	sum = a + b + c;
	for (int i = 0; i < 6; i++) {
		coef_5[i] = (coef_6_3[i] * a / sum) + (coef_6_4[i] * b / sum) + (coef_6_5[i] * c / sum);
		cout << "coef_5[" << i << "] = " << coef_5[i] << "\n";
	}*/

	// global color change
	Mat global;
	//result.copyTo(global);
	
	// region_0 & warp_0 
	temp = img_bgr_to_g(warp_0);
	/*color_trans(global, region_0, coef_0[0], coef_0[1], 0);
	color_trans(global, region_0, coef_0[2], coef_0[3], 1);
	color_trans(global, region_0, coef_0[4], coef_0[5], 2);*/
	color_trans(warp_0, temp, coef_0[0], coef_0[1], 0);
	color_trans(warp_0, temp, coef_0[2], coef_0[3], 1);
	color_trans(warp_0, temp, coef_0[4], coef_0[5], 2);

	// region_1 & warp_1
	temp = img_bgr_to_g(warp_1);
	/*color_trans(global, region_1, coef_1[0], coef_1[1], 0);
	color_trans(global, region_1, coef_1[2], coef_1[3], 1);
	color_trans(global, region_1, coef_1[4], coef_1[5], 2);*/
	color_trans(warp_1, temp, coef_1[0], coef_1[1], 0);
	color_trans(warp_1, temp, coef_1[2], coef_1[3], 1);
	color_trans(warp_1, temp, coef_1[4], coef_1[5], 2);

	// region_2 & warp_2
	temp = img_bgr_to_g(warp_2);
	/*color_trans(global, region_2, coef_2[0], coef_2[1], 0);
	color_trans(global, region_2, coef_2[2], coef_2[3], 1);
	color_trans(global, region_2, coef_2[4], coef_2[5], 2);*/
	color_trans(warp_2, temp, coef_2[0], coef_2[1], 0);
	color_trans(warp_2, temp, coef_2[2], coef_2[3], 1);
	color_trans(warp_2, temp, coef_2[4], coef_2[5], 2);

	// region_3 & warp_3
	temp = img_bgr_to_g(warp_3);
	/*color_trans(global, region_3, coef_3[0], coef_3[1], 0);
	color_trans(global, region_3, coef_3[2], coef_3[3], 1);
	color_trans(global, region_3, coef_3[4], coef_3[5], 2);*/
	color_trans(warp_3, temp, coef_3[0], coef_3[1], 0);
	color_trans(warp_3, temp, coef_3[2], coef_3[3], 1);
	color_trans(warp_3, temp, coef_3[4], coef_3[5], 2);

	// region_4 & warp_4
	temp = img_bgr_to_g(warp_4);
	color_trans(warp_4, temp, coef_4[0], coef_4[1], 0);
	color_trans(warp_4, temp, coef_4[2], coef_4[3], 1);
	color_trans(warp_4, temp, coef_4[4], coef_4[5], 2);

	// region_5 & warp_5
	/*temp = img_bgr_to_g(warp_5);
	color_trans(warp_5, temp, coef_5[0], coef_5[1], 0);
	color_trans(warp_5, temp, coef_5[2], coef_5[3], 1);
	color_trans(warp_5, temp, coef_5[4], coef_5[5], 2);*/

	// region_6 & warp_6
	/*temp = img_bgr_to_g(warp_6);
	color_trans(warp_6, temp, coef_6[0], coef_6[1], 0);
	color_trans(warp_6, temp, coef_6[2], coef_6[3], 1);
	color_trans(warp_6, temp, coef_6[4], coef_6[5], 2);*/

	/*
	// fill the hole by equal the value after global
	hole_fill(global, warp_0, warp_1, warp_2);
	imwrite("global.png", global);
	*/

	//imwrite("global.png", global);
	imwrite("global_warp_0.png", warp_0);
	imwrite("global_warp_1.png", warp_1);
	imwrite("global_warp_2.png", warp_2);
	imwrite("global_warp_3.png", warp_3);
	imwrite("global_warp_4.png", warp_4);
	//imwrite("global_warp_5.png", warp_5);
	//imwrite("global_warp_6.png", warp_6);

	/********************************** local ***************************************/
	// cal ADW size
	int adw_len = 91;
	// cal inner square(local working region) in overlap region
	int left_up[2], right_up[2], left_down[2], right_down[2];
	Mat local;
	//global.copyTo(local);	
	// do overlap_0_1, overlap_0_2, overlap_0_3, overlap_1_2, overlap_1_3, overlap_2_3
	
	// draw to clarify working region
	// 0_1	
	find_in_square(overlap_0_1, left_up, right_up, left_down, right_down);	
	//in_square(overlap_0_1, left_up, right_down);
	right_up[0] = left_up[0], right_up[1] = right_down[1];
	left_down[0] = right_down[0], left_down[1] = left_up[1];
	temp = draw_local(overlap_0_1, left_up[0], left_up[1], left_down[0] - left_up[0], right_up[1] - left_up[1], adw_len);
	imwrite("local_1_work.png",temp);
	
	for (int ch = 0; ch < 3; ch++) {
		cal_grid(local,region_0,warp_0,warp_1,left_up[0],left_up[1],left_down[0]-left_up[0],right_up[1]-left_up[1],adw_len,ch);
	}

	// draw to clarify working region
	// 0_2
	//find_in_square(overlap_0_2, left_up, right_up, left_down, right_down);	
	////in_square(overlap_0_2, left_up, right_down);
	//right_up[0] = left_up[0], right_up[1] = right_down[1];
	//left_down[0] = right_down[0], left_down[1] = left_up[1];
	//temp = draw_local(overlap_0_2, left_up[0], left_up[1], left_down[0] - left_up[0], right_up[1] - left_up[1], adw_len);
	//imwrite("local_2_work.png", temp);
	//
	//for (int ch = 0; ch < 3; ch++) {
	//	cal_grid(local, region_0, warp_0, warp_2, left_up[0], left_up[1], left_down[0] - left_up[0], right_up[1] - left_up[1], adw_len, ch);
	//}

	// draw to clarify working region
	// 0_3
	//find_in_square(overlap_0_3, left_up, right_up, left_down, right_down);
	//temp = draw_local(overlap_0_3, left_up[0], left_up[1], left_down[0] - left_up[0], right_up[1] - left_up[1], adw_len);
	////imwrite("local_3_work.png", temp);

	//for (int ch = 0; ch < 3; ch++) {
	//	cal_grid(local, region_0, warp_0, warp_3, left_up[0], left_up[1], left_down[0] - left_up[0], right_up[1] - left_up[1], adw_len, ch);
	//}
	
	// 1_2
	//find_in_square(overlap_1_2, left_up, right_up, left_down, right_down);	
	in_square(overlap_1_2, left_up, right_down);
	right_up[0] = left_up[0], right_up[1] = right_down[1];
	left_down[0] = right_down[0], left_down[1] = left_up[1];
	temp = draw_local(overlap_1_2, left_up[0], left_up[1], left_down[0] - left_up[0], right_up[1] - left_up[1], adw_len);
	imwrite("local_2_work.png", temp);
		
	for (int ch = 0; ch < 3; ch++) {
		cal_grid(local, region_1, warp_1, warp_2, left_up[0], left_up[1], left_down[0] - left_up[0], right_up[1] - left_up[1], adw_len, ch);
	}

	// draw to clarify working region
	// 1_3
	//find_in_square(overlap_1_3, left_up, right_up, left_down, right_down);
	////in_square(overlap_1_3, left_up, right_down);
	//temp = draw_local(overlap_1_3, left_up[0], left_up[1], left_down[0] - left_up[0], right_up[1] - left_up[1], adw_len);
	//imwrite("local_4_work.png", temp);

	//for (int ch = 0; ch < 3; ch++) {
	//	cal_grid(local, region_1, warp_1, warp_3, left_up[0], left_up[1], left_down[0] - left_up[0], right_up[1] - left_up[1], adw_len, ch);
	//}

	// draw to clarify working region
	// 1_4
	///*find_in_square(overlap_1_4, left_up, right_up, left_down, right_down);
	//temp = draw_local(overlap_1_4, left_up[0], left_up[1], left_down[0] - left_up[0], right_up[1] - left_up[1], adw_len);*/
	//imwrite("local_4_work.png", temp);

	//for (int ch = 0; ch < 3; ch++) {
	//	cal_grid(local, region_1, warp_1, warp_4, left_up[0], left_up[1], left_down[0] - left_up[0], right_up[1] - left_up[1], adw_len, ch);
	//}*/

	// draw to clarify working region
	// 1_5
	/*find_in_square(overlap_1_5, left_up, right_up, left_down, right_down);
	temp = draw_local(overlap_1_5, left_up[0], left_up[1], left_down[0] - left_up[0], right_up[1] - left_up[1], adw_len);*/
	//imwrite("local_5_work.png", temp);

	/*for (int ch = 0; ch < 3; ch++) {
		cal_grid(local, region_1, warp_1, warp_5, left_up[0], left_up[1], left_down[0] - left_up[0], right_up[1] - left_up[1], adw_len, ch);
	}*/

	// draw to clarify working region	
	// 2_3
	//find_in_square(overlap_2_3, left_up, right_up, left_down, right_down);
	in_square(overlap_2_3, left_up, right_down);
	right_up[0] = left_up[0], right_up[1] = right_down[1];
	left_down[0] = right_down[0], left_down[1] = left_up[1];
	temp = draw_local(overlap_2_3, left_up[0], left_up[1], left_down[0] - left_up[0], right_up[1] - left_up[1], adw_len);
	imwrite("local_3_work.png", temp);

	for (int ch = 0; ch < 3; ch++) {
		cal_grid(local, region_2, warp_2, warp_3, left_up[0], left_up[1], left_down[0] - left_up[0], right_up[1] - left_up[1], adw_len, ch);
	}

	// draw to clarify working region
	// 2_4
	//find_in_square(overlap_2_4, left_up, right_up, left_down, right_down);
	////in_square(overlap_2_4, left_up, right_down);
	//temp = draw_local(overlap_2_4, left_up[0], left_up[1], left_down[0] - left_up[0], right_up[1] - left_up[1], adw_len);
	//imwrite("local_6_work.png", temp);

	//for (int ch = 0; ch < 3; ch++) {
	//	cal_grid(local, region_2, warp_2, warp_4, left_up[0], left_up[1], left_down[0] - left_up[0], right_up[1] - left_up[1], adw_len, ch);
	//}

	// draw to clarify working region
	// 2_5
	/*find_in_square(overlap_2_5, left_up, right_up, left_down, right_down);
	temp = draw_local(overlap_2_5, left_up[0], left_up[1], left_down[0] - left_up[0], right_up[1] - left_up[1], adw_len);*/
	//imwrite("local_7_work.png", temp);

	/*for (int ch = 0; ch < 3; ch++) {
		cal_grid(local, region_2, warp_2, warp_5, left_up[0], left_up[1], left_down[0] - left_up[0], right_up[1] - left_up[1], adw_len, ch);
	}*/
	
	// 3_4
	//find_in_square(overlap_3_4, left_up, right_up, left_down, right_down);
	in_square(overlap_3_4, left_up, right_down);
	right_up[0] = left_up[0], right_up[1] = right_down[1];
	left_down[0] = right_down[0], left_down[1] = left_up[1];
	temp = draw_local(overlap_3_4, left_up[0], left_up[1], left_down[0] - left_up[0], right_up[1] - left_up[1], adw_len);
	imwrite("local_4_work.png", temp);

	for (int ch = 0; ch < 3; ch++) {
		cal_grid(local, region_3, warp_3, warp_4, left_up[0], left_up[1], left_down[0] - left_up[0], right_up[1] - left_up[1], adw_len, ch);
	}

	// draw to clarify working region
	// 3_5
	/*find_in_square(overlap_3_5, left_up, right_up, left_down, right_down);
	temp = draw_local(overlap_3_5, left_up[0], left_up[1], left_down[0] - left_up[0], right_up[1] - left_up[1], adw_len);
	imwrite("local_8_work.png", temp);

	for (int ch = 0; ch < 3; ch++) {
		cal_grid(local, region_3, warp_3, warp_5, left_up[0], left_up[1], left_down[0] - left_up[0], right_up[1] - left_up[1], adw_len, ch);
	}*/

	// draw to clarify working region
	// 3_6
	/*find_in_square(overlap_3_5, left_up, right_up, left_down, right_down);
	temp = draw_local(overlap_3_5, left_up[0], left_up[1], left_down[0] - left_up[0], right_up[1] - left_up[1], adw_len);*/
	//imwrite("local_9_work.png", temp);

	/*for (int ch = 0; ch < 3; ch++) {
		cal_grid(local, region_3, warp_3, warp_6, left_up[0], left_up[1], left_down[0] - left_up[0], right_up[1] - left_up[1], adw_len, ch);
	}*/
		
	// 4_5
	//find_in_square(overlap_4_5, left_up, right_up, left_down, right_down);
	////in_square(overlap_4_5, left_up, right_down);
	//right_up[0] = left_up[0], right_up[1] = right_down[1];
	//left_down[0] = right_down[0], left_down[1] = left_up[1];
	//temp = draw_local(overlap_4_5, left_up[0], left_up[1], left_down[0] - left_up[0], right_up[1] - left_up[1], adw_len);
	//imwrite("local_9_work.png", temp);

	//for (int ch = 0; ch < 3; ch++) {
	//	cal_grid(local, region_4, warp_4, warp_5, left_up[0], left_up[1], left_down[0] - left_up[0], right_up[1] - left_up[1], adw_len, ch);
	//}

	// draw to clarify working region
	// 4_6
	/*find_in_square(overlap_4_6, left_up, right_up, left_down, right_down);
	temp = draw_local(overlap_4_6, left_up[0], left_up[1], left_down[0] - left_up[0], right_up[1] - left_up[1], adw_len);*/
	//imwrite("local_10_work.png", temp);

	/*for (int ch = 0; ch < 3; ch++) {
		cal_grid(local, region_4, warp_4, warp_6, left_up[0], left_up[1], left_down[0] - left_up[0], right_up[1] - left_up[1], adw_len, ch);
	}*/

	// draw to clarify working region
	// 5_4
	/*find_in_square(overlap_5_4, left_up, right_up, left_down, right_down);
	temp = draw_local(overlap_5_4, left_up[0], left_up[1], left_down[0] - left_up[0], right_up[1] - left_up[1], adw_len);
	imwrite("local_12_work.png", temp);

	for (int ch = 0; ch < 3; ch++) {
		cal_grid(local, region_5, warp_5, warp_4, left_up[0], left_up[1], left_down[0] - left_up[0], right_up[1] - left_up[1], adw_len, ch);
	}*/
	// 5_6
	/*find_in_square(overlap_5_6, left_up, right_up, left_down, right_down);
	temp = draw_local(overlap_5_6, left_up[0], left_up[1], left_down[0] - left_up[0], right_up[1] - left_up[1], adw_len);*/
	//imwrite("local_10_work.png", temp);

	/*for (int ch = 0; ch < 3; ch++) {
		cal_grid(local, region_5, warp_5, warp_6, left_up[0], left_up[1], left_down[0] - left_up[0], right_up[1] - left_up[1], adw_len, ch);
	}*/


	//imwrite("local.png", local);
	
	/*
	temp = img_bgr_to_g(warp_0);
	local.copyTo(warp_0, temp);
	
	temp = img_bgr_to_g(warp_1);
	local.copyTo(warp_1, temp);
	
	temp = img_bgr_to_g(warp_2);
	local.copyTo(warp_2, temp);
	*/

	imwrite("0_warp.png", warp_0);
	imwrite("1_warp.png", warp_1);
	imwrite("2_warp.png", warp_2);
	imwrite("3_warp.png", warp_3);
	imwrite("4_warp.png", warp_4);
	//imwrite("5_warp.png", warp_5);
	//imwrite("6_warp.png", warp_6);

	int k = waitKey(0);
	return 0;
}