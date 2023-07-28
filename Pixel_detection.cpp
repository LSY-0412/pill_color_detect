#include <opencv2/opencv.hpp>
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace cv;
using namespace std;

// dilate�� edge�� ���Ͽ� �� ��翡 �����(�˾�) edge�鸸 �������� �׸��� �� ���� ������ ���� bgrValueList�� ��ȯ�ϴ� �Լ�.
void getContours(Mat imgDil, Mat img, vector<vector<Scalar>>& bgrValuesList);

int main() {
	//�̹��� �ҷ�����
	String imgpath = "test_3_resize.bmp";
	Mat image = imread(imgpath, IMREAD_COLOR);

	if (image.empty()) {
		cout << "Could not open or find the image." << endl;
		return -1;
	}

	// ���� ������ �ϱ����� �̹��� ��� ��ȯ
	Mat grayImage;
	cvtColor(image, grayImage, COLOR_BGR2GRAY);

	// ����þ� ���͸� ����Ͽ� �̹��� ������ ����
	GaussianBlur(grayImage, grayImage, Size(3, 3), 0, 0);

	// ĳ�� ���� ������ ����Ͽ� ���� ������ 50~80�� ������ ����
	Mat edges;
	double threshold1 = 50;
	double threshold2 = 80;

	Canny(grayImage, edges, threshold1, threshold2);

	// ������ ��Ȯ�ϰ� �ϱ����Ͽ� dilate���� ����
	Mat dilatedEdges, erodeEdges;
	dilate(edges, dilatedEdges, Mat(), Point(-1, -1));

	// BGR���� �����ϱ� ���� bgrValueList vector ����
	vector<vector<Scalar>> bgrValuesList;

	// dilate�� edge�� ���Ͽ� �� ��翡 �����(�˾�) edge�鸸 contour�� �׸��� �� ���� ������ ���� bgrValueList�� ��ȯ.
	getContours(dilatedEdges, image, bgrValuesList);

	// �� ������ ���ο� �ִ� BGR���� combinedBgrValues ���� ���� �߰�. (���� �ٸ� �������� BGR���� ���� ���ͷ� ����)
	vector<Scalar> combinedBgrValues;
	for (const auto& bgrValues : bgrValuesList) {
		combinedBgrValues.insert(combinedBgrValues.end(), bgrValues.begin(), bgrValues.end());
	}

	// ���յ� BGR ���� k-means�� ������ �������� ��ȯ.
	Mat bgrData(combinedBgrValues.size(), 3, CV_32F);
	for (size_t i = 0; i < combinedBgrValues.size(); i++) {
		Scalar bgr = combinedBgrValues[i];
		bgrData.at<float>(i, 0) = bgr[0]; // B
		bgrData.at<float>(i, 1) = bgr[1]; // G
		bgrData.at<float>(i, 2) = bgr[2]; // R
	}

	// k-means�� cluster�� ����.
	int numClusters = 5;

	// bgrData�� kmeans�� ����. labels�� �����Ͱ� ��� Ŭ�����Ϳ� �Ҵ��ϴ���. centers�� �� Ŭ�������� ��ǥ BGR���� ����.
	Mat labels, centers;
	kmeans(bgrData, numClusters, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0), 3, KMEANS_RANDOM_CENTERS, centers);

	// Mat�������� �Ǿ��ִ� centers�� BGR���� Scalar�� ��ȯ �� bgrValuesReduced�� ����.
	vector<Scalar> bgrValuesReduced;
	for (int i = 0; i < numClusters; i++) {
		Scalar bgrCenter(centers.at<float>(i, 0), centers.at<float>(i, 1), centers.at<float>(i, 2));
		bgrValuesReduced.push_back(bgrCenter);
	}

	// bgrValuesReduced�� �ִ� BGR���� reducedBGRValues�� Vec3b ���·� ����.
	Mat3b reducedBGRValues(bgrValuesReduced.size(), 1);
	for (size_t i = 0; i < bgrValuesReduced.size(); i++) {
		Scalar bgr = bgrValuesReduced[i];
		reducedBGRValues(i, 0) = Vec3b(bgr[0], bgr[1], bgr[2]);
	}

	// kmeans�� Ŭ�����͸��� 5���� ��ǥ BGR���� ���.
	cout << "Reduced BGR Values: " << endl;
	for (size_t i = 0; i < reducedBGRValues.rows; i++) {
		Vec3b bgr = reducedBGRValues(i, 0);
		cout << "(B=" << static_cast<int>(bgr[0]) << ", G=" << static_cast<int>(bgr[1]) << ", R=" << static_cast<int>(bgr[2]) << ")" << endl;
	}

	// ���� �񱳸� ���Ͽ� reducedBGRValues�� BGR���� HSV�� ��ȯ.
	Mat3b reducedValuesHSV;
	cvtColor(reducedBGRValues, reducedValuesHSV, COLOR_BGR2HSV);


	for (const Scalar& bgr : bgrValuesReduced) {
		Mat3b bgrPixel(1, 1, Vec3b(bgr[0], bgr[1], bgr[2]));
		Mat3b hsvPixel;
		cvtColor(bgrPixel, hsvPixel, COLOR_BGR2HSV);


		Vec3b hsv = hsvPixel(0, 0);
		int hue = hsv[0];  
		float saturation = hsv[1]; 
		float value = hsv[2]; 

		cout << "H=" << hue << ", S=" << saturation << ", V=" << value << endl;

		Mat img = imread(imgpath, IMREAD_COLOR);
		Mat img_hsv;
		
		cvtColor(img, img_hsv, COLOR_BGR2HSV);

		Mat white_mask, white_image;
		int hue_tolerance = 10;  // Hue ������ ���� ��� ������ ����.
		int saturation_tolerance = 100; // saturation ������ ���� ��� ������ ����.
		int value_tolerance = 50; // value ������ ���� ��� ������ ����.

		Scalar lower_white = Scalar(hue - hue_tolerance, saturation - saturation_tolerance, value - value_tolerance);
		Scalar upper_white = Scalar(hue + hue_tolerance, saturation + saturation_tolerance, value + value_tolerance);

		// HSV ���� �Ʒ� ������ �ʰ��ϴ� ��� ó��
		lower_white[0] = MAX(lower_white[0], 0);
		upper_white[0] = MIN(upper_white[0], 360);
		lower_white[1] = MAX(lower_white[1], 0);
		upper_white[1] = MIN(upper_white[1], 255);
		lower_white[2] = MAX(lower_white[2], 0);
		upper_white[2] = MIN(upper_white[2], 255);

		inRange(img_hsv, lower_white, upper_white, white_mask);

		//// dilation ����
		erode(white_mask, white_mask, Mat::ones(Size(5, 5), CV_8UC1), Point(-1, -1),2);
		dilate(white_mask, white_mask, Mat::ones(Size(5, 5), CV_8UC1), Point(-1, -1),2);
		bitwise_and(img, img, white_image, white_mask);

		imshow("white_image", white_image);
		waitKey(0);


	}

	// kmeans�Ϸ�� ������ ���.
	Mat displayReducedBGRValues(100, bgrValuesReduced.size() * 100, CV_8UC3, Scalar(255, 255, 255));

	// ���ȭ�� BGR ���� �ݺ��Ͽ� �̹����� ǥ��
	for (size_t i = 0; i < bgrValuesReduced.size(); i++) {
		Scalar bgr = bgrValuesReduced[i];
		Mat colorPatch(displayReducedBGRValues, Rect(i * 100, 0, 100, 100));
		colorPatch.setTo(Vec3b(bgr[0], bgr[1], bgr[2]));
	}

	// �̹��� ũ�Ժ���
	resize(displayReducedBGRValues, displayReducedBGRValues, Size(displayReducedBGRValues.cols * 2, displayReducedBGRValues.rows * 2));

	imshow("Reduced BGR Values", displayReducedBGRValues);
	//imwrite("Reduced BGR Values_1.bmp", displayReducedBGRValues);
	waitKey(0);
	return 0;
}

void getContours(Mat imgDil, Mat img, vector<vector<Scalar>>& bgrValuesList) {
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(imgDil, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	for (size_t i = 0; i < contours.size(); i++) {
		int area = contourArea(contours[i]);

		if (area > 1000) {
			vector<vector<Point>> conPoly(contours.size());
			float peri = arcLength(contours[i], true);
			approxPolyDP(contours[i], conPoly[i], 0.01 * peri, true);

			int objCor = static_cast<int>(conPoly[i].size());

			// �ѷ����� �������� �������� 4�� �̻��϶��� �Ǵ�.
			if (objCor > 4) {
				// �������� �󸶳� ���� ������� �Ǵ��ϴ� circuaity.
				double circularity = (4 * M_PI * area) / (peri * peri);

				// circuaity�� threshold�� ����.
				double circularityThreshold = 0.075; // Adjust this value as needed

				if (circularity > circularityThreshold) {
					// ��� �������� �׸���.
					drawContours(img, conPoly, i, Scalar(0, 255, 0), 2);

					// ������ ���� ������ ���Ͽ� mask�� ����. ������ ���θ� ������� ĥ��.
					Mat maskInternal = Mat::zeros(img.size(), CV_8UC1);
					drawContours(maskInternal, conPoly, i, Scalar(255), FILLED);

					// ������ �ܺ� ������ ���Ͽ� mask�� ����. �������� ��踦 ������� ĥ��.
					Mat maskExternal = Mat::zeros(img.size(), CV_8UC1);
					drawContours(maskExternal, conPoly, i, Scalar(255), 2); // 2�� �β��� ���� �������� �ܺο����� ���Ͽ� �׸���.

					// �� ���� ����ũ�� �����Ͽ� ���� ������ ����ũ�� ����.
					Mat mask = maskInternal - maskExternal;

					// ���� ������ ���� BGR���� ����.
					vector<Scalar> bgrValues;
					for (int y = 0; y < img.rows; y++) {
						for (int x = 0; x < img.cols; x++) {
							if (mask.at<uchar>(y, x) == 255) {
								bgrValues.push_back(img.at<Vec3b>(y, x));
							}
						}
					}

					// bgrValuesList�� BGR������ ����.
					bgrValuesList.push_back(bgrValues);
				}
			}
		}
	}
}
