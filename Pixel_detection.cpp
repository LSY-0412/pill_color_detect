#include <opencv2/opencv.hpp>
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace cv;
using namespace std;

// dilate한 edge에 대하여 원 모양에 비슷한(알약) edge들만 윤곽선을 그리고 그 내부 영역에 대한 bgrValueList를 반환하는 함수.
void getContours(Mat imgDil, Mat img, vector<vector<Scalar>>& bgrValuesList);

int main() {
	//이미지 불러오기
	String imgpath = "test_3_resize.bmp";
	Mat image = imread(imgpath, IMREAD_COLOR);

	if (image.empty()) {
		cout << "Could not open or find the image." << endl;
		return -1;
	}

	// 엣지 검출을 하기위한 이미지 흑백 전환
	Mat grayImage;
	cvtColor(image, grayImage, COLOR_BGR2GRAY);

	// 가우시안 필터를 사용하여 이미지 노이즈 제거
	GaussianBlur(grayImage, grayImage, Size(3, 3), 0, 0);

	// 캐니 엣지 검출을 사용하여 엣지 강도가 50~80인 엣지를 검출
	Mat edges;
	double threshold1 = 50;
	double threshold2 = 80;

	Canny(grayImage, edges, threshold1, threshold2);

	// 엣지를 명확하게 하기위하여 dilate연산 수행
	Mat dilatedEdges, erodeEdges;
	dilate(edges, dilatedEdges, Mat(), Point(-1, -1));

	// BGR값을 저장하기 위한 bgrValueList vector 생성
	vector<vector<Scalar>> bgrValuesList;

	// dilate한 edge에 대하여 원 모양에 비슷한(알약) edge들만 contour를 그리고 그 내부 영역에 대한 bgrValueList를 반환.
	getContours(dilatedEdges, image, bgrValuesList);

	// 각 윤곽선 내부에 있는 BGR값을 combinedBgrValues 벡터 끝에 추가. (서로 다른 윤곽선의 BGR값을 단일 벡터로 결합)
	vector<Scalar> combinedBgrValues;
	for (const auto& bgrValues : bgrValuesList) {
		combinedBgrValues.insert(combinedBgrValues.end(), bgrValues.begin(), bgrValues.end());
	}

	// 결합된 BGR 값을 k-means에 적합한 형식으로 변환.
	Mat bgrData(combinedBgrValues.size(), 3, CV_32F);
	for (size_t i = 0; i < combinedBgrValues.size(); i++) {
		Scalar bgr = combinedBgrValues[i];
		bgrData.at<float>(i, 0) = bgr[0]; // B
		bgrData.at<float>(i, 1) = bgr[1]; // G
		bgrData.at<float>(i, 2) = bgr[2]; // R
	}

	// k-means의 cluster수 지정.
	int numClusters = 5;

	// bgrData에 kmeans를 적용. labels는 데이터가 어느 클러스터에 할당하는지. centers는 각 클러스터의 대표 BGR값을 저장.
	Mat labels, centers;
	kmeans(bgrData, numClusters, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0), 3, KMEANS_RANDOM_CENTERS, centers);

	// Mat형식으로 되어있는 centers의 BGR값을 Scalar로 변환 후 bgrValuesReduced에 저장.
	vector<Scalar> bgrValuesReduced;
	for (int i = 0; i < numClusters; i++) {
		Scalar bgrCenter(centers.at<float>(i, 0), centers.at<float>(i, 1), centers.at<float>(i, 2));
		bgrValuesReduced.push_back(bgrCenter);
	}

	// bgrValuesReduced에 있는 BGR값을 reducedBGRValues에 Vec3b 형태로 저장.
	Mat3b reducedBGRValues(bgrValuesReduced.size(), 1);
	for (size_t i = 0; i < bgrValuesReduced.size(); i++) {
		Scalar bgr = bgrValuesReduced[i];
		reducedBGRValues(i, 0) = Vec3b(bgr[0], bgr[1], bgr[2]);
	}

	// kmeans로 클러스터링된 5가지 대표 BGR값을 출력.
	cout << "Reduced BGR Values: " << endl;
	for (size_t i = 0; i < reducedBGRValues.rows; i++) {
		Vec3b bgr = reducedBGRValues(i, 0);
		cout << "(B=" << static_cast<int>(bgr[0]) << ", G=" << static_cast<int>(bgr[1]) << ", R=" << static_cast<int>(bgr[2]) << ")" << endl;
	}

	// 색상 비교를 위하여 reducedBGRValues의 BGR값을 HSV로 변환.
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
		int hue_tolerance = 10;  // Hue 범위에 대한 허용 오차를 설정.
		int saturation_tolerance = 100; // saturation 범위에 대한 허용 오차를 설정.
		int value_tolerance = 50; // value 범위에 대한 허용 오차를 설정.

		Scalar lower_white = Scalar(hue - hue_tolerance, saturation - saturation_tolerance, value - value_tolerance);
		Scalar upper_white = Scalar(hue + hue_tolerance, saturation + saturation_tolerance, value + value_tolerance);

		// HSV 값이 아래 범위를 초과하는 경우 처리
		lower_white[0] = MAX(lower_white[0], 0);
		upper_white[0] = MIN(upper_white[0], 360);
		lower_white[1] = MAX(lower_white[1], 0);
		upper_white[1] = MIN(upper_white[1], 255);
		lower_white[2] = MAX(lower_white[2], 0);
		upper_white[2] = MIN(upper_white[2], 255);

		inRange(img_hsv, lower_white, upper_white, white_mask);

		//// dilation 적용
		erode(white_mask, white_mask, Mat::ones(Size(5, 5), CV_8UC1), Point(-1, -1),2);
		dilate(white_mask, white_mask, Mat::ones(Size(5, 5), CV_8UC1), Point(-1, -1),2);
		bitwise_and(img, img, white_image, white_mask);

		imshow("white_image", white_image);
		waitKey(0);


	}

	// kmeans완료된 색상을 출력.
	Mat displayReducedBGRValues(100, bgrValuesReduced.size() * 100, CV_8UC3, Scalar(255, 255, 255));

	// 평균화된 BGR 값을 반복하여 이미지에 표시
	for (size_t i = 0; i < bgrValuesReduced.size(); i++) {
		Scalar bgr = bgrValuesReduced[i];
		Mat colorPatch(displayReducedBGRValues, Rect(i * 100, 0, 100, 100));
		colorPatch.setTo(Vec3b(bgr[0], bgr[1], bgr[2]));
	}

	// 이미지 크게보기
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

			// 둘러쌓인 윤곽선의 꼭짓점이 4개 이상일때만 판단.
			if (objCor > 4) {
				// 윤곽선이 얼마나 원에 가까운지 판단하는 circuaity.
				double circularity = (4 * M_PI * area) / (peri * peri);

				// circuaity의 threshold를 지정.
				double circularityThreshold = 0.075; // Adjust this value as needed

				if (circularity > circularityThreshold) {
					// 녹색 윤곽선을 그린다.
					drawContours(img, conPoly, i, Scalar(0, 255, 0), 2);

					// 윤곽선 내부 영역에 대하여 mask를 생성. 윤곽선 내부를 흰색으로 칠함.
					Mat maskInternal = Mat::zeros(img.size(), CV_8UC1);
					drawContours(maskInternal, conPoly, i, Scalar(255), FILLED);

					// 윤곽선 외부 영역에 대하여 mask를 생성. 윤곽선의 경계를 흰색으로 칠함.
					Mat maskExternal = Mat::zeros(img.size(), CV_8UC1);
					drawContours(maskExternal, conPoly, i, Scalar(255), 2); // 2의 두께를 가진 윤곽선을 외부영역에 대하여 그린다.

					// 두 개의 마스크를 결합하여 내부 영역의 마스크를 얻음.
					Mat mask = maskInternal - maskExternal;

					// 내부 영역에 대한 BGR값을 추출.
					vector<Scalar> bgrValues;
					for (int y = 0; y < img.rows; y++) {
						for (int x = 0; x < img.cols; x++) {
							if (mask.at<uchar>(y, x) == 255) {
								bgrValues.push_back(img.at<Vec3b>(y, x));
							}
						}
					}

					// bgrValuesList에 BGR값들을 저장.
					bgrValuesList.push_back(bgrValues);
				}
			}
		}
	}
}
