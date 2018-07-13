#include "stdafx.h"
#include "camera.h"
/** Function Headers */

using namespace face;
/** Global variables */
Picture::Picture(){
	//-- 1. Load the cascades  
	if (!face_cascade.load(face_cascade_name)) {
		printf("--(!)Error loading face cascade\n");
		return ;
	}
	else
		printf("���������������ɹ���\n");
	if (!eyes_cascade.load(eyes_cascade_name)) {
		printf("--(!)Error loading eyes cascade\n");
		return ;
	}
	else
		printf("�������۷������ɹ���\n");
};
void Picture::saveTracedata(string path,string name){
	VideoCapture cap;
	cap.open(0);
	if (!cap.isOpened()) {
		cout << "Camera opened failed!" << endl;
		return ;
	}

	Mat frame;
	Mat frame_gray;
	namedWindow("videoTest");
	int i = 0;
	while(i<10)
	{
		try {
			char key = waitKey(100);
			cap >> frame;
			//capture.read(frame);
			circleFace(frame);
			imshow("videoTest", frame);
			if (key == 'q')
			{
				destroyWindow("videoTest");
				break;
			}
			string filename =path + format("\\%s\\%d.jpg",name, i);
	
			switch (key)
			{
			case'p':
				i++;
				frame_gray = cutPicture(frame);
				imwrite(filename, frame_gray);
				cout << "��д��" << filename << endl;
				imshow("photo", frame);
				waitKey(1000);
				destroyWindow("photo");
				//imshow("photo_gray", frame_gray);
				//waitKey(1000);
				//destroyWindow("photo_gray");
				break;
			default:
				break;
			}
		}
		catch (Exception e) {
			cerr << "δʶ���������" << endl;
			i = i - 1;
			continue;
		}
	}
	destroyWindow("videoTest");
}

Mat Picture::cutPicture(Mat frame) {
	try {

		std::vector<Rect> faces;//����
		Mat frame_gray;//�Ҷ�ͼ

		cvtColor(frame, frame_gray, COLOR_BGR2GRAY);//ת��Ϊ�Ҷ�ͼ

													//imshow("gray_ori", frame_gray);

		equalizeHist(frame_gray, frame_gray);//���⻯�Ҷ�ͼ

											 //imshow("gray_eql", frame_gray);

											 //-- Detect faces  
		face_cascade.detectMultiScale(frame_gray, faces, 1.1, 3, CV_HAAR_DO_ROUGH_SEARCH, Size(70, 70), Size(500, 500));
		//ͼƬ���洢��⵽��Ŀ��λ�ã�ÿ�ηŴ�1.1�����ظ�ʶ������Ϊ�Ϸ����������Բ�ѯ����С70x70�����100x100��
		if (faces.empty()) {
			cerr << "δʶ�������!" << endl;
			Exception e;
			throw e;
		}
		for (size_t i = 0; i < faces.size(); i++)
		{
			//Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);  
			//ellipse(frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);  
			//rectangle(frame, faces[i], Scalar(255, 0, 0), 1, 8, 0);
			//ͼ�񣬾��Σ�����������ɫ��������ϸ���������ͣ�shift������С����λ����
			//Ȧ������λ��

			//imshow("���ƾ���", frame);
			Mat faceROI = frame_gray(faces[i]);//roi��Ȥ����
			printf("��ȡ�����ɹ���\n");
			Mat myFace;
			resize(faceROI, myFace, Size(92, 112));

			return myFace;
			//imshow("��Ȥ����", faceROI);

/*std::vector<Rect> eyes;

//-- In each face, detect eyes
eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 1, CV_HAAR_DO_ROUGH_SEARCH, Size(3, 3));

for (size_t j = 0; j < eyes.size(); j++)
{
	Rect rect(faces[i].x + eyes[j].x, faces[i].y + eyes[j].y, eyes[j].width, eyes[j].height);

	//Point eye_center(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
	//int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);
	//circle(frame, eye_center, radius, Scalar(255, 0, 0), 4, 8, 0);
	rectangle(frame, rect, Scalar(0, 255, 0), 1, 8, 0);
	//Ȧ���۾�λ��
}*/
		}

		//-- Show what you got  
		//namedWindow(window_name, 2);
		//imshow(window_name, frame);
	}
	catch (Exception e) {
		cerr << "��������ʧ�ܣ�" << endl;
		throw e;
	}
}

void Picture::circleFace(Mat frame) {

	std::vector<Rect> faces;//����
	Mat frame_gray;//�Ҷ�ͼ

	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);//ת��Ϊ�Ҷ�ͼ

	equalizeHist(frame_gray, frame_gray);//���⻯�Ҷ�ͼ

	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 3, CV_HAAR_DO_ROUGH_SEARCH, Size(70, 70), Size(500, 500));
	//ͼƬ���洢��⵽��Ŀ��λ�ã�ÿ�ηŴ�1.1�����ظ�ʶ������Ϊ�Ϸ����������Բ�ѯ����С70x70�����100x100��

	for (size_t i = 0; i < faces.size(); i++)
	{
		//Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);  
		//ellipse(frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);  
		rectangle(frame, faces[i], Scalar(255, 0, 0), 1, 8, 0);
		//ͼ�񣬾��Σ�����������ɫ��������ϸ���������ͣ�shift������С����λ����
		//Ȧ������λ��
		
		Mat faceROI = frame_gray(faces[i]);//roi��Ȥ����

		std::vector<Rect> eyes;

		//-- In each face, detect eyes
		eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 1, CV_HAAR_DO_ROUGH_SEARCH, Size(3, 3));
 
		/*for (size_t j = 0; j < eyes.size(); j++)
		{
		Rect rect(faces[i].x + eyes[j].x, faces[i].y + eyes[j].y, eyes[j].width, eyes[j].height);

		//Point eye_center(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
		//int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);
		//circle(frame, eye_center, radius, Scalar(255, 0, 0), 4, 8, 0);
		rectangle(frame, rect, Scalar(0, 255, 0), 1, 8, 0);
		//Ȧ���۾�λ��
		}*/
	}
	//return frame;
	//-- Show what you got  
	//namedWindow(window_name, 2);
	//imshow(window_name, frame);
}

bool Picture::doTrain() {
	string fn_csv = "orl_faces\\at.txt";
	vector<Mat> images;
	vector<int> labels;
	try {//�������ļ����˳�
		read_csv(fn_csv, images, labels, ';');
	}
	catch (cv::Exception&e) {
		cerr << "Error opening file\"" << fn_csv << "\".Reason:" << e.msg << endl;
		return false;
	}
	if (images.size() <= 1) {
		string error_message = "ͼƬ�����������飬�޷��������С�";
		CV_Error(CV_StsError, error_message);
	}
	
	Mat testSample = images[0];
	int testLabel = labels[0];
	cout << "����ѵ��ģ�ͣ�" << endl;

	Ptr<FaceRecognizer> model = createEigenFaceRecognizer();
	model->train(images, labels);
	model->save("MyFacePCAModel.xml");

	Ptr<FaceRecognizer> model1 = createFisherFaceRecognizer();
	model1->train(images, labels);
	model1->save("MyFaceFisherModel.xml");

	Ptr<FaceRecognizer> model2 = createLBPHFaceRecognizer();
	model2->train(images, labels);
	model2->save("MyFaceLBPHModel.xml");

	// ����Բ���ͼ�����Ԥ�⣬predictedLabel��Ԥ���ǩ��� 
	int predictedLabel = model->predict(testSample);
	int predictedLabel1 = model1->predict(testSample);
	int predictedLabel2 = model2->predict(testSample);

	int result = labels[predictedLabel];
	int result1 = labels[predictedLabel1];
	int result2 = labels[predictedLabel2];

	string result_message = format("Predicted class = %d / Actual class = %d.", result, testLabel);
	string result_message1 = format("Predicted class = %d / Actual class = %d.", result1, testLabel);
	string result_message2 = format("Predicted class = %d / Actual class = %d.", result2, testLabel);
	cout << result_message << endl;
	cout << result_message1 << endl;
	cout << result_message2 << endl;
	images.clear();
	labels.clear();
	return true;

}

void Picture::read_csv(const string &filename, vector<Mat> &images, vector<int> &labels, char separator ) {
	std::ifstream file(filename.c_str(), ifstream::in); 
	if (!file) {
		string error_Message = "No valid input file was given,please check the given filename.";
		CV_Error(CV_StsBadArg, error_Message);
	}
	string line, path,classlabel,name;
	Mat a;
	cout << "��ȡcsv�ļ���" << endl;
	while (getline(file, line)) {
		stringstream liness(line);
		getline(liness, path, separator);
		getline(liness, classlabel,separator);
		getline(liness, name);
		if (!path.empty() && !classlabel.empty()) {
			images.push_back(imread(path, 0));
			labels.push_back(atoi(classlabel.c_str()));
		}
	}
}

static Mat norm_0_255(InputArray _src) {
	Mat src = _src.getMat();

	Mat dst;
	switch (src.channels()) {
	case 1:
		cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
		break;
	case 3:
		cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
		break;
	default:
		src.copyTo(dst);
		break;
	}
	return dst;
}

int Picture::recFace(Mat face){
	CascadeClassifier cascade;
	bool stop = false;
	//������������ģ��
	cascade.load("haarcascade_frontalface_alt.xml");
	//����ѵ��ģ��
	Ptr<FaceRecognizer> modelPCA = createEigenFaceRecognizer();
	modelPCA->load("MyFacePCAModel.xml");
	cout << "ѵ��ģ�ͼ��سɹ���" << endl;
	
	int predictPCA = modelPCA->predict(face);
	cout << predictPCA << endl;
	return predictPCA;

}

string Picture::getName(int i) {
	return names[i];
}

void Picture::perFace() {
}
void Picture::setName(string name) {
	names.push_back(name);
}
void Picture::clearName() {
	names.clear();
}