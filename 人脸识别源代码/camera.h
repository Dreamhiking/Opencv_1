#pragma once
class Picture {
private:
	String face_cascade_name = "haarcascades\\haarcascade_frontalface_default.xml";//����������������
	String eyes_cascade_name = "haarcascades\\haarcascade_eye_tree_eyeglasses.xml";//���۷�����
	CascadeClassifier face_cascade;   //��������������  
	CascadeClassifier eyes_cascade;   //�������۷�����  
	String window_name = "Capture - Face detection";
	vector<string> names;

public:
	Picture();
	void saveTracedata(string path,string name);
	Mat cutPicture(Mat pic);
	void perFace();
	bool doTrain();
	int recFace(Mat pic);
	string getName(int k);
	void circleFace(Mat Pic);
	void read_csv(const string &filename, vector<Mat> &images, vector<int>&labels, char separator );
	void setName(string name);
	void clearName();
};
