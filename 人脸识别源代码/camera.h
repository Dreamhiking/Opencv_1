#pragma once
class Picture {
private:
	String face_cascade_name = "haarcascades\\haarcascade_frontalface_default.xml";//正面人脸分类检测器
	String eyes_cascade_name = "haarcascades\\haarcascade_eye_tree_eyeglasses.xml";//人眼分类器
	CascadeClassifier face_cascade;   //定义人脸分类器  
	CascadeClassifier eyes_cascade;   //定义人眼分类器  
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
