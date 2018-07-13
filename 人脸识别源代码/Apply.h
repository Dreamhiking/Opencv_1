#pragma once
#include"camera.h"
class Apply {
private:
	Picture p;
	string path = "D:\\computer\\workspace\\C++\\FacePerception\\FacePerception\\orl_faces";
	void printout(string path, int number, string fileName, ofstream &out);
public:
	Apply();
	void perFace();
	void addFace();
	void delFace();
	void updFace();
	void creatCSV();
	void simpleUI();
};