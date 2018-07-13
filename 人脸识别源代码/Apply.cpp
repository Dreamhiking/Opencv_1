#include"stdafx.h"
#include"Apply.h"
#include<io.h>
#include<direct.h>
using namespace std;
Apply::Apply() {
	creatCSV();
	p.doTrain();
	system("pause");
}

void Apply::perFace() {
	VideoCapture cap;
	cap.open(0);
	if (!cap.isOpened()) {
		cout << "Camera opened failed!" << endl;
		return;
	}

	Mat frame;
	Mat frame_gray;
	namedWindow("videoTest");
	bool i = true;
	int k = 0;
	while (i)
	{
		try {
			char key = waitKey(100);
			cap >> frame;
			//capture.read(frame);
			p.circleFace(frame);
			imshow("videoTest", frame);
			if (key == 'q')
			{
				destroyWindow("videoTest");
				break;
			}


			switch (key)
			{
			case'p':
				i = false;
				frame_gray = p.cutPicture(frame);
				k = p.recFace(frame_gray);

				cout << "识别结果为：" << p.getName(k) << endl;
				imshow("photo", frame);
				waitKey(1000);
				destroyWindow("photo");
				waitKey(1000);
				system("pause");
				//imshow("photo_gray", frame_gray);
				//waitKey(1000);
				//destroyWindow("photo_gray");
				break;
			default:
				break;
			}
		}
		catch (Exception e) {
			cerr << "未识别出人脸！" << endl;
			i = i - 1;
			continue;
		}
	}
	destroyWindow("videoTest");
	//if ((char)c == 27) { return 0; } // escape  
	//}  



}

void Apply::addFace() {
	string name;
	cout << "请输入要添加或修改的学生姓名：" << endl;
	cin >> name;
	
	_finddata_t file;
	long handle;
	string findpath = path + "\\*";
	handle = _findfirst(findpath.c_str(), &file);

	ofstream out;
	string outpath = path + "\\at.txt";
	out.open(outpath.c_str(),ios::app);

	if (handle == -1l) {
		cerr << "文件查找失败！" << endl;
	}
	
	int i = 0;
	bool flag = false;
	do {
		if (file.attrib & _A_SUBDIR) {
			if ((strcmp(file.name, ".") != 0) && (strcmp(file.name, "..") != 0)) {
				if (strcmp(file.name, name.c_str()) == 0)
					flag = true;
				i++;
			}
		}
	} while (_findnext(handle, &file) == 0);
	
	if (!flag) {
		_mkdir((path + "\\" + name).c_str());
		for (int k = 0; k < 10; k++) {
			out << path << "\\" << name << "\\" << format("d%.jpg", k) << ";" << i << ";" << name << endl;
		}
	}
	p.saveTracedata(path, name);
	system("pause");
}

void Apply::updFace() {

}

void Apply::delFace() {
	string name;
	cout << "请输入要删除的学生名：" << endl;
	cin >> name;

	_finddata_t file;
	long handle;
	string filepath = path + "\\" + name+"\\*";
	handle = _findfirst(filepath.c_str(), &file);

	if (handle == -1l) {
		cerr << "文件查找失败！" << endl;
	}

	do {
		if ((strcmp(file.name, ".") != 0) && (strcmp(file.name, "..") != 0)) {
		string s = "del " + path + "\\" + name + "\\" + file.name;
		system(s.c_str());
		cout << s.c_str()<< endl;
	}
	} while (_findnext(handle, &file) == 0);
	string s= "rd " + path + "\\" + name +"\n";

	cout << s.c_str() << endl;
	_findclose(handle);	
	system(s.c_str());
	system("pause");


}

void Apply::creatCSV() {
	p.clearName();
	_finddata_t file;
	long handle;
	string findpath = path + "\\*";
	handle = _findfirst(findpath.c_str(), &file);

	ofstream out;
	string outfile = path + "\\at.txt";
	remove(outfile.c_str());
	out.open(outfile.c_str(), ios::app);

	if (handle == -1l) {
		cerr << "文件查找失败！" << endl;
	}
	int i = 0;

	do {
		if (file.attrib & _A_SUBDIR) {
			if ((strcmp(file.name, ".") != 0) && (strcmp(file.name, "..") != 0)) {
				p.setName(file.name);
				string newPath = path + "\\" + file.name;
				printout(newPath, i, file.name, out);
				i++;
			}
		}
	} while (_findnext(handle, &file) == 0);
	out.flush();
	out.close();
	cout << "csv文件创建完成！" << endl;

}
void Apply::printout(string path, int number, string fileName, ofstream &out) {
	string findpath = path + "\\*";
	_finddata_t file;
	long handle;
	handle = _findfirst(findpath.c_str(), &file);
	if (handle == -1l) {
		cout << "文件查找失败2" << endl;
		system("pause");
		return;
	}
	do {
		if ((strcmp(file.name, ".") != 0) && (strcmp(file.name, "..") != 0)) {
			out << path << "\\" << file.name << ";" << number << ";" << fileName << endl;
		}
	} while (_findnext(handle, &file) == 0);
}

void Apply::simpleUI() {
	int i = 1;
	cout << "**************************欢迎进入人脸识别系统*******************************" << endl << endl;
	while (i != '0') {
		system("cls");
		cout << "1------------------------------------------------------------进行人脸识别" << endl << endl;
		cout << "2------------------------------------------------------------添加学生" << endl << endl;
		cout << "3------------------------------------------------------------删除学生" << endl << endl;
		cout << "4------------------------------------------------------------更新模型" << endl << endl;
		cout << "0------------------------------------------------------------退出系统" << endl << endl;

		i = getchar();
		while (getchar() != '\n');
		switch (i) {
		case '1': 
			perFace();

			break;

		case '2':addFace();	{
			cout << "更新中。。。" << endl;
			creatCSV();
			p.doTrain();
			cout << "信息更新成功！" << endl;
			system("pause");
			break;
		}
		case '3':delFace();	{
			cout << "更新中。。。" << endl;
			creatCSV();
			p.doTrain();
			cout << "信息更新成功！" << endl;
			system("pause");
			break;
		}
		case '4':addFace();	{	
			cout << "更新中。。。" << endl;
			creatCSV();
			p.doTrain();
			cout << "信息更新成功！" << endl;	
			system("pause");
			break;
		}
		case '0':break;
		default: {
			cerr << "您的输入有误，请重新输入！" << endl;
			system("pause");
			continue;
		}
		}
	}
	cout << "已退出！" << endl;
	system("pause");
}
