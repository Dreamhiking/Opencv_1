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

				cout << "ʶ����Ϊ��" << p.getName(k) << endl;
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
			cerr << "δʶ���������" << endl;
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
	cout << "������Ҫ��ӻ��޸ĵ�ѧ��������" << endl;
	cin >> name;
	
	_finddata_t file;
	long handle;
	string findpath = path + "\\*";
	handle = _findfirst(findpath.c_str(), &file);

	ofstream out;
	string outpath = path + "\\at.txt";
	out.open(outpath.c_str(),ios::app);

	if (handle == -1l) {
		cerr << "�ļ�����ʧ�ܣ�" << endl;
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
	cout << "������Ҫɾ����ѧ������" << endl;
	cin >> name;

	_finddata_t file;
	long handle;
	string filepath = path + "\\" + name+"\\*";
	handle = _findfirst(filepath.c_str(), &file);

	if (handle == -1l) {
		cerr << "�ļ�����ʧ�ܣ�" << endl;
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
		cerr << "�ļ�����ʧ�ܣ�" << endl;
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
	cout << "csv�ļ�������ɣ�" << endl;

}
void Apply::printout(string path, int number, string fileName, ofstream &out) {
	string findpath = path + "\\*";
	_finddata_t file;
	long handle;
	handle = _findfirst(findpath.c_str(), &file);
	if (handle == -1l) {
		cout << "�ļ�����ʧ��2" << endl;
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
	cout << "**************************��ӭ��������ʶ��ϵͳ*******************************" << endl << endl;
	while (i != '0') {
		system("cls");
		cout << "1------------------------------------------------------------��������ʶ��" << endl << endl;
		cout << "2------------------------------------------------------------���ѧ��" << endl << endl;
		cout << "3------------------------------------------------------------ɾ��ѧ��" << endl << endl;
		cout << "4------------------------------------------------------------����ģ��" << endl << endl;
		cout << "0------------------------------------------------------------�˳�ϵͳ" << endl << endl;

		i = getchar();
		while (getchar() != '\n');
		switch (i) {
		case '1': 
			perFace();

			break;

		case '2':addFace();	{
			cout << "�����С�����" << endl;
			creatCSV();
			p.doTrain();
			cout << "��Ϣ���³ɹ���" << endl;
			system("pause");
			break;
		}
		case '3':delFace();	{
			cout << "�����С�����" << endl;
			creatCSV();
			p.doTrain();
			cout << "��Ϣ���³ɹ���" << endl;
			system("pause");
			break;
		}
		case '4':addFace();	{	
			cout << "�����С�����" << endl;
			creatCSV();
			p.doTrain();
			cout << "��Ϣ���³ɹ���" << endl;	
			system("pause");
			break;
		}
		case '0':break;
		default: {
			cerr << "���������������������룡" << endl;
			system("pause");
			continue;
		}
		}
	}
	cout << "���˳���" << endl;
	system("pause");
}
