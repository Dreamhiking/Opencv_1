#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>

using namespace cv;


int main()
{
	//�򿪵�һ������ͷ
	VideoCapture cap(0);
	//����Ƿ�򿪳ɹ�
	if(!cap.isOpened())
	{
		return -1;
	}
	
	Mat frame;
	Mat edges;

	bool stop = false;
	while(!stop)
	{
		//��cap�ж�һ֡���浽frame
		cap>>frame;
		if(frame.empty())
			break;
		//��������ͼ��תΪΪ�Ҷ�ͼ
		cvtColor(frame, edges, CV_BGR2GRAY);

		GaussianBlur(edges, edges, Size(7,7), 1.5, 1.5);
		//���б�Ե��ȡ
		Canny(edges, edges, 0, 30, 3);
		//��ʾ���
		imshow("��ǰ��Ƶ",edges);
		//�ȴ�30�룬����������˳�ѭ��
		if(waitKey(30) >=0)
			stop = true;
	}
	return 0;
}