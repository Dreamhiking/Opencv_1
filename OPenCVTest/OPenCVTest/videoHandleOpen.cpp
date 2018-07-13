#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>

using namespace cv;


int main()
{
	//打开第一个摄像头
	VideoCapture cap(0);
	//检测是否打开成功
	if(!cap.isOpened())
	{
		return -1;
	}
	
	Mat frame;
	Mat edges;

	bool stop = false;
	while(!stop)
	{
		//从cap中读一帧，存到frame
		cap>>frame;
		if(frame.empty())
			break;
		//将读到的图像转为为灰度图
		cvtColor(frame, edges, CV_BGR2GRAY);

		GaussianBlur(edges, edges, Size(7,7), 1.5, 1.5);
		//进行边缘提取
		Canny(edges, edges, 0, 30, 3);
		//显示结果
		imshow("当前视频",edges);
		//等待30秒，如果按键则退出循环
		if(waitKey(30) >=0)
			stop = true;
	}
	return 0;
}