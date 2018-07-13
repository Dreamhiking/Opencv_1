/*#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main()
{
	char filename[100];
	 
	for (int i = 1; i < 4; i++)
	{
		sprintf(filename, "C:\\Users\\lenovo\\Desktop\\测试图片\\%d.jpg", i);
		// 读入图片
		Mat Img = imread(filename, 0);
		if (!Img.data)
		{
			break;
		}
		
		namedWindow("显示图片"+i, 0);
		imshow("显示图片"+i, Img);


		// 可以在此对于Img的图像处理或者其他操作
		
        // 等待时间为1s
		waitKey(1000);

	}

	system("pause");
	return 0;
}*/
/*
#include<iostream>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>


using namespace std;
using namespace cv;
int main(){
	char filename[100],savename[100];
	Mat image1;
	image1 = imread("C:\\Users\\lenovo\\Desktop\\测试图片\\模板.jpg");
	int flag=0;
	for (int i = 1; i < 13; i++){
		sprintf(filename, "C:\\Users\\lenovo\\Desktop\\测试图片\\%d.jpg", i);
		Mat image2 = imread(filename);
		if((!image1.data) || (!image2.data))
		  return -1;
	    Mat test=image2.clone();
	    int rows=image1.rows;
	    int cols=image1.cols;
	    for(int i=0;i<rows;i++){
		for(int j=0;j<cols;j++){
			test.at<Vec3b>(i,j)[0]=image1.at<Vec3b>(i,j)[0]==0 ? 0 : test.at<Vec3b>(i,j)[0];
			test.at<Vec3b>(i,j)[1]=image1.at<Vec3b>(i,j)[1]==0 ? 0 : test.at<Vec3b>(i,j)[1];
			test.at<Vec3b>(i,j)[2]=image1.at<Vec3b>(i,j)[1]==0 ? 0 : test.at<Vec3b>(i,j)[2];
		}
	}
	
	//imshow("image1",image1);
	//imshow("image2",image2);
//	imshow("test"+i,test);         //最终显示的生成图片
	sprintf(savename, "C:\\Users\\lenovo\\Desktop\\测试图片\\NEW\\%d.jpg", flag);
	imwrite(savename, test);
	flag++;
	waitKey(1000);
	    
	}

	system("pause");
	return 0;
	
	
}
*/
/*
#include <windows.h>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

char* WcharToChar(const wchar_t* wp)  
{  
    char *m_char;
    int len= WideCharToMultiByte(CP_ACP,0,wp,wcslen(wp),NULL,0,NULL,NULL);  
    m_char=new char[len+1];  
    WideCharToMultiByte(CP_ACP,0,wp,wcslen(wp),m_char,len,NULL,NULL);  
    m_char[len]='\0';  
    return m_char;  
}  

wchar_t* CharToWchar(const char* c)  
{   
	wchar_t *m_wchar;
    int len = MultiByteToWideChar(CP_ACP,0,c,strlen(c),NULL,0);  
    m_wchar=new wchar_t[len+1];  
    MultiByteToWideChar(CP_ACP,0,c,strlen(c),m_wchar,len);  
    m_wchar[len]='\0';  
    return m_wchar;  
}  

wchar_t* StringToWchar(const string& s)  
{  
    const char* p=s.c_str();  
    return CharToWchar(p);  
} 
int main(){
	const string fileform = "*.jpg";
    const string perfileReadPath = "testSamples";
	const string perfileReadPath1 = "testSamples1";
	string  fileReadName,
            fileReadPath;
	char temp[256];
	sprintf(temp, "%d", 3);
   //读取每个类文件夹下所有图像
		int j = 0;//每一类读取图像个数计数
		fileReadPath = perfileReadPath + "/" + temp + "/" + fileform;

		HANDLE hFile;
		LPCTSTR lpFileName = StringToWchar(fileReadPath);//指定搜索目录和文件类型，如搜索d盘的音频文件可以是"D:\\*.mp3"
		WIN32_FIND_DATA pNextInfo;  //搜索得到的文件信息将储存在pNextInfo中;
		hFile = FindFirstFile(lpFileName,&pNextInfo);//请注意是 &pNextInfo , 不是 pNextInfo;
		if(hFile == INVALID_HANDLE_VALUE)
		{
			exit(-1);//搜索失败
		}
		//do-while循环读取
		do
		{	
			if(pNextInfo.cFileName[0] == '.')//过滤.和..
				continue;
			j++;//读取一张图
			//wcout<<pNextInfo.cFileName<<endl;
			printf("%s\n",WcharToChar(pNextInfo.cFileName));
			//对读入的图片进行处理
			Mat srcImage = imread( perfileReadPath + "/" + temp + "/" + WcharToChar(pNextInfo.cFileName),CV_LOAD_IMAGE_GRAYSCALE);
			int xRoi=200;
			int yRoi=140;
			int widthRoi=80;
			int heightRoi=60;
			Mat roiImage(srcImage.rows,srcImage.cols,CV_8UC3);
			cout<<srcImage.rows<<" "<<srcImage.cols<<endl;
			srcImage(Rect(xRoi,yRoi,widthRoi,heightRoi)).copyTo(roiImage);
			imwrite(perfileReadPath1 + "/" + temp + "/" + WcharToChar(pNextInfo.cFileName),roiImage);
		} while (FindNextFile(hFile,&pNextInfo) && j<74);//如果设置读入的图片数量，则以设置的为准，如果图片不够，则读取文件夹下所有图片
        
    
    return 0;
}
*/

/*
#include <windows.h>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

char* WcharToChar(const wchar_t* wp)  
{  
    char *m_char;
    int len= WideCharToMultiByte(CP_ACP,0,wp,wcslen(wp),NULL,0,NULL,NULL);  
    m_char=new char[len+1];  
    WideCharToMultiByte(CP_ACP,0,wp,wcslen(wp),m_char,len,NULL,NULL);  
    m_char[len]='\0';  
    return m_char;  
}  

wchar_t* CharToWchar(const char* c)  
{   
	wchar_t *m_wchar;
    int len = MultiByteToWideChar(CP_ACP,0,c,strlen(c),NULL,0);  
    m_wchar=new wchar_t[len+1];  
    MultiByteToWideChar(CP_ACP,0,c,strlen(c),m_wchar,len);  
    m_wchar[len]='\0';  
    return m_wchar;  
}  

wchar_t* StringToWchar(const string& s)  
{  
    const char* p=s.c_str();  
    return CharToWchar(p);  
}  

int main()
{
	const string fileform = "*.jpg";
    const string perfileReadPath = "Samples";

    const int sample_mun_perclass = 20;//训练字符每类数量
    const int class_mun = 2;//训练字符类数

    const int image_cols = 30;
    const int image_rows = 30;
    string  fileReadName,
            fileReadPath;
    char temp[256];

    float trainingData[class_mun*sample_mun_perclass][image_rows*image_cols] = {{0}};//每一行一个训练样本
    float labels[class_mun*sample_mun_perclass][class_mun]={{0}};//训练样本标签

    for(int i=0;i<=class_mun-1;++i)//不同类
    {
		//读取每个类文件夹下所有图像
		int j = 0;//每一类读取图像个数计数
        sprintf(temp, "%d", i);
		fileReadPath = perfileReadPath + "/" +temp+"/" + fileform;
		cout<<"文件夹"<<i<<endl;
		HANDLE hFile;
		LPCTSTR lpFileName = StringToWchar(fileReadPath);//指定搜索目录和文件类型，如搜索d盘的音频文件可以是"D:\\*.mp3"
		WIN32_FIND_DATA pNextInfo;  //搜索得到的文件信息将储存在pNextInfo中;
		hFile = FindFirstFile(lpFileName,&pNextInfo);//请注意是 &pNextInfo , 不是 pNextInfo;
		if(hFile == INVALID_HANDLE_VALUE)
		{
			exit(-1);//搜索失败
		}
		//do-while循环读取
		do
		{	
			if(pNextInfo.cFileName[0] == '.')//过滤.和..
				continue;
			j++;//读取一张图
			//wcout<<pNextInfo.cFileName<<endl;
			printf("%s\n",WcharToChar(pNextInfo.cFileName));
			//对读入的图片进行处理
			Mat srcImage = imread( perfileReadPath + "/" + temp + "/" + WcharToChar(pNextInfo.cFileName),CV_LOAD_IMAGE_GRAYSCALE);
			Mat resizeImage;
			Mat trainImage;
			Mat result;

			resize(srcImage,resizeImage,Size(image_cols,image_rows),(0,0),(0,0),CV_INTER_AREA);//使用象素关系重采样。当图像缩小时候，该方法可以避免波纹出现
			//threshold(resizeImage,trainImage,0,255,CV_THRESH_BINARY|CV_THRESH_OTSU);

			for(int k = 0; k<image_rows*image_cols; ++k)
            {
				trainingData[i*sample_mun_perclass+(j-1)][k] = (float)resizeImage.data[k];
                //trainingData[i*sample_mun_perclass+(j-1)][k] = (float)trainImage.at<unsigned char>((int)k/8,(int)k%8);//(float)train_image.data[k];
                //cout<<trainingData[i*sample_mun_perclass+(j-1)][k] <<" "<< (float)trainImage.at<unsigned char>(k/8,k%8)<<endl;
            }

		} while (FindNextFile(hFile,&pNextInfo) && j<sample_mun_perclass);//如果设置读入的图片数量，则以设置的为准，如果图片不够，则读取文件夹下所有图片
		
    }

	// Set up training data Mat
	Mat trainingDataMat(class_mun*sample_mun_perclass, image_rows*image_cols, CV_32FC1, trainingData);
	cout<<"trainingDataMat——OK！"<<endl;

    // Set up label data 
    for(int i=0;i<=class_mun-1;++i)
    {
        for(int j=0;j<=sample_mun_perclass-1;++j)
        {
            for(int k = 0;k<class_mun;++k)
            {
                if(k==i)
                    labels[i*sample_mun_perclass + j][k] = 1;
                else labels[i*sample_mun_perclass + j][k] = 0;
            }
        }
    }
    Mat labelsMat(class_mun*sample_mun_perclass, class_mun, CV_32FC1,labels);
	cout<<"labelsMat:"<<endl;
	cout<<labelsMat<<endl;
	cout<<"labelsMat——OK！"<<endl;

	//训练代码

	cout<<"training start...."<<endl;
    CvANN_MLP bp;
    // Set up BPNetwork's parameters
    CvANN_MLP_TrainParams params;
    params.train_method=CvANN_MLP_TrainParams::BACKPROP;
    params.bp_dw_scale=0.001;
    params.bp_moment_scale=0.1;
    params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,10000,0.0001);  //设置结束条件
    //params.train_method=CvANN_MLP_TrainParams::RPROP;
    //params.rp_dw0 = 0.1;
    //params.rp_dw_plus = 1.2;
    //params.rp_dw_minus = 0.5;
    //params.rp_dw_min = FLT_EPSILON;
    //params.rp_dw_max = 50.;

    //Setup the BPNetwork
    Mat layerSizes=(Mat_<int>(1,5) << image_rows*image_cols,128,128,128,class_mun);
    bp.create(layerSizes,CvANN_MLP::SIGMOID_SYM,1.0,1.0);//CvANN_MLP::SIGMOID_SYM
                                               //CvANN_MLP::GAUSSIAN
                                               //CvANN_MLP::IDENTITY
    cout<<"training...."<<endl;
    bp.train(trainingDataMat, labelsMat, Mat(),Mat(), params);

    bp.save("../bpcharModel.xml"); //save classifier
    cout<<"training finish...bpModel1.xml saved "<<endl;


	//测试神经网络
	cout<<"测试："<<endl;
	Mat test_image = imread("REC0009_148.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	Mat test_temp;
	resize(test_image,test_temp,Size(image_cols,image_rows),(0,0),(0,0),CV_INTER_AREA);//使用象素关系重采样。当图像缩小时候，该方法可以避免波纹出现
	threshold(test_temp,test_temp,0,255,CV_THRESH_BINARY|CV_THRESH_OTSU);
	Mat_<float>sampleMat(1,image_rows*image_cols); 
	for(int i = 0; i<image_rows*image_cols; ++i)  
    {  
        //sampleMat.at<float>(0,i) = (float)test_temp.at<uchar>(i/8,i%8);
		sampleMat.at<float>(0,i) = (float)test_temp.data[i];
    }  
	
	Mat responseMat;  
	bp.predict(sampleMat,responseMat);  
	Point maxLoc;
	double maxVal = 0;
	minMaxLoc(responseMat,NULL,&maxVal,NULL,&maxLoc);
	cout<<"识别结果："<<maxLoc.x<<"	相似度:"<<maxVal*100<<"%"<<endl;
	imshow("test_image",test_image);  
	waitKey(0);
	
	return 0;
}
*/

	/*
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>

using namespace cv;

void Pic2Gray(Mat camerFrame,Mat &gray)
{
	//普通台式机3通道BGR,移动设备为4通道
	if (camerFrame.channels() == 3)
	{
		cvtColor(camerFrame, gray, CV_BGR2GRAY);
	}
	else if (camerFrame.channels() == 4)
	{
		cvtColor(camerFrame, gray, CV_BGRA2GRAY);
	}
	else
		gray = camerFrame;
}


int main()
{
	//加载Haar或LBP对象或人脸检测器
	CascadeClassifier faceDetector;
	std::string faceCascadeFilename = "haarcascade_frontalface_default.xml";

	//友好错误信息提示
	try{
		faceDetector.load(faceCascadeFilename);
	}
	catch (cv::Exception e){}
	if (faceDetector.empty())
	{
		std::cerr << "脸部检测器不能加载 (";
		std::cerr << faceCascadeFilename << ")!" << std::endl;
		exit(1);
	}

	//打开摄像头
	VideoCapture camera(0);
	while (true)
	{
		Mat camerFrame;
	//	camerFrame=imread("图片多人脸.png");
		camera >> camerFrame;
		if (camerFrame.empty())
		{
			std::cerr << "无法获取摄像头图像" << std::endl;
			getchar();
			exit(1);
		}
		Mat displayedFrame(camerFrame.size(),CV_8UC3);


		//人脸检测只试用于灰度图像
		Mat gray;
		Pic2Gray(camerFrame, gray);



		//直方图均匀化(改善图像的对比度和亮度)
		Mat equalizedImg;
		equalizeHist(gray, equalizedImg);

		//人脸检测用Cascade Classifier::detectMultiScale来进行人脸检测

		//int flags = CASCADE_FIND_BIGGEST_OBJECT|CASCADE_DO_ROUGH_SEARCH;	//只检测脸最大的人
		int flags = CASCADE_SCALE_IMAGE;	//检测多个人
		Size minFeatureSize(30, 30);
		float searchScaleFactor = 1.1f;
		int minNeighbors = 4;
		std::vector<Rect> faces;
		faceDetector.detectMultiScale(equalizedImg, faces, searchScaleFactor, minNeighbors, flags, minFeatureSize);

		//画矩形框
		cv::Mat face;
		cv::Point text_lb;
		for (size_t i = 0; i < faces.size(); i++)
		{
			if (faces[i].height > 0 && faces[i].width > 0)
			{
				face = gray(faces[i]);
				text_lb = cv::Point(faces[i].x, faces[i].y);
				cv::rectangle(equalizedImg, faces[i], cv::Scalar(255, 0, 0), 1, 8, 0);
				cv::rectangle(gray, faces[i], cv::Scalar(255, 0, 0), 1, 8, 0);
				cv::rectangle(camerFrame, faces[i], cv::Scalar(255, 0, 0), 1, 8, 0);
			}
		}


		imshow("直方图均匀化", equalizedImg);
		imshow("灰度化", gray);
		imshow("原图", camerFrame);

		waitKey(20);
	}

	getchar();
	return 0;
}
*/

/*
#include <opencv2/opencv.hpp>
#include <iostream>
#include <Windows.h>

using namespace std;
using namespace cv;

void AllImagePro( string src, string dst, static int number );
void detectAndCut( Mat img ,string dir, string filename);
char* WcharToChar(const wchar_t* wp);
wchar_t* CharToWchar(const char* c);
wchar_t* CharToWchar(const char* c);
wchar_t* StringToWchar(const string& s);

CascadeClassifier face_cascade;
String face_cascade_name = "haarcascade_frontalface_alt.xml";

int main()
{
    //在文件夹0图片中寻找人脸并截取存放到文件夹0cut中
    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
    AllImagePro( "student", "0cut" ,20);
    cout<<"cut……OK！"<<endl;

    return 0;
}

//读取目录src下min（number，所有图像）图像提取人脸并保存到srccut目录，
//参数：原图片目录src       剪切图片保存目录dst     读取最大数量number    
void AllImagePro( string src, string dst, static int number )
{
    int count=0;
    string src1 = src;  
    string src1cut = dst;
    HANDLE hFile;
    LPCTSTR lpFileName = StringToWchar(src1+"/"+"*.*"); //指定搜索目录和文件类型，如搜索d盘的音频文件可以是"D:\\*.mp3"
    WIN32_FIND_DATA pNextInfo;  //搜索得到的文件信息将储存在pNextInfo中;
    hFile = FindFirstFile(lpFileName,&pNextInfo);//请注意是 &pNextInfo , 不是 pNextInfo;
    if(hFile == INVALID_HANDLE_VALUE)
    {
        //搜索失败
        exit(-1);
    }
    cout<<"文件夹"<<src<<"找到的图片："<<endl;
    do
    {
        if(pNextInfo.cFileName[0] == '.')//过滤.和..
            continue;
        count++;
        printf("%s\n",WcharToChar(pNextInfo.cFileName));
        Mat img = imread( src1 + "/" + WcharToChar(pNextInfo.cFileName) , 1 );
        detectAndCut( img ,src1cut ,WcharToChar(pNextInfo.cFileName) );
    }while (FindNextFile(hFile,&pNextInfo) && count<number);//如果设置读入的图片数量，则以设置的为准，如果图片不够，则读取文件夹下所有图片
}

//人脸检测
//参数：待检测图像img       保存路径dir     保存文件名name
void detectAndCut( Mat img ,string dir, string filename)
{
   std::vector<Rect> faces;
   Mat img_gray;

   cvtColor( img, img_gray, COLOR_BGR2GRAY );
   equalizeHist( img_gray, img_gray );
   //-- Detect faces
   face_cascade.detectMultiScale( img_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );

   for( size_t i = 0; i < faces.size(); i++ )
    {
      Point rec( faces[i].x, faces[i].y );
      Point rec2( faces[i].x + faces[i].width, faces[i].y + faces[i].height );
      Mat roi_img = img( Range(faces[i].y,faces[i].y + faces[i].height), Range(faces[i].x,faces[i].x + faces[i].width) );
      imwrite( dir+"/"+filename, roi_img );   
   }
}


char* WcharToChar(const wchar_t* wp)  
{  
    char *m_char;
    int len= WideCharToMultiByte(CP_ACP,0,wp,wcslen(wp),NULL,0,NULL,NULL);  
    m_char=new char[len+1];  
    WideCharToMultiByte(CP_ACP,0,wp,wcslen(wp),m_char,len,NULL,NULL);  
    m_char[len]='\0';  
    return m_char;  
}  

wchar_t* CharToWchar(const char* c)  
{   
    wchar_t *m_wchar;
    int len = MultiByteToWideChar(CP_ACP,0,c,strlen(c),NULL,0);  
    m_wchar=new wchar_t[len+1];  
    MultiByteToWideChar(CP_ACP,0,c,strlen(c),m_wchar,len);  
    m_wchar[len]='\0';  
    return m_wchar;  
}  

wchar_t* StringToWchar(const string& s)  
{  
    const char* p=s.c_str();  
    return CharToWchar(p);  
}  
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <windows.h>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndCut( Mat img ,string dir ,string filename );
void AllImagePro( string src, string dst, const int number );
char* WcharToChar(const wchar_t* wp);
wchar_t* CharToWchar(const char* c);
wchar_t* StringToWchar(const string& s);
string getstring ( const int n );

CascadeClassifier face_cascade;
String face_cascade_name = "haarcascade_frontalface_alt.xml";

//主函数
int main()
{
    const int sample_mun_perclass = 12;//训练每类图片数量
    const int class_mun = 2;//训练类数 一类是美女，一类是丑女 ^-^

    const int image_cols = 30;
    const int image_rows = 30;

    string  fileReadName,fileReadPath;

    float trainingData[class_mun*sample_mun_perclass][image_rows*image_cols] = {{0}};//每一行一个训练样本
    float labels[class_mun*sample_mun_perclass][class_mun]={{0}};//训练样本标签

    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
    AllImagePro( "0", "0cut" ,sample_mun_perclass);
    AllImagePro( "1", "1cut" ,sample_mun_perclass);
    cout<<"cut……OK！"<<endl;

     for(int i=0;i<class_mun;++i)//不同类
    {
        //读取每个类文件夹下所有图像
        int j = 0;//每一类读取图像个数计数
        fileReadPath = getstring(i) + "cut/" + "*.*";
        cout<<"文件夹"<<i<<endl;
        HANDLE hFile;
        LPCTSTR lpFileName = StringToWchar(fileReadPath);//指定搜索目录和文件类型，如搜索d盘的音频文件可以是"D:\\*.mp3"
        WIN32_FIND_DATA pNextInfo;  //搜索得到的文件信息将储存在pNextInfo中;
        hFile = FindFirstFile(lpFileName,&pNextInfo);//请注意是 &pNextInfo , 不是 pNextInfo;
        if(hFile == INVALID_HANDLE_VALUE)
        {
            exit(-1);//搜索失败
        }
        //do-while循环读取
        do
        {
            if(pNextInfo.cFileName[0] == '.')//过滤.和..
                continue;
            //wcout<<pNextInfo.cFileName<<endl;
            j++;
            printf("%s\n",WcharToChar(pNextInfo.cFileName));
            //对读入的图片进行处理
            Mat srcImage = imread( getstring(i) + "/" + WcharToChar(pNextInfo.cFileName),CV_LOAD_IMAGE_GRAYSCALE);
            Mat trainImage;

            resize(srcImage,trainImage,Size(image_cols,image_rows),(0,0),(0,0),CV_INTER_AREA);//使用象素关系重采样。当图像缩小时候，该方法可以避免波纹出现
           // threshold(trainImage,trainImage,0,255,CV_THRESH_BINARY|CV_THRESH_OTSU);
            Canny(trainImage ,trainImage ,150,100,3,false);
            for(int k = 0; k<image_rows*image_cols; ++k)
            {
                trainingData[i*sample_mun_perclass+(j-1)][k] = (float)trainImage.data[k];
                //trainingData[i*sample_mun_perclass+(j-1)][k] = (float)trainImage.at<unsigned char>((int)k/8,(int)k%8);//(float)train_image.data[k];
                //cout<<trainingData[i*sample_mun_perclass+(j-1)][k] <<" "<< (float)trainImage.at<unsigned char>(k/8,k%8)<<endl;
            }

        } while (FindNextFile(hFile,&pNextInfo) );

    }

    // Set up training data Mat
    Mat trainingDataMat(class_mun*sample_mun_perclass, image_rows*image_cols, CV_32FC1, trainingData);
    //cout<<"trainingDataMat:"<<endl;
    //cout<<trainingDataMat<<endl;
    cout<<"trainingDataMat——OK！"<<endl;

    // Set up label data  
    for(int i=0;i<=class_mun-1;++i)
    {
        for(int j=0;j<=sample_mun_perclass-1;++j)
        {
            for(int k = 0;k<class_mun;++k)
            {
                if(k==i)
                    labels[i*sample_mun_perclass + j][k] = 1;
                else labels[i*sample_mun_perclass + j][k] = 0;
            }
        }
    }

    // Set up label data 
    Mat labelsMat(class_mun*sample_mun_perclass, class_mun, CV_32FC1,labels);
    cout<<"labelsMat:"<<endl;
    cout<<labelsMat<<endl;
    cout<<"labelsMat——OK！"<<endl;

    //训练代码

    cout<<"training start...."<<endl;
    CvANN_MLP bp;

    // Set up BPNetwork's parameters
    CvANN_MLP_TrainParams params;
    params.train_method=CvANN_MLP_TrainParams::BACKPROP;
    params.bp_dw_scale=0.001;
    params.bp_moment_scale=0.1;
    params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,10000,0.0001);  //设置结束条件
    //params.train_method=CvANN_MLP_TrainParams::RPROP;
    //params.rp_dw0 = 0.1;
    //params.rp_dw_plus = 1.2;
    //params.rp_dw_minus = 0.5;
    //params.rp_dw_min = FLT_EPSILON;
    //params.rp_dw_max = 50.;

    //Setup the BPNetwork
    Mat layerSizes=(Mat_<int>(1,4) << image_rows*image_cols,int(image_rows*image_cols/2),int(image_rows*image_cols/2),class_mun);
    bp.create(layerSizes,CvANN_MLP::SIGMOID_SYM,1.0,1.0);//CvANN_MLP::SIGMOID_SYM
                                               //CvANN_MLP::GAUSSIAN
                                               //CvANN_MLP::IDENTITY
    cout<<"training...."<<endl;
    bp.train(trainingDataMat, labelsMat, Mat(),Mat(), params);

    bp.save("bpcharModel.xml"); //save classifier
    cout<<"training finish...bpModel1.xml saved "<<endl;

    //测试神经网络
    cout<<"测试："<<endl;

    Mat test_image = imread("yuner0.jpg",CV_LOAD_IMAGE_GRAYSCALE);
    Mat test_temp;
    resize(test_image,test_temp,Size(image_cols,image_rows),(0,0),(0,0),CV_INTER_AREA);//使用象素关系重采样。当图像缩小时候，该方法可以避免波纹出现
    //threshold(test_temp,test_temp,0,255,CV_THRESH_BINARY|CV_THRESH_OTSU);
    Canny(test_temp ,test_temp ,150,100,3,false);
    Mat_<float>sampleMat(1,image_rows*image_cols); 
    for(int i = 0; i<image_rows*image_cols; ++i)  
    {  
        sampleMat.at<float>(0,i) = (float)test_temp.data[i];
       // sampleMat.at<float>(0,i) = (float)test_temp.at<uchar>(i/8,i%8);  //(float)resizeImage.data[k]
    }  

    Mat responseMat;  
    bp.predict(sampleMat,responseMat); 

    float* p=responseMat.ptr<float>(0);  
    float max= -1,min =0;  
    int index = 0;  
    for(int k=0;k<class_mun;++k)  
    {  
        cout<<(float)(*(p+k))<<" ";  
        if(k==class_mun-1)  
            cout<<endl;  
        if((float)(*(p+k))>max)  
        {  
            min = max;  
            max = (float)(*(p+k));  
            index = k;  
        }  
        else  
        {  
            if(min < (float)(*(p+k)))  
                min = (float)(*(p+k));  
        }  
    }  
    //对应上美、丑
    string judge = "";
    if (index==0)
        judge = "路人";
    if (index==1)
        judge = "林允儿";
    cout<<"识别结果:"<<judge<<endl<<"识别置信度:"<<(((max-min)*100) > 100 ? 100:((max-min)*100))<<endl;

    /*Point maxLoc;
    double maxVal = 0;
    minMaxLoc(responseMat,NULL,&maxVal,NULL,&maxLoc);
    cout<<"识别结果："<<maxLoc.x<<"  置信度:"<<maxVal*100<<"%"<<endl;*/
    imshow("test_image",test_image);  
    imshow("test_temp",test_temp);  
    waitKey(0);

    return 0;
}



//读取目录src下min（number，所有图像）图像提取人脸并保存到srccut目录，
//参数：原图片目录src       剪切图片保存目录dst     读取最大数量number    
void AllImagePro( string src, string dst, static int number )
{
    int count=0;
    string src1 = src;  
    string src1cut = dst;
    HANDLE hFile;
    LPCTSTR lpFileName = StringToWchar(src1+"/"+"*.*"); //指定搜索目录和文件类型，如搜索d盘的音频文件可以是"D:\\*.mp3"
    WIN32_FIND_DATA pNextInfo;  //搜索得到的文件信息将储存在pNextInfo中;
    hFile = FindFirstFile(lpFileName,&pNextInfo);//请注意是 &pNextInfo , 不是 pNextInfo;
    if(hFile == INVALID_HANDLE_VALUE)
    {
        //搜索失败
        exit(-1);
    }
    cout<<"文件夹"<<src<<"找到的图片："<<endl;
    do
    {
        if(pNextInfo.cFileName[0] == '.')//过滤.和..
            continue;
        count++;
        printf("%s\n",WcharToChar(pNextInfo.cFileName));
        Mat img = imread( src1 + "/" + WcharToChar(pNextInfo.cFileName) , 1 );
        detectAndCut( img ,src1cut ,WcharToChar(pNextInfo.cFileName) );
    }while (FindNextFile(hFile,&pNextInfo) && count<number);//如果设置读入的图片数量，则以设置的为准，如果图片不够，则读取文件夹下所有图片
}

//人脸检测
//参数：待检测图像img       保存路径dir     保存文件名name
void detectAndCut( Mat img ,string dir, string filename)
{
   std::vector<Rect> faces;
   Mat img_gray;

   cvtColor( img, img_gray, COLOR_BGR2GRAY );
   equalizeHist( img_gray, img_gray );
   //-- Detect faces
   face_cascade.detectMultiScale( img_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );

   for( size_t i = 0; i < faces.size(); i++ )
    {
      Point rec( faces[i].x, faces[i].y );
      Point rec2( faces[i].x + faces[i].width, faces[i].y + faces[i].height );
      Mat roi_img = img( Range(faces[i].y,faces[i].y + faces[i].height), Range(faces[i].x,faces[i].x + faces[i].width) );
      imwrite( dir+"/"+filename, roi_img );   
   }
}


char* WcharToChar(const wchar_t* wp)  
{  
    char *m_char;
    int len= WideCharToMultiByte(CP_ACP,0,wp,wcslen(wp),NULL,0,NULL,NULL);  
    m_char=new char[len+1];  
    WideCharToMultiByte(CP_ACP,0,wp,wcslen(wp),m_char,len,NULL,NULL);  
    m_char[len]='\0';  
    return m_char;  
}  

wchar_t* CharToWchar(const char* c)  
{   
    wchar_t *m_wchar;
    int len = MultiByteToWideChar(CP_ACP,0,c,strlen(c),NULL,0);  
    m_wchar=new wchar_t[len+1];  
    MultiByteToWideChar(CP_ACP,0,c,strlen(c),m_wchar,len);  
    m_wchar[len]='\0';  
    return m_wchar;  
}  

wchar_t* StringToWchar(const string& s)  
{  
    const char* p=s.c_str();  
    return CharToWchar(p);  
}  

string getstring ( const int n )
{
    std::stringstream newstr;
    newstr<<n;
    return newstr.str();
}