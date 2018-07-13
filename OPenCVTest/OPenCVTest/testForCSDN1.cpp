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
		sprintf(filename, "C:\\Users\\lenovo\\Desktop\\����ͼƬ\\%d.jpg", i);
		// ����ͼƬ
		Mat Img = imread(filename, 0);
		if (!Img.data)
		{
			break;
		}
		
		namedWindow("��ʾͼƬ"+i, 0);
		imshow("��ʾͼƬ"+i, Img);


		// �����ڴ˶���Img��ͼ���������������
		
        // �ȴ�ʱ��Ϊ1s
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
	image1 = imread("C:\\Users\\lenovo\\Desktop\\����ͼƬ\\ģ��.jpg");
	int flag=0;
	for (int i = 1; i < 13; i++){
		sprintf(filename, "C:\\Users\\lenovo\\Desktop\\����ͼƬ\\%d.jpg", i);
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
//	imshow("test"+i,test);         //������ʾ������ͼƬ
	sprintf(savename, "C:\\Users\\lenovo\\Desktop\\����ͼƬ\\NEW\\%d.jpg", flag);
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
   //��ȡÿ�����ļ���������ͼ��
		int j = 0;//ÿһ���ȡͼ���������
		fileReadPath = perfileReadPath + "/" + temp + "/" + fileform;

		HANDLE hFile;
		LPCTSTR lpFileName = StringToWchar(fileReadPath);//ָ������Ŀ¼���ļ����ͣ�������d�̵���Ƶ�ļ�������"D:\\*.mp3"
		WIN32_FIND_DATA pNextInfo;  //�����õ����ļ���Ϣ��������pNextInfo��;
		hFile = FindFirstFile(lpFileName,&pNextInfo);//��ע���� &pNextInfo , ���� pNextInfo;
		if(hFile == INVALID_HANDLE_VALUE)
		{
			exit(-1);//����ʧ��
		}
		//do-whileѭ����ȡ
		do
		{	
			if(pNextInfo.cFileName[0] == '.')//����.��..
				continue;
			j++;//��ȡһ��ͼ
			//wcout<<pNextInfo.cFileName<<endl;
			printf("%s\n",WcharToChar(pNextInfo.cFileName));
			//�Զ����ͼƬ���д���
			Mat srcImage = imread( perfileReadPath + "/" + temp + "/" + WcharToChar(pNextInfo.cFileName),CV_LOAD_IMAGE_GRAYSCALE);
			int xRoi=200;
			int yRoi=140;
			int widthRoi=80;
			int heightRoi=60;
			Mat roiImage(srcImage.rows,srcImage.cols,CV_8UC3);
			cout<<srcImage.rows<<" "<<srcImage.cols<<endl;
			srcImage(Rect(xRoi,yRoi,widthRoi,heightRoi)).copyTo(roiImage);
			imwrite(perfileReadPath1 + "/" + temp + "/" + WcharToChar(pNextInfo.cFileName),roiImage);
		} while (FindNextFile(hFile,&pNextInfo) && j<74);//������ö����ͼƬ�������������õ�Ϊ׼�����ͼƬ���������ȡ�ļ���������ͼƬ
        
    
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

    const int sample_mun_perclass = 20;//ѵ���ַ�ÿ������
    const int class_mun = 2;//ѵ���ַ�����

    const int image_cols = 30;
    const int image_rows = 30;
    string  fileReadName,
            fileReadPath;
    char temp[256];

    float trainingData[class_mun*sample_mun_perclass][image_rows*image_cols] = {{0}};//ÿһ��һ��ѵ������
    float labels[class_mun*sample_mun_perclass][class_mun]={{0}};//ѵ��������ǩ

    for(int i=0;i<=class_mun-1;++i)//��ͬ��
    {
		//��ȡÿ�����ļ���������ͼ��
		int j = 0;//ÿһ���ȡͼ���������
        sprintf(temp, "%d", i);
		fileReadPath = perfileReadPath + "/" +temp+"/" + fileform;
		cout<<"�ļ���"<<i<<endl;
		HANDLE hFile;
		LPCTSTR lpFileName = StringToWchar(fileReadPath);//ָ������Ŀ¼���ļ����ͣ�������d�̵���Ƶ�ļ�������"D:\\*.mp3"
		WIN32_FIND_DATA pNextInfo;  //�����õ����ļ���Ϣ��������pNextInfo��;
		hFile = FindFirstFile(lpFileName,&pNextInfo);//��ע���� &pNextInfo , ���� pNextInfo;
		if(hFile == INVALID_HANDLE_VALUE)
		{
			exit(-1);//����ʧ��
		}
		//do-whileѭ����ȡ
		do
		{	
			if(pNextInfo.cFileName[0] == '.')//����.��..
				continue;
			j++;//��ȡһ��ͼ
			//wcout<<pNextInfo.cFileName<<endl;
			printf("%s\n",WcharToChar(pNextInfo.cFileName));
			//�Զ����ͼƬ���д���
			Mat srcImage = imread( perfileReadPath + "/" + temp + "/" + WcharToChar(pNextInfo.cFileName),CV_LOAD_IMAGE_GRAYSCALE);
			Mat resizeImage;
			Mat trainImage;
			Mat result;

			resize(srcImage,resizeImage,Size(image_cols,image_rows),(0,0),(0,0),CV_INTER_AREA);//ʹ�����ع�ϵ�ز�������ͼ����Сʱ�򣬸÷������Ա��Ⲩ�Ƴ���
			//threshold(resizeImage,trainImage,0,255,CV_THRESH_BINARY|CV_THRESH_OTSU);

			for(int k = 0; k<image_rows*image_cols; ++k)
            {
				trainingData[i*sample_mun_perclass+(j-1)][k] = (float)resizeImage.data[k];
                //trainingData[i*sample_mun_perclass+(j-1)][k] = (float)trainImage.at<unsigned char>((int)k/8,(int)k%8);//(float)train_image.data[k];
                //cout<<trainingData[i*sample_mun_perclass+(j-1)][k] <<" "<< (float)trainImage.at<unsigned char>(k/8,k%8)<<endl;
            }

		} while (FindNextFile(hFile,&pNextInfo) && j<sample_mun_perclass);//������ö����ͼƬ�������������õ�Ϊ׼�����ͼƬ���������ȡ�ļ���������ͼƬ
		
    }

	// Set up training data Mat
	Mat trainingDataMat(class_mun*sample_mun_perclass, image_rows*image_cols, CV_32FC1, trainingData);
	cout<<"trainingDataMat����OK��"<<endl;

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
	cout<<"labelsMat����OK��"<<endl;

	//ѵ������

	cout<<"training start...."<<endl;
    CvANN_MLP bp;
    // Set up BPNetwork's parameters
    CvANN_MLP_TrainParams params;
    params.train_method=CvANN_MLP_TrainParams::BACKPROP;
    params.bp_dw_scale=0.001;
    params.bp_moment_scale=0.1;
    params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,10000,0.0001);  //���ý�������
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


	//����������
	cout<<"���ԣ�"<<endl;
	Mat test_image = imread("REC0009_148.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	Mat test_temp;
	resize(test_image,test_temp,Size(image_cols,image_rows),(0,0),(0,0),CV_INTER_AREA);//ʹ�����ع�ϵ�ز�������ͼ����Сʱ�򣬸÷������Ա��Ⲩ�Ƴ���
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
	cout<<"ʶ������"<<maxLoc.x<<"	���ƶ�:"<<maxVal*100<<"%"<<endl;
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
	//��̨ͨʽ��3ͨ��BGR,�ƶ��豸Ϊ4ͨ��
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
	//����Haar��LBP��������������
	CascadeClassifier faceDetector;
	std::string faceCascadeFilename = "haarcascade_frontalface_default.xml";

	//�Ѻô�����Ϣ��ʾ
	try{
		faceDetector.load(faceCascadeFilename);
	}
	catch (cv::Exception e){}
	if (faceDetector.empty())
	{
		std::cerr << "������������ܼ��� (";
		std::cerr << faceCascadeFilename << ")!" << std::endl;
		exit(1);
	}

	//������ͷ
	VideoCapture camera(0);
	while (true)
	{
		Mat camerFrame;
	//	camerFrame=imread("ͼƬ������.png");
		camera >> camerFrame;
		if (camerFrame.empty())
		{
			std::cerr << "�޷���ȡ����ͷͼ��" << std::endl;
			getchar();
			exit(1);
		}
		Mat displayedFrame(camerFrame.size(),CV_8UC3);


		//�������ֻ�����ڻҶ�ͼ��
		Mat gray;
		Pic2Gray(camerFrame, gray);



		//ֱ��ͼ���Ȼ�(����ͼ��ĶԱȶȺ�����)
		Mat equalizedImg;
		equalizeHist(gray, equalizedImg);

		//���������Cascade Classifier::detectMultiScale�������������

		//int flags = CASCADE_FIND_BIGGEST_OBJECT|CASCADE_DO_ROUGH_SEARCH;	//ֻ�����������
		int flags = CASCADE_SCALE_IMAGE;	//�������
		Size minFeatureSize(30, 30);
		float searchScaleFactor = 1.1f;
		int minNeighbors = 4;
		std::vector<Rect> faces;
		faceDetector.detectMultiScale(equalizedImg, faces, searchScaleFactor, minNeighbors, flags, minFeatureSize);

		//�����ο�
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


		imshow("ֱ��ͼ���Ȼ�", equalizedImg);
		imshow("�ҶȻ�", gray);
		imshow("ԭͼ", camerFrame);

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
    //���ļ���0ͼƬ��Ѱ����������ȡ��ŵ��ļ���0cut��
    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
    AllImagePro( "student", "0cut" ,20);
    cout<<"cut����OK��"<<endl;

    return 0;
}

//��ȡĿ¼src��min��number������ͼ��ͼ����ȡ���������浽srccutĿ¼��
//������ԭͼƬĿ¼src       ����ͼƬ����Ŀ¼dst     ��ȡ�������number    
void AllImagePro( string src, string dst, static int number )
{
    int count=0;
    string src1 = src;  
    string src1cut = dst;
    HANDLE hFile;
    LPCTSTR lpFileName = StringToWchar(src1+"/"+"*.*"); //ָ������Ŀ¼���ļ����ͣ�������d�̵���Ƶ�ļ�������"D:\\*.mp3"
    WIN32_FIND_DATA pNextInfo;  //�����õ����ļ���Ϣ��������pNextInfo��;
    hFile = FindFirstFile(lpFileName,&pNextInfo);//��ע���� &pNextInfo , ���� pNextInfo;
    if(hFile == INVALID_HANDLE_VALUE)
    {
        //����ʧ��
        exit(-1);
    }
    cout<<"�ļ���"<<src<<"�ҵ���ͼƬ��"<<endl;
    do
    {
        if(pNextInfo.cFileName[0] == '.')//����.��..
            continue;
        count++;
        printf("%s\n",WcharToChar(pNextInfo.cFileName));
        Mat img = imread( src1 + "/" + WcharToChar(pNextInfo.cFileName) , 1 );
        detectAndCut( img ,src1cut ,WcharToChar(pNextInfo.cFileName) );
    }while (FindNextFile(hFile,&pNextInfo) && count<number);//������ö����ͼƬ�������������õ�Ϊ׼�����ͼƬ���������ȡ�ļ���������ͼƬ
}

//�������
//�����������ͼ��img       ����·��dir     �����ļ���name
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

//������
int main()
{
    const int sample_mun_perclass = 12;//ѵ��ÿ��ͼƬ����
    const int class_mun = 2;//ѵ������ һ������Ů��һ���ǳ�Ů ^-^

    const int image_cols = 30;
    const int image_rows = 30;

    string  fileReadName,fileReadPath;

    float trainingData[class_mun*sample_mun_perclass][image_rows*image_cols] = {{0}};//ÿһ��һ��ѵ������
    float labels[class_mun*sample_mun_perclass][class_mun]={{0}};//ѵ��������ǩ

    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
    AllImagePro( "0", "0cut" ,sample_mun_perclass);
    AllImagePro( "1", "1cut" ,sample_mun_perclass);
    cout<<"cut����OK��"<<endl;

     for(int i=0;i<class_mun;++i)//��ͬ��
    {
        //��ȡÿ�����ļ���������ͼ��
        int j = 0;//ÿһ���ȡͼ���������
        fileReadPath = getstring(i) + "cut/" + "*.*";
        cout<<"�ļ���"<<i<<endl;
        HANDLE hFile;
        LPCTSTR lpFileName = StringToWchar(fileReadPath);//ָ������Ŀ¼���ļ����ͣ�������d�̵���Ƶ�ļ�������"D:\\*.mp3"
        WIN32_FIND_DATA pNextInfo;  //�����õ����ļ���Ϣ��������pNextInfo��;
        hFile = FindFirstFile(lpFileName,&pNextInfo);//��ע���� &pNextInfo , ���� pNextInfo;
        if(hFile == INVALID_HANDLE_VALUE)
        {
            exit(-1);//����ʧ��
        }
        //do-whileѭ����ȡ
        do
        {
            if(pNextInfo.cFileName[0] == '.')//����.��..
                continue;
            //wcout<<pNextInfo.cFileName<<endl;
            j++;
            printf("%s\n",WcharToChar(pNextInfo.cFileName));
            //�Զ����ͼƬ���д���
            Mat srcImage = imread( getstring(i) + "/" + WcharToChar(pNextInfo.cFileName),CV_LOAD_IMAGE_GRAYSCALE);
            Mat trainImage;

            resize(srcImage,trainImage,Size(image_cols,image_rows),(0,0),(0,0),CV_INTER_AREA);//ʹ�����ع�ϵ�ز�������ͼ����Сʱ�򣬸÷������Ա��Ⲩ�Ƴ���
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
    cout<<"trainingDataMat����OK��"<<endl;

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
    cout<<"labelsMat����OK��"<<endl;

    //ѵ������

    cout<<"training start...."<<endl;
    CvANN_MLP bp;

    // Set up BPNetwork's parameters
    CvANN_MLP_TrainParams params;
    params.train_method=CvANN_MLP_TrainParams::BACKPROP;
    params.bp_dw_scale=0.001;
    params.bp_moment_scale=0.1;
    params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,10000,0.0001);  //���ý�������
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

    //����������
    cout<<"���ԣ�"<<endl;

    Mat test_image = imread("yuner0.jpg",CV_LOAD_IMAGE_GRAYSCALE);
    Mat test_temp;
    resize(test_image,test_temp,Size(image_cols,image_rows),(0,0),(0,0),CV_INTER_AREA);//ʹ�����ع�ϵ�ز�������ͼ����Сʱ�򣬸÷������Ա��Ⲩ�Ƴ���
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
    //��Ӧ��������
    string judge = "";
    if (index==0)
        judge = "·��";
    if (index==1)
        judge = "���ʶ�";
    cout<<"ʶ����:"<<judge<<endl<<"ʶ�����Ŷ�:"<<(((max-min)*100) > 100 ? 100:((max-min)*100))<<endl;

    /*Point maxLoc;
    double maxVal = 0;
    minMaxLoc(responseMat,NULL,&maxVal,NULL,&maxLoc);
    cout<<"ʶ������"<<maxLoc.x<<"  ���Ŷ�:"<<maxVal*100<<"%"<<endl;*/
    imshow("test_image",test_image);  
    imshow("test_temp",test_temp);  
    waitKey(0);

    return 0;
}



//��ȡĿ¼src��min��number������ͼ��ͼ����ȡ���������浽srccutĿ¼��
//������ԭͼƬĿ¼src       ����ͼƬ����Ŀ¼dst     ��ȡ�������number    
void AllImagePro( string src, string dst, static int number )
{
    int count=0;
    string src1 = src;  
    string src1cut = dst;
    HANDLE hFile;
    LPCTSTR lpFileName = StringToWchar(src1+"/"+"*.*"); //ָ������Ŀ¼���ļ����ͣ�������d�̵���Ƶ�ļ�������"D:\\*.mp3"
    WIN32_FIND_DATA pNextInfo;  //�����õ����ļ���Ϣ��������pNextInfo��;
    hFile = FindFirstFile(lpFileName,&pNextInfo);//��ע���� &pNextInfo , ���� pNextInfo;
    if(hFile == INVALID_HANDLE_VALUE)
    {
        //����ʧ��
        exit(-1);
    }
    cout<<"�ļ���"<<src<<"�ҵ���ͼƬ��"<<endl;
    do
    {
        if(pNextInfo.cFileName[0] == '.')//����.��..
            continue;
        count++;
        printf("%s\n",WcharToChar(pNextInfo.cFileName));
        Mat img = imread( src1 + "/" + WcharToChar(pNextInfo.cFileName) , 1 );
        detectAndCut( img ,src1cut ,WcharToChar(pNextInfo.cFileName) );
    }while (FindNextFile(hFile,&pNextInfo) && count<number);//������ö����ͼƬ�������������õ�Ϊ׼�����ͼƬ���������ȡ�ļ���������ͼƬ
}

//�������
//�����������ͼ��img       ����·��dir     �����ļ���name
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