#include<bits/stdc++.h>
#include "./lib/NN.h"

using namespace std;


struct Image{
	vector<double> image;
	int lable;
};

unsigned int reverseint(unsigned int A)
{
	return ((((uint32_t)(A) & 0xff000000) >> 24) |
                      (((uint32_t)(A) & 0x00ff0000) >> 8 ) |
                      (((uint32_t)(A) & 0x0000ff00) << 8 ) |
                      (((uint32_t)(A) & 0x000000ff) << 24));
}

unsigned char reverse(unsigned char A)
{
	return ((((A) & (unsigned char)0xff00) >> 8 ) |
                      (((A) & (unsigned char)0x00ff) << 8 ));

}

int main()
{
	const int testSize = 10000;
	Image testImageList[testSize];
	const int sizeOfTest = 60000;
	Image imageList[sizeOfTest];
	ifstream inFile("train-images.idx3-ubyte",ios::in|ios::binary);
	if(!inFile){
		cout<<"error"<<endl;
		return -1;
	}

	unsigned int n,a;

	inFile.read((char*) &a,sizeof(unsigned int));
	cout<<"magic number: "<<reverseint(a)<<endl;

	inFile.read((char*) &n,sizeof(unsigned int));
	cout<<"number of images: "<<reverseint(n)<<endl;

	inFile.read((char*) &a,sizeof(unsigned int));
	cout<<"number of rows: "<<reverseint(a)<<endl;
	inFile.read((char*) &a,sizeof(unsigned int));
	cout<<"number of colums"<<reverseint(a)<<endl;


	for(int k = 0; k < sizeOfTest;k++)
	{
		for(int i = 0 ;i < 28;i++)
		{
			for(int j = 0;j < 28;j++)
			{
				unsigned char temp;
				inFile.read((char*) &temp,sizeof(unsigned char));
				//temp = reverse(temp);
				imageList[k].image.push_back( (double)temp/255);
				//cout<<(int) temp<<" ";
			}
			//cout<<endl;
		}
	}

	ifstream inFileLable("train-labels.idx1-ubyte",ios::in|ios::binary);
	if(!inFileLable){
		cout<<"error"<<endl;
		return -1;
	}


	inFileLable.read((char*) &a,sizeof(unsigned int));
	cout<<"magic number: "<<reverseint(a)<<endl;

	inFileLable.read((char*) &n,sizeof(unsigned int));
	cout<<"number of images: "<<reverseint(n)<<endl;

	for(int k = 0; k < sizeOfTest;k++)
	{

		unsigned char temp;
		inFileLable.read((char*) &temp,sizeof(unsigned char));
		imageList[k].lable = temp;
		//cout<<"Lable: "<<(int) temp<<endl;

	}


	srand(time(0)*3049);

	NN::NNcore nn;
	nn.init(vector<int> {28*28,128,68,10},0.01);

	int correctnum=0,wrongnum=0;
	for(int j = 0; j< 10;j++)
	{
		correctnum=0,wrongnum=0;
		for(int i = 0; i< sizeOfTest;i++)
		{
			nn.forward(imageList[i].image,false);

			vector<double> answer = vector<double>(10) ;
			answer[imageList[i].lable] = 1;

			nn.pushBack(answer);

			if(nn.choice()==imageList[i].lable)
			{
				correctnum++;
			}
			else
			{
				wrongnum++;
			}
			if(i%500 == 0)
				cout<<"Term: "<<j<<" Train Time"<<i<<"  "<<correctnum/(double)(correctnum+wrongnum)<<endl;

	}

	}

	ifstream testFileImage("t10k-images.idx3-ubyte",ios::in|ios::binary);
	if(!inFile){
		cout<<"error"<<endl;
		return -1;
	}

	testFileImage.read((char*) &a,sizeof(unsigned int));
	cout<<"magic number: "<<reverseint(a)<<endl;

	testFileImage.read((char*) &n,sizeof(unsigned int));
	cout<<"number of images: "<<reverseint(n)<<endl;

	testFileImage.read((char*) &a,sizeof(unsigned int));
	cout<<"number of rows: "<<reverseint(a)<<endl;
	testFileImage.read((char*) &a,sizeof(unsigned int));
	cout<<"number of colums: "<<reverseint(a)<<endl;


	for(int k = 0; k < testSize;k++)
	{
		for(int i = 0 ;i < 28;i++)
		{
			for(int j = 0;j < 28;j++)
			{
				unsigned char temp;
				testFileImage.read((char*) &temp,sizeof(unsigned char));
				testImageList[k].image.push_back( (double)temp/255);
			}
		}
	}

	ifstream TestFileLable("t10k-labels.idx1-ubyte",ios::in|ios::binary);
	if(!inFileLable){
		cout<<"error"<<endl;
		return -1;
	}


	TestFileLable.read((char*) &a,sizeof(unsigned int));
	cout<<"magic number: "<<reverseint(a)<<endl;

	TestFileLable.read((char*) &n,sizeof(unsigned int));
	cout<<"number of images: "<<reverseint(n)<<endl;

	for(int k = 0; k < testSize;k++)
	{

		unsigned char temp;
		TestFileLable.read((char*) &temp,sizeof(unsigned char));
		testImageList[k].lable = temp;
		//cout<<"Lable: "<<(int) temp<<endl;

	}

	cout<<"\n\n\nstartTest:"<<endl<<endl<<endl;

	correctnum=0,wrongnum=0;
	for(int i = 0; i< testSize;i++)
	{

		//cout<<"output: ";

		nn.forward(testImageList[i].image,false);

		if(nn.choice()==testImageList[i].lable)
		{
			correctnum++;
		}
		else
		{
			wrongnum++;
		}
		if(i%1000==0)
			cout<<correctnum/(double)(correctnum+wrongnum)<<endl;

	}
	return 0;

}

