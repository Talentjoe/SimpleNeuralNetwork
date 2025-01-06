#include<bits/stdc++.h>

using namespace std;

class NN{
	int							size;
	double						studyRate;

	vector<vector<double>> 		layers;
	vector<vector<double>> 		layersZ;
	vector<int> 				layerSize;

	vector<vector<double>>			b;

	vector<vector<vector<double>>>	w;

	public:

	NN()
	{
		size = 0;
		studyRate = 0;
	}

	double getRandomDoubleNumber(double max = 1,double min = -1)
	{
		return min + (double)(rand()) / ((double)(RAND_MAX / (max - min)));
	}

	double sigmoidP(double x)
	{
		return sigmoid(x)*(1-sigmoid(x));
	}

	double sigmoid(double x)
	{
		return 1/(1+exp(-x));
	}

	vector<double> forward(vector<double> inNums,bool printRes = false)
	{
		if(inNums.size()!=layerSize[0])
		{
			cout<<"Size Not Mathch !! "<<endl;
			return {};
		}
		layers[0] = inNums;

		for(auto& row : layersZ) // 将内容归零
		{
			fill(row.begin(), row.end(), 0.0);
		}

		for(int i = 1; i < size;i++)
		{
			for(int k = 0;k < layerSize[i];k++)
			{
				for(int j = 0; j < layerSize[i-1];j++)
					layersZ[i][k] += layers[i-1][j]*w[i-1][j][k];
				layersZ[i][k] += b[i][k];
				layers[i][k] = sigmoid(layersZ[i][k]);
			}
		}
		if(printRes)
		{
			for(auto i : layers[size-1]) cout<<setw(10)<<i;
			cout<<endl;
		}
		return layers[size-1];
	}


	double pushBack(vector<double> correctOut)
	{
		vector<vector<double> > delta(size);
		for(int i = 1; i < size; i++)
		{
			delta[i].resize(layerSize[i]);
		}

		for(int i = 0;i < layerSize[size-1];i++)
		{
			delta[size-1][i] = (layers[size-1][i]-correctOut[i])*sigmoidP(layersZ[size-1][i]); //BP1
		}
		for(int i = size-2; i > 0; i--)
		{
			for(int j = 0; j< layerSize[i];j++)
			{
				double value = 0;
				for(int k = 0; k< layerSize[i+1]; k++)
				{
					value += w[i][j][k]*delta[i+1][k];
				}
				delta[i][j] = sigmoidP(layersZ[i][j])*value;// BP2
			}
		}
		for(int i = 1; i < size; i++)
		{
			for(int j = 0;j < layerSize[i];j++)
			{
				b[i][j] -= delta[i][j]*studyRate;

				for(int k = 0 ;k < layerSize[i-1];k++)
				{
					//cout<<layers[i-1][k]*delta[i][j]*studyRate;
					w[i-1][k][j] -= layers[i-1][k]*delta[i][j]*studyRate;
				}
			}

		}

		double pre = CalCost(correctOut);
		forward(layers[0]);
		double after = CalCost(correctOut);

		return after-pre;
	}
	/*
	double CalCost(vector<double> correctOut)
	{
		double cost=0;
		if(correctOut.size()!=layerSize[Size-1])
		{
			cout<<"Size Error"<<endl;
			return 0;
		}
		for(int i = 0;i < layerSize[Size-1];i++)
		{
			cost += (layers[Size-1][i]-correctOut[i])*(layers[Size-1][i]-correctOut[i])/2.0;
		}
		cost/=layerSize[Size-1];
		return cost;
	}*/

	double CalCost(vector<double> correctOut)
	{
		double cost=0;
		if(correctOut.size()!=layerSize[size-1])
		{
			cout<<"Size Error"<<endl;
			return 0;
		}
		for(int i = 0;i < layerSize[size-1];i++)
		{
			cost += (correctOut[i]*log( layers[size-1][i])+(1-correctOut[i])*log(1-layers[size-1][i]))/2.0;
		}
		cost/=-layerSize[size-1];
		return cost;
	}

	void printLayers()
	{
		for(int i = 0; i<size;i++)
		{
			cout<<"Layer "<<setw(2)<<i<<": ";
			for(int j = 0;j<layerSize[i];j++)
				cout<<setw(10)<<layers[i][j]<<" ";

			cout<<endl<<endl;
		}
	}

	void printW(int layerNumberToPrint)
	{
		cout<<"W from layer "<<layerNumberToPrint<<" to "<<layerNumberToPrint+1<<": "<<endl;
		cout<<"From↓   To-> ";
		for(int i= 0;i<layerSize[layerNumberToPrint+1];i++)
		{
			cout<<setw(10)<<i<<" ";
		}
		cout<<endl<<endl;

		for(int i = 0; i<layerSize[layerNumberToPrint];i++)
		{
			cout<<"From "<<setw(2)<<i<<" note:";
			for(int j = 0;j<layerSize[layerNumberToPrint+1];j++)
				cout<<setw(10)<<w[layerNumberToPrint][i][j]<<" ";

			cout<<endl<<endl;
		}
	}

	int choice()
	{
		double max = 0;
		int res;
		for(int i = 0; i<layerSize[size-1];i++)
		{
			if(layers[size-1][i]>max)
			{
				max = layers[size-1][i];
				res = i;
			}
		}
		return res;
	}

	void init(vector<int> LayerS,double studyR)
	{
		size = LayerS.size();
		layerSize = LayerS;
		layers.resize(size);
		layersZ.resize(size);
		b.resize(size);
		w.resize(size-1);

		studyRate = studyR;

		for(int i = 0; i<size;i++)
		{
			layers[i].resize(layerSize[i]);

			if(i!=0)
			{
				layersZ[i].resize(layerSize[i]);
				b[i].resize(layerSize[i]);
			}

			if(i<size-1)
			{
				w[i].resize(layerSize[i]);
				for(int j = 0;j < layerSize[i];j++)
				{
					w[i][j].resize(layerSize[i+1]);
				}
			}
		}

		cout<<"RESIZED"<<endl;

		for(int i = 0; i<size;i++)
		{
			for(int j = 0; j < layerSize[i];j++)
			{
				layers[i][j] = getRandomDoubleNumber();
				if(i!=0)
				{
					layersZ[i][j] = getRandomDoubleNumber();
				 	b[i][j] = getRandomDoubleNumber();
				}
				if(i<size-1)
				{
					for(int k = 0;k < layerSize[i+1];k++)
					{
						w[i][j][k] = getRandomDoubleNumber();
					}
				}
			}
		}
	}
	void changeStudyRate(double rate)
	{
		studyRate = rate;
	}
};

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

	NN nn;
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

