/*####################################################################################################################*/
//
//	Name:	Rahul Maheshwari
//	Source.cpp :  This assignment implements "Bag of Words" Object Recognition using some OpenCV functions.
//
//  Reference: http://stackoverflow.com/questions/5056645/sorting-stdmap-using-value
//
/*####################################################################################################################*/


// include necessary header files
#include"stdafx.h"
#include <iostream>
#include <string>
#include <sstream>
#include <cv.h>
#include <highgui.h>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/nonfree/features2d.hpp>

using namespace cv;
using namespace std;

// This function alter the key, value in HashMap. Required for sorting using value.
template<typename A, typename B>
std::pair<B,A> flip_pair(const std::pair<A,B> &p)
{
    return std::pair<B,A>(p.second, p.first);
}

// This function alter the key, value in HashMap. Required for sorting using value.
template<typename A, typename B>
std::map<B,A> flip_map(const std::map<A,B> &src)
{
    std::map<B,A> dst;
    std::transform(src.begin(), src.end(), std::inserter(dst, dst.begin()), 
                   flip_pair<A,B>);
    return dst;
}

// This function returns label based on the weight of 5 class
int returnLabel(double count1, double count2, double count3, double count4, double count5)
{
	double temp1 = (count1>count2) ? count1 : count2;
	double temp2 = (temp1>count3) ? temp1 : count3;
	double temp3 = (temp2>count4) ? temp2 : count4;
	double max = (temp3>count5) ? temp3 : count5;

	if(max==count1)
		return 1;
	else if(max==count2)
		return 2;
	else if(max==count3)
		return 3;
	else if(max==count4)
		return 4;
	else if(max==count5)
		return 5;

	return 0;
}

// Main function
int main( int argc, char** argv )
{
	//----------------------------------------------------------Training Start----------------------------------------------------------------------//
	cout << "---------------Training Starts------------------" << endl << endl;
	// Read training images-------------------------------------------------------------
	cout << "Reading training images..." << endl;
	vector<Mat> training_images;  // to store all the training images
	Mat image;

	for(unsigned int i=1; i<21; i++)
	{
		stringstream filename; 
		filename << "training/car/train_car_" << i << ".jpg";
		image = imread(filename.str(), 1);
		if(!image.data)    // check whether reading is successful or not
		{
			cout << "No image data in file: " << filename.str() << endl;
			return -1;
		}
		training_images.push_back(image);
	}
	for(unsigned int i=1; i<21; i++)
	{
		stringstream filename; 
		filename << "training/face/train_face_" << i << ".jpg";
		image = imread(filename.str(), 1);
		if(!image.data)    // check whether reading is successful or not
		{
			cout << "No image data in file: " << filename.str() << endl;
			return -1;
		}
		training_images.push_back(image);
	}
	for(unsigned int i=1; i<21; i++)
	{
		stringstream filename; 
		filename << "training/laptop/train_laptop_" << i << ".jpg";
		image = imread(filename.str(), 1);
		if(!image.data)    // check whether reading is successful or not
		{
			cout << "No image data in file: " << filename.str() << endl;
			return -1;
		}
		training_images.push_back(image);
	}
	for(unsigned int i=1; i<21; i++)
	{
		stringstream filename; 
		filename << "training/motorbike/train_motorbike_" << i << ".jpg";
		image = imread(filename.str(), 1);
		if(!image.data)    // check whether reading is successful or not
		{
			cout << "No image data in file: " << filename.str() << endl;
			return -1;
		}
		training_images.push_back(image);
	}
	for(unsigned int i=1; i<21; i++)
	{
		stringstream filename; 
		filename << "training/pigeon/train_pigeon_" << i << ".jpg";
		image = imread(filename.str(), 1);
		if(!image.data)    // check whether reading is successful or not
		{
			cout << "No image data in file: " << filename.str() << endl;
			return -1;
		}
		training_images.push_back(image);
	}
	cout << endl;
	// ---------------------------------------------------------------------------------------
	
	// Compute SIFT feature for each training images------------------------------------------
	cout << "Computing SIFT features of each training images..." << endl;
	vector<Mat> training_SIFT_descriptors;
	vector<KeyPoint> keypoints;
	Mat descriptors;
	Mat all_training_SIFT_descriptors; // to store the SIFT features of all training images in one matrix

	for(unsigned int i=0; i<training_images.size(); i++)
	{
		SIFT siftobject;
		siftobject.operator()(training_images[i], Mat(), keypoints, descriptors);
		training_SIFT_descriptors.push_back(descriptors);

		all_training_SIFT_descriptors.push_back(descriptors);
	}
	cout << "training data size is: " << all_training_SIFT_descriptors.rows  << "*" << all_training_SIFT_descriptors.cols << endl;
	cout << endl;
	// ----------------------------------------------------------------------------------------

	// Compute covariance and eigenvectors for all the images----------------------------------
	cout << "Computing covariance and eigenvalues..." << endl;
	Mat covar;
	Mat mean;
	calcCovarMatrix(all_training_SIFT_descriptors, covar, mean, CV_COVAR_NORMAL | CV_COVAR_ROWS, CV_32F);
	cout << "Covariance matrix size is: " << covar.rows  << "*" << covar.cols << endl;

	Mat eigenvalues;
	Mat eigenvectors;
	eigen(covar, eigenvalues, eigenvectors);
	cout << "EigenValue matrix size is: " << eigenvalues.rows  << "*" << eigenvalues.cols << endl;
	cout << "EigenVector matrix size is: " << eigenvectors.rows  << "*" << eigenvectors.cols << endl;

	int pcaReduced = 20;  // set the dimension reduction value
	Mat reduced_eigenvectors(pcaReduced, eigenvectors.cols, CV_32F, 0.0);
	for(int i=0; i<pcaReduced; i++)
	{
		for(int j=0; j<eigenvectors.cols; j++)
		{
			reduced_eigenvectors.at<float>(i, j) = eigenvectors.at<float>(i, j);
		}
	}
	reduced_eigenvectors = reduced_eigenvectors.t();
	cout << endl;
	// ----------------------------------------------------------------------------------------
 
	// Computing PCA SIFT feature for each training images-------------------------------------
	cout << "Computing PCA SIFT features of training images..." << endl;
	vector<Mat> training_PCA_SIFT_features;
	Mat all_training_PCA_SIFT_features; // for k-means clustering we require it in one matrix
	Mat projection;
	for(unsigned int i=0; i<training_images.size(); i++)
	{
		projection = training_SIFT_descriptors[i]*reduced_eigenvectors;
		training_PCA_SIFT_features.push_back(projection);

		all_training_PCA_SIFT_features.push_back(projection);
	}
	cout << "training data size after reducing features is: " << all_training_PCA_SIFT_features.rows  << "*" << all_training_PCA_SIFT_features.cols << endl;
	cout << endl;
	// ----------------------------------------------------------------------------------------
	
	// Compute Codewords, Cluster the PCA SIFT features-----------------------------------------
	cout << "Computing Code words, Vocabulary of training images by clustering PCA SIFT features..." << endl;
	int nclusters = 100;  // set the number of clusters you want
	Mat labels;
	int attempts = 20;
	Mat centers;
	TermCriteria criteria;
    criteria.epsilon = 1e-4;
    criteria.maxCount = 1;

	kmeans(all_training_PCA_SIFT_features, nclusters, labels, criteria, attempts, KMEANS_RANDOM_CENTERS, centers);
	cout << "training centers size: " << centers.rows  << "*" << centers.cols << endl;
	cout << endl;
	// -----------------------------------------------------------------------------------------
	
	// Calculating Histogram for images---------------------------------------------------------
	cout << "Calculating histogram for each images..." << endl;
	vector< vector<double> > training_histogram;
	for(unsigned int i=0; i<training_images.size(); i++)
	{
		training_histogram.push_back(vector< double >(nclusters, 0.0));
	}

	int index = 0;
	for(int i=0; i<training_images.size(); i++)
	{
		for(int j=0; j<training_SIFT_descriptors[i].rows; j++)
		{
			training_histogram[i][labels.at<uchar>(index, 0)]++;
			index++;
		}
		// Normalize histogram if requires, uncomment below
		/*for(int j=0; j<nclusters; j++)
		{
			training_histogram[i][j] = (training_histogram[i][j]*100)/training_SIFT_descriptors[i].rows;
		}*/
	}
	cout << endl;
	// -----------------------------------------------------------------------------
	//----------------------------------------------------------Training Done----------------------------------------------------------------------//

	cout << "---------------Testing Starts------------------" << endl << endl;

	//----------------------------------------------------------Testing Start----------------------------------------------------------------------//
	// Read test images-------------------------------------------------------------
	cout << "Reading test images..." << endl;
	vector<Mat> testing_images;

	for(unsigned int i=1; i<11; i++)
	{
		stringstream filename; 
		filename << "testing/car/test_car_" << i << ".jpg";
		image = imread(filename.str(), 1);
		if(!image.data)    // check whether reading is successful or not
		{
			cout << "No image data in file: " << filename.str() << endl;
			return -1;
		}
		testing_images.push_back(image);
	}
	for(unsigned int i=1; i<11; i++)
	{
		stringstream filename; 
		filename << "testing/face/test_face_" << i << ".jpg";
		image = imread(filename.str(), 1);
		if(!image.data)    // check whether reading is successful or not
		{
			cout << "No image data in file: " << filename.str() << endl;
			return -1;
		}
		testing_images.push_back(image);
	}
	for(unsigned int i=1; i<11; i++)
	{
		stringstream filename; 
		filename << "testing/laptop/test_laptop_" << i << ".jpg";
		image = imread(filename.str(), 1);
		if(!image.data)    // check whether reading is successful or not
		{
			cout << "No image data in file: " << filename.str() << endl;
			return -1;
		}
		testing_images.push_back(image);
	}
	for(unsigned int i=1; i<11; i++)
	{
		stringstream filename; 
		filename << "testing/motorbike/test_motorbike_" << i << ".jpg";
		image = imread(filename.str(), 1);
		if(!image.data)    // check whether reading is successful or not
		{
			cout << "No image data in file: " << filename.str() << endl;
			return -1;
		}
		testing_images.push_back(image);
	}
	for(unsigned int i=1; i<11; i++)
	{
		stringstream filename; 
		filename << "testing/pigeon/test_pigeon_" << i << ".jpg";
		image = imread(filename.str(), 1);
		if(!image.data)    // check whether reading is successful or not
		{
			cout << "No image data in file: " << filename.str() << endl;
			return -1;
		}
		testing_images.push_back(image);
	}
	cout << endl;
	// ---------------------------------------------------------------------------------------
	
	// Compute SIFT feature for each test images----------------------------------------------
	cout << "Computing SIFT features of each testing images..." << endl;
	vector<Mat> testing_SIFT_descriptors;

	for(unsigned int i=0; i<testing_images.size(); i++)
	{
		SIFT siftobject;
		siftobject.operator()(testing_images[i], Mat(), keypoints, descriptors);
		testing_SIFT_descriptors.push_back(descriptors);
	}
	cout << endl;
	// ----------------------------------------------------------------------------------------
	
	// Computing PCA SIFT feature for each testing images--------------------------------------
	cout << "Computing PCA SIFT features of testing images..." << endl;
	vector<Mat> testing_PCA_SIFT_features;
	
	for(unsigned int i=0; i<testing_images.size(); i++)
	{
		projection = testing_SIFT_descriptors[i]*reduced_eigenvectors;
		testing_PCA_SIFT_features.push_back(projection);
	}
	cout << endl;
	// ----------------------------------------------------------------------------------------
	
	//cout << "Computing histogram for each testing images-------------------------------------
	cout << "Computing histogram of each test images..." << endl;
	vector< vector<double> > testing_histogram;
	for(unsigned int i=0; i<testing_images.size(); i++)
	{
		testing_histogram.push_back(vector< double >(nclusters, 0.0));
	}

	// histogram based on nearest neighborhood 
	for(unsigned int i=0; i<testing_images.size(); i++)
	{
		for(int j=0; j<testing_PCA_SIFT_features[i].rows; j++)
		{
			double min_distance = INT_MAX;
			int label = 0;

			for(int k=0; k<nclusters; k++)
			{
				double euclidian_distance = 0.0;
				for(int l=0; l<pcaReduced; l++)
				{
					euclidian_distance += ( pow( (testing_PCA_SIFT_features[i].at<float>(j, l) - centers.at<float>(k, l)), 2.0) ); 
				}
				euclidian_distance = sqrt(euclidian_distance);
				if(euclidian_distance<min_distance)
				{
					min_distance = euclidian_distance;
					label = k;
				}
			}
			testing_histogram[i][label]++;
		}
		// Normalize histogram if requires, uncomment below
		/*for(int j=0; j<nclusters; j++)
		{
			testing_histogram[i][j] = (testing_histogram[i][j]*100)/testing_SIFT_descriptors[i].rows;
		}*/
	}
	cout << endl;
	// ----------------------------------------------------------------------------------------

	// Computing k-nearest neighbor for test images--------------------------------------------
	cout << "Computing k-nearest neighbor for finding class of object..." << endl;
	map<int, double> dist[50];
	map<double, int> sortedDist[50];
	map<double, int>::iterator it;

	// KNN for nearset neighbor based on euclidian distance
	for(unsigned int i=0; i<testing_images.size(); i++)
	{	
		for(int j=0; j<training_images.size(); j++)
		{
			double euclidian_distance = 0.0;
			for(int k=0; k<nclusters; k++)
			{
				euclidian_distance += ( pow( (testing_histogram[i][k]-training_histogram[j][k]), 2.0) );
			}
			dist[i].insert(pair<int, double>(j, sqrt(euclidian_distance)));
		}
		sortedDist[i] = flip_map(dist[i]);
	}
	
	int kNN = 10; // set for number of nearest neighbor
	//find the class based on weight of K nearest neighbors
	int classifyLabel[50] = {0};
	for(int i=0; i<testing_images.size(); i++)
	{
		cout << "For image " << i+1 << " : ";
		it=sortedDist[i].begin();
		double count1=0.0, count2=0.0, count3=0.0, count4=0.0, count5=0.0;
		for(int j=0; j<kNN; j++)
		{
			int label = (it)->second;
			
			if(label>=0 && label<20)
			{
				count1 += (1/it->first);
			}
			else if(label>=20 && label<40)
			{
				count2 += (1/it->first);
			}
			else if(label>=40 && label<60)
			{
				count3 += (1/it->first);
			}
			else if(label>=60 && label<80)
			{
				count4 += (1/it->first);
			}
			else if(label>=80 && label<100)
			{
				count5 += (1/it->first);
			}
			
			//cout << (it)->second << ", ";
			it++;
		}
		int labelget = returnLabel(count1, count2, count3, count4, count5);
		classifyLabel[i] = labelget;
		cout << labelget;

		cout << endl;
	}
	cout << endl;
	// ----------------------------------------------------------------------------------------

	// Display Confusion Matrix----------------------------------------------------------------
	cout << "Displaying confusion matrix: " << endl;
	int confusionMat[5][5] = {0}; // create cnfusion matrix
	for(unsigned int i=0; i<5; i++)
	{
		int count1=0, count2=0, count3=0, count4=0, count5=0;
		for(int j=0; j<10; j++)
		{
			if(classifyLabel[(10*i)+j]==1)
				count1++;
			else if(classifyLabel[(10*i)+j]==2)
				count2++;
			else if(classifyLabel[(10*i)+j]==3)
				count3++;
			else if(classifyLabel[(10*i)+j]==4)
				count4++;
			else if(classifyLabel[(10*i)+j]==5)
				count5++;
		}
		confusionMat[i][0] = count1;
		confusionMat[i][1] = count2;
		confusionMat[i][2] = count3;
		confusionMat[i][3] = count4;
		confusionMat[i][4] = count5;
	}
	cout << "\t\tCAR" << "\tFACE" <<  "\tLAPTOP" << "\tM_BIKE" << "\tPIGEON" << endl;
	cout << endl;
	string arr[5] = { "CAR", "FACE", "LAPTOP", "M_BIKE", "PIGEON"};
	for(int i=0; i<5; i++)
	{
		cout << arr[i] << "\t"; 
		for(int j=0; j<5; j++)
		{
			cout << "\t" << confusionMat[i][j];
		}
		cout << endl << endl;
	}
	// ----------------------------------------------------------------------------------------

	// Display Accuracy of the program---------------------------------------------------------
	cout << "Performance of the program: " << endl;
	cout << "Number of clusters taken: " << nclusters << endl;
	cout << "PCA dimension reduced from 128 to : " << pcaReduced << endl;
	cout << "KNN is chosen to be : " << kNN << endl << endl;
	
	cout << "Accuracy for CAR images is : " << (confusionMat[0][0]*100)/10.0 << "%" << endl;
	cout << "Accuracy for FACE images is : " << (confusionMat[1][1]*100)/10.0 << "%" << endl;
	cout << "Accuracy for LAPTOP images is : " << (confusionMat[2][2]*100)/10.0 << "%" << endl;
	cout << "Accuracy for MOTORBIKE images is : " << (confusionMat[3][3]*100)/10.0 << "%" << endl;
	cout << "Accuracy for PIGEON images is : " << (confusionMat[4][4]*100)/10.0 << "%" << endl;
	cout << endl; 

	double average_accuracy = (confusionMat[0][0]+confusionMat[1][1]+confusionMat[2][2]+confusionMat[3][3]+confusionMat[4][4])*100/50.0;
	cout << "Average Accuracy is : " << average_accuracy << "%" << endl;
	// ----------------------------------------------------------------------------------------
	//----------------------------------------------------------Testing Done----------------------------------------------------------------------//
	
	// Disply windows
	namedWindow("Input Image", CV_WINDOW_AUTOSIZE);
	
	imshow("Input Image", testing_images[0]);
	
	waitKey(0);  // wait for stroke key

	cout << "End of Program: " << endl << endl;

	return 0;
}
