// Adaptation of SLIC for RGB depth clustering

#include <iostream>
#include <vector>
using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>


// #include "slic.h"
#include "SlicClustering.h"

string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

string cvmat_info( const cv::Mat& mat )
{
    std::stringstream buffer;
    buffer << "shape=" << mat.rows << "," << mat.cols << "," << mat.channels() ;
    buffer << "\t" << "dtype=" << type2str( mat.type() );
    return buffer.str();
}


int main( int argc, char ** argv )
{
    cout << "====\n\t" << argv[0] << "\n====\n";

    cv::Mat imA_org = cv::imread( "../images/rgb-d/rgb/1.png");
    cv::Mat imA_lab;
    cv::cvtColor( imA_org, imA_lab, CV_BGR2Lab );
    cv::Mat imA = imA_org;

    cout << "imA Dims: " << cvmat_info( imA_lab ) << endl;
    cv::Mat imA_depth = cv::imread( "../images/rgb-d/depth/1.png", CV_LOAD_IMAGE_ANYDEPTH);
    cout << "imA_depth Dims: " << cvmat_info( imA_depth ) << endl;

    // cout << imA_depth << endl;

    // params
    int w = imA.cols, h = imA.rows;
    int nr_superpixels = 400;
    int nc = 20;
    double step = sqrt((w * h) / ( (double) nr_superpixels ) ); ///< step size per cluster
    cout << "Params:\n";
    cout << "step size per cluster: " << step << endl;
    cout << "Weight: " << nc << endl;
    cout << "Number of superpixel: "<< nr_superpixels << endl;
    cout << "===\n";


    SlicClustering slic_obj;
    slic_obj.generate_superpixels( imA_lab, imA_depth, step, nc );

    //viz
    // slic_obj.display_center_grid( imA, cv::Scalar(0,0,255) );
    // slic_obj.colour_with_cluster_means( imA );
    slic_obj.display_contours( imA,  cv::Scalar(0,0,255) );



    cv::imshow( "result", imA );
    cv::waitKey( 0 );

}
