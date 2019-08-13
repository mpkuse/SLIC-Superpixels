#include "SlicClustering.h"


/*
 * Constructor. Nothing is done here.
 */
SlicClustering::SlicClustering() {

}

/*
 * Destructor. Clear any present data.
 */
SlicClustering::~SlicClustering() {
    clear_data();
}

/*
 * Clear the data as saved by the algorithm.
 *
 * Input : -
 * Output: -
 */
void SlicClustering::clear_data() {
    clusters.clear();
    distances.clear();
    centers.clear();
    center_counts.clear();
}



void SlicClustering::init_data( const cv::Mat& image, const cv::Mat& depth )
{
    /* asserts  */
    assert( image.rows > 0 && image.cols > 0 );
    assert( image.rows == depth.rows && image.cols == depth.cols );
    #if COLOR_CHANNEL == 1
    assert( image.channels() == 1 );
    assert( image.type() == CV_8UC1 );
    assert( depth.type() == CV_16UC1 );
    #else
    assert( image.channels() == 3 );
    assert( image.type() == CV_8UC3 );
    assert( depth.type() == CV_16UC1 );
    #endif

    /* Initialize the cluster and distance matrices (ie. at pixel-wise cluster marker). */
    for (int j = 0; j < image.rows; j++) {
        vector<int> cr; cr.clear();
        vector<double> dr; dr.clear();
        for (int i = 0; i < image.cols; i++) {
            cr.push_back(-1);
            dr.push_back(FLT_MAX);
        }
        clusters.push_back(cr);
        distances.push_back(dr);
    }

    /* Initialize the centers and counters. */
    int cc = 0;
    for (int j = step; j < image.rows - step/2; j += step) {
        for (int i = step; i < image.cols - step/2; i += step) {
            // vector<double> center;
            /* Find the local minimum (gradient-wise). */
            // cv::Point nc = find_local_minimum(image, cvPoint(i,j));
            // CvScalar colour = cvGet2D(image, nc.y, nc.x);

            PixElement center;
            cv::Point nc;// = find_local_minimum(image, cvPoint(i,j)); //TODO
            nc = cv::Point( i, j ); //quick fix, just set it as center and get by

            /* Generate the center vector. */
            // center.push_back(colour[0]);
            // center.push_back(colour[1]);
            // center.push_back(colour[2]);
            // center.push_back(nc.x);
            // center.push_back(nc.y);
            center.u = j; //rowIdx
            center.v = i; //colIdx


            float depth_val = depth.at<uint16_t>( nc.y, nc.x ); //TODO: if Z is zero (aka invalid depth initialize cluster center with someother point. )
            back_project( center.v, center.u,  depth_val,  center.X, center.Y, center.Z );

            #if COLOR_CHANNEL == 3
            cv::Scalar colour = image.at<cv::Vec3b>( nc.y, nc.x );
            center.red = colour[0];
            center.green = colour[1];
            center.blue = colour[2];
            #else
            uchar colour = image.at<uchar>( nc.y, nc.x );
            center.intensity = (float) colour;
            #endif



            /* Append to vector of centers. */
            cout << "init cluster#"  <<  cc << " :\t";
            PixElement::pretty_print( center );
            centers.push_back(center);
            center_counts.push_back(0);
            cc++;
        }
    }
}


void SlicClustering::generate_superpixels(   const cv::Mat& image, const cv::Mat& depth, int step, int nc )
{
    this->step = step;
    this->nc = nc;
    this->ns = step;

    /* Clear previous data (if any), and re-initialize it. */
    clear_data();
    init_data(image, depth);


    /* Run EM for 10 iterations (as prescribed by the algorithm). */
    for (int i = 0; i < NR_ITERATIONS; i++) {
        cout << ">>>>>>>>>>>>>>>>> EM Iteration#" << i << endl;
        /* Reset distance values. */
        for (int k = 0;k < image.rows; k++) {
            for (int j = 0; j < image.cols; j++) {
                distances[k][j] = FLT_MAX;
            }
        }

        for (int j = 0; j < (int) centers.size(); j++) {
            /* Only compare to pixels in a 2 x step by 2 x step region. */
            for (int l = centers[j].v - step; l < centers[j].v + step; l++) {
                for (int k = centers[j].u - step; k < centers[j].u + step; k++) {

                    if (k >= 0 && k < image.rows && l >= 0 && l < image.cols ) {
                        PixElement tmp;
                        tmp.u = k; tmp.v = l;

                        float depth_val = depth.at<uint16_t>( k, l );
                        back_project( l, k,  depth_val,  tmp.X, tmp.Y, tmp.Z );


                        #if COLOR_CHANNEL == 3
                        cv::Scalar colour = image.at< cv::Vec3b >( k, l );
                        tmp.red = colour[0]; tmp.green = colour[1]; tmp.blue = colour[2];
                        #else
                        tmp.intensity = (float) image.at< uchar >( k, l );
                        #endif

                        // double d = compute_dist(j, cvPoint(k,l), colour);
                        double d = PixElement::distance( centers[j], tmp );

                        /* Update cluster allocation if the cluster minimizes the
                           distance. */
                        if (d < distances[k][l]) {
                            distances[k][l] = d;
                            clusters[k][l] = j;
                        }
                    }
                }
            }
        }

        /* Clear the center values. */
        for (int j = 0; j < (int) centers.size(); j++) {
            // centers[j][0] = centers[j][1] = centers[j][2] = centers[j][3] = centers[j][4] = 0;
            centers[j].reset();
            center_counts[j] = 0;
        }

        /* Compute the new cluster centers. */
        for (int j = 0; j < image.cols; j++) {
            for (int k = 0; k < image.rows; k++) {
                int c_id = clusters[k][j];

                if (c_id != -1) {
                    PixElement tmp;
                    tmp.u = k; tmp.v = j;

                    float depth_val = depth.at<uint16_t>( tmp.u, tmp.v );
                    back_project( tmp.v, tmp.u,  depth_val,  tmp.X, tmp.Y, tmp.Z );



                    centers[c_id].u += tmp.u;
                    centers[c_id].v += tmp.v;
                    centers[c_id].X += tmp.X;
                    centers[c_id].Y += tmp.Y;
                    centers[c_id].Z += tmp.Z;

                    #if COLOR_CHANNEL == 3
                    cv::Scalar colour = image.at< cv::Vec3b >( tmp.u, tmp.v );
                    tmp.red = colour[0]; tmp.green = colour[1]; tmp.blue = colour[2];
                    centers[c_id].red += tmp.red;
                    centers[c_id].green += tmp.green;
                    centers[c_id].blue += tmp.blue;
                    #else
                    tmp.intensity = (float) image.at< uchar >( k, l );
                    centers[c_id].intensity += tmp.intensity;
                    #endif

                    // CvScalar colour = cvGet2D(image, k, j);
                    // centers[c_id][0] += colour.val[0];
                    // centers[c_id][1] += colour.val[1];
                    // centers[c_id][2] += colour.val[2];
                    // centers[c_id][3] += j;
                    // centers[c_id][4] += k;

                    center_counts[c_id] += 1;
                }
            }
        }

        #if 1
        /* Normalize the clusters. */
        for (int j = 0; j < (int) centers.size(); j++) {
            // centers[j][0] /= center_counts[j];
            // centers[j][1] /= center_counts[j];
            // centers[j][2] /= center_counts[j];
            // centers[j][3] /= center_counts[j];
            // centers[j][4] /= center_counts[j];

            centers[j].u /= center_counts[j];
            centers[j].v /= center_counts[j];
            centers[j].X /= center_counts[j];
            centers[j].Y /= center_counts[j];
            centers[j].Z /= center_counts[j];
            #if COLOR_CHANNEL == 3
            centers[j].red /= center_counts[j];
            centers[j].green /= center_counts[j];
            centers[j].blue /= center_counts[j];
            #else
            centers[j].intensity /= center_counts[j];
            #endif


        }
        #endif
    }
}



void SlicClustering::display_center_grid(cv::Mat& image, cv::Scalar colour) {
    for (int i = 0; i < (int) centers.size(); i++) {
        // cvCircle(image, cvPoint(centers[i][3], centers[i][4]), 2, colour, 2);
        cout << "center#" << i << "\t count = " << center_counts[i] << "\t";
        PixElement::pretty_print( centers[i] );
        // cout << endl;

        cv::Point pt( centers[i].v, centers[i].u );
        cv::circle(image, pt, 2, colour, -1, CV_AA);

    }
}



void SlicClustering::display_center_grid() {
    cout << "[SlicClustering::display_center_grid]\n";
    for (int i = 0; i < (int) centers.size(); i++) {

        cout << "center#" << i << "\t count = " << center_counts[i] << "\t";
        PixElement::pretty_print( centers[i] );
        // cout << endl;

    }
}



void SlicClustering::colour_with_cluster_means(cv::Mat& image) {
    assert( image.channels() == 3 );

    vector<cv::Scalar> colours(centers.size());

    /* Gather the colour values per cluster. */
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            int index = clusters[i][j];
            cv::Scalar colour = image.at<cv::Vec3b>( i, j );

            colours[index][0] += colour[0];
            colours[index][1] += colour[1];
            colours[index][2] += colour[2];
        }
    }

    /* Divide by the number of pixels per cluster to get the mean colour. */
    for (int i = 0; i < (int)colours.size(); i++) {
        colours[i][0] /= center_counts[i];
        colours[i][1] /= center_counts[i];
        colours[i][2] /= center_counts[i];
    }

    /* Fill in. */
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            cv::Scalar ncolour = colours[ clusters[i][j] ];

            cv::Vec3b tmp = cv::Vec3b( (uchar)ncolour[2], (uchar)ncolour[1], (uchar)ncolour[0] );
            image.at<cv::Vec3b>(i,j) = tmp;
        }
    }
}




void SlicClustering::display_contours(cv::Mat& image, cv::Scalar colour) {
    assert( image.channels() == 3 );
    const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};

	/* Initialize the contour vector and the matrix detailing whether a pixel
	 * is already taken to be a contour. */
	vector<cv::Point> contours;
	vec2db istaken;
	for (int i = 0; i < image.rows; i++) {
        vector<bool> nb;
        for (int j = 0; j < image.cols; j++) {
            nb.push_back(false);
        }
        istaken.push_back(nb);
    }

    /* Go through all the pixels. */
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            int nr_p = 0;

            /* Compare the pixel to its 8 neighbours. */
            for (int k = 0; k < 8; k++) {
                int x = i + dx8[k], y = j + dy8[k];

                if (x >= 0 && x < image.rows && y >= 0 && y < image.cols) {
                    if (istaken[x][y] == false && clusters[i][j] != clusters[x][y]) {
                        nr_p += 1;
                    }
                }
            }

            /* Add the pixel to the contour list if desired. */
            if (nr_p >= 2) {
                contours.push_back(cv::Point(j,i));
                istaken[i][j] = true;
            }
        }
    }

    /* Draw the contour pixels. */
    cv::Vec3b colo = cv::Vec3b( colour[0], colour[1], colour[2] );
    for (int i = 0; i < (int)contours.size(); i++) {
        // cvSet2D(image, contours[i].y, contours[i].x, colour);
        image.at< cv::Vec3b >( contours[i] ) = colo;
    }
}
