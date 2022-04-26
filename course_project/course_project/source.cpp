#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;


int main()
{
    Mat photo_bgr = imread("photo_6.jpg", 1);
    Mat photo_rgb, test, histo_before, histo_after, grey_src;

    cvtColor(photo_bgr, grey_src, COLOR_BGR2GRAY);

    cvtColor(photo_bgr, photo_rgb, COLOR_BGR2RGB);

    vector<Mat> lab_planes(3);
    split(photo_rgb, lab_planes);

    Ptr<CLAHE> clahe = createCLAHE(1.5, Size(8, 8));
    Mat R, G, B;

    clahe->apply(lab_planes[0], R);
    clahe->apply(lab_planes[1], G);
    clahe->apply(lab_planes[2], B);

    R.copyTo(lab_planes[0]);
    G.copyTo(lab_planes[1]);
    B.copyTo(lab_planes[2]);
    merge(lab_planes, photo_rgb);

    cvtColor(photo_rgb, photo_rgb, COLOR_RGB2BGR);



    Mat rgb = photo_rgb;
    //pyrDown(photo_rgb, rgb);
    Mat small;
    cvtColor(rgb, small, COLOR_BGR2GRAY);
    Mat grad;
    Mat morphKernel = getStructuringElement(MORPH_ELLIPSE, Size(9, 9));
    morphologyEx(small, grad, MORPH_GRADIENT, morphKernel);
    Mat bw;
    //imshow("тест", grad);
    //waitKey(0);
    threshold(grad, bw, 0.0, 255.0, THRESH_BINARY | THRESH_OTSU);
    Mat connected;
    morphKernel = getStructuringElement(MORPH_RECT, Size(15, 3));
    morphologyEx(bw, connected, MORPH_CLOSE, morphKernel);
    Mat mask = Mat::zeros(bw.size(), CV_8UC1);
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(connected, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE, Point(0, 0));
    for (int idx = 0; idx >= 0; idx = hierarchy[idx][0])
    {
        Rect rect = boundingRect(contours[idx]);
        Mat maskROI(mask, rect);
        maskROI = Scalar(0, 0, 0);
        drawContours(mask, contours, idx, Scalar(255, 255, 255), -1);
        double r = (double)countNonZero(maskROI) / (rect.width * rect.height);

        if (r > .45
            &&
            (rect.height > 35 && rect.width > 35) 
            )
        {
            cout << rect.x << " " << rect.y << endl;
            rectangle(rgb, rect, Scalar(0, 255, 0), 2);
        }
    }
    imwrite("6.jpg", rgb);

    return 0;
}