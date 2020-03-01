#include <iostream>

#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>


#define PI 3.1415926
#define ROBOT_RADIOS 10

using namespace std;
using namespace Eigen;
using namespace cv;
namespace bg = boost::geometry;

typedef bg::model::d2::point_xy<double> DPoint;
typedef bg::model::segment<DPoint> DSegment;

vector<Vector2d> partricle_ste;




double evalfunc(double parameter[], int FUNC = 1)
{
    if (FUNC == 1)
    {
        double val = 0;
        val = pow(parameter[0],2) + 1 * pow((parameter[1] - pow(parameter[0],2)),2);
        return val;
    }

    if (FUNC == 2)
    {
        double val = 0;
        for (int i = 0; i < NVARS; i++)
        {
            val += (parameter[i] * parameter[i] - 10 * cos(2 * PI * parameter[i] ) + 10.0);
        }
        return val;
    }
}

















int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
