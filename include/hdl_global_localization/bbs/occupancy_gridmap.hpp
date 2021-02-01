#ifndef HDL_GLOBAL_LOCALIZATION_OCCUPANCY_GRID_HPP
#define HDL_GLOBAL_LOCALIZATION_OCCUPANCY_GRID_HPP

#include <vector>
#include <Eigen/Core>
#include <nav_msgs/OccupancyGrid.h>

#include <opencv2/opencv.hpp>

namespace hdl_global_localization {

class OccupancyGridMap {
public:
  using Ptr = std::shared_ptr<OccupancyGridMap>;

  OccupancyGridMap(double resolution, const cv::Mat& values) {
    this->resolution = resolution;
    this->values = values;
  }

  OccupancyGridMap(double resolution, int width, int height) {
    this->resolution = resolution;
    values = cv::Mat1f(height, width, 0.0f);
  }

  double grid_resolution() const { return resolution; }
  int width() const { return values.cols; }
  int height() const { return values.rows; }
  const float* data() const { return reinterpret_cast<const float*>(values.data); }

  void insert_points(const std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f>>& points, int min_points_per_grid) {
    for (const auto& pt : points) {
      auto loc = grid_loc(pt); //计算map下每个point在grid_map下的行列坐标
      if (in_map(loc)) { //在grid_map内
        value(loc) += 1; //每当一个map point落入grid_map某个位置，对应位置元素加1
      }
    }

    values /= min_points_per_grid; //注意: 对每个grid_map元素归一化
    for (auto& x : values) {
      x = std::min(x, 1.0f); //落入对应位置点的个数超过max_points_per_cell时，强制为1
    }
  }

  double calc_score(const std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f>>& points) const {
    double sum_dists = 0.0;
    for (const auto& pt : points) {
      auto loc = grid_loc(pt);
      loc.x() = std::max<int>(0, std::min<int>(values.cols - 1, loc.x()));
      loc.y() = std::max<int>(0, std::min<int>(values.rows - 1, loc.y()));
      sum_dists += value(loc);
    }

    return sum_dists;
  }

  OccupancyGridMap::Ptr pyramid_up() const {
    cv::Mat1f small_map(values.rows / 2, values.cols / 2);
    for (int i = 0; i < values.rows / 2; i++) {
      for (int j = 0; j < values.cols / 2; j++) {
        float x = values.at<float>(i * 2, j * 2);
        x = std::max(x, values.at<float>(i * 2 + 1, j * 2));
        x = std::max(x, values.at<float>(i * 2, j * 2 + 1));
        x = std::max(x, values.at<float>(i * 2 + 1, j * 2 + 1));
        small_map.at<float>(i, j) = x;
      }
    }

    return std::make_shared<OccupancyGridMap>(resolution * 2.0, small_map);
  }

  nav_msgs::OccupancyGridConstPtr to_rosmsg() const {
    nav_msgs::OccupancyGridPtr msg(new nav_msgs::OccupancyGrid);
    msg->header.frame_id = "map";
    msg->header.stamp = ros::Time(0);

    msg->data.resize(values.rows * values.cols);
    std::transform(values.begin(), values.end(), msg->data.begin(), [=](auto x) {
      double x_ = x * 100.0;
      return std::max(0.0, std::min(100.0, x_));
    });

    msg->info.map_load_time = ros::Time::now();
    msg->info.width = values.cols;
    msg->info.height = values.rows;
    msg->info.resolution = resolution;

    msg->info.origin.position.x = -resolution * values.cols / 2;
    msg->info.origin.position.y = -resolution * values.rows / 2;
    msg->info.origin.position.z = 0.0; //map坐标系(grid_map中心位置，x右，y前)的左下角

    return msg;
  }

private:
  bool in_map(const Eigen::Vector2i& pix) const {
    bool left_bound = (pix.array() >= Eigen::Array2i::Zero()).all();
    bool right_bound = (pix.array() < Eigen::Array2i(values.cols, values.rows)).all();
    return left_bound && right_bound;
  }
  
  //cv::Mat的行列分别对应grid_map的(y, x)
  float value(const Eigen::Vector2i& loc) const { return values.at<float>(loc.y(), loc.x()); }
  float& value(const Eigen::Vector2i& loc) { return values.at<float>(loc.y(), loc.x()); }
  
  //计算在grid_map中的行列坐标，grid_map在map坐标系的左下角。x轴，y轴分别与map坐标系的x轴，y轴对齐。
  Eigen::Vector2i grid_loc(const Eigen::Vector2f& pt) const {
    Eigen::Vector2i loc = (pt / resolution).array().floor().cast<int>();
    Eigen::Vector2i offset = Eigen::Vector2i(values.cols / 2, values.rows / 2);
    return loc + offset;
  }

private:
  double resolution;
  cv::Mat1f values;  //落入对应位置点的个数超过max_points_per_cell时，强制为1，元素值的大小在[0, 1]
};

}  // namespace hdl_global_localization

#endif