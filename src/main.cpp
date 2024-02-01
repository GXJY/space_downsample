#include <ikd-Tree/ikd_Tree.h>
#include <omp.h>
#include <pcl/common/common.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <unistd.h>

#include <iostream>
#include <mutex>
#include <vector>

pcl::PointCloud<pcl::PointXYZI>::Ptr make_cubic() {
  std::string save_path = "../data/cubic.pcd";
  pcl::PointCloud<pcl::PointXYZI>::Ptr cubic_cloud(
      new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointXYZI pt;
  int size = 2;
  double step = 0.01;
  for (double x = -size * 0.5; x <= size * 0.5; x += step) {
    for (double y = -size * 0.5; y <= size * 0.5; y += step) {
      for (double z = -size * 0.5; z <= size * 0.5; z += step) {
        pt.x = x;
        pt.y = y;
        pt.z = z;
        pt.intensity = 1;
        cubic_cloud->points.push_back(pt);
      }
    }
  }
  pcl::PCDWriter pcdwriter;
  pcdwriter.writeBinary(save_path, *cubic_cloud);
  std::cout << "Saved cubic cloud to " << save_path << std::endl;
  return cubic_cloud;
}

void space_downsample(const pcl::PointCloud<pcl::PointXYZI>::Ptr input_pc,
                      const double space_size_) {
  auto pro_start = std::chrono::high_resolution_clock::now();
  // ikdtree add-tree
  KD_TREE<pcl::PointXYZI>* ikd_tree = new KD_TREE<pcl::PointXYZI>();
  pcl::PointCloud<PointType>::Ptr space_pc_(new pcl::PointCloud<PointType>);
  space_pc_->push_back(input_pc->points[0]);
  PointVector need_to_add;
  float max_dist = 0.2;
  bool rebuild = true;
  ikd_tree->Build(space_pc_->points);
  std::cout << "input_pc->size(): " << input_pc->size() << std::endl;
  std::cout << "starting filtering " << std::endl;
  for (int i = 0; i < input_pc->size(); i++) {
    if (i % 10000 == 0) {
      auto pro_mid = std::chrono::high_resolution_clock::now();
      auto duration_mid = std::chrono::duration_cast<std::chrono::milliseconds>(pro_mid - pro_start);
      std::cout << "space ds process: " << i  << " of " << input_pc->size() << std::endl;
      std::cout << " using time: " << duration_mid.count() / 1000 << " sec" << std::endl;
    }
    PointVector Nearest_Points;
    vector<float> KPoint_Distance(1, 0);
    if (max_dist < space_size_) {
      std::cout << "max_dist should bigger than space_size_ " << std::endl;
    }
    ikd_tree->Nearest_Search(input_pc->points[i], 1, Nearest_Points,
                            KPoint_Distance, max_dist);
    // std::cout << "Nearest_Points size: " << Nearest_Points.size() << std::endl;
    if (Nearest_Points.size() < 1) {
      space_pc_->push_back(input_pc->points[i]);
      need_to_add.push_back(input_pc->points[i]);
      ikd_tree->Add_Points(need_to_add, false);
      need_to_add.clear();
      continue;
    }
    if ((input_pc->points[i].x - Nearest_Points[0].x) *
                (input_pc->points[i].x - Nearest_Points[0].x) +
            (input_pc->points[i].y - Nearest_Points[0].y) *
                (input_pc->points[i].y - Nearest_Points[0].y) +
            (input_pc->points[i].z - Nearest_Points[0].z) *
                (input_pc->points[i].z - Nearest_Points[0].z) <
        space_size_ * space_size_) {
      continue;
    } else {
      space_pc_->push_back(input_pc->points[i]);
      need_to_add.push_back(input_pc->points[i]);
      ikd_tree->Add_Points(need_to_add, false);
      need_to_add.clear();
    }
  }
  std::cout << "saving " << space_pc_->size() << " data points to "
            << "../data/space_ds.pcd" << std::endl;
  pcl::PCDWriter pcd_writer;
  pcd_writer.writeBinary<pcl::PointXYZI>("../data/space_ds.pcd", *space_pc_);
  auto pro_end = std::chrono::high_resolution_clock::now();
  auto duration_end = std::chrono::duration_cast<std::chrono::milliseconds>(pro_end - pro_start);
  std::cout << "all process used time: " << duration_end.count() / 1000 << " sec" << std::endl;
}

int main() {
  pcl::PointCloud<pcl::PointXYZI>::Ptr ori_pc(
      new pcl::PointCloud<pcl::PointXYZI>);
  // ori_pc = make_cubic();
  pcl::io::loadPCDFile("../data/test.pcd", *ori_pc);
  double space_size = 0.1;
  space_downsample(ori_pc, space_size);

  return 0;
}
