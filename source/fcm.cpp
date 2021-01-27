#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <math.h>
#include <Eigen/Dense>
#include <float.h>
#include "fcm.h"
#include "record.h"

extern vector<traj_base> traj;
extern vector<centers_fcm_base> centers_fcm;

FCM::FCM(double m, double epsilon) {
  m_epsilon = epsilon;
  m_m = m;
  m_membership = nullptr;
  m_data = nullptr;
  m_cluster_center = nullptr;
  m_num_clusters = 0;
  m_num_dimensions = 0;
}

FCM::~FCM() {
  if (m_data != nullptr) {
    delete m_data;
    m_data = nullptr;
  }

  if (m_membership != nullptr) {
    delete m_membership;
    m_membership = nullptr;
  }

  if (m_cluster_center != nullptr) {
    delete m_cluster_center;
    m_cluster_center = nullptr;
  }
}


double FCM::update_membership() {

  long k, i;
  float new_uik;
  double max_diff = 0.0, diff;

  if (m_data == nullptr || m_data->rows() == 0) {
    throw std::logic_error("ERROR: data should not be empty when updating the membership");
  }

  if (m_membership == nullptr || m_membership->rows() == 0 || m_membership->rows() != m_data->rows()) {
    this->init_membership();
  }
  if (m_num_clusters == 0) {
    throw std::logic_error("ERROR: the number of clusters should be set");
  }

  for (i = 0; i < m_num_clusters; i++) {
    for (k = 0; k < m_data->rows(); k++) {
      //cout << "point: " << k << " and cluster" << i <<endl;
      //cout << "\nwill ask for the new new_uik"<< endl;
      new_uik = this->compute_membership_point(i, k);
      diff = new_uik - (*m_membership)(k, i); // We need the membership inversed which is more natural for us
      if (diff > max_diff) {
        max_diff = diff;
      }
      (*m_membership)(k, i) = new_uik;
    }
  }
  return max_diff;
}


void FCM::compute_centers() {
  long i, j, k;
  double numerator, denominator;
  MatrixXf t;
  t.resize(m_data->rows(), m_num_clusters);
  if (m_data == nullptr || m_data->rows() == 0) {
    throw std::logic_error("ERROR: number of rows is zero");
    return;
  }
  for (i = 0; i < m_data->rows(); i++) { // compute (u^m) for each cluster for each point
    for (j = 0; j < m_num_clusters; j++) {
      t(i, j) = float(pow((*m_membership)(i, j), m_m));
    }
  }
  for (j = 0; j < m_num_clusters; j++) { // loop for each cluster
    for (k = 0; k < m_num_dimensions; k++) { // for each dimension
      numerator = 0.0;
      denominator = 0.0;
      for (i = 0; i < m_data->rows(); i++) {
        numerator += t(i, j) * (*m_data)(i, k);
        denominator += t(i, j);
      }
      (*m_cluster_center)(j, k) = float(numerator / denominator);
    }
  }
}

double FCM::get_dist(long i, long k) {
  /*
   * distance which is denoted in the paper as d
   * k is the data point
   * i is the cluster center point
  */
  //cout<<"get_dist: point: "<<k<<" and cluster "<<i<<endl;
  long j;
  double sqsum = 0.0;
  if (m_num_clusters == 0) {
    throw std::logic_error("ERROR: number of clusters should not be zero\n");
  }
  if (m_num_dimensions == 0) {
    throw std::logic_error("ERROR: number of dimensions should not be zero\n");
  }
  for (j = 0; j < m_num_dimensions; j++) {
    sqsum += pow(((*m_data)(k, j) - (*m_cluster_center)(i, j)), 2);
  }
  return sqrt(sqsum);
}

float FCM::compute_membership_point(long i, long k) {
  /*
   * i the cluster
   * k is the data point
  */
  //cout << __func__ <<"  num of cluster: "<<m_num_clusters<<endl;
  long j;
  double t, seg = 0.0;
  double exp = 2 / (m_m - 1);
  double dik, djk;
  if (m_num_clusters == 0) {
    throw std::logic_error("ERROR: number of clusters should not be zero\n");
  }
  for (j = 0; j < m_num_clusters; j++) {
    //std::cout << i << "  " << j << "  " << k << std::endl;
    dik = this->get_dist(i, k);
    djk = this->get_dist(j, k);
    //std::cout << dik << "  " << djk << std::endl;
    if (djk == 0) {
      djk = DBL_MIN;
    }
    t = dik / djk;
    t = pow(t, exp);
    //cout << "cluster: " << i << "data: " << k << " - " << "t: "<<t<<endl;
    seg += t;
  }
  //std::cin.get();

  //cout << "seg: "<<seg << " u: "<<(1.0/seg)<<endl;
  return float(1.0 / seg);
}


void FCM::set_data(MatrixXf *data) {
  if (m_data != nullptr) {
    delete m_data;
  }
  if (data->rows() == 0) {
    throw std::logic_error("ERROR: seting empty data");
  }
  m_data = data;
  m_num_dimensions = m_data->cols();
}

void FCM::set_membership(MatrixXf *membership) {
  if (m_data == 0) {
    throw std::logic_error("ERROR: the data should present before setting up the membership");
  }
  if (m_num_clusters == 0) {
    if (membership->cols() == 0) {
      throw std::logic_error("ERROR: the number of clusters is 0 and the membership matrix is empty");
    }
    else {
      this->set_num_clusters(membership->cols());
    }
  }
  if (m_membership != nullptr) {
    delete m_membership;
  }
  m_membership = membership;
  if (m_membership->rows() == 0) {
    m_membership->resize(m_data->rows(), m_num_clusters);
  }
}

void FCM::init_membership() {
  long i, j;
  double mem;
  if (m_num_clusters == 0) {
    throw std::logic_error("ERROR: the number of clusters is 0");
  }
  if (m_data == nullptr) {
    throw std::logic_error("ERROR: the data should present before setting up the membership");
  }
  if (m_membership != nullptr) {
    delete m_membership;
  }
  m_membership = new MatrixXf;
  m_membership->resize(m_data->rows(), m_num_clusters);
  mem = 1.0 / m_num_clusters;
  for (j = 0; j < m_num_clusters; j++) {
    for (i = 0; i < m_data->rows(); i++) {
      (*m_membership)(i, j) = float(mem);
    }
  }
}

void FCM::set_num_clusters(long num_clusters) {
  m_num_clusters = num_clusters;
  if (m_cluster_center) {
    delete m_cluster_center;
  }
  m_cluster_center = new MatrixXf;
  m_cluster_center->resize(m_num_clusters, m_num_dimensions);
}

MatrixXf * FCM::get_data() {
  return m_data;
}

MatrixXf * FCM::get_membership() {
  return m_membership;
}

MatrixXf * FCM::get_cluster_center() {
  return m_cluster_center;
}

// DUNN INDEX //
double d_eu_distance(vector<double> v1, vector<double> v2) {
  double d_eu = 0.0;
  for (int n = 0; n < v1.size(); ++n)
    d_eu += pow((v1[n] - v2[n]), 2);

  if (d_eu > 0) return sqrt(d_eu);
  else return 0.0;
}
//-------------------------------------------------------------------
double intra_distance(int i) {
  double max_dist = 0.0;
  for (int n = 0; n < traj.size(); ++n)
    if (traj[n].means_class == i)
      for (int m = 0; m < traj.size(); ++m)
        if (traj[m].means_class == i)
        {
          vector<double> v1 = { traj[n].average_speed, traj[n].v_max, traj[n].v_min };
          vector<double> v2 = { traj[m].average_speed, traj[m].v_max, traj[m].v_min };
          double dist = d_eu_distance(v1, v2);
          if (max_dist < dist)
            max_dist = dist;
        }

  return max_dist;
}
//-------------------------------------------------------------------
double inter_distance(int n, int m) {
  double min_dist = 1e6;
  for (auto &t1: traj)
    if (t1.means_class == n)
      for (auto &t2:traj)
        if (t2.means_class == m)
        {
          vector<double> v1 = { t1.average_speed, t1.v_max, t1.v_min };
          vector<double> v2 = { t2.average_speed, t2.v_max, t2.v_min };
          double dist = d_eu_distance(v1, v2);
          if (min_dist > dist)
            min_dist = dist;
        }

  return min_dist;
}
//-------------------------------------------------------------------
double measure_dunn_index() {
  double num=  1e6;
  double den = 0.0;

  for (auto &c : centers_fcm) {
    double intra_dist_ci = intra_distance(c.idx);
    if (intra_dist_ci > den)
      den = intra_dist_ci;
  }

  for (int n=0; n<centers_fcm.size()-1; ++n)
    for (int m = n + 1; m < centers_fcm.size(); ++m) {
      double inter_n_m = inter_distance(n, m);
      if (inter_n_m < num)
        num = inter_n_m;
    }

  //std::cout << "num: " << num << " den: " << den << std::endl;
  return (num/den);
}
//-------------------------------------------------------------------








