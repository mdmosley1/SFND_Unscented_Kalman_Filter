#ifndef UKF_H
#define UKF_H

#include "Eigen/Dense"
#include "measurement_package.h"

struct PredictionData
{
    // predicted sigma points
    std::vector<Eigen::VectorXd> Xsig_pred;
    // predicted Sigma points in measurement space
    std::vector<Eigen::VectorXd> Zsig; 

    Eigen::VectorXd x; // predicted mean state
    Eigen::MatrixXd P; // predicted covariance
    Eigen::VectorXd z; // predicted measurement
    Eigen::MatrixXd S; // predicted measurement covariance
};

class UKF
{
 public:
  /**
   * Constructor
   */
  UKF();

  /**
   * Destructor
   */
  virtual ~UKF();

  /**
   * ProcessMeasurement
   * @param meas_package The latest measurement data of either radar or laser
   */
  void ProcessMeasurement(MeasurementPackage meas_package);
  
  
  PredictionData Prediction(const double delta_t,
                            const Eigen::VectorXd& x,
                            const Eigen::MatrixXd& P) const;

  Eigen::VectorXd GetState() const;

  Eigen::MatrixXd GetCovariance() const;
    

 private:

  void SetStateFromMeasurement(Eigen::VectorXd _z, MeasurementPackage::SensorType _type);

  std::tuple<Eigen::VectorXd, Eigen::MatrixXd, std::vector<Eigen::VectorXd>> PredictRadarMeasurement(const std::vector<Eigen::VectorXd> predictedSigmaPoints);

  std::tuple<Eigen::VectorXd, Eigen::MatrixXd> PredictMeanAndCovariance(const std::vector<Eigen::VectorXd>& predictedSigmaPoints) const;

  std::vector<Eigen::VectorXd> GenerateAugmentedSigmaPoints(Eigen::VectorXd _x, Eigen::MatrixXd _P) const;
  std::vector<Eigen::VectorXd> GenerateSigmaPoints(Eigen::VectorXd _x, Eigen::MatrixXd _P) const;
  
  void UpdateState(const MeasurementPackage& meas, PredictionData& _pData);
        
  /**
   * Updates the state and the state covariance matrix using a laser measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateLidar(MeasurementPackage meas_package);

  /**
   * Updates the state and the state covariance matrix using a radar measurement
   * @param meas_package The measurement at k+1
   */
  
  void UpdateRadar(const PredictionData& _pData, const Eigen::VectorXd _z);


  // initially set to false, set to true in first call of ProcessMeasurement
  bool is_initialized_ = false;

  // if this is false, laser measurements will be ignored (except for init)
  bool use_laser_ = true;

  // if this is false, radar measurements will be ignored (except for init)
  bool use_radar_ = true;

  // state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
  Eigen::VectorXd x_;

  // state covariance matrix
  Eigen::MatrixXd P_;

  // time when the state is true, in us
  long long time_us_;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  double std_a_;

  // Process noise standard deviation yaw acceleration in rad/s^2
  double std_yawdd_;

  // Laser measurement noise standard deviation position1 in m
  double std_laspx_;

  // Laser measurement noise standard deviation position2 in m
  double std_laspy_;

  // Radar measurement noise standard deviation radius in m
  double std_radr_;

  // Radar measurement noise standard deviation angle in rad
  double std_radphi_;

  // Radar measurement noise standard deviation radius change in m/s
  double std_radrd_ ;

  // Weights of sigma points
  std::vector<double> weights_;

  // State dimension
   int n_x_ = 5;

  // Augmented state dimension
   int n_aug_ = 7;

  // Sigma point spreading parameter
    double lambda_ = 3 - n_aug_;

    double lastTime_ = 0.0;
    
};

#endif  // UKF_H
