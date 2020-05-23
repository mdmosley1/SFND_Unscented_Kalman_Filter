#include "ukf.h"
#include "Eigen/Dense"

#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Constants

// size of radar measurement space
int n_z = 3;


/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF()
{
    std::cout << "UKF constructor!" << "\n";
    
    // if this is false, laser measurements will be ignored (except during init)
    use_laser_ = false;

    // if this is false, radar measurements will be ignored (except during init)
    use_radar_ = true;

    // initial state vector (px, py, v, theta, theta_dot) (constant velocity, constant turn rate)
    x_ = VectorXd(5);
    x_.fill(0.0);

    // initial covariance matrix
    P_ = MatrixXd::Identity(5, 5);

    // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_ = 2.5;

    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = 0.9;
  
    /**
     * DO NOT MODIFY measurement noise values below.
     * These are provided by the sensor manufacturer.
     */

    // Laser measurement noise standard deviation position1 in m
    std_laspx_ = 0.15;

    // Laser measurement noise standard deviation position2 in m
    std_laspy_ = 0.15;

    // Radar measurement noise standard deviation radius in m
    std_radr_ = 0.3;
    //std_radr_ = 0.003;

    // Radar measurement noise standard deviation angle in rad
    std_radphi_ = 0.03;
    //std_radphi_ = 0.0003;

    // Radar measurement noise standard deviation radius change in m/s
    std_radrd_ = 0.3;
    //std_radrd_ = 0.003;
  
    /**
     * End DO NOT MODIFY section for measurement noise values 
     */

    weights_.push_back(lambda_/(lambda_ + n_aug_));
    for (int i = 1; i<2*n_aug_ + 1; ++i)
        weights_.push_back(0.5/(lambda_ + n_aug_));
}

UKF::~UKF() {}


void NormalizeAngle(double& d)
{
    //assert(std::fabs(d) < 10.0);
    while (d >  M_PI)
    {
        d -= 2.*M_PI;
    }
    while (d < -M_PI)
    {
        d += 2.*M_PI;
    }
}

void PrintStateVector(VectorXd x)
{
    std::cout << "posx      = " << x[0] << "\n";
    std::cout << "posy      = " << x[1] << "\n";
    std::cout << "vel       = " << x[2] << "\n";
    std::cout << "yaw       = " << x[3] << "\n";
    std::cout << "yaw rate  = " << x[4] << "\n";
}


/**
 * TODO: Complete this function! Use radar data to update the belief 
 * about the object's position. Modify the state vector, x_, and 
 * covariance, P_.
 * You can also calculate the radar NIS, if desired.
 */
// void UpdateRadar: update the state from radar measurement
// Inputs
// (*) x_p: predicted state mean
// (*) P_p: predicted state covariance
// (*) z: the radar measurement
void UKF::UpdateRadar(const PredictionData& _pData, const VectorXd _z)
{
    std::cout << __FUNCTION__ << std::endl;
    // create matrix for cross correlation Tc
    MatrixXd Tc = MatrixXd(n_x_, n_z);

    // calculate cross correlation matrix
    Tc.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; ++i)
    {
        // residual
        VectorXd z_diff = _pData.Zsig[i] - _pData.z;
        // angle normalization
        NormalizeAngle(z_diff(1));

        // state difference
        VectorXd x_diff = _pData.Xsig_pred[i] - _pData.x;
        // angle normalization
        NormalizeAngle(x_diff(3));

        Tc = Tc + weights_[i] * x_diff * z_diff.transpose();
    }

    // Kalman gain K;
    MatrixXd K = Tc * _pData.S.inverse();

    // residual
    VectorXd z_diff = _z - _pData.z;

    // angle normalization
    // NormalizeAngle(z_diff(1));

    auto x_Previous = x_;
    // update state mean and covariance matrix
    // the updated state is a combination of the prediction and the measurement
    x_ = _pData.x + K * z_diff;
    P_ = _pData.P - K*_pData.S*K.transpose();

    //NormalizeAngle(x_(3));

    // print result
    std::cout << "\n\n";
    std::cout << "UKF:UpdateRadar" << "\n";
    std::cout << "residual z_diff: \n" << z_diff << "\n";
    std::cout << "S: \n" << _pData.S << "\n";
    std::cout << "S^-1: \n" << _pData.S.inverse() << "\n";
    std::cout << "K: \n" << K << "\n";
    std::cout << "Tc: \n" << Tc << "\n";
    std::cout << "Previous state x: " << std::endl;
    PrintStateVector(x_Previous);
    std::cout << "Updated state x: " << std::endl;
    PrintStateVector(x_);
    std::cout << "\n\n";
    //std::cout << "Updated state covariance P: " << std::endl << P_ << std::endl;
}


void PrintMeasurementVector(VectorXd z)
{
    std::cout << "range      = " << z[0] << "\n";
    std::cout << "angle      = " << z[1] << "\n";
    std::cout << "range rate = " << z[2] << "\n";    

    
}
// output is 
std::tuple<VectorXd, MatrixXd, std::vector<VectorXd>> UKF::PredictRadarMeasurement(const std::vector<Eigen::VectorXd> predictedSigmaPoints)
{
    std::cout << __FUNCTION__ << std::endl;
    // create vector for sigma points in measurement space
    std::vector<VectorXd> Zsig;

    // mean predicted measurement
    VectorXd z_pred = VectorXd(n_z);
    // measurement covariance matrix S
    MatrixXd S = MatrixXd(n_z,n_z);

    // transform sigma points into measurement space
    for (int i = 0; i < 2 * n_aug_ + 1; ++i)
    {  // 2n+1 simga points
        // extract values for better readability
        auto& sigmaPoint = predictedSigmaPoints[i];
        double p_x = sigmaPoint(0);
        double p_y = sigmaPoint(1);
        double v   = sigmaPoint(2);
        double yaw = sigmaPoint(3);

        double v1 = cos(yaw)*v;
        double v2 = sin(yaw)*v;

        // the radar measurement model
        VectorXd predictedSigmaPointMeas = VectorXd(n_z);
        predictedSigmaPointMeas(0) = sqrt(p_x*p_x + p_y*p_y);                       // r
        predictedSigmaPointMeas(1) = atan2(p_y,p_x);                                // phi
        predictedSigmaPointMeas(2) = (p_x*v1 + p_y*v2) / sqrt(p_x*p_x + p_y*p_y);   // r_dot
        Zsig.push_back(predictedSigmaPointMeas);
    }

    // mean predicted measurement
    z_pred.fill(0.0);
    for (int i = 0; i < 2*n_aug_ + 1; ++i)
        z_pred = z_pred + weights_[i] * Zsig[i];

    std::cout << "Print mean predicted measurement:" << "\n";
    PrintMeasurementVector(z_pred);


    // innovation covariance matrix S
    S.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; ++i)
    {  // 2n+1 simga points
        // residual
        VectorXd z_diff = Zsig[i] - z_pred;
        // angle normalization
        NormalizeAngle(z_diff(1));
        S = S + weights_[i] * z_diff * z_diff.transpose();
    }

    // add measurement noise covariance matrix
    MatrixXd R = MatrixXd(n_z,n_z);
    R <<  std_radr_*std_radr_, 0, 0,
          0, std_radphi_*std_radphi_, 0,
        0, 0, std_radrd_*std_radrd_;
    S = S + R;

    std::cout << __FUNCTION__ << ": EXIT" << std::endl;

    return std::make_tuple(z_pred, S, Zsig);
}



// transform the predicted state into the measurement space so it can
// be compared to the physical measurement. Use the measurement model
// z = h(x), which depends on the sensor

void UKF::UpdateState(const MeasurementPackage& _meas, PredictionData& _pData)
{
    // Predict Measurement
    // TODO: Make these two predict functions a method of PredictionData
    if (_meas.sensor_type_ == MeasurementPackage::RADAR)
        std::tie(_pData.z, _pData.S, _pData.Zsig) = PredictRadarMeasurement(_pData.Xsig_pred);
    else if (_meas.sensor_type_ == MeasurementPackage::LASER)
    {
        //std::tie(_pData.z, _pData.S, _pData.Zsig) = PredictLidarMeasurement(_pData.Xsig_pred);
    }
    else
        std::cout << "Unknown measurement type!" << "\n";

    std::cout << "ABOUT TO UPDATE STATE!" << "\n";
        
    // Update State
    if (_meas.sensor_type_ == MeasurementPackage::RADAR)
        UpdateRadar(_pData, _meas.raw_measurements_);
    else if (_meas.sensor_type_ == MeasurementPackage::LASER)
        UpdateLidar(_meas);
    else
        std::cout << "Unknown measurement type!" << "\n";
}

std::vector<Eigen::VectorXd> PredictSigmaPoints(std::vector<Eigen::VectorXd> _sigmaPts, double delta_t) 
{
    // predict sigma points
    std::vector<Eigen::VectorXd> predictedSigmaPts;
    for (auto& sigmaPt : _sigmaPts)
    {
        // extract values for better readability
        double p_x = sigmaPt[0];
        double p_y = sigmaPt[1];
        double v = sigmaPt[2];
        double yaw = sigmaPt[3];
        double yawd = sigmaPt[4];
        double nu_a = sigmaPt[5];
        double nu_yawdd = sigmaPt[6];

        double px_p, py_p;
        // Use the constant velocity, constant yaw rate model
        if (fabs(yawd) > 0.001)
        {
            px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
            py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
        }
        else
        {        // avoid division by zero
            px_p = p_x + v*delta_t*cos(yaw);
            py_p = p_y + v*delta_t*sin(yaw);
        }

        double v_p = v;
        double yaw_p = yaw + yawd*delta_t;
        double yawd_p = yawd;

        // add noise
        px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
        py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
        v_p = v_p + nu_a*delta_t;

        yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
        yawd_p = yawd_p + nu_yawdd*delta_t;

        // write predicted sigma point into right column
        Eigen::VectorXd predictedSigmaPt(5);
        predictedSigmaPt[0] = px_p;
        predictedSigmaPt[1] = py_p;
        predictedSigmaPt[2] = v_p;
        predictedSigmaPt[3] = yaw_p;
        predictedSigmaPt[4] = yawd_p;
        predictedSigmaPts.push_back(predictedSigmaPt);
    }

    // std::cout << "Predicted Sigma Points:" << "\n";
    // for (auto& p : predictedSigmaPts)
    //     std::cout << "sigma point:\n " << p << "\n\n";
    
    return predictedSigmaPts;
}

/**
 * Prediction Predicts sigma points, the state, and the state covariance
 * matrix
 * @param delta_t Time between k and k+1 in s
 */
PredictionData UKF::Prediction(double delta_t) const
{
    delta_t = delta_t / 1000000.0; // convert time into seconds
    // (I) generate the sigma points and weights
    // also return weights. create new struct called weightedSigmaPoint
    auto sigmaPts = GenerateAugmentedSigmaPoints(x_, P_);
    
    // (II) use sigma points to predict the state and error covariance
    auto predictedSigmaPts = PredictSigmaPoints(sigmaPts, delta_t);

    // (III) Compute the predicted mean and covariance from the sigma points
    VectorXd x_predict; // predicted mean state
    MatrixXd P_predict; // predicted covariance
    std::tie(x_predict, P_predict) = PredictMeanAndCovariance(predictedSigmaPts);

    std::cout << "Mean predicted state:\n ";
    PrintStateVector(x_predict);

    PredictionData pData;
    pData.Xsig_pred = predictedSigmaPts;
    pData.x = x_predict;
    pData.P = P_predict;
    
    return pData;
}

Eigen::VectorXd UKF::GetState()
{
    //std::cout << "Get State!" << std::endl;
    return x_;
}

void UKF::SetStateFromMeasurement(Eigen::VectorXd _z, MeasurementPackage::SensorType _type)
{
    if (_type == MeasurementPackage::LASER)
    {
        assert(_z.size() == 2);
        x_[0] = _z[0];
        x_[1] = _z[1];
    }
    else if (_type == MeasurementPackage::RADAR)
    {
        assert(_z.size() == 3);
        // compute position from radar measurement
        double range = _z[0];
        double angle = _z[1];
        x_[0] = range*cos(angle);
        x_[1] = range*sin(angle);
    }
    else
        std::cout << "ERROR: unrecognzied measurement type!" << "\n";
}

void UKF::ProcessMeasurement(MeasurementPackage meas_package)
{
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR && !use_radar_)
        return;

    if (meas_package.sensor_type_ == MeasurementPackage::LASER && !use_laser_)
        return;

    std::cout << "ProcessMeasurment" << std::endl;
    if (!is_initialized_)
    {
        is_initialized_ = true;
        //std::cout << "Raw measurement: " << meas_package.raw_measurements_ << "\n";
        SetStateFromMeasurement(meas_package.raw_measurements_, meas_package.sensor_type_);
        return;
    }
    
    static double lastTime = 0.0;
    double dt = meas_package.timestamp_ - lastTime;
    lastTime = meas_package.timestamp_;

    PredictionData p = Prediction(dt);
    UpdateState(meas_package, p);
}

// this function also takes into account process noise when generating the sigma points
std::vector<Eigen::VectorXd> UKF::GenerateAugmentedSigmaPoints(Eigen::VectorXd _x, Eigen::MatrixXd _P) const
{
    std::cout << __FUNCTION__ << std::endl;

    // create augmented mean vector
    VectorXd x_aug = VectorXd(n_aug_);

    // create augmented state covariance
    MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

    // create augmented mean state
    x_aug.head(5) = _x;
    x_aug(5) = 0;
    x_aug(6) = 0;

    // create augmented covariance matrix
    P_aug.fill(0.0);
    P_aug.topLeftCorner(5,5) = _P;
    P_aug(5,5) = std_a_*std_a_;
    P_aug(6,6) = std_yawdd_*std_yawdd_;

    // create square root matrix
    MatrixXd L = P_aug.llt().matrixL();

    // create augmented sigma points
    std::vector<Eigen::VectorXd> sigmaPts;
    sigmaPts.push_back(x_aug);
    for (int i = 0; i< n_aug_; ++i)
    {
        sigmaPts.push_back(x_aug + sqrt(lambda_ + n_aug_) * L.col(i));
        sigmaPts.push_back(x_aug - sqrt(lambda_ + n_aug_) * L.col(i));
    }
    
    return sigmaPts;
}


std::vector<Eigen::VectorXd> UKF::GenerateSigmaPoints(Eigen::VectorXd _x, Eigen::MatrixXd _P) const
{
    // predicted sigma points matrix
    std::vector<Eigen::VectorXd> sigmaPts;

    // define spreading parameter
    double lambda = 3 - n_x_;

    // calculate square root of P
    MatrixXd A = _P.llt().matrixL();

    // set first sigma point to the state vector
    sigmaPts.push_back(_x);

    // set remaining sigma points
    for (int i = 0; i < n_x_; ++i)
    {
        sigmaPts.push_back(_x + sqrt(lambda+n_x_) * A.col(i));
        sigmaPts.push_back(_x - sqrt(lambda+n_x_) * A.col(i));
    }

    return sigmaPts;
}


std::tuple<VectorXd, MatrixXd> UKF::PredictMeanAndCovariance(const std::vector<Eigen::VectorXd>& predictedSigmaPoints) const
{ 
    std::cout << __FUNCTION__ << std::endl;    
    
    // predicted state mean
    VectorXd x = VectorXd(n_x_);
    x.fill(0.0);
    // iterate over sigma points
    for (int i = 0; i < 2 * n_aug_ + 1; ++i) 
        x = x + weights_[i] * predictedSigmaPoints[i];

    // predicted state covariance matrix
    MatrixXd P = MatrixXd(n_x_, n_x_);
    P.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; ++i)
    {  // iterate over sigma points
        // state difference
        VectorXd x_diff = predictedSigmaPoints[i] - x;
        // angle normalization
        NormalizeAngle(x_diff(3));
        P = P + weights_[i] * x_diff * x_diff.transpose();
    }

    std::cout << __FUNCTION__ << " EXIT "<< std::endl;
    
    return std::make_tuple(x, P);
}


void UKF::UpdateLidar(MeasurementPackage meas_package)
{
    std::cout << __FUNCTION__ << std::endl;
    /**
     * TODO: Complete this function! Use lidar data to update the belief 
     * about the object's position. Modify the state vector, x_, and 
     * covariance, P_.
     * You can also calculate the lidar NIS, if desired.
     */

    // (1) Compute the predicted measurement from the last state based on the model
    

    // (2) Update the state estimate
}

