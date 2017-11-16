/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */
#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  num_particles = 10;

  // This lines creates a normal (Gaussian) distribution
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  particles.clear();
  for (int i = 0; i < num_particles; ++i) {
    Particle p = {i, dist_x(gen), dist_y(gen), dist_theta(gen), 1};
    particles.push_back(p);
  }

  is_initialized = true;
}



void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // This lines creates a normal (Gaussian) distribution
  normal_distribution<double> noise_x(0, std_pos[0]);
  normal_distribution<double> noise_y(0, std_pos[1]);
  normal_distribution<double> noise_theta(0, std_pos[2]);

  for(int i=0; i<particles.size(); i++){
    double nx = noise_x(gen);
    double ny = noise_y(gen);
    double nt = noise_theta(gen);

    // Prediction if yaw rate is not 0
    if( fabs(yaw_rate) > 0.001 ){
      double theta_new = particles[i].theta + (yaw_rate*delta_t);
      double vel_yr = velocity/yaw_rate;
      particles[i].x += vel_yr*(sin(theta_new)-sin(particles[i].theta)) + nx;
      particles[i].y += vel_yr*(cos(particles[i].theta)-cos(theta_new)) + ny;
      particles[i].theta = theta_new + nt;
    } else {
      double vel_dt = velocity*delta_t;
      particles[i].x += vel_dt*cos(particles[i].theta) + nx;
      particles[i].y += vel_dt*sin(particles[i].theta) + ny;
      particles[i].theta += nt;
    }
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {

  for(int i=0; i<observations.size(); i++){
    double closest_dist = 99999;
    double new_dist  = 99999;

    // Check if there is a closer landmark
    for(int j=0; j<predicted.size(); j++){
      new_dist = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y );

      if( new_dist < closest_dist){
        closest_dist = new_dist;
        observations[i].id = predicted[j].id;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {

  // This lines creates a normal (Gaussian) distribution
  normal_distribution<double> noise_lm_x(0, std_landmark[0]);
  normal_distribution<double> noise_lm_y(0, std_landmark[1]);

  weights.clear();
  for(int i=0; i<particles.size(); i++){
    double Xp = particles[i].x;
    double Yp = particles[i].y;
    double Tp = particles[i].theta;

    // Get landmarks in range of sensor
    std::vector<LandmarkObs> predicted;
    for(int j=0; j<map_landmarks.landmark_list.size(); j++){
      double Xm = map_landmarks.landmark_list[j].x_f;
      double Ym = map_landmarks.landmark_list[j].y_f;
      if( dist(Xp, Yp, Xm, Ym) <= sensor_range){
        LandmarkObs obs;
        obs.id = map_landmarks.landmark_list[j].id_i;
        obs.x = Xm;
        obs.y = Ym;
        predicted.push_back(obs);
      }
    }

    // Transform observations to Map space
    std::vector<LandmarkObs> transformed_observations;
    for(int j=0; j<observations.size(); j++){
      LandmarkObs obs;
      obs.id = 0;
      obs.x = Xp + (cos(Tp)*observations[j].x)-(sin(Tp)*observations[j].y) + noise_lm_x(gen);
      obs.y = Yp + (sin(Tp)*observations[j].x)+(cos(Tp)*observations[j].y) + noise_lm_y(gen);
      transformed_observations.push_back(obs);
    }

    // Associate observation to landmark
    if( predicted.size()>0 &&  transformed_observations.size()>0){
      dataAssociation(predicted, transformed_observations);
    }


    // Debug draw lines
    std::vector<int> associations;
    std::vector<double> sense_x;
    std::vector<double> sense_y;
    for(int j=0; j<transformed_observations.size(); j++){
      associations.push_back( transformed_observations[j].id );
      sense_x.push_back( transformed_observations[j].x );
      sense_y.push_back( transformed_observations[j].y );
    }
    SetAssociations(particles[i], associations, sense_x, sense_y);

    // Update Weight
    particles[i].weight = 1;
    for(int j=0; j<transformed_observations.size(); j++){
      int Ox = transformed_observations[j].x;
      int Oy = transformed_observations[j].y;

      int Mx = 0;
      int My = 0;
      for(int k=0; k<predicted.size(); k++){
        if( predicted[k].id == transformed_observations[j].id){
          Mx = predicted[k].x;
          My = predicted[k].y;
        }
      }

      // Multivariate-Gaussian probability
      double gauss_norm = 1/(2*M_PI*std_landmark[0]*std_landmark[1]);
      double exponent = pow(M_E, -( pow(Ox-Mx,2)/(2*pow(std_landmark[0],2)) + pow(Oy-My,2)/(2*pow(std_landmark[1],2)) ));
      particles[i].weight *=  gauss_norm * exponent;
    }

    weights.push_back( particles[i].weight );
  }
}

void ParticleFilter::resample() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::discrete_distribution<> d(weights.begin(), weights.end());

  std::vector<Particle> resampled_particles;
  for(int n=0; n<num_particles; ++n) {
    resampled_particles.push_back( particles[d(gen)] );
  }

  particles = resampled_particles;
}

void ParticleFilter::SetAssociations(Particle &particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
