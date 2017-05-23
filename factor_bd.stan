/*
Fit count data by factoring log firing rate as W * V^*
with W local-global sparse. Columns of W are successively sparser.
V uses the same prior
Follows Bhattacharya and Dunson (2011)
*/

data {
  int<lower=1> N;  # number of observations
  int<lower=1> S;  # number of stims
  int<lower=1> U;  # number of units
  int<lower=1> K;  # number of features

  int<lower=1> unit[N];  # stimulus index for each trial
  int<lower=1> stim[N];  # stimulus index for each trial
  int<lower=0> count[N];  # spike count for each trial
}

parameters {
  # baseline firing rates
  vector[U] b;

  # log firing rate matrix factors
  matrix<lower=0>[S, K] W_raw;  # stimulus feature loading matrix
  matrix[U, K] V_raw;  # neuron response matrix

  # elementwise local precision component
  matrix<lower=0>[S, K] phi_W;
  matrix<lower=0>[U, K] phi_V;

  # stick lengths for global precision component
  vector<lower=0>[K] delta_W;
  vector<lower=0>[K] delta_V;
}

transformed parameters {
  # global columnwise precision components
  vector<lower=0>[K] tau_W;
  vector<lower=0>[K] tau_V;

  matrix<lower=0>[S, K] W;  # stimulus feature loading matrix
  matrix[U, K] V;  # neuron response matrix

  # log firing rates
  matrix[S, U] eta;
  matrix[S, U] lambda;

  tau_W = exp(cumulative_sum(log(delta_W)));
  tau_V = exp(cumulative_sum(log(delta_V)));

  for (k in 1:K) {
    W[:, k] = W_raw[:, k] ./ (tau_W[k] * phi_W[:, k]);
    V[:, k] = V_raw[:, k] ./ (tau_V[k] * phi_V[:, k]);
  }

  eta = W * V';
  for (s in 1:S) {
    for (u in 1:U) {
      lambda[s, u] = b[u] + eta[s, u];
    }
  }
}

model {
  b ~ normal(-1, 1);

  for (k in 1:K) {
    delta_W[k] ~ gamma(1.5, 1.);
    delta_V[k] ~ gamma(1.5, 1.);
    for (s in 1:S) {
      W_raw[s, k] ~ normal(0, 1);
      phi_W[s, k] ~ gamma(3/2, 3/2);
    }
    for (u in 1:U) {
      V_raw[u, k] ~ normal(0, 1);
      phi_V[u, k] ~ gamma(3/2, 3/2);
    }
  }

  for (i in 1:N) {
    count[i] ~ poisson_log(lambda[stim[i], unit[i]]);
  }
}
