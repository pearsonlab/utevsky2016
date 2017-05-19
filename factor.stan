/*
Fit count data by factoring log firing rate as W * V^*
with W local-global sparse. Columns of W are successively sparser.
V uses the same prior
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
  # log firing rate matrix factors
  matrix[S, K] W_raw;  # stimulus feature loading matrix
  matrix[U, K] V_raw;  # neuron response matrix

  # global columnwise variance components
  vector<lower=0>[K] tau_W;
  vector<lower=0>[K] tau_V;

  # elementwise local variance component
  matrix<lower=0>[S, K] phi_W;
  matrix<lower=0>[U, K] phi_V;
}

transformed parameters {
  matrix[S, K] W;  # stimulus feature loading matrix
  matrix[U, K] V;  # neuron response matrix

  # log firing rates
  matrix[S, U] eta;

  for (k in 1:K) {
    W[:, k] = tau_W[k] * phi_W[:, k] .* W_raw[:, k];
    V[:, k] = tau_V[k] * phi_V[:, k] .* V_raw[:, k];
  }

  eta = W * V';
}

model {
  for (k in 1:K) {
    tau_W[k] ~ cauchy(0, 1);
    tau_V[k] ~ cauchy(0, 1);
    for (s in 1:S) {
      W_raw[s, k] ~ normal(0, 1);
      phi_W[s, k] ~ cauchy(0, 1);
    }
    for (u in 1:U) {
      V_raw[u, k] ~ normal(0., 1.);
      phi_V[u, k] ~ cauchy(0, 1);
    }
  }

  for (i in 1:N) {
    count[i] ~ poisson_log(eta[stim[i], unit[i]]);
  }
}
