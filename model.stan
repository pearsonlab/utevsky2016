data {
  int<lower=1> N;  # number of trials
  int<lower=1> NU;  # number of units
  int<lower=1> NS;  # number of stims
  int<lower=0> P;  # number of pre-specified regressors
  int<lower=1> K;  # number of latent features

  int<lower=1> unit[N];  # stimulus index for each trial
  int<lower=1> stim[N];  # stimulus index for each trial
  int<lower=0> count[N];  # spike count for each trial
  matrix[N, P] X;  # matrix of pre-specified regressors
}
transformed data {
}
parameters {
  # baselines
  vector[NU] A;

  # regression coefficients
  matrix[NU, P] B;

  # latent state responses
  matrix [NU, K] C;
  vector<lower=0>[K] tau;  # global variance
  matrix<lower=0>[NU, K] lambda;  # local variance

  # latent states
  # (these are actually continuous, as would result from marginalizing over Z;
  # this should really be thought of as a matrix of weights w[k, s] in [0, 1])
  vector<lower=0, upper=1>[K] delta;  # differential probability of each column appearing
  matrix<lower=0, upper=1>[K, NS] Z;

  # overdispersion
  real<lower=0> sig;

  # overdispersion (per trial)
  real eps;
}
transformed parameters {
  vector<lower=0, upper=1>[K] pi;  # probability of each column appearing

  # stick breaking construction for ibp
  for (k in 1:K) {
    if (k > 1) {
      pi[k] = delta[k];
    } else {
      pi[k] = pi[k - 1] * delta[k];
    }
  }
}
model {
  A ~ normal(0, 2);  # equivalent to mean baseline ~15 Hz

  for (u in 1:NU) {
    B[u] ~ normal(0, 1);
  }

  for (k in 1:K)
    tau[k] ~ cauchy(0, sig);

  for (u in 1:NU) {
    for (k in 1:K) {
      lambda[u, k] ~ cauchy(0, 0.1);
      C[u, k] ~ normal(0, tau[k] * lambda[u, k]);
    }
  }

  delta ~ beta(3, 1);
  for (k in 1:K) {
    for (s in 1:NS)
      Z[k, s] ~ bernoulli(pi[k]);
  }

  sig ~ cauchy(0, 0.1);

  for (i in 1:N) {
    real eta;
    eps ~ normal(0, sig);
    eta = A[unit[i]] + B[unit[i], :] * X[i, :]' + C[unit[i], :] * Z[:, stim[i]] + eps;
    count[i] ~ poisson_log(eta);
  }


}
