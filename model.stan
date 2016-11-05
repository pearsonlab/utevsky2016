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
  
  # lambda has a horseshoe prior:
  # p(x|t) ~ \int d\phi N(0, t * phi)
  # phi ~ Ca+(0, 1);
  matrix<lower=0>[NU, K] phi;  
  
  # latent states 
  # (these are actually continuous, as would result from marginalizing over Z;
  # this should really be thought of as a matrix of weights w[k, s] in [0, 1])
  matrix<lower=0, upper=1>[K, NS] Z;
  
  # overdispersion (per unit)
  real<lower=0> sig[NU];
  
  # overdispersion (per trial)
  real eps[N]; 
}
transformed parameters {
  real eta[N];
  
  for (i in 1:N)
    eta[i] = A[unit[i]] + B[unit[i], :] * X[i, :]' + C[unit[i], :] * Z[:, stim[i]] + eps[i];
}
model {
  A ~ normal(2.5, 1);  # equivalent to mean baseline ~15 Hz
  
  for (u in 1:NU) {
    B[u] ~ normal(0, 1);
  }
  
  for (k in 1:K) 
    tau[k] ~ cauchy(0, .1);
    
  for (u in 1:NU) {
    for (k in 1:K) {
      lambda[u, k] ~ cauchy(0, .1);
      C[u, k] ~ normal(0, tau[k] * lambda[u, k]);
    }
  }
  
  for (k in 1:K) {
    Z[k] ~ beta(1, 100);
  }
    
  for (u in 1:NU)
    sig[u] ~ cauchy(0, 0.1);
    
  for (i in 1:N) {
    eps[i] ~ normal(0, 1); 
    count[i] ~ poisson_log(eta[i] + sig[unit[i]] * eps[i]);
  }
  
  
}
