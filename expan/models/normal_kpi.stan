data {
	int<lower=0> Nc; 	// number of entities in the control group
	int<lower=0> Nt; 	// number of entities in the treatment group
	real y[Nc]; 		// normally distributed KPI in the control group
	real x[Nt]; 		// normally distributed KPI in the treatment group
}

parameters {
	real mu;			// population mean
	real<lower=0> sigma;// population variance
	real alpha;         // normalized version of delta
}

transformed parameters {
	real delta;			// absolute difference of mean
	delta = alpha * sigma;
}

model {
	alpha ~ cauchy(0, 1);
	mu ~ cauchy(0, 1);
	sigma ~ gamma(2, 2);
	x ~ normal(mu+delta, sigma);
	y ~ normal(mu, sigma);
}

