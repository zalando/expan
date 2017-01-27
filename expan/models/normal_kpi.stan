data {
	int<lower=0> Nc; 	// number of entities in the control group
	int<lower=0> Nt; 	// number of entities in the treatment group
	real x[Nc]; 		// normally distributed KPI in the control group
	real y[Nt]; 		// normally distributed KPI in the treatment group
}

parameters {
	real mu;			// population mean
	real<lower=0> sigma;// population variance
	real delta;
}

transformed parameters {
	real alpha;			// total effect size
	alpha = delta * sigma;
}

model {
	delta ~ cauchy(0, 1);
	mu ~ cauchy(0, 1);
	x ~ normal(mu-alpha/2, sigma);
	y ~ normal(mu+alpha/2, sigma);
}

