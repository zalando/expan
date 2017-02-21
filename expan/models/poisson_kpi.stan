data {
	int<lower=0> Nc; 	// number of entities in the control group
	int<lower=0> Nt; 	// number of entities in the treatment group
	int<lower=0> x[Nc]; 		// KPI in the control group
	int<lower=0> y[Nt]; 		// KPI in the treatment group
}

parameters {
	real<lower=0> lambda;			
	real delta;
}

transformed parameters {
	real alpha;			// absolute effect size
	alpha = delta * sqrt(lambda);
}

model {
	delta ~ cauchy(0, 1);
	x ~ poisson(lambda);
	y ~ poisson(lambda+alpha);
}

