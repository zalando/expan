data {
	int<lower=0> Nc; 	// number of entities in the control group
	int<lower=0> Nt; 	// number of entities in the treatment group
	int<lower=0> y[Nc]; 		// KPI in the control group
	int<lower=0> x[Nt]; 		// KPI in the treatment group
}

parameters {
	real<lower=0> lambda;			
	real<lower=-lambda> delta;
}

transformed parameters {
	//real delta;			// absolute effect size
	//alpha = lambda_t - lambda;
}

model {
	delta ~ cauchy(0, 1);
	lambda ~ gamma(2, 2);
	x ~ poisson(lambda+delta);
	y ~ poisson(lambda);
}

