# Sequential time-window learning with approximate Bayesian computation: an application to epidemic forecasting

This branch contains all the necessary code for reproducing the results presented in the work "Sequential time-window learning with approximate Bayesian
computation: an application to epidemic forecasting" by Valeriano et al.

---

In the **data** folder, there is the data on COVID-19 in multiple countries, collected from [Our World in Data](https://github.com/owid/covid-19-data/tree/master/public/data), and the data on population size of countries, used to define the prior distribution of the effective population size parameter of the epidemic model.

In the **main** folder, there is all the necessary code, both for generating the results of the inference via ABC-SMC algorithm, and for analyzing such results and producing the figures present in the paper. The ABC-SMC code can be executed serially on in parallel, with MPI.

In the **logs** folder, there is enough results, generated with the given code, for running the analysis script without needing to generate the ABC-SMC inference results again, if a user is interested in that.