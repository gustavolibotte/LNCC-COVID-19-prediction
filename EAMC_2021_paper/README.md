# EAMC_2021_paper :chart_with_upwards_trend: :page_facing_up:

Dataset used was retrieved from [Kaggle](https://www.kaggle.com/imdevskp/corona-virus-report)

## Organization

The repository is divided into two folders:

### Data

Contains the data used for the simulations/evaluation of our work. Inside it, there is only one [file](https://github.com/gustavolibotte/LNCC-COVID-19-prediction/blob/master/EAMC_2021_paper/data/population_by_country_2020.csv), corresponding to the dataset of population sizes throughout the world. We used China's population as a range of the uniform distribution used as prior for fitting.

### Main

The main folder contains all necessary code for reproduction of our work and supplementary analysis on the correlation between epidemiological parameters considered by the SEIRD model.

* .csv files :file_folder:

One .csv files for the complete dataset of daily registration of infections, deaths and recoveries in China ([covid_19_clean_complete.csv](https://github.com/gustavolibotte/LNCC-COVID-19-prediction/blob/master/EAMC_2021_paper/main/covid_19_clean_complete.csv)) and another .csv with the final posterior for all parameters in all 5 considerations proposed in our work ([CorrelParams.csv](https://github.com/gustavolibotte/LNCC-COVID-19-prediction/blob/master/EAMC_2021_paper/main/CorrelParams.csv)). The first 20 rows of the csv correspond to the situation I mentioned in the article, rows 21 to 40 correspond to situation II, and so on.


* Python code :snake:

```python
def function(x):
  return something
```

* [Jupyter notebook](https://github.com/gustavolibotte/LNCC-COVID-19-prediction/blob/master/EAMC_2021_paper/main/Correlation_Param.ipynb) of Correlation between Parameters :notebook:

A small jupyter notebook with supplementary analysis of our work, regarding the correlation between each epidemiological parameter used in the SEIRD model. The final result of the notebook is the correlation matrix given bellow.

Correlation Matrix: 
![alt text][logo]

[logo]: https://github.com/gustavolibotte/LNCC-COVID-19-prediction/blob/master/EAMC_2021_paper/main/Correl_Params.png "Correlation Matrix"

Some other stuff here (VAVA ADD WHATEVER YOU WANT)
