#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 09:48:12 2020

@author: gustavolibotte
"""

import pandas as pd  # data processing
from proj_consts import ProjectConsts

class LoadData:
    
    @staticmethod
    def getBrazilDataFrame(min_confirmed: int = 5, store: bool = False) -> pd.DataFrame:
        """
        Get updated data on the epidemic in Brazil of all states by day.
        
        Source: Número de casos confirmados de COVID-19 no Brasil (on GitHub).
        https://raw.githubusercontent.com/wcota/covid19br/master/cases-brazil-states.csv
        
        Parameters
        ----------
        min_confirmed: int
            Minimum number of confirmed cases in time series.
        store: bool
            Determines whether data obtained from web is saved on the hard drive.
            
        Return
        ------
        pd.DataFrame
            COVID-19 cumulative data (confirmed, dead, recovered) in Brazil per day for all states.
        """
        df_brazil_states_cases = pd.read_csv(
            ProjectConsts.CASES_BRAZIL_STATES_URL,
            usecols=["date", "state", "totalCases", "deaths", "recovered"],
            parse_dates=["date"],
        )

        df_brazil_states_cases["date"] = pd.to_datetime(df_brazil_states_cases["date"]).dt.date
        df_brazil_states_cases.fillna(value={"recovered": 0}, inplace=True, downcast="int64")
        columns_rename = {"totalCases": "confirmed", "deaths": "dead"}
        df_brazil_states_cases.rename(columns=columns_rename, inplace=True)
    
        df_brazil_states_cases = df_brazil_states_cases[df_brazil_states_cases.confirmed > min_confirmed]
        df_brazil_states_cases["day"] = df_brazil_states_cases.date.apply(
                lambda x: (x - df_brazil_states_cases.date.min()).days
        )
        df_brazil_states_cases = df_brazil_states_cases.reset_index(drop=True)
        if store == True:
            last_date = str(df_brazil_states_cases.date.iloc[-1])
            df_brazil_states_cases.to_csv(f"{ProjectConsts.DATA_PATH}/" + last_date + "_Brazil_by_day.csv", index=False)
        return df_brazil_states_cases
    
    @staticmethod
    def getBrazilStateDataFrame(df_brazil_states_cases: pd.DataFrame, state_name: str) -> pd.DataFrame:
        """
        Get updated data on the epidemic of a specific state by day
        
        Parameters
        ----------
        df_brazil_states_cases: pd.DataFrame
            DataFrame containing information of all states
        state_name: str
            State initials (e.g., RJ)
        
        Return
        ------
        pd.DataFrame
            COVID-19 cumulative data (confirmed, dead and recovered) in Brazil per day for a specific state
        """
        df_brazil_states_cases = df_brazil_states_cases[df_brazil_states_cases["state"] == state_name]
        df_brazil_states_cases = df_brazil_states_cases.drop(columns = ["state"])
        df_brazil_states_cases = df_brazil_states_cases.reset_index(drop=True)
        return df_brazil_states_cases
    
    @staticmethod
    def getBrazilStateCityDataFrame(state_name: str, store: bool = False) -> pd.DataFrame:
        """
        Get updated data on the epidemic of all cities in a specific state by day
        
        Source: Número de casos confirmados de COVID-19 no Brasil (on GitHub)
        https://raw.githubusercontent.com/wcota/covid19br/master/cases-brazil-cities-time.csv
        
        Parameters
        ----------
        state_name: str
            State initials (e.g., RJ)
        store: bool
            Determines whether data obtained from web is saved on the hard drive.
        
        Return
        ------
        pd.DataFrame
            COVID-19 cumulative data (confirmed and dead) in Brazil per day for all cities of a specific state
        """
        df_state_city_cases = pd.read_csv(ProjectConsts.CASES_BRAZIL_CITIES_URL,
            usecols=["date", "state", "city", "totalCases", "deaths"],
            parse_dates=["date"],
        )

        df_state_city_cases["date"] = pd.to_datetime(df_state_city_cases["date"]).dt.date
        df_state_city_cases = df_state_city_cases[df_state_city_cases["state"] == state_name]
        df_state_city_cases["day"] = df_state_city_cases.date.apply(
            lambda x: (x - df_state_city_cases.date.min()).days
        )
        column_names = {"totalCases": "confirmed", "deaths": "dead"}
        df_state_city_cases = df_state_city_cases.rename(columns=column_names)
        df_state_city_cases = df_state_city_cases[["date", "state", "city", "dead", "confirmed", "day"]]
        df_state_city_cases = df_state_city_cases.reset_index(drop=True)
        if store == True:
            last_date = str(df_state_city_cases.date.iloc[-1])
            df_state_city_cases.to_csv(f"{ProjectConsts.DATA_PATH}/"  + last_date + "_" + state_name + "_cities_by_day.csv", index=False)
        return df_state_city_cases
    
    @staticmethod
    def getBrazilCityDataFrame(df_state_city_cases: pd.DataFrame, city_name: str) -> pd.DataFrame:
        """
        Get updated data on the epidemic of a specific city by day. There is no recovered data within the cities.
        
        Parameters
        ----------
        df_state_city_cases: pd.DataFrame
            DataFrame containing information of all cities in a specific state
        city_name: str
            City name (e.g., Petrópolis/RJ)
        
        Return
        ------
        pd.DataFrame
            COVID-19 cumulative data (confirmed and dead) in Brazil per day for a specific city
        """
        df_city_cases = df_state_city_cases[df_state_city_cases["city"] == city_name]
        df_city_cases = df_city_cases.drop(columns = ["state", "city"])
        df_city_cases = df_city_cases.reset_index(drop=True)
        return df_city_cases
    