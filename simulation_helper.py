import geopandas as gpd
import numpy as np
from scipy.stats import weibull_min, chi2
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import expon, pareto
from scipy.stats import gamma

def get_fire_data(greater_than_100=False):
    # Read the fire data
    gdf = gpd.read_file("nfdb/NFDB_point_20250519.shp")

    # read in the historical fire management zones
    zones = gpd.read_file("historical_fire_zones\Historical_Fire_Management_Zones.shp")

    # Ensure both GeoDataFrames use the same CRS
    gdf = gdf.to_crs(zones.crs)

    # join the two GeoDataFrames
    data = gpd.sjoin(gdf, zones, how="inner", predicate="within")

    # drop the data in FMZ_DESIGN == 'Parks Zone'
    data = data[data.FMZ_DESIGN != "Parks Zone"]

    data.NFDBFIREID.duplicated().sum()

    # show me the rows where NFDBFIREID is duplicated
    duplicates = data[data.NFDBFIREID.duplicated(keep=False)]

    # remove the duplicates
    data = data[~data.NFDBFIREID.duplicated(keep="first")]

    data = data[(data.CAUSE == "N") | (data.CAUSE == "U")]

    # drop rows where the REP_DATE is null
    joined = data[data.REP_DATE.notnull()]

    # make a new column for whether the SIZE_HA is greater than 100\
    joined["SIZE_HA_100"] = joined["SIZE_HA"].apply(lambda x: True if x > 100 else False)

    # remove the rows with YEAR before 1975
    joined = joined[joined.YEAR >= 1976]

    # make dictionary to change values of FMZ_DESIGN
    fmz_dict = {
        "Hudson Bay Zone": "Extensive",
        "Great Lakes/St. Lawrence Zone": "Intensive Measured",
        "Boreal Zone": "Intensive Measured",
        "Northern Boreal Zone": "Intensive Measured"
    }

    # make a new column in the joined GeoDataFrame
    joined["FMZ_ZONE"] = joined["FMZ_DESIGN"].map(fmz_dict)

    season_dict = {1: "Winter",
                2: "Winter",
                3: "Spring",
                    4: "Spring",
                    5: "Spring",
                    6: "Summer",
                    7: "Summer",
                    8: "Summer",
                    9: "Fall",
                    10: "Fall",
                    11: "Fall",
                    12: "Winter"}

    # make a new column for the season based on the column called "MONTH"
    joined["SEASON"] = joined["MONTH"].map(season_dict)


    # make a new column based on the column year which is whether it is after 2006 or not
    def after_2005(year):
        if year > 2005:
            return "After 2005"
        else:
            return "Before 2005"

    joined["AFTER_2005"] = joined["YEAR"].apply(after_2005)

    joined = joined[['YEAR', 'REP_DATE', 'SIZE_HA', 'FMZ_ZONE', 'SEASON', 'AFTER_2005', 'SIZE_HA_100']]

    # split into two by the FMZ_ZONE
    extensive = joined[joined.FMZ_ZONE == "Extensive"]
    intensive = joined[joined.FMZ_ZONE == "Intensive Measured"]

    # if greater_than_100 is True, return only the fires greater than 100 ha
    if greater_than_100:
        extensive = extensive[extensive.SIZE_HA > 100]
        intensive = intensive[intensive.SIZE_HA > 100]
        joined = joined[joined.SIZE_HA > 100]

    return joined, intensive, extensive

def fit_fire_size_distribution(data):
    # get the average fire size to fit an expnential distribution
    exp_mean = data.mean()

    # weibull parameters
    shape, loc, scale = weibull_min.fit(data, floc=0)

    # pareto parameters
    shape_pareto, loc_pareto, scale_pareto = pareto.fit(data, floc=0)

    # return the paremeters
    return {
        "exponential": exp_mean,
        "weibull": (shape, loc, scale),
        "pareto": (shape_pareto, loc_pareto, scale_pareto)
    }

def fit_cox_process(data):
    # Fit a gamma distribution to the number of large fires
    shape_gamma, loc_gamma, scale_gamma = gamma.fit(data, floc=0)

    # fit an exponential distribution to the inter-event times
    exp_mean = data.mean()

    return {
        "gamma": (shape_gamma, loc_gamma, scale_gamma),
        "exponential": exp_mean
    }

def generate_samples(parameters, size=1000):
    # Generate samples from the fitted distributions cox process
    sim_lambdas = {"gamma": gamma.rvs(*parameters["gamma"], size=size),
               "exponential": expon.rvs(scale=parameters["exponential"], size=size)}
        

    #use simulated_lambas to generate samples using poisson process
    samples = {}
    for key, lambdas in sim_lambdas.items():
        samples[key] = []
        for lam in lambdas:
            # Generate a sample from a Poisson process with rate lam
            sample = np.random.poisson(lam)
            samples[key].append(sample)
    return samples

def generate_fire_area(number_of_fires, parameters):
    # Generate fire areas based on the fitted distributions
    fire_areas = {}
    for key, param in parameters.items():
        if key == "exponential":
            fire_areas[key] = expon.rvs(scale=param, size=number_of_fires).sum()
        elif key == "weibull":
            fire_areas[key] = weibull_min.rvs(*param, size=number_of_fires).sum()
        elif key == "pareto":
            fire_areas[key] = pareto.rvs(*param, size=number_of_fires).sum()
    return fire_areas