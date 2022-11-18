import pandas
import numpy
from statsmodels.api import OLS,add_constant

cbsa_hom = []
years = range(2010,2021)
for year in years:
    year = str(int(year))
    race_eth = pandas.read_csv('/home/andrewstier/Downloads/census_all_tracts/ACSDT5Y%s.B02001_data_with_overlays_2022-06-16T001037.csv' % year)
    total_column = 'B02001_001E'
    white_column = 'B02001_002E'
    black_column = 'B02001_003E'
    native_column = 'B02001_004E'
    asian_column = 'B02001_005E'
    island_column = 'B02001_006E'
    other_column = 'B02001_007E'
    eth_cols = [white_column,black_column,native_column,asian_column,island_column,other_column]
    geo_colummn = 'GEO_ID'
    race_eth = race_eth[race_eth[geo_colummn].str.contains('US')]
    race_eth['county'] = race_eth[geo_colummn].map(lambda x: x.split('US')[1][:5])

    cbsas = pandas.read_csv('/home/andrewstier/Downloads/census_all_tracts/delineation_2020.csv',skiprows=2)
    cbsas['FIPS State Code'] = cbsas['FIPS State Code'].astype(str).map(lambda x: x if len(x)==2 else '0'+x)
    cbsas['FIPS County Code'] = cbsas['FIPS County Code'].astype(str).map(lambda x: x if len(x)==3 else ('0'+ x) if len(x)==2 else '00'+x)
    cbsas['county'] =  cbsas['FIPS State Code'].astype(str) + cbsas['FIPS County Code'].astype(str)
    cbsa_codes = numpy.unique(cbsas['CBSA Code'])

    cbsa_pop = pandas.read_csv('/home/andrewstier/Downloads/census_all_tracts/ACSDT5Y%s.B01003_data_with_overlays_2022-07-18T054021.csv' % year)
    cbsa_pop = cbsa_pop[cbsa_pop[geo_colummn].str.contains('US')]
    cbsa_pop['cbsa_code'] = cbsa_pop[geo_colummn].map(lambda x: x.split('US')[1])

    income = pandas.read_csv('/home/andrewstier/Downloads/census_all_tracts/ACSDT5Y%s.B19013_data_with_overlays_2022-07-18T053702.csv' % year)
    income = income[income[geo_colummn].str.contains('US')]
    income['cbsa_code'] = income[geo_colummn].map(lambda x: x.split('US')[1])

    gdp_macro = pandas.read_csv('/home/andrewstier/Downloads/census_all_tracts/bea_gdp_cbsa.csv',skiprows=4)
    gdp_micro = pandas.read_csv('/home/andrewstier/Downloads/census_all_tracts/bea_gdp_micro.csv', skiprows=4)
    gdp = pandas.concat([gdp_macro,gdp_micro])
    gdp['cbsa_code'] = gdp['GeoFips'].astype(int).astype(str)
    gdp = gdp[['cbsa_code',year]]

    joined_race_eth = race_eth.set_index('county').join(cbsas.set_index('county')).reset_index()

    c=0
    for cbsa in cbsa_codes:
        cbsa_race_eth = joined_race_eth[joined_race_eth['CBSA Code']==cbsa]
        ratios = numpy.array([
            cbsa_race_eth[x].astype(float).sum() / cbsa_race_eth[total_column].astype(float).sum()
            for x in eth_cols
        ])
        tot = cbsa_race_eth[total_column].astype(float)
        tot = tot.map(lambda x: x if x>0 else numpy.nan)
        local_ratio = [
            cbsa_race_eth[x].astype(float) / tot for x in eth_cols
        ]
        homophily = []
        seggregation_index = []
        gini_index = []
        inter_exposure = []
        for i in range(len(eth_cols)):
            homophily.append(numpy.abs(
                local_ratio[i]-ratios[i]
            ).mean())
            seggregation_index.append(numpy.nansum(numpy.abs(
                (local_ratio[i] - ratios[i]) * tot
            )) / (
                                     2 * numpy.nansum(tot) * ratios[i] * (1 - ratios[i])
                             )
            )
            gini_index.append(
              numpy.nansum([
                  [numpy.abs(local_ratio[i].values[j]-local_ratio[i].values[k])*tot.values[j]*tot.values[k]
                for j in range(len(local_ratio[i]))] for k in range(len(local_ratio[i]))])/
              (2 * numpy.nansum(tot)**2 * ratios[i] * (1 - ratios[i]))
            )
            inter_exposure.append(numpy.nansum(tot*local_ratio[i]**2)
                                  /(numpy.nansum(local_ratio[i]*tot)*(1-ratios[i]))
                                  -ratios[i]/(1-ratios[i]))

        correction = 1
        hom_correction = 0
        for i in range(len(homophily)-1):
            hom_correction += ratios[i]**2*(homophily[i] if ~numpy.isnan(homophily[i]) else 0)
            for j in range(i+1,len(homophily)):
                correction -= ratios[i]*ratios[j]*((homophily[i] if ~numpy.isnan(homophily[i]) else 0)+(homophily[j] if ~numpy.isnan(homophily[j]) else 0))
        correction_seg_idx = 1
        hom_correction_seg_idx = 0
        for i in range(len(seggregation_index) - 1):
            hom_correction_seg_idx += ratios[i] ** 2 * (seggregation_index[i] if ~numpy.isnan(seggregation_index[i]) else 0)
            for j in range(i + 1, len(seggregation_index)):
                correction_seg_idx -= ratios[i] * ratios[j] * ((seggregation_index[i] if ~numpy.isnan(seggregation_index[i]) else 0) +
                                                               (
                    seggregation_index[j] if ~numpy.isnan(seggregation_index[j]) else 0))
        correction_gini = 1
        hom_correction_gini = 0
        for i in range(len(gini_index) - 1):
            hom_correction_gini += ratios[i] ** 2 * (
                gini_index[i] if ~numpy.isnan(gini_index[i]) else 0)
            for j in range(i + 1, len(gini_index)):
                correction_gini -= ratios[i] * ratios[j] * (
                            (gini_index[i] if ~numpy.isnan(gini_index[i]) else 0) +
                            (
                                gini_index[j] if ~numpy.isnan(gini_index[j]) else 0))
        correction_exp = 1
        hom_correction_exp = 0
        for i in range(len(inter_exposure) - 1):
            hom_correction_exp += ratios[i] ** 2 * (
                inter_exposure[i] if ~numpy.isnan(inter_exposure[i]) else 0)
            for j in range(i + 1, len(inter_exposure)):
                correction_exp -= ratios[i] * ratios[j] * (
                        (inter_exposure[i] if ~numpy.isnan(inter_exposure[i]) else 0) +
                        (
                            inter_exposure[j] if ~numpy.isnan(inter_exposure[j]) else 0))

        ord = numpy.argsort(ratios)
        pop_shape = cbsa_pop[cbsa_pop['cbsa_code'] == str(cbsa)].shape[0]
        gdp_cbsa = gdp[gdp['cbsa_code'] == str(cbsa)][year]
        cbsa_hom.append([year,cbsa,
                         correction,
                         hom_correction,
                         correction_seg_idx,
                         hom_correction_seg_idx,
                         correction_gini,
                         hom_correction_gini,
                         correction_exp,
                         hom_correction_exp,
                         cbsa_pop[cbsa_pop['cbsa_code'] == str(cbsa)]['B01003_001E'].values[0] if pop_shape>0 else numpy.nan,
                         income[income['cbsa_code'] == str(cbsa)]['B19013_001E'].values[0] if pop_shape>0 else numpy.nan,
                         gdp_cbsa.values[
                             0] if gdp_cbsa.shape[0] > 0 else numpy.nan,
                       ] + homophily + seggregation_index + gini_index +inter_exposure+ratios.tolist())
        c+=1


cbsa_hom = numpy.vstack(cbsa_hom)

cbsas_hom_df = pandas.DataFrame(cbsa_hom,columns=[
    "year",
    "cbsa_code",
    "homophily_correction",
    "homophily_correction_pos",
    "homophily_correction_seg_idx",
    "homophily_correction_pos_seg_idx",
    "homophily_correction_gini",
    "homophily_correction_pos_gini",
    "homophily_correction_exp",
    "homophily_correction_pos_exp",
    "cbsa_population",
    "cbsa_median_income",
    "cbsa_gdp",
    "homophily_white",
    "homophily_black",
    "homophily_native",
    "homophily_asian",
    "homophily_island",
    "homophily_other",

    "homophily_white_seg_idx",
    "homophily_black_seg_idx",
    "homophily_native_seg_idx",
    "homophily_asian_seg_idx",
    "homophily_island_seg_idx",
    "homophily_other_seg_idx",

    "homophily_white_gini",
    "homophily_black_gini",
    "homophily_native_gini",
    "homophily_asian_gini",
    "homophily_island_gini",
    "homophily_other_gini",

    "homophily_white_exp",
    "homophily_black_exp",
    "homophily_native_exp",
    "homophily_asian_exp",
    "homophily_island_exp",
    "homophily_other_exp",

    "population_white",
    "population_black",
    "population_native",
    "population_asian",
    "population_island",
    "population_other",
])

resids = []
for year in years:
    year = str(int(year))
    year_data = cbsas_hom_df[cbsas_hom_df['year'] == year]
    keep = ~numpy.isnan(year_data['cbsa_population'].astype(float)) & ~numpy.isnan(year_data['cbsa_median_income'].astype(float))
    scaling_data = year_data[keep]
    scaling_fit = OLS(numpy.log(scaling_data['cbsa_median_income'].astype(float)), add_constant(numpy.log(scaling_data['cbsa_population'].astype(float)))).fit()
    c=0
    for i in range(len(keep)):
        if keep.values[i]:
            resids.append(scaling_fit.resid.values[c])
            c += 1
        else:
            resids.append(numpy.nan)

cbsas_hom_df['scaling_residuals'] = resids

resids = []
for year in years:
    year = str(int(year))
    year_data = cbsas_hom_df[cbsas_hom_df['year'] == year]
    keep = ~numpy.isnan(year_data['cbsa_population'].astype(float)) & ~numpy.isnan(year_data['cbsa_gdp'].astype(float))
    scaling_data = year_data[keep]
    scaling_fit = OLS(numpy.log(scaling_data['cbsa_gdp'].astype(float)), add_constant(numpy.log(scaling_data['cbsa_population'].astype(float)))).fit()
    c=0
    for i in range(len(keep)):
        if keep.values[i]:
            resids.append(scaling_fit.resid.values[c])
            c += 1
        else:
            resids.append(numpy.nan)

cbsas_hom_df['scaling_residuals_gdp'] = resids

cbsas_hom_df.to_csv('/home/andrewstier/Downloads/scaling_homophily/data/race_eth_homophily_cbsas_census_tracts.csv')
