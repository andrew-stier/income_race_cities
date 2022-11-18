import pandas
import numpy
from statsmodels.api import OLS,add_constant

cbsa_hom = []
years = range(2010,2021)
for year in years:
    year = str(int(year))
    income_tracts = pandas.read_csv(
        '/home/andrewstier/Downloads/census_all_tracts/ACSDT5Y%s.B19001_data_with_overlays_2022-07-18T090053.csv' % year)
    total_column = 'B19001_001E'
    geo_colummn = 'GEO_ID'
    income_cols = ['B19001_0%sE' % (str(i) if i >= 10 else '0' + str(i)) for i in range(2, 18)]
    income_tracts = income_tracts[income_tracts[geo_colummn].str.contains('US')]
    income_tracts['county'] = income_tracts[geo_colummn].map(lambda x: x.split('US')[1][:5])
    income_tracts['below_med_inc'] = income_tracts[income_cols[:10]].sum(1)
    income_tracts['above_med_inc'] = income_tracts[income_cols[10:]].sum(1)
    income_cols = ['below_med_inc','above_med_inc']


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

    gdp_macro = pandas.read_csv('/home/andrewstier/Downloads/census_all_tracts/bea_gdp_cbsa.csv', skiprows=4)
    gdp_micro = pandas.read_csv('/home/andrewstier/Downloads/census_all_tracts/bea_gdp_micro.csv', skiprows=4)
    gdp = pandas.concat([gdp_macro, gdp_micro])
    gdp['cbsa_code'] = gdp['GeoFips'].astype(int).astype(str)
    gdp = gdp[['cbsa_code', year]]

    joined_income = income_tracts.set_index('county').join(cbsas.set_index('county')).reset_index()

    c=0
    for cbsa in cbsa_codes:
        cbsa_income = joined_income[joined_income['CBSA Code'] == cbsa]
        ratios = numpy.array([
            cbsa_income[x].astype(float).sum() / cbsa_income[total_column].astype(float).sum()
            for x in income_cols
        ])
        # ratios = [ratios[0]] + [ratios[1:4].sum()] + [ratios[4:7].sum()] + [ratios[7:10].sum()] + [ratios[10:13].sum()] + [ratios[13:].sum()]
        tot = cbsa_income[total_column].astype(float)
        tot = tot.map(lambda x: x if x>0 else numpy.nan)
        local_ratio = [
            cbsa_income[x].astype(float) / tot for x in income_cols
        ]
        homophily = []
        for i in range(len(income_cols)):
            homophily.append(numpy.abs(
                local_ratio[i]-ratios[i]
            ).mean())
        correction = 1
        for i in range(len(homophily)-1):
            for j in range(i+1,len(homophily)):
                correction -= ratios[i]*ratios[j]*(homophily[i]+homophily[j])
        ord = numpy.argsort(ratios)
        gdp_cbsa = gdp[gdp['cbsa_code'] == str(cbsa)][year]
        pop_shape = cbsa_pop[cbsa_pop['cbsa_code'] == str(cbsa)].shape[0]
        cbsa_hom.append([year,cbsa,
                         correction,
                         cbsa_pop[cbsa_pop['cbsa_code'] == str(cbsa)]['B01003_001E'].values[0] if pop_shape>0 else numpy.nan,
                         income[income['cbsa_code'] == str(cbsa)]['B19013_001E'].values[0] if pop_shape>0 else numpy.nan,
                         gdp_cbsa.values[
                             0] if gdp_cbsa.shape[0] > 0 else numpy.nan,

                       ])
        c+=1


cbsa_hom = numpy.vstack(cbsa_hom)

cbsas_hom_df = pandas.DataFrame(cbsa_hom,columns=[
    "year",
    "cbsa_code",
    "homophily_correction",
    "cbsa_population",
    "cbsa_median_income",
    "cbsa_gdp",
    # "homophily_white",
    # "homophily_black",
    # "homophily_native",
    # "homophily_asian",
    # "homophily_island",
    # "homophily_other",
    # "population_1",
    # "population_2",
    # "population_3",
    # "population_asian",
    # "population_island",
    # "population_other",
])

resids = []
for year in years:
    year = str(int(year))
    year_data = cbsas_hom_df[cbsas_hom_df['year'] == year]
    keep = ~numpy.isnan(year_data['cbsa_population'].astype(float)) \
           & ~numpy.isnan(year_data['cbsa_median_income'].astype(float))
    scaling_data = year_data[keep]
    scaling_fit = OLS(numpy.log(scaling_data['cbsa_median_income'].astype(float)),
                      add_constant(numpy.log(scaling_data['cbsa_population'].astype(float)))).fit()
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


cbsas_hom_df.to_csv('/home/andrewstier/Downloads/scaling_homophily/data/income_homophily_cbsas_census_tracts.csv')
