var_clim = ['aridity', 'moisture_index', 'pet_mean',  'precip_mean',
       'seasonality', 'snow_frac', 'snow_mean', 'temp_gradient',
       'temp_mean','precip_high_dur','precip_high_freq', 'precip_low_dur', 'precip_low_freq','temp_high_dur', 'temp_high_freq', 'temp_low_dur', 'temp_low_freq',]


var_catch_attr =["elv_mean",	"elv_std", "elv_min",
"elv_max","gradient_mean","gradient_std",
"upArea_max","forest","irrigated",
"other","rice","sealed","water",
"land_use_main","cropcoef_mean","cropcoef_std",
"chanbnkf_mean","chanbw_mean","changrad_mean",
"chanlength_sum","chanman_mean","chanmct_sum",
"ksat1","ksat2","ksat3","lambda1","lambda2",
"lambda3","genua1","genua2","genua3",
"soildepth1","soildepth2","soildepth3",
"thetas1","thetas2","thetas3","thetar1",
"thetar2","thetar3","lai01","lai02","lai03",
"lai04","lai05","lai06","lai07","lai08","lai09","lai10","lai11","lai12",
"laiyrmean","laiyrmax","laiyrmin","no_reservoirs","storage_reservoirs","no_lakes","area_lakes"]


list_var_inflow = ["inflow_mean","inflow_std","inflow_min","inflow_max","inflow_num"]
list_var = var_clim +var_catch_attr

var_output = ['CalChanMan1','CalChanMan3', 'GwLoss', 'GwPercValue', "UpperZoneTimeConstant","LowerZoneTimeConstant",
       'LZThreshold', 'PowerPrefFlow',
        'SnowMeltCoef','b_Xinanjiang','TransSub',]
