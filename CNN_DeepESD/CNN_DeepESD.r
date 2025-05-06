# Downscaling Multi-Model Climate Projection Ensembles with Deep Learning (DeepESD): Contribution to CORDEX EUR-44
# Geoscientific Model Development
# J. Baño-Medina, R. Manzanas, E. Cimadevilla, J. Fernández, J. González-Abad, A.S. Cofiño, and J.M. Gutiérrez

# GitHub repository at https://github.com/SantanderMetGroup/DeepDownscaling

# This notebook reproduces the results presented in **Downscaling Multi-Model Climate Projection Ensembles with Deep Learning (DeepESD): Contribution to CORDEX EUR-44***, submitted to Geoscientific Model Development by J. Baño-Medina, R. Manzanas, E. Cimadevilla, J. Fernández, J. González-Abad, A.S. Cofiño and J.M. Gutiérrez. This paper presents DeepESD, the first dataset of high-resolution (0.5º) climate change projections (up to 2100) of daily precipitation and temperature over Europe obtained with deep learning techniques (in particular convolutional neural networks) from an ensemble of eight global climate models from the Coupled Model Intercomparison Project version 5 (CMIP5).

# Note: The technical specifications of the machine used to run the code presented herein can be found at the end of the notebook.
# 1. Preparing the R environment and working directories

# This notebook is written in the free programming language R (version 3.6.1) and builds on climate4R (hereafter C4R), a suite of R packages developed by the Santander Met Group for transparent climate data access, post processing (including bias correction and downscaling) and visualization. For details on climate4R (C4R hereafter), the interested reader is referred to Iturbide et al. 2019.

# In particular, the following C4R libraries are used along the notebook: loadeR and loadeR.2nc (data loading), transformeR (data manipulation), downscaleR and downscaleR.keras (downscaling with neural networks) and visualizeR (visualization). To install them you may use the devtools package (e.g., devtools::install_github("SantanderMetGroup/downscaleR.keras@v1.0.0") to install downscaleR.keras). Alternatively, you may directly install the entire C4R framework through conda, C4R version 1.5.0, following the instructions provided at the end of this page. The latter option is highly recommended. Note that even with C4R v1.5.0 installed via conda, you still need to upgrade libraries climate4R.UDG and VALUE with the devtools package by typing: devtools::install_github(c("SantanderMetGroup/climate4R.UDG@v0.2.2","SantanderMetGroup/VALUE@v2.2.2")).

#######################################################################################################################################################

options(java.parameters = "-Xmx8g")  # expanding Java memory

# C4R libraries
library(loadeR) # version v1.7.0
library(loadeR.2nc) # version v0.1.1
library(transformeR) # version v2.1.0
library(downscaleR) # version v3.3.2
library(visualizeR) # version v1.6.0
library(downscaleR.keras) # version v1.0.0 that build on Keras version 2.3.0 and tensorflow version 2.2.0 
library(climate4R.value) # version v0.0.2 and relies on VALUE version v2.2.2
library(climate4R.UDG) # version v0.2.2

# Other useful libraries
library(magrittr) # to operate with '%>%' or '%<>%'

# For visualization purposes
library(RColorBrewer)
library(gridExtra)
library(ggplot2)
     

# The predictions and models generated along the notebook are saved in a well-organized set of directories. Please use the dir.create function to create two new folders (Data and models) in your working directory. Within each of these folders, create subsequently two more subfolders, named temperature and precip. Finally, we create also the figures directory directly in the working directory.

## Uncomment to create directories
dir.create("./CNN_DeepESD/Data")
dir.create("./CNN_DeepESD/Data/precip")
dir.create("./CNN_DeepESD/Data/temperature")
dir.create("./CNN_DeepESD/models")
dir.create("./CNN_DeepESD/models/precip")
dir.create("./CNN_DeepESD/models/temperature")
dir.create("./CNN_DeepESD/figures")
     
# We are now ready to load into our R environment all the data we are going to work with, which can be freely accessed through the Climate Data Service developed by the Santander Met Group (non registered users need to register first here). Use the loginUDG function to log into the service with your own credentials.

loginUDG(username = "youruser", password = "yourpassword") # login into the Santander CDS
     

# The following Table lists the Santander Climate Data Service (CDS) endpoint of the datasets used in this study (except for E-OBS whose observational records can be found in their website). For a wide variety of datasets, C4R uses labels to point to these endpoints. The available labels can be displayed by typing UDG.datasets() into an R terminal. Throughout the notebook we lean on the corresponding labels to load the data into our R environment.
# Dataset 	CDS endpoint:
# ERA-Interim 	https://data.meteo.unican.es/tds5/catalog/catalogs/interim/interim_DM_predictors.html?dataset=interim/daily/interim20_daily.ncml
# CMIP5 	https://data.meteo.unican.es/tds5/catalog/catalogs/cmip5/cmip5Datasets.html
# CORDEX 	https://data.meteo.unican.es/thredds/catalog/devel/c3s34d/catalog.html
# DeepESD 	https://data.meteo.unican.es/thredds/catalog/esgcet/catalog.html

# The following block of code allows for loading the ERA-Interim predictor variables, which are needed to train our neural networks, for the period 1979-2005 by using the loadGridData function. Subsequently, the makeMultiGrid creates a unique C4R object containing all this information.

# Predictor variables considered (see -*-)
vars  <- c("psl","z@500","z@700","z@850", 
          "hus@500","hus@700","hus@850",
          "ta@500","ta@700","ta@850",
          "ua@500","ua@700","ua@850",
          "va@500","va@700","va@850")
# We loop over the variables and then use  makeMultiGrid, to bind the variables in a single C4R object
x <- lapply(vars, function(z) {
  loadGridData(dataset = "ECMWF_ERA-Interim-ESD",  # "ECMWF_ERA-Interim-ESD" is the label that identifies the dataset of interest in the Santander CDS
               var = z,
               lonLim = c(-8,34),  # domain of interest for the predictors
               latLim = c(34,76),  # domain of interest for the predictors
               years = 1979:2005)
}) %>% makeMultiGrid()
     

# As predictands we use temperature and precipitation from E-OBS, which can be obtained as netCDF files here. Once downloaded, these data can be imported in R with the loadGridData function. Subsequently, we upscale these E-OBS fields from their native 0.25º to the 0.5º regular grid our projections are delivered by using the interpGrid from transformeR. In the cell below, we illustrate how to load both precipitation and temperature with loadGridData, however, note that you should only load one at a time, or change the object name y to e.g., y_rr or y_tg.

grid05 = list("x" = c(-9.75,30.25),"y" = c(34.25,74.25))  # boundaries of our projections' domain
attr(grid05,"resX") <- attr(grid05,"resY") <- 0.5  # target spatial resolution for our projections

## Please load only one predictand variable (either temperature or precipitation) 
## or give a different name to each variable (e.g. 'y_rr' and 'y_tg') ----------------------------------

# To load E-OBS precipitation (previously downloaded as netCDF file)
y <- loadGridData(dataset = "rr_ens_mean_0.25deg_reg_v20.0e.nc",
                  var = "rr",
                  lonLim = c(-10,30),
                  latLim = c(34,74), 
                  years = 1979:2005) %>% interpGrid(new.coordinates = grid05, method = "bilinear")

# To load E-OBS temperature (previously downloaded as netCDF file)
y <- loadGridData(dataset = "tg_ens_mean_0.25deg_reg_v20.0e.nc",
                  var = "tg",
                  lonLim = c(-10,30),
                  latLim = c(34,74), 
                  years = 1979:2005) %>% interpGrid(new.coordinates = grid05, method = "bilinear")
     

# We recommend the user to save the predictor x and predictand y data into .rda objects since these loading steps can be quite time-consuming.

# Convolutional Neural Networks (CNNs)

# To build DeepESD we rely on the convolutional neural networks (CNN) presented in Baño-Medina et al. 2020; in particular, on the CNN -*- models, which were found to provide robust results for precipitation (temperature) both in ''perfect-prognosis'' conditions but also in the GCM space. The cell below shows how to build these CNN models based on Keras, and save them in a custom function called modelCNN. Note that precipitation and temperature CNN models are different, so please (un)comment the needed lines depending on your particular target variable of interest.

# Note: We refer the reader to Baño-Medina et al. 2020 for further details about the exact configuration of the CNNs used herein.

## Please select one ----------------------------------------------------------------------------------------------------------------
## ----------------------------------------------------------------------------------------------------------------------------------
## See https://gmd.copernicus.org/articles/13/2109/2020 for technical details
## ----------------------------------------------------------------------------------------------------------------------------------

# Precipitation model
modelCNN <- function(inp) {
    # Input layer
    inputs <- layer_input(shape = dim(inp$x.global)[2:4])
    # Hidden layers
    l1 = layer_conv_2d(inputs,filters = 50, kernel_size = c(3,3), activation = 'relu', padding = "same")
    l2 = layer_conv_2d(l1,filters = 25, kernel_size = c(3,3), activation = 'relu', padding = "same")
    l3 = layer_conv_2d(l2,filters = 1, kernel_size = c(3,3), activation = 'relu', padding = "same")
    l4 = layer_flatten(l3)
    # Output layer
    l51 = layer_dense(l4,units = dim(inp
Data)[2], activation = 'sigmoid') 
    l52 = layer_dense(l4,units = dim(inpData)[2], activation = 'linear') 
    l53 = layer_dense(l4,units = dim(inpData)[2], activation = 'linear') 
    outputs <- layer_concatenate(list(l51,l52,l53))      
    model <- keras_model(inputs = inputs, outputs = outputs) 
}

## ----------------------------------------------------------------------------------------------------------------------------------

# Temperature model
modelCNN <- function(inp) {
    # Input layer    
    inputs <- layer_input(shape = dim(inp$x.global)[2:4])
    # Hidden layers    
    l1 = layer_conv_2d(inputs,filters = 50, kernel_size = c(3,3), activation = 'relu', padding = "valid")
    l2 = layer_conv_2d(l1,filters = 25, kernel_size = c(3,3), activation = 'relu', padding = "valid")
    l3 = layer_conv_2d(l2,filters = 10, kernel_size = c(3,3), activation = 'relu', padding = "valid") 
    l4 = layer_flatten(l3)
    # Output layer    
    l51 = layer_dense(l4,units = dim(inpData)[2], activation = 'linear') 
    l52 = layer_dense(l4,units = dim(inp
Data)[2], activation = 'linear') 
    outputs <- layer_concatenate(list(l51,l52))      
    model <- keras_model(inputs = inputs, outputs = outputs) 
}

## ----------------------------------------------------------------------------------------------------------------------------------
     
# 2. DeepESD

# In this section we 1) load the predictor variables of interest from the 8 GCM considered from the Santander CDS, 2) harmonize and standardize these predictor fields, 3) save these processed fields in rda objects to avoid repeating these steps in the future, 4) build the CNN models based on ERA-Interim predictors and E-OBS predictands and 5) apply these models to the GCM predictor variables to obtain the final high-resolution (downscaled at 0.5º) projections up to 2100.
# 2.1 Preparing the predictor datasets

# The dh and df objects below contain the labels that identify the 8 GCMs considered in this work in the Santander CDS, for the historical and RCP.8.5 scenario, respectively. These lables are used when calling the loadGridData function for data loading.

## Use UDG.datasets() to obtain the labels of the desired GCMs

# Historical scenario
dh <- c("CMIP5-subset_CanESM2_r1i1p1_historical",
       "CMIP5-subset_CNRM-CM5_r1i1p1_historical",
       "CMIP5-subset_MPI-ESM-MR_r1i1p1_historical",
       "CMIP5-subset_MPI-ESM-LR_r1i1p1_historical",
       "CMIP5-subset_NorESM1-M_r1i1p1_historical", 
       "CMIP5-subset_GFDL-ESM2M_r1i1p1_historical",
       "CMIP5-subset_EC-EARTH_r12i1p1_historical",
       "CMIP5-subset_IPSL-CM5A-MR_r1i1p1_historical")

# RCP8.5 scenario
df <- c("CMIP5-subset_CanESM2_r1i1p1_rcp85",
       "CMIP5-subset_CNRM-CM5_r1i1p1_rcp85", 
       "CMIP5-subset_MPI-ESM-MR_r1i1p1_rcp85",
       "CMIP5-subset_MPI-ESM-LR_r1i1p1_rcp85",
       "CMIP5-subset_NorESM1-M_r1i1p1_rcp85",
       "CMIP5-subset_GFDL-ESM2M_r1i1p1_rcp85",
       "CMIP5-subset_EC-EARTH_r12i1p1_rcp85",
       "CMIP5-subset_IPSL-CM5A-MR_r1i1p1_rcp85")

# Labels used to identify the 8 GCMs
dName <- c("CanESM2","CNRM-CM5","MPI-ESM-MR","MPI-ESM-LR","NorESM1","GFDL","EC-Earth","IPSL")
     

# The following loop allows us to load the predictors from the above GCMs over our target domain for the reference (1975-2005) and future (early-future: 2006-2040, mid-future: 2041-2070, far-future: 2071-2100) periods of interest. Note that the historical (RCP8.5) scenario is used for the reference (future) periods. Note also that, within the loop, all the GCMs are interpolated to the spatial resolution of the ERA-Interim predictors which were used to fit the CNNs. Once loaded, the GCM predictors are saved as .rda files.

lapply(c("h","ef","mf","ff"), FUN = function(sc) {
    if (sc == "h")  years <- 1975:2005  # historical period of interest
    if (sc == "ef") years <- 2006:2040  # early-future
    if (sc == "mf") years <- 2041:2070  # mid-future
    if (sc == "ff") years <- 2071:2100  # far-future
    if (sc == "h"){d <- dh} else {d <- df}
    
    # We loop over the GCMs
    lapply(1:length(d), FUN = function(zz) {    
    x <- lapply(vars, function(z) loadGridData(dataset = d[zz],var = z,
                                               lonLim = c(-8,34),latLim = c(34,76),
                                               years = years) %>% interpGrid(new.coordinates = getGrid(x))) %>% makeMultiGrid()      
    
    # Since the IPSL contains NA values in the 850hPa level at certain gridpoints, we replace these NA values with the numeric values of their closest neighbours.
    if (dName[zz] == "IPSL") {
      ind850 <- grepl("@850",x
varName,fixed = TRUE) %>% which()
      indGP <- apply(x$Data[ind850[1],1,,,], MARGIN = c(2,3), anyNA) %>% which(arr.ind = TRUE)
      for (i in 1:nrow(indGP)) {
        indTime <- is.na(x$Data[ind850[1],1,,indGP[i,1],indGP[i,2]]) %>% which()
        x$Data[ind850,1,indTime,indGP[i,1],indGP[i,2]] <- x$Data[ind850,1,indTime,indGP[i,1],indGP[i,2]-1]
      }
    }
                
    # We save the predictor fields as .rda files            
    if (sc == "h") {
        xh <- x
        save(xh, file = paste0("./Data/xh_",dName[zz],".rda"))
        rm(x,xh)
    } else {
        xf <- x
        save(xf, file = paste0("./Data/x",sc,"_",dName[zz],".rda"))
        rm(x,xf)
    }            
  })  
})  
     

# The following loop allows us to harmonize and standardize the GCM predictors loaded in the previous step to assure they reasonable resemble the ERA-Interim variables used to train the CNN models (note this is one of the key assumptions that are done in ''perfect-prognosis'' downscaling). For this harmonization+standardization step, which is different depending on the particular scenario of interest (the reader is referred again to Baño-Medina et al. 2020 for details about this process), the scaleGrid function from transformeR is used. The so-processed GCM predictors, wich will be used as inputs to the CNN models, are saved as rda files.

# We loop over the GCMs
lapply(1:length(dName), FUN = function(zz) {
  load(paste0("./Data/xh_",dName[zz],".rda")) 
  # We loop over the temporal periods  
  lapply(c("h","ef","mf","ff"), FUN = function(sc) {
      
    # We harmonize and standardize the historical scenario  
    if (sc == "h") {
      xh <- scaleGrid(xh, base = subsetGrid(xh, years = 1979:2005), ref = x, type = "standardize", spatial.frame = "gridbox", time.frame = "monthly") # harmonization      
      xn <- scaleGrid(xh, base = subsetGrid(xh, years = 1979:2005), type = "standardize")  # standardization
        
    # We harmonize and standardize the RCP8.5 scenario    
    } else {
      load(paste0("./Data/x",sc,"_",dName[zz],".rda"))  
      xf <- scaleGrid(xf, base = subsetGrid(xh, years = 1979:2005), ref = x, type = "standardize", spatial.frame = "gridbox", time.frame = "monthly") # harmonization    
      xh <- scaleGrid(xh, base = subsetGrid(xh, years = 1979:2005), ref = x, type = "standardize", spatial.frame = "gridbox", time.frame = "monthly") # harmonization     
      xn <- xf %>% scaleGrid(base = subsetGrid(xh, years = 1979:2005), type = "standardize")  # standardization                     
    }
      
    # We save the standardized predictor fields as `rda` objects  
    save(xn, file = paste0("./Data/xn",sc,"_",dName[zz],".rda"))  
  })
})
     
# 2.2 Precipitation downscaling

# This section shows how to fit the CNN model which links the large-scale predictors from ERA-Interim with the high-resolution E-OBS precipitation at surface. The steps to take would be the following:

    Prepare the predictor and predictand tensors with the prepareData.keras function from downscaleR.keras.
    Standardize the ERA-Interim predictors with the scaleGrid function from transformeR.
    For a better fit of the Gamma distribution, 0.99 is substracted from observed precipitation and negative values are ignored (note that this step implies that rainy days are defined as those receiving 1 or more mm of precipitation). To do this, the gridArithmetics and binaryGrid functions from transformeR are used.
    Train the CNN model encapsuled in the modelCNN (which has been previously defined) with the downscaleTrain.keras function from downscaleR.keras. To optimize the negative log-likelihood of the Bernoulli-Gamma distribution, we employ the custom loss function bernouilliGammaLoss from downscaleR.keras. The network is fitted using the adam optimizer and a learning rate of 1e-4. Early-stopping with a patience of 30 epochs is applied and the best model (epoch) is saved in the working directory as a .h5 file.


# NOTE: Running this cell takes about 1 hour 

# Preparing predictor and predictand data for downscaling with downscaleR.keras
xyT <- prepareData.keras(x = scaleGrid(x,type = "standardize"), y = binaryGrid(gridArithmetics(y,0.99,operator = "-"),condition = "GE",threshold = 0,partial = TRUE),
                         first.connection = "conv",last.connection = "dense",channels = "last")  

# Training the CNN model to downscale precipitation
downscaleTrain.keras(obj = xyT,model = modelCNN(xyT),clear.session = TRUE,
                     compile.args = list("loss" = bernouilliGammaLoss(last.connection = "dense"),"optimizer" = optimizer_adam(lr = 0.0001)),
                     fit.args = list("batch_size" = 100,"epochs" = 10000,"validation_split" = 0.1,"verbose" = 1,
                                     "callbacks" = list(callback_early_stopping(patience = 30),callback_model_checkpoint(filepath='./models/precip/CNN1.h5',monitor='val_loss', save_best_only=TRUE))))


     

# Once the model is trained, we use it to predict in both the train (training period using ERA-Interim variables) and the GCM spaces. As per the former, we are interested in the estimation of the parameter p (probability of rain), since it is needed to later adjust the frequency of rain in the high-resolution projections obtained from the GCM (see the manuscript for details). To compute p in the train period the following is done:

#    Prepare the predictors which will serve as inputs for the CNN model with the prepareNewData.keras function. Subsequently, use them to predict in the train set with the downscalePredict.keras function. The model argument indicates the path where the CNN model was previously stored, and C4Rtemplate is a C4R object used as template for the predictions which provides de proper metadata. Since downscalePredict.keras outputs 3 parameters (the probability of rain, p, and the logarithmic of the shape and scale parameters of the Gamma distribution, log_alpha and log_beta), the subsetGrid is applied in order to keep only p.


# Preparing predictor data
xyt <- prepareNewData.keras(scaleGrid(x,type = "standardize"), xyT) 
pred_ocu_train <- downscalePredict.keras(newdata = xyt,C4R.template = y,clear.session = TRUE,loss = "bernouilliGammaLoss",
                                         model = list("filepath" = './models/precip/CNN1.h5',"custom_objects" = c("custom_loss" = bernouilliGammaLoss(last.connection = "dense")))) %>% subsetGrid(var = "p")  
rm(xyt) # to save memory
     

# At this point, the trained CNN model is used to generate the high-resolution projections building on the 8 GCMs considered in this work. To do so, we perform a loop over the distinct GCMs in which the corresponding predictors (which had been previously saved) are loaded and conveniently transformed using the prepareNewData.keras function. Finally, the log_alpha, log_beta and p parameters, which are obtained with the downscalePredict.keras function are saved in the pred object.

#    On the one hand, log_alpha and log_beta are used to obtain the rainfall amount with the computeRainfall function from downscaleR.keras. The argument simulate allows us for specifying if either a stochastic or a deterministic outcome is wanted. The argument bias is used to re-center the Gamma distribution to 1mm/day.
#    On the other hand, we use the p parameter to derive the binary event occurrence/non occurrence through the bynaryGrid function.
#    Finally, both series (binary and continuous) are multiplied to produce the complete precipitation time series.

#In the following block of code we compute the (deterministic) frequency and (stochastic) amount of rain. The generated projections are saved in .nc format with the grid2nc function.

#Note: If a purely deterministic, or a purely stochastic version of the projections is wanted, please uncomment the corresponding lines.

# NOTE: Running this cell takes about 4 hours

# We loop over the GCMs
lapply(1:length(dName), FUN = function(zz) {
    
  # We loop over the temporal periods  
  lapply(c("ef","mf"), FUN = function(sc) {
    load(paste0("./Data/xn2",sc,"_",dName[zz],".rda"))
    xyt <- prepareNewData.keras(xn,xyT)  
    pred <- downscalePredict.keras(newdata = xyt,C4R.template = y,clear.session = TRUE,loss = "bernouilliGammaLoss",
                                   model = list("filepath" = './models/precip/CNN1.h5',"custom_objects" = c("custom_loss" = bernouilliGammaLoss(last.connection = "dense")))) 
    
    ## Frequency (deterministic) and amount of rain (stochastic) ------------------------------------------
    pred2 <- computeRainfall(log_alpha = subsetGrid(pred,var = "log_alpha"),log_beta = subsetGrid(pred,var = "log_beta"),bias = 1,simulate = TRUE) %>% gridArithmetics(binaryGrid(subsetGrid(pred,var = "p"),ref.obs = binaryGrid(y,threshold = 1, condition = "GE"),ref.pred = pred_ocu_train))  
    grid2nc(pred2,NetCDFOutFile = paste0("./Data/precip/CNN2_",sc,"_",dName[zz],".nc4"))   
  })    
})
     
# 2.3 Temperature downscaling

# This section shows how to train the CNN model which links the large-scale predictors from ERA-Interim with the high-resolution E-OBS temperature at surface. As for precipitation, the steps to take would be the following:

#    Prepare the predictor and predictand tensors with the prepareData.keras function from downscaleR.keras.
#    Standardize the ERA-Interim predictors with the scaleGrid function from transformeR.
#    Train the CNN model encapsuled in the modelCNN (which has been previously defined) with the downscaleTrain.keras function from downscaleR.keras. To optimize the negative log-likelihood of the Gaussian distribution, we use the custom loss function GaussianLoss from downscaleR.keras. Whe network is fitted using the adam optimizer and a learning rate of 1e-4. Early-stopping with a patience of 30 epochs is applied and the best model (epoch) is saved in the working directory as a .h5 file.


# NOTE: Running this cell takes about 1 hour

# Preparing predictor and predictand data for downscaling with downscaleR.keras
xyT <- prepareData.keras(x = scaleGrid(x,type = "standardize"),y = y,
                         first.connection = "conv",last.connection = "dense",channels = "last")  

# Training the CNN model to downscale temperature
downscaleTrain.keras(obj = xyT,model = modelCNN(xyT),clear.session = TRUE,
                     compile.args = list("loss" = gaussianLoss(last.connection = "dense"),"optimizer" = optimizer_adam(lr = 0.0001)),
                     fit.args = list("batch_size" = 100,"epochs" = 10000,"validation_split" = 0.1,"verbose" = 1,
                                     "callbacks" = list(callback_early_stopping(patience = 30),callback_model_checkpoint(filepath='./models/temperature/CNN10.h5',monitor='val_loss', save_best_only=TRUE))))
     

# Once trained, the CNN model is used to generate the high-resolution projections of temperature. Similarly as for precipitation, we apply for this task the prepareNewData.keras and downscalePredict.keras functions, saving the mean and log_var parameters in the pred object. The code below allows for obtaining deterministic projections (note that only the mean parameter is needed in this case) and saving them in .nc format with the grid2nc function.

# Note: If stochastic projections are wanted, please uncomment the corresponding lines. he projections as .ncusing function grid2nc.

# NOTE: Running this cell takes about 1 hour and 20 minutes

lapply(1:length(dName), FUN = function(zz) {
  lapply(c("h","ef","mf","ff"), FUN = function(sc) {
    load(paste0("./Data/xn",sc,"_",dName[zz],".rda"))
    xyt <- prepareNewData.keras(xn,xyT)
    pred <- downscalePredict.keras(newdata = xyt,C4R.template = y,clear.session = TRUE,loss = "gaussianLoss",
                                   model = list("filepath" = './models/temperature/CNN10.h5',"custom_objects" = c("custom_loss" = gaussianLoss(last.connection = "dense")))) 
    
    ## Deterministic version 
    pred <- subsetGrid(pred, var = "mean")
    pred
varName <- "tas"
    grid2nc(pred,NetCDFOutFile = paste0("./Data/temperature/CNN_",sc,"_",dName[zz],".nc4"))               
      
    k_clear_session()  
  })    
})
     
# 3 Dynamical climate models

# To assess the credibility of DeepESD, it is compared against two different ensembles of dynamical models (see the Technical Validation section), the first/second of them formed by Global/Regional Climate Models (GCMs/RCMs). Since DeepESD covers only land, we start by creating a 0.5º land-sea mask which will be later applied to eliminate sea points from both GCMs and RCMs.

mask <- gridArithmetics(subsetGrid(y,year = 1990),0) %>% gridArithmetics(1,operator = "+") %>% climatology()
grid2nc(mask, NetCDFOutFile = "./Data/mask.nc4")  
     
# 3.1 Ensemble of Global Climate Models (GCMs)

# We perform a loop over the temporal periods of interest (1975-2005 for the historical scenario plus 2006-2040, 2041-2070 and 2071-2100 for RCP8.5) and save the GCM ensemble as netCDF files (grid2nc function) in a multi-member C4R object. All GCMs are interpolated to our target 0.5º resolution (E-OBS grid), using conservative remapping. To do this interpolation, we rely on the cdo library and use function system to invoke the OS command. Please note that you can install the cdo library with conda, by typing conda install cdo in a terminal. Finally, sea points are removed by applying the land-sea mask we have previously created.

## Please select one:   -----------------------------------------------------------------------
# Parameter Setting for precipitation
variable <- "precip"
var <- "pr"
# Parameter Setting for temperature
variable <- "temperature"
var <- "tas"
## --------------------------------------------------------------------------------------------

# We loop over the temporal periods of interest
lapply(c("h","ef","mf","ff"), FUN = function(z) {
  if (z == "h")  years <- 1975:2005  # historical
  if (z == "ef") years <- 2006:2040  # RCP8.5
  if (z == "mf") years <- 2041:2070  # RCP8.5
  if (z == "ff") years <- 2071:2100  # RCP8.5
  if (z == "h") {d <- dh} else {d <- df} 
    
  # We loop over the GCM labels  
  lapply(1:length(d), FUN = function(zzz) {    
    # Load the data and interpolate to the target resolution with interpGrid  
    yy <- loadGridData(dataset = d[zzz],var = var,years = years,lonLim = c(-10,30),latLim = c(34,74))
    grid2nc(yy, NetCDFOutFile = "./aux.nc4") # we save the GCM cropped to the European domain, and save as .nc in an auxiliary variable 
    system(paste0("cdo remapcon,", "./Data/mask.nc4", " ", "./aux.nc4", " ", "./aux2.nc4"))  # We use system function to call the cdo library and interpolate the grid using conservative remapping
    yy <- loadGridData("./aux2.nc4", var = var) # we load the interpolated GCM field
    file.remove(c("./aux.nc4","./aux2.nc4"))  
    # Apply land-sea mask 
    yy <- lapply(1:getShape(yy,"time"), FUN = function(z) gridArithmetics(subsetDimension(yy, dimension = "time", indices = z),mask)) %>% bindGrid(dimension = "time")            
    # Save the GCM members    
    grid2nc(yy,NetCDFOutFile = paste0("./Data/",variable,"/y_",z,"_",dName[zzz],".nc4"))
  }) 
})
     
# 3.2 Ensemble of Regional Climate Models (RCMs)

# In this section we form an ensemble of EURO-CORDEX RCMs which can be easily loaded from the Santander CDS by using the appropiate labels (see the block below).

# Labels for the historical scenario
dh <- c("CORDEX-EUR-44_CCCma-CanESM2_historical_r1i1p1_SMHI-RCA4_v1",
        "CORDEX-EUR-44_CNRM-CERFACS-CNRM-CM5_historical_r1i1p1_ETH-CLMcom-CCLM5-0-6_v1",
        "CORDEX-EUR-44_CNRM-CERFACS-CNRM-CM5_historical_r1i1p1_SMHI-RCA4_v1",
        "CORDEX-EUR-44_MPI-M-MPI-ESM-LR_historical_r1i1p1_CLMcom-CCLM4-8-17_v1",
        "CORDEX-EUR-44_MPI-M-MPI-ESM-LR_historical_r1i1p1_MPI-CSC-REMO2009_v1",
        "CORDEX-EUR-44_NCC-NorESM1-M_historical_r1i1p1_SMHI-RCA4_v1",
        "CORDEX-EUR-44_NOAA-GFDL-GFDL-ESM2M_historical_r1i1p1_SMHI-RCA4_v1",
        "CORDEX-EUR-44_ICHEC-EC-EARTH_historical_r12i1p1_SMHI-RCA4_v1",
        "CORDEX-EUR-44_ICHEC-EC-EARTH_historical_r12i1p1_ETH-CLMcom-CCLM5-0-6_v1",
        "CORDEX-EUR-44_IPSL-IPSL-CM5A-MR_historical_r1i1p1_SMHI-RCA4_v1",
        "CORDEX-EUR-44_IPSL-IPSL-CM5A-MR_historical_r1i1p1_IPSL-INERIS-WRF331F_v1")
# Labels for the RCP8.5 scenario
df <- c("CORDEX-EUR-44_CCCma-CanESM2_rcp85_r1i1p1_SMHI-RCA4_v1",
        "CORDEX-EUR-44_CNRM-CERFACS-CNRM-CM5_rcp85_r1i1p1_ETH-CLMcom-CCLM5-0-6_v1",
        "CORDEX-EUR-44_CNRM-CERFACS-CNRM-CM5_rcp85_r1i1p1_SMHI-RCA4_v1",
        "CORDEX-EUR-44_MPI-M-MPI-ESM-LR_rcp85_r1i1p1_CLMcom-CCLM4-8-17_v1",
        "CORDEX-EUR-44_MPI-M-MPI-ESM-LR_rcp85_r1i1p1_MPI-CSC-REMO2009_v1",
        "CORDEX-EUR-44_NCC-NorESM1-M_rcp85_r1i1p1_SMHI-RCA4_v1",
        "CORDEX-EUR-44_NOAA-GFDL-GFDL-ESM2M_rcp85_r1i1p1_SMHI-RCA4_v1",
        "CORDEX-EUR-44_ICHEC-EC-EARTH_rcp85_r12i1p1_SMHI-RCA4_v1",
        "CORDEX-EUR-44_ICHEC-EC-EARTH_rcp85_r12i1p1_ETH-CLMcom-CCLM5-0-6_v1",
        "CORDEX-EUR-44_IPSL-IPSL-CM5A-MR_rcp85_r1i1p1_SMHI-RCA4_v1",
        "CORDEX-EUR-44_IPSL-IPSL-CM5A-MR_rcp85_r1i1p1_IPSL-INERIS-WRF331F_v1")
     

# We perfrom a loop over the temporal periods of interest (1975-2005 for the historical scenario plus 2006-2040, 2041-2070 and 2071-2100 for RCP8.5) and save the RCM ensemble as netCDF files (grid2nc function) in a multi-member C4R object. All RCMs are interpolated to our target 0.5º resolution (E-OBS grid) and sea points are removed by applying the land-sea mask we have previously created.

## Please select one:   -----------------------------------------------------------------------
# Precipitation
variable <- "precip"
var <- "pr"
## Temperature
variable <- "temperature"
var <- "tas"
## --------------------------------------------------------------------------------------------

# We loop over the temporal periods of interest
lapply(c("h","ef","mf","ff"), FUN = function(z) {
  if (z == "h")  years <- 1975:2005  # historical
  if (z == "ef") years <- 2006:2040  # RCP8.5
  if (z == "mf") years <- 2041:2070  # RCP8.5
  if (z == "ff") years <- 2071:2099  # RCP8.5
  if (z == "h") {d <- dh} else {d <- df}
    
  # We loop over the RCM labels  
  yy <- lapply(1:length(d), FUN = function(zzz) { 
        # Load the data and interpolate to the target resolution with interpGrid
        yy <- loadGridData(dataset = d[zzz],var = var,years = years,lonLim = c(-10,30),latLim = c(34,74)) %>% interpGrid(getGrid(y))
        # Apply land-sea mask
        yy <- lapply(1:getShape(yy,"time"), FUN = function(z) gridArithmetics(subsetDimension(yy, dimension = "time", indices = z),mask)) %>% bindGrid(dimension = "time")  
        # Save the RCM members
        grid2nc(yy,NetCDFOutFile = paste0("./Data/",variable,"/yRCM_",z,"_member",zzz,".nc4"))
  })
})
     
# 4. Results

# This section provides the code needed to reproduce the figures presented in the manuscript. Note we mostly rely on the visualizeR package for plotting since it supports both spatial maps and temporal series.
# 4.1 Ensemble mean and bias with respect to E-OBS

# Figure 1 in the manuscript shows the climatology of the different ensembles built (GCM, RCM and DeepESD), along with the corresponding mean error (bias) with respect to the observed pattern in the historical period. We start thus by computing the climatology of the different contributing members forming each ensemble and saving them as netCDF files in the working directory.

## Please select one:   -----------------------------------------------------------------------
# Precipitation
variable <- "precip"
var <- "pr"
# Temperature
variable <- "temperature"
var <- "tas"
## --------------------------------------------------------------------------------------------

dates <- list(start = "2006-01-01 12:00:00 GMT", end = "2041-01-01 12:00:00 GMT") # because a member of the RCM ensemble misses a value on the 01-Jan-2006, so to preserve temporal consistency in the ensemble we add this date to the ensemble mean metadata
dName1 <- c("CanESM2","CNRM-CM5","MPI-ESM-MR","MPI-ESM-LR","NorESM1","GFDL","EC-Earth","IPSL")
dName2 <- 1:11
nn <- c("GCM","RCM","CNN")
lapply(1:length(nn), FUN = function(zz) {
  if (nn[zz] == "RCM") {dName <- dName2} else {dName <- dName1}  
  lapply(c("h","ef","mf","ff"), FUN = function(z) {  
    pred <- lapply(1:length(dName), FUN = function(zzz) { 
      if (zz == 1) path <- paste0("./Data/",variable,"/y_",z,"_",dName[zzz],".nc4") # GCM ensemble
      if (zz == 2) path <- paste0("./Data/",variable,"/yRCM_",z,"_member",zzz,".nc4") # RCM ensemble           
      if (zz == 3) path <- paste0("./Data/",variable,"/CNN_",z,"_",dName[zzz],".nc4") # DeepESD ensemble   
      grid <- loadGridData(dataset = path,var = var)
      grid <- valueIndex(grid, index.code = "Mean")$Index  # computing the mean of each member 
      if (z == "ef") grid$Dates <- dates 
      return(grid)  
    }) %>% bindGrid(dimension = "member")  # bind the member means in a single C4R object along the `member` dimension
    
    pred$InitializationDates <- NULL
    # Saving the ensemble mean in netCDF format  
    grid2nc(pred,NetCDFOutFile = paste0("./Data/",variable,"/",nn[zz],"_",z,"_ensemble.nc4"))
    pred <- NULL  
  })
})  
     

# Next, the mean climatology for each ensemble is obtained with the aggregateGrid function (note the aggregation is done along the member dimension) from transformeR. Afterwards, we can already use spatialPlot to plot the corresponding spatial pattern. The resulting figure is saved in pdf format in the path indicated in the pdfOutput object.

# Note: Depending on the target variable of interest the user should comment/uncomment the appropriate lines at the beginning of the cell, which define the plotting parameters better suited for precipitation and temperature. This applies to the rest of the notebook from now on.

## Please select one:   -----------------------------------------------------------------------
# Precipitation
cb <- c("#FFFFFF",brewer.pal(n = 9, "BuPu"))
cb <- cb %>% colorRampPalette()
at <- seq(0,8,0.5)    
units <- "mm/day"
pdfOutput <- "./figures/ensembleMean_pr.pdf" 
var <- "precip"
# Temperature
cb <- c("#FFFFFF",brewer.pal(n = 9, "OrRd"))
cb <- cb %>% colorRampPalette()
at <- seq(-5, 20,2.5) 
units <- "ºC"
pdfOutput <- "./figures/ensembleMean_tas.pdf" 
var <- "temperature"
## --------------------------------------------------------------------------------------------

nn <- c("GCM","RCM","CNN")
figs <- lapply(1:length(nn), FUN = function(z) {
  # We store in `grid` object the ensemble of climatologies  
  grid <- loadGridData(paste0("./Data/",var,"/",nn[z],"_h_ensemble.nc4"), var = "Mean")
  # Compute the ensemble mean  
  gridMean <-  aggregateGrid(grid,aggr.mem = list(FUN = "mean", na.rm = TRUE)) 
  # We depict the ensemble mean with spatialPlot function  
  spatialPlot(gridMean,
              backdrop.theme = "coastline",
              main = paste0("Ensemble Mean (",units,") - ",nn[z]),
              ylab = "1975-2005",
              col.regions = cb,
              at = at,
              set.min = at[1], set.max = at[length(at)])
}) 
pdf(pdfOutput, width = 15, height = 10)   
grid.arrange(grobs = figs, ncol = 3)                     
dev.off()
     

# Now we plot the bias with respecto to the observed (i.e. E-OBS) climatology. Again, we rely on spatialPlot to depict the spatial fields, and aggregateGrid and gridArithmetics, to compute the ensemble mean and its bias, respectively. The resulting figures are saved in pdf format in the path indicated by pdfOutput.

## Please select one:   -----------------------------------------------------------------------
# Precipitation
cb <- brewer.pal(n = 11, "BrBG")
cb[6] <- "#FFFFFF"; cb <- cb %>% colorRampPalette()
at <- c(seq(-2, -0.5,0.5),-0.25,0.25,seq(0.5, 2,0.5))    
units <- "mm/day"
pdfOutput <- "./figures/bias_pr.pdf" 
var <- "precip"
# Temperature
cb <- rev(brewer.pal(n = 11, "RdBu"))
cb[6] <- "#FFFFFF"; cb <- cb %>% colorRampPalette()
at <- c(seq(-2, -0.5,0.5),-0.25,0.25, seq(0.5,2,0.5)) 
units <- "ºC"
pdfOutput <- "./figures/bias_tas.pdf"
var <- "temperature"
## --------------------------------------------------------------------------------------------

nn <- c("GCM","RCM","CNN")
figs <- lapply(1:length(nn), FUN = function(z) {
  # Compute the ensemble mean  
  grid <- loadGridData(paste0("./Data/",var,"/",nn[z],"_h_ensemble.nc4"), var = "Mean") %>% aggregateGrid(aggr.mem = list(FUN = "mean", na.rm = TRUE))
  # Compute the bias with respect to the observed temporal climatology for the same period
  grid  %<>% gridArithmetics(valueIndex(y, index.code = "Mean")$Index,operator = "-")
  # Depict the bias of the ensemble mean
  spatialPlot(grid,
              backdrop.theme = "coastline",
              main = paste0("Bias Ensemble Mean (",units,") - ",nn[z]),
              ylab = "1975-2005",
              col.regions = cb,
              at = at,
              set.min = at[1], set.max = at[length(at)])  
}) 
pdf(pdfOutput, width = 15, height = 10)   
grid.arrange(grobs = figs, ncol = 3)                     
dev.off()
     
# 4.2 Climate change signals

# To produce Figure 2 in the manuscript we perform a loop, for each ensemble (GCM, RCM and DeepESD), over the different RCP8.5 periods of interest and sequentially compute the difference between the future climatology and the historical one. These climate change signals are then averaged along the member dimension using the aggregateGrid function. The resulting figures are saved in pdf format in the path indicated by pdfOutput.

## Please select one:   -----------------------------------------------------------------------
# Precipitation
cb <- brewer.pal(n = 11, "BrBG")
cb[6] <- "#FFFFFF"; cb <- cb %>% colorRampPalette()
at <- c(seq(-1, -0.25,0.25),-0.125,0.125,seq(0.25, 1,0.25)) 
pdfOutput <- "./figures/deltas_pr.pdf" 
var <- "precip"
# Temperature
cb <- c("#FFFFFF",brewer.pal(n = 11, "OrRd"))
at <- c(seq(0,4,0.5),5,6)
pdfOutput <- "./figures/deltas_tas.pdf"
var <- "temperature"
## --------------------------------------------------------------------------------------------

nn <- c("GCM","RCM","CNN")
figs <- lapply(c("ef","mf","ff"), FUN = function(z) {  
    lapply(1:length(nn), FUN = function(zz) {
    gridh <- loadGridData(paste0("./Data/",var,"/",nn[zz],"_h_ensemble.nc4"), var = "Mean") 
    gridf <- loadGridData(paste0("./Data/",var,"/",nn[zz],"_",z,"_ensemble.nc4"), var = "Mean") 
    grid <- gridArithmetics(gridf,gridh,operator = "-")
    gridMean <-  aggregateGrid(grid,aggr.mem = list(FUN = "mean", na.rm = TRUE)) 

    if (z == "ef") period <- c("2006-2040")
    if (z == "mf") period <- c("2041-2070")
    if (z == "ff") period <- c("2071-2100")
    spatialPlot(gridMean,
                backdrop.theme = "coastline",
                main = paste("CC. signal wrt 1975-2005"),
                ylab = period,
                col.regions = cb,
                at = at,
                set.min = at[1], set.max = at[length(at)])  
  }) 
}) %>% unlist(recursive = FALSE)   
pdf(pdfOutput, width = 15, height = 10)   
grid.arrange(grobs = figs, ncol = 3)                     
dev.off()
     
# 4.3 Time-series

# The following block of code allows for plotting the time-series of the climate change signals. For precipitation (temperature), we perform a loop over the validation metrics of interest: R01, SDII (Mean). For details about these metrics please see the manuscript or type show.indices() in a new cell. At each iteration of the loop we define a doCall.args list which contains the aggr.y arguments needed for the aggregateGrid function (note the validation is done at an annual basis), which is finally passed t do.call. At the end of the loop, the resulting figures are saved in pdf format.

## Please select one:   -----------------------------------------------------------------------
# Precipitation
indices <- c("R01","SDII")
variable  <- "precip"
var <- "pr"
# Temperature
indices <- c("Mean")
variable <- "temperature"
var <- "tas"

## --------------------------------------------------------------------------------------------
figs <- lapply(indices, FUN = function(zz) {
  doCall.args <- list() 
  doCall.args[["aggr.y"]] <- list()
  
  # The R01 do.call arguments  
  if (zz == "R01")  {
    doCall.args[["aggr.y"]][["FUN"]] <- "index.freq"
    doCall.args[["aggr.y"]][["freq.type"]] <- "rel"
    doCall.args[["aggr.y"]][["condition"]] <- "GE"
    doCall.args[["aggr.y"]][["threshold"]] <- 1
    ylim <- c(0.24,0.54)  
  }
  # The SDII do.call arguments  
  if (zz == "SDII"){
    doCall.args[["aggr.y"]][["FUN"]] <- "index.meanGE"
    doCall.args[["aggr.y"]][["threshold"]] <- 1
    ylim <- c(2,9)  
  } 
  # The Mean do.call arguments    
  if (zz == "Mean"){
    doCall.args[["aggr.y"]][["FUN"]]   <- "mean"
    doCall.args[["aggr.y"]][["na.rm"]] <- TRUE
    ylim <- c(0,18)  
  }    
    
  # We compute the index for the GCM ensemble. To do this, we loop over the temporal periods and then bind the serie along the time dimension with bindGrid function    
  pred1 <- lapply(c("h","ef","mf","ff"), FUN = function(z) { 
    lapply(1:length(dName), FUN = function(zzz) {            
      doCall.args[["grid"]] <- loadGridData(dataset = paste0("./Data/",variable,"/y_",z,"_",dName[zzz],".nc4"),var = var)
      do.call("aggregateGrid",doCall.args)
    }) %>% bindGrid(dimension = "member")
  }) %>% bindGrid(dimension = "time") 
  
  # We compute the index for the RCM ensemble. To do this, we loop over the temporal periods and then bind the serie along the time dimension with bindGrid function
  pred3 <- lapply(c("h","ef","mf","ff"), FUN = function(z) { 
    lapply(1:11, FUN = function(zzz) {            
      doCall.args[["grid"]] <- loadGridData(dataset = paste0("./Data/",variable,"/yRCM_",z,"_member",zzz,".nc4"),var = var)
      do.call("aggregateGrid",doCall.args)
    }) %>% bindGrid(dimension = "member")
  }) %>% bindGrid(dimension = "time")
    
    # We compute the index for the DeepESD ensemble. To do this, we loop over the temporal periods and then bind the serie along the time dimension with bindGrid function    
  pred2 <- lapply(c("h","ef","mf","ff"), FUN = function(z) { 
    lapply(1:length(dName), FUN = function(zzz) {            
      doCall.args[["grid"]] <- loadGridData(dataset = paste0("./Data/",variable,"/CNN_",z,"_",dName[zzz],".nc4"),var = var)
      do.call("aggregateGrid",doCall.args)
    }) %>% bindGrid(dimension = "member")
  }) %>% bindGrid(dimension = "time")
      
  # We compute the index for the observed temporal serie 
  doCall.args[["grid"]] <- y
  y <- do.call("aggregateGrid",doCall.args)
  
  # We call temporalPlot to plot the times-series 
  temporalPlot("OBS" = y,"GCM" = pred1,"RCM" = pred3,"CNN" = pred2, cols = c("black","red","blue","green"),xyplot.custom = list(ylim = ylim))       
})

# Saving the resulting figures in .pdf format
pdf(paste0("./figures/serie_",var,".pdf"), width = 15, height = 4)
grid.arrange(grobs = figs, ncol = 3)  
dev.off()
     
# Technical specifications

# Please note this notebook was run on a machine with the following technical specifications:

#    Operating system: Ubuntu 18.04.3 LTS (64 bits)
#    Memory: 60 GiB
#    Processor: 2x Intel(R) Xeon(R) CPU E5-2670 0 @ 2.60GHz (16 cores, 32 threads)

