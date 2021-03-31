library(hsi)
library(magrittr)
library(raster)
library(usethis)
library(devtools)
library(rgdal)

load(file ="/shared_volume/e_puma_test.RData" )

stopifnot(inherits(e_test, "sp.temporal.env"))

n_nas <- floor(dim(e_test$env_data_train)[1]*0.1)

env_train <- e_test$env_data_train

rm_layers <- unlist(sapply( 1:dim(env_train)[2], function(x){
  if(length(which(is.na(env_train[,x]))) > n_nas) return(x)
} ))

if(!is.null(rm_layers)){
     env_train <- stats::na.omit(env_train[,-rm_layers])
  }

numericIDs <- which(sapply(env_train, is.numeric))

cor_matrix <- stats::cor(env_train[,numericIDs])

cor_threshold=0.9
ellipsoid_level=0.975
nvars_to_fit=3
plot3d=FALSE
E = 0.05
RandomPercent = 50
NoOfIteration=1000
parallel=FALSE
n_cores=4

find_cor   <- correlation_finder(cor_mat = cor_matrix,
                                 threshold = cor_threshold,
                                 verbose = F)

cor_filter <- find_cor$descriptors


combinatoria_vars <- combn(length(cor_filter),nvars_to_fit)


year_to_search <- min(as.numeric(names(e_test$layers_path_by_year)))


cat("The total number of models to be tested are: ", dim(combinatoria_vars)[2],"...\n\n")

env_layers <- raster::stack(e_test$layers_path_by_year[[paste0(year_to_search)]])

this_species <- e_test

#10:40 didnt work in 30, testing:

modelos <- lapply(1:969,function(x){
      cat(sprintf("Doing model: %d of %d \n", x , dim(combinatoria_vars)[2]))

      print("Varaibles filtadas por combinatiria de las mas representativas")
      vars_model <- cor_filter[combinatoria_vars[,x]]
      ellip <- try(cov_center(env_train[,vars_model],
                              level = ellipsoid_level ,vars = vars_model),silent = T)
      if(class(ellip)=="try-error") return("what to return?")

      print("Datos de presencia de la sp en el ambiente")
      occs_env <- this_species$env_data_train[,vars_model]

      print("Ajuste del modelo de elipsoide")

      sp_model <- ellipsoidfit(data = env_layers[[vars_model]],
                               centroid =ellip$centroid,
                               covar =  ellip$covariance,
                               level = ellipsoid_level,
                               size = 3,
                               plot = plot3d)

      valData <- this_species$test_data[,c(1,2)]
      valData$sp_name <- "sp"
      valData <- valData[,c(3,1,2)]
      print("PartialROC")
      p_roc<- PartialROC(valData = valData,
                         PredictionFile = sp_model$suitRaster,
                         E = E,
                         RandomPercent = RandomPercent,
                         NoOfIteration = NoOfIteration)
      p_roc$auc_pmodel <- paste0(x)

      return(list(model = sp_model$suitRaster,
                  pRoc=p_roc[,c("auc_ratio","auc_pmodel")],
                  metadata=ellip))

    })

#parallel:
future::plan(tweak(multiprocess, workers = n_cores))
modelos <- 1:dim(combinatoria_vars)[2] %>%
  furrr::future_map(function(x){
    cat("Doing model: ", x," of ", dim(combinatoria_vars)[2],"\n")

    # Varaibles filtadas por combinatiria de las mas representativas
    vars_model <- cor_filter[combinatoria_vars[,x]]
    ellip <- try(cov_center(env_train[,vars_model],
                            level = ellipsoid_level ,vars = vars_model),silent = T)
    if(class(ellip)=="try-error") return()

    # Datos de presencia de la sp en el ambiente
    occs_env <- this_species$env_data_train[,vars_model]

    # Ajuste del modelo de elipsoide

    sp_model <- ellipsoidfit(data = env_layers[[vars_model]],
                             centroid =ellip$centroid,
                             covar =  ellip$covariance,
                             level = ellipsoid_level,
                             size = 3,
                             plot = plot3d)

    if(length(ellip$centroid)==3 && plot3d){
      # Presencias de la sp en el ambiente
      rgl::points3d(occs_env,size=10)

      # Ejes del elipsoide

      rgl::segments3d(x = ellip$axis_coordinates[[1]][,1],
                      y = ellip$axis_coordinates[[1]][,2],
                      z = ellip$axis_coordinates[[1]][,3],
                      lwd=3)


      rgl::segments3d(x = ellip$axis_coordinates[[2]][,1],
                      y = ellip$axis_coordinates[[2]][,2],
                      z = ellip$axis_coordinates[[2]][,3],
                      lwd=3)

      rgl::segments3d(x = ellip$axis_coordinates[[3]][,1],
                      y = ellip$axis_coordinates[[3]][,2],
                      z = ellip$axis_coordinates[[3]][,3],
                      lwd=3)

    }

    valData <- this_species$test_data[,c(1,2)]
    valData$sp_name <- "sp"
    valData <- valData[,c(3,1,2)]
    p_roc<- PartialROC(valData = valData,
                       PredictionFile = sp_model$suitRaster,
                       E = E,
                       RandomPercent = RandomPercent,
                       NoOfIteration = NoOfIteration)
    p_roc$auc_pmodel <- paste0(x)

    return(list(model = sp_model$suitRaster,
                pRoc=p_roc[,c("auc_ratio","auc_pmodel")],
                metadata=ellip))
  },.progress = TRUE)