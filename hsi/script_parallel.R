library(hsi)
library(magrittr)
library(raster)
library(usethis)
library(devtools)
library(rgdal)
library(future)


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

###############
###############parallel:

future::plan(tweak(multiprocess, workers = n_cores))
modelos <- 1:dim(combinatoria_vars)[2] %>%
  furrr::future_map(function(x){
    #cat("Doing model: ", x," of ", dim(combinatoria_vars)[2],"\n")
    cat(sprintf("Doing model: %d of %d \n", x , dim(combinatoria_vars)[2]))

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

###############
###############(end) parallel


#after last execution this came out:

#4: UNRELIABLE VALUE: Future (‘<none>’) unexpectedly generated random numbers without specifying argument '[future.]seed'. There is a risk that those random numbers are not statistically sound and the overall results might be invalid. To fix this, specify argument '[future.]seed', e.g. 'seed=TRUE'. This ensures that proper, parallel-safe random numbers are produced via the L'Ecuyer-CMRG method. To disable this check, use [future].seed=NULL, or set option 'future.rng.onMisuse' to "ignore". 


#####continuation

  procs <- lapply(1:length(modelos),function(x) {
    proc <- modelos[[x]][[2]]
  })
  procs <- do.call("rbind.data.frame",procs)
  procs$auc_pmodel <- as.factor(procs$auc_pmodel)

  m1 <- lm(auc_ratio ~ auc_pmodel, data = procs)
  model_means <- sapply(levels(procs$auc_pmodel), function(y){
    model_index <- which(procs$auc_pmodel == y)
    media_model <- mean(procs[model_index,1],na.rm=T)
    return(media_model)
  })

  best_model <-names(model_means)[which(model_means==max(model_means,na.rm = TRUE))]

  models_meta_data <- lapply(1:length(modelos), function(x){
    matadata <- modelos[[x]][[3]]
  })

  best_model_metadata <- modelos[[as.numeric(best_model)]][[3]]

  sp.temp.best.model <- list(sp_coords = this_species$sp_coords,
                             coords_env_data_all = this_species$coords_env_data_all,
                             env_data_train = this_species$env_data_train,
                             env_data_test = this_species$env_data_test,
                             test_data = this_species$test_data,
                             sp_occs_year = this_species$sp_occs_year,
                             oocs_data = this_species$oocs_data,
                             lon_lat_vars = this_species$lon_lat_vars,
                             layers_path_by_year = this_species$layers_path_by_year,
                             best_model_metadata= best_model_metadata,
                             ellipsoid_level =ellipsoid_level,
                             pROC_table = procs,
                             models_meta_data=models_meta_data)
  class(sp.temp.best.model) <- c("list", "sp.temporal.modeling","sp.temporal.env","sp.temp.best.model")

ponca_mask <- raster::raster("/shared_volume/Ponca_DV/poncamask.tif")

#create dir /shared_volume/new_model_parallel

temporal_projection(this_species = sp.temp.best.model,
                      save_dir = "/shared_volume/new_model_parallel",
                      sp_mask = ponca_mask,
                      crs_model = NULL,
                      sp_name ="pan_onca",
                      plot3d = FALSE)

#out:

...
[1] "/shared_volume/new_model_parallel/temporal_modeling_pan_onca/2011"
[1] "/shared_volume/new_model_parallel/temporal_modeling_pan_onca/2012"
[1] "/shared_volume/new_model_parallel/temporal_modeling_pan_onca/2013"
[1] "/shared_volume/new_model_parallel/temporal_modeling_pan_onca/2014"
[1] "/shared_volume/new_model_parallel/temporal_modeling_pan_onca/niche_comparations_results/final_results_pan_onca"
     comparation suit_change
-1  2004vs.X2005    44.70591
-11 2004vs.X2006    39.48869
-12 2004vs.X2007    36.19155
-13 2004vs.X2008    38.58205
-14 2004vs.X2009    36.10716
-15 2004vs.X2010    39.21707
-16 2004vs.X2011    26.07860
-17 2004vs.X2012    19.24669
-18 2004vs.X2013    12.32983
-19 2004vs.X2014    10.64748