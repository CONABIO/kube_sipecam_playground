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



#save(modelos,file = "/shared_volume/modelos.RData")
#save(modelos,file = "/shared_volume/modelos_compressed.RData", compress=TRUE)

#shared_volume is mapped in /LUSTRE/MADMEX/tasks/2020

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

#create dir /shared_volume/new_model

temporal_projection(this_species = sp.temp.best.model,
                      save_dir = "/shared_volume/new_model",
                      sp_mask = ponca_mask,
                      crs_model = NULL,
                      sp_name ="pan_onca",
                      plot3d = FALSE)

#out:
...
[1] "/shared_volume/new_model/temporal_modeling_pan_onca/2014"
[1] "/shared_volume/new_model/temporal_modeling_pan_onca/niche_comparations_results/final_results_pan_onca"
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


#next is not necessary:

#save(sp.temp.best.model,file = "/shared_volume/sp_best_model.RData")

#save desaggregated:

save(e_test,file = "/shared_volume/results_dir/e_test.RData")
save(best_model_metadata,file = "/shared_volume/results_dir/best_model_metadata.RData")
save(ellipsoid_level,file = "/shared_volume/results_dir/ellipsoid_level.RData")
save(procs,file = "/shared_volume/results_dir/pROC_table.RData")
save(models_meta_data,file = "/shared_volume/results_dir/models_meta_data.RData")


#create dir /shared_volume/results_dir

save(e_test$sp_coords,file = "/shared_volume/results_dir/sp_coords.RData")
save(e_test$coords_env_data_all,file = "/shared_volume/results_dir/coords_env_data_all.RData")
save(e_test$env_data_train,file = "/shared_volume/results_dir/env_data_train.RData")
save(e_test$env_data_test,file = "/shared_volume/results_dir/env_data_test.RData")
save(e_test$test_data,file = "/shared_volume/results_dir/test_data.RData")
save(e_test$sp_occs_year,file = "/shared_volume/results_dir/sp_occs_year.RData")
save(e_test$oocs_data,file = "/shared_volume/results_dir/oocs_data.RData")
save(e_test$lon_lat_vars,file = "/shared_volume/results_dir/lon_lat_vars.RData")
save(e_test$layers_path_by_year,file = "/shared_volume/results_dir/layers_path_by_year.RData")
