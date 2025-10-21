# ------------------
## Step 0
## Hello model!

cdc_risk <- function(x, base_risk = 0.00003) {
  rratio <- rep(7900, nrow(x))
  rratio[which(x$Age < 84.5)] <- 2800
  rratio[which(x$Age < 74.5)] <- 1100
  rratio[which(x$Age < 64.5)] <- 400
  rratio[which(x$Age < 49.5)] <- 130
  rratio[which(x$Age < 39.5)] <- 45
  rratio[which(x$Age < 29.5)] <- 15
  rratio[which(x$Age < 17.5)] <- 1
  rratio[which(x$Age < 4.5)]  <- 2
  rratio * base_risk
}

steve <- data.frame(Age = 25, Diabetes = "Yes")
cdc_risk(steve)


library("DALEX")
model_cdc <- DALEX::explain(cdc_risk,
                            predict_function = function(m, x) m(x),
                            type  = "classification",
                            label = "CDC")
predict(model_cdc, steve)


# ------------------
## Step 1
## Data Exploration (EDA)

covid_spring <- read.table("covid_spring.csv", sep =";", header = TRUE, stringsAsFactors = TRUE)
covid_summer <- read.table("covid_summer.csv", sep =";", header = TRUE, stringsAsFactors = TRUE)


library("ggplot2")
ggplot(covid_spring) + geom_histogram(aes(Age, fill = Death))

library("tableone")
CreateTableOne(vars = colnames(covid_spring)[1:10],
               data = covid_spring, 
               strata = "Death")


selected_vars <- c("Gender", "Age", "Cardiovascular.Diseases", 
                   "Diabetes", "Neurological.Diseases", "Kidney.Diseases",
                   "Cancer", "Death")

# use only selected variables
covid_spring <- covid_spring[,selected_vars]
covid_summer <- covid_summer[,selected_vars]

# ------------------
## Step 2
## Model performance

model_cdc <-  DALEX::explain(cdc_risk,
                                 predict_function = function(m, x) m(x),
                                 data  = covid_summer,
                                 y     = covid_summer$Death == "Yes",
                                 type  = "classification",
                                 label = "CDC")

mp_cdc <- model_performance(model_cdc, cutoff = 0.1)
mp_cdc

plot(mp_cdc, geom = "roc")

plot(mp_cdc, geom = "lift")


# ------------------
## Step 3
## Grow a tree

library("partykit")
tree <- ctree(Death ~., covid_spring, 
              control = ctree_control(alpha = 0.0001))
plot(tree)



model_tree <- DALEX::explain(tree,
                             predict_function = function(m, x) 
                               predict(m, x, type = "prob")[,2],
                             data = covid_summer,
                             y = covid_summer$Death == "Yes",
                             type = "classification", label = "Tree")

(mp_tree <- model_performance(model_tree, cutoff = 0.1))

plot(mp_tree, mp_cdc, geom="roc")

# ------------------
## Step 4
## Plant a forest

library("ranger")

forest <- ranger(Death ~., covid_spring, probability = TRUE)
forest


library("mlr3")
(covid_task <- TaskClassif$new(id = "covid_spring", 
                               backend = covid_spring, 
                               target = "Death",  positive = "Yes"))


library("mlr3learners")
library("ranger")
covid_ranger <- lrn("classif.ranger", predict_type="prob", 
                    num.trees=25)

covid_ranger$train(covid_task)

model_ranger <- explain(covid_ranger,
                        predict_function = function(m,x)
                          predict(m, x, predict_type = "prob")[,2],
                        data = covid_summer,
                        y = covid_summer$Death == "Yes",
                        type = "classification", label = "Ranger")

(mp_ranger <- model_performance(model_ranger))

plot(mp_ranger, mp_tree, mp_cdc, geom= "roc")

# ------------------
## Step 5
## Hyperparameter Optimisation

library("mlr3tuning")
library("paradox")
search_space = ps(
  num.trees = p_int(lower = 50, upper = 500),
  max.depth = p_int(lower = 1, upper = 10),
  splitrule = p_fct(levels = c("gini", "extratrees"))
)

tuned_ranger = AutoTuner$new(
  learner    = covid_ranger,
  resampling = rsmp("cv", folds = 5),
  measure    = msr("classif.auc"),
  search_space = search_space,
  terminator = trm("evals", n_evals = 10),
  tuner      = tnr("random_search") )

tuned_ranger$train(covid_task)
tuned_ranger$tuning_result
#    num.trees max.depth    minprop splitrule
# 1:       264         9 0.06907318      gini
#    learner_param_vals  x_domain classif.auc
# 1:          <list[4]> <list[4]>   0.9272979


model_tuned = DALEX::explain(tuned_ranger,
                       predict_function = function(m,x)
                           m$predict_newdata(newdata = x)$prob[,2],
                       data = covid_summer,
                       y = covid_summer$Death == "Yes",
                       type = "classification", label = "AutoTune")


(mp_tuned <- model_performance(model_tuned))
# Measures for:  classification
# recall     : 0.02575107 
# precision  : 0.4 
# f1         : 0.0483871 
# accuracy   : 0.9764 
# auc        : 0.9447171


plot(mp_tuned, mp_ranger, mp_tree, mp_cdc, geom = "roc")


# Step 5B

library("rms")
lmr_rcs <- lrm(Death ~ Gender + rcs(Age, 3) + Cardiovascular.Diseases + Diabetes +
                 Neurological.Diseases + Kidney.Diseases + Cancer, covid_spring)
lmr_rcs


model_lmr_rcs <-  DALEX::explain(lmr_rcs,
                                 data = covid_summer[,-8],
                                 y = covid_summer$Death == "Yes",
                                 type = "classification",
                                 label = "LMR",
                                 verbose = FALSE)

mp_lrm <- model_performance(model_lmr_rcs, cutoff = 0.1)
mp_lrm

plot(mp_ranger, mp_tree, mp_tuned, mp_lrm, geom = "roc")

do.call(rbind, list(tree  = mp_tree$measures,
                    lrm    = mp_lrm$measures,
                    forest = mp_ranger$measures,
                    tuned  = mp_tuned$measures))

# ------------------
## Step 6
## Variable Importance

mpart_forest <- model_parts(model_ranger)
mpart_forest

mpart_forest <- model_parts(model_ranger, type = "difference")
mpart_forest

plot(mpart_forest, show_boxplots = FALSE, bar_width=4) +
  ggtitle("Variable importance","")



# ------------------
## Step 7
## Break down

Steve <- data.frame(Gender = factor("Male", c("Female", "Male")),
                    Age = 76,
                    Cardiovascular.Diseases = factor("Yes", c("No", "Yes")), 
                    Diabetes = factor("No", c("No", "Yes")), 
                    Neurological.Diseases = factor("No", c("No", "Yes")), 
                    Kidney.Diseases = factor("No", c("No", "Yes")), 
                    Cancer = factor("No", c("No", "Yes")))
predict(model_ranger, Steve)

ppart_tree <- predict_parts(model_tree, Steve)
plot(ppart_tree)

ppart_forest <- predict_parts(model_tree, Steve, type = "shap")
plot(ppart_forest)

ppart_lmr_rcs <- predict_parts(model_lmr_rcs, Steve, type = "shap")
plot(ppart_lmr_rcs)


ppart_forest <- predict_parts(model_ranger, Steve, type = "shap")
pl1 <- plot(ppart_forest) + ggtitle("Shapley values for Ranger")

ppart_forest <- predict_parts(model_ranger, Steve)
pl2 <- plot(ppart_forest) + ggtitle("Break-down for Ranger")

library("patchwork")
pl1 + pl2

# ------------------
## Step 8
## Ceteris Paribus

cp_ranger <- predict_profile(model_ranger, Steve)
cp_ranger

plot(cp_ranger, variables = "Age")
plot(cp_ranger, variables = "Cardiovascular.Diseases", 
     categorical_type = "lines")

cp_cdc <- predict_profile(model_cdc, Steve)
cp_tree <- predict_profile(model_tree, Steve)
cp_tune <- predict_profile(model_tuned, Steve)

plot(cp_cdc, cp_tree, cp_ranger, cp_tune, variables = "Age")

predict_parts(model_ranger, Steve, type = "oscillations")


# ------------------
## Step 9
## Partial Dependence

mprof_forest <- predict_profile(model_ranger, Steve, "Age")
plot(mprof_forest)


mprof_forest <- predict_profile(model_ranger, variable_splits = list(Age=0:100), Steve)
mprof_tree <- predict_profile(model_tree, variable_splits = list(Age=0:100), Steve)
mprof_lmr_rcs <- predict_profile(model_lmr_rcs, variable_splits = list(Age=0:100), Steve)

plot(mprof_forest, mprof_lmr_rcs, mprof_tree)


mprof_forest <- predict_profile(model_ranger, variables = "Age", Steve)
pl1 <- plot(mprof_forest) + ggtitle("Ceteris paribus for Ranger") 

mprof_forest2 <- predict_profile(model_ranger, variables = "Cardiovascular.Diseases", Steve)
pl2 <- plot(mprof_forest2, variable_type = "categorical", variables = "Cardiovascular.Diseases", categorical_type = "lines")  + ggtitle("Ceteris paribus for Ranger")

library("patchwork")
pl1 + pl2



mprof_forest <- model_profile(model_ranger, "Age")
plot(mprof_forest) +
  ggtitle("PD profile","")


mgroup_forest <- model_profile(model_ranger, variable_splits = list(Age = 0:100), 
                               groups = "Diabetes")
plot(mgroup_forest)+
  ggtitle("PD profiles for groups","") + ylab("") + theme(legend.position = "top")


