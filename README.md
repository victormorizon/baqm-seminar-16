# The file is structured in the following way:

* One should create a "data" folder that contains all data files (from Aegon, PC4 from CBS and the livability index from the government)
* "baseline" contains the  different models that were tried and tuned for the baseline model (each file corresponds to a model). model_victor (lightgbm) contains the final model with 'best' parameters. It also contains the code the apply the predictions and get the plots (and data for tables) from the report
* "customer_segmentation" contains the code for customer segmentations ('victor' being the final one that was chosen, segments.csv is the assigned segment per customer)
* "first_stage", "first_stage_2" and "second_stage" contain files that were used for training and exploration (no final model)
* "reference" contains the files given by Aegon to us
* "data_prep" is the file that was used to de-panelise the data and augment it with the external data
* "plots" contains all the plots used in the report
* "full_model" essentially contains all the data used for the final DML model (including sensitivity analysis and ATE). It is decomposed as follows:
  * "functions" contains all the functions used to run the entire model (it is used as an import)
  * "main" imports functions from the previous files, as well as the data and runs the entire DoubleML model. It then prints and plots the ATE, GATE's, robustness valuea and the sensitivity (contour) plots. It also makes some of the result plots from the report.
