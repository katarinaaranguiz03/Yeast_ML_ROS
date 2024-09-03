# Code for "Machine learning reveals genes impacting oxidative stress resistance across yeasts"
Within this repository are all required materials to run a semi-automated ML pipeline and produce SHAP values using our model's output.

## File Descriptions
- **PRACTICE_smallset.tsv**: This .tsv file contains a condensed version of the dataset used in this manuscript for practice.
- **ml_pipeline_env.yml**: This .yml file contains all required dependencies for running all scripts.
- **ML_preprocess.py**: This .py script prepares input data matrix for use by subsequent algorithms.
- **test_set.py**: This .py script sets aside a percentage of instances to be used for model validation. These instances are NOT used to  train the model.
- **Feature_Selection.py**: This .py script selects informative features for use by the classification algorithm.
- **ML_classification.py**: This .py script classifies instances in the provided input matrix based on feature patterns and relationships.
- **ML_ROS_rf_50.sh**: This .sh file contains all executible files ran to generate the model used in this manuscript.
- **ML_ROS_rf_50_shapdf.py**: This .py script parses the model's .pkl file, estimates SHAP values, and compiles those SHAP values in a Pandas Dataframe. 
- **ML_ROS_model_output.pkl**: This .pkl file contains all model output from the model generated for this manuscript in a compressed format.
- **ML_functions.py**: This .py script is utilized by the classification script to provide required functions.

## Authors and acknowledgment
Feature_Selection.py, ML_classification.py, and test_set.py were directly copied from https://github.com/ShiuLab/ML-Pipeline.
ml_pipeline_env.yml, ML_ROS_rf_50.sh, ML_ROS_rf_50_shapdf.py, and ML_ROS_model_output.pkl are original for this project.

## License
Licensing for the Shiu Lab's scripts can be found at https://github.com/ShiuLab/ML-Pipeline.
