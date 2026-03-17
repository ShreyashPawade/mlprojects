## confusion matrix and classification report ,r2 sqare and mean absolute error

## train diff models and find out best model
## try each and every algorithm 
## save the best model

import os
import sys

from dataclasses import dataclass


from sklearn.ensemble import (AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor    

from src.exception import CustomException
from src.logger import logging
from src.utlis import save_object,evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path= os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Spiliting training and test input data")
            x_train,y_train,x_test,y_test=(
                
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            models={
                
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "Linear Regression":LinearRegression(),
                "K-Neighbors Regressor":KNeighborsRegressor(),
                "XGBRegressor":XGBRegressor(),
                "CatBoosting Regressor":CatBoostRegressor(verbose=False),
                "AdaBoost Regressor":AdaBoostRegressor(),
                        
                
                
                
                
                
                
                
            }
            
            model_report:dict=evaluate_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,
                                             models=models)
            
            best_model_score= max(sorted(model_report.values()))
            
            best_model_name= list(model_report.keys())[
                list(model_report.values()).index(best_model_score)]
            
            
            
            best_model=models[best_model_name]
            
            
            if best_model_score<0.6:
                raise CustomException("No best model found")
            
            
            logging.info(f"Best found model on both training and testing dataset is {best_model_name} with r2 score of {best_model_score}")
            
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted=best_model.predict(x_test)
            r2=r2_score(y_test,predicted)
            
            
        
        
        except Exception as e:
            raise CustomException(e,sys)
                
        
        