from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoLarsIC
import numpy as np

class LARS_model():
    
    def __init__(self,
                 max_iter: int,
                 tol: float,
                 selection: str):

        self.max_iter = max_iter
        self.tol = tol
        self.selection = selection
        self.lassoLarsAIC = LassoLarsIC(max_iter = self.max_iter)
        
    def fit(self, X, y):
        self.lassoLarsAIC.fit(X,y)
        self.alpha = self.lassoLarsAIC.alpha_
        self.lasso_model = Lasso(alpha = self.alpha,
                                 max_iter = self.max_iter, 
                                 tol = self.tol, 
                                 selection = self.selection)
        
        self.fitted_model = self.lasso_model.fit(X, y)
        self.coef_ = self.fitted_model.coef_
        return self.fitted_model

    def predict(self, X):
        return self.lasso_model.predict(X)