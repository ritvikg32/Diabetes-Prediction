from django.shortcuts import render
from django.http import HttpResponse
import joblib
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Create your views here.
def home(request):
    return render(request,"form.html")

def get_std_scaler():
    std = joblib.load('standard_scaler.sav')
    return std



def result(request):
    model = joblib.load('diabetes_model.sav')

    

    input_list = []

    user_name = request.GET['name']
    input_list.append(request.GET['pregnant'])
    input_list.append(request.GET['glucose'])
    input_list.append(request.GET['bp'])
    input_list.append(request.GET['thickness'])
    input_list.append(request.GET['insulin'])
    input_list.append(request.GET['bmi'])
    input_list.append(request.GET['age'])
    input_list.append(request.GET['pf'])

    input_list = [input_list]

    

    

  
    std=get_std_scaler()
    
    # X_train = std.fit_transform(input_list)

    X_test=std.transform(input_list)
    print(X_test)

    
    ans = model.predict(X_test)

    sentence = 'are'
    
    if ans[0] != 1:
        sentence = 'are not'
    
    


    return render(request,"result.html",{'ans':ans[0], 'user_name':user_name, 'result':sentence})


