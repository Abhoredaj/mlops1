import json
import logging
import os
import joblib
import pytest
from prediction_service.prediction import form_response, api_response
import prediction_service

input_data = {
    "incorrect_range": 
    {"accelerations": 7897897, 
    "prolongued_decelerations": 555, 
    "abnormal_short_term_variability": 99, 
    "percentage_of_time_with_abnormal_long_term_variability": 99, 
    "histogram_mode": 200, 
    "histogram_mean": 789
    },

    "correct_range":
    {"accelerations": 0.01, 
    "prolongued_decelerations": 0.001, 
    "abnormal_short_term_variability": 45.0, 
    "percentage_of_time_with_abnormal_long_term_variability": 45.0, 
    "histogram_mode": 90.0, 
    "histogram_mean": 100.0
    },

    "incorrect_col":
    {"accelerations": 0.01, 
    "prolongued decelerations": 0.001, 
    "abnormal short_term_variability": 45.0, 
    "percentage of_time_with_abnormal_long_term_variability": 45.0, 
    "histogram mode": 90.0, 
    "histogram mean": 100.0
    }
}

TARGET_range = {
    "min": 1.0,
    "max": 3.0
}

def test_form_response_correct_range(data=input_data["correct_range"]):
    res = form_response(data)
    assert  TARGET_range["min"] <= res <= TARGET_range["max"]

def test_api_response_correct_range(data=input_data["correct_range"]):
    res = api_response(data)
    assert  TARGET_range["min"] <= res["response"] <= TARGET_range["max"]

def test_form_response_incorrect_range(data=input_data["incorrect_range"]):
    with pytest.raises(prediction_service.prediction.NotInRange):
        res = form_response(data)

def test_api_response_incorrect_range(data=input_data["incorrect_range"]):
    res = api_response(data)
    assert res["response"] == prediction_service.prediction.NotInRange().message

def test_api_response_incorrect_col(data=input_data["incorrect_col"]):
    res = api_response(data)
    assert res["response"] == prediction_service.prediction.NotInCols().message 