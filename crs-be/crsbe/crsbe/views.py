from django.http import JsonResponse
import joblib
from django.views.decorators.csrf import csrf_exempt
import json

# Sample data
#input_data1 = [[13.49, 4.12, 139452, 0.9, 69, 368, 2668]] 

# print("Predicted credit score:", predicted_score[0])
@csrf_exempt
def predict(request):
    data = json.loads(request.body)
    profit_margin = data.get('profit_margin')
    return_on_total_assets = data.get('return_on_total_assets')
    credit_limit = data.get('credit_limit')
    likelihood_of_failure_percentage = data.get('likelihood_of_failure_percentage')
    no_of_employees = data.get('no_of_employees')
    gearing = data.get('gearing')
    net_current_assets = data.get('net_current_assets')
    random_forest_model = joblib.load('crsbe/rf_model.pkl')
    input_data = [[profit_margin, return_on_total_assets,credit_limit, likelihood_of_failure_percentage, no_of_employees, gearing, net_current_assets]] 
    predicted_score = random_forest_model.predict(input_data)[0]
    predicted_score = predicted_score.item()
    return JsonResponse({'Predicted Score': predicted_score})