import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings


file_path = os.path.join(settings.BASE_DIR, 'prediction', 'data', 'housing_price_dataset.csv')
data = pd.read_csv(file_path)


data = data.drop(data.columns[3], axis=1)


X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error of the model: {mse}')
r2 = r2_score(y_test, y_pred)
print(f'R-squared value of the model: {r2}')


@csrf_exempt
def predict(request):
    if request.method == 'POST':
        input_data = request.POST.get('input_data')
        input_data = [float(i) for i in input_data.split(',')]
        prediction = model.predict([input_data])
        return JsonResponse({'prediction': prediction[0]})
    return JsonResponse({'error': 'Invalid request method'}, status=400)

@csrf_exempt
def data_sample(request):
    if request.method == 'GET':
        sample_data = data.sample(10).to_dict(orient='records')
        return JsonResponse({'data_sample': sample_data})
    return JsonResponse({'error': 'Invalid request method'}, status=400)
