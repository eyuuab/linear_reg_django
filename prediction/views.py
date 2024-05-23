import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings

# Load the dataset and drop the 4th feature
file_path = os.path.join(settings.BASE_DIR, 'prediction', 'data', 'housing_price_dataset.csv')
data = pd.read_csv(file_path)
data = data.drop(data.columns[3], axis=1)

# Assume the last column is the target variable
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

@csrf_exempt
def predict(request):
    if request.method == 'POST':
        # Extract data from the request
        input_data = request.POST.get('input_data')
        input_data = [float(i) for i in input_data.split(',')]
        
        # Make prediction
        prediction = model.predict([input_data])
        
        # Return the result
        return JsonResponse({'prediction': prediction[0]})
    return JsonResponse({'error': 'Invalid request method'}, status=400)
