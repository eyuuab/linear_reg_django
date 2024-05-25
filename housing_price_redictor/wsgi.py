
import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'housing_price_redictor.settings')

application = get_wsgi_application()

app = application
