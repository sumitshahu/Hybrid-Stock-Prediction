from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('search_stocks/', views.search_stocks, name='search_stocks'),
    path('stock/<str:stock_name>/', views.stock_details, name='stock_details'),  # Keep only one stock details path

    

    # API Endpoints
    path('api/live_price/<str:stock_name>/', views.get_live_price, name='live_price'),
    path('api/stock_data/<str:stock_name>/', views.get_stock_data, name='stock_data'),

    # Updated Forecast API to support model selection
    # path('api/forecast/<str:stock_name>/<str:model>/', views.get_forecast, name='forecast'),
]
