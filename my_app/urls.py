from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('search_stocks/', views.search_stocks, name='search_stocks'),
    path('stock/<str:stock_symbol>/', views.stock_details, name='stock_details'),
    path('stock/<str:stock_symbol>/historical_data/', views.get_historical_data, name='get_historical_data'),
]
