from django.urls import path
from .views import *

app_name = "predictionIA"

urlpatterns = [
    path("", index, name="index"),
    path("chat/", chat, name="chat"),
    path('inscription/', inscription, name='inscription'),
    path('deconnexion/',deconnexion, name='deconnexion'),
    path("predict/", predict_ajax, name="predict"),
    path("tree.png", tree_png, name="tree"),
    path("history/", history, name="history"),

]
