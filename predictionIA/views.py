import json, io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from django.http import JsonResponse, HttpResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout
from django.contrib import messages

from sklearn.tree import plot_tree
from .models import Prediction
from .form import CustomUserCreationForm
from .utils import predict_from_dict, MODEL_CLASS

# --- liste réduite des features + unités ---
FEATURE_HINT = [
    ("Vitesse_vent", "[m/s]"),
    ("Hauteur_vagues", "[m]"),
    ("Vitesse_courant", "[m/s]"),
    ("Taux_corrosion", "[mn/ans]"),
    ("Periode_vagues", "[s]"),
]

# --- auth ---
def inscription(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, 'Inscription réussie !')
            return redirect('predictionIA:index')
        else:
            for field in form:
                for error in field.errors:
                    messages.error(request, error)
    else:
        form = CustomUserCreationForm()
    return render(request, 'inscription.html', {'form': form})

def index(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('predictionIA:chat')
        else:
            messages.error(request, "Nom d'utilisateur ou mot de passe incorrect")
    return render(request, "index.html")

@login_required
def chat(request):
    return render(request, "chat.html", {"feature_hint": FEATURE_HINT})

def deconnexion(request):
    logout(request)
    return redirect('predictionIA:index')

# --- prédiction ---
@csrf_exempt
@login_required
def predict_ajax(request):
    if request.method != "POST":
        return HttpResponseBadRequest("Only POST allowed")
    try:
        payload = json.loads(request.body.decode("utf-8"))
        features = payload.get("features", payload)

        result = predict_from_dict(features)

        Prediction.objects.create(
            user=request.user,
            input_features=features,
            etat_label=result["etat_label"],
            etat_proba=result["etat_proba"],
            duree_restant=result["duree_restant"]
        )

        return JsonResponse({"success": True, "result": result})
    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)}, status=400)

@login_required
def history(request):
    predictions = Prediction.objects.filter(user=request.user).order_by("-created_at")
    return render(request, "history.html", {"predictions": predictions})

def tree_png(request):
    if not hasattr(MODEL_CLASS, "estimators_"):
        return HttpResponseBadRequest("Model has no trees to show")

    # Choix de l’indice d’arbre
    i = int(request.GET.get("i", 0))
    i = max(0, min(i, len(MODEL_CLASS.estimators_) - 1))
    estimator = MODEL_CLASS.estimators_[i]

    # Création du plot
    fig, ax = plt.subplots(figsize=(12, 8))
    feature_names = getattr(estimator, "feature_names_in_", None)

    plot_tree(
        estimator,
        feature_names=feature_names,
        max_depth=3,
        filled=True,
        ax=ax
    )

    # Export en PNG
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    return HttpResponse(buf.getvalue(), content_type="image/png")

