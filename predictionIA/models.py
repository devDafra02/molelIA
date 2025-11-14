from django.db import models
from django.contrib.auth.models import User

class Prediction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)  # utilisateur qui a fait la prédiction
    created_at = models.DateTimeField(auto_now_add=True)      # date et heure de la prédiction
    input_features = models.JSONField()                       # dictionnaire des valeurs entrées

    # Résultats classification
    etat_label = models.CharField(max_length=50)              
    etat_proba = models.FloatField(null=True, blank=True)     

    # Résultats régression
    duree_restant = models.FloatField(null=True, blank=True)  

    def __str__(self):
        return f"{self.user.username} - {self.etat_label} ({self.created_at})"
