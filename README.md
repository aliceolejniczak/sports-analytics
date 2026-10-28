# Sports Analytics Task

Projet de data pipeline et de modélisation pour analyser la relation entre charge d'entraînement, sommeil, bien-être, nutrition et blessures dans un suivi longitudinal sportif.

## Description

Ce projet agrège des données provenant de plusieurs sources :

- PMsys : questionnaires de bien-être, sRPE et blessures ;
- Fitbit : activité, sommeil, calories, exercices, fréquence cardiaque ;
- Google Docs : repas déclarés, poids, hydratation, alcool ;
- food-images : photos de repas et métadonnées EXIF pour estimation nutritionnelle.

Le pipeline produit ensuite :

- un dataset journalier agrégé par participant et par date ;
- un dataset nettoyé prêt pour l'analyse ;
- un dataset de modélisation avec variables utiles pour l'apprentissage ;
- des rapports EDA et de modélisation.

Deux tâches de prédiction sont couvertes :

- estimation d'un indice de performance entre 0 et 100 ;
- prédiction du risque de blessure dans les 3 jours suivants.

## Structure du dépôt

```text
.
├── data/
│   ├── raw/
│   │   ├── p01/
│   │   ├── p03/
│   │   └── p05/
│   └── processed/
│       ├── raw_merged_dataset.csv
│       ├── test_dataset.csv
│       ├── model_dataset.csv
│       └── meals_metadata.csv
├── src/
│   ├── build_dataset.py
│   ├── image_processing.py
│   ├── preprocess.py
│   ├── eda.py
│   └── train.py
├── reports/
│   ├── eda/
│   ├── preprocessing/
│   ├── modeling/
│   └── project_report.md
├── notebooks/
│   └── 01_EDA_and_cleaning.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

Créer un environnement virtuel et installer les dépendances :

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Le traitement des images utilise le modèle pré-entraîné `nateraw/food`. À la première exécution, ses poids sont téléchargés depuis Hugging Face, donc une connexion Internet est nécessaire.

## Exécution complète

Depuis la racine du projet :

```bash
venv/bin/python src/build_dataset.py
venv/bin/python src/preprocess.py
venv/bin/python src/eda.py
venv/bin/python src/train.py
```

### Rôle de chaque étape

1. `src/build_dataset.py` : fusionne les données et construit le dataset agrégé par `participant_id + date`.
2. `src/preprocess.py` : nettoie les types, les unités, les valeurs manquantes, les outliers et crée les cibles.
3. `src/eda.py` : génère le rapport d'exploration des distributions et de la structure.
4. `src/train.py` : entraîne et évalue les modèles et exporte les métriques.

## Datasets produits

### Dataset complet

`data/processed/test_dataset.csv` contient les données agrégées, nettoyées et enrichies de toutes les sources. Une ligne correspond à un participant pour un jour.

### Dataset de modélisation

`data/processed/model_dataset.csv` contient les variables sélectionnées pour l'apprentissage. Il est prêt à être utilisé par les modèles, avec les variables textuelles exclues de la partie apprentissage.

### Données nutritionnelles issues des images

`data/processed/meals_metadata.csv` contient les estimations de calories, protéines, glucides et lipides par repas détecté à partir des photos.

## Rapports générés

Les sorties sont disponibles dans `reports/` :

- `reports/eda/` : structure, statistiques descriptives, distributions ;
- `reports/preprocessing/` : choix de nettoyage, valeurs manquantes et outliers ;
- `reports/modeling/` : métriques de performance et de blessure ;
- `reports/project_report.md` : rapport plus détaillé du projet.

## Modèles

### Performance

- Modèle : `RandomForestRegressor`
- Objectif : prédire un indice de performance sur 100
- Métriques : RMSE, MAE, R²

### Blessure

- Modèle : `XGBClassifier`
- Objectif : prédire un risque de blessure dans les 3 jours suivants
- Métriques : balanced accuracy, précision, rappel, F1, ROC-AUC, PR-AUC

## Résultats actuels

Les résultats doivent être interprétés avec prudence car le jeu de données est petit et le nombre d’athlètes limité.

### Performance

- RMSE : 18.30
- MAE : 13.69
- R² : -1.43
- baseline RMSE : 15.15

### Blessure

- balanced accuracy : 0.552
- precision : 0.304
- recall : 0.333
- F1 : 0.318
- ROC-AUC : 0.557
- PR-AUC : 0.305

Le modèle est légèrement meilleur que la baseline sur la blessure, mais le signal prédictif reste faible.

## Limites du projet

- seulement 3 participants ;
- durée d'observation limitée ;
- estimation nutritionnelle par portion standard à partir des images ;
- données incomplètes imputées dans le pipeline ;
- résultats non validés cliniquement.

## Contribution / développement

Pour modifier le pipeline ou ajouter de nouvelles variables :

- commencer par la source de données concernée ;
- ajuster les étapes de nettoyage dans `src/preprocess.py` ;
- vérifier l'impact sur les datasets produits ;
- relancer la pipeline complète pour reproduire les rapports.

