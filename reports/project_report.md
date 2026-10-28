# Rapport de projet : Sports Analytics

## 1. Objectif et perimetre

L'objectif est de construire une base longitudinale journaliere permettant d'etudier la relation entre charge d'entrainement, sommeil, bien-etre, nutrition et blessures. L'unite d'observation choisie est le couple `(participant_id, date)` : chaque ligne decrit un athlete pour une journee donnee.

Le projet repond aux quatre volets suivants :

1. aggregation multi-source ;
2. pretraitement et analyse de qualite ;
3. analyse exploratoire de la structure et des distributions ;
4. creation de cibles et entrainement de deux modeles predictifs.

## 2. Aggregation des donnees

### Sources

Les donnees sont organisees par athlete (`p01`, `p03`, `p05`) et proviennent de :

- **PMsys** : wellness, sRPE et blessures ;
- **Fitbit** : pas, calories, distance, frequence cardiaque, exercices, sommeil et zones cardiaques ;
- **Google Docs** : repas, poids, hydratation et alcool ;
- **food-images** : images des repas et leurs metadonnees.

### Methode

`src/build_dataset.py` convertit les dates au niveau du jour, agrege les mesures intrajournalieres et effectue des jointures externes sur `participant_id` et `date`. Les jointures externes sont importantes car toutes les sources ne contiennent pas une observation chaque jour.

Les dates Google Docs sont interpretees avec `dayfirst=True`, car elles sont au format `dd/mm/YYYY`. Les dates PMsys et Fitbit sont des timestamps ISO et sont traitees separement.

Le dataset agrege comprend actuellement 456 lignes. La structure est reguliere : trois participants et 152 jours par participant.

## 3. Traitement des images et nutrition

Pour chaque image, `src/image_processing.py` :

1. cherche la date EXIF `DateTimeOriginal` ;
2. utilise la date de modification du fichier comme fallback si l'EXIF est absent ;
3. applique le modele pre-entraine Food-101 `nateraw/food` ;
4. conserve la classe predite et sa probabilite ;
5. associe la classe a une valeur calorique par portion standard ;
6. estime les proteines, glucides et lipides ;
7. agrege ces resultats par participant et par jour.

La formule des macronutriments utilise une repartition documentee de 20 % proteines, 50 % glucides et 30 % lipides :


proteines = calories * 0.20 / 4
glucides  = calories * 0.50 / 4
lipides   = calories * 0.30 / 9


Cette methode est plus reproductible qu'une valeur aleatoire et utilise un modele specialise pour reconnaitre les plats. Elle ne permet toutefois pas de mesurer la portion reelle. Les colonnes `food_model_confidence` et `calorie_estimation_confidence` rendent cette incertitude explicite.

## 4. Data preprocessing

Le traitement principal est dans `src/preprocess.py`.

### Gestion des types et des unites

- les dates sont converties avec `pd.to_datetime` ;
- les identifiants sont convertis en valeurs numeriques ;
- les autres champs quantitatifs sont convertis avec `pd.to_numeric` ;
- les durees Fitbit de sommeil sont converties en heures ;
- les durees Fitbit d'exercice sont converties en minutes ;
- les champs textuels sont conserves mais ne sont pas fournis directement aux modeles.

### Valeurs manquantes : options examinees

**Suppression des lignes.** Simple, mais inadaptee : elle ferait perdre des jours entiers et reduirait encore un dataset deja petit.

**Moyenne globale.** Rapide, mais elle efface les differences entre athletes et est sensible aux valeurs extremes.

**Mediane par participant.** Robuste et coherente pour le poids, l'activite et la nutrition. Elle est retenue pour ces variables.

**Propagation temporelle.** Le forward-fill puis backward-fill exploite la structure longitudinale des questionnaires et du sommeil. Cette methode est retenue pour les series wellness et sommeil.

**KNN ou MICE.** Plus avances, mais difficiles a justifier avec seulement trois participants. Ils peuvent aussi utiliser indirectement des informations futures dans une serie temporelle.

### Choix final pour les valeurs manquantes

- absence de sRPE ou de duree : `0`, interprete comme absence de seance ;
- wellness et sommeil : propagation par participant, puis mediane ;
- activite et nutrition : mediane du participant, puis mediane globale ;
- texte absent : `Unknown` ;
- compteurs sans observation : `0`.

### Detection et traitement des outliers

Deux niveaux sont utilises :

1. des bornes metier pour les echelles bornees, par exemple `0-10` pour fatigue, stress, readiness et RPE ;
2. la regle statistique `1.5 * IQR` pour les variables numeriques suffisamment variees.

Les lignes ne sont pas supprimees : une journee d'entrainement extreme peut etre reelle et informative. Les valeurs impossibles deviennent manquantes. Les valeurs extremes plausibles sont plafonnees aux bornes IQR, puis traitees par la strategie d'imputation.

Les audits sont sauvegardes dans :

- `reports/preprocessing/outlier_summary.csv` ;
- `reports/preprocessing/missing_summary.csv` ;
- `reports/preprocessing/preprocessing_choices.md`.

Le pipeline produit actuellement un dataset sans valeur manquante.

## 5. Overview de la base et EDA

`src/eda.py` genere les tableaux et graphiques demandes pour l'overview :

- nombre de lignes et de colonnes ;
- nombre de participants ;
- nombre de jours par participant ;
- nombre de colonnes numeriques et textuelles ;
- statistiques descriptives des variables principales ;
- distribution des pas, sommeil, calories, charge et performance ;
- distribution des blessures ;
- bilan des valeurs manquantes.

Les sorties sont dans `reports/eda/`.

Les groupes de variables sont :

- wellness : fatigue, stress, soreness, mood, readiness ;
- sommeil : duree, scores Fitbit, sommeil profond, efficacite ;
- activite : pas, calories, distance et exercices ;
- nutrition : calories, macronutriments, poids, hydratation, repas et alcool ;
- cibles : performance et blessure.

## 6. Creation des cibles

### Indice de performance

`performance_index` est un indice de 0 a 100 inspire d'une logique de Hooper Index adaptee aux variables disponibles. Il combine :

- readiness positif ;
- score global de sommeil positif ;
- fatigue negative ;
- soreness negative ;
- stress negatif.

Le score brut est ensuite normalise avec `MinMaxScaler` entre 0 et 100. Il s'agit d'un indice construit pour le projet, pas d'une mesure clinique validee.

### Risque de blessure

`injury_next_3_days` vaut 1 lorsqu'une blessure est observee dans la fenetre des trois jours suivants pour le meme participant. Cette fenetre est choisie car elle correspond a un horizon court et actionnable pour la gestion de charge et la prevention.

## 7. Modelisation

### Preparation temporelle

Les donnees sont triees par participant et par date. Les variables explicatives sont decalees d'un jour : le modele utilise les mesures du jour precedent pour predire la cible du jour. Cette precaution reduit la fuite d'information, en particulier pour `performance_index`, qui est construit a partir de variables wellness et sommeil.

Le decoupage est chronologique : 80 % des observations les plus anciennes pour l'entrainement et 20 % des observations les plus recentes pour le test. La validation interne utilise `TimeSeriesSplit`.

### Modele de performance

Un `RandomForestRegressor` est utilise car :

- la cible est continue ;
- les relations entre sommeil, charge et performance peuvent etre non lineaires ;
- le modele est robuste aux echelles et interactions ;
- il fonctionne avec un dataset de taille modeste.

Les hyperparametres explodes dans `src/train.py` sont :

- `n_estimators` : 100 ou 200 ;
- `max_depth` : `None` ou 10 ;
- `min_samples_split` : 2 ou 5.

Cette grille est choisie pour equilibrier flexibilite et stabilite. Un nombre trop faible d'arbres peut sous-apprendre ; un nombre trop grand, avec peu de donnees, augmente le risque de surapprentissage. De meme, une profondeur trop importante peut apprendre des motifs trop specifique a quelques jours, alors qu'un arbre plus limite est plus stable sur des donnees longitudinales courtes. Le parametre `min_samples_split` impose une division seulement si un nœud contient assez d'observations, ce qui limite les regles trop fines dans un echantillon peu volumineux.

La selection est realisee par `GridSearchCV` avec `TimeSeriesSplit`, ce qui garantit que l'optimisation se fait sur des validations temporelles et non sur des observations futuristes. Le score retenu est `neg_root_mean_squared_error`, ce qui correspond a une minimisation de l'erreur RMSE sur les donnees de test prospectives.

Les metriques sont RMSE, MAE et R2, comparees a une baseline qui predit la moyenne d'entrainement.

Resultats actuels :

- RMSE : 18.30 ;
- MAE : 13.69 ;
- R2 : -1.43 ;
- baseline RMSE : 15.15.

Le R2 negatif montre que le modele ne depasse pas la baseline sur le test. Il faut le signaler comme une limite, et non masquer ce resultat.

### Modele de blessure

Un `XGBClassifier` est utilise car :

- la cible est binaire ;
- les interactions entre charge, sommeil et bien-etre peuvent etre non lineaires ;
- `scale_pos_weight` compense le desequilibre entre blessures et non-blessures.

Les hyperparametres explodes dans `src/train.py` sont :

- `n_estimators` : 100 ou 200 ;
- `learning_rate` : 0.05 ou 0.1 ;
- `max_depth` : 3 ou 5.

Le choix de `learning_rate` faible est classique pour XGBoost sur petits jeux de donnees : il permet un apprentissage plus stable et reduit le risque de surapprentissage. Une profondeur moderee de 3 ou 5 limite la complexite des arbres lorsque la classe positive est rare. Le nombre d'arbres est egalement garde modere afin de ne pas suradapter les observations d'entraînement. La valeur `scale_pos_weight = neg_class / pos_class` est essentielle dans ce contexte : le dataset est fortement desequilibre, avec beaucoup plus de jours sans blessure que de jours avec blessure. Sans cette correction, le modele aurait tendance a predire la classe majoritaire et a obtenir des performances artificiellement faibles sur la classe positive.

La selection est realisee par `GridSearchCV` avec `TimeSeriesSplit` et le score `f1`. Ce choix est coherent pour un probleme de detection de blessure rare : on cherche a maximiser la qualite globale de la prediction positive sans ignorer le desequilibre de classes. Les metriques incluent balanced accuracy, precision, rappel, F1, ROC-AUC, PR-AUC et la matrice de confusion.

Resultats actuels :

- balanced accuracy : 0.552 ;
- precision : 0.304 ;
- rappel : 0.333 ;
- F1 : 0.318 ;
- ROC-AUC : 0.557 ;
- PR-AUC : 0.305 ;
- matrice : TN=54, FP=16, FN=14, TP=7.

Le modele est legerement meilleur que la baseline en balanced accuracy, mais le pouvoir predictif reste faible.

## 8. Fichiers de sortie

- `data/processed/raw_merged_dataset.csv` : aggregation brute ;
- `data/processed/test_dataset.csv` : dataset complet pretraite ;
- `data/processed/model_dataset.csv` : variables selectionnees pour l'apprentissage ;
- `data/processed/meals_metadata.csv` : agregation nutritionnelle des images ;
- `reports/eda/` : analyse exploratoire ;
- `reports/preprocessing/` : audits de nettoyage ;
- `reports/modeling/` : choix et metriques des modeles.

## 9. Limites et ameliorations

Les limites principales sont le nombre tres faible de participants, la duree d'observation, l'imputation de certaines absences, l'incertitude sur les portions alimentaires et la faible performance predictive obtenue.

Les ameliorations prioritaires seraient :

1. augmenter le nombre d'athletes et de saisons ;
2. obtenir les portions pesees ou des informations nutritionnelles validees ;
3. valider le modele sur des athletes completement independants ;
4. comparer davantage de baselines et de modeles ;
5. effectuer une calibration des probabilites de blessure ;
6. analyser les variables Fitbit dans leurs unites documentees d'origine.
