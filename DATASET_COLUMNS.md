# Documentation des Colonnes du Dataset Diabetes

## Vue d'ensemble

Ce document explique en détail toutes les colonnes présentes dans les datasets générés pour ce projet MLOps.

---

## Colonnes Originales (11 colonnes)

Ces colonnes proviennent du dataset Diabetes de Scikit-learn et sont **normalisées** (valeurs entre environ -2 et 2).

### 1. age - Âge
- **Type**: Numérique (normalisé)
- **Description**: Âge du patient
- **Signification médicale**: Facteur de risque important - le risque de diabète augmente avec l'âge
- **Interprétation**: Valeurs positives = plus âgé, valeurs négatives = plus jeune

### 2. sex - Sexe
- **Type**: Numérique (normalisé)
- **Description**: Sexe biologique du patient
- **Valeurs originales**: 1 = masculin, 2 = féminin
- **Signification médicale**: Certaines formes de diabète ont des prévalences différentes selon le sexe

### 3. bmi - Body Mass Index (IMC)
- **Type**: Numérique (normalisé)
- **Description**: Indice de masse corporelle = Poids (kg) / Taille² (m²)
- **Signification médicale**: **Facteur de risque majeur** pour le diabète de type 2
- **Interprétation**: 
  - Valeurs négatives = IMC faible (sous-poids/normal)
  - Valeurs positives = IMC élevé (surpoids/obésité)
- **Importance**: Un IMC élevé est fortement corrélé à la résistance à l'insuline

### 4. bp - Blood Pressure (Pression artérielle)
- **Type**: Numérique (normalisé)
- **Description**: Pression artérielle moyenne
- **Signification médicale**: L'hypertension est souvent associée au diabète (syndrome métabolique)
- **Interprétation**: 
  - Valeurs négatives = pression normale/basse
  - Valeurs positives = pression élevée/hypertension

### 5. s1 - TC (Total Cholesterol)
- **Type**: Numérique (normalisé)
- **Description**: Cholestérol total sérique
- **Signification médicale**: Marqueur du profil lipidique, important pour le risque cardiovasculaire
- **Valeurs normales**: < 200 mg/dL (souhaitable)
- **Importance**: Le diabète affecte le métabolisme des lipides

### 6. s2 - LDL (Low-Density Lipoprotein)
- **Type**: Numérique (normalisé)
- **Description**: Lipoprotéines de basse densité ("mauvais cholestérol")
- **Signification médicale**: Principal facteur de risque cardiovasculaire
- **Valeurs normales**: < 100 mg/dL (optimal)
- **Importance**: Les diabétiques ont souvent un LDL élevé

### 7. s3 - HDL (High-Density Lipoprotein)
- **Type**: Numérique (normalisé)
- **Description**: Lipoprotéines de haute densité ("bon cholestérol")
- **Signification médicale**: Protecteur contre les maladies cardiovasculaires
- **Valeurs normales**: > 60 mg/dL (souhaitable)
- **Importance**: Les diabétiques ont souvent un HDL bas

### 8. s4 - TCH (Total Cholesterol / HDL Ratio)
- **Type**: Numérique (normalisé)
- **Description**: Ratio cholestérol total / HDL
- **Signification médicale**: Indicateur de risque cardiovasculaire
- **Valeurs normales**: < 5 (souhaitable)
- **Importance**: Ratio élevé = risque cardiovasculaire élevé

### 9. s5 - LTG (Log of Triglycerides)
- **Type**: Numérique (normalisé, logarithme)
- **Description**: Logarithme du taux de triglycérides sériques
- **Signification médicale**: Graisses dans le sang, partie du profil lipidique
- **Valeurs normales**: < 150 mg/dL (normal)
- **Importance**: Triglycérides élevés = risque de diabète et maladies cardiaques

### 10. s6 - GLU (Blood Glucose)
- **Type**: Numérique (normalisé)
- **Description**: Taux de glucose sanguin (glycémie)
- **Signification médicale**: **Marqueur direct du diabète**
- **Valeurs normales**: 70-100 mg/dL (à jeun)
- **Importance**: Mesure centrale pour diagnostiquer et surveiller le diabète

### 11. target - Progression du Diabète
- **Type**: Numérique (non normalisé)
- **Description**: Mesure quantitative de la progression de la maladie après 1 an
- **Plage**: 25 à 346
- **Signification**: **Variable cible à prédire**
- **Interprétation**: 
  - Valeurs basses (< 100) = progression faible
  - Valeurs moyennes (100-200) = progression modérée
  - Valeurs élevées (> 200) = progression importante

---

## Nouvelles Colonnes Dérivées (6 colonnes)

Ces colonnes ont été ajoutées par notre script pour enrichir le dataset avec des informations médicalement significatives.

### 12. bmi_category - Catégorie d'IMC
- **Type**: Catégoriel
- **Valeurs**: 'Underweight', 'Normal', 'Overweight', 'Obese'
- **Calcul**: Basé sur la distribution normalisée du BMI
- **Utilité**: 
  - Classification médicale standard
  - Facilite l'interprétation
  - Utile pour l'analyse par groupe
- **Standards médicaux**:
  - Underweight: IMC < 18.5
  - Normal: IMC 18.5-24.9
  - Overweight: IMC 25-29.9
  - Obese: IMC ≥ 30

### 13. bp_category - Catégorie de Pression Artérielle
- **Type**: Catégoriel
- **Valeurs**: 'Normal', 'Elevated', 'High'
- **Calcul**: Basé sur la distribution normalisée de la pression
- **Utilité**: Identification rapide des patients hypertendus
- **Standards médicaux**:
  - Normal: < 120/80 mmHg
  - Elevated: 120-129/< 80 mmHg
  - High: ≥ 130/80 mmHg

### 14. cholesterol_ratio - Ratio LDL/HDL
- **Type**: Numérique
- **Formule**: s2 / s3
- **Signification médicale**: **Indicateur clé du risque cardiovasculaire**
- **Interprétation**:
  - Ratio < 2: Excellent
  - Ratio 2-3: Bon
  - Ratio 3-4: Acceptable
  - Ratio > 4: Risque élevé
- **Importance**: Plus précis que le cholestérol total seul

### 15. metabolic_risk - Score de Risque Métabolique
- **Type**: Numérique (0 à 1)
- **Formule**: Moyenne pondérée de:
  - 30% BMI
  - 20% Pression artérielle
  - 20% LDL
  - 20% Triglycérides
  - 10% Glucose
- **Utilité**: **Score composite du syndrome métabolique**
- **Interprétation**:
  - < 0.3: Risque faible
  - 0.3-0.5: Risque modéré
  - 0.5-0.7: Risque élevé
  - > 0.7: Risque très élevé
- **Signification**: Combine plusieurs facteurs de risque en un seul indicateur

### 16. age_group - Groupe d'Âge
- **Type**: Catégoriel
- **Valeurs**: 'Young', 'Middle', 'Senior'
- **Calcul**: Basé sur la distribution normalisée de l'âge
- **Utilité**: 
  - Stratification par âge
  - Analyse démographique
  - Identification des populations à risque
- **Approximation**:
  - Young: < 45 ans
  - Middle: 45-65 ans
  - Senior: > 65 ans

### 17. diabetes_risk - Niveau de Risque Diabète
- **Type**: Catégoriel
- **Valeurs**: 'Low', 'Medium', 'High', 'Very High'
- **Calcul**: Basé sur les quartiles de la variable target
- **Utilité**: **Classification en groupes de risque**
- **Distribution**:
  - Low: 25% des patients (target < Q1)
  - Medium: 25% des patients (Q1 ≤ target < Q2)
  - High: 25% des patients (Q2 ≤ target < Q3)
  - Very High: 25% des patients (target ≥ Q3)
- **Application**: Permet de cibler les interventions médicales

---

## Statistiques par Version

### Version 1: Original (442 échantillons, 17 features)
- Dataset de référence avec colonnes dérivées
- Distribution équilibrée des catégories
- Baseline pour comparaison

### Version 2: Augmenté (542 échantillons, 17 features)
- +100 échantillons synthétiques (variation moyenne 15%)
- Maintient les distributions des colonnes dérivées
- Augmente la diversité

### Version 3: Varié (642 échantillons, 17 features)
- +200 échantillons avec 3 niveaux de variation (5%, 15%, 30%)
- Plus grande diversité dans les colonnes dérivées
- Meilleure robustesse du modèle

---

## Utilisation des Colonnes pour la Modélisation

### Features Principales à Utiliser
1. **Numériques continues**: age, bmi, bp, s1-s6, cholesterol_ratio, metabolic_risk
2. **Catégorielles**: bmi_category, bp_category, age_group, diabetes_risk

### Features Engineering Recommandé
- Interactions: bmi × age, bp × bmi (déjà implémentées)
- Ratios: LDL/HDL (cholesterol_ratio déjà calculé)
- Scores composites: metabolic_risk (déjà calculé)

### Variables Cibles
- **Régression**: target (valeur continue 25-346)
- **Classification**: diabetes_risk (4 classes)

---

## Importance Médicale

### Facteurs de Risque Majeurs (corrélation forte avec target)
1. **bmi**: Facteur de risque #1 pour diabète type 2
2. **s6 (glucose)**: Marqueur direct du diabète
3. **metabolic_risk**: Score composite des facteurs métaboliques
4. **cholesterol_ratio**: Indicateur du profil lipidique

### Facteurs de Risque Modérés
1. **bp**: Hypertension souvent associée au diabète
2. **s5 (triglycérides)**: Partie du syndrome métabolique
3. **age**: Risque augmente avec l'âge

### Facteurs Informatifs
1. **sex**: Différences de prévalence selon le sexe
2. **s1, s2, s3, s4**: Profil lipidique complet

---

## Références Médicales

- **Syndrome Métabolique**: Combinaison de facteurs (obésité abdominale, hypertension, dyslipidémie, hyperglycémie)
- **Diabète Type 2**: Maladie métabolique caractérisée par l'hyperglycémie chronique
- **BMI Standards**: OMS - Organisation Mondiale de la Santé
- **Blood Pressure**: AHA - American Heart Association guidelines
- **Cholesterol**: NCEP ATP III - National Cholesterol Education Program

---

**Note**: Toutes les valeurs sont normalisées dans le dataset original. Les interprétations "élevé/faible" se basent sur les écarts par rapport à la moyenne de l'échantillon.

**Date de création**: Décembre 2025  
**Auteur**: Ahmed Ben Abderrazak  
**Version**: 1.0
