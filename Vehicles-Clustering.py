#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering

sv_data = pd.read_csv("vehicles.csv")

print(sv_data.head(), "\n")
print(sv_data.info())

# Remplacement NaN
sv_data.isna().sum()
sv_data = sv_data.fillna(sv_data.mean())
print(sv_data.isna().sum())


classe = sv_data["class"]
sv_data = sv_data.drop("class", axis=1)
print(sv_data.info())
print(classe.value_counts())
print(classe)


###### VISUALISATION ######

# Boxplot

liste = [sv_data["compactness"],sv_data["circularity"],sv_data["distance_circularity"], sv_data["radius_ratio"],
        sv_data["pr.axis_aspect_ratio"], sv_data["max.length_aspect_ratio"], sv_data["scatter_ratio"],
        sv_data["elongatedness"],sv_data["pr.axis_rectangularity"], sv_data["scaled_variance"],
        sv_data["scaled_variance.1"], sv_data["scaled_radius_of_gyration"], sv_data["scaled_radius_of_gyration.1"],
        sv_data["skewness_about"],sv_data["skewness_about.1"], sv_data["skewness_about.2"], sv_data["hollows_ratio"]]

lab_liste = ["compactness", "circularity", "distance circularity", "radius_ratio", "pr axis aspect ratio","max length aspect ratior",
            "scatter ratio", "elongatedness", "pr axis rectangularity", "scaled variance", "scaled var 1", "scaled radius gyration",
             "scaled radius gyration 1", "skewness about", "skewness about 1", "skewness about 2",
            "hollows ratio"]

print(len(liste), len(lab_liste))

plt.figure(figsize = (12,10))
plt.boxplot(liste, labels=lab_liste)
plt.xticks(rotation=90)
plt.show();

sv_data.describe()


###### NORMALISATION ######

# Normalisation MinMax
from sklearn.preprocessing import MinMaxScaler

scaler= MinMaxScaler().fit(sv_data)
sv_data_scaled = scaler.transform(sv_data)


# Kmeans 50 clusters
clf = KMeans(n_clusters = 50)
clf.fit(sv_data_scaled)

labels = clf.labels_
centroids = clf.cluster_centers_

print(centroids.shape)
print(labels.shape)


# Méthode du coude avec 50 clusters
from scipy.spatial.distance import cdist

liste_k = range(2,50)
distortion = []

for k in liste_k:
    cluster=KMeans(n_clusters=k)
    cluster.fit(sv_data_scaled)
    distortion.append(sum(np.min(cdist(sv_data_scaled,cluster.cluster_centers_,'euclidean'), axis=1))/ np.size(sv_data_scaled, axis=0))

plt.plot(liste_k, distortion);

# => difficile d'identifier un coude avec 50 clusters


# Méthode du coude avec 10 clusters
from scipy.spatial.distance import cdist

liste_k = range(2,10)
distortion = []

for k in liste_k:
    cluster=KMeans(n_clusters=k)
    cluster.fit(sv_data_scaled)
    distortion.append(sum(np.min(cdist(sv_data_scaled,cluster.cluster_centers_,'euclidean'), axis=1))/ np.size(sv_data_scaled, axis=0))

plt.plot(liste_k, distortion);

# => Observation d'un petit coude à 3 clusters


# Construction d'un dendrogramme
from scipy.cluster.hierarchy import linkage, dendrogram

Z = linkage(centroids, method="ward", metric="euclidean")

plt.figure(figsize = (12,10))

dendrogram(Z,leaf_rotation=7);
# => apparemment 2 clusters, voir 3
# Peut être que la distortion est une métrique plus adaptée que dendrogramme car il va avoir tendance
# à grouper les classes proches



# Classification ascendante hiérarchique avec 3 clusters
cah = AgglomerativeClustering(n_clusters=3)
cah.fit(centroids)

labels = cah.labels_
print(labels)


# Calcul des centres de gravité de chaque nouveau groupe
nv_centroids = pd.DataFrame(centroids)
nv_centroids["labels"] = labels

nv_centroids = nv_centroids.groupby("labels").mean()
nv_centroids


# Consolidation du KMeans
clf_2 = KMeans(n_clusters = 3, init=nv_centroids)
clf_2.fit(sv_data_scaled)

labels_final = clf_2.labels_
centroids_final= clf_2.cluster_centers_

print(labels_final)


# Comparaison aux classes de véhicules 

classe = classe.replace({"car":1,"bus":2,"van":0})
classe=pd.DataFrame(classe)
labels_final = pd.DataFrame(labels_final)



df = pd.concat([classe, labels_final], axis=1)
df = df.rename({0:"labels"}, axis=1)

print(df.head())


# matrice de confusion
print(pd.crosstab(df["class"], df["labels"]),"\n")
print("labels : ","\n",df["labels"].value_counts(),"\n")
print("classes : ","\n",df["class"].value_counts())


"""

van : Sur-représentation de la classe van (0) (548 obs aprés classif multiple, 199 obs dans sv_data)
    => souvent confondues avec les deux autres classes 

car : classe souvent confondue avec van (196/429) 
    => caractéristiques proches de van

bus : classe souvent confondue avec van (159/218) et parfois avec car 
    => caract proches de van et parfois similaires à car
    
    => Les vans sont souvent classifiés comme des car ou bus

Les vans sont une sorte d'hybride entre voiture et bus, il est donc possible que les deux autres classes partagent des 
caractéristiques avec celle-ci, pouvant affecter la classification de l'algorithme. Les informations du dataset 
par rapport à la silhouette du véhicule ne permettent pas de de bien discriminer les classes.

=> Il faudrait ajouter des variables explicatives permettant de mieux discriminer bus et car de van.
=> Essayer d'identifier et d'écarter des variables explicatives liant les classes entre elles 


L'utilisation de 3 clusters semble pertinent, comme observé avec la méthode du coude et légérement sur le dendrogramme.
En effet, le dendrogramme indiquait que deux classes étaient très proches, il doit s'agir de car et de van 
Une séparation en 2 classes aurait été de type : gros véhicule (bus) et petit véhicule (car, van).

"""


Z = linkage(nv_centroids, method="ward", metric="euclidean")

plt.figure(figsize = (5,5))

dendrogram(Z)
plt.legend();


# Score de silhouette
from sklearn.metrics import silhouette_score
k_clust = range(2,10)
s_scores=[]

for k in k_clust:
    clf = KMeans(n_clusters = k)
    clf.fit(sv_data_scaled)

    labels = clf.labels_
    centroids = clf.cluster_centers_
    
    # Classification ascendante hiérarchique avec 3 clusters
    cah = AgglomerativeClustering(n_clusters=k)
    cah.fit(centroids)
    labels = cah.labels_
    
    nv_centroids = pd.DataFrame(centroids)
    nv_centroids["labels"] = labels
    nv_centroids = nv_centroids.groupby("labels").mean()

    # Consolidation du KMeans
    clf_2 = KMeans(n_clusters = k, init=nv_centroids)
    clf_2.fit(sv_data_scaled)

    labels_final = clf_2.labels_
    
    s_score = silhouette_score(sv_data_scaled, labels_final, metric="sqeuclidean")
    s_scores.append(s_score)

plt.plot(k_clust, s_scores)


# Score de silhouette le plus élevé pour n_clusters = 2
# => meilleur nombre pour l'homogénéité intra-cluster et la séparation inter-cluster

# MAIS ne permet pas une bonne séparation inter-cluster, comme vu au dessus (van et bus très proche)
# Choisir 3 clusters semble donc pertinent quitte à avoir moins d'homogénéité intra-cluster

clf = KMeans(n_clusters = 3)
clf.fit(sv_data_scaled)

labels = clf.labels_
centroids = clf.cluster_centers_


# Construction d'un dendrogramme
from scipy.cluster.hierarchy import linkage, dendrogram

Z = linkage(centroids, method="ward", metric="euclidean")

plt.figure(figsize = (12,10))

dendrogram(Z,leaf_rotation=7);
# => apparemment 2 clusters, voir 3
# Peut être que la distortion est une métrique plus adaptée que dendrogramme car il va avoir tendance
# à grouper les classes proches


