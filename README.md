# ClusterLab

4 modèles de segmentation dans une seule app Streamlit :
clients télécom (CAH), insuffisance cardiaque (KMeans), clients grossiste (KMeans), graines de blé (KMeans).

## Lancer en local
```bash
pip install -r requirements.txt
# copier les 4 CSV dans data/ (voir data/LISEZMOI.txt)
streamlit run app.py
```

## Déployer (Streamlit Community Cloud)
1. Poussez ce dossier sur GitHub (avec les CSV dans `data/`).
2. Sur share.streamlit.io : New app, choisissez le dépôt, fichier principal `app.py`.

Si un CSV manque, la page du modèle propose de le déposer dans l'interface.
