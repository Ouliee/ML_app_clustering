"""ClusterLab : 4 modèles de segmentation réunis dans une seule application Streamlit.

Lancer :  streamlit run app.py
Les modèles sont entraînés au démarrage (et mis en cache) à partir des CSV du dossier data/.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, normalize

BASE = Path(__file__).parent
DATA_DIRS = [BASE / "data", BASE]
PALETTE = ["#6B4A35", "#C29A6B", "#A8604A", "#8E9A7E", "#B9A58C", "#4A3426"]
INK = "#3B2A20"
DARK = "#2B1D15"
GRID = "#EADFCB"
YN = {0: "Non", 1: "Oui"}


def F(col, label, kind, **kw):
    return dict(col=col, label=label, kind=kind, **kw)


# ───────────────────────────── Configuration des 4 modèles ─────────────────────────────
MODELS: dict[str, dict] = {
    "customer": dict(
        page="Clients télécom", icon="📱", grad=("#5C3D2E", "#9A6B47"),
        title="Segmentation des clients télécom",
        tagline="Un client est-il fidèle ou non ? Réponse à partir de son comportement d'usage.",
        file="Customer_dataset.csv", algo="CAH · liaison moyenne", k=2, ncols=3,
        columns=["Call Failure", "Complains", "Subscription Length", "Charge Amount", "Seconds of Use",
                 "Frequency of use", "Frequency of SMS", "Distinct Called Numbers", "Age Group",
                 "Status", "Age", "Customer Value"],
        fields=[
            F("Call Failure", "Appels échoués", "int"),
            F("Complains", "Plainte déposée", "bin", opts=YN),
            F("Subscription Length", "Durée d'abonnement (mois)", "sint"),
            F("Charge Amount", "Niveau de facturation (0-10)", "sint"),
            F("Seconds of Use", "Secondes d'utilisation", "int", step=100),
            F("Frequency of use", "Fréquence d'appels", "int"),
            F("Frequency of SMS", "Fréquence de SMS", "int"),
            F("Distinct Called Numbers", "Numéros distincts appelés", "int"),
            F("Age Group", "Tranche d'âge (code)", "cat", opts={i: f"Catégorie {i}" for i in range(1, 6)}),
            F("Status", "Statut", "cat", opts={1: "Actif", 2: "Inactif"}),
            F("Age", "Âge", "sint"),
            F("Customer Value", "Valeur client", "float", step=10.0, fmt="%.2f"),
        ],
        pipeline=[
            "**Normalisation L2** de chaque client (`normalize`).",
            "**Classification ascendante hiérarchique** : distance euclidienne, liaison moyenne, 2 classes.",
            "Un nouveau client est affecté au **centroïde le plus proche**, comme dans le déploiement Gradio du notebook.",
            "**ACP à 2 composantes**, utilisée uniquement pour la carte de visualisation.",
        ],
        notes=["Les classes sont nommées d'après leur profil : le groupe qui utilise le plus le service est « Fidèle ».",
               "La CAH obtient une meilleure silhouette que le KMeans (k=3), d'où son choix pour le déploiement."],
    ),
    "heart": dict(
        page="Insuffisance cardiaque", icon="🫀", grad=("#5E2B22", "#A65A42"),
        title="Profils de patients en insuffisance cardiaque",
        tagline="Rapprochez un patient des 4 profils cliniques identifiés par KMeans.",
        file="heart_failure_clinical_records_dataset.csv", algo="KMeans", k=4, ncols=3,
        columns=["age", "anaemia", "creatinine_phosphokinase", "diabetes", "ejection_fraction",
                 "high_blood_pressure", "platelets", "serum_creatinine", "serum_sodium", "sex",
                 "smoking", "time", "DEATH_EVENT"],
        fields=[
            F("age", "Âge", "sint"),
            F("anaemia", "Anémie", "bin", opts=YN),
            F("creatinine_phosphokinase", "CPK (mcg/L)", "int"),
            F("diabetes", "Diabète", "bin", opts=YN),
            F("ejection_fraction", "Fraction d'éjection (%)", "sint"),
            F("high_blood_pressure", "Hypertension", "bin", opts=YN),
            F("platelets", "Plaquettes (kiloplaquettes/mL)", "float", step=1000.0, fmt="%.0f"),
            F("serum_creatinine", "Créatinine sérique (mg/dL)", "float", step=0.1, fmt="%.2f"),
            F("serum_sodium", "Sodium sérique (mEq/L)", "sint"),
            F("sex", "Sexe", "bin", opts={0: "Femme", 1: "Homme"}),
            F("smoking", "Fumeur", "bin", opts=YN),
        ],
        features=None,
        pipeline=[
            "Variables d'entrée : 11 caractéristiques cliniques (**sans** `time` ni `DEATH_EVENT`).",
            "**log(1+x)** sur la CPK, très asymétrique.",
            "**Standardisation** (`StandardScaler`).",
            "**KMeans** à 4 classes (`init='random'`, 500 itérations), k choisi par la méthode du coude.",
        ],
        notes=["La silhouette est faible : les groupes se chevauchent. Lisez-les comme des tendances, pas comme des frontières nettes."],
        notice="Outil pédagogique de segmentation non supervisée. Il ne remplace pas un avis médical.",
    ),
    "wholesale": dict(
        page="Clients grossiste", icon="🛒", grad=("#4A3828", "#8F7452"),
        title="Segmentation des clients d'un grossiste",
        tagline="Regroupez les clients selon leurs dépenses annuelles par catégorie de produits.",
        file="Wholesale customers data.csv", algo="KMeans", k=4, ncols=3,
        columns=["Channel", "Region", "Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"],
        fields=[
            F("Fresh", "Produits frais", "int", step=500),
            F("Milk", "Lait & produits laitiers", "int", step=500),
            F("Grocery", "Épicerie", "int", step=500),
            F("Frozen", "Surgelés", "int", step=500),
            F("Detergents_Paper", "Détergents & papier", "int", step=500),
            F("Delicassen", "Traiteur", "int", step=500),
        ],
        pipeline=[
            "Variables d'entrée : les 6 dépenses annuelles (`Channel` et `Region` servent seulement à décrire les groupes).",
            "**log(1+x)** pour réduire l'asymétrie des dépenses.",
            "**Standardisation** (`StandardScaler`).",
            "**KMeans** à 4 classes (`init='random'`, 500 itérations).",
        ],
        notes=["Les dépenses sont en unités monétaires par an. Les groupes se chevauchent (silhouette modeste)."],
    ),
    "wheat": dict(
        page="Graines de blé", icon="🌾", grad=("#6E5428", "#B8913F"),
        title="Classification des graines de blé",
        tagline="Réglez les 7 mesures morphologiques d'une graine et découvrez son groupe.",
        file="wheat_seeds_dataset.csv", algo="KMeans", k=4, ncols=2,
        columns=["area A", "perimeter", "compactness", "length of kernel", "width of kernel",
                 "asymmetry coefficient", "length of kernel groove"],
        fields=[
            F("area A", "Aire", "sfloat", step=0.01),
            F("perimeter", "Périmètre", "sfloat", step=0.01),
            F("compactness", "Compacité", "sfloat", step=0.001, fmt="%.4f"),
            F("length of kernel", "Longueur du grain", "sfloat", step=0.001, fmt="%.3f"),
            F("width of kernel", "Largeur du grain", "sfloat", step=0.001, fmt="%.3f"),
            F("asymmetry coefficient", "Coefficient d'asymétrie", "sfloat", step=0.01),
            F("length of kernel groove", "Longueur du sillon", "sfloat", step=0.001, fmt="%.3f"),
        ],
        pipeline=[
            "**Standardisation** (`StandardScaler`) des 7 variables morphologiques, pour qu'aucune ne domine les distances.",
            "**ACP à 2 composantes** pour la visualisation.",
            "**KMeans** à 4 classes (`init='random'`, 500 itérations), k choisi par la méthode du coude.",
        ],
        notes=["Contrairement au notebook d'origine, les variables sont standardisées : l'aire et le périmètre ne dominent plus le KMeans.",
               "Après standardisation, le k optimal peut changer : vérifiez la silhouette pour k=3 et k=4 avant de figer le choix."],
    ),
}
for _k, _c in MODELS.items():
    _c["features"] = [f["col"] for f in _c["fields"]]
    if _k == "customer":
        assert _c["features"] == _c["columns"]


# ───────────────────────────── Chargement des données ─────────────────────────────
def _slug(s: str) -> str:
    return re.sub(r"[\s_]+", "_", s.lower())


def find_file(name: str):
    for d in DATA_DIRS:
        if d.exists():
            for q in d.glob("*.csv"):
                if _slug(q.name) == _slug(name):
                    return q
    return None


def clean(raw: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = raw.rename(columns=lambda c: re.sub(r"\s+", " ", str(c)).strip())
    if not all(c in df.columns for c in columns):
        if df.shape[1] < len(columns):
            raise ValueError(f"{len(columns)} colonnes attendues, {df.shape[1]} trouvées.")
        df = df.iloc[:, : len(columns)].copy()
        df.columns = columns
    return df[columns].apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)


def get_df(key: str, interactive: bool = True):
    cfg = MODELS[key]
    try:
        p = find_file(cfg["file"])
        up = st.session_state.get(f"up:{key}")
        if p is not None:
            return clean(pd.read_csv(p, sep=None, engine="python"), cfg["columns"])
        if up is not None:
            return clean(pd.read_csv(io.BytesIO(up), sep=None, engine="python"), cfg["columns"])
    except Exception as e:  # noqa: BLE001
        if not interactive:
            return None
        st.error(f"Impossible de lire le fichier : {e}")
    if not interactive:
        return None
    st.markdown(hero_html(cfg), unsafe_allow_html=True)
    st.warning(f"Le fichier **{cfg['file']}** est introuvable dans `data/`. "
               "Déposez-le ci-dessous pour tester l'application (pour un déploiement, placez-le dans `data/`).")
    f = st.file_uploader(f"Charger {cfg['file']}", type="csv", key=f"uploader:{key}")
    if f is not None:
        st.session_state[f"up:{key}"] = f.getvalue()
        st.rerun()
    st.stop()


# ───────────────────────────── Entraînement ─────────────────────────────
@dataclass
class Seg:
    X: pd.DataFrame
    labels: np.ndarray
    names: list
    descs: list
    centers: np.ndarray
    to_model: Callable
    pca: PCA
    coords: np.ndarray
    sil: float
    var2d: float
    mu: np.ndarray
    sd: np.ndarray
    z: pd.DataFrame
    means: pd.DataFrame
    sizes: pd.Series
    short: list
    extra: dict = field(default_factory=dict)


def describe(zrow: pd.Series, short: list, thr: float = 0.5) -> str:
    idx = np.argsort(-np.abs(zrow.values))[:3]
    parts = [f"{'↑' if zrow.iloc[i] > 0 else '↓'} {short[i]}" for i in idx if abs(zrow.iloc[i]) >= thr]
    return " · ".join(parts) if parts else "Profil proche de la moyenne"


@st.cache_resource(show_spinner="Entraînement du modèle…")
def build(key: str, df: pd.DataFrame) -> Seg:
    cfg = MODELS[key]
    feats, k = cfg["features"], cfg["k"]
    A = df[feats].astype(float).values

    if key == "customer":
        to_model = lambda a: normalize(np.asarray(a, float))  # noqa: E731
    elif key == "heart":
        j = feats.index("creatinine_phosphokinase")

        def pre(a):
            a = np.array(a, float, copy=True)
            a[:, j] = np.log1p(a[:, j])
            return a

        sc = StandardScaler().fit(pre(A))
        to_model = lambda a: sc.transform(pre(a))  # noqa: E731
    elif key == "wholesale":
        sc = StandardScaler().fit(np.log1p(A))
        to_model = lambda a: sc.transform(np.log1p(np.asarray(a, float)))  # noqa: E731
    else:  # wheat
        sc = StandardScaler().fit(A)
        to_model = lambda a: sc.transform(np.asarray(a, float))  # noqa: E731
    M = to_model(A)

    if key == "customer":
        raw = AgglomerativeClustering(n_clusters=k, metric="euclidean", linkage="average").fit_predict(M)
        j = feats.index("Seconds of Use")
        order = list(np.argsort([A[raw == c, j].mean() for c in range(k)])[::-1])
        names = ["Fidèle", "Non fidèle"]
    else:
        raw = KMeans(n_clusters=k, init="random", max_iter=500, n_init=10, random_state=42).fit_predict(M)
        order = list(pd.Series(raw).value_counts().index)
        names = [f"Groupe {i + 1}" for i in range(k)]
    remap = {old: new for new, old in enumerate(order)}
    labels = np.array([remap[r] for r in raw])

    pca = PCA(n_components=2).fit(M)
    mu, sd = A.mean(0), A.std(0)
    sd[sd == 0] = 1.0
    means = pd.DataFrame(A, columns=feats).groupby(labels).mean()
    z = (means - mu) / sd
    short = [re.sub(r"\s*\(.*?\)", "", f["label"]) for f in cfg["fields"]]
    seg = Seg(
        X=pd.DataFrame(A, columns=feats), labels=labels, names=names,
        descs=[describe(z.iloc[c], short) for c in range(k)],
        centers=np.vstack([M[labels == c].mean(0) for c in range(k)]),
        to_model=to_model, pca=pca, coords=pca.transform(M),
        sil=float(silhouette_score(M, labels)), var2d=float(pca.explained_variance_ratio_.sum() * 100),
        mu=mu, sd=sd, z=z, means=means,
        sizes=pd.Series(np.bincount(labels, minlength=k), index=names), short=short,
    )
    if key == "customer":
        km = KMeans(n_clusters=3, init="random", max_iter=500, n_init=10, random_state=42).fit_predict(M)
        seg.extra["compare"] = {"KMeans (k=3)": float(silhouette_score(M, km)), "CAH (k=2)": seg.sil}
    if key == "wholesale":
        ch = df["Channel"].map({1: "Horeca", 2: "Retail"})
        seg.extra["channel"] = pd.crosstab(labels, ch, normalize="index") * 100
    if key == "heart":
        seg.extra["death"] = df["DEATH_EVENT"].groupby(labels).mean() * 100
    return seg


def predict(seg: Seg, values):
    m = seg.to_model(np.array(values, float).reshape(1, -1))
    d = cdist(m, seg.centers)[0]
    o = d.argsort()
    margin = (d[o[1]] - d[o[0]]) / (d[o[1]] + d[o[0]] + 1e-12)
    return int(o[0]), d, float(margin), seg.pca.transform(m)[0]


# ───────────────────────────── Style ─────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Playfair+Display:wght@500;600&display=swap');
.stApp {font-family:'Inter',system-ui,sans-serif; color:#3B2A20; background:#FAF6EF;}
.block-container {padding-top:2.2rem; max-width:1120px;}
#MainMenu, footer {visibility:hidden;}
h1,h2,h3,h4,h5 {font-family:'Playfair Display',Georgia,serif; font-weight:600; color:#3B2A20; letter-spacing:0;}
section[data-testid="stSidebar"] {background:#F3EADB; border-right:1px solid #E6D9C3;}
.stTabs [data-baseweb="tab-list"] {gap:1.4rem; border-bottom:1px solid #E6D9C3;}
.stTabs [data-baseweb="tab"] {font-weight:500; font-size:.95rem; color:#8A7360; padding-left:0; padding-right:0;}
.stTabs [aria-selected="true"] {color:#3B2A20; font-weight:600;}
.stTabs [data-baseweb="tab-highlight"] {background-color:#6B4A35;}
.stButton > button, .stDownloadButton > button {border-radius:10px; font-weight:500; border:1px solid #DCCBAE;
  background:#FFFDF9; color:#5A3E2B;}
.stButton > button:hover, .stDownloadButton > button:hover {border-color:#6B4A35; color:#3B2A20; background:#F6EEDF;}
.stFormSubmitButton > button {border-radius:10px; font-weight:600; background:#6B4A35; color:#FFF8EE;
  border:1px solid #6B4A35; padding:.5rem 1.8rem;}
.stFormSubmitButton > button:hover {background:#553826; border-color:#553826; color:#fff;}
div[data-testid="stVerticalBlockBorderWrapper"] {border-radius:16px; border-color:#E6D9C3;}
.hero {background:#F1E6D3; border:1px solid #E3D3B6; border-radius:18px; padding:1.9rem 2.2rem 1.6rem; margin-bottom:1.2rem;}
.hero .e {font-size:.72rem; text-transform:uppercase; letter-spacing:.16em; color:#8A6A4F; font-weight:600; margin-bottom:.5rem;}
.hero .t {font-family:'Playfair Display',Georgia,serif; font-size:2.2rem; font-weight:600; line-height:1.18; color:#3B2A20;}
.hero .s {margin-top:.6rem; font-size:1rem; color:#6B5644; max-width:720px; line-height:1.6;}
.chip {display:inline-block; padding:.2rem .7rem; border-radius:999px; font-size:.74rem; font-weight:500;
  margin:1rem .4rem 0 0; background:#FFFDF9; border:1px solid #E3D3B6; color:#6B4A35;}
.chip.dark {margin:.8rem .35rem 0 0;}
.chip.warn {background:#F8EAD0; border-color:#E9CE98; color:#8A5A12; margin:.8rem .35rem 0 0;}
.kpi {background:#FFFDF9; border:1px solid #E6D9C3; border-radius:14px; padding:.9rem 1.1rem;}
.kpi-l {font-size:.68rem; text-transform:uppercase; letter-spacing:.12em; color:#9A8470; font-weight:600;}
.kpi-v {font-family:'Playfair Display',Georgia,serif; font-size:1.65rem; font-weight:600; color:#3B2A20; line-height:1.35;}
.kpi-s {font-size:.78rem; color:#8A7360;}
.result {background:#FFFDF9; border:1px solid #E6D9C3; border-left:6px solid var(--c); border-radius:14px;
  padding:1.3rem 1.8rem; margin:1.2rem 0 1rem;}
.result .r-top {font-size:.7rem; text-transform:uppercase; letter-spacing:.16em; color:#9A8470; font-weight:600;}
.result .r-name {font-family:'Playfair Display',Georgia,serif; font-size:2.1rem; font-weight:600; line-height:1.25; color:#3B2A20;}
.result .r-desc {color:#6B5644; font-size:.98rem; margin-top:.15rem;}
.ico {width:46px; height:46px; border-radius:12px; display:flex; align-items:center; justify-content:center;
  font-size:1.4rem; background:#F1E6D3; border:1px solid #E3D3B6; margin-bottom:.8rem;}
.card-t {font-family:'Playfair Display',Georgia,serif; font-size:1.25rem; font-weight:600; color:#3B2A20;}
.card-p {color:#7A6652; font-size:.92rem; margin-top:.25rem; line-height:1.5;}
.step {display:flex; gap:.8rem; align-items:flex-start; margin:.5rem 0;}
.step b.n {flex:0 0 28px; height:28px; border-radius:50%; background:#F1E6D3; border:1px solid #E3D3B6; color:#6B4A35;
  display:flex; align-items:center; justify-content:center; font-size:.8rem;}
/* ── Robustesse : même rendu crème/marron en mode sombre ou si config.toml est absent ── */
:root {color-scheme: light;}
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {background:#FAF6EF !important; color:#3B2A20;}
[data-testid="stHeader"], [data-testid="stToolbar"] {background:#FAF6EF !important;}
[data-testid="stHeader"] *, [data-testid="stToolbar"] * {color:#6B4A35 !important;}
[data-testid="stSidebar"], [data-testid="stSidebarContent"], [data-testid="stSidebarHeader"] {background:#F3EADB !important;}
[data-testid="stSidebar"] a, [data-testid="stSidebar"] span, [data-testid="stSidebar"] p,
[data-testid="stSidebar"] small, [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {color:#5A3E2B !important;}
[data-testid="stSidebarNav"] a[aria-current="page"], [data-testid="stSidebarNavLink"][aria-current="page"] {background:#E9DCC4 !important;}
[data-testid="stSidebarNav"] a:hover {background:#EBDFC9 !important;}
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5 {color:#3B2A20 !important;}
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] strong, [data-testid="stMarkdownContainer"] em {color:#3B2A20;}
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] * {color:#5A3E2B !important;}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * {color:#8A7360 !important;}
[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="select"] > div {
  background:#FFFDF9 !important; border-color:#DCCBAE !important; border-radius:10px !important;}
[data-baseweb="input"] input, [data-baseweb="select"] div, [data-baseweb="select"] span {
  color:#3B2A20 !important; -webkit-text-fill-color:#3B2A20 !important;}
[data-baseweb="select"] svg, [data-testid="stNumberInput"] svg {fill:#6B4A35 !important; color:#6B4A35 !important;}
[data-testid="stNumberInput"] button {background:#F3EADB !important; color:#5A3E2B !important; border-color:#DCCBAE !important;}
[data-baseweb="popover"] [data-baseweb="menu"], [data-baseweb="popover"] ul {background:#FFFDF9 !important;}
[data-baseweb="popover"] li, [data-baseweb="popover"] li * {color:#3B2A20 !important;}
[data-baseweb="popover"] li:hover, [data-baseweb="popover"] li[aria-selected="true"] {background:#F1E6D3 !important;}
[data-testid="stAlert"] {background:#F6EBD5 !important; border:1px solid #E6D2A8 !important; border-radius:12px !important;}
[data-testid="stAlert"] * {color:#6B4A35 !important;}
.stTabs [data-baseweb="tab"] p {color:#8A7360 !important;}
.stTabs [aria-selected="true"] p {color:#3B2A20 !important; font-weight:600;}
.stTabs [data-baseweb="tab-border"] {background-color:#E6D9C3 !important;}
.stTabs [data-baseweb="tab-highlight"] {background-color:#6B4A35 !important;}
[data-testid="stSlider"] [role="slider"] {background-color:#6B4A35 !important; box-shadow:none !important;}
[data-testid="stSliderThumbValue"], [data-testid="stSliderThumbValue"] * {color:#6B4A35 !important;}
[data-testid="stTickBarMin"], [data-testid="stTickBarMax"] {color:#9A8470 !important;}
[data-testid="stSlider"] [data-baseweb="slider"] > div > div:first-child {filter:hue-rotate(-330deg) saturate(.45) brightness(.85);}
[data-testid="stDataFrame"] {border:1px solid #E6D9C3; border-radius:12px; overflow:hidden;}
"""


def hero_html(cfg, chips=()):
    ch = "".join(f'<span class="chip">{c}</span>' for c in chips)
    return (f'<div class="hero"><div class="e">{cfg["icon"]} &nbsp;{cfg["page"]}</div><div class="t">{cfg["title"]}</div>'
            f'<div class="s">{cfg["tagline"]}</div>{ch}</div>')


def kpi(col, label, value, sub=""):
    col.markdown(f'<div class="kpi"><div class="kpi-l">{label}</div><div class="kpi-v">{value}</div>'
                 f'<div class="kpi-s">{sub}</div></div>', unsafe_allow_html=True)


def sil_quality(s: float) -> str:
    return ("Séparation forte" if s >= 0.7 else "Bonne séparation" if s >= 0.5
            else "Structure modérée" if s >= 0.25 else "Structure faible")


def style(fig, h=380):
    fig.update_layout(template="plotly_white", height=h, margin=dict(l=10, r=10, t=45, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter, sans-serif", size=12, color=INK),
                      title_font=dict(family="Playfair Display, Georgia, serif", size=17, color=INK),
                      legend=dict(orientation="h", y=-0.18, x=0))
    fig.update_xaxes(gridcolor=GRID, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, linecolor=GRID)
    return fig


def map_fig(seg: Seg, point=None, title="Carte des groupes (ACP)"):
    fig = go.Figure()
    for c, n in enumerate(seg.names):
        m = seg.labels == c
        fig.add_trace(go.Scatter(x=seg.coords[m, 0], y=seg.coords[m, 1], mode="markers", name=n,
                                 marker=dict(size=7, color=PALETTE[c], opacity=0.6)))
    if point is not None:
        fig.add_trace(go.Scatter(x=[point[0]], y=[point[1]], mode="markers", name="Votre saisie",
                                 marker=dict(size=22, symbol="star", color=DARK,
                                             line=dict(color="#FAF5EC", width=2))))
    fig.update_xaxes(title="PC1", zeroline=False)
    fig.update_yaxes(title="PC2", zeroline=False)
    fig.update_layout(title=title)
    return style(fig)


# ───────────────────────────── Saisie ─────────────────────────────
def cast(f, v):
    k = f["kind"]
    if k in ("bin", "cat"):
        opts = list(f["opts"])
        return min(opts, key=lambda o: abs(o - float(v)))
    if k in ("int", "sint"):
        return int(round(float(v)))
    return float(v)


def load_example(key, cfg, X):
    row = X.sample(1).iloc[0]
    for f in cfg["fields"]:
        st.session_state[f"{key}:{f['col']}"] = cast(f, row[f["col"]])


def reset_defaults(key, cfg, X):
    for f in cfg["fields"]:
        st.session_state[f"{key}:{f['col']}"] = cast(f, X[f["col"]].median())


def make_widget(key, f, X, holder):
    sk, lo, hi = f"{key}:{f['col']}", float(X[f["col"]].min()), float(X[f["col"]].max())
    hlp = f"Plage observée : {lo:g} à {hi:g}"
    with holder:
        k = f["kind"]
        if k in ("bin", "cat"):
            return st.selectbox(f["label"], list(f["opts"]), format_func=f["opts"].get, key=sk)
        if k == "int":
            return st.number_input(f["label"], min_value=0, step=int(f.get("step", 1)), key=sk, help=hlp)
        if k == "float":
            return st.number_input(f["label"], min_value=0.0, step=float(f["step"]), format=f["fmt"], key=sk, help=hlp)
        if k == "sint":
            return st.slider(f["label"], int(lo), int(hi), key=sk)
        span = hi - lo
        return st.slider(f["label"], round(max(0.0, lo - 0.05 * span), 4), round(hi + 0.05 * span, 4),
                         step=float(f["step"]), format=f.get("fmt", "%.2f"), key=sk)


# ───────────────────────────── Onglets d'un modèle ─────────────────────────────
def predict_tab(key, cfg, seg: Seg):
    X, fields = seg.X, cfg["fields"]
    for f in fields:
        sk = f"{key}:{f['col']}"
        if sk not in st.session_state:
            st.session_state[sk] = cast(f, X[f["col"]].median())
    if cfg.get("notice"):
        st.info(cfg["notice"], icon="⚕️")

    b1, b2, _ = st.columns([1.1, 1.1, 3])
    b1.button("🎲 Exemple du jeu de données", on_click=load_example, args=(key, cfg, X))
    b2.button("↺ Valeurs médianes", on_click=reset_defaults, args=(key, cfg, X))

    with st.container(border=True):
        st.markdown("##### Caractéristiques à analyser")
        with st.form(f"form_{key}", border=False):
            cols = st.columns(cfg["ncols"], gap="large")
            vals = [make_widget(key, f, X, cols[i % cfg["ncols"]]) for i, f in enumerate(fields)]
            st.form_submit_button("Analyser →", type="primary")

    c, d, margin, xy = predict(seg, vals)
    outside = [f["label"] for f, v in zip(fields, vals)
               if v < X[f["col"]].min() - 1e-9 or v > X[f["col"]].max() + 1e-9]
    if outside:
        st.warning("Hors de la plage d'entraînement : " + ", ".join(outside) + ". La prédiction est une extrapolation.")

    st.markdown(f'<div class="result" style="--c:{PALETTE[c]}"><div class="r-top">Groupe prédit</div>'
                f'<div class="r-name">{seg.names[c]}</div><div class="r-desc">{seg.descs[c]}</div></div>',
                unsafe_allow_html=True)
    k1, k2, k3 = st.columns(3)
    kpi(k1, "Netteté de l'affectation", f"{margin * 100:.0f} %", "écart entre les 2 groupes les plus proches")
    kpi(k2, "Taille du groupe", f"{int(seg.sizes.iloc[c])}", f"{seg.sizes.iloc[c] / len(X) * 100:.1f} % des observations")
    kpi(k3, "Distance au centre", f"{d[c]:.3g}", "espace du modèle")
    st.write("")

    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(map_fig(seg, xy, "Où se situe votre saisie ?"))
    with g2:
        z_in = (np.array(vals, float) - seg.mu) / seg.sd
        fig = go.Figure()
        fig.add_bar(y=seg.short, x=seg.z.iloc[c].values, orientation="h", name=f"Moyenne · {seg.names[c]}",
                    marker_color=PALETTE[c], opacity=0.5)
        fig.add_bar(y=seg.short, x=z_in, orientation="h", name="Votre saisie", marker_color=DARK)
        fig.update_layout(barmode="group", title="Profil comparé au groupe")
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(title="Écart à la moyenne (en écarts-types)")
        st.plotly_chart(style(fig, max(380, 34 * len(seg.short) + 130)))


def explore_tab(key, cfg, df, seg: Seg):
    colors = [PALETTE[i] for i in range(len(seg.names))]
    c1, c2 = st.columns([1, 1.5])
    with c1:
        fig = go.Figure(go.Bar(x=seg.names, y=seg.sizes.values, marker_color=colors,
                               text=seg.sizes.values, textposition="outside"))
        fig.update_layout(title="Effectif par groupe", showlegend=False)
        st.plotly_chart(style(fig))
    with c2:
        st.plotly_chart(map_fig(seg))

    lim = float(min(3, max(1, np.abs(seg.z.values).max())))
    fig = px.imshow(seg.z.T.values, x=seg.names, y=seg.short, color_continuous_scale=[[0, "#4F6D7A"], [0.5, "#FAF5EC"], [1, "#8B4A2B"]],
                    zmin=-lim, zmax=lim, aspect="auto", text_auto=".2f")
    fig.update_layout(title="Profil de chaque groupe (écart à la moyenne générale, en écarts-types)",
                      coloraxis_colorbar=dict(title="σ"))
    st.plotly_chart(style(fig, max(340, 34 * len(seg.short) + 120)))

    st.markdown("##### Valeurs moyennes par groupe")
    tbl = seg.means.copy()
    tbl.index = seg.names
    tbl.columns = seg.short
    st.dataframe(tbl.round(2).T)

    ex = seg.extra
    if "compare" in ex:
        fig = go.Figure(go.Bar(x=list(ex["compare"]), y=[round(v, 3) for v in ex["compare"].values()],
                               marker_color=[PALETTE[4], PALETTE[0]], text=[f"{v:.3f}" for v in ex["compare"].values()],
                               textposition="outside"))
        fig.update_layout(title="Indice de silhouette : KMeans vs CAH", showlegend=False)
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(style(fig, 320))
    if "channel" in ex:
        ch = ex["channel"]
        fig = go.Figure([go.Bar(name=col, x=seg.names, y=ch[col].values) for col in ch.columns])
        fig.update_layout(barmode="stack", title="Répartition par canal de vente (%)")
        st.plotly_chart(style(fig, 320))
    if "death" in ex:
        fig = go.Figure(go.Bar(x=seg.names, y=ex["death"].round(1).values, marker_color=colors,
                               text=[f"{v:.1f} %" for v in ex["death"].values], textposition="outside"))
        fig.update_layout(title="Décès observés pendant le suivi, par groupe (%)", showlegend=False)
        st.plotly_chart(style(fig, 320))
        st.caption("Description des données uniquement, ce n'est pas une analyse causale ni un pronostic.")

    out = df.copy()
    out["Groupe"] = [seg.names[i] for i in seg.labels]
    st.download_button("⬇️ Télécharger les données avec leur groupe", out.to_csv(index=False).encode("utf-8"),
                       file_name=f"{key}_groupes.csv", mime="text/csv")


def about_tab(cfg, df, seg: Seg):
    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.markdown("##### Pipeline du modèle")
        for i, s in enumerate(cfg["pipeline"], 1):
            st.markdown(f'<div class="step"><b class="n">{i}</b><div>{s}</div></div>', unsafe_allow_html=True)
        st.markdown("##### Bon à savoir")
        for n in cfg["notes"]:
            st.markdown(f"- {n}")
    with c2:
        st.markdown("##### Fiche technique")
        st.dataframe(pd.DataFrame({
            "Algorithme": cfg["algo"], "Groupes (k)": cfg["k"], "Observations": len(df),
            "Variables du modèle": len(cfg["features"]), "Silhouette": f"{seg.sil:.3f}",
            "Variance ACP (2 axes)": f"{seg.var2d:.1f} %",
        }, index=["Valeur"]).T.astype(str))
        st.caption("Les modèles sont entraînés au démarrage (graine fixée) : les groupes sont reproductibles.")


def model_page(key: str):
    cfg = MODELS[key]
    df = get_df(key)
    seg = build(key, df)
    st.markdown(hero_html(cfg, [cfg["algo"], f"k = {cfg['k']}", f"{len(df)} observations"]), unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    kpi(k1, "Observations", f"{len(df):,}".replace(",", " "), "jeu d'entraînement")
    kpi(k2, "Variables", len(cfg["features"]), "utilisées par le modèle")
    kpi(k3, "Silhouette", f"{seg.sil:.2f}", sil_quality(seg.sil))
    kpi(k4, "Variance ACP", f"{seg.var2d:.0f} %", "portée par la carte 2D")
    st.write("")
    t1, t2, t3 = st.tabs(["🎯 Prédiction", "📊 Exploration des groupes", "ℹ️ Le modèle"])
    with t1:
        predict_tab(key, cfg, seg)
    with t2:
        explore_tab(key, cfg, df, seg)
    with t3:
        about_tab(cfg, df, seg)


# ───────────────────────────── Pages ─────────────────────────────
PAGES: dict = {}


def page_home():
    st.markdown('<div class="hero"><div class="e">Segmentation · Machine learning</div><div class="t">ClusterLab</div>'
                '<div class="s">Quatre modèles de segmentation dans une seule application : choisissez un jeu de données, '
                'saisissez un profil et découvrez à quel groupe il appartient.</div>'
                '<span class="chip">KMeans</span><span class="chip">CAH</span><span class="chip">ACP</span>'
                '<span class="chip">Silhouette</span></div>', unsafe_allow_html=True)
    cols = st.columns(2, gap="large")
    for i, key in enumerate(MODELS):
        cfg = MODELS[key]
        df = get_df(key, interactive=False)
        chips = f'<span class="chip dark">{cfg["algo"]}</span><span class="chip dark">k = {cfg["k"]}</span>'
        if df is not None:
            seg = build(key, df)
            chips += (f'<span class="chip dark">Silhouette {seg.sil:.2f}</span>'
                      f'<span class="chip dark">{len(df)} obs.</span>')
        else:
            chips += '<span class="chip warn">Données à fournir</span>'
        with cols[i % 2]:
            with st.container(border=True):
                st.markdown(f'<div class="ico">{cfg["icon"]}</div>'
                            f'<div class="card-t">{cfg["title"]}</div><div class="card-p">{cfg["tagline"]}</div>{chips}',
                            unsafe_allow_html=True)
                st.write("")
                st.page_link(PAGES[key], label="Ouvrir le modèle", icon="➡️")
    st.write("")
    st.markdown("##### Comment ça marche")
    s1, s2, s3 = st.columns(3)
    s1.markdown("**1 · Choisissez un modèle**  \nChaque page correspond à l'un de vos notebooks.")
    s2.markdown("**2 · Saisissez un profil**  \nOu chargez un exemple tiré du jeu de données.")
    s3.markdown("**3 · Interprétez**  \nGroupe prédit, position sur la carte et profil comparé au groupe.")


def page_customer(): model_page("customer")
def page_heart(): model_page("heart")
def page_wholesale(): model_page("wholesale")
def page_wheat(): model_page("wheat")


def main():
    st.set_page_config(page_title="ClusterLab · Segmentation", page_icon="🧩", layout="wide")
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
    PAGES["home"] = st.Page(page_home, title="Accueil", icon="🏠", url_path="accueil", default=True)
    for key, fn in (("customer", page_customer), ("heart", page_heart),
                    ("wholesale", page_wholesale), ("wheat", page_wheat)):
        PAGES[key] = st.Page(fn, title=MODELS[key]["page"], icon=MODELS[key]["icon"], url_path=key)
    nav = st.navigation(list(PAGES.values()))
    st.sidebar.caption("ClusterLab · scikit-learn & Streamlit")
    nav.run()


if __name__ == "__main__":
    main()
