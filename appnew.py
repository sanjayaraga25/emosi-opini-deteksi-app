# -*- coding: utf-8 -*-
# app.py — Analisis Emosi & Ujaran Kebencian + Laporan Agregat (All-in-One)

import os, io, math, time, json, platform, random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import streamlit as st
from transformers import AutoTokenizer, AutoModel

# =============== Page ===============
st.set_page_config(page_title="Analisis Emosi & Ujaran Kebencian (All-in-One)", layout="wide")
st.markdown("""
<style>
  .pill{display:inline-block;padding:6px 10px;margin:4px 6px 0 0;border-radius:999px;font-weight:600}
  .pill-emo{background:#F3F6FF}.pill-hs{background:#FFF6F6}
</style>
""", unsafe_allow_html=True)

# =============== Sidebar: Konfigurasi ===============
st.sidebar.title("⚙️ Pengaturan")
threshold  = st.sidebar.slider("Threshold Probabilitas (global)", 0.0, 1.0, 0.50, 0.01)
batch_size = st.sidebar.slider("Ukuran Batch (inferensi dataset)", 8, 256, 64, 8)

st.sidebar.subheader("📁 Path Wajib (.pt + Backbone)")
EMO_PT = st.sidebar.text_input("File model Emosi .pt", r"F:\SKRIPSI ALL FILE\EVALUASIPERLABEL\emosi_S2\S2a\best_model.pt")
HS_PT  = st.sidebar.text_input("File model HS .pt",   r"F:\SKRIPSI ALL FILE\EVALUASIPERLABEL\hs_out_S3\32\best_model.pt")
BACKBONE_DIR = st.sidebar.text_input(
    "Folder backbone IndoBERTweet (cache lokal)",
    r"F:\HF-CACHE\indolem\indobertweet-base-uncased",
    help="Jika kosong/tidak ada, akan fallback ke HF Hub (perlu internet satu kali untuk cache)."
)

st.sidebar.subheader("📦 Direktori keluaran laporan")
OUT_DIR = Path(st.sidebar.text_input("Folder output", r"F:\SKRIPSI ALL FILE\APLICATION-NEW\output")).expanduser()
OUT_DIR.mkdir(parents=True, exist_ok=True)

st.sidebar.divider()

# ====== FIX: seeding jangan di-reset tiap rerun ======
seed = st.sidebar.number_input("Seed (sampling)", 0, 10000, 42, 1)

def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)

# Hanya set seed kalau belum pernah atau seed berubah
if "has_seeded" not in st.session_state or st.session_state.get("last_seed") != seed:
    set_seed(seed)
    st.session_state["has_seeded"] = True
    st.session_state["last_seed"] = seed
# ================================================

st.sidebar.divider()
st.sidebar.subheader("🔧 Threshold per label (opsional)")
use_per_label = st.sidebar.checkbox("Aktifkan threshold per label", value=False)

# =============== Label set & device ===============
label_emosi = ["anger","fear","sadness"]
emoji_emosi = ["😡","😨","😢"]
label_hs    = ["HS","Abusive","HS_Individual","HS_Group","HS_Race","HS_Gender","HS_Other"]
emoji_hs    = ["🛑","🤬","👤","👥","🏳️","🚻","❓"]

emo_pretty = [f"{emoji_emosi[i]} {label_emosi[i]}" for i in range(len(label_emosi))]
hs_pretty  = [f"{emoji_hs[i]} {label_hs[i]}" for i in range(len(label_hs))]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# threshold map (default: global)
thr_map = {lbl: threshold for lbl in label_emosi + label_hs}
if use_per_label:
    st.sidebar.markdown("**Emosi**")
    for lbl in label_emosi:
        thr_map[lbl] = st.sidebar.slider(f"{lbl}", 0.0, 1.0, threshold, 0.01, key=f"thr_emo_{lbl}")
    st.sidebar.markdown("**Hate Speech**")
    for lbl in label_hs:
        thr_map[lbl] = st.sidebar.slider(f"{lbl}", 0.0, 1.0, threshold, 0.01, key=f"thr_hs_{lbl}")

# =============== Helpers umum ===============
def _load_tok_backbone(backbone_dir: str):
    """Load tokenizer IndoBERTweet dari folder lokal jika ada, jika tidak fallback ke HF Hub."""
    if backbone_dir and os.path.isdir(backbone_dir):
        tok = AutoTokenizer.from_pretrained(backbone_dir, local_files_only=True)
        return tok, backbone_dir, True, backbone_dir
    base_id = "indolem/indobertweet-base-uncased"
    tok = AutoTokenizer.from_pretrained(base_id)  # cache HF
    return tok, base_id, False, "HF hub/cache: indolem/indobertweet-base-uncased"

def _normalize_sd(sd):
    """Normalisasi state_dict agar kompatibel: buang 'module.' dan ekstrak state_dict di wrapper umum."""
    if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
    if isinstance(sd, dict) and "model_state_dict" in sd: sd = sd["model_state_dict"]
    if isinstance(sd, dict) and any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd

# =============== Definisi model dinamis (sesuai checkpoint) ===============
class EmotionModelDynamic(nn.Module):
    """Mampu memuat checkpoint emosi baik mode bert+lstm maupun bert+proj (tanpa LSTM)."""
    def __init__(self, backbone_id: str, local_only: bool,
                 use_lstm=False, hidden_size=256, bidirectional=True,
                 in_feat=768, num_labels=3, dropout=0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(backbone_id, local_files_only=local_only)
        self.use_lstm = use_lstm
        self.dropout = nn.Dropout(dropout)
        if use_lstm:
            self.lstm = nn.LSTM(768, hidden_size, batch_first=True, bidirectional=bidirectional)
            feat = hidden_size * (2 if bidirectional else 1)
            self.classifier = nn.Linear(feat, num_labels)
        else:
            self.proj = nn.Identity() if in_feat == 768 else nn.Linear(768, in_feat)
            self.classifier = nn.Linear(in_feat, num_labels)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        x = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if self.use_lstm:
            x, _ = self.lstm(x)
            x = x[:, 0]
            return self.classifier(self.dropout(x))
        cls = x[:, 0, :]
        cls = self.proj(self.dropout(cls))
        return self.classifier(cls)

class HateModelDynamic(nn.Module):
    """HS: selalu bert + BiLSTM + classifier linier (sesuai training)."""
    def __init__(self, backbone_id: str, local_only: bool, hidden_size=256, bidirectional=True, num_labels=7, dropout=0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(backbone_id, local_files_only=local_only)
        self.lstm = nn.LSTM(768, hidden_size, batch_first=True, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        feat = hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Linear(feat, num_labels)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        x = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x, _ = self.lstm(x)
        x = x[:, 0]
        return self.classifier(self.dropout(x))

@st.cache_resource(show_spinner=False)
def load_emotion_from_pt(emo_pt_path: str, backbone_dir: str):
    if not emo_pt_path or not os.path.isfile(emo_pt_path):
        raise RuntimeError(f"File .pt emosi tidak ditemukan: {emo_pt_path}")
    tok, base_id, local_only, tok_src = _load_tok_backbone(backbone_dir)
    sd = _normalize_sd(torch.load(emo_pt_path, map_location="cpu"))
    info = {"tokenizer_source": tok_src}

    if "lstm.weight_ih_l0" in sd:
        hidden = sd["lstm.weight_ih_l0"].shape[0] // 4
        bidir  = "lstm.weight_ih_l0_reverse" in sd
        mdl = EmotionModelDynamic(base_id, local_only, use_lstm=True, hidden_size=hidden, bidirectional=bidir, num_labels=3)
        missing, unexpected = mdl.load_state_dict(sd, strict=False)
        info.update({"mode": "bert+lstm", "hidden": hidden, "bidirectional": bidir})
    else:
        clf_w = sd.get("classifier.weight", None)
        if clf_w is None: raise RuntimeError("Checkpoint emosi tidak memiliki 'classifier.weight'.")
        in_feat = int(clf_w.shape[1]); num_labels = int(clf_w.shape[0])
        mdl = EmotionModelDynamic(base_id, local_only, use_lstm=False, in_feat=in_feat, num_labels=num_labels)
        # remap proyeksi bila namanya beda
        pw = next((k for k,v in sd.items() if k.endswith(".weight") and tuple(v.shape)==(in_feat, 768)), None)
        pb = next((k for k,v in sd.items() if k.endswith(".bias")   and tuple(v.shape)==(in_feat,)), None)
        if pw: sd["proj.weight"] = sd.pop(pw)
        if pb: sd["proj.bias"]   = sd.pop(pb)
        missing, unexpected = mdl.load_state_dict(sd, strict=False)
        info.update({"mode": "bert+proj", "in_feat": in_feat, "remapped_proj": bool(pw or pb)})

    mdl.eval().to(DEVICE)
    info.update({"missing_keys": missing, "unexpected_keys": unexpected})
    return tok, mdl, info

@st.cache_resource(show_spinner=False)
def load_hs_from_pt(hs_pt_path: str, backbone_dir: str):
    if not hs_pt_path or not os.path.isfile(hs_pt_path):
        raise RuntimeError(f"File .pt HS tidak ditemukan: {hs_pt_path}")
    tok, base_id, local_only, tok_src = _load_tok_backbone(backbone_dir)
    sd = _normalize_sd(torch.load(hs_pt_path, map_location="cpu"))

    w_ih = sd.get("lstm.weight_ih_l0", None)
    if w_ih is None: raise RuntimeError("Checkpoint HS tidak memiliki 'lstm.weight_ih_l0'.")
    hidden = w_ih.shape[0] // 4
    bidir  = "lstm.weight_ih_l0_reverse" in sd
    clf_w  = sd.get("classifier.weight", None)
    num_labels = int(clf_w.shape[0]) if clf_w is not None else 7

    mdl = HateModelDynamic(base_id, local_only, hidden_size=hidden, bidirectional=bidir, num_labels=num_labels)
    missing, unexpected = mdl.load_state_dict(sd, strict=False)
    mdl.eval().to(DEVICE)
    info = {"tokenizer_source": tok_src, "inferred_hidden": hidden, "inferred_bidirectional": bidir,
            "inferred_num_labels": num_labels, "missing_keys": missing, "unexpected_keys": unexpected}
    return tok, mdl, info

# =============== Load model + ukur cold start ===============
with st.spinner("🔄 Memuat model…"):
    t0 = time.perf_counter()
    tokenizer_emo, model_emo, emo_info = load_emotion_from_pt(EMO_PT, BACKBONE_DIR)
    emo_cold = time.perf_counter() - t0

    t1 = time.perf_counter()
    tokenizer_hs, model_hs, hs_info   = load_hs_from_pt(HS_PT, BACKBONE_DIR)
    hs_cold  = time.perf_counter() - t1

# =============== Fungsi inferensi ===============
@torch.no_grad()
def predict_single(text: str):
    emo_inputs = tokenizer_emo(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    emo_inputs = {k: v.to(DEVICE) for k, v in emo_inputs.items()}
    emo_args   = {k: emo_inputs[k] for k in ("input_ids", "attention_mask") if k in emo_inputs}
    emo_probs  = torch.sigmoid(model_emo(**emo_args)).squeeze().detach().cpu().numpy()

    hs_inputs = tokenizer_hs(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    hs_inputs = {k: v.to(DEVICE) for k, v in hs_inputs.items()}
    hs_args   = {k: hs_inputs[k] for k in ("input_ids", "attention_mask") if k in hs_inputs}
    hs_probs  = torch.sigmoid(model_hs(**hs_args)).squeeze().detach().cpu().numpy()
    return emo_probs.astype(float), hs_probs.astype(float)

@torch.no_grad()
def predict_batch(texts, batch=64):
    N = len(texts)
    emo_out = np.zeros((N, len(label_emosi)), dtype=np.float32)
    hs_out  = np.zeros((N, len(label_hs)), dtype=np.float32)
    steps = math.ceil(N / batch)
    prog = st.progress(0, text="⏳ Memproses batch…")
    t0 = time.perf_counter()
    for i in range(steps):
        sl = slice(i*batch, min((i+1)*batch, N))
        chunk = ["" if pd.isna(x) else str(x) for x in texts[sl]]

        emo_inputs = tokenizer_emo(chunk, return_tensors='pt', truncation=True, padding=True, max_length=128)
        emo_inputs = {k: v.to(DEVICE) for k, v in emo_inputs.items()}
        emo_args   = {k: emo_inputs[k] for k in ("input_ids", "attention_mask") if k in emo_inputs}
        emo_out[sl, :] = torch.sigmoid(model_emo(**emo_args)).detach().cpu().numpy()

        hs_inputs = tokenizer_hs(chunk, return_tensors='pt', truncation=True, padding=True, max_length=128)
        hs_inputs = {k: v.to(DEVICE) for k, v in hs_inputs.items()}
        hs_args   = {k: hs_inputs[k] for k in ("input_ids", "attention_mask") if k in hs_inputs}
        hs_out[sl, :] = torch.sigmoid(model_hs(**hs_args)).detach().cpu().numpy()

        prog.progress((i+1)/steps, text=f"⏳ Memproses batch… ({i+1}/{steps})")
    total_time = time.perf_counter() - t0
    prog.empty()
    tps = (N / total_time) if total_time > 0 else float("inf")
    return emo_out, hs_out, total_time, tps

def active_with_map(probs, labels, emojis, thr_default, thr_map=None):
    out = []
    for i, lbl in enumerate(labels):
        thr = thr_map.get(lbl, thr_default) if thr_map else thr_default
        if probs[i] >= thr:
            out.append(f"{emojis[i]} {lbl}")
    return out

def df_bar(row, names):
    return pd.DataFrame({"Label": names, "Prob": [float(x) for x in row]}).sort_values("Label")

def download_csv(df, fname):
    buf = io.StringIO(); df.to_csv(buf, index=False)
    st.download_button("💾 Unduh CSV", buf.getvalue(), file_name=fname, mime="text/csv")

# ======= util aggregasi =======
def prevalence_table(binary_mat, labels):
    counts = binary_mat.sum(axis=0).astype(int)
    total  = int(binary_mat.shape[0]) if binary_mat.size else 0
    prop   = (counts / total) if total > 0 else np.zeros_like(counts, dtype=float)
    return pd.DataFrame({"label": labels, "count": counts, "proportion": prop})

# =============== UI ===============
c1, c2, c3 = st.columns([1.6,1,1])
with c1: st.title("📊 Analisis Emosi & Ujaran Kebencian — IndoBERTweet-BiLSTM")
with c2: st.metric("💻 Device", "CUDA" if DEVICE=="cuda" else "CPU")
with c3: st.metric("⚖️ Threshold", f"{threshold:.2f}")
st.caption("Checkpoint terbaik: Emosi (1e-5, BS=64), Hate Speech (5e-5, BS=32).")

tab1, tab2 = st.tabs(["📄 Analisis Kalimat", "📂 Analisis Dataset (+ Laporan)"])

# --------- Tab 1: satu kalimat ---------
with tab1:
    st.subheader("Masukkan Kalimat / Tweet")
    txt = st.text_area("Teks:", height=150, placeholder="Tulis teks di sini…")
    if st.button("🔍 Analisis"):
        if not txt.strip():
            st.warning("Teks tidak boleh kosong.")
        else:
            with st.spinner("Menganalisis…"):
                emo_p, hs_p = predict_single(txt)
            act_emo = active_with_map(emo_p, label_emosi, emoji_emosi, threshold, thr_map if use_per_label else None)
            act_hs  = active_with_map(hs_p,  label_hs,  emoji_hs,  threshold, thr_map if use_per_label else None)
            L, R = st.columns(2)
            with L:
                st.markdown("**🎭 Emosi**")
                st.markdown(" ".join([f"<span class='pill pill-emo'>{x}</span>" for x in act_emo]) or "—", unsafe_allow_html=True)
                st.bar_chart(df_bar(emo_p, emo_pretty), x="Label", y="Prob", height=240)
            with R:
                st.markdown("**🚨 Hate Speech**")
                st.markdown(" ".join([f"<span class='pill pill-hs'>{x}</span>" for x in act_hs]) or "—", unsafe_allow_html=True)
                st.bar_chart(df_bar(hs_p, hs_pretty), x="Label", y="Prob", height=240)

# --------- Tab 2: dataset + laporan agregat ---------
with tab2:
    st.subheader("Unggah Dataset (.csv / .xlsx)")
    up = st.file_uploader("Unggah file:", type=["csv","xlsx"])
    if up:
        try:
            df_raw = pd.read_csv(up) if up.name.lower().endswith(".csv") else pd.read_excel(up)
        except Exception as e:
            st.error(f"Gagal baca file: {e}"); st.stop()
        if df_raw.empty: st.error("File kosong."); st.stop()

        text_cols = [c for c in df_raw.columns if df_raw[c].dtype == object] or list(df_raw.columns)
        text_col = st.selectbox("Pilih kolom teks", text_cols, index=0)
        st.dataframe(df_raw.head(10), use_container_width=True)

        c = st.columns(3)

        # ---- Proses Semua: hanya PROSES, belum simpan ----
        if c[0].button("▶️ Proses Semua"):
            texts = list(df_raw[text_col].astype(str))
            with st.spinner(f"Menganalisis {len(texts)} baris…"):
                emo_probs, hs_probs, inf_time, tps = predict_batch(texts, batch=batch_size)

            # Prob DF
            prob_cols = label_emosi + label_hs
            prob_df = pd.DataFrame(np.hstack([emo_probs, hs_probs]), columns=prob_cols).add_suffix("_prob")

            # Bin DF (dengan suffix _pred) + per-label threshold bila aktif
            thr_arr = np.array([thr_map[c] for c in (label_emosi + label_hs)])
            bin_mat = (pd.DataFrame(np.hstack([emo_probs, hs_probs]), columns=(label_emosi + label_hs)).values >= thr_arr).astype(int)
            bin_df  = pd.DataFrame(bin_mat, columns=(label_emosi + label_hs)).add_suffix("_pred")

            # Hindari bentrok nama kolom input
            safe_df = df_raw.drop(columns=[c for c in df_raw.columns if c in (label_emosi + label_hs)], errors="ignore")
            out = pd.concat([safe_df.reset_index(drop=True), prob_df, bin_df], axis=1)

            # ===== Laporan agregat (belum disimpan, hanya disiapkan) =====
            hs_bin = out[[f"{l}_pred" for l in label_hs]].values
            emo_bin = out[[f"{l}_pred" for l in label_emosi]].values

            hs_prev = prevalence_table(hs_bin, label_hs)
            emo_prev = prevalence_table(emo_bin, label_emosi)

            hs_any = (hs_bin.sum(axis=1) > 0).astype(int)
            df_cross = pd.DataFrame({"HS_any": hs_any, **{l: emo_bin[:, i] for i, l in enumerate(label_emosi)}})
            ct = df_cross.groupby("HS_any")[label_emosi].sum().astype(int)

            # metadata runtime (cold start & throughput per model)
            meta = {
                "env": {
                    "device": DEVICE,
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "transformers": __import__("transformers").__version__,
                    "platform": platform.platform(),
                    "gpu_name": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None,
                },
                "ops": {
                    "num_tweets": len(texts),
                    "batch_size": int(batch_size),
                    "thresholds_hs": {l: float(thr_map[l]) for l in label_hs},
                    "thresholds_emo": {l: float(thr_map[l]) for l in label_emosi},
                    "cold_start_sec": {"hs": float(hs_cold), "emo": float(emo_cold)},
                    "inference_time_sec": {
                        "hs": float(inf_time),   # total gabungan; jika mau pisah HS/Emo, pecah timing per forward
                        "emo": float(inf_time)
                    },
                    "throughput_tweets_per_sec": {
                        "hs": float(tps),
                        "emo": float(tps)
                    }
                }
            }

            # simpan di session_state, belum ditulis ke disk
            st.session_state["out_all"] = out
            st.session_state["hs_prev_all"] = hs_prev
            st.session_state["emo_prev_all"] = emo_prev
            st.session_state["ct_all"] = ct
            st.session_state["meta_all"] = meta

            st.success("✅ Analisis selesai. Preview hasil ada di bawah. Klik tombol simpan kalau sudah yakin.")

        # ---- Prediksi 1 baris acak ----
        if 'rand_key' not in st.session_state: st.session_state.rand_key = 0
        if c[2].button("🎲 Prediksi 1 Baris Acak", key=f"btn_rand_{st.session_state.rand_key}"):
            st.session_state.rand_key += 1
            row = df_raw.sample(1).iloc[0]
            tx = str(row[text_col]); st.write("**Tweet Acak:**", tx)
            with st.spinner("Memprediksi…"):
                emo_p, hs_p = predict_single(tx)
            act_emo = active_with_map(emo_p, label_emosi, emoji_emosi, threshold, thr_map if use_per_label else None)
            act_hs  = active_with_map(hs_p,  label_hs,  emoji_hs,  threshold, thr_map if use_per_label else None)
            L, R = st.columns(2)
            with L:
                st.markdown("**🎭 Emosi**")
                st.markdown(" ".join([f"<span class='pill pill-emo'>{x}</span>" for x in act_emo]) or "—", unsafe_allow_html=True)
                st.bar_chart(df_bar(emo_p, emo_pretty), x="Label", y="Prob", height=240)
            with R:
                st.markdown("**🚨 Hate Speech**")
                st.markdown(" ".join([f"<span class='pill pill-hs'>{x}</span>" for x in act_hs]) or "—", unsafe_allow_html=True)
                st.bar_chart(df_bar(hs_p, hs_pretty), x="Label", y="Prob", height=240)

        # ---- Proses Sampling (tampil & unduh CSV prediksi saja) ----
        if c[1].button("🎯 Proses Sampling"):
            n = st.number_input("Jumlah sampel", 1, len(df_raw), min(200, len(df_raw)))
            samp = df_raw.sample(int(n), random_state=seed).reset_index(drop=True)
            texts = list(samp[text_col].astype(str))
            with st.spinner(f"Menganalisis {len(samp)} baris…"):
                emo_probs, hs_probs, _, _ = predict_batch(texts, batch=batch_size)

            prob_df = pd.DataFrame(np.hstack([emo_probs, hs_probs]), columns=(label_emosi + label_hs)).add_suffix("_prob")
            thr_arr = np.array([thr_map[c] for c in (label_emosi + label_hs)])
            bin_df  = (pd.DataFrame(np.hstack([emo_probs, hs_probs]), columns=(label_emosi + label_hs)).values >= thr_arr).astype(int)
            bin_df  = pd.DataFrame(bin_df, columns=(label_emosi + label_hs)).add_suffix("_pred")

            safe_df = samp.drop(columns=[c for c in samp.columns if c in (label_emosi + label_hs)], errors="ignore")
            out_samp = pd.concat([safe_df, prob_df, bin_df], axis=1)
            st.dataframe(out_samp.head(20), use_container_width=True)
            st.markdown("### 📊 Distribusi Label (Sampling)")
            st.bar_chart(bin_df.sum().sort_values(ascending=False))
            download_csv(out_samp, "hasil_sampling.csv")

        # ======== Preview & Tombol Simpan (setelah Proses Semua) ========
        if "out_all" in st.session_state:
            out = st.session_state["out_all"]
            hs_prev = st.session_state["hs_prev_all"]
            emo_prev = st.session_state["emo_prev_all"]
            ct = st.session_state["ct_all"]

            st.markdown("### 📄 Hasil (preview)")
            st.dataframe(out.head(50), use_container_width=True)
            st.markdown("### 📊 Ringkasan Prevalensi")
            cL, cR = st.columns(2)
            with cL:
                st.caption("Hate Speech")
                st.dataframe(hs_prev, use_container_width=True)
            with cR:
                st.caption("Emosi")
                st.dataframe(emo_prev, use_container_width=True)
            st.markdown("### 🔗 Emosi di dalam HS (crosstab)")
            st.dataframe(ct, use_container_width=True)

            if st.button("💾 Simpan Laporan ke Folder Output"):
                out_path = OUT_DIR / "hasil_prediksi.csv"
                out.to_csv(out_path, index=False)
                hs_prev.to_csv(OUT_DIR / "summary_hatespeech_prevalence.csv", index=False)
                emo_prev.to_csv(OUT_DIR / "summary_emotion_prevalence.csv", index=False)
                ct.to_csv(OUT_DIR / "crosstab_emotion_within_HS.csv")

                meta = st.session_state.get("meta_all", {})
                with open(OUT_DIR / "run_metadata.json", "w") as f:
                    json.dump(meta, f, indent=2)

                st.success(f"✅ Laporan disimpan di: {OUT_DIR}")

# =============== Footer kecil ===============
st.caption(f"Cold-start: Emosi {emo_cold:.2f}s • HS {hs_cold:.2f}s  |  Output folder: {OUT_DIR}")
