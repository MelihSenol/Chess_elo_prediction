import customtkinter as ctk
from tkinter import messagebox
from tkinter.scrolledtext import ScrolledText
import chess.pgn
import joblib
import pandas as pd
import numpy as np
import re
import random
from io import StringIO

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

saat_dzn = re.compile(r'\[%clk (\d+:\d+:\d+|\d+:\d+)\]')

def tum_oyunlari_yukle(dosya_adi):
    oyunlar = []
    try:
        with open(dosya_adi, "r", encoding="utf-8") as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                oyun_io = StringIO()
                print(game, file=oyun_io)
                oyunlar.append(oyun_io.getvalue())
    except Exception as e:
        messagebox.showerror("Hata", f"Oyunlar yüklenemedi: {str(e)}")
    return oyunlar

oyun_listesi = tum_oyunlari_yukle("Test_verileri.pgn")

def saat_çözümle(comment):
    match = saat_dzn.search(comment)
    return match.group(1) if match else None

def saat_saniye_cevir(saat_dizi):
    part = list(map(int, saat_dizi.split(':')))
    return part[0]*3600 + part[1]*60 + part[2] if len(part)==3 else part[0]*60 + part[1]

def zaman_kontrolü(tc):
    if tc == '-': return 0, 0
    part = tc.split('+')
    return int(part[0]), int(part[1]) if len(part) > 1 else 0

def process_pgn(pgn_text):
    try:
        pgn = StringIO(pgn_text)
        game = chess.pgn.read_game(pgn)
        if not game:
            raise ValueError("Geçersiz PGN formatı.")

        white_elo = int(game.headers.get('WhiteElo', 0))
        black_elo = int(game.headers.get('BlackElo', 0))
        opening = game.headers.get('Opening', 'Unknown')
        tc = game.headers.get('TimeControl', '0+0')
        initial, increment = zaman_kontrolü(tc)

        beyaz_zaman, siyah_zaman = [], []
        board = game.board()
        for node in game.mainline():
            comment = node.comment
            saat_dizi = saat_çözümle(comment)
            if saat_dizi:
                saat_saniye = saat_saniye_cevir(saat_dizi)
                if board.turn == chess.WHITE:
                    beyaz_zaman.append(saat_saniye)
                else:
                    siyah_zaman.append(saat_saniye)
            board.push(node.move)

        def zaman_hesapla(saatler, initial_ek):
            if not saatler:
                return [0]*5
            kullanılan_zaman = []
            süre_hesapla = initial_ek
            for saat in reversed(saatler):
                harcanan_zaman = süre_hesapla + increment - saat
                kullanılan_zaman.append(harcanan_zaman)
                süre_hesapla = saat
            kullanılan_zaman = list(reversed(kullanılan_zaman))
            return [
                np.mean(kullanılan_zaman),
                np.std(kullanılan_zaman) if len(kullanılan_zaman) > 1 else 0,
                max(kullanılan_zaman),
                min(kullanılan_zaman),
                sum(kullanılan_zaman)
            ]

        white_features = zaman_hesapla(beyaz_zaman, initial)
        black_features = zaman_hesapla(siyah_zaman, initial)

        return {
            'white_elo': white_elo,
            'opening': opening,
            'tc_initial': initial,
            'tc_inc': increment,
            'white_time_mean': white_features[0],
            'white_time_std': white_features[1],
            'white_time_max': white_features[2],
            'white_time_min': white_features[3],
            'white_time_total': white_features[4],
            'black_time_mean': black_features[0],
            'black_time_std': black_features[1],
            'black_time_max': black_features[2],
            'black_time_min': black_features[3],
            'black_time_total': black_features[4],
            'target': black_elo
        }
    except Exception as e:
        raise ValueError(f"PGN işlenirken hata oluştu: {str(e)}")

def predict_elo():
    try:
        pgn_text = pgn_text_alani.get("1.0", "end").strip()
        if not pgn_text:
            messagebox.showerror("Hata", "Lütfen PGN metni girin.")
            return

        sample_data = process_pgn(pgn_text)

        opening_df = pd.DataFrame({'opening': [sample_data['opening']]})
        opening_encoded = encoder.transform(opening_df)
        opening_features = pd.DataFrame(opening_encoded, columns=opening_cols)

        features = pd.DataFrame({
            'white_elo': [sample_data['white_elo']],
            'tc_initial': [sample_data['tc_initial']],
            'tc_inc': [sample_data['tc_inc']],
            'white_time_mean': [sample_data['white_time_mean']],
            'white_time_std': [sample_data['white_time_std']],
            'white_time_max': [sample_data['white_time_max']],
            'white_time_min': [sample_data['white_time_min']],
            'white_time_total': [sample_data['white_time_total']],
            'black_time_mean': [sample_data['black_time_mean']],
            'black_time_std': [sample_data['black_time_std']],
            'black_time_max': [sample_data['black_time_max']],
            'black_time_min': [sample_data['black_time_min']],
            'black_time_total': [sample_data['black_time_total']]
        })

        features = pd.concat([features, opening_features], axis=1)
        predicted_elo = model.predict(features)

        sonuc_kart.configure(
            text=f"Tahmin Edilen ELO: {predicted_elo[0]:.1f}\nGerçek ELO: {sample_data['target']}"
        )
    except Exception as e:
        messagebox.showerror("Hata", f"Hata oluştu: {str(e)}")

def rastgele_pgn_yukle():
    try:
        if not oyun_listesi:
            raise ValueError("Yüklenecek oyun yok.")
        secilen_oyun = random.choice(oyun_listesi)
        pgn_text_alani.delete("1.0", "end")
        pgn_text_alani.insert("end", secilen_oyun)
    except Exception as e:
        messagebox.showerror("Hata", f"PGN yüklenemedi: {str(e)}")

model = joblib.load("chess_elo_model.pkl")
encoder = joblib.load("encoder.pkl")
opening_cols = joblib.load("opening_cols.pkl")

app = ctk.CTk()
app.geometry("900x700")
app.title("♟️ Satranç ELO Tahmini")

baslik = ctk.CTkLabel(app, text="📊 Satranç ELO Tahmin Aracı", font=ctk.CTkFont(size=24, weight="bold"))
baslik.pack(pady=20)

pgn_frame = ctk.CTkFrame(app, corner_radius=10)
pgn_frame.pack(padx=20, pady=10, fill="both", expand=True)

pgn_label = ctk.CTkLabel(pgn_frame, text="Oyun Verisi:", font=ctk.CTkFont(size=16))
pgn_label.pack(anchor="w", pady=(10, 0), padx=10)

pgn_text_alani = ScrolledText(pgn_frame, height=20, font=("Courier", 10), bg="#1e1e1e", fg="white", insertbackground="white")
pgn_text_alani.pack(padx=10, pady=10, fill="both", expand=True)

buton_frame = ctk.CTkFrame(app)
buton_frame.pack(pady=10)

tahmin_buton = ctk.CTkButton(buton_frame, text="🎯 Tahmin Et", width=150, command=predict_elo)
tahmin_buton.grid(row=0, column=0, padx=10)

rastgele_buton = ctk.CTkButton(buton_frame, text="🔀 Rastgele Oyun", width=150, command=rastgele_pgn_yukle)
rastgele_buton.grid(row=0, column=1, padx=10)

sonuc_kart = ctk.CTkLabel(
    app,
    text="Tahmin Edilen ELO: -\nGerçek ELO: -",
    font=ctk.CTkFont(size=16),
    width=400,
    height=100,
    corner_radius=12,
    fg_color="#333333",
    text_color="white",
    justify="center"
)
sonuc_kart.pack(pady=20)

app.mainloop()
