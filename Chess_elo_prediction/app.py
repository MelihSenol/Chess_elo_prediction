import tkinter as tk
from tkinter import messagebox, scrolledtext
import chess.pgn
import joblib
import pandas as pd
import numpy as np
import re
from io import StringIO

saat_dzn = re.compile(r'\[%clk (\d+:\d+:\d+|\d+:\d+)\]')

def saat_çözümle(comment):
    match = saat_dzn.search(comment)
    return match.group(1) if match else None

def saat_saniye_cevir(saat_dizi):
    part = list(map(int, saat_dizi.split(':')))
    if len(part) == 3:
        return part[0]*3600 + part[1]*60 + part[2]
    elif len(part) == 2:
        return part[0]*60 + part[1]
    return 0

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
                max(kullanılan_zaman) if kullanılan_zaman else 0,
                min(kullanılan_zaman) if kullanılan_zaman else 0,
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
        pgn_text = pgn_text_alanı.get("1.0", tk.END).strip()
        if not pgn_text:
            messagebox.showerror("Hata", "Lütfen PGN metnini girin.")
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
        
        result_label.config(text=f"Tahmin Edilen ELO: {predicted_elo[0]:.1f}\nGerçek ELO: {sample_data['target']}")
    except Exception as e:
        messagebox.showerror("Hata", f"Bir hata oluştu: {str(e)}")

model = joblib.load('chess_elo_model.pkl')
encoder = joblib.load('encoder.pkl')
opening_cols = joblib.load('opening_cols.pkl')

root = tk.Tk()
root.title("Satranç ELO Tahmin Uygulaması")

tk.Label(root, text="PGN Metni:").grid(row=0, column=0)
pgn_text_alanı = scrolledtext.ScrolledText(root, width=60, height=20)
pgn_text_alanı.grid(row=1, column=0, columnspan=2)

predict_button = tk.Button(root, text="Tahmin Et", command=predict_elo)
predict_button.grid(row=2, column=0, columnspan=2)

result_label = tk.Label(root, text="Tahmin Edilen ELO: -\nGerçek ELO: -")
result_label.grid(row=3, column=0, columnspan=2)

root.mainloop()