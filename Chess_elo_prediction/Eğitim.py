import chess.pgn
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
import joblib

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

def load_pgn(file_path):
    games = []
    with open(file_path, 'r', encoding='utf-8') as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if not game:
                break
            games.append(game)
    return games

def process_games(games):
    data = []
    for game in games:
        try:
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

            def zaman_hesapla(saatler, ilk_zaman_ek):
                if not saatler:
                    return [0]*5
                
                kullanılan_zaman = []
                önceki_saat = ilk_zaman_ek
                for saat in reversed(saatler):
                    harcanan_zaman = önceki_saat + increment - saat
                    kullanılan_zaman.append(harcanan_zaman)
                    önceki_saat = saat
                
                kullanılan_zaman = list(reversed(kullanılan_zaman))
                
                return [
                    np.mean(kullanılan_zaman),
                    np.std(kullanılan_zaman) if len(kullanılan_zaman) > 1 else 0,
                    max(kullanılan_zaman) if kullanılan_zaman else 0,
                    min(kullanılan_zaman) if kullanılan_zaman else 0,
                    sum(kullanılan_zaman)
                ]

            beyaz_özellik = zaman_hesapla(beyaz_zaman, initial)
            siyah_özellik = zaman_hesapla(siyah_zaman, initial)

            data.append({
                'white_elo': white_elo,
                'opening': opening,
                'tc_initial': initial,
                'tc_inc': increment,
                'white_time_mean': beyaz_özellik[0],
                'white_time_std': beyaz_özellik[1],
                'white_time_max': beyaz_özellik[2],
                'white_time_min': beyaz_özellik[3],
                'white_time_total': beyaz_özellik[4],
                'black_time_mean': siyah_özellik[0],
                'black_time_std': siyah_özellik[1],
                'black_time_max': siyah_özellik[2],
                'black_time_min': siyah_özellik[3],
                'black_time_total': siyah_özellik[4],
                'target': black_elo
            })
        except Exception as e:
            print(f"Hata oluştu: {str(e)}")
            continue
    return pd.DataFrame(data)

def train_model(df, model_path='chess_elo_model.pkl', encoder_path='encoder.pkl', opening_cols_path='opening_cols.pkl'):
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    opening_encoded = encoder.fit_transform(df[['opening']])
    opening_cols = [f"opening_{i}" for i in range(opening_encoded.shape[1])]
    
    X = pd.concat([
        df.drop(['opening', 'target'], axis=1),
        pd.DataFrame(opening_encoded, columns=opening_cols)
    ], axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        objective='reg:squarederror',
        n_estimators=500,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)

    joblib.dump(model, model_path)
    joblib.dump(encoder, encoder_path)
    joblib.dump(opening_cols, opening_cols_path)
    print(f"Model başarıyla kaydedildi: {model_path}")
    print(f"Encoder başarıyla kaydedildi: {encoder_path}")
    print(f"Opening columns başarıyla kaydedildi: {opening_cols_path}")

    preds = model.predict(X_test)
    print(f"Test MAE: {mean_absolute_error(y_test, preds):.2f}")
    print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, preds)):.2f}")
    print(f"Test R²: {r2_score(y_test, preds):.2f}")

if __name__ == "__main__":
    games = load_pgn('games_3000.pgn')
    df = process_games(games)

    train_model(df)