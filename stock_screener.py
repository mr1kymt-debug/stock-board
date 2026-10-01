"""株スクリーナー（自分用）

使い方:
    pip install yfinance pandas
    python stock_screener.py

3つの見方で銘柄を評価し、日本株・米国株それぞれの市場内でランキングします。
  1. 上昇の勢い … 株価の上がり方と、売上・利益の伸び
  2. 頭打ちリスク … 上がってきた株が、この先足踏みしそうかどうか
  3. 1か月後の見立て … 上の2つに、直近のニュース（news_view.json）と市場全体の状況を加えた判断
     あわせて、過去1年の値動きの大きさから「1か月後の株価の目安の範囲」も出します。

総合スコア = 上昇の勢い75% + 頭打ちしにくさ25%

分析する銘柄は、2段階で選びます。
  1. 東証（日本株）とS&P500（米国株）のほぼ全銘柄を対象に、値動きの勢いだけで順位をつけます
     （売上や利益までは見ません。値動きだけなら、まとめて速く調べられるためです）
  2. 各市場で勢いの良かった上位 SHORTLIST_SIZE 社に、UNIVERSE（標準の10社）と
     my_stocks.txt の「自分の銘柄」を加え、そこだけ詳しく分析します
     （売上・利益・頭打ちリスクなどは、この詳しい分析でだけ出ます）

全銘柄の一覧が取得できなかったときは、1をスキップし、2の銘柄だけで分析します。

あくまで候補を絞り込む補助ツールで、将来の値上がりを保証するものではありません。
"""
import datetime
import json
import pathlib
import io
import re
import unicodedata
import warnings

import requests

import numpy as np
import pandas as pd
import yfinance as yf

# 標準の対象銘柄（日米10社ずつ）。日本株は末尾に ".T"
# 自分で足したい銘柄は、ここではなく my_stocks.txt に書くのがおすすめです
UNIVERSE = {
    "US": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "JPM", "WMT"],
    "JP": ["7203.T", "6758.T", "9984.T", "8035.T", "6861.T", "8306.T", "6501.T", "7974.T", "9983.T", "6857.T"],
}
MY_STOCKS_FILE = "my_stocks.txt"

# 「市場全体」の一次選抜（値動きの勢いだけで見る、2段階選抜の1段目）
SHORTLIST_SIZE = 150                 # 各市場で、この上位社数だけを詳しく分析する
SP500_LIST_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
JPX_LIST_URL = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xlsx"  # JPXは拡張子を変えることがあるので、取得に失敗したら自動で.xlsに切り替える
FETCH_TIMEOUT = 30
BULK_CHUNK = 150                     # 値動きだけをまとめて取得するときの、1回あたりの銘柄数
# 上昇の勢い（高いほど良い指標）。合計1。仮の値なので、使いながら調整してください
WEIGHTS = {
    "ret_3m": 0.25,           # 3か月の値上がり率
    "ret_1m": 0.10,           # 1か月の値上がり率
    "above_ma200": 0.10,      # 約10か月の平均株価との差（長い目で見た上昇傾向）
    "volume_ratio": 0.10,      # 直近5日÷60日の取引量（注目度の変化）
    "revenue_growth": 0.20,   # 売上の伸び
    "earnings_growth": 0.20,  # 利益の伸び
    "pe_improve": 0.05,       # 実績PER÷予想PER（1より大きいと増益予想）
}
STAGE1_KEYS = ["ret_3m", "ret_1m", "above_ma200", "volume_ratio"]  # 一次選抜で使う、値動きだけの指標
STAGE1_WEIGHTS = {k: WEIGHTS[k] for k in STAGE1_KEYS}

# 頭打ちリスク（高いほど頭打ちしやすい指標）。合計1
PLATEAU_WEIGHTS = {
    "rsi14": 0.30,         # 買われすぎ度（0〜100。70超は過熱ぎみ）
    "slowdown": 0.30,      # 上昇ペースの鈍り（3か月の月平均ペース − 直近1か月）
    "forward_pe": 0.25,    # 株価は今後の利益の何倍か（高いほど割高）
    "volume_ratio": 0.15,  # 取引の細り（取引量が少ないほどリスク大）
}
LOW_IS_RISKY = {"volume_ratio"}  # 値が「小さい」ほどリスクが高い指標

PLATEAU_SHARE = 0.25   # 総合スコアのうち「頭打ちしにくさ」が占める割合
HIGH_RISK = 67         # これ以上なら「頭打ち注意」
LOW_RISK = 33          # これ以下なら「上昇継続の可能性」

# 1か月後の見立て（news_view.json のニュースと市場全体の状況を加味）
NEWS_FILE = "news_view.json"
NEWS_MAX_AGE_DAYS = 10                              # これより古いニュースは見立てに使わない
TONE_VALUE = {"追い風": 1, "中立": 0, "逆風": -1}
TICKER_NEWS_WEIGHT = 0.75                           # 個別銘柄のニュースの効き方
MARKET_NEWS_WEIGHT = 0.50                           # 市場全体の状況の効き方
PLATEAU_PENALTY = 0.5                               # 「頭打ち注意」のときの減点
TRADING_DAYS_1M = 21                                # 1か月の営業日数


def compute_rsi(close: pd.Series, n: int = 14) -> float:
    """買われすぎ度（RSI）。上げ幅と下げ幅の平均の比から0〜100で表す。"""
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = up.iloc[-1] / down.iloc[-1]
        return float(100 - 100 / (1 + rs))


def compute_price_features(hist: pd.DataFrame) -> dict:
    """株価履歴（Close, Volume列）から価格系の指標を計算する。"""
    close = hist["Close"].dropna()
    volume = hist["Volume"].dropna()
    feats = {}
    if len(close) >= 1:
        feats["price"] = float(close.iloc[-1])
    if len(close) >= 22:
        feats["ret_1m"] = close.iloc[-1] / close.iloc[-22] - 1
    if len(close) >= 64:
        feats["ret_3m"] = close.iloc[-1] / close.iloc[-64] - 1
    if len(close) >= 200:
        feats["above_ma200"] = close.iloc[-1] / close.rolling(200).mean().iloc[-1] - 1
    if len(volume) >= 60 and volume.tail(60).mean() > 0:
        feats["volume_ratio"] = volume.tail(5).mean() / volume.tail(60).mean()
    if len(close) >= 30:
        rsi = compute_rsi(close)
        if np.isfinite(rsi):
            feats["rsi14"] = rsi
    if "ret_1m" in feats and "ret_3m" in feats:
        feats["slowdown"] = feats["ret_3m"] / 3 - feats["ret_1m"]
    if len(close) >= 60:
        # 過去約1年の日々の値動きの大きさ → 1か月分に換算
        daily = np.log(close).diff().dropna().tail(252).std()
        if np.isfinite(daily):
            feats["vol_1m"] = float(daily * np.sqrt(TRADING_DAYS_1M))
    return feats


def fetch_features(ticker: str) -> dict:
    """1銘柄分のデータを取得して指標にまとめる。取得失敗時は空の指標を返す。"""
    feats = {}
    try:
        t = yf.Ticker(ticker)
        feats.update(compute_price_features(t.history(period="1y")))
        info = t.info
        feats["revenue_growth"] = info.get("revenueGrowth")
        feats["earnings_growth"] = info.get("earningsGrowth")
        trailing, forward = info.get("trailingPE"), info.get("forwardPE")
        if trailing and forward and trailing > 0 and forward > 0:
            feats["pe_improve"] = trailing / forward
        if forward and forward > 0:
            feats["forward_pe"] = forward
    except Exception as e:  # ネットワークエラーや銘柄コード違いなど
        print(f"[警告] {ticker} の取得に失敗: {e}")
    return feats


def normalize_code(raw: str):
    """銘柄コードを整える。全角→半角、小文字→大文字、日本株は末尾に .T を付ける。使えない書き方は None。"""
    code = unicodedata.normalize("NFKC", raw).strip().upper()
    if not code:
        return None
    if re.fullmatch(r"\d{3}[0-9A-Z]", code):        # 日本株の4桁コード（例：7203、285A）
        code += ".T"
    return code if re.fullmatch(r"[0-9A-Z]{1,6}(\.T|-[A-Z])?", code) else None


def market_of(code: str) -> str:
    return "JP" if code.endswith(".T") else "US"


def load_my_stocks(path=MY_STOCKS_FILE):
    """my_stocks.txt から「自分の銘柄」を読み込む。1行に1つ。「#」から後ろはメモ。"""
    p = pathlib.Path(path)
    if not p.exists():
        return []
    codes = []
    for line in p.read_text(encoding="utf-8-sig").splitlines():
        line = line.replace("＃", "#").split("#", 1)[0]
        for part in re.split(r"[,\s、，]+", line):
            if not part:
                continue
            code = normalize_code(part)
            if code is None:
                print(f"[警告] my_stocks.txt の「{part}」は銘柄コードとして読めないため、飛ばしました。")
            elif code not in codes:
                codes.append(code)
    return codes


def build_universe(extra):
    """分析する銘柄を決める。
    東証・S&P500のほぼ全銘柄を対象に、値動きの勢いだけで一次選抜し（bulk_price_features）、
    各市場の上位 SHORTLIST_SIZE 社に、標準の銘柄（UNIVERSE）と自分の銘柄（extra）を必ず加える。
    一覧が取得できない市場は、一次選抜をせず、標準の銘柄＋自分の銘柄だけで進める。
    """
    must = {m: set(UNIVERSE[m]) for m in UNIVERSE}
    for code in extra:
        must[market_of(code)].add(code)

    full = {"US": fetch_sp500_universe(), "JP": fetch_tse_universe()}
    uni = {}
    for m in UNIVERSE:
        if full[m]:
            uni[m] = shortlist_market(m, full[m], must[m])
        else:
            uni[m] = sorted(must[m])
    return uni


def fetch_sp500_universe():
    """S&P500の構成銘柄の一覧を取得する（データはWikipediaを整理して配布しているもの）。失敗したら None。"""
    try:
        r = requests.get(SP500_LIST_URL, timeout=FETCH_TIMEOUT)
        r.raise_for_status()
        codes = []
        for raw in pd.read_csv(io.StringIO(r.text))["Symbol"]:
            c = normalize_code(str(raw).replace(".", "-"))  # BRK.B のような表記を BRK-B に合わせる
            if c:
                codes.append(c)
        return sorted(set(codes)) or None
    except Exception as e:
        print(f"[警告] S&P500の一覧を取得できませんでした（{e}）。米国株は標準の銘柄だけで進めます。")
        return None


def _fetch_tse_universe_from(url):
    """指定したURLからJPXの一覧表を読み込み、国内株式のコード一覧を返す（内部用、例外はそのまま投げる）。"""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    r = requests.get(url, headers=headers, timeout=FETCH_TIMEOUT)
    r.raise_for_status()
    buf = io.BytesIO(r.content)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            df = pd.read_excel(buf, engine="openpyxl")   # 現在の形式（.xlsx）
        except Exception:
            buf.seek(0)
            df = pd.read_excel(buf, engine="xlrd")        # 以前の形式（.xls）だった場合の保険
    market_col = next((c for c in df.columns if "市場" in str(c) and "商品" in str(c)), None)
    code_col = next((c for c in df.columns if "コード" == str(c).strip()), None)
    if market_col is None or code_col is None:
        raise ValueError("想定した列（コード／市場・商品区分）が見つかりません")
    stock = df[df[market_col].astype(str).str.contains("内国株式", na=False)]
    codes = [normalize_code(str(c)) for c in stock[code_col]]
    return sorted({c for c in codes if c}) or None


def fetch_tse_universe():
    """東証の上場銘柄の一覧を取得する（JPXが公開している一覧表）。国内株式だけに絞る。失敗したら None。"""
    try:
        return _fetch_tse_universe_from(JPX_LIST_URL)
    except Exception as e:
        # JPXはファイルの拡張子（.xls / .xlsx）を、月によって変えることがあるため、もう一方を1回だけ試す
        alt = JPX_LIST_URL.replace(".xlsx", ".xls") if JPX_LIST_URL.endswith(".xlsx") else JPX_LIST_URL.replace(".xls", ".xlsx")
        if alt != JPX_LIST_URL:
            try:
                return _fetch_tse_universe_from(alt)
            except Exception:
                pass
        print(f"[警告] 東証の銘柄一覧を取得できませんでした（{e}）。日本株は標準の銘柄だけで進めます。")
        return None


def bulk_price_features(tickers):
    """多くの銘柄の、値動きの指標だけをまとめて取得する（売上や利益は見ない、一次選抜用）。"""
    rows = []
    uniq = sorted(set(tickers))
    for i in range(0, len(uniq), BULK_CHUNK):
        chunk = uniq[i : i + BULK_CHUNK]
        print(f"  値動きを取得中: {i + 1}〜{i + len(chunk)} / {len(uniq)}")
        try:
            raw = yf.download(chunk, period="1y", auto_adjust=True, progress=False, group_by="column", threads=True)
        except Exception as e:
            print(f"  [警告] この{len(chunk)}件の取得に失敗しました（{e}）。飛ばします。")
            continue
        if raw is None or raw.empty:
            continue
        close_all = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]].rename(columns={"Close": chunk[0]})
        volume_all = raw["Volume"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Volume"]].rename(columns={"Volume": chunk[0]})
        for t in chunk:
            if t not in close_all.columns:
                continue
            f = compute_price_features(pd.DataFrame({"Close": close_all[t], "Volume": volume_all.get(t)}).dropna(how="all"))
            rows.append({"ticker": t, **{k: f.get(k) for k in STAGE1_KEYS}})
    return pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame(columns=STAGE1_KEYS)


def shortlist_market(market, full_list, must_include):
    """値動きの勢いだけで一次選抜し、上位 SHORTLIST_SIZE 社（＋必ず入れる銘柄）を返す。"""
    candidates = sorted(set(full_list) | set(must_include))
    feats = bulk_price_features(candidates)
    if feats.empty:
        return sorted(set(must_include))
    score = sum(w * feats[c].rank(pct=True).fillna(0.5) for c, w in STAGE1_WEIGHTS.items())
    ranked = score.sort_values(ascending=False).index.tolist()
    top = ranked[:SHORTLIST_SIZE]
    print(f"  {market}: 全{len(candidates)}銘柄のうち、値動きの勢いで上位{len(top)}銘柄を選びました")
    return sorted(set(top) | set(must_include))


def load_news(path=NEWS_FILE, today=None):
    """news_view.json を読み込む。無い・古い・壊れているときは None（ニュースなしで計算）。"""
    p = pathlib.Path(path)
    if not p.exists():
        print("[情報] news_view.json がないため、ニュースは見立てに使いません。")
        return None
    try:
        news = json.loads(p.read_text(encoding="utf-8"))
        as_of = datetime.date.fromisoformat(news["asOf"])
    except Exception as e:
        print(f"[警告] news_view.json を読み取れませんでした（{e}）。ニュースは使いません。")
        return None
    age = ((today or datetime.date.today()) - as_of).days
    if age > NEWS_MAX_AGE_DAYS:
        print(f"[警告] ニュースが{age}日前の情報なので、見立てには使いません。")
        return None
    return news


def _rank(df: pd.DataFrame, col: str, high_is_top: bool = True) -> pd.Series:
    """同じ市場の中での順位（0〜1）。データがない銘柄は真ん中（0.5）として扱う。"""
    return df.groupby("market")[col].rank(pct=True, ascending=high_is_top).fillna(0.5)


def direction_label(points: float) -> str:
    if points >= 1.0:
        return "上向き"
    if points >= 0.4:
        return "やや上向き"
    if points > -0.4:
        return "横ばい"
    if points > -1.0:
        return "やや下向き"
    return "下向き"


def build_scores(df: pd.DataFrame, news=None) -> pd.DataFrame:
    """上昇の勢い・頭打ちリスク・総合スコア・1か月後の見立てを計算する。"""
    df = df.copy()
    for col in set(WEIGHTS) | set(PLATEAU_WEIGHTS) | {"ret_3m", "above_ma200", "price", "vol_1m"}:
        if col not in df.columns:
            df[col] = np.nan

    rising = sum(w * _rank(df, c) for c, w in WEIGHTS.items())
    risk = sum(w * _rank(df, c, high_is_top=c not in LOW_IS_RISKY) for c, w in PLATEAU_WEIGHTS.items())
    df["rising_score"] = (rising * 100).round(1)
    df["plateau_risk"] = (risk * 100).round(1)

    known = df["ret_3m"].notna() & df["above_ma200"].notna()
    is_rising = (df["ret_3m"] > 0) & (df["above_ma200"] > 0)
    df["outlook"] = np.select(
        [~known, ~is_rising, df["plateau_risk"] >= HIGH_RISK, df["plateau_risk"] <= LOW_RISK],
        ["判定できない", "上昇していない", "頭打ち注意", "上昇継続の可能性"],
        default="様子見",
    )
    df["score"] = ((1 - PLATEAU_SHARE) * df["rising_score"] + PLATEAU_SHARE * (100 - df["plateau_risk"])).round(1)

    # --- 1か月後の見立て（スコア + 頭打ち + ニュース + 市場全体） ---
    news = news or {}
    ticker_tone = df.index.to_series().map(lambda t: (news.get("tickers", {}).get(t) or {}).get("tone"))
    market_tone = df["market"].map(lambda m: (news.get("market", {}).get(m) or {}).get("tone"))
    df["news_tone"] = ticker_tone.fillna("")
    df["market_tone"] = market_tone.fillna("")
    base = ((df["score"] - 50) / 25).clip(-1, 1)
    points = (
        base
        - np.where(df["outlook"] == "頭打ち注意", PLATEAU_PENALTY, 0.0)
        + TICKER_NEWS_WEIGHT * ticker_tone.map(TONE_VALUE).fillna(0)
        + MARKET_NEWS_WEIGHT * market_tone.map(TONE_VALUE).fillna(0)
    )
    df["direction"] = points.map(direction_label)

    # --- 1か月後の株価の目安（過去1年の値動きの大きさから計算。方向は決めつけない） ---
    for name, z in (("range", 1.0), ("wide", 1.96)):   # 約7割 / 約95%
        df[f"{name}_low"] = (df["price"] * np.exp(-z * df["vol_1m"])).round(2)
        df[f"{name}_high"] = (df["price"] * np.exp(z * df["vol_1m"])).round(2)
    df["price"] = df["price"].round(2)
    df["vol_1m"] = df["vol_1m"].round(4)
    return df.sort_values(["market", "score"], ascending=[True, False])


def main():
    news = load_news()
    mine = load_my_stocks()
    if mine:
        print(f"自分の銘柄: {', '.join(mine)}")
    rows = []
    for market, tickers in build_universe(mine).items():
        for ticker in tickers:
            print(f"取得中: {ticker}")
            rows.append({"ticker": ticker, "market": market, "mine": int(ticker in mine), **fetch_features(ticker)})
    result = build_scores(pd.DataFrame(rows).set_index("ticker"), news)

    pd.set_option("display.float_format", lambda x: f"{x:,.1f}")
    print("\n=== ランキング（市場別） ===")
    print(result[["market", "score", "outlook", "direction", "price", "range_low", "range_high"]].to_string())
    result.to_csv("screener_result.csv", encoding="utf-8-sig")
    print("\nscreener_result.csv に保存しました")


if __name__ == "__main__":
    main()
