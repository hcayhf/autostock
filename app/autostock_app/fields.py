"""検索可能なカラムの単一の定義元。

DB スキーマ・指標計算・スクリーナ・API の /meta・Web UI が
すべてこの定義を参照する。カラムを増やすときはここに 1 行足す。

単位について:
    数値はすべて「人が検索窓に打ち込む単位」でそのまま格納する。
    比率は % (0.0532 ではなく 5.32)、時価総額は億円、売買代金は百万円。
    UI 側で単位変換をしないので、SQL を直接叩いても直感どおりの値が出る。
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field


@dataclass(frozen=True)
class Field:
    key: str
    label: str
    group: str  # identity | price | technical | fundamental
    dtype: str  # text | real
    unit: str = ""
    desc: str = ""
    #: 検索フォームで既定表示するか（多すぎるので主要なものだけ true）
    primary: bool = False
    #: 負値を取りうる指標か。True のときだけ表示に符号を付け、色分けする。
    #: 配当利回りやボラティリティのように常に正の指標に "+" が付くのを避ける。
    signed: bool = False

    @property
    def numeric(self) -> bool:
        return self.dtype == "real"


# --- 銘柄属性 (stocks 由来・文字列) -----------------------------------------

IDENTITY_FIELDS: tuple[Field, ...] = (
    Field("code", "コード", "identity", "text", desc="4桁の証券コード"),
    Field("name", "銘柄名", "identity", "text"),
    Field("market", "市場区分", "identity", "text", primary=True),
    Field("sector33", "業種(33)", "identity", "text", primary=True),
    Field("sector17", "業種(17)", "identity", "text"),
    Field("scale", "規模区分", "identity", "text", desc="TOPIX Core30 / Large70 など"),
)

#: カテゴリとして IN 検索できる文字列カラム。
CATEGORICAL_KEYS: tuple[str, ...] = ("market", "sector33", "sector17", "scale")

#: 部分一致検索できる文字列カラム。
TEXT_SEARCH_KEYS: tuple[str, ...] = ("code", "name")


# --- 株価そのもの ------------------------------------------------------------

PRICE_FIELDS: tuple[Field, ...] = (
    Field("close", "終値", "price", "real", "円", primary=True),
    Field("volume", "出来高", "price", "real", "株"),
    Field("turnover_20d", "売買代金(20日平均)", "price", "real", "百万円",
          desc="流動性フィルタ用。終値×出来高の20日平均", primary=True),
)


# --- テクニカル指標 ----------------------------------------------------------

TECHNICAL_FIELDS: tuple[Field, ...] = (
    # モメンタム
    Field("ret_5d", "騰落率(5日)", "technical", "real", "%", signed=True),
    Field("ret_20d", "騰落率(20日)", "technical", "real", "%", primary=True, signed=True),
    Field("ret_60d", "騰落率(60日)", "technical", "real", "%", signed=True),
    Field("ret_120d", "騰落率(120日)", "technical", "real", "%", signed=True),
    Field("ret_250d", "騰落率(250日≒1年)", "technical", "real", "%", primary=True, signed=True),
    # トレンド
    Field("sma_dev_20", "20日線乖離率", "technical", "real", "%", signed=True),
    Field("sma_dev_60", "60日線乖離率", "technical", "real", "%", signed=True),
    Field("sma_dev_200", "200日線乖離率", "technical", "real", "%",
          desc="中長期のトレンド判定。プラスなら長期上昇トレンド", primary=True, signed=True),
    # 位置
    Field("high_60d_pct", "60日高値からの位置", "technical", "real", "%",
          desc="0%が60日高値。マイナスほど高値から離れている", signed=True),
    Field("low_60d_pct", "60日安値からの位置", "technical", "real", "%",
          desc="0%が60日安値。プラスほど安値から離れている", signed=True),
    Field("high_250d_pct", "52週高値からの位置", "technical", "real", "%",
          desc="0%が年初来高値圏", primary=True, signed=True),
    Field("low_250d_pct", "52週安値からの位置", "technical", "real", "%", primary=True, signed=True),
    Field("max_drawdown_250d", "最大ドローダウン(52週)", "technical", "real", "%",
          desc="直近52週で経験した最大の下落幅。0以下の値で、大きいほど値持ちが良い", signed=True),
    # オシレーター
    Field("rsi_14", "RSI(14)", "technical", "real", "", desc="0〜100。30以下で売られすぎ", primary=True),
    Field("macd_hist", "MACDヒストグラム", "technical", "real", "%", desc="株価で正規化済み", signed=True),
    Field("bb_pct_b", "ボリンジャー%B", "technical", "real", "",
          desc="0〜100に正規化。0で下限バンド、100で上限バンド"),
    Field("bb_bandwidth", "ボリンジャーバンド幅", "technical", "real", "%"),
    # リスク
    Field("volatility_20d", "ボラティリティ(20日)", "technical", "real", "%", desc="日次騰落率の標準偏差"),
    Field("volatility_60d", "ボラティリティ(60日)", "technical", "real", "%", primary=True),
    # 出来高
    Field("volume_ratio_5_20", "出来高比(5日/20日)", "technical", "real", "%",
          desc="プラスなら直近で出来高が膨らんでいる", signed=True),
)


# --- ファンダメンタルズ ------------------------------------------------------

FUNDAMENTAL_FIELDS: tuple[Field, ...] = (
    Field("market_cap", "時価総額", "fundamental", "real", "億円", primary=True),
    Field("per", "PER", "fundamental", "real", "倍", desc="実績PER。赤字企業は欠損", primary=True),
    Field("pbr", "PBR", "fundamental", "real", "倍", primary=True),
    Field("dividend_yield", "配当利回り", "fundamental", "real", "%", primary=True),
    Field("roe", "ROE", "fundamental", "real", "%", primary=True, signed=True),
    Field("equity_ratio", "自己資本比率", "fundamental", "real", "%"),
    Field("eps", "EPS", "fundamental", "real", "円", signed=True),
    Field("bps", "BPS", "fundamental", "real", "円"),
)


ALL_FIELDS: tuple[Field, ...] = (
    IDENTITY_FIELDS + PRICE_FIELDS + TECHNICAL_FIELDS + FUNDAMENTAL_FIELDS
)

FIELDS_BY_KEY: dict[str, Field] = {f.key: f for f in ALL_FIELDS}

#: indicators テーブルに入る列（= compute_indicators が返す列）。
INDICATOR_KEYS: tuple[str, ...] = tuple(
    f.key for f in PRICE_FIELDS + TECHNICAL_FIELDS
)

#: fundamentals テーブルに入る列。
FUNDAMENTAL_KEYS: tuple[str, ...] = tuple(f.key for f in FUNDAMENTAL_FIELDS)

#: 数値レンジ検索・ソートに使えるカラム。
NUMERIC_KEYS: tuple[str, ...] = tuple(f.key for f in ALL_FIELDS if f.numeric)

GROUP_LABELS = {
    "identity": "銘柄属性",
    "price": "株価・流動性",
    "technical": "テクニカル",
    "fundamental": "ファンダメンタルズ",
}


def field_or_raise(key: str) -> Field:
    """未知のカラム名を弾く。SQL 組み立て前の検証に使う。"""
    try:
        return FIELDS_BY_KEY[key]
    except KeyError:
        raise ValueError(f"unknown field: {key!r}") from None
