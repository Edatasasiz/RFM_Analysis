###############################################################
# RFM ile Müşteri Segmentasyonu (Customer Segmentation with RFM)
###############################################################

###############################################################
# GÖREV 1: Veriyi  Hazırlama ve Anlama (Data Understanding)
###############################################################

# 1. flo_data_20K.csv verisini okuyunuz. Dataframe’in kopyasını oluşturunuz.

import datetime as dt
import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


df1 = pd.read_csv("CRM_Analytics/RFM/FLO_case1/flo_data_20k.csv")
df = df1.copy()
df.head()

# 2. Veri setinde
        # a. İlk 10 gözlem,
            df.head(10)
        # b. Değişken isimleri,
            df.columns
        # c. Betimsel istatistik,
            df.describe().T
        # d. Boş değer,
            df.isnull().sum()
        # e. Değişken tipleri, incelemesi yapınız.
            df.info()

# 3. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir. Herbir müşterinin toplam
#     alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.
df["total_order"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["total_value"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]
df.head()



# 4. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
df.info()
for col in df.columns:
    if "date" in col:
        df[col] = pd.to_datetime(df[col])


# 5. Alışveriş kanallarındaki müşteri sayısının, toplam alınan ürün sayısının ve toplam harcamaların dağılımına bakınız.
df.groupby("order_channel").agg({"total_order" : "count", "total_value" : "sum"})

# 6. En fazla kazancı getiren ilk 10 müşteriyi sıralayınız.
df.groupby("master_id").agg({"total_value" : "sum"}).sort_values("total_value",ascending=False).head(10)

# 7. En fazla siparişi veren ilk 10 müşteriyi sıralayınız.
df.groupby("master_id").agg({"total_order" : "sum"}).sort_values("total_order",ascending=False).head(10)

# 8. Veri ön hazırlık sürecini fonksiyonlaştırınız.
def data_preparation(df):
    df["total_order"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
    df["total_value"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]

    for col in df.columns:
        if "date" in col:
            df[col] = pd.to_datetime(df[col])

        return df



###############################################################
# GÖREV 2: RFM Metriklerinin Hesaplanması
###############################################################

# Adım 1: Recency, Frequency ve Monetary tanımlarını yapınız.

# Recency: Müşterinin son sipariş tarihinden bu yana kaç gün geçti?
# Frequency: Sipariş sıklığını ifade ediyor, Müşterinin toplam sipariş sayısıdır.
# Monetary: Müşteri şirkete toplamda ne kadar para ödedi?

# Adım 2: Müşteri özelinde Recency, Frequency ve Monetary metriklerini hesaplayınız.
# Adım 3: Hesapladığınız metrikleri rfm isimli bir değişkene atayınız.

df["last_order_date"].max()     # 2021-05-30

today_date = dt.datetime(2021, 6, 1)    # analiz 2 gün sonra

rfm = df.groupby('master_id').agg({'last_order_date': lambda last_order_date: (today_date - last_order_date.max()).days,
                                   'total_order': lambda total_order: total_order.nunique(),
                                   'total_value': lambda total_value: total_value.sum()})
rfm.head()

# Adım 4: Oluşturduğunuz metriklerin isimlerini recency, frequency ve monetary olarak değiştiriniz.
rfm.columns = ['recency', 'frequency', 'monetary']



###############################################################
# GÖREV 3: RF ve RFM Skorlarının Hesaplanması (Calculating RF and RFM Scores)
###############################################################

# Adım 1: Recency, Frequency ve Monetary metriklerini qcut yardımı ile 1-5 arasında skorlara çeviriniz.
# Adım 2: Bu skorları recency_score, frequency_score ve monetary_score olarak kaydediniz.

rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])
rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])


# Adım 3: recency_score ve frequency_score’u tek bir değişken olarak ifade ediniz ve RF_SCORE olarak kaydediniz.

rfm["RF_SCORE"] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str))



###############################################################
# GÖREV 4: RF Skorlarının Segment Olarak Tanımlanması
###############################################################

# Adım 1: Oluşturulan RF skorları için segment tanımlamaları yapınız.
# Adım 2: Aşağıdaki seg_map yardımı ile skorları segmentlere çeviriniz.

seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm['segment'] = rfm['RF_SCORE'].replace(seg_map, regex=True)
rfm.head()



###############################################################
# GÖREV 5: Aksiyon zamanı!
###############################################################

# 1. Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz.

rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])


# 2. RFM analizi yardımı ile 2 case için ilgili profildeki müşterileri bulunuz ve müşteri id'lerini csv ye kaydediniz.

# a. FLO bünyesine yeni bir kadın ayakkabı markası dahil ediyor. Dahil ettiği markanın ürün fiyatları genel müşteri
# tercihlerinin üstünde. Bu nedenle markanın tanıtımı ve ürün satışları için ilgilenecek profildeki müşterilerle özel olarak
# iletişime geçmek isteniliyor. Sadık müşterilerinden(champions, loyal_customers) ve kadın kategorisinden alışveriş
# yapan kişiler özel olarak iletişim kurulacak müşteriler. Bu müşterilerin id numaralarını csv dosyasına kaydediniz.

target_id = rfm[rfm["segment"].isin(["champions", "loyal_customers"])].index
target_customers = df[(df["master_id"].isin(target_id)) &
              (df["interested_in_categories_12"].str.contains("KADIN"))]["master_id"]
target_customers.head()
target_customers.shape      # (1820,)    2487

target_customers.to_csv("woman_customers.csv")

# b. Erkek ve Çocuk ürünlerinde %40'a yakın indirim planlanmaktadır. Bu indirimle ilgili kategorilerle ilgilenen geçmişte
# iyi müşteri olan ama uzun süredir alışveriş yapmayan kaybedilmemesi gereken müşteriler, uykuda olanlar ve yeni
# gelen müşteriler özel olarak hedef alınmak isteniyor. Uygun profildeki müşterilerin id'lerini csv dosyasına kaydediniz.


target_id = rfm[rfm["segment"].isin(["cantloose", "hibernating", "new_customers"])].index
target_customers = df[(df["master_id"].isin(target_id)) &
                      ((df["interested_in_categories_12"].str.contains("ERKEK")) |
                      (df["interested_in_categories_12"].str.contains("ÇOCUK")))]["master_id"]
target_customers.head()
target_customers.shape      # (1420,)     2771

target_customers.to_csv("man_and_child_customers.csv")
