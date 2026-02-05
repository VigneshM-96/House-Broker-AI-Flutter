from sklearn.preprocessing import LabelEncoder

def encode_location(df):
    encoder = LabelEncoder()
    df["Location"] = encoder.fit_transform(df["Location"])
    return df, encoder