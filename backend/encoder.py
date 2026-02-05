location_map = {}

def fit_encoder(df):
    global location_map
    locations = df['location'].unique()
    location_map = {loc: i for i, loc in enumerate(locations)}
    return location_map

def encode_location(loc):
    return location_map.get(loc, -1)  # -1 for unknown locations

def get_location_map():
    return location_map