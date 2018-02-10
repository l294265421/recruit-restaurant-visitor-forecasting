from data.raw_data import data

air_genre_name = data['air_store_info']['air_genre_name']
print(len(air_genre_name.unique()))
for name in air_genre_name.unique():
    print(name)
print(max([len(name.split()) for name in air_genre_name.unique()]))

air_area_name = data['air_store_info']['air_area_name']
print(len(air_area_name.unique()))
for name in air_area_name.unique():
    print(name)
print(max([len(name.split()) for name in air_area_name.unique()]))

latitude = data['air_store_info']['latitude']
print(max(latitude))
print(min(latitude))

longitude = data['air_store_info']['longitude']
print(max(longitude))
print(min(longitude))