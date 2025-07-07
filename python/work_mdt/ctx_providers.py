from contextily import providers
name_2_provider = {
    "Mapnik":providers.OpenStreetMap.Mapnik,
    "DE":providers.OpenStreetMap.DE,
    "France":providers.OpenStreetMap.France,
    "HOT":providers.OpenStreetMap.HOT,
    "Positron":providers.CartoDB.Positron,
    "Voyager":providers.CartoDB.Voyager,
    "DarkMatter":providers.CartoDB.DarkMatter,
    "WorldStreetMap":providers.Esri.WorldStreetMap,
    "WorldImagery":providers.Esri.WorldImagery,
    "WorldTopoMap":providers.Esri.WorldTopoMap,
    "WorldTerrain":providers.Esri.WorldTerrain,
    "NatGeoWorldMap":providers.Esri.NatGeoWorldMap,
    "ModisTerraTrueColorCR":providers.NASAGIBS.ModisTerraTrueColorCR,
    "ModisTerraBands367CR":providers.NASAGIBS.ModisTerraBands367CR,
    "ViirsEarthAtNight2012":providers.NASAGIBS.ViirsEarthAtNight2012
}
