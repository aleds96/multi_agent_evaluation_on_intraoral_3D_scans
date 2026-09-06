LANDMARK_PALETTE = {
    "Mesial":        [1.0, 0.0, 0.0],   #rosso
    "Distal":        [0.0, 1.0, 0.0],   #verde
    "Cusp":          [0.0, 0.0, 1.0],   #blu
    "InnerPoint":    [1.0, 1.0, 0.0],   #giallo
    "OuterPoint":    [0.0, 1.0, 1.0],   #ciano
    "FacialPoint":    [1.0, 0.0, 1.0],   #magenta
}
FDI_GROUPS = {
    "incisors":  [11,12,21,22,31,32,41,42],
    "canines":   [13,23,33,43],
    "premolars": [14,15,24,25,34,35,44,45],
    "molars":    [16,17,18,26,27,28,36,37,38,46,47,48],
    "gingiva":   [0]
}

#Reverse map: dente => gruppo
TOOTH_TO_GROUP = {t: g for g, lst in FDI_GROUPS.items() for t in lst}
TOOTH_GROUP_PALETTE = {
    "incisors":  [0.5, 0.5, 0.5],      #grigio
    "canines":   [1.0, 0.68, 0.45],    #arancione pastello
    "premolars": [0.95, 0.55, 0.75],   #rosa pastello
    "molars":    [0.45, 0.70, 1.0],    #azzurro pastello
    "gingiva":   [0.95, 0.95, 0.95],   #bianco
}

CATEGORIES = list(LANDMARK_PALETTE.keys())
print("CATEGORIES ==>", CATEGORIES)