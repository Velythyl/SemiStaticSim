def xyztuple2dict(tup):
    return {'x': tup[0], 'y': tup[1], 'z': tup[2]}

def dict2xyztuple(dic):
    return (dic['x'], dic['y'], dic['z'])

def xyztuple_precision(tup):
    return (round(tup[0], 3), round(tup[1],3), round(tup[2],3))