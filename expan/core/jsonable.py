import json

class Jsonable(object):
    """ the properties of the class should be:
     - either int, float, bool, or str
     - or     lists, sets, tuples made of them
     - or     dicts made of them
     - or     other instances of Jsonable
     - or     any combinations of above
     see function jsonify() below for the logic of recursive json building
     """
    def to_json(self):
        return json.dumps(jsonify(self))

def jsonify(x):
    def isin(x, types):
        return any(isinstance(x, t) for t in types)

    if isin(x, [bool, int, float, str]):
        return x
    elif isin(x, [list, set, tuple]):
        return [jsonify(y) for y in x]
    elif isin(x, [dict]):
        return {jsonify(k): jsonify(v) for (k, v) in x.items()}
    elif isinstance(x, Jsonable):
        return jsonify(x.__dict__)
    else:
        print("don't know how to jsonify", x)
        return None
