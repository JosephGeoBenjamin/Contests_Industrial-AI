"""
Script for parsing the data labels provided by BDD
"""

import json

import xmltodict
import glob

def BDD_extract_2dbbox(json_file,
                out_json = "BDD_objectdetect_2dbbox_train.json" ):
    '''
    json_file - format:
    list({ "name": "image_id.jpg",
            "labels": [{
                "category": "object_class",
                "box2d": { "x1": float, "y1": float, "x2": float, "y2": float}
            },]
            "attributes": { "weather": "overcast", "scene": "city street", "timeofday": "daytime"},
        })
    '''

    json_data = json.load(open(json_file))

    final_data = []
    for i, dic in enumerate(json_data):
        data = {}
        data["name"] = dic["name"]
        data["attributes"] = dic["attributes"]
        data['size'] ={ "height":720 , "width":1280 }
        data["labels"] = []
        for j, jdic in enumerate(dic["labels"]):
            jdata = {}
            try:
                jdata["category"] = jdic["category"]
                jdata["box2d"] = jdic["box2d"]
                data["labels"].append(jdata)
            except:
                pass
        final_data.append(data)

    print(len(final_data))
    with open(out_json, "w") as f:
        json.dump(final_data, f, indent=2)


def PascalVOC_extract_2dbbox(xmls_folder_path,
                out_json = "pascal_voc_2dbbox_train.json" ):
    '''
    json_file - format:
    list({ "name": "image_id.jpg", "height": int, "width": int,
            "labels": [{
                "category": "object_class",
                "box2d": { "x1": float, "y1": float, "x2": float, "y2": float}
            },]
        })
    '''
    def _2int(x):
        try:
            return int(x)
        except:
            return int(float(x))

    final_data = []
    for a in glob.glob(xmls_folder_path+"/*.xml"):
        y = xmltodict.parse(open(a).read())
        ddict = {}
        ddict['name'] = y["annotation"]["filename"]
        ddict['size'] ={ "height":_2int(y["annotation"]["size"]["height"]),
                         "width": _2int(y["annotation"]["size"]["width"]) }
        x = y["annotation"]["object"]
        if not isinstance(x,list): x = [x]

        ddict['labels'] = []
        for i in x:
            ldict = {}
            ldict['category'] = i["name"]
            ldict['box2d'] = { 'x1': _2int(i["bndbox"]["xmin"]), 'y1': _2int(i["bndbox"]["ymin"]),
                               'x2': _2int(i["bndbox"]["xmax"]), 'y2': _2int(i["bndbox"]["ymax"]),
                            }
            ddict['labels'].append(ldict)

        final_data.append(ddict)

    print(len(final_data))
    with open(out_json, "w") as f:
        json.dump(final_data, f, indent=2)

if __name__ == "__main__":
    pass