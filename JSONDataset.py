import json
import os

class JSONDataset:
    def __init__(self, directory):
        self.directory = directory
        self.files = os.listdir(directory)

    def posts(self, constraints={}):
        for filename in self.files:
            _json = json.load(open("%s/%s" % (self.directory, filename)))
            if {} == constraints:
                yield _json
            else:
                constraints_met = True
                for key, constraint in constraints.iteritems():
                    if not constraint(_json[key]):
                        constraints_met = False
                        break
                if constraints_met:
                    yield _json


