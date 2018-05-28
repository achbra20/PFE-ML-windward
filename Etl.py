import  requests
import  pandas as pd
import  json
import  os

class Etl_data:

    # todo satatic methode Consumate  Api to dataframe
    @staticmethod
    def web_service_response(url):
        print("log api")
        response = requests.get(url)
        data= response.json()
        return  pd.DataFrame(data)

    #todo Static mmethode to delete and write json file in 'path' and the name is 'fileName' and the date is 'data'
    @staticmethod
    def writeToJSONFile(path, fileName, data):
        print('log file create')
        filePathNameWExt = path + '/' + fileName + '.json'
        if os.path.isfile(filePathNameWExt):
            os.remove(filePathNameWExt)
        with open(filePathNameWExt, 'w') as fp:
            json.dump(data, fp)

    #todo static methode to open json file from the path 'path' and yhe name of file is 'file'
    @staticmethod
    def open_json(file,path):
        with open(path+"/"+file+'.json') as json_data:
            data_dict = json.load(json_data)
        return pd.DataFrame(data_dict)