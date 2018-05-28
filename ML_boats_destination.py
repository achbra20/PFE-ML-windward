from Etl import Etl_data
from Ml_destination import Etl_Model_Ml_Destination
from ML_boats import Recommendation_boats
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error


class Etl_boats_destination:

    @staticmethod
    def tremestre(month):
        if int(month) < 3:
            i = 1
        elif int(month)  < 6:
            i = 2
        elif int(month)  < 9:
            i = 3
        else:
            i = 4
        return i

class Create_data_Ml_boat_destination:

    def __init__(self ,api_data_ww ,api_data_crm,api_data_crm2 ,api_boats_id_ww ,api_boats_generic,api_all_destination):
        self.data_ww         = Etl_data.web_service_response(api_data_ww)
        self.data_crm        = Etl_data.web_service_response(api_data_crm)
        self.data_crm2       = Etl_data.web_service_response(api_data_crm2)
        self.boats_id_ww     = Etl_data.web_service_response(api_boats_id_ww)
        self.boats_generic   = Etl_data.web_service_response(api_boats_generic)
        self.all_destination = Etl_data.web_service_response(api_all_destination)

    def tremestre(self ,df_boat_destination):
        i = 0
        tremestre = []
        df_boat_destination['month'] = df_boat_destination.month.astype(int)
        for index, request in df_boat_destination.iterrows():

            if request['month'] < 3:
                i = 1
            elif request['month'] < 6:
                i = 2
            elif request['month'] < 9:
                i = 3
            else:
                i = 4
            tremestre.append(
                {'id_gen': request['id_gen'], 'id_location': request['id'], 'tremestre': i, 'year': request['year']})
        return pd.DataFrame(tremestre)

    def rank(self ,lists):
        Statistique = []
        unique = list(lists.id_gen.unique())
        for i in range(0, len(unique)):
            one_dest = pd.DataFrame(lists[lists["id_gen"] == unique[i]])
            Statistique2 = []
            for index, request in one_dest.iterrows():
                notfound = True
                for j in range(0, len(Statistique2)):
                    if Statistique2[j]["year"] == request['year'] and Statistique2[j]["tremestre"] == request[
                        "tremestre"] and Statistique2[j]["id_location"] == request["id_location"]:
                        Statistique2[j]["counts"] = Statistique2[j]["counts"] + 1
                        notfound = False
                        break
                if len(Statistique2) == 0 or notfound == True:
                    Statistique2.append({"id_gen": request["id_gen"], "counts": 1, 'tremestre': request["tremestre"],
                                         'year': request['year'], "id_location": request["id_location"]})
            Statistique = Statistique + Statistique2
        Statistique = sorted(Statistique, key=lambda k: k["counts"], reverse=True)
        return pd.DataFrame(Statistique)

    def boat_found_id(self,boat_data, boat_id):
        boats_id = []
        for index, boat in boat_data.iterrows():
            for index, boats in boat_id.iterrows():
                ch = boats["boat_brand"] + " " + boats["boat_model"] + " " + boats["shipyard_name"]
                ch2 = boats["shipyard_name"] + " " + boats["boat_brand"] + " " + boats["boat_model"]
                print(boat["boat"])
                if ch.upper() in boat["boat"].upper() or boat["boat"].upper() in ch.upper():
                    boats_id.append(
                        {"id_gen": boats["generic"], "name": ch, 'boat': boat["boat"]})
                    break
                elif ch2.upper() in boat["boat"].upper():
                    boats_id.append(
                        { "id_gen": boats["generic"], "name": ch, "boat": boat["boat"]})
                    break
        return boats_id

    def found_id_for_all_boat(self,df_boats):
        boats_req = df_boats.drop_duplicates(subset="boat", keep="first")
        uniqueid = self.boats_id_ww.drop_duplicates(subset="generic", keep="first")
        model_test_1 = self.boat_found_id(boats_req, uniqueid)
        df_boats_final = pd.merge(pd.DataFrame(df_boats), pd.DataFrame(model_test_1), on='boat', how='inner')
        return df_boats_final

    def speration(self,lists, champ, sp):
        Statistique = []
        b = True
        for index, request in lists.iterrows():
            if request[champ] != "":
                x = request[champ].split(sp)
                for i in range(len(x)):
                    Statistique.append({"destination": x[i], "id_gen": request["id_gen"], "year": request["year"],
                                        "month": request["month"], "day": request["day"]})
        return pd.DataFrame(Statistique)

    def found_destination_id(self,list_p, champ_p, champ_id, list_f, champ_f):
        rank = []
        for index, request in list_f.iterrows():
            for index, req in list_p.iterrows():
                if req[champ_p].upper() in request[champ_f].upper() and req[champ_p] != "":
                    rank.append(
                        {'label': req[champ_p], 'destination': request[champ_f], 'id_par': req["parent_location"],
                         "type": req["location_type"], 'id': req[champ_id]})
                    break
        return rank

    def found_all_request_id_destination(self,lists):
        final = []
        all_destination_lf = pd.DataFrame(self.all_destination[self.all_destination["ref_language"] == "1"])
        for index, request in lists.iterrows():
            if request["type"] == "country" or request["type"] == "ocean":
                final.append(
                    {'id': request['id'], "year": request["year"], 'month': request["month"], 'day': request["day"],
                     "id_gen": request["id_gen"]})
            else:
                one_dest = pd.DataFrame(all_destination_lf[all_destination_lf["location_id"] == request['id_par']])
                for index, req in one_dest.iterrows():
                    if req["location_type"] == "zone":
                        final.append({'id': req['parent_location'], "year": request["year"], 'month': request["month"],'day': request["day"], "id_gen": request["id_gen"]})
                    else:
                        final.append({'id': req["location_id"], "year": request["year"], 'month': request["month"],'day': request["day"], "id_gen": request["id_gen"]})
        return final

    def found_destination(self,Model_boats_destination):
        Model_boats_destination    = self.speration(Model_boats_destination, "request_destination", "/")
        Model_boats_destination    = self.speration(Model_boats_destination, "destination", ",")
        uniqueid = Model_boats_destination.drop_duplicates(subset="destination", keep="first")
        Model_boats_destination_id = self.found_destination_id(self.all_destination, "location_name", "location_id", uniqueid, "destination")
        model_destination_boats    = pd.merge(pd.DataFrame(Model_boats_destination), pd.DataFrame(Model_boats_destination_id), on='destination', how='inner')
        model_destination_boats    = self.found_all_request_id_destination(model_destination_boats)
        return  pd.DataFrame(model_destination_boats)

    def create_data_ML(self,root_model):

        df_boats = self.data_ww.append(self.data_crm)
        df_boats = df_boats.append(self.data_crm2)

        print(df_boats.shape )
        print(df_boats.info())

        df_boats = df_boats[df_boats["boat"] != ""]
        df_boats = df_boats[df_boats["month"] != "0"]
        df_boats = df_boats[df_boats["request_destination"] != ""]

        df_boats = self.found_id_for_all_boat(df_boats)
        df_boats = self.found_destination(df_boats)

        Etl_data.writeToJSONFile(root_model,"data_boat_destination_statestique",df_boats.to_dict('records'))

        df_boats_final    = self.tremestre(df_boats)
        df_boats_final    = self.rank(df_boats_final)

        req_generic       = self.boats_generic.loc[:,['boat_id','loa', 'beam', 'fuel']]
        req_generic       = req_generic.rename(columns={'boat_id': 'id_gen'})
        Model_boats_final = pd.merge(df_boats_final, req_generic, on='id_gen', how='inner')

        Model_boats_final['loa']         = Model_boats_final.loa.astype(float)
        Model_boats_final['beam']        = Model_boats_final.beam.astype(float)
        Model_boats_final['fuel']        = Model_boats_final.fuel.astype(int)
        Model_boats_final['id_location'] = Model_boats_final.id_location.astype(int)
        Model_boats_final['id_gen']      = Model_boats_final.id_gen.astype(int)
        Model_boats_final['year']        = Model_boats_final.year.astype(int)
        Model_boats_final['tremestre']   = Model_boats_final.tremestre.astype(int)

        Etl_data.writeToJSONFile(root_model, "data_final_boats_destination", Model_boats_final.to_dict('records'))


class Ml_boat_destination:

    def __init__(self,root_model,name_model):
        self.df_boats_destination = Etl_data.open_json("data_final_boats_destination",root_model)
        self.training_model(self.df_boats_destination,root_model,name_model)

    def training_model(self,df_boats_destination,root_model,name_model):

        print("begin of training")

        feature_col_names = ['year', 'tremestre', 'id_gen', 'id_location', 'fuel', 'loa', 'beam']
        predicted_class_names = ['counts']

        X = df_boats_destination[feature_col_names].values
        y = df_boats_destination[predicted_class_names].values
        split_test_size = 0.25

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_test_size, random_state=23)

        model = RandomForestRegressor(n_estimators=100)
        model.fit(X_train, y_train)

        joblib.dump(model,root_model+"/"+name_model)

        #####################################
        score_train = model.score(X_train, y_train)
        score_test = model.score(X_test, y_test)
        print(score_train, score_test)

        x_pred      = model.predict(X_train)
        error_train = mean_squared_error(y_train, x_pred)
        print(error_train)

        y_pred = model.predict(X_test)
        error_test = mean_squared_error(y_test, y_pred)
        print(error_test)
        ####################
        info = []
        info.append({"scoretrai":score_train,"errortr":error_train,"scoretes":score_test,"errorte":error_test})
        Etl_data.writeToJSONFile(root_model,"score_ml_boats_destination",info)
        ####################
        #predection = Predection_distination(root_name+"/"+name_model)
        #predection.restart(root_name+"/"+name_model)
        #####################################
        print("training done !")


def singleton(theClass):
    """ decorator for a class to make a singleton out of it """
    classInstances = {}
    def getInstance(*args, **kwargs):
        """ creating or just return the one and only class instance.
            The singleton depends on the parameters used in __init__ """
        key = (theClass, args, str(kwargs))
        if key not in classInstances:
            classInstances[key] = theClass(*args, **kwargs)
        return classInstances[key]
    return getInstance

@singleton
class Ml_predection_boat_destination:

    def __init__(self,name):
        self.model = joblib.load(name)

    def restart(self,name):
        self.model = joblib.load(name)

    def predict(self,year,tremestre,id_generic,id_location,fuel,loa,beam):
        X_new = [[year,tremestre,id_generic,id_location,fuel,loa,beam]]
        predection   =  self.model.predict(X_new)
        return predection


class Recommendation_boats_destenation:

    def __init__(self,api_boats_generic,api_all_destination,root_model,name_model):

        self.boats_generic   = Etl_data.web_service_response(api_boats_generic)
        self.all_destination = Etl_data.web_service_response(api_all_destination)
        self.root_model      = root_model
        self.name_model      = name_model

    def found_boats(self,country):

        recommandation_boat = Recommendation_boats(self.root_model)
        boat_recommended    = recommandation_boat.recommended_boats(country,30)
        return boat_recommended

    def found_parent_location(self, id_location):

        df_destination = self.all_destination[self.all_destination["location_id"] == id_location]
        for index, destination in df_destination.iterrows():
            if destination["location_type"] == "country" or destination["location_type"] == "ocean":
                return destination["location_id"]
            elif destination["location_type"] == "zone":
                return destination["parent_location"]
            else:
               return self.found_parent_location(destination["parent_location"])

    def found_tremestre(self,date):

        date_in = date.split("-")
        i = 0
        if int(date_in[1]) <= 3:
            i = 1
        elif int(date_in[1]) <= 6:
            i = 2
        elif int(date_in[1]) <= 9:
            i = 3
        else:
            i = 4

        return i,date_in[2]

    def  scored_boat(self,country,id_location,date):

        boat_recommended  = self.found_boats(country)
        boat_recommended  = pd.DataFrame(boat_recommended).rename(columns={'id_generic': 'id_gen'})
        req_generic       = self.boats_generic.loc[:,['boat_id','loa', 'beam', 'fuel']]
        req_generic       = req_generic.rename(columns={'boat_id': 'id_gen'})
        Model_boats_final = pd.merge(boat_recommended, req_generic, on='id_gen', how='inner')
        parent_location   = self.found_parent_location(id_location)
        tremestre,year    = self.found_tremestre(date)
        boats_prediction  = Ml_predection_boat_destination(self.root_model+"/"+self.name_model)
        final_recommend   = []

        for index,boat in Model_boats_final.iterrows():
           prediction = boats_prediction.predict(int(year),tremestre,int(boat['id_gen']),int(parent_location),boat['fuel'],boat['loa'],boat['beam'])
           final_recommend.append({"date": date, "score": round(float(prediction[0]),2) ,'boat':boat["name"] ,'id':boat["id_gen"] })

        final_recommend= sorted(final_recommend, key=lambda k: k["score"], reverse=True)

        return final_recommend






