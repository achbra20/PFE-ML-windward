from Ml_destination import Recommendation_distnation,Predection_distination
from  Etl import Etl_data
import datetime
import pandas as pd

from country_list import countries_for_language

import base64
import os
import io

class Statestique_all_destination:

    #all code country
    def all_countries(self):
        countries = pd.DataFrame(countries_for_language('en'))
        return countries.to_dict("record")

    #todo scored the destination parent for index page
    def scored_day(self,name_model, root_model,api_all_destination):
        all_destination = Etl_data.web_service_response(api_all_destination)
        predection = Predection_distination(root_model+"/"+name_model)
        date = datetime.datetime.now()
        date = date.strftime('%Y/%m/%d')
        destination = pd.DataFrame(all_destination[all_destination["ref_language"] == "1"])

        rank = []
        date_in = date.split("/")
        for index, request in destination.iterrows():
            if request["location_type"] == "country" or request["location_type"] == "ocean":
                predicit = predection.predict(date_in[0], date_in[1], date_in[2], request["location_id"])
                rank.append({"date": date_in[0]+"/"+date_in[1]+"/"+date_in[2], "score": predicit[0],'destination': request["location_name"], 'id': request["location_id"]})

        return sorted(rank, key=lambda k: k["score"], reverse=True)[:20]

    #todo write the score of ML in index page
    def score_ml_destination(self,root_model):
        score = Etl_data.open_json("score_ml_destination",root_model)
        for index ,sc in score.iterrows():
            return sc["scoretrai"],sc["scoretes"],sc["errortr"],sc["errorte"]

    #todo data for the graph in the index page
    def courbe_destination(self,root_model,location = 21):
        destination          = Etl_data.open_json("data_Ml_destination",root_model)
        destination          = destination[destination["id"] == location]
        destination          = self.rank(destination)
        destination          = pd.DataFrame(destination)
        df_destination_choix = self.reg_time(destination)
        return df_destination_choix.to_dict('records')

    #todo found the list of destination existe for the selected option
    def found_tuple(self,root_model,api_all_destination):
        all_destination      = Etl_data.web_service_response(api_all_destination)
        destination = Etl_data.open_json("data_Ml_destination", root_model)
        all_destination = all_destination[all_destination["ref_language"] == "1"]
        unique = list(destination.id.unique())
        id_destination_f =[]
        for i in range(0,len(unique)):
            id_destination = all_destination[all_destination["location_id"] == unique[i]]
            for index, des in id_destination.iterrows():
                id_destination_f.append({"id":des["location_id"],"destination":des["location_name"]})

        return sorted(id_destination_f, key=lambda k: k["destination"], reverse=False)

    #todo called in courbe_destination to add date
    def reg_time(self,df_destination_choix):
        df_destination_choix["dates"] = pd.DatetimeIndex(data=df_destination_choix.dates)
        df_destination_choix = df_destination_choix.sort_values(by='dates')
        df_destination_choix["dates"] = df_destination_choix["dates"].dt.strftime('%Y/%m')
        return df_destination_choix

    #todo ranked the destination from json by month
    def rank(self,lists):
        Statistique = []
        unique = list(lists.id.unique())
        for i in range(0, len(unique)):
            one_dest = pd.DataFrame(lists[lists["id"] == unique[i]])
            Statistique = []
            for index, request in one_dest.iterrows():
                notfound = True
                for j in range(0, len(Statistique)):
                    if Statistique[j]["dates"] == request["month"]+"/"+request["year"]:
                        Statistique[j]["score"] = Statistique[j]["score"] + request["counts"]
                        notfound = False
                        break
                if len(Statistique) == 0 or notfound == True:
                    Statistique.append({ "dates":request["month"]+"/"+request["year"],"score": request["counts"]})
        return Statistique

    @staticmethod
    def top_destination_request(root_model,api_all_destination):
        df_destination  =  Etl_data.open_json("data_Ml_destination",root_model)
        all_destination =  Etl_data.web_service_response(api_all_destination)
        all_destination = all_destination[all_destination["ref_language"] == "1"]
        destination = list(df_destination.id.unique())
        df_score = []
        for i in range(0,len(destination)):
            one_destination = pd.DataFrame(df_destination[df_destination["id"] == destination[i]])
            score = 0
            for index , des in one_destination.iterrows():
                score = score + des["counts"]
            df_score.append({"location_id":destination[i],"score":score})

        df_score = pd.DataFrame(sorted(df_score, key=lambda k: k["score"], reverse=True))
        df_destination = pd.merge(df_score, all_destination, on='location_id', how='inner')
        return df_destination.to_dict('records')

class Statestique_bosts:

    #todo Number of boat bay type rent all year in this month
    def statestique_type_boat(self,api_ww,apicrm,country):
        req_ww    = Etl_data.web_service_response(api_ww)
        req_crm   = Etl_data.web_service_response(apicrm)
        req_ww    = req_ww[req_ww["country"] == country.upper()]
        req_crm   = req_crm[req_crm["pays"]  == country.upper()]
        boats_crm = self.boats_clean_type(req_crm,"type_bateau","pays",["3","1","2"])
        boats     = self.boats_clean_type(req_ww, "boat_type", "country", ["Motoryacht", "Monohull", "Catamaran"])
        boats     =  self.somme_boats(boats,boats_crm)
        return boats

    #todo add the count of crm and ww
    def somme_boats(self,boats,boats_crm):
        boats[0]["value"] = boats[0]["value"]+ boats_crm[0]["value"]
        boats[1]["value"] = boats[1]["value"]+ boats_crm[1]["value"]
        boats[2]["value"] = boats[2]["value"]+ boats_crm[2]["value"]

        return boats

    #todo calculate the number of boats by taype
    def boats_clean_type(self,dataframe,label_type,label_country,lists):
        df_boats = []
        type_mot  = 0
        type_mo   = 0
        type_cat  = 0
        for index, req in dataframe.iterrows():
            if req[label_type] == lists[0]:
                type_mot = type_mot + 1
            if req[label_type] == lists[1]:
                type_mo = type_mo + 1
            if req[label_type] == lists[2]:
                type_cat = type_cat + 1
        df_boats.append({"label": "Catamaran", "value":type_cat})
        df_boats.append({"label": "Monohull", "value": type_mo})
        df_boats.append({"label": "Motoryacht", "value": type_mot})
        print(df_boats)
        return df_boats

    @staticmethod
    def all_boats(api_all_boat):
        boats = Etl_data.web_service_response(api_all_boat)
        boats = boats.drop_duplicates(subset="generic", keep="first")
        boats_id = []
        for index, boat in boats.iterrows():
            ch = boat["boat_brand"] + " " + boat["boat_model"] + " " + boat["shipyard_name"]
            boats_id.append({ "id_gen": boat["generic"], "name": ch})

        boats_id = sorted(boats_id, key=lambda k: k["name"], reverse=False)

        return boats_id

    @staticmethod
    def all_boats_destination(root_model,api_all_boat,api_all_destination):

        boats_destination = Etl_data.open_json("data_boat_destination_statestique", root_model)
        all_boat         = Statestique_bosts.all_boats(api_all_boat)
        all_boat         = pd.DataFrame(all_boat)
        df_boats         = boats_destination.drop_duplicates(subset="id_gen", keep="first")
        df_boats         = pd.merge(df_boats, all_boat, on='id_gen', how='inner')
        df_boats         = df_boats.to_dict('records')
        df_boats         = sorted(df_boats, key=lambda k: k["name"], reverse=False)
        all_destination  = Etl_data.web_service_response(api_all_destination)
        df_destenation   = boats_destination.drop_duplicates(subset="id", keep="first")
        all_destination  = all_destination[all_destination["ref_language"] == "1"]
        all_destination  = all_destination.rename(columns={'location_id': 'id'})
        df_destenation   = pd.merge(df_destenation, all_destination, on='id', how='inner')
        df_destenation   = df_destenation.to_dict('records')
        df_destenation   = sorted(df_destenation, key=lambda k: k["location_name"], reverse=False)

        return df_destenation,df_boats

    def rank_boat(self,lists):
        Statistique = []
        for index, request in lists.iterrows():
            notfound = True
            for j in range(0, len(Statistique)):
                if Statistique[j]["dates"] == request["month"]+"/"+request["year"]:
                    Statistique[j]["score"] = Statistique[j]["score"] + 1
                    notfound = False
                    break
            if len(Statistique) == 0 or notfound == True:
                Statistique.append({"dates":request["month"]+"/"+request["year"],"score": 1})
        return Statistique

    def satestique_boat_generique(self,id_generic,root_model):

        boats = Etl_data.open_json("data_boat_destination_statestique",root_model)
        df_boats = boats[boats["id_gen"] == id_generic]
        if len(df_boats) < 1:
            return {"dates":"","score": 0}
        df_boats = self.rank_boat(df_boats)
        df_boats = pd.DataFrame(df_boats)
        df_boats["dates"] = pd.DatetimeIndex(data=df_boats.dates)
        df_boats          = df_boats.sort_values(by='dates')
        df_boats["dates"] = df_boats["dates"].dt.strftime('%Y/%m')

        return df_boats.to_dict('records')

    def satestique_boat_generique_destination(self,id_generic,id_location,root_model):

        boats = Etl_data.open_json("data_boat_destination_statestique", root_model)
        df_boats = boats[boats["id_gen"] == id_generic]
        df_boats = df_boats[df_boats["id"] == id_location]
        if len(df_boats) < 1:
            return {"dates": 0 ,"score": 0}
        df_boats = self.rank_boat(df_boats)
        df_boats = pd.DataFrame(df_boats)
        df_boats["dates"] = pd.DatetimeIndex(data=df_boats.dates)
        df_boats          = df_boats.sort_values(by='dates')
        df_boats["dates"] = df_boats["dates"].dt.strftime('%Y/%m')

        return df_boats.to_dict('records')

    @staticmethod
    def top_boats(api_all_boats,root_model):
        boats    = Etl_data.open_json("data_boat_destination_statestique", root_model)
        all_boat = Statestique_bosts.all_boats(api_all_boats)
        boat     = list(boats.id_gen.unique())
        df_score = []
        for i in range(0,len(boat)):
            one_boat = pd.DataFrame(boats[boats["id_gen"] == boat[i]])
            score = len(one_boat)
            df_score.append({"id_gen":boat[i],"score":score})

        df_score = pd.DataFrame(sorted(df_score, key=lambda k: k["score"], reverse=True))
        all_boat = pd.DataFrame(all_boat)
        df_boats = pd.merge(df_score, all_boat, on='id_gen', how='inner')

        return df_boats.to_dict('records')

    def boat_type_destination(self,root_model,id_location,api_all_boats):
        df_boats = Etl_data.open_json("data_boat_destination_statestique", root_model)
        df_boats = df_boats[df_boats["id"] == id_location]
        req_generic = Etl_data.web_service_response(api_all_boats)
        req_generic = req_generic.rename(columns={'boat_id': 'id_gen'})
        df_boats = pd.merge(df_boats, req_generic, on='id_gen', how='inner')
        df_boats_type =[]
        motoryacht = 0
        monohull   = 0
        catamaran  = 0
        for index,boat in df_boats.iterrows():
            if boat["hull"].upper() == "MONOHULL":
                if boat["propulsion"].upper() == "SAILING":
                    monohull=monohull+1
                else:
                    motoryacht = motoryacht + 1
            else:
                catamaran = catamaran+ 1
        df_boats_type.append({'label': 'Catamaran',  'value': catamaran})
        df_boats_type.append({'label': 'Monohull',   'value': monohull})
        df_boats_type.append({'label': 'Motoryacht', 'value': motoryacht})
        return df_boats_type

    @staticmethod
    def destination_boat_statetique(root_model,id_location,api_all_boats):
        df_boats = Etl_data.open_json("data_boat_destination_statestique", root_model)
        df_boats = df_boats[df_boats["id"] == id_location]
        boat     = list(df_boats.id_gen.unique())
        df_score = []
        for i in range(0,len(boat)):
            score    = 0
            one_boat = pd.DataFrame(df_boats[df_boats["id_gen"] == boat[i]])
            score    = len(one_boat)
            df_score.append({"id_gen":boat[i],"score":score})
        df_score  = pd.DataFrame(df_score)
        all_boats = Etl_data.web_service_response(api_all_boats)
        all_boats = all_boats.rename(columns={'generic': 'id_gen'})
        all_boats = all_boats.drop_duplicates(subset="id_gen", keep="first")
        df_boats  = pd.merge(df_score, all_boats, on='id_gen', how='inner')
        df_final  = []
        for index , boat in df_boats.iterrows():
            df_final.append({"id_gen":boat["id_gen"],"name_boat":boat["boat_brand"] + " " + boat["boat_model"] + " " + boat["shipyard_name"],"score":boat["score"]})

        return df_final

    @staticmethod
    def country_boat_statistique(root_model,code_country):
        df_boats = Etl_data.open_json("recommandation_boats", root_model)
        df_boats = df_boats[df_boats["country"] == code_country]
        boat = list(df_boats.id_gen.unique())
        df_score = []
        for i in range(0,len(boat)):
            score    = 0
            one_boat = pd.DataFrame(df_boats[df_boats["id_generic"] == boat[i]])
            score    = len(one_boat)
            df_score.append({"id_gen":boat[i],"score":score,'name':one_boat["name"]})
        return df_score







