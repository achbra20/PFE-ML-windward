import requests
import pandas as pd
import json
import os
import numpy as np
from geopy.geocoders import Nominatim
from geopy.distance import vincenty
from Etl import Etl_data

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
import scipy.sparse as sparse
import random
import implicit


from sklearn.svm import SVR


class Etl_Model_Ml_Destination:

    #todo separation in request how have many destinatination
    @staticmethod
    def separation(lists,champ,label,sp):
        print("log separetion")
        speration = []
        for index, request in lists.iterrows():
            if request[champ] != "":
                x = request[champ].split(sp)
                for i in range(len(x)):
                    speration.append({label: x[i], "year": request["year"], "month": request["month"],"country": request["country"]})
        return pd.DataFrame(speration)

    #todo found the id of destination from location_name
    @staticmethod
    def found_id(list_id, champ_p, champ_id, list_req, champ_req):
        lists = []
        for index, request in list_req.iterrows():
            for index, req in list_id.iterrows():
                if req[champ_p].upper() in request[champ_req].upper() and req[champ_p] != "":
                    lists.append(
                        {'label': req[champ_p], 'destination': request[champ_req],'id': req[champ_id] ,'id_par': req["parent_location"],'type': req["location_type"]})
                    break
        return pd.DataFrame(lists)

    #todo found the parent_location from location_id
    @staticmethod
    def found_parent_destination(request,all_destination):
        if request["type"].upper() == "COUNTRY" or request["type"].upper() == "OCEAN":
            one_dest = all_destination[all_destination["location_id"] == request['id']]
            for index , destination in one_dest.iterrows():
                return  request['id'],destination["location_name"]
        if request["type"].upper() == "CITY" or request["type"].upper() == "ZONE":
            one_dest_par = all_destination[all_destination["location_id"] == request["id_par"]]
            for index, destination in one_dest_par.iterrows():
                new_request = {"type": destination["location_type"],"id": request["id_par"],"id_par":destination["parent_location"]}
                return  Etl_Model_Ml_Destination.found_parent_destination(new_request,all_destination)

class  Creation_Model:

    def __init__(self,api_ww,api_crm,api_alldestination,path):
        self.df_destination  = Etl_data.web_service_response(api_ww).append(Etl_data.web_service_response(api_crm))
        self.destination_id  = Etl_data.web_service_response(api_alldestination)
        self.path            = path

    # todo count the number of reservation for destination in dd/mm/yyyy
    def ranked_destination(self,df_destination):
        print('ranked')
        unique_destination = list(df_destination.destination.unique())
        result =[]
        for i in range(0, len(unique_destination)):
            one_destination = df_destination[df_destination["destination"] == unique_destination[i]]
            no_repetation   = one_destination.drop_duplicates(subset=['year','country','month'],keep='first')
            for index, request in no_repetation.iterrows():
                one = one_destination[one_destination["country"] == request['country']]
                one = one[one["year"] == request['year']]
                one = one[one["month"] == request['month']]
                result.append({"destination": request["destination"], "counts": len(one), "year": request["year"],"month": request["month"], "country": request["country"]})
        return pd.DataFrame(result)

    #todo found the parent id for a destination and put it in dictionary
    def found_id_parent(self,model_destination_all):
        print("found id location")
        unique_destination = model_destination_all.drop_duplicates(subset="destination", keep="first")
        destenation_id = Etl_Model_Ml_Destination.found_id(self.destination_id, "location_name", "location_id", unique_destination, "destination")
        model_destination = pd.merge(model_destination_all,destenation_id, on='destination', how='inner')
        final = []
        all_destination_lf = pd.DataFrame(self.destination_id [self.destination_id ["ref_language"] == "1"])
        for index, request in model_destination.iterrows():
            id_destination,name_destination = Etl_Model_Ml_Destination.found_parent_destination(request,all_destination_lf)
            final.append({'id': id_destination , "year": request["year"], 'month': request["month"],'country': request["country"], "counts": request["counts"]})
        return pd.DataFrame(final)

    #todo split the composed  destination in dataframe
    def separtion(self):
        print("log put all request on one dataframe")
        model_destination = self.df_destination[self.df_destination["month"] != "0"]
        model_destination = model_destination[model_destination["request_destination"] != ""]
        model_destination = Etl_Model_Ml_Destination.separation(model_destination, "request_destination", "destination", "/")
        model_destination = Etl_Model_Ml_Destination.separation(model_destination, "destination", "destination", ",")
        return model_destination

    #todo create the dataframe for training and save in json file
    def create_model(self):
        df_destination    = self.separtion()
        model_destination = self.ranked_destination(df_destination)
        Etl_data.writeToJSONFile(self.path, "data_Ml_destination_test", model_destination.to_dict('records'))
        #model_destination = Etl_data.open_json("data_Ml_destination_test",self.path)
        model_destination = self.found_id_parent(model_destination)
        Etl_data.writeToJSONFile(self.path, "data_Ml_destination", model_destination.to_dict('records'))
        print("log Json created")
        print("log Creation done !")

class Training_model_destination:

    def __init__(self,path,name_model):
        self.path = path
        self.name_model = name_model
        self.data   = self.index_country()

    def index_country(self):
        model_destination = Etl_data.open_json("data_Ml_destination",self.path)
        model_destination = model_destination[model_destination["country"] != ""]
        index_country = Etl_data.open_json("indexed_countryt",self.path)
        unique_country = list(model_destination.country.unique())
        for i in range(0, len(unique_country)):
            one_index = index_country[index_country["label"] == unique_country[i]]
            for index , one_contry in one_index.iterrows():
                model_destination.loc[model_destination['country'] == unique_country[i], ['country']] = one_contry["index"]
        return  model_destination

    #todo training the algorithme with the data and save it in file.model and save the score and MSE
    def training(self):
        print("begin of training")
        print(self.data)
        #model_destination = model_destination.append(model_destination)

        feature_col_names     = ['year', 'month', 'id' , 'country']
        predicted_class_names = ['counts']
        X                     = self.data[feature_col_names].values
        y                     = self.data[predicted_class_names].values

        split_test_size       = 0.20
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_test_size, random_state=42)

        model = KNeighborsRegressor(weights='distance',n_neighbors=6)
        model.fit(X_train, y_train)



        joblib.dump(model,self.path+"/"+self.name_model)

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
        ############################
        info = []
        info.append({"scoretrai":score_train,"errortr":error_train,"scoretes":score_test,"errorte":error_test})
        Etl_data.writeToJSONFile(self.path,"score_ml_destination",info)
        ###########################
        predection = Predection_distination(self.path+"/"+self.name_model )
        predection.restart()
        ###########################
        print("training done !")
        return "training done !"

class Creation_Model_recommendation:

    def __init__(self,api_ww, api_crm, all_destination,path):
        self.path            = path
        self.data_ww         = Etl_data.web_service_response(api_ww)
        self.data_crm        = Etl_data.web_service_response(api_crm)
        self.all_destination = Etl_data.web_service_response(all_destination)

    def ranked_destination(self,df_destination):
        print('log ranked the destination with country')
        unique_destination = list(df_destination.id.unique())
        result =[]
        for i in range(len(unique_destination)):
            one_destination = df_destination[df_destination["id"] == unique_destination[i]]
            no_repetation   = one_destination.drop_duplicates(subset=['country'],keep='first')
            for index, request in no_repetation.iterrows():
                one = one_destination[one_destination["country"] == request['country']]
                result.append({"id":request["id"],"destination":request["destination"], "counts": len(one), "country": request["country"]})
        return pd.DataFrame(sorted(result, key=lambda k: k["counts"], reverse=True))

    def found_id_destination(self,destination_id):
        print("log found id parent for destination")
        final = []
        all_destination_lf = self.all_destination[self.all_destination["ref_language"] == "1"]
        for index, request in destination_id.iterrows():
            id_destination, name_destination = Etl_Model_Ml_Destination.found_parent_destination(request,all_destination_lf)
            final.append( {'id': id_destination, "destination": name_destination,'country': request["country"]})
        return pd.DataFrame(final)

    def do_separation(self,list, champ, country, sp):
        Statistique = []
        b = True
        for index, request in list.iterrows():
            if request[champ] != "":
                x = request[champ].split(sp)
                for i in range(len(x)):
                    Statistique.append({"destination": x[i], "country": request[country]})
        return pd.DataFrame(Statistique)

    def separation(self):
        print('log clean data')
        distination_ww  = self.do_separation(self.data_ww, "request_destination", "country", "/")
        distination_crm = self.do_separation(self.data_crm, "destination_lib_francais", "pays", "/")
        distination_crm = self.do_separation(distination_crm, "destination", "country", ",")
        destination     = distination_crm.append(distination_ww)
        destination     = destination[destination["destination"] != ""]
        destination     = destination[destination["country"] != ""]
        return destination

    def indexed_country(self,df_destination):
        print("indexed country")
        unique = list(df_destination.country.unique())
        indexed_country=[]
        for i in range(0,len(unique)):
            indexed_country.append({"label": unique[i], "index": i})
            df_destination.loc[df_destination['country'] == unique[i], ['country']] = i
        Etl_data.writeToJSONFile(self.path,"indexed_country" , indexed_country)

        return df_destination

    def create_m_recomendation(self):

        destination        = self.separation()
        unique_destination = destination.drop_duplicates(subset="destination", keep="first")
        all_destination2   = self.all_destination.drop_duplicates(subset="location_name", keep="first")
        destenation_id     = Etl_Model_Ml_Destination.found_id(all_destination2, "location_name", "location_id", unique_destination, "destination")
        destenation        = pd.merge(destination, destenation_id, on='destination', how='inner')

        destination_country = self.found_id_destination(destenation)
        print(destination_country)
        destination         = self.ranked_destination(destination_country)
        print(destination)
        destination         = self.indexed_country(destination)

        Etl_data.writeToJSONFile(self.path, "recommendation_destination", destination.to_dict('records'))

        recommendation_distance = Recommendation_distnation(self.path)
        recommendation_distance.restart()

        print ("creation done")

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
class Predection_distination():

    def __init__(self,name):
        self.model = joblib.load(name)
        self.name  = name

    def restart(self,):
        self.model = joblib.load(self.name)

    def predict(self,year, month, id ,country):
        X_new = [[year , month , id , country]]
        knr = self.model.predict(X_new)
        return knr

@singleton
class Recommendation_distnation:

    alpha = 15

    def __init__(self,root_model):
        self.name_model = "recommendation_destination"
        self.root_model = root_model
        self.purchases_sparse, self.countrys, self.distinations, self.indexed_destination = self.saprce()
        self.distination_train, self.distination_test, self.distination_users_altered     = self.make_train()
        self.country_vecs, self.distination_vecs, self.distinations_arr, self.countrys_arr= self.recommend_destination()


    def restart(self):
        self.purchases_sparse, self.countrys, self.distinations, self.indexed_destination = self.saprce()
        self.distination_train, self.distination_test, self.distination_users_altered     = self.make_train()
        self.country_vecs, self.distination_vecs, self.distinations_arr, self.countrys_arr= self.recommend_destination()

    def saprce(self):
        model_destination = Etl_data.open_json(self.name_model, self.root_model)
        model_destination = pd.DataFrame(model_destination)

        indexed_destination = model_destination[["id", "destination"]].drop_duplicates()
        indexed_destination['id'] = indexed_destination.id.astype(str)

        model_destination['country'] = model_destination.country.astype(int)
        model_destination = model_destination[['id', 'counts', 'country']]
        grouped_purchased = model_destination.groupby(['country', 'id']).sum().reset_index()

        countrys = list(np.sort(grouped_purchased.country.unique()))
        distinations = list(grouped_purchased.id.unique())
        quantity = list(grouped_purchased.counts)

        rows = grouped_purchased.country.astype('category', categories=countrys).cat.codes
        # Get the associated row indices
        cols = grouped_purchased.id.astype('category', categories=distinations).cat.codes
        # Get the associated column indices
        purchases_sparse = sparse.csr_matrix((quantity, (rows, cols)), shape=(len(countrys), len(distinations)))
        return purchases_sparse ,countrys ,distinations ,indexed_destination

    def make_train(self, pct_test=0.2):

        test_set = self.purchases_sparse.copy()
        test_set[test_set != 0] = 1
        training_set = self.purchases_sparse.copy()
        nonzero_inds = training_set.nonzero()
        nonzero_pairs = list(
            zip(nonzero_inds[0], nonzero_inds[1]))
        random.seed(0)
        num_samples = int(
            np.ceil(pct_test * len(nonzero_pairs)))
        samples = random.sample(nonzero_pairs,
                                num_samples)
        user_inds = [index[0] for index in samples]
        item_inds = [index[1] for index in samples]
        training_set[user_inds, item_inds] = 0
        training_set.eliminate_zeros()
        return training_set, test_set, list(set(user_inds))

    def recommend_destination(self):
        countrys_arr = np.array(self.countrys)
        distinations_arr = np.array(self.distinations)
        user_vecs, item_vecs = implicit.alternating_least_squares((self.distination_train * self.alpha).astype('double'),
                                                                  factors=20,
                                                                  regularization=0.1,
                                                                  iterations=50)
        return  user_vecs, item_vecs, distinations_arr, countrys_arr

    def get_items_purchased(self,country_id, mf_train, country_list, destination_list, des_lookup):
        country_ind = np.where(country_list == country_id)[0][0]
        purchased_ind = mf_train[country_ind, :].nonzero()[1]
        prod_codes = destination_list[purchased_ind]
        destination = des_lookup.loc[des_lookup.id.isin(prod_codes)]
        return destination

    def rec_items(self,country_id, mf_train, country_vecs, destination_vecs, country_list, destination_list, des_lookup, nbdes=10):

        country_ind = np.where(country_list == country_id)[0][0]
        pref_vec = mf_train[country_ind, :].toarray()
        pref_vec = pref_vec.reshape(-1) + 1
        pref_vec[pref_vec > 1] = 0
        rec_vector = country_vecs[country_ind, :].dot(destination_vecs.T)
        min_max = MinMaxScaler()
        rec_vector_scaled = min_max.fit_transform(rec_vector.reshape(-1, 1))[:, 0]
        recommend_vector = pref_vec * rec_vector_scaled
        dis_idx = np.argsort(recommend_vector)[::-1][:nbdes]
        rec_list = []
        for index in dis_idx:
            code = destination_list[index]
            single = des_lookup[des_lookup["id"] == code]
            for index, sing in single.iterrows():
                rec_list.append([code, sing["destination"]])
        codes = [item[0] for item in rec_list]
        descriptions = [item[1] for item in rec_list]
        final_frame = pd.DataFrame({'id': codes, 'destination': descriptions})
        return final_frame[['id', 'destination']]

    def list_recommended(self,country,date,nb_destionation,api_all_des,model_name):
        df_country  = Etl_data.open_json("indexed_country", self.root_model)
        df_country_ml = Etl_data.open_json("indexed_countryt", self.root_model)
        df_country  = pd.DataFrame(df_country)
        index_c     = df_country[df_country["label"] == country.upper()]
        index_ml = df_country_ml[df_country_ml["label"] == country.upper()]
        if len(index_c)>0:
            print("existe")
            destination = self.get_items_purchased(int(index_c["index"]), self.distination_train, self.countrys_arr, self.distinations_arr, self.indexed_destination)
            if len(destination)< int(nb_destionation):
                print(len(destination))
                #destination_rec = self.rec_items(int(index_c["index"]),self.distination_train,self.country_vecs,self.distination_vecs, self.countrys_arr, self.distinations_arr, self.indexed_destination, 20)
                destination_distance = self.list_for_any(country, api_all_des)
                destination_distance = pd.DataFrame(destination_distance)
                destination = destination.append(destination_distance[['id', 'destination']][:int(nb_destionation)-len(destination)])
        else:
            print("not existe")
            destination = self.list_for_any(country, api_all_des)
            destination = pd.DataFrame(destination[:int(nb_destionation)])

        destination = self.scored_destnation(destination,date,model_name,int(index_ml["index"]))
        return  destination[:int(nb_destionation)]

    def scored_destnation(self,destination,date,name_model,country):
        predection = Predection_distination(self.root_model + "/" + name_model)
        destination = destination.drop_duplicates(subset="id", keep="first")
        rank = []
        date_in = date.split("-")
        for index, request in destination.iterrows():
            predicit = predection.predict(date_in[2], date_in[1], request["id"],country)
            rank.append({"date": date_in[0]+"/"+date_in[1]+"/"+date_in[2], "score": int(predicit[0]),'destination': request["destination"], 'id': request["id"]})

        return sorted(rank, key=lambda k: k["score"], reverse=True)

    def list_for_any(self,country,api_all):
        all_destination = Etl_data.web_service_response(api_all)
        all_destination = pd.DataFrame(all_destination)
        all_destination = pd.DataFrame(all_destination[all_destination["location_type"] == "country"])
        one_dest = pd.DataFrame(all_destination[all_destination["location_type"] == "ocean"])
        all_destination.append(one_dest)
        all_destination = all_destination.drop_duplicates(subset="location_id", keep="first")
        geolocator = Nominatim()
        location = geolocator.geocode(country)

        newport_ri = (location.latitude, location.longitude)
        distance = []
        for index, distination in all_destination.iterrows():
            cleveland_oh = (distination["latitude"], distination["longitude"])
            distance.append({"id":distination["location_id"],"destination":distination["location_name"],"distance":vincenty(newport_ri, cleveland_oh).miles})
        distance = sorted(distance, key=lambda k: k["distance"], reverse=False)
        return   distance

