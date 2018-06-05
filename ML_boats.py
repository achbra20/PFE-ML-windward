from Etl import Etl_data
import pandas as pd

import scipy.sparse as sparse
import numpy as np
import random
import implicit
import datetime
import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib

class  Etl_data_boat:

    @staticmethod
    def found_id_boat(boat_id,boats,champ_boat):
        ch  = boat_id["boat_brand"] + " " + boat_id["boat_model"] + " " + boat_id["shipyard_name"]
        ch2 = boat_id["shipyard_name"] + " " + boat_id["boat_brand"] + " " + boat_id["boat_model"]
        if ch.upper() in boats[champ_boat].upper() or boats[champ_boat].upper() in ch.upper():
            return ch , boat_id['generic']
        elif ch2.upper() in boats[champ_boat].upper():
            return ch , boat_id['generic']
        else:
            return "null","null"

class Create_Data_Recommendation_Boats:

    def __init__(self,api_data_ww,api_data_crm,api_boats_id_ww):
        self.data_ww     = Etl_data.web_service_response(api_data_ww)
        self.data_crm    = Etl_data.web_service_response(api_data_crm)
        self.boats_id_ww = Etl_data.web_service_response(api_boats_id_ww)

    def boat_found_all_id(self):
        unique_boat_id_crm = self.data_crm.drop_duplicates(subset="boat_model", keep="first")
        boats_crm = self.boat_found_id(unique_boat_id_crm, self.boats_id_ww)
        print(boats_crm)
        unique_boat_id_ww = self.data_ww.drop_duplicates(subset="boat_model", keep="first")
        boats_ww = self.boat_found_id(unique_boat_id_ww, self.boats_id_ww)

        boats_all_crm = pd.merge(pd.DataFrame(boats_crm), pd.DataFrame(self.data_crm), on='boat_model', how='inner')
        boats_all_ww = pd.merge(pd.DataFrame(boats_ww), pd.DataFrame(self.data_ww), on='boat_model', how='inner')
        boats_all_ww = boats_all_ww[['name', "boat_model", 'country', "id_generic", "id"]]
        boat_final = boats_all_ww.append(boats_all_crm)
        return  boat_final

    def boat_found_id(self,boat_request_data,boat_id_data):
        print('log found id_generic boat')
        boats_id = []
        for index, boat_req in boat_request_data.iterrows():
            for index, boat_id in boat_id_data.iterrows():
                name , generic_id = Etl_data_boat.found_id_boat(boat_id,boat_req,"boat_model")
                if (name != "null"):
                    print(name)
                    boats_id.append({"id": boat_id["boat_id"], "id_generic": generic_id, "name": name,'boat_model': boat_req["boat_model"]})
                    break
        return boats_id

    def ranked_boat(self,df_boats):
        print("log ranked boats")
        result =[]
        unique_country = list(df_boats.country.unique())
        for i in range(0,len(unique_country)):
            one_country   = df_boats[df_boats["country"] == unique_country[i]]
            unique_boats = list(one_country.id_generic.unique())
            for j in range(0,len(unique_boats)):
                one  = one_country[one_country["id_generic"] == unique_boats[j]]
                name = list(one.name.unique())
                result.append({ "id_generic": unique_boats[j], "counts": len(one),"name": name[0],"country": unique_country[i]})

        return pd.DataFrame(sorted(result, key=lambda k: k["counts"], reverse=True))

    def created_data_for_recommendation_boat(self,root_model):
        print("log refrech data start")
        boats_data = self.boat_found_all_id()
        boats_data = boats_data[boats_data["country"] != ""]
        boats_data_final = self.ranked_boat(boats_data)
        Etl_data.writeToJSONFile(root_model,"recommandation_boats",boats_data_final.to_dict("records"))
        recommendation = Recommendation_boats(root_model)
        recommendation.restart(root_model)

class ML_boat_data:

    def __init__(self,api_data_ww,api_data_crm1,api_data_crm2,api_boats_id_ww):
        self.data_ww    = Etl_data.web_service_response(api_data_ww)
        self.data_crm1  = Etl_data.web_service_response(api_data_crm1)
        self.data_crm2  = Etl_data.web_service_response(api_data_crm2)
        self.boat_id_ww = Etl_data.web_service_response(api_boats_id_ww)
        self.all_boat   = self.all_boat_int(self.data_ww,self.data_crm1,self.data_crm2)

    def all_boat_int(self,data1,data2,data3):
        df_boats = data1.append(data2)
        df_boats = df_boats.append(data3)
        df_boats = df_boats[df_boats['year']!= "0"]
        df_boats = df_boats[df_boats['boat'] != ""]
        return df_boats

    def boat_found_id(self,boat_request_data,boat_id_data):
        print('log found id_generic boat')
        boats_id = []
        for index, boat_req in boat_request_data.iterrows():
            for index, boat_id in boat_id_data.iterrows():
                name , generic_id = Etl_data_boat.found_id_boat(boat_id,boat_req,"boat")
                if (name != "null"):
                    print(name)
                    boats_id.append({ "id_gen": generic_id, "name": name,'boat': boat_req["boat"]})
                    break

        return boats_id

    def ranged_week(self,df_boats):
        new_df_boats = []
        for index, req in df_boats.iterrows():
            datetime_object = datetime.date(int(req["year"]), int(req["month"]), int(req["day"]))
            wk = datetime_object.isocalendar()[1]
            new_df_boats.append({'id_gen': req['id_gen'], 'day': wk, 'year': req['year']})
        return pd.DataFrame(new_df_boats)

    def ranked_boat(self,df_boats):
        print("log ranked boats")
        result =[]
        unique_boats = list(df_boats.id_gen.unique())
        for i in range(len(unique_boats)):
            one_boats = df_boats[df_boats["id_gen"] == unique_boats[i]]
            no_repetation = one_boats.drop_duplicates(subset=['year'], keep='first')
            for index, request in no_repetation.iterrows():
                one_year = one_boats[one_boats["year"] == request['year']]
                unique_day= list(one_year.day.unique())
                for j in range(len(unique_day)):
                    one_day = one_year[one_year["day"] == unique_day[j]]
                    result.append({ "id_gen": unique_boats[i], "counts": len(one_day),"day": unique_day[j],"year": request["year"]})
        return pd.DataFrame(sorted(result, key=lambda k: k["counts"], reverse=True))

    def create_data_ml_boats(self,root_model):
        boat           = self.all_boat.drop_duplicates(subset='boat',keep='first')
        boat_id        = self.boat_id_ww.drop_duplicates(subset='generic',keep='first')
        boat_final     = self.boat_found_id(boat,boat_id)
        df_boats       = pd.merge(pd.DataFrame(self.all_boat), pd.DataFrame(boat_final), on='boat', how='inner')
        df_boats       = self.ranged_week(df_boats)
        df_boats_final = self.ranked_boat(df_boats)
        Etl_data.writeToJSONFile(root_model,'data_Ml_boat',df_boats_final.to_dict("records"))
        return "done"

class Training_boat:

    def __init__(self,root_model,name_model):
        self.path       = root_model
        self.name_model = name_model
        self.boat_data  = Etl_data.open_json('data_Ml_boat',self.path)
        self.training_data_boat()

    def training_data_boat(self):
        feature_col_names = ['year','day','id_gen' ]
        predicted_class_names = ['counts']

        X = self.boat_data[feature_col_names].values  # predictor feature columns
        y = self.boat_data[predicted_class_names].values  # predicted class (score) column (1 X m)
        split_test_size = 0.25

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_test_size, random_state=20)

        regrossor = RandomForestRegressor(n_estimators=500, random_state=0)
        regrossor.fit(X_train, y_train)

        joblib.dump(regrossor, self.path  + "/" + self.name_model)

        score_train = math.fabs(regrossor.score(X_train,y_train))
        score_test  = math.fabs(regrossor.score(X_test,y_test))
        print(score_train, score_test)

        x_pred      = regrossor.predict(X_train)
        error_train = mean_squared_error(y_train, x_pred)
        print(error_train)

        y_pred = regrossor.predict(X_test)
        error_test = mean_squared_error(y_test, y_pred)
        print(error_test)

        prediction = Prediction(self.path  + "/" + self.name_model)
        prediction.restart()

        info = []
        info.append({"scoretrai":score_train,"errortr":error_train,"scoretes":score_test,"errorte":error_test})
        Etl_data.writeToJSONFile(self.path ,"score_ml_boats",info)

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
class Recommendation_boats:

    alpha = 15
    def __init__(self,root_model):
        self.indexed_country,self.boats_data                                   = self.index_country(root_model)
        self.purchases_sparse, self.countrys, self.boats, self.indexed_boats   = self.saprce()
        self.boat_train, self.boat_test, self.boats_countrys_altered           = self.make_train()
        self.countrys_vecs, self.boats_vecs, self.boats_arr, self.countrys_arr = self.recommend_boats()

    def restart(self,root_model):
        self.indexed_country,self.boats_data                                   = self.index_country(root_model)
        self.purchases_sparse, self.countrys, self.boats, self.indexed_boats   = self.saprce()
        self.boat_train, self.boat_test, self.boats_countrys_altered           = self.make_train()
        self.countrys_vecs, self.boats_vecs, self.boats_arr, self.countrys_arr = self.recommend_boats()

    def index_country(self,root_model):
        boats_data = Etl_data.open_json("recommandation_boats",root_model)
        index_country = Etl_data.open_json("indexed_countryt",root_model)
        unique_country = list(boats_data.country.unique())
        for i in range(0, len(unique_country)):
            one_index = index_country[index_country["label"] == unique_country[i]]
            for index , one_contry in one_index.iterrows():
                boats_data.loc[boats_data['country'] == unique_country[i], ['country']] = one_contry["index"]
        return index_country, boats_data

    def saprce(self):

        data_recomendation_boats = pd.DataFrame(self.boats_data)

        indexed_boats               = data_recomendation_boats[['name','id_generic']].drop_duplicates()
        indexed_boats['id_generic'] = indexed_boats.id_generic.astype(str)

        data_recomendation_boats['country'] = data_recomendation_boats.country.astype(int)
        data_recomendation_boats            = data_recomendation_boats[['id_generic', 'counts', 'country']]
        grouped_purchased                   = data_recomendation_boats.groupby(['country', 'id_generic']).sum().reset_index()

        countrys = list(np.sort(grouped_purchased.country.unique()))
        boats    = list(grouped_purchased.id_generic.unique())
        quantity = list(grouped_purchased.counts)

        rows = grouped_purchased.country.astype('category', categories=countrys).cat.codes
        # Get the associated row indices
        cols = grouped_purchased.id_generic.astype('category', categories=boats).cat.codes
        # Get the associated column indices
        purchases_sparse = sparse.csr_matrix((quantity, (rows, cols)), shape=(len(countrys), len(boats)))
        return purchases_sparse ,countrys ,boats ,indexed_boats

    def make_train(self, pct_test=0.2):

        test_set                = self.purchases_sparse.copy()
        test_set[test_set != 0] = 1

        training_set  = self.purchases_sparse.copy()
        nonzero_inds  = training_set.nonzero()
        nonzero_pairs = list(zip(nonzero_inds[0], nonzero_inds[1]))
        random.seed(0)

        num_samples    = int(np.ceil(pct_test * len(nonzero_pairs)))
        samples        = random.sample(nonzero_pairs,num_samples)
        country_inds   = [index[0] for index in samples]
        boats_inds     = [index[1] for index in samples]

        training_set[country_inds, boats_inds] = 0
        training_set.eliminate_zeros()

        return training_set, test_set, list(set(country_inds))

    def recommend_boats(self):
        countrys_arr = np.array(self.countrys)
        boats_arr = np.array(self.boats)
        countrys_vecs, boats_vecs = implicit.alternating_least_squares((self.boat_train * self.alpha).astype('double'),factors=20,regularization=0.1,iterations=50)
        return  countrys_vecs, boats_vecs, boats_arr, countrys_arr

    def get_boats_purchased(self,country_id):
        country_ind   = np.where(self.countrys_arr == country_id)[0][0]
        purchased_ind = self.boat_train[country_ind, :].nonzero()[1]
        prod_codes    = self.boats_arr[purchased_ind]
        boat_rec      = self.indexed_boats.loc[self.indexed_boats.id_generic.isin(prod_codes)]
        return boat_rec

    def rec_boats(self,country_id,num_boats=10):

        country_ind = np.where( self.countrys_arr == country_id)[0][0]
        pref_vec = self.boat_train[country_ind, :].toarray()
        pref_vec = pref_vec.reshape(-1) + 1
        pref_vec[pref_vec > 1] = 0
        rec_vector = self.countrys_vecs[country_ind, :].dot(self.boats_vecs.T)
        min_max = MinMaxScaler()
        rec_vector_scaled = min_max.fit_transform(rec_vector.reshape(-1, 1))[:, 0]
        recommend_vector = pref_vec * rec_vector_scaled
        boat_idx = np.argsort(recommend_vector)[::-1][:num_boats]
        rec_list = []
        for index in boat_idx:
            code = self.boats_arr[index]
            rec_list.append([code, self.indexed_boats.name.loc[self.indexed_boats.id_generic == code].iloc[0]])
        codes = [item[0] for item in rec_list]
        descriptions = [item[1] for item in rec_list]
        final_frame = pd.DataFrame({'id_generic': codes, 'name': descriptions})
        return final_frame[['id_generic', 'name']]
    #just for test
    def recommended_boats(self,country,nb_recommendation):

        indexed_country = pd.DataFrame(self.indexed_country)
        index_c = indexed_country[indexed_country["label"] == country.upper()]
        if len(index_c)> 0 :
            boats_taked = self.get_boats_purchased(int(index_c["index"]))
            if len(boats_taked) < int(nb_recommendation) :
                nb_rec = int(nb_recommendation) - len(boats_taked)
                boats_recomendation = self.rec_boats(int(index_c["index"]), nb_rec)
                boats_final = boats_taked.append(boats_recomendation)
                return boats_final.to_dict('records')
            return boats_taked.to_dict('records')
        else:
            rec_default_boat = self.recommendation_for_any()
            return rec_default_boat[:int(nb_recommendation)]

    def recommendation_for_any(self):
        rec_default_boat = []
        unique_boat = list(self.boats_data.id_generic.unique())
        for i in range(0,len(unique_boat)):
            count = 0
            one_boat = pd.DataFrame(self.boats_data[self.boats_data["id_generic"] == unique_boat[i]])
            for index , boat in one_boat.iterrows():
                count = count + boat["counts"]
                name  = boat["name"]
            rec_default_boat.append({"id_generic":unique_boat[i],"name":name,"counts":count})
            rec_default_boat = sorted(rec_default_boat, key=lambda k: k["counts"], reverse=True)
        return   rec_default_boat

    def scored_recommendation(self,name_model,root_model,date,boats):
        boats   = pd.DataFrame(boats)
        date_in = date.split("-")
        datetime_object = datetime.date(int(date_in[2]),int(date_in[1]), int(date_in[0]))
        wk          = datetime_object.isocalendar()[1]
        predction   = Prediction(root_model+'/'+name_model)
        scored_boat = []
        for index ,boat in boats.iterrows():
            score = predction.predict(int(date_in[2]),wk,int(boat['id_generic']))
            scored_boat.append({'id_generic':boat['id_generic'],'name':boat['name'],'score':score[0]})
        return pd.DataFrame(sorted(scored_boat, key=lambda k: k["score"], reverse=True))

    def recommandation_boat(self,country,date,name_model,root_model,nb_recommendation):
        indexed_country = pd.DataFrame(self.indexed_country)
        index_c = indexed_country[indexed_country["label"] == country.upper()]
        if len(index_c) > 0 :
            boat_taked = self.get_boats_purchased(int(index_c["index"]))
            if len(boat_taked) < int(nb_recommendation):
                nb_rec = int(nb_recommendation) - len(boat_taked)
                boats_recomendation = self.rec_boats(int(index_c["index"]),nb_rec)
                boat_taked = boat_taked.append(boats_recomendation)
            result = self.scored_recommendation(name_model, root_model, date, boat_taked)
        else:
            all_boat = self.boats_data.drop_duplicates(subset="id_generic",keep='first')
            result = self.scored_recommendation(name_model,root_model,date,all_boat)

        result = result[:int(nb_recommendation)]
        return  result.to_dict('records')

@singleton
class Prediction:

    def __init__(self,name):
        self.name  = name
        self.model = joblib.load(name)
        print('instanced')

    def restart(self):
        self.model = joblib.load(self.name)

    def predict(self,year,day,id_generic):
        X_new = [[year,day,id_generic]]
        predection   =  self.model.predict(X_new)
        return predection

